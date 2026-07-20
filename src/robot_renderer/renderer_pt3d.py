from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ._pt3d import _require_pt3d
from .obj_mesh_colors import load_obj_with_mtl_vertex_colors
from .robot_model import RobotKinematics

try:
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRendererWithFragments,
        MeshRasterizer,
        BlendParams,
        HardPhongShader,
        PointLights,
        TexturesVertex,
    )
except Exception:
    # Broad on purpose: a mismatched compiled extension can raise OSError or
    # RuntimeError, not just ImportError. Keep the module importable so
    # _require_pt3d() can raise the friendly install message at construction.
    load_obj = Meshes = PerspectiveCameras = RasterizationSettings = None  # type: ignore[assignment]
    MeshRendererWithFragments = MeshRasterizer = BlendParams = None  # type: ignore[assignment]
    HardPhongShader = PointLights = TexturesVertex = None  # type: ignore[assignment]


def _to_tensor(x: Union[np.ndarray, torch.Tensor], dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype)
    return x.to(dtype=dtype)


def _apply_face_colors(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_colors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mesh tensors that preserve one color per face.

    A shared vertex cannot carry different colors for adjacent faces. Keep the
    original topology for a uniform material, and otherwise duplicate the
    triangle corners so every face retains its own color.
    """
    if faces.shape[0] != face_colors.shape[0]:
        raise ValueError(
            f"Expected one color per face, got {face_colors.shape[0]} colors "
            f"for {faces.shape[0]} faces."
        )
    if faces.numel() == 0:
        return verts, faces, torch.ones(
            (verts.shape[0], 3), dtype=face_colors.dtype, device=face_colors.device
        )
    if torch.all(face_colors == face_colors[:1]):
        colors = face_colors[:1].expand(verts.shape[0], -1)
        return verts, faces, colors

    expanded_verts = verts[faces.reshape(-1)]
    expanded_faces = torch.arange(
        faces.numel(), dtype=faces.dtype, device=faces.device
    ).reshape(-1, 3)
    expanded_colors = (
        face_colors[:, None, :].expand(-1, 3, -1).reshape(-1, face_colors.shape[-1])
    )
    return expanded_verts, expanded_faces, expanded_colors


def gaussian_blur_chw(
    img: torch.Tensor,
    sigma: float = 4.0,
) -> torch.Tensor:
    """
    Fast approximate Gaussian blur via bilinear downsample and upsample.

    Not a true Gaussian kernel: the double bilinear resampling gives a
    visually similar low-pass result at a fraction of the cost, which is all
    the background-blurring use case needs.

    Args:
        img:   (3, H, W) float tensor.
        sigma: Blur strength; downsamples to max(1, H/sigma) then upsamples back.

    Returns:
        (3, H, W) float tensor on the same device as input.
    """
    _, H, W = img.shape
    ds_h = max(1, int(H / sigma))
    ds_w = max(1, int(W / sigma))
    tiny = F.interpolate(
        img.unsqueeze(0), size=(ds_h, ds_w), mode="bilinear", align_corners=False
    )
    blurred = F.interpolate(tiny, size=(H, W), mode="bilinear", align_corners=False)
    return blurred.squeeze(0)


class RobotPyTorch3DRenderer:
    """
    Renders a robot given joint configuration using PyTorch3D.

    Loads OBJ meshes per link, uses robot.get_joint_R_t(q) to pose them,
    assembles a single combined mesh, then renders with PT3D.

    Usage:
        renderer = RobotPyTorch3DRenderer(robot, mesh_files, camera_K_mat, ...)
        renderer.to(device)
        meshes = renderer.transform_mesh2robot_config(joint_angles)
        rgb, frags = renderer.phong_render_batched(meshes, R_batch, T_batch)
    """

    def __init__(
        self,
        robot: RobotKinematics,
        mesh_files: Union[List[str], List[List[str]]],
        camera_K_mat: Union[np.ndarray, torch.Tensor],
        image_size_hw: Optional[Tuple[int, int]] = None,
        device: Optional[Union[str, torch.device]] = None,
        color_mapping: Optional[Dict[str, Union[List[float], torch.Tensor]]] = None,
    ):
        """
        Args:
            robot:          RobotKinematics implementation.
            mesh_files:     Nested list of OBJ paths, one list per link.
            camera_K_mat:   3x3 intrinsics (fx, 0, cx; 0, fy, cy; 0, 0, 1).
            image_size_hw:  (H, W) output resolution. Inferred from K if None.
            device:         Torch device. Defaults to CPU (PyTorch convention);
                            pass "cuda" or call .to("cuda") to use the GPU.
            color_mapping:  Optional keyword-to-RGB dict. Keys matched to mesh
                            filename substrings; "default" as fallback.
        """
        _require_pt3d()
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.image_size_hw = image_size_hw
        self.camera_K_mat = camera_K_mat
        self._background_image: Optional[torch.Tensor] = None

        self._color_mapping: Dict[str, torch.Tensor] = {}
        if color_mapping:
            for k, v in color_mapping.items():
                self._color_mapping[k] = (
                    torch.tensor(v, dtype=torch.float32)
                    if not isinstance(v, torch.Tensor)
                    else v.to(torch.float32)
                )

        self.preload_verts: List[List[torch.Tensor]] = []
        self.preload_faces: List[List[torch.Tensor]] = []
        self.colors: List[List[torch.Tensor]] = []

        for m_file in mesh_files:
            if not isinstance(m_file, list):
                m_file = [m_file]
            verts_link, faces_link, colors_link = [], [], []
            for file in m_file:
                v, f_idx, aux = load_obj(file, load_textures=True)
                f = f_idx.verts_idx if hasattr(f_idx, "verts_idx") else f_idx[0]
                v, f, color = self._build_mesh_with_colors(v, f, aux, file)
                verts_link.append(v)
                faces_link.append(f)
                colors_link.append(color)
            # Sanity-check: each mesh's face indices must be within [0, that mesh's V).
            # Uses per-mesh vertex count, not the combined total; a mesh pointing into
            # another mesh's vertex range is also corrupt and must be caught.
            for vi, fi, file in zip(verts_link, faces_link, m_file, strict=True):
                valid = fi[fi >= 0]
                if valid.numel() > 0 and int(valid.max().item()) >= vi.shape[0]:
                    raise ValueError(
                        f"Mesh file '{file}' has face indices up to {int(valid.max().item())} "
                        f"but only {vi.shape[0]} vertices were loaded. "
                        "The OBJ file may be corrupt or incorrectly converted from DAE/STL."
                    )
            # Move to the target device right away. load_obj returns CPU
            # tensors, and rendering mixes these with FK tensors that live on
            # self.device, so leaving them on CPU would crash the first render
            # on a CUDA machine unless the caller happened to call .to().
            self.preload_verts.append([v.to(self.device) for v in verts_link])
            self.preload_faces.append([f.to(self.device) for f in faces_link])
            self.colors.append([c.to(self.device) for c in colors_link])

        self._build_pt3d_renderers(camera_K_mat, image_size_hw)

    def to(self, device: Union[str, torch.device]) -> "RobotPyTorch3DRenderer":
        """Move all renderer state to device. Returns self for chaining."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.preload_verts = [
            [v.to(device) for v in link] for link in self.preload_verts
        ]
        self.preload_faces = [
            [f.to(device) for f in link] for link in self.preload_faces
        ]
        self.colors = [
            [c.to(device) for c in link] for link in self.colors
        ]

        self._build_pt3d_renderers(self.camera_K_mat, self.image_size_hw)
        return self

    def _build_pt3d_renderers(
        self,
        camera_K_mat: Union[np.ndarray, torch.Tensor],
        image_size_hw: Optional[Tuple[int, int]],
    ) -> None:
        focal_len, principal_point, image_size = self._unpack_camera_K_mat(
            camera_K_mat, image_size_hw=image_size_hw
        )
        self.cameras = PerspectiveCameras(
            focal_length=focal_len,
            principal_point=principal_point,
            in_ndc=False,
            image_size=image_size,
            device=self.device,
        )
        h, w = int(image_size[0, 0].item()), int(image_size[0, 1].item())

        raster_phong = RasterizationSettings(
            blur_radius=0.0,
            faces_per_pixel=1,
            image_size=(h, w),
            max_faces_per_bin=None,  # updated per-render in phong_render_batched based on mesh face count
            # Robot CAD assets are not guaranteed to be closed or consistently
            # wound. Culling is only a performance optimisation, and can silently
            # remove valid surfaces (notably after mesh simplification).
            cull_backfaces=False,
        )
        point_lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 0.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
        )
        self.phong_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_phong),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=point_lights,
                blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
            ),
        )

    def set_background_image(self, image_hw3: Optional[torch.Tensor]) -> None:
        """
        Set a full-image background (H, W, 3) for Phong rendering.
        Pass None to revert to flat-color mode.
        Kept for direct use; prefer passing background_image to render_templates().
        """
        if image_hw3 is not None:
            self._background_image = image_hw3.to(device=self.device, dtype=torch.float32)
        else:
            self._background_image = None

    def _assemble_mesh_tensors(
        self,
        robot_config: Union[List[float], np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (verts, faces, verts_rgb) for the robot at the given joint configuration.

        All link meshes are transformed to world frame and concatenated.
        verts_rgb is (V, 3), with no batch dimension.
        """
        q = robot_config.cpu().numpy() if isinstance(robot_config, torch.Tensor) else robot_config
        R, t = self.robot.get_joint_R_t(q)
        R = _to_tensor(R).to(self.device)
        t = _to_tensor(t).to(self.device)

        if len(R) != len(self.preload_verts):
            raise ValueError(
                f"FK returned {len(R)} link poses but {len(self.preload_verts)} mesh "
                "groups are registered. mesh_files must match get_joint_R_t() output "
                "in order AND length."
            )

        verts_list, faces_list, verts_rgb_list = [], [], []
        vert_counter = 0
        for ri, ti, vi, fi, ci in zip(
            R, t, self.preload_verts, self.preload_faces, self.colors, strict=True
        ):
            for vii, fii, cii in zip(vi, fi, ci, strict=True):
                vii = vii @ ri.T + ti
                fii = fii + vert_counter
                vert_counter += int(vii.shape[0])
                verts_list.append(vii)
                faces_list.append(fii)
                verts_rgb_list.append(cii.to(dtype=vii.dtype, device=vii.device))

        return (
            torch.cat(verts_list, dim=0),
            torch.cat(faces_list, dim=0),
            torch.cat(verts_rgb_list, dim=0),
        )

    def transform_mesh2robot_config(
        self,
        robot_config: Union[List[float], np.ndarray, torch.Tensor],
    ) -> Meshes:
        """
        Build a single combined PT3D Meshes for the robot at the given
        joint configuration, with all link meshes transformed to world frame.
        """
        verts, faces, verts_rgb = self._assemble_mesh_tensors(robot_config)
        meshes = Meshes(
            verts=[verts], faces=[faces],
            textures=TexturesVertex(verts_rgb.unsqueeze(0)),
        )
        if meshes.isempty():
            raise RuntimeError("Assembled robot mesh is empty")
        return meshes

    def phong_render_batched(
        self,
        meshes_world: Meshes,
        R_batch: torch.Tensor,
        T_batch: torch.Tensor,
        backgrounds: Optional[List[List[float]]] = None,
    ) -> Tuple[torch.Tensor, object]:
        """
        Render meshes_world from N camera views in a single PT3D forward pass.

        Args:
            meshes_world: Single-batch Meshes (will be extended to N).
            R_batch:      (N, 3, 3) camera rotations.
            T_batch:      (N, 3)   camera translations.
            backgrounds:  Optional list of N [R,G,B] per-view background colors.
                          If self._background_image is set, it overrides this.

        Returns:
            rgba:      (N, H, W, 4) rendered RGBA on device.
            fragments: PT3D Fragments object with pix_to_face (N, H, W, K).
        """
        N = R_batch.shape[0]
        R_batch = R_batch.to(self.device)
        T_batch = T_batch.to(self.device)

        # Camera center in world frame: solve cam @ R + T = 0, so cam = -T @ R.T
        # R_batch is (N, 3, 3) so R_batch.transpose(-2,-1) gives R.T per view.
        cam_centers = -(T_batch.unsqueeze(1) @ R_batch.transpose(-2, -1)).squeeze(1)  # (N, 3)

        # Build only PerspectiveCameras (cheap) and update the existing renderer's
        # camera + lights in-place. This avoids recreating MeshRendererWithFragments,
        # MeshRasterizer, and HardPhongShader every call, which was the main source of
        # per-frame Python overhead in the previous implementation.
        cameras_batch = PerspectiveCameras(
            focal_length=self.cameras.focal_length.expand(N, -1),
            principal_point=self.cameras.principal_point.expand(N, -1),
            in_ndc=False,
            image_size=self.cameras.image_size.expand(N, -1),
            R=R_batch,
            T=T_batch,
            device=self.device,
        )
        self.phong_renderer.rasterizer.cameras = cameras_batch
        self.phong_renderer.shader.cameras = cameras_batch
        self.phong_renderer.shader.lights = PointLights(
            device=self.device,
            location=cam_centers,
            ambient_color=[[0.5, 0.5, 0.5]],
        )

        # Prevent coarse-bin overflow when the arm is small in frame (faces concentrate
        # into 1-2 bins). PT3D default max(10000, faces/5) can be too small; setting it
        # to the actual face count is the tight upper bound and has no runtime cost,
        # only memory: about 196 bins * n_faces * 4 bytes (about 23 MB for 30k faces).
        n_faces = int(meshes_world.num_faces_per_mesh().max().item())
        self.phong_renderer.rasterizer.raster_settings.max_faces_per_bin = max(n_faces, 10000)

        meshes_batch = meshes_world.extend(N)
        rgba, fragments = self.phong_renderer(meshes_world=meshes_batch)

        rgba = self._composite_backgrounds(rgba, backgrounds, fragments)
        return rgba, fragments

    def _composite_backgrounds(
        self,
        rgba: torch.Tensor,
        backgrounds: Optional[List[List[float]]],
        fragments=None,
    ) -> torch.Tensor:
        """
        Replace background pixels in rgba with per-view colors or a background image.

        Uses fragments.zbuf (z-depth buffer) when available for a precise foreground
        mask (any pixel with zbuf > -1 hit the mesh).  Falls back to a white-pixel
        threshold (all channels > 0.99) when fragments is None.
        """
        N = rgba.shape[0]
        # Build foreground mask: (N, H, W, 1), True = mesh pixel, False = background.
        if fragments is not None and hasattr(fragments, "zbuf"):
            fg_mask = (fragments.zbuf[..., 0] > -1).unsqueeze(-1)  # (N, H, W, 1)
        else:
            fg_mask = ~(rgba[..., :3] > 0.99).all(dim=-1, keepdim=True)  # (N, H, W, 1)

        if self._background_image is not None:
            bg = self._background_image.to(self.device)       # (H, W, 3)
            bg_batch = bg.unsqueeze(0).expand(N, -1, -1, -1) # (N, H, W, 3)
            rgba_rgb = torch.where(fg_mask, rgba[..., :3], bg_batch)
            rgba = torch.cat([rgba_rgb, rgba[..., 3:]], dim=-1)
        elif backgrounds is not None:
            # Build all background colours in one tensor (N, 3) then broadcast
            # over (H, W). This avoids a Python loop and N small tensor allocations.
            bg_colors = torch.tensor(
                [backgrounds[i % len(backgrounds)] for i in range(N)],
                dtype=torch.float32,
                device=self.device,
            )                                                  # (N, 3)
            bg_batch = bg_colors.view(N, 1, 1, 3)             # broadcast to (N, H, W, 3)
            rgba_rgb = torch.where(fg_mask, rgba[..., :3], bg_batch.expand_as(rgba[..., :3]))
            rgba = torch.cat([rgba_rgb, rgba[..., 3:]], dim=-1)
        return rgba

    def _unpack_camera_K_mat(
        self,
        camera_K_mat: Union[np.ndarray, torch.Tensor],
        image_size_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = (
            torch.from_numpy(camera_K_mat).to(dtype=torch.float32)
            if isinstance(camera_K_mat, np.ndarray)
            else camera_K_mat.to(dtype=torch.float32)
        ).to(device=self.device)
        focal_len = torch.tensor([[K[0,0].item(), K[1,1].item()]], dtype=torch.float32, device=self.device)
        principal_point = K[:2, 2].reshape(1, 2).to(torch.float32)
        if image_size_hw is not None:
            h, w = int(image_size_hw[0]), int(image_size_hw[1])
            image_size = torch.tensor([[h, w]], dtype=torch.float32, device=self.device)
        else:
            import warnings
            image_size = (2 * principal_point.flip(dims=[1])).to(torch.float32)
            warnings.warn(
                "image_size_hw not provided; inferring image size as 2*(cx, cy) from the "
                "intrinsic matrix. This is only correct when the principal point is exactly "
                "at the image centre. Pass image_size_hw=(H, W) explicitly to avoid this.",
                UserWarning,
                stacklevel=3,
            )
        return focal_len, principal_point, image_size

    def _build_mesh_with_colors(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        aux,
        file: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build mesh tensors with colors from OBJ material assignments.

        Priority:
          1. Explicit color_mapping override.
          2. PT3D aux.material_colors + aux.face_material_idx (fast path).
          3. Custom MTL parser (load_obj_with_mtl_vertex_colors), a reliable fallback
             when PT3D fails to parse non-standard MTL filenames (e.g. material.lib).
             Multi-material meshes use exploded topology to retain face boundaries.
          4. Uniform default color.
        """
        override = self._match_color_override(file)
        if override is not None:
            colors = override.unsqueeze(0).expand(verts.shape[0], -1)
            return verts, faces, colors

        mat_colors = getattr(aux, "material_colors", None)
        face_mat_idx = getattr(aux, "face_material_idx", None)
        if mat_colors and face_mat_idx is not None:
            mat_names = list(mat_colors.keys())
            kd_list = []
            for name in mat_names:
                mat = mat_colors[name]
                kd = mat.get("diffuse_color", torch.tensor([1.0, 1.0, 1.0]))
                kd_list.append(kd[:3].float())
            kd_tensor = torch.stack(kd_list)  # (M, 3)
            fmi = face_mat_idx.long()  # (F,)
            face_colors = torch.ones(faces.shape[0], 3, dtype=torch.float32)
            valid = (fmi >= 0) & (fmi < kd_tensor.shape[0])
            face_colors[valid] = kd_tensor[fmi[valid]]
            return _apply_face_colors(verts, faces, face_colors)

        # PT3D didn't parse the MTL. Try our own parser for colors, then
        # apply its per-face result to the PT3D triangle geometry.
        if file.lower().endswith(".obj"):
            fb = self._match_name_to_color_np(file)
            parsed = load_obj_with_mtl_vertex_colors(file, fb)
            if parsed is not None:
                # parsed uses exploded topology (3 verts/face); extract per-face color
                # (all 3 corners have the same Kd).
                c_exp = parsed[2]  # (3F, 3), one color per face corner
                face_colors_np = c_exp[::3]  # one color per face, shape (F, 3)
                if len(face_colors_np) == faces.shape[0]:
                    face_colors = torch.from_numpy(face_colors_np.astype(np.float32))
                    return _apply_face_colors(verts, faces, face_colors)
                # Face count mismatch: PT3D and our parser triangulated quads/ngons
                # with different strategies. Fall through to uniform-color fallback.

        color = self._match_name2color(file)  # (3,)
        colors = color.unsqueeze(0).expand(verts.shape[0], -1)
        return verts, faces, colors

    def _match_name_to_color_np(self, file: str) -> np.ndarray:
        t = self._match_name2color(file)
        return t.detach().cpu().numpy().astype(np.float64)

    def _match_color_override(self, file: str) -> Optional[torch.Tensor]:
        if not self._color_mapping:
            return None
        file_name = Path(file).stem
        for part in file_name.split("_"):
            if part in self._color_mapping:
                return self._color_mapping[part]
        return self._color_mapping.get("default")

    def _match_name2color(self, file: str) -> torch.Tensor:
        """Return a matching override or the uniform fallback color."""
        default = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32)
        override = self._match_color_override(file)
        return override if override is not None else default
