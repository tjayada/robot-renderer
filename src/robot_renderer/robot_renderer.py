from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import trimesh

from .config import ViewConfig
from .registry import get_robot_entry
from .renderer_pt3d import RobotPyTorch3DRenderer
from .template_pipeline_pt3d import render_templates
from .types import RenderResult


def _resolve_mesh_files(mesh_dir: Path) -> List[List[str]]:
    """Discover OBJ files in mesh_dir and return them grouped per-link."""
    mesh_dir = Path(mesh_dir)
    if not mesh_dir.is_dir():
        return []
    subdirs = sorted([d for d in mesh_dir.iterdir() if d.is_dir()])
    if subdirs:
        groups = []
        for sd in subdirs:
            objs = sorted(sd.glob("*.obj")) + sorted(sd.glob("*.OBJ"))
            if objs:
                groups.append([str(o) for o in objs])
        return groups
    objs = sorted(mesh_dir.glob("*.obj")) + sorted(mesh_dir.glob("*.OBJ"))
    return [[str(o)] for o in objs]


def _asset_hint(name: str) -> str:
    if name in {"owi535", "meca500"}:
        return f" Run `make assets-{name}` from the repository root."
    return ""


def _validate_asset_paths(name: str, urdf: Path, mesh_files) -> None:
    if not urdf.is_file():
        raise FileNotFoundError(
            f"URDF file for robot '{name}' is missing: {urdf}.{_asset_hint(name)}"
        )
    missing = []
    for group in mesh_files:
        paths = [group] if isinstance(group, (str, Path)) else group
        missing.extend(str(path) for path in paths if not Path(path).is_file())
    if missing:
        preview = ", ".join(missing[:3])
        if len(missing) > 3:
            preview += f", and {len(missing) - 3} more"
        raise FileNotFoundError(
            f"Robot '{name}' is missing mesh files: {preview}.{_asset_hint(name)}"
        )


def _mesh_transform_from_center(center: np.ndarray, scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center
    return transform


class RobotRenderer:
    """
    High-level robot renderer (PyTorch3D backend).

    Looks up the robot in the registry, builds a :class:`RobotPyTorch3DRenderer`,
    and exposes :meth:`render_templates` as the single entry point.

    Args:
        name:          Registered robot name (e.g. ``"panda"``), or a custom name
                       after calling :func:`robot_renderer.register`.
        K:             (3, 3) camera intrinsics matrix.
        config:        :class:`ViewConfig` controlling resolution, viewset, orientation, etc.
        device:        Torch device for rendering and all tensor outputs.
                       Defaults to CPU (PyTorch convention); pass "cuda" or
                       call .to("cuda") to use the GPU.
        mesh_files:    Optional per-link OBJ list; overrides auto-discovery from the registry.
        color_mapping: Optional keyword-to-RGB dict passed to the renderer for per-link color
                       overrides. Keys are matched against the ``_``-separated tokens of the
                       mesh filename stem; ``"default"`` acts as a fallback when no token
                       matches. When ``None`` (default), colors come from the OBJ MTL files.
    """

    def __init__(
        self,
        name: str,
        K: Union[np.ndarray, torch.Tensor],
        config: Optional[ViewConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        mesh_files: Optional[List[List[str]]] = None,
        color_mapping: Optional[dict] = None,
    ):
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.config = config or ViewConfig()
        self.name = name

        entry = get_robot_entry(name)
        adapter_cls = entry["adapter_cls"]
        if adapter_cls is None:
            raise NotImplementedError(
                f"No adapter_cls registered for '{name}'. "
                "Provide one via robot_renderer.register(..., adapter_cls=MyAdapter)."
            )
        if mesh_files is None:
            mesh_files = entry.get("mesh_files") or _resolve_mesh_files(entry["mesh_dir"])
        if not mesh_files:
            raise FileNotFoundError(
                f"No mesh files found for robot '{name}' in {entry['mesh_dir']}. "
                "Check mesh_dir or provide mesh_files explicitly."
            )
        _validate_asset_paths(name, Path(entry["urdf"]), mesh_files)
        self._kinematics = adapter_cls(entry["urdf"])

        K_np = K.cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K, dtype=np.float64)
        self._K = torch.tensor(K_np, dtype=torch.float32, device=device)

        self._renderer = RobotPyTorch3DRenderer(
            robot=self._kinematics,
            mesh_files=mesh_files,
            camera_K_mat=K_np,
            image_size_hw=(self.config.render_size, self.config.render_size),
            device=device,
            color_mapping=color_mapping,
        )

    def to(self, device: Union[str, torch.device]) -> "RobotRenderer":
        """Move renderer state and tensor outputs to *device*. Returns self for chaining."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self._K = self._K.to(device)
        self._renderer.to(device)
        return self

    def render_templates(
        self,
        joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
        background_image: Optional[torch.Tensor] = None,
        config_override: Optional[ViewConfig] = None,
    ) -> RenderResult:
        """
        Render multi-view templates for the given joint configuration.

        Args:
            joint_angles:     Robot joint angles (length matches the registered adapter).
            background_image: Optional (H, W, 3) or (3, H, W) float tensor to use as
                              background behind the robot in all rendered views.
            config_override:  Override ``self.config`` for this call only.

        Returns:
            :class:`RenderResult` with all tensors on ``self.device``.
        """
        cfg = config_override or self.config
        if cfg.render_size != self.config.render_size:
            raise ValueError(
                "config_override.render_size must match the renderer construction "
                f"size ({self.config.render_size}), got {cfg.render_size}. "
                "Construct a new RobotRenderer for a different render size."
            )
        result = render_templates(
            renderer=self._renderer,
            joint_angles=joint_angles,
            config=cfg,
            background_image=background_image,
            robot_kinematics=self._kinematics,
        )
        return _result_to_device(result, self.device)

    def _build_posed_trimesh(
        self,
        joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
        scale: float,
    ) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        Assemble the posed robot as a single ``trimesh.Trimesh`` with vertex
        colors, vertices scaled by *scale* and centred at the bounding-box
        centre.

        Returns:
            mesh:   ``trimesh.Trimesh``, centred, vertex colours preserved.
            center: (3,) bbox centre that was subtracted (in scaled units).
        """
        verts, faces, verts_rgb = self._renderer._assemble_mesh_tensors(joint_angles)

        verts_np = verts.cpu().numpy().astype(np.float64) * scale
        faces_np = faces.cpu().numpy()
        colors_np = (verts_rgb.cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)

        center = (verts_np.max(axis=0) + verts_np.min(axis=0)) / 2.0
        rgba = np.concatenate(
            [colors_np, np.full((len(colors_np), 1), 255, dtype=np.uint8)], axis=1
        )
        mesh = trimesh.Trimesh(
            vertices=verts_np - center, faces=faces_np, vertex_colors=rgba, process=False,
        )
        return mesh, center

    def export_posed_trimesh(
        self,
        joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
    ) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        Return the robot at the given joint configuration as an in-memory
        ``trimesh.Trimesh`` in millimetres, centred at the bounding-box centre.

        Use this when another renderer expects an in-memory mesh in millimetres.

        Returns:
            mesh           : ``trimesh.Trimesh``, vertices in mm, origin at
                             bounding-box centre, vertex colours preserved.
            mesh_transform : (4, 4) float64 ndarray that maps URDF-base (metres) to
                             the centred-mm mesh frame.  NOTE: [:3,:3] = 1000*I
                             (a scale, not a rotation), so you CANNOT simply
                             compute ``T_m2c @ mesh_transform``, because that embeds the
                             1000 scale factor into the rotation block. Recover
                             ``T_base_cam`` in metres as::

                                 R = T_m2c[:3, :3]
                                 t = T_m2c[:3, 3]
                                 center_mm = -mesh_transform[:3, 3]
                                 T_base_cam[:3, :3] = R
                                 T_base_cam[:3, 3]  = (t - R @ center_mm) / 1000.0
        """
        mesh, center_mm = self._build_posed_trimesh(joint_angles, scale=1000.0)

        # mesh_transform maps the URDF base in m to the centred mesh frame in mm:
        #   v_centered_mm = v_urdf_m * 1000 - center_mm
        mesh_transform = _mesh_transform_from_center(center_mm, scale=1000.0)

        return mesh, mesh_transform

    def export_posed_mesh(
        self,
        joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
        output_path: Union[str, Path],
        scale_mm: bool = True,
    ) -> Tuple[Path, np.ndarray]:
        """
        Export the robot at the given joint configuration as a single merged mesh.

        Assembles all link meshes (with FK transforms applied) into one mesh and
        writes it to disk. Vertex colors are preserved from the per-link color
        mapping. Suitable for callers that consume a posed CAD mesh directly.

        Args:
            joint_angles: Robot joint angles.
            output_path:  Output file path. Format is inferred from the extension;
                          ``.ply`` (recommended, keeps vertex colors) and
                          ``.obj`` are both supported.
            scale_mm:     If True (default), multiply vertices by 1000 to convert
                          from URDF metres to millimetres.

        Returns:
            output_path: Resolved absolute output path.
            mesh_transform: (4, 4) transform from the URDF base frame to the
                            exported centred mesh frame.
        """
        scale = 1000.0 if scale_mm else 1.0
        mesh, center = self._build_posed_trimesh(joint_angles, scale=scale)

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
        return output_path, _mesh_transform_from_center(center, scale=scale)


def _result_to_device(result: RenderResult, device: torch.device) -> RenderResult:
    """Move all tensors in a RenderResult to *device*."""
    from .types import RenderedView
    views = [
        RenderedView(
            image=v.image.to(device),
            R=v.R.to(device),
            T=v.T.to(device),
            gt_pose=v.gt_pose.to(device),
        )
        for v in result.views
    ]
    return RenderResult(
        views=views,
        mesh_transform=result.mesh_transform.to(device),
        mesh_verts=result.mesh_verts.to(device),
        visible_surface_points=result.visible_surface_points.to(device),
        sphere_radius=result.sphere_radius,
        anchor_gt_pose=result.anchor_gt_pose.to(device),
        anchor_idx=result.anchor_idx,
        orientation_center=(
            result.orientation_center.to(device)
            if result.orientation_center is not None else None
        ),
        zbuf=result.zbuf.to(device) if result.zbuf is not None else None,
    )
