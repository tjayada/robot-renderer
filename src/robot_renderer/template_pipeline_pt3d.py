from __future__ import annotations
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from ._pt3d import _require_pt3d
from .config import ViewConfig
from .mesh_ops import orient_mesh
from .renderer_pt3d import RobotPyTorch3DRenderer
from .types import RenderedView, RenderResult
from .view_strategy import score_and_sort_views, select_diverse_views

try:
    from pytorch3d.renderer import TexturesVertex, look_at_view_transform
    from pytorch3d.structures import Meshes
except ImportError:
    look_at_view_transform = None  # type: ignore[assignment]
    Meshes = TexturesVertex = None  # type: ignore[assignment]


def _rt_batch_to_T_m2c_opencv(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Convert batched PyTorch3D (R, T) to (N, 4, 4) OpenCV-style T_m2c.

    PT3D convention (row vectors, Y-up, Z-toward-viewer):
        p_view = p_world @ R + T
    So the PT3D world-to-view matrix is [[R, 0], [T, 1]].

    .transpose(1, 2) converts to column-major: [[R.T, T.T], [0, 1]].
    C = diag([-1,-1,1,1]) then applies the PyTorch3D-to-OpenCV coordinate fix.
    """
    N = R.shape[0]
    dev = R.device
    C = torch.diag(torch.tensor([-1., -1., 1., 1.], dtype=torch.float32, device=dev))

    # Build the PT3D world-to-view matrix directly. This avoids constructing PerspectiveCameras.
    M = torch.zeros(N, 4, 4, dtype=torch.float32, device=dev)
    M[:, :3, :3] = R      # rotation block
    M[:, 3, :3] = T       # translation row
    M[:, 3, 3] = 1.0

    return C.view(1, 4, 4) @ M.transpose(1, 2)


def fill_frame_crop(
    images: torch.Tensor,   # (N, 3, H, W) rendered views, float [0,1]
    zbuf: torch.Tensor,     # (N, H, W) metric depth, background = -1
    out_size: int,
    pad_frac: float = 0.1,
    min_px: int = 64,
    antialias: bool = True,
) -> torch.Tensor:
    """Per-view: crop each rendered view to its object silhouette (square, padded,
    shifted to stay in-frame) and resize to
    out_size, so every template fills the frame to a consistent fraction regardless
    of viewing angle / joint config. Views with too few silhouette pixels are left
    full-frame. Returns (N, 3, out_size, out_size).

    Only the images change; template cameras and 3D geometry are untouched.
    """
    N, _, H, W = images.shape
    out = images.new_empty(N, 3, out_size, out_size)
    for i in range(N):
        fg = zbuf[i] > -1.0  # silhouette: foreground depth is positive, bg = -1
        if int(fg.sum().item()) < min_px:
            crop = images[i:i + 1]  # degenerate view: keep full frame
        else:
            xs = torch.where(fg.any(dim=0))[0]
            ys = torch.where(fg.any(dim=1))[0]
            x0, x1 = int(xs[0].item()), int(xs[-1].item()) + 1
            y0, y1 = int(ys[0].item()), int(ys[-1].item()) + 1
            s = min(float(max(x1 - x0, y1 - y0)) * (1.0 + pad_frac), float(H), float(W))
            s_i = max(1, int(s))
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            bx = int(min(max(cx - s / 2.0, 0.0), W - s_i))
            by = int(min(max(cy - s / 2.0, 0.0), H - s_i))
            crop = images[i:i + 1, :, by:by + s_i, bx:bx + s_i]
        out[i] = F.interpolate(
            crop, size=(out_size, out_size),
            mode="bilinear", align_corners=False, antialias=antialias,
        )[0].clamp(0.0, 1.0)
    return out


def render_templates(
    renderer: RobotPyTorch3DRenderer,
    joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
    config: ViewConfig,
    background_image: Optional[torch.Tensor] = None,
    robot_kinematics=None,
) -> RenderResult:
    """
    Render multi-view templates using PyTorch3D Phong shading.

    View placement uses ``look_at_view_transform``; ``gt_pose`` / ``R`` / ``T`` are in
    OpenCV-style T_m2c convention (Z-forward, Y-down).
    """
    _require_pt3d()
    device = renderer.device

    if isinstance(joint_angles, torch.Tensor):
        joints_np = joint_angles.detach().cpu().numpy().astype(np.float64)
        q_tensor = joint_angles.to(device=device, dtype=torch.float32)
    else:
        joints_np = np.asarray(joint_angles, dtype=np.float64)
        q_tensor = torch.as_tensor(joints_np, dtype=torch.float32, device=device)

    verts_t, faces_t, vert_rgb = renderer._assemble_mesh_tensors(q_tensor)
    verts_t = verts_t.to(device=device, dtype=torch.float32)
    faces_t = faces_t.to(device=device, dtype=torch.long)
    vert_rgb = vert_rgb.to(device=device, dtype=torch.float32)

    verts_aligned_t, mesh_transform, center_t = orient_mesh(
        verts=verts_t,
        orientation=config.orientation,
        device=device,
        robot_kinematics=robot_kinematics,
        joint_angles=joints_np,
    )

    meshes_aligned = Meshes(
        verts=[verts_aligned_t],
        faces=[faces_t],
        textures=TexturesVertex(verts_features=vert_rgb.unsqueeze(0)),
    )

    verts_min = verts_aligned_t.min(dim=0).values
    verts_max = verts_aligned_t.max(dim=0).values
    center_bbox = (verts_min + verts_max) / 2.0
    center_bbox_np = center_bbox.detach().cpu().numpy().astype(np.float64)

    sphere_radius = float(
        (verts_aligned_t - center_bbox).norm(dim=1).max().item()
    )
    dist = sphere_radius * config.sphere_distance_factor

    candidate_views = config.candidate_views
    if not candidate_views:
        raise RuntimeError(
            f"Viewset '{config.viewset}' with elevation_range={config.elevation_range} "
            "produced zero candidate views. Widen elevation_range or increase fibonacci N."
        )

    scored_views = score_and_sort_views(candidate_views, verts_aligned_t, faces_t)
    if config.diverse_selection:
        final_views = select_diverse_views(scored_views, config.num_views,
                                           anchor_elevation_range=config.anchor_elevation_range,
                                           device=device)
    else:
        final_views = scored_views[: config.num_views]

    # Vectorised camera placement: all N eye positions computed at once with numpy broadcasting.
    azimuths = np.radians([az for az, _ in final_views])   # (N,)
    elevations = np.radians([el for _, el in final_views]) # (N,)
    cos_el = np.cos(elevations)
    offsets = np.stack(                                     # (N, 3)
        [cos_el * np.sin(azimuths), np.sin(elevations), cos_el * np.cos(azimuths)], axis=1
    )
    eyes_t = torch.from_numpy(
        center_bbox_np + dist * offsets                     # (N, 3) float64
    ).to(dtype=torch.float32, device=device)
    at_t = torch.tensor(center_bbox_np, dtype=torch.float32, device=device).unsqueeze(0).expand(
        len(final_views), -1
    )

    R_cam, T_cam = look_at_view_transform(
        eye=eyes_t,
        at=at_t,
        up=((0.0, 1.0, 0.0),),
        device=device,
    )

    prev_bg = renderer._background_image
    try:
        if background_image is not None:
            bg = background_image.detach()
            if bg.shape[0] == 3:
                bg = bg.permute(1, 2, 0)
            renderer.set_background_image(bg.to(device=device, dtype=torch.float32))
        else:
            renderer.set_background_image(None)

        rgba, fragments = renderer.phong_render_batched(
            meshes_aligned,
            R_cam,
            T_cam,
            backgrounds=config.backgrounds,
        )
    finally:
        renderer.set_background_image(prev_bg)

    # zbuf: (N, H, W, K); keep nearest face only for (N, H, W).
    # Positive = metric camera-Z of the nearest mesh surface; -1 = background.
    zbuf = fragments.zbuf[..., 0]  # (N, H, W)

    # Clipping guard: if the arm's silhouette touches the render border, part of
    # it lies outside the frustum and templates (and zbuf anchors) are corrupted.
    # Fix by increasing sphere_distance_factor (large arms under real-camera
    # intrinsics can clip at factors that look generous, e.g. 2.25).
    fg_border = (
        (zbuf[:, 0, :] > 0).any(-1) | (zbuf[:, -1, :] > 0).any(-1)
        | (zbuf[:, :, 0] > 0).any(-1) | (zbuf[:, :, -1] > 0).any(-1)
    )
    if bool(fg_border.any()):
        import logging
        logging.getLogger(__name__).warning(
            "render_templates: %d/%d views have the arm touching the render "
            "border, so the mesh is clipped by the frustum; increase "
            "sphere_distance_factor (current distance = %.2f x bounding-sphere "
            "radius).", int(fg_border.sum()), zbuf.shape[0],
            config.sphere_distance_factor,
        )

    # Visibility-filtered surface points: face centroids of all faces that are
    # visible in at least one of the N rendered views.
    # pix_to_face: (N, H, W, 1), the GLOBAL face index per pixel in the packed
    # batch, -1 for background.  meshes_aligned.extend(N) creates N copies of
    # the mesh, so image n's faces are at global indices [n*F, (n+1)*F).
    # % F converts to LOCAL indices valid for indexing into faces_t (shape (F,3)).
    pix_to_face = fragments.pix_to_face[..., 0]          # (N, H, W)
    F = faces_t.shape[0]
    global_idxs = pix_to_face[pix_to_face >= 0]
    visible_face_idxs = (global_idxs % F).unique()       # local indices in [0, F)
    v0 = verts_aligned_t[faces_t[visible_face_idxs, 0]]
    v1 = verts_aligned_t[faces_t[visible_face_idxs, 1]]
    v2 = verts_aligned_t[faces_t[visible_face_idxs, 2]]
    visible_surface_points = (v0 + v1 + v2) / 3.0        # (M, 3) face centroids

    gt_poses = _rt_batch_to_T_m2c_opencv(R_cam, T_cam)

    n_views = len(final_views)
    images = rgba[..., :3].clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)

    # Per-view fill-frame crop: make each template fill the frame to a
    # consistent fraction (zbuf gives the silhouette). Images-only; geometry untouched.
    if config.fill_frame:
        images = fill_frame_crop(
            images, zbuf, config.render_size,
            pad_frac=config.fill_frame_pad, min_px=config.fill_frame_min_px,
        )

    rendered_views: List[RenderedView] = [
        RenderedView(
            image=images[i],
            R=gt_poses[i, :3, :3].clone(),
            T=gt_poses[i, :3, 3].clone(),
            gt_pose=gt_poses[i].clone(),
        )
        for i in range(n_views)
    ]

    return RenderResult(
        views=rendered_views,
        mesh_transform=mesh_transform,
        mesh_verts=verts_aligned_t,
        visible_surface_points=visible_surface_points,
        sphere_radius=sphere_radius,
        anchor_gt_pose=rendered_views[0].gt_pose,
        anchor_idx=0,
        orientation_center=center_t,
        zbuf=zbuf,
    )
