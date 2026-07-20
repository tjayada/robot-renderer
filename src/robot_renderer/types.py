from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class RenderedView:
    """One rendered template view.

    All poses are OpenCV-convention T_m2c in the *aligned* mesh frame;
    translations are in metres.
    """
    image: torch.Tensor      # (3, H, W) float [0, 1] at render_size
    R: torch.Tensor          # (3, 3) camera rotation  (OpenCV convention)
    T: torch.Tensor          # (3,)   camera translation, metres (OpenCV convention)
    gt_pose: torch.Tensor    # (4, 4) T_m2c, translation in metres (OpenCV convention)


@dataclass
class RenderResult:
    """Full output of render_templates().

    All geometry is in **metres**, expressed in the aligned mesh frame
    (see ``mesh_transform``). The only millimetre interfaces in the package
    are ``export_posed_trimesh`` and ``export_posed_mesh``.
    """

    views: List[RenderedView]
    mesh_transform: torch.Tensor          # (4, 4) URDF-to-aligned transform, metres
    mesh_verts: torch.Tensor              # (V, 3) aligned mesh vertices, metres
    visible_surface_points: torch.Tensor  # (M, 3) visible face centroids, metres
    sphere_radius: float                  # bounding-sphere radius, metres
    anchor_gt_pose: torch.Tensor          # (4, 4) OpenCV convention, translation in metres
    anchor_idx: int = 0
    orientation_center: Optional[torch.Tensor] = None  # (3,) orientation pivot in URDF frame, metres
    zbuf: Optional[torch.Tensor] = None   # (N, H, W) camera-Z depth per view, metres; -1 = background
