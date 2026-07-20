from __future__ import annotations
import math
from typing import Optional, Tuple

import torch


def rotation_align_vector_to_up(
    v: torch.Tensor,
    up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute a 3 x 3 rotation matrix R such that R @ v is approximately up.

    Args:
        v:  (3,) source direction.
        up: (3,) target direction. Defaults to Y-up [0, 1, 0].

    Returns:
        R: (3, 3) rotation matrix on the same device as v.
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=v.dtype, device=v.device)

    v_n = v / (v.norm() + 1e-8)
    up_n = up / (up.norm() + 1e-8)
    cos_theta = torch.clamp(torch.dot(v_n, up_n), -1.0, 1.0)

    if cos_theta > 1.0 - 1e-4:
        return torch.eye(3, dtype=v.dtype, device=v.device)
    if cos_theta < -1.0 + 1e-4:
        tmp = torch.tensor([1.0, 0.0, 0.0], dtype=v.dtype, device=v.device)
        if torch.allclose(v_n.abs(), tmp.abs(), atol=1e-3):
            tmp = torch.tensor([0.0, 1.0, 0.0], dtype=v.dtype, device=v.device)
        k = torch.linalg.cross(v_n, tmp)
        k = k / (k.norm() + 1e-8)
        angle = torch.tensor(math.pi, dtype=v.dtype, device=v.device)
    else:
        k = torch.linalg.cross(v_n, up_n)
        k = k / (k.norm() + 1e-8)
        angle = torch.acos(cos_theta)

    K_mat = torch.zeros((3, 3), dtype=v.dtype, device=v.device)
    K_mat[0, 1] = -k[2]
    K_mat[0, 2] =  k[1]
    K_mat[1, 0] =  k[2]
    K_mat[1, 2] = -k[0]
    K_mat[2, 0] = -k[1]
    K_mat[2, 1] =  k[0]
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + torch.sin(angle) * K_mat + (1.0 - torch.cos(angle)) * (K_mat @ K_mat)


def _pca_principal_axis(verts: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Return the principal axis (largest-eigenvalue eigenvector) of the centred vertex cloud."""
    X = verts - center
    cov = (X.T @ X) / float(max(X.shape[0] - 1, 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs[:, int(torch.argmax(eigvals).item())]


def orient_mesh(
    verts: torch.Tensor,
    orientation: str,
    device: torch.device,
    robot_kinematics=None,
    joint_angles=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Orient a robot mesh and return the aligned vertices with the 4 x 4 rigid
    transform from the URDF frame to the aligned frame.

    Args:
        verts:            (V, 3) float32 tensor of mesh vertices.
        orientation:      One of "simple_upright", "align_mesh_longest_axis_upright",
                          "align_mesh_longest_axis_top_right", or "end_effector_frame"
                          (rotate into the EE orientation about the EE origin;
                          the mesh is not re-centred).
        device:           Torch device for intermediate computations.
        robot_kinematics: Required only for "end_effector_frame".
        joint_angles:     Required only for "end_effector_frame".

    Returns:
        verts_aligned:  (V, 3) float32, vertices after orientation transform.
        mesh_transform: (4, 4) float32, the rigid transform from URDF to aligned frame.
        center:         (3,)   float32, orientation pivot in URDF frame.
    """
    verts = verts.to(device=device, dtype=torch.float32)
    center = (verts.max(dim=0).values + verts.min(dim=0).values) / 2.0

    if orientation == "simple_upright":
        base_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        R_align = rotation_align_vector_to_up(base_z)

    elif orientation == "align_mesh_longest_axis_upright":
        principal = _pca_principal_axis(verts, center)
        R_align = (
            torch.eye(3, dtype=torch.float32, device=device)
            if principal.norm() < 1e-8
            else rotation_align_vector_to_up(principal)
        )

    elif orientation == "align_mesh_longest_axis_top_right":
        # Align principal axis to the upper-right diagonal (45 degree X-right/Y-up) so the
        # arm appears in the top-right corner of the rendered view.
        principal = _pca_principal_axis(verts, center)
        top_right = torch.tensor(
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0],
            dtype=torch.float32, device=device,
        )
        R_align = (
            torch.eye(3, dtype=torch.float32, device=device)
            if principal.norm() < 1e-8
            else rotation_align_vector_to_up(principal, up=top_right)
        )

    elif orientation == "end_effector_frame":
        # Rotates the mesh into the end-effector's orientation, pivoting about
        # the EE origin (which stays in place; the mesh is NOT translated to
        # put the EE at the world origin).
        import numpy as np
        if robot_kinematics is None or joint_angles is None:
            raise ValueError("end_effector_frame requires robot_kinematics and joint_angles")
        q = np.asarray(joint_angles, dtype=np.float64)
        R_ee_arr, t_ee_arr = robot_kinematics.get_joint_R_t(q)
        R_ee = torch.tensor(R_ee_arr[-1], dtype=torch.float32, device=device)
        t_ee = torch.tensor(t_ee_arr[-1], dtype=torch.float32, device=device)
        R_align = R_ee.T
        center = t_ee  # EE origin as pivot

    else:
        raise ValueError(
            f"Unknown orientation {orientation!r}. "
            "Choose 'simple_upright', 'align_mesh_longest_axis_upright', "
            "'align_mesh_longest_axis_top_right', or 'end_effector_frame'."
        )

    verts_aligned = (verts - center) @ R_align.T + center

    mesh_transform = torch.eye(4, dtype=torch.float32, device=device)
    mesh_transform[:3, :3] = R_align
    mesh_transform[:3, 3] = (torch.eye(3, device=device) - R_align) @ center

    return verts_aligned, mesh_transform, center
