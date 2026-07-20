"""
Unit tests for the PyTorch3D-free parts of robot-renderer: view sampling,
mesh orientation, the OBJ+MTL fallback parser, the fill-frame crop, and the
FK adapters. These run on any machine with the base dependencies installed
(torch, numpy, roboticstoolbox). No PyTorch3D or GPU required.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_renderer.config import ViewConfig
from robot_renderer.mesh_ops import orient_mesh, rotation_align_vector_to_up
from robot_renderer.obj_mesh_colors import load_obj_with_mtl_vertex_colors
from robot_renderer.registry import get_robot_entry, register
from robot_renderer.renderer_pt3d import (
    RobotPyTorch3DRenderer,
    _apply_face_colors,
)
from robot_renderer.robot_renderer import (
    RobotRenderer,
    _mesh_transform_from_center,
    _validate_asset_paths,
)
from robot_renderer.template_pipeline_pt3d import fill_frame_crop
from robot_renderer.view_strategy import (
    build_viewset,
    fibonacci_sphere_directions,
    score_and_sort_views,
    select_diverse_views,
)

ROBOTS_DIR = Path(__file__).resolve().parents[1] / "src" / "robot_renderer" / "robots"


# ---------------------------------------------------------------------------
# view_strategy
# ---------------------------------------------------------------------------


def test_build_viewset_fibonacci_count_and_elevation_band():
    views = build_viewset("fibonacci_64", num_views=8, elevation_range=(-5.0, 5.0))
    assert len(views) == 64
    for _, el in views:
        assert -5.0 <= el <= 5.0


def test_build_viewset_presets():
    assert len(build_viewset("four_sides", 99)) == 4
    assert len(build_viewset("eight_sides", 99)) == 8
    assert build_viewset("ring_8", 99) == build_viewset("eight_sides", 99)
    hemi = build_viewset("hemisphere_8", 99)
    assert len(hemi) == 8
    assert {el for _, el in hemi} == {0.0, 30.0}


def test_build_viewset_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown viewset"):
        build_viewset("fibonnaci_256", num_views=8)  # deliberate typo


def test_fibonacci_directions_are_unit_length():
    dirs = fibonacci_sphere_directions(128)
    assert dirs.shape == (128, 3)
    assert torch.allclose(dirs.norm(dim=1), torch.ones(128), atol=1e-5)


def test_select_diverse_views_count_and_anchor():
    views = build_viewset("fibonacci_64", num_views=8, elevation_range=(-30.0, 30.0))
    selected = select_diverse_views(views, num_views=8)
    assert len(selected) == 8
    assert len(set(selected)) == 8          # no duplicates
    assert selected[0] == views[0]          # anchor = first (best-scored) view


def test_score_and_sort_prefers_face_on_view():
    # A single large triangle facing +Z: the (az=0, el=0) direction looks
    # straight down +Z and must out-score the edge-on (az=90, el=0) view.
    verts = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    ordered = score_and_sort_views([(0.0, 0.0), (90.0, 0.0)], verts, faces)
    assert ordered[0] == (0.0, 0.0)


def test_candidate_views_reflect_config_edits():
    cfg = ViewConfig(viewset="fibonacci_32")
    assert len(cfg.candidate_views) == 32
    cfg.viewset = "four_sides"              # must not return stale cached views
    assert len(cfg.candidate_views) == 4


# ---------------------------------------------------------------------------
# mesh_ops
# ---------------------------------------------------------------------------


def test_rotation_align_vector_to_up():
    torch.manual_seed(0)
    up = torch.tensor([0.0, 1.0, 0.0])
    for v in [
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.0, -1.0, 0.0]),     # antiparallel branch
        torch.randn(3),
    ]:
        R = rotation_align_vector_to_up(v)
        aligned = R @ (v / v.norm())
        assert torch.allclose(aligned, up, atol=1e-4)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)  # proper rotation


def test_orient_mesh_transform_consistency():
    # verts_aligned must equal applying the returned 4x4 mesh_transform to verts.
    torch.manual_seed(1)
    verts = torch.randn(200, 3)
    aligned, T, center = orient_mesh(verts, "simple_upright", device=torch.device("cpu"))
    homog = torch.cat([verts, torch.ones(200, 1)], dim=1)
    applied = (T @ homog.T).T[:, :3]
    assert torch.allclose(applied, aligned, atol=1e-5)
    bbox_center = (verts.max(dim=0).values + verts.min(dim=0).values) / 2.0
    assert torch.allclose(center, bbox_center)


def test_orient_mesh_unknown_orientation_raises():
    with pytest.raises(ValueError, match="Unknown orientation"):
        orient_mesh(torch.randn(10, 3), "not_a_mode", device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# obj_mesh_colors
# ---------------------------------------------------------------------------


def test_obj_mtl_parser_applies_kd_per_face(tmp_path):
    (tmp_path / "cube.mtl").write_text(
        "newmtl red\nKd 1.0 0.0 0.0\nnewmtl blue\nKd 0.0 0.0 1.0\n"
    )
    (tmp_path / "tri.obj").write_text(
        "mtllib cube.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "usemtl red\nf 1 2 3\n"
        "usemtl blue\nf 2 4 3\n"
    )
    parsed = load_obj_with_mtl_vertex_colors(str(tmp_path / "tri.obj"), np.ones(3))
    assert parsed is not None
    verts, faces, colors = parsed
    assert verts.shape == (6, 3)            # exploded: 3 verts per face
    assert faces.shape == (2, 3)
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0])   # red face
    np.testing.assert_allclose(colors[3], [0.0, 0.0, 1.0])   # blue face


def test_obj_mtl_parser_default_color_for_missing_material(tmp_path):
    (tmp_path / "plain.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    fallback = np.array([0.2, 0.4, 0.6])
    parsed = load_obj_with_mtl_vertex_colors(str(tmp_path / "plain.obj"), fallback)
    assert parsed is not None
    np.testing.assert_allclose(parsed[2][0], fallback)


def test_face_material_boundaries_duplicate_shared_vertices():
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    face_colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    out_verts, out_faces, out_colors = _apply_face_colors(
        verts, faces, face_colors
    )

    assert out_verts.shape == (6, 3)
    assert torch.equal(out_faces, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert torch.all(out_colors[:3] == face_colors[0])
    assert torch.all(out_colors[3:] == face_colors[1])
    assert torch.equal(out_verts[out_faces].reshape(-1, 3), verts[faces].reshape(-1, 3))


def test_explicit_color_override_precedes_obj_materials():
    renderer = RobotPyTorch3DRenderer.__new__(RobotPyTorch3DRenderer)
    renderer._color_mapping = {"arm": torch.tensor([0.2, 0.4, 0.6])}
    verts = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]])
    aux = SimpleNamespace(
        material_colors={"red": {"diffuse_color": torch.tensor([1.0, 0.0, 0.0])}},
        face_material_idx=torch.tensor([0]),
    )

    out_verts, out_faces, out_colors = renderer._build_mesh_with_colors(
        verts, faces, aux, "robot_arm.obj"
    )

    assert out_verts is verts
    assert out_faces is faces
    assert torch.allclose(out_colors, torch.tensor([[0.2, 0.4, 0.6]]).expand(3, -1))


# ---------------------------------------------------------------------------
# fill_frame_crop
# ---------------------------------------------------------------------------


def test_fill_frame_crop_enlarges_small_silhouette():
    H = W = 64
    images = torch.zeros(1, 3, H, W)
    zbuf = torch.full((1, H, W), -1.0)
    images[0, :, 28:36, 28:36] = 1.0        # small 8x8 white square
    zbuf[0, 28:36, 28:36] = 0.5
    out = fill_frame_crop(images, zbuf, out_size=H, pad_frac=0.0, min_px=4)
    assert out.shape == (1, 3, H, W)
    # The 8px silhouette (12.5% of width) must now fill most of the frame.
    fg_frac = (out[0, 0] > 0.5).float().mean().item()
    assert fg_frac > 0.5


def test_fill_frame_crop_keeps_degenerate_view_full_frame():
    H = W = 64
    images = torch.rand(1, 3, H, W)
    zbuf = torch.full((1, H, W), -1.0)      # no silhouette at all
    out = fill_frame_crop(images, zbuf, out_size=H, min_px=4)
    assert torch.allclose(out, images.clamp(0.0, 1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# public API guards and path handling
# ---------------------------------------------------------------------------


def test_custom_registry_paths_are_resolved_at_registration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    register(
        "relative_path_test",
        urdf="robot.urdf",
        mesh_dir="meshes",
        mesh_files=[["meshes/link.obj"]],
    )
    entry = get_robot_entry("relative_path_test")
    assert entry["urdf"] == tmp_path / "robot.urdf"
    assert entry["mesh_dir"] == tmp_path / "meshes"
    assert entry["mesh_files"] == [[str(tmp_path / "meshes" / "link.obj")]]


def test_optional_asset_error_has_fetch_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"make assets-owi535"):
        _validate_asset_paths("owi535", tmp_path / "owi535.urdf", [])


def test_render_size_override_rejected_before_rendering():
    renderer = RobotRenderer.__new__(RobotRenderer)
    renderer.config = ViewConfig(render_size=64)
    with pytest.raises(ValueError, match="must match"):
        renderer.render_templates(
            joint_angles=np.zeros(1),
            config_override=ViewConfig(render_size=32),
        )


def test_mesh_transform_maps_base_units_to_centered_export_units():
    center = np.array([10.0, 20.0, 30.0])
    transform = _mesh_transform_from_center(center, scale=1000.0)
    point_m = np.array([0.1, 0.2, 0.3, 1.0])
    np.testing.assert_allclose(
        transform @ point_m,
        np.array([90.0, 180.0, 270.0, 1.0]),
    )


# ---------------------------------------------------------------------------
# FK adapters (need roboticstoolbox but not PyTorch3D)
# ---------------------------------------------------------------------------


def test_panda_adapter_shapes_and_validation():
    from robot_renderer.robots.panda import PandaAdapter
    adapter = PandaAdapter(ROBOTS_DIR / "panda" / "panda.urdf")
    R, t = adapter.get_joint_R_t(np.zeros(7))
    assert R.shape == (9, 3, 3)
    assert t.shape == (9, 3)
    # Base link pose is identity (fixed link, FK start).
    np.testing.assert_allclose(R[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t[0], np.zeros(3), atol=1e-12)
    with pytest.raises(ValueError, match="expects 7 joint angles"):
        adapter.get_joint_R_t(np.zeros(6))


def test_panda_adapter_rotations_are_orthonormal():
    from robot_renderer.robots.panda import PandaAdapter
    adapter = PandaAdapter(ROBOTS_DIR / "panda" / "panda.urdf")
    q = np.linspace(-0.5, 0.5, 7)
    R, _ = adapter.get_joint_R_t(q)
    for Ri in R:
        np.testing.assert_allclose(Ri @ Ri.T, np.eye(3), atol=1e-10)
        assert math.isclose(np.linalg.det(Ri), 1.0, abs_tol=1e-10)
