"""
End-to-end smoke test: renders Panda views through the full pipeline.

Requires PyTorch3D; this doubles as the install check (`make render-check`).
Uses the built-in "panda" robot so no paths or registration are needed.
The renderer is built once for the module and the template render is shared
between tests to keep the installation check fast.
"""
import numpy as np
import torch
import pytest

import robot_renderer as rr

RENDER_SIZE = 128
NUM_VIEWS = 2
JOINT_ANGLES = np.zeros(7, dtype=np.float32)


@pytest.fixture(scope="module")
def renderer():
    K = np.array([
        [600.0,   0.0, 320.0],
        [  0.0, 600.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float32)
    cfg = rr.ViewConfig(render_size=RENDER_SIZE, viewset="fibonacci_32",
                        num_views=NUM_VIEWS)
    return rr.RobotRenderer("panda", K=K, config=cfg, device="cpu")


@pytest.fixture(scope="module")
def templates(renderer):
    return renderer.render_templates(JOINT_ANGLES)


def test_render_templates_shapes(renderer, templates):
    assert len(templates.views) == NUM_VIEWS
    for view in templates.views:
        assert view.image.shape == (3, RENDER_SIZE, RENDER_SIZE)
        assert view.image.device.type == renderer.device.type
        assert view.gt_pose.shape == (4, 4)
        assert view.R.shape == (3, 3)
        assert view.T.shape == (3,)

    assert templates.mesh_transform.shape == (4, 4)
    assert templates.anchor_gt_pose.shape == (4, 4)
    assert templates.visible_surface_points.ndim == 2
    assert templates.visible_surface_points.shape[1] == 3
    assert templates.sphere_radius > 0.0


def test_renderer_is_double_sided(renderer):
    settings = renderer._renderer.phong_renderer.rasterizer.raster_settings
    assert settings.cull_backfaces is False


def test_render_with_background_image(renderer):
    bg = torch.ones(RENDER_SIZE, RENDER_SIZE, 3) * 0.5
    result = renderer.render_templates(JOINT_ANGLES, background_image=bg)
    assert len(result.views) == NUM_VIEWS


def test_to_device(renderer):
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert renderer.to(target) is renderer
    result = renderer.render_templates(JOINT_ANGLES)
    assert result.views[0].image.device.type == target.type
