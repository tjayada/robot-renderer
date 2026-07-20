"""
robot-renderer: PyTorch3D-based multi-view renderer for articulated robots.

Public API
----------
    import robot_renderer as rr

    renderer = rr.RobotRenderer("panda", K=K, config=rr.ViewConfig())
    result   = renderer.render_templates(joint_angles)
"""

from .config import ViewConfig
from .registry import register
from .renderer_pt3d import gaussian_blur_chw
from .robot_renderer import RobotRenderer
from .types import RenderedView, RenderResult

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ViewConfig",
    "register",
    "RobotRenderer",
    "gaussian_blur_chw",
    "RenderedView",
    "RenderResult",
]
