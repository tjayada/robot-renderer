from __future__ import annotations

_PT3D_AVAILABLE: bool | None = None
_PT3D_ERROR: Exception | None = None

_INSTALL_MSG = """
PyTorch3D is required by robot-renderer but is not installed.

PyTorch3D has no universal wheel. You must install a build that matches
your Python, CUDA, and PyTorch versions exactly.

Quick install (Python 3.10, CUDA 12.1, PyTorch 2.4.1):
    pip install fvcore iopath
    pip install --no-index --no-cache-dir pytorch3d \\
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html

For other versions, find the matching wheel at:
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/

See also: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
""".strip()


def _require_pt3d() -> None:
    """Raise ImportError with install instructions if pytorch3d is not usable.

    Imports the submodules the renderer actually uses, not just the top-level
    package. This catches the common "installed but the compiled extension does
    not load" case (usually a torch/pytorch3d build mismatch), which a plain
    `import pytorch3d` misses. The original error is chained so the real cause
    is visible under the install instructions.
    """
    global _PT3D_AVAILABLE, _PT3D_ERROR
    if _PT3D_AVAILABLE is None:
        try:
            from pytorch3d.io import load_obj  # noqa: F401
            from pytorch3d.renderer import MeshRasterizer  # noqa: F401
            _PT3D_AVAILABLE = True
        except Exception as e:  # ImportError, OSError, RuntimeError, ...
            _PT3D_AVAILABLE = False
            _PT3D_ERROR = e

    if not _PT3D_AVAILABLE:
        raise ImportError(_INSTALL_MSG) from _PT3D_ERROR
