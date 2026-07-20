PY ?= python

# PyTorch3D wheel: must match Python + CUDA + PyTorch exactly.
# Override on the CLI if your setup differs, e.g.:
#   make install-pt3d PT3D_WHEEL=https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
PT3D_WHEEL ?= https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl

# --- Step-by-step setup ---
#
# Recommended flow (matches the README setup steps):
#
#   conda create -n robot-renderer python=3.10 pip -c conda-forge -y
#   conda activate robot-renderer
#   make install-torch
#   make install-pt3d
#   make install
#   make check
#
# Or all at once after activating the env:
#   make setup

.PHONY: help setup install-torch install-pt3d install check \
        assets assets-owi535 assets-meca500 clean-asset-cache test render-check lint

help:
	@echo "robot-renderer targets:"
	@echo "  make setup             full install: torch -> pytorch3d -> robot-renderer"
	@echo "  make install-torch     PyTorch + torchvision CUDA wheels (2.4.1 + cu121)"
	@echo "  make install-pt3d      fvcore, iopath, and the matching PyTorch3D wheel"
	@echo "  make install           robot-renderer (editable) with assets + test extras"
	@echo "  make check             import torch, pytorch3d, and robot_renderer"
	@echo "  make test              run dependency-light tests"
	@echo "  make render-check      render Panda views; verifies the PyTorch3D install"
	@echo "  make lint              ruff check on src/tests/tools"
	@echo "  make assets            download + rebuild the assets we cannot ship (owi535, meca500)."
	@echo "                         Safe to re-run: robots that are already there are skipped."
	@echo "  make assets-owi535     OWI-535 URDF and meshes from the RoboPose/INRIA archive"
	@echo "  make assets-meca500    Meca500 URDF and meshes from meca500_ros2 (pinned commit)"
	@echo "  make clean-asset-cache delete the download cache (tools/_downloads)"
	@echo ""
	@echo "Why some assets are not shipped: see MESH_LICENSES/README.md."

install-torch:
	pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
	    --index-url https://download.pytorch.org/whl/cu121

install-pt3d:
	pip install fvcore==0.1.5.post20221221 iopath==0.1.10
	pip install $(PT3D_WHEEL)

install:
	pip install -e ".[assets,test]"

setup: install-torch install-pt3d install

check:
	$(PY) -c "import torch; print('torch:', torch.__version__)"
	$(PY) -c "from pytorch3d.renderer import MeshRasterizer; print('pytorch3d: ok')"
	$(PY) -c "import robot_renderer; print('robot_renderer: ok')"
	@echo "All checks passed."

test:
	$(PY) -m pytest tests/test_geometry.py tests/test_assets.py -v

render-check:
	$(PY) -m pytest tests/test_smoke.py -v

lint:
	$(PY) -m ruff check src tests tools

assets:
	$(PY) tools/fetch_assets.py --robot all

assets-owi535:
	$(PY) tools/fetch_assets.py --robot owi535

assets-meca500:
	$(PY) tools/fetch_assets.py --robot meca500

clean-asset-cache:
	rm -rf tools/_downloads
