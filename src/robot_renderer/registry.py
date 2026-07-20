from __future__ import annotations
from pathlib import Path
from typing import Dict, Union

_REGISTRY: Dict[str, dict] = {}
_BUILTIN_DIR = Path(__file__).parent / "robots"
_BUILTINS_REGISTERED: bool = False


def _absolute(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _absolute_mesh_files(mesh_files):
    normalized = []
    for group in mesh_files:
        if isinstance(group, (str, Path)):
            normalized.append(str(_absolute(group)))
        else:
            normalized.append([str(_absolute(path)) for path in group])
    return normalized


def register(
    name: str,
    urdf: Union[str, Path],
    mesh_dir: Union[str, Path],
    adapter_cls=None,
    mesh_files=None,
) -> None:
    """
    Register a robot for use with RobotRenderer.

    Args:
        name:        Identifier used in RobotRenderer(name, ...).
        urdf:        Path to the robot's URDF file.
        mesh_dir:    Directory containing the robot's OBJ mesh files.
        adapter_cls: Optional custom RobotKinematics implementation class.
        mesh_files:  Optional explicit per-link mesh file list: a list of lists
                     of OBJ paths, one inner list per link, in FK output order.
                     When omitted, mesh files are auto-discovered from mesh_dir.
    """
    entry: dict = {
        "urdf": _absolute(urdf),
        "mesh_dir": _absolute(mesh_dir),
        "adapter_cls": adapter_cls,
    }
    if mesh_files is not None:
        entry["mesh_files"] = _absolute_mesh_files(mesh_files)
    _REGISTRY[name] = entry


def _register_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    panda_dir = _BUILTIN_DIR / "panda"
    if panda_dir.exists():
        from .robots.panda import PandaAdapter
        # Explicit per-link mesh list matching PANDA_LINK_INDICES = [0,1,2,3,4,5,6,7,9].
        # Order MUST match the FK output order from PandaAdapter.get_joint_R_t().
        # finger.obj is excluded; it is not part of the rendered kinematic chain.
        visual = panda_dir / "meshes" / "visual"
        # Visual meshes use a per-link subdirectory layout (linkN/linkN.obj).
        # Use the high-detail visual meshes. Collision meshes are too coarse for
        # appearance templates.
        panda_mesh_files = [
            [str(visual / "link0" / "link0.obj")],
            [str(visual / "link1" / "link1.obj")],
            [str(visual / "link2" / "link2.obj")],
            [str(visual / "link3" / "link3.obj")],
            [str(visual / "link4" / "link4.obj")],
            [str(visual / "link5" / "link5.obj")],
            [str(visual / "link6" / "link6.obj")],
            [str(visual / "link7" / "link7.obj")],
            [str(visual / "hand" / "hand.obj")],
        ]
        _REGISTRY.setdefault("panda", {
            "urdf": panda_dir / "panda.urdf",
            "mesh_dir": visual,
            "mesh_files": panda_mesh_files,
            "adapter_cls": PandaAdapter,
        })

    baxter_dir = _BUILTIN_DIR / "baxter"
    if baxter_dir.exists():
        from .robots.baxter import BaxterAdapter
        # Mesh order MUST match BAXTER_LEFT_LINK_INDICES = [30,31,32,33,34,36,37]
        # i.e. the FK output order from BaxterAdapter.get_joint_R_t(), 7 links total:
        #   S0 (shoulder yaw), S1 (shoulder pitch),
        #   E0 (upper elbow),  E1 (lower elbow),
        #   W0 (upper forearm), W1 (lower forearm), W2 (wrist)
        # S0 and S1 are rendered but hidden behind / fused to the torso in most views.
        visual = baxter_dir / "meshes" / "visual"
        baxter_mesh_files = [
            [str(visual / "S0" / "S0.obj")],  # link 30, shoulder yaw (mostly occluded)
            [str(visual / "S1" / "S1.obj")],  # link 31, shoulder pitch (mostly occluded)
            [str(visual / "E0" / "E0.obj")],
            [str(visual / "E1" / "E1.obj")],
            [str(visual / "W0" / "W0.obj")],
            [str(visual / "W1" / "W1.obj")],
            [str(visual / "W2" / "W2.obj")],
        ]
        _REGISTRY.setdefault("baxter_left_arm", {
            "urdf": baxter_dir / "baxter.urdf",
            "mesh_dir": visual,
            "mesh_files": baxter_mesh_files,
            "adapter_cls": BaxterAdapter,
        })

    owi535_dir = _BUILTIN_DIR / "owi535"
    if owi535_dir.exists():
        from .robots.owi535 import Owi535Adapter

        def _m(name: str) -> str:
            return str(owi535_dir / name)

        # Mesh lists per link, in FK output order: Model, Rotation, Base, Elbow, Wrist.
        # Each inner list contains all unique visual mesh files for that link (from URDF).
        owi535_mesh_files = [
            # Model (7 meshes)
            [_m("Untitled-mesh.0.obj"), _m("Untitled_001-mesh.0.obj"),
             _m("Untitled_001-mesh.1.obj"), _m("Untitled_001-mesh.2.obj"),
             _m("Untitled_001-mesh.3.obj"), _m("Untitled_002-mesh.0.obj"),
             _m("Untitled_003-mesh.0.obj")],
            # Rotation (1 mesh)
            [_m("Untitled_064-mesh.0.obj")],
            # Base (6 meshes)
            [_m("Untitled_058-mesh.0.obj"), _m("Untitled_059-mesh.0.obj"),
             _m("Untitled_060-mesh.0.obj"), _m("Untitled_061-mesh.0.obj"),
             _m("Untitled_062-mesh.0.obj"), _m("Untitled_063-mesh.0.obj")],
            # Elbow (26 meshes)
            [_m("Untitled_004-mesh.0.obj"), _m("Untitled_005-mesh.0.obj"),
             _m("Untitled_006-mesh.0.obj"), _m("Untitled_007-mesh.0.obj"),
             _m("Untitled_008-mesh.0.obj"), _m("Untitled_009-mesh.0.obj"),
             _m("Untitled_010-mesh.0.obj"), _m("Untitled_011-mesh.0.obj"),
             _m("Untitled_040-mesh.0.obj"), _m("Untitled_041-mesh.0.obj"),
             _m("Untitled_042-mesh.0.obj"), _m("Untitled_043-mesh.0.obj"),
             _m("Untitled_044-mesh.0.obj"), _m("Untitled_045-mesh.0.obj"),
             _m("Untitled_046-mesh.0.obj"), _m("Untitled_047-mesh.0.obj"),
             _m("Untitled_048-mesh.0.obj"), _m("Untitled_049-mesh.0.obj"),
             _m("Untitled_050-mesh.0.obj"), _m("Untitled_051-mesh.0.obj"),
             _m("Untitled_052-mesh.0.obj"), _m("Untitled_053-mesh.0.obj"),
             _m("Untitled_054-mesh.0.obj"), _m("Untitled_055-mesh.0.obj"),
             _m("Untitled_056-mesh.0.obj"), _m("Untitled_057-mesh.0.obj")],
            # Wrist (34 meshes, gripper excluded, same policy as Panda and Baxter)
            [_m("Untitled_012-mesh.0.obj"), _m("Untitled_013-mesh.0.obj"),
             _m("Untitled_014-mesh.0.obj"), _m("Untitled_015-mesh.0.obj"),
             _m("Untitled_016-mesh.0.obj"), _m("Untitled_017-mesh.0.obj"),
             _m("Untitled_018-mesh.0.obj"), _m("Untitled_019-mesh.0.obj"),
             _m("Untitled_019-mesh.1.obj"), _m("Untitled_019-mesh.2.obj"),
             _m("Untitled_020-mesh.0.obj"), _m("Untitled_021-mesh.0.obj"),
             _m("Untitled_022-mesh.0.obj"), _m("Untitled_022-mesh.1.obj"),
             _m("Untitled_022-mesh.2.obj"), _m("Untitled_023-mesh.0.obj"),
             _m("Untitled_024-mesh.0.obj"), _m("Untitled_025-mesh.0.obj"),
             _m("Untitled_026-mesh.0.obj"), _m("Untitled_026-mesh.1.obj"),
             _m("Untitled_027-mesh.0.obj"), _m("Untitled_028-mesh.0.obj"),
             _m("Untitled_029-mesh.0.obj"), _m("Untitled_030-mesh.0.obj"),
             _m("Untitled_031-mesh.0.obj"), _m("Untitled_032-mesh.0.obj"),
             _m("Untitled_033-mesh.0.obj"), _m("Untitled_034-mesh.0.obj"),
             _m("Untitled_035-mesh.0.obj"), _m("Untitled_036-mesh.0.obj"),
             _m("Untitled_037-mesh.0.obj"), _m("Untitled_037-mesh.1.obj"),
             _m("Untitled_038-mesh.0.obj"), _m("Untitled_039-mesh.0.obj")],
        ]
        _REGISTRY.setdefault("owi535", {
            "urdf": owi535_dir / "owi535.urdf",
            "mesh_dir": owi535_dir,
            "mesh_files": owi535_mesh_files,
            "adapter_cls": Owi535Adapter,
        })


    lbr_med7_dir = _BUILTIN_DIR / "lbr_med7"
    if lbr_med7_dir.exists():
        from .robots.lbr_med7 import LBRMed7Adapter
        # Mesh order matches LBR_MED7_LINK_INDICES = [1,2,3,4,5,6,7,8]
        # i.e. FK output order from LBRMed7Adapter: link_0 through link_7
        # (8 links; the fixed base link_0 is rendered too).
        visual = lbr_med7_dir / "meshes" / "visual"
        lbr_med7_mesh_files = [
            [str(visual / "link_0" / "link_0.obj")],  # fixed base
            [str(visual / "link_1" / "link_1.obj")],
            [str(visual / "link_2" / "link_2.obj")],
            [str(visual / "link_3" / "link_3.obj")],
            [str(visual / "link_4" / "link_4.obj")],
            [str(visual / "link_5" / "link_5.obj")],
            [str(visual / "link_6" / "link_6.obj")],
            [str(visual / "link_7" / "link_7.obj")],
        ]
        _REGISTRY.setdefault("lbr_med7", {
            "urdf": lbr_med7_dir / "lbr_med7.urdf",
            "mesh_dir": visual,
            "mesh_files": lbr_med7_mesh_files,
            "adapter_cls": LBRMed7Adapter,
        })


    xarm7_dir = _BUILTIN_DIR / "xarm7"
    if xarm7_dir.exists():
        from .robots.xarm7 import XArm7Adapter
        # Mesh order matches XARM7_LINK_INDICES = [1,2,3,4,5,6,7,8]
        # i.e. FK output order: link_base, link1 through link7
        visual = xarm7_dir / "meshes" / "visual"
        xarm7_mesh_files = [
            [str(visual / "link_base" / "link_base.obj")],
            [str(visual / "link1"     / "link1.obj")],
            [str(visual / "link2"     / "link2.obj")],
            [str(visual / "link3"     / "link3.obj")],
            [str(visual / "link4"     / "link4.obj")],
            [str(visual / "link5"     / "link5.obj")],
            [str(visual / "link6"     / "link6.obj")],
            [str(visual / "link7"     / "link7.obj")],
        ]
        _REGISTRY.setdefault("xarm7", {
            "urdf": xarm7_dir / "xarm7.urdf",
            "mesh_dir": visual,
            "mesh_files": xarm7_mesh_files,
            "adapter_cls": XArm7Adapter,
        })


    meca500_dir = _BUILTIN_DIR / "meca500"
    if meca500_dir.exists():
        from .robots.meca500 import Meca500Adapter
        visual = meca500_dir / "meshes" / "visual"
        meca500_mesh_files = [
            [str(visual / "meca_500_r3_base" / "meca_500_r3_base.obj")],
            [str(visual / "meca_500_r3_j1"   / "meca_500_r3_j1.obj")],
            [str(visual / "meca_500_r3_j2"   / "meca_500_r3_j2.obj")],
            [str(visual / "meca_500_r3_j3"   / "meca_500_r3_j3.obj")],
            [str(visual / "meca_500_r3_j4"   / "meca_500_r3_j4.obj")],
            [str(visual / "meca_500_r3_j5"   / "meca_500_r3_j5.obj")],
            [str(visual / "meca_500_r3_j6"   / "meca_500_r3_j6.obj")],
        ]
        _REGISTRY.setdefault("meca500", {
            "urdf": meca500_dir / "meca500.urdf",
            "mesh_dir": visual,
            "mesh_files": meca500_mesh_files,
            "adapter_cls": Meca500Adapter,
        })


def get_robot_entry(name: str) -> dict:
    """Return the registry entry for name, raising KeyError if not found."""
    _register_builtins()
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Robot '{name}' not registered. "
            f"Available: {available}. "
            f"Register with robot_renderer.register('{name}', urdf=..., mesh_dir=...)."
        )
    return _REGISTRY[name]
