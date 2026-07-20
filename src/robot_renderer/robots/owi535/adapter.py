"""
OWI-535 kinematics adapter for RobotRenderer.

Kinematic chain (4 actuated DOF):
    World --fixed--> Model --Y--> Rotation --X--> Base --X--> Elbow --X--> Wrist

Link indices in the RTB link list (kinematic tree order):
    0 = World   (fixed base, not rendered)
    1 = Model
    2 = Rotation
    3 = Base
    4 = Elbow
    5 = Wrist

Mesh OBJ files are in metres (visual origins baked in, vertices scaled mm to m).
FK translations are also converted to metres (URDF units are mm; T_SCALE is 0.001).

Joint order: [Model_Rotation, Rotation_Base, Base_Elbow, Elbow_Wrist] (radians).
"""
from __future__ import annotations
from typing import List

from .. import LinkChainAdapter

OWI535_LINK_INDICES = [1, 2, 3, 4, 5]  # Model, Rotation, Base, Elbow, Wrist
OWI535_BASE_LINK_IDX = 0               # World
OWI535_NUM_JOINTS = 4


class Owi535Adapter(LinkChainAdapter):
    """
    RobotKinematics adapter for the OWI-535 4-DOF arm.

    Returns 5 (R, t) pairs for links [Model, Rotation, Base, Elbow, Wrist].
    Translations are in metres (URDF joint origins are in mm, converted by 0.001).
    """

    LINK_INDICES: List[int] = OWI535_LINK_INDICES
    BASE_LINK_IDX = OWI535_BASE_LINK_IDX
    NUM_JOINTS = OWI535_NUM_JOINTS
    T_SCALE = 0.001  # URDF joint origins are in mm; convert to metres
