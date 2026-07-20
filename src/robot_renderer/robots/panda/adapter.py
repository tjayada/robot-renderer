"""
Franka Panda kinematics adapter.

Ported from the robot model's reference implementation.
"""
from __future__ import annotations
from typing import List

from .. import LinkChainAdapter

# FK is evaluated for these link indices (0-based in the URDF link list).
# Link 8 is the finger joint; excluded because the gripper is not part of the rendered chain.
PANDA_LINK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9]


class PandaAdapter(LinkChainAdapter):
    """
    RobotKinematics adapter for the Franka Panda arm.

    Returns 9 (R, t) pairs for links [Link0 through Link7, Hand]
    (PANDA_LINK_INDICES = [0,1,2,3,4,5,6,7,9], gripper finger excluded).
    """

    LINK_INDICES: List[int] = PANDA_LINK_INDICES
    BASE_LINK_IDX = 0
    NUM_JOINTS = 7
    MANUFACTURER = "Franka"
