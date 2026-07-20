"""
KUKA LBR Med 7 R800 kinematics adapter for RobotRenderer.

URDF source: lbr-stack/lbr_fri_ros2_stack (Apache 2.0)
  https://github.com/lbr-stack/lbr_fri_ros2_stack

The Med 7 is a 7-DOF arm. ERobot.URDF_read gives 10 links:
    [0]  world         (fixed, world anchor)
    [1]  lbr_link_0    (fixed, base; FK start)
    [2]  lbr_link_1    (revolute, A1)
    [3]  lbr_link_2    (revolute, A2)
    [4]  lbr_link_3    (revolute, A3)
    [5]  lbr_link_4    (revolute, A4)
    [6]  lbr_link_5    (revolute, A5)
    [7]  lbr_link_6    (revolute, A6)
    [8]  lbr_link_7    (revolute, A7)
    [9]  lbr_link_ee   (fixed, end-effector frame)

We render all 8 visual links (link_0 through link_7, fixed base included),
computing FK from lbr_link_0. The base link's own FK (end == start on a
fixed link) is the identity, so it renders correctly in every view.
"""
from __future__ import annotations
from typing import List

from .. import LinkChainAdapter

# Indices into robot.links for all 8 visual links (link_0 through link_7).
LBR_MED7_LINK_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]
# Index of lbr_link_0, the fixed base from which FK is computed.
LBR_MED7_BASE_LINK_IDX = 1


class LBRMed7Adapter(LinkChainAdapter):
    """
    RobotKinematics adapter for the KUKA LBR Med 7 R800.

    Returns 8 (R, t) pairs for visual links [link_0 through link_7] (fixed base
    included), expressed in the lbr_link_0 base frame, matching the 8 mesh
    groups registered for lbr_med7.
    """

    LINK_INDICES: List[int] = LBR_MED7_LINK_INDICES
    BASE_LINK_IDX = LBR_MED7_BASE_LINK_IDX
    NUM_JOINTS = 7
    MANUFACTURER = "KUKA"
