"""
UFACTORY xArm 7 kinematics adapter for RobotRenderer.

URDF source: xArm-Developer/xarm_ros2 (BSD-3-Clause)
  https://github.com/xArm-Developer/xarm_ros2

ERobot.URDF_read gives 10 links:
    [0]  world       (fixed, world anchor)
    [1]  link_base   (fixed, base; FK start)
    [2]  link1       (revolute, joint1)
    [3]  link2       (revolute, joint2)
    [4]  link3       (revolute, joint3)
    [5]  link4       (revolute, joint4)
    [6]  link5       (revolute, joint5)
    [7]  link6       (revolute, joint6)
    [8]  link7       (revolute, joint7)
    [9]  link_eef    (fixed, end-effector frame)

All 8 visual links (link_base through link7) are rendered.
"""
from __future__ import annotations
from typing import List

from .. import LinkChainAdapter

# link_base (fixed) + link1 through link7 (revolute)
XARM7_LINK_INDICES  = [1, 2, 3, 4, 5, 6, 7, 8]
XARM7_BASE_LINK_IDX = 1


class XArm7Adapter(LinkChainAdapter):
    """
    RobotKinematics adapter for the UFACTORY xArm 7.

    Returns 8 (R, t) pairs for visual links [link_base, link1 through link7],
    expressed in the link_base frame.
    """

    LINK_INDICES: List[int] = XARM7_LINK_INDICES
    BASE_LINK_IDX = XARM7_BASE_LINK_IDX
    NUM_JOINTS = 7
    MANUFACTURER = "UFACTORY"
