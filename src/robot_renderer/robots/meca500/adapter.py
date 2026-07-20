"""
Mecademic Meca500 R3 kinematics adapter for RobotRenderer.

URDF source: Vanderbilt-Applied-Robotics-Lab/meca500_ros2
  https://github.com/Vanderbilt-Applied-Robotics-Lab/meca500_ros2
  That source declares no license. The source-derived URDF and meshes are
  generated locally by `make assets` instead of being shipped with the repo.
  See MESH_LICENSES/README.md.

ERobot.URDF_read gives 7 links:
    [0]  meca_base_link    (fixed, base; FK start)
    [1]  meca_axis_1_link  (revolute, joint1, axis z)
    [2]  meca_axis_2_link  (revolute, joint2, axis y)
    [3]  meca_axis_3_link  (revolute, joint3, axis y)
    [4]  meca_axis_4_link  (revolute, joint4, axis x)
    [5]  meca_axis_5_link  (revolute, joint5, axis y)
    [6]  meca_axis_6_link  (revolute, joint6, axis x)

All 7 visual links (base through axis_6) are rendered.
DAE scene transforms are identity, so no coordinate correction is required.
"""
from __future__ import annotations
from typing import List

from .. import LinkChainAdapter

MECA500_LINK_INDICES  = [0, 1, 2, 3, 4, 5, 6]
MECA500_BASE_LINK_IDX = 0


class Meca500Adapter(LinkChainAdapter):
    """
    RobotKinematics adapter for the Mecademic Meca500 R3.

    Returns 7 (R, t) pairs for visual links [base, axis_1 through axis_6],
    expressed in the meca_base_link frame.
    """

    LINK_INDICES: List[int] = MECA500_LINK_INDICES
    BASE_LINK_IDX = MECA500_BASE_LINK_IDX
    NUM_JOINTS = 6
    MANUFACTURER = "Mecademic"
