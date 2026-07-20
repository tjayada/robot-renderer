"""
Baxter left-arm kinematics adapter for RobotRenderer.

Ported from CtRNet BaxterLeftArm (https://github.com/ucsdarclab/CtRNet-robot-pose-estimation,
MIT License). The full Baxter URDF has 15 DOF; this adapter exposes the 7 left-arm joints
and pads the joint vector accordingly before calling FK.

Link indices in the full URDF link list (roboticstoolbox), relative to base link 30:
    30 = left_upper_shoulder  (S0 child)
    31 = left_lower_shoulder  (S1 child)
    32 = left_upper_elbow     (E0 child)
    33 = left_lower_elbow     (E1 child)
    34 = left_upper_forearm   (W0 child)
    36 = left_lower_forearm   (W1 child)  # index 35 is left_arm_itb (fixed), skipped
    37 = left_wrist           (W2 child)

Base-frame convention (verified numerically): roboticstoolbox's
``fkine(q, end, start)`` INCLUDES the start link's own joint variable, so with
start = links[30] the effective base frame is the *parent* side of the s0 joint,
i.e. the ``left_arm_mount`` frame, identical to CtRNet's ``bl`` frame (their DH
chain in run_eval_baxter.py matches this FK's wrist position to <0.1 mm). The
estimator's T_base_cam is therefore directly comparable to CtRNet's cTb.
"""
from __future__ import annotations
from typing import List

import numpy as np

from .. import LinkChainAdapter

BAXTER_LEFT_LINK_INDICES = [30, 31, 32, 33, 34, 36, 37]
BAXTER_LEFT_BASE_LINK_IDX = 30
BAXTER_TOTAL_JOINTS = 15  # full robot DOF expected by roboticstoolbox


class BaxterAdapter(LinkChainAdapter):
    """
    RobotKinematics adapter for the Baxter left arm.

    Returns 7 (R, t) pairs for links [S0, S1, E0, E1, W0, W1, W2].
    The 7 provided joint angles are padded into the full 15-DOF vector before FK.
    """

    LINK_INDICES: List[int] = BAXTER_LEFT_LINK_INDICES
    BASE_LINK_IDX = BAXTER_LEFT_BASE_LINK_IDX
    NUM_JOINTS = 7
    MANUFACTURER = "Rethink"

    def _full_q(self, q: np.ndarray) -> np.ndarray:
        # Pad 7 left-arm angles into the full 15-DOF joint vector (right side stays at 0).
        q_full = np.zeros(BAXTER_TOTAL_JOINTS, dtype=np.float64)
        q_full[-7:] = q
        return q_full
