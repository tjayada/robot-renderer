from __future__ import annotations
from typing import Protocol, Tuple, Union

import numpy as np


class RobotKinematics(Protocol):
    """
    Structural interface for robot forward kinematics.

    Implementations must return one (R, t) pair per mesh group, in the same
    order as the mesh_files list passed to the renderer.

    R shape: (n_links, 3, 3), rotation matrices in world/base frame
    t shape: (n_links, 3),    translation vectors in world/base frame, in
                               **metres** (URDF convention). Adapters for
                               URDFs authored in mm must convert (see
                               ``robots.LinkChainAdapter.T_SCALE``).

    joint_angles are in radians.
    """

    def get_joint_R_t(
        self, joint_angles: Union[list, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]: ...
