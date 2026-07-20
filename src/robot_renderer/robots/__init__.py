from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def _load_erobot(urdf_path, manufacturer: Optional[str] = None):
    """Load a roboticstoolbox ERobot from a URDF path."""
    from roboticstoolbox.robot.ERobot import ERobot
    links, name, urdf_string, urdf_filepath = ERobot.URDF_read(str(urdf_path))
    kwargs = {"name": name, "urdf_string": urdf_string, "urdf_filepath": urdf_filepath}
    if manufacturer is not None:
        kwargs["manufacturer"] = manufacturer
    return ERobot(links, **kwargs)


class LinkChainAdapter:
    """
    Shared FK adapter base: poses each rendered link via roboticstoolbox
    ``fkine(q, end=link, start=base_link)``.

    Subclasses set the class attributes below (and may override
    :meth:`_full_q` when the URDF has more DOF than the exposed joint vector,
    e.g. Baxter). Note that ``fkine(start=...)`` INCLUDES the start link's own
    joint variable; for a *fixed* base link ``fkine(end==start)`` is the
    identity, while for a joint-carrying base link (Baxter S0) the base mesh
    correctly rotates with its joint.

    Class attributes:
        LINK_INDICES:  Indices into ``robot.links`` for the rendered links,
                       in mesh_files order.
        BASE_LINK_IDX: Index of the FK base link (``start`` of every fkine).
        NUM_JOINTS:    Expected length of the joint_angles argument.
        MANUFACTURER:  Optional manufacturer string for the ERobot.
        T_SCALE:       Factor applied to FK translations (e.g. 0.001 for a
                       URDF authored in mm).
    """

    LINK_INDICES: List[int] = []
    BASE_LINK_IDX: int = 0
    NUM_JOINTS: int = 0
    MANUFACTURER: Optional[str] = None
    T_SCALE: float = 1.0

    def __init__(
        self,
        urdf_path: Union[str, Path],
        link_indices: Optional[List[int]] = None,
        base_link_idx: Optional[int] = None,
    ) -> None:
        self._link_indices = link_indices if link_indices is not None else list(self.LINK_INDICES)
        self._base_link_idx = base_link_idx if base_link_idx is not None else self.BASE_LINK_IDX
        self._robot = _load_erobot(urdf_path, manufacturer=self.MANUFACTURER)

    def _full_q(self, q: np.ndarray) -> np.ndarray:
        """Map the exposed joint vector to the full URDF joint vector (identity by default)."""
        return q

    def get_joint_R_t(
        self, joint_angles: Union[List[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FK for the given joint angles (radians).

        Returns:
            R: (n_links, 3, 3) rotation matrices in the base-link frame.
            t: (n_links, 3)   translation vectors in the base-link frame (metres).
        """
        q = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.NUM_JOINTS:
            raise ValueError(
                f"{type(self).__name__} expects {self.NUM_JOINTS} joint angles, got {q.shape[0]}"
            )
        q = self._full_q(q)
        base_link = self._robot.links[self._base_link_idx]
        R_list, t_list = [], []
        for idx in self._link_indices:
            T = self._robot.fkine(q, end=self._robot.links[idx], start=base_link)
            R_list.append(T.R)
            t_list.append(T.t)
        return np.array(R_list), np.array(t_list) * self.T_SCALE
