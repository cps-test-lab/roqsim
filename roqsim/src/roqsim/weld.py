"""A weld's relative pose, written at run time.

A plugin that welds two bodies where they stand writes the weld's row of ``model.eq_data`` while the
run goes on: the anchor, then body2's pose in body1's frame. ``mj_setConst`` recomputes that pose
from ``qpos0`` for a body weld the compiler marks with a non-zero ``eq_data[11]`` -- one compiled
without a ``relpose``, as ``roqsim_motion``'s holds are -- so a later ``mj_setConst`` (a payload's
mass, ``model_override``'s ``body_mass``) would move the welded body back to where it was compiled.
:func:`set_relpose` writes the pose and clears that mark, which is how MuJoCo's changelog says a
weld's pose is set directly.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: The ``eq_data`` column MuJoCo marks a body weld whose pose ``mj_setConst`` recomputes with.
_RECOMPUTE = 11


def set_relpose(model: mujoco.MjModel, eq: int, anchor, pos, quat) -> None:
    """Weld *eq* holds body2 at *pos*, *quat* in body1's frame, about *anchor* on body1.

    ``mj_setConst`` leaves the pose as written afterwards. The weld's torque scale
    (``eq_data[10]``) is the caller's.
    """
    row = model.eq_data[eq]
    row[0:3] = np.asarray(anchor, dtype=float)
    row[3:6] = np.asarray(pos, dtype=float)
    row[6:10] = np.asarray(quat, dtype=float)
    row[_RECOMPUTE] = 0.0
