"""Derived kinematic quantities MuJoCo does not hand over directly.

Its own module rather than a helper inside ``state`` or ``capture`` because the callers span packages
-- ``roqsim.state`` and ``roqsim.capture`` here, ``roqsim_ros_bridge.sim_interfaces`` and
``roqsim_walker.nav.controller`` out of tree -- and because ``state`` already imports ``capture``
transitively (``state -> recording -> capture``), so a helper in either would have to be duplicated
to be reachable from both.
"""

from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np


class Twist(NamedTuple):
    """A body's spatial velocity, world-aligned, at the body's own frame origin.

    Named rather than a bare 6-vector on purpose: MuJoCo returns **rotational first**, so a caller
    unpacking ``lin, ang = vel[:3], vel[3:]`` gets them backwards, and the mistake is invisible in
    any planar test (a robot driving on a floor has ``ang.x = ang.y = 0`` and ``lin.z = 0``, so the
    swapped vectors look plausible).
    """

    linear: tuple[float, float, float]
    angular: tuple[float, float, float]


def body_twist(model, data, body_id: int) -> Twist:
    """The world-frame twist of body ``body_id``.

    ``mj_objectVelocity`` with ``flg_local=0`` rather than ``data.cvel``: cvel is expressed in the
    com-based frame of the body's kinematic subtree, so its linear part is the velocity *at the
    subtree centre of mass* and differs from the body origin's by omega x r -- correct for MuJoCo's
    own dynamics, wrong for "how fast is this robot moving".

    Requires ``data`` to be posed -- live during a step, or after ``mj_forward`` when restored from
    a recording.
    """
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    return Twist(
        linear=(float(vel[3]), float(vel[4]), float(vel[5])),
        angular=(float(vel[0]), float(vel[1]), float(vel[2])),
    )
