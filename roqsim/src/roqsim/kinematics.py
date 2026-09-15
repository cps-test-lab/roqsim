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


#: ``qpos`` entries a joint of each type occupies. A free joint carries a position and a quaternion,
#: a ball joint a quaternion alone.
JOINT_WIDTH = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}

#: ``qvel`` entries a joint of each type occupies. Narrower than :data:`JOINT_WIDTH` for the two
#: rotating types: a quaternion needs four numbers to state an orientation and three to state how
#: fast it is changing.
JOINT_DOFS = {
    int(mujoco.mjtJoint.mjJNT_FREE): 6,
    int(mujoco.mjtJoint.mjJNT_BALL): 3,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def joint_width(model, jid: int) -> int:
    """How many ``qpos`` entries joint ``jid`` occupies."""
    return JOINT_WIDTH[int(model.jnt_type[jid])]


def joint_dofs(model, jid: int) -> int:
    """How many ``qvel`` entries joint ``jid`` occupies."""
    return JOINT_DOFS[int(model.jnt_type[jid])]


def joint_dof_indices(model, names) -> list[int]:
    """Every ``qvel`` index belonging to the named joints, in order, skipping names the model lacks.

    Sliced from each joint's own ``jnt_dofadr`` rather than indexed by joint id, for the reason
    :func:`roqsim.state.joint_columns` gives about ``qpos``: a free or ball joint occupies several
    entries, so ``qvel[jid]`` would silently read a neighbour's value.
    """
    out: list[int] = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            continue
        adr = int(model.jnt_dofadr[jid])
        out.extend(range(adr, adr + joint_dofs(model, jid)))
    return sorted(set(out))


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
