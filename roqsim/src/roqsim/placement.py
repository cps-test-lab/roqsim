# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Putting an entity where a trial wants it.

The counterpart of :mod:`roqsim.presence`, which decides *whether* an entity can be perceived. This
decides *where* it is, and the two together are what ``SpawnEntity`` does in one transaction.

Here rather than in each caller, and for the reason :func:`roqsim.pose.rpy_to_quat` already gives
about conventions: which bodies can take a pose, and what taking one means for each, is a fact about
the compiled model. A consumer that reimplements it is not reusing this simulator, it is
re-describing it -- and two descriptions drift. They had: the same twenty lines lived in the OSC
access layer and in the ROS bridge, and had already diverged on where the forward-kinematics refresh
happened and on where a body's base joint was looked up.
"""

from __future__ import annotations


def place_body(ctx, entity, pos, quat, vel=None) -> bool:
    """Put *entity* at ``pos``/``quat``. ``True`` if it took the pose. Physics thread only.

    Two kinds of body can take one, for different reasons.

    A **mocap** body (``motion: driven``) is placed by writing ``mocap_pos``/``mocap_quat``. It has
    no degrees of freedom, so the solver never owns its pose: it stays where it is put, cannot be
    shoved off that placement by whatever bumps into it, and a placement that happens to intersect
    other geometry is not answered by launching it. That is what an experiment means by an obstacle AT
    a position -- the pose is the experiment's variable, not the solver's output -- and it is what
    an SDF ``<static>true</static>`` obstacle does on the other simulator, so a trial that moves
    scenery behaves the same on both.

    A **free** body (``motion: physics``) is placed by writing its base joint's ``qpos``. Its pose
    is the solver's from the next step on, which is what a trial wants only when the body is meant
    to move, fall or be pushed.

    A **welded** body has neither and cannot be placed; ``False`` says so, and the caller turns that
    into whatever refusal its own surface owes.

    ``vel`` is the six-vector a free body is to carry away, or ``None`` for "at rest" -- which is
    what placing something means, and what every caller before it could ask for anything else
    relied on. A mocap body ignores it: there is no DOF to carry a velocity, and the placement
    itself was applied in full, so this reports success rather than refusing.

    The forward-kinematics refresh happens here, so a caller cannot place a body and then read a
    stale pose off it.
    """
    import mujoco

    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, getattr(entity, "body", "") or "")
    mocapid = int(ctx.model.body_mocapid[bid]) if bid >= 0 else -1
    if mocapid >= 0:
        ctx.data.mocap_pos[mocapid] = pos
        ctx.data.mocap_quat[mocapid] = quat
        mujoco.mj_forward(ctx.model, ctx.data)
        return True

    # Resolved here, not by the caller: which joint carries a body is the model's business, and two
    # callers looking it up separately is how they came to disagree about it.
    joint_name = (getattr(entity, "meta", None) or {}).get("base_joint")
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) if joint_name else -1
    if jid < 0 or ctx.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        return False

    q = ctx.model.jnt_qposadr[jid]
    ctx.data.qpos[q : q + 3] = pos
    ctx.data.qpos[q + 3 : q + 7] = quat
    dof = ctx.model.jnt_dofadr[jid]
    ctx.data.qvel[dof : dof + 6] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if vel is None else vel
    mujoco.mj_forward(ctx.model, ctx.data)
    return True


def base_joint_of(entity) -> str | None:
    """The joint a free body is placed through, or ``None`` when it has none.

    Exposed because a caller that wants to explain WHY a placement was refused needs the name this
    looked for, and reaching into ``entity.meta`` to guess it is what put the lookup in two places.
    """
    return (getattr(entity, "meta", None) or {}).get("base_joint")


__all__ = ["base_joint_of", "place_body"]
