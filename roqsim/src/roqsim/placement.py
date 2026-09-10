# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Putting an entity where a trial wants it.

The counterpart of :mod:`roqsim.presence`, which decides *whether* an entity can be perceived. This
decides *where* it is, and the two together are what ``SpawnEntity`` does in one transaction.

Which bodies can take a pose, and what taking one means for each, is a fact about the compiled
model rather than about any one way of asking for it. So it lives with the model, and every
consumer gets the same answer: an OpenSCENARIO action, a ROS service and an in-process plugin all
place a body by calling this, and none of them has to know that a mocap body is written through
``mocap_pos`` while a free one is written through its joint's ``qpos``, nor which of the two a
given entity is.

That is what makes the behaviour portable across transports instead of merely similar. A consumer
that spelled the rule out itself would be re-describing the simulator rather than using it, and two
descriptions of one fact are two things to keep in step -- so the refusal below, the velocity
default, and the forward-kinematics refresh are decided once, here, for all of them.
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

    # Resolved here, not by the caller: which joint carries a body is a property of the model, so
    # asking the model is the one lookup that cannot disagree with it.
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

    Exposed so a caller explaining WHY a placement was refused can name the joint that was looked
    for without repeating the lookup. One function answering for that name is what keeps every
    caller's account of a refusal true of the model it is refusing about.
    """
    return (getattr(entity, "meta", None) or {}).get("base_joint")


__all__ = ["base_joint_of", "place_body"]
