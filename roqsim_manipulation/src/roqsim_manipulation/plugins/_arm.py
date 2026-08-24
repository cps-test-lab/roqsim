"""Shared helpers for the manipulator plugins (joint/actuator discovery, pose math)."""

from __future__ import annotations

import math

import mujoco


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """(w, x, y, z) quaternion from roll/pitch/yaw (rad), fixed-axis XYZ (ROS/URDF convention)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def prefixed_joints(model, prefix: str) -> list[int]:
    """Joint ids whose name starts with ``prefix``, in model declaration order.

    Only hinge/slide joints are returned (the arm's DOFs); a free/ball base joint,
    if any, is skipped so the list lines up with the controllable joints.
    """
    out = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if not name.startswith(prefix):
            continue
        # int(): mjt* enums compare False against a numpy scalar (`x in (...)` puts the enum on the
        # left of `==`), which would leave this list empty and the arm with no reportable joints.
        if int(model.jnt_type[jid]) in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            out.append(jid)
    return out


def prefixed_actuators(model, prefix: str) -> tuple[list[tuple[int, int]], list[int]]:
    """Split the arm's actuators into joint-driven and auxiliary (e.g. tendon gripper).

    Returns ``(joint_acts, aux_acts)`` where ``joint_acts`` is a list of
    ``(actuator_id, joint_id)`` for position-servo joint actuators and ``aux_acts`` is a
    list of actuator ids with a non-joint transmission (held at a fixed ctrl by the
    controller).
    """
    joint_acts: list[tuple[int, int]] = []
    aux_acts: list[int] = []
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if not name.startswith(prefix):
            continue
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_acts.append((aid, int(model.actuator_trnid[aid, 0])))
        else:
            aux_acts.append(aid)
    return joint_acts, aux_acts


def strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix) :] if prefix and name.startswith(prefix) else name


def named_actuators(
    model, prefix: str, joints: list[str], gripper_actuator: str | None = None
) -> tuple[list[tuple[int, int]], list[int]]:
    """Resolve exactly *joints* (and optionally one aux actuator) instead of scanning by prefix.

    The prefix scan in :func:`prefixed_actuators` assumes the prefix selects one subsystem, which
    holds for a standalone arm but not for a robot whose arm shares its entity with other actuated
    parts -- a humanoid's legs, a mobile manipulator's wheels. There the scan claims those too and the
    controller then fights whichever plugin owns them, writing position targets into what may be
    torque actuators. Naming the joints scopes ownership explicitly.

    Returns the same ``(joint_acts, aux_acts)`` shape as :func:`prefixed_actuators`. Raises when a
    named joint or actuator does not exist: a silently dropped joint is an arm that moves in
    ``joint_states`` but never responds to a trajectory.
    """
    # Joint -> its driving actuator. Resolved this way round, not by actuator name, because an
    # actuator is not required to share its joint's name and generally does not: the UR10e drives
    # `shoulder_pan_joint` from an actuator called `shoulder_pan`.
    by_joint: dict[int, int] = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            by_joint.setdefault(int(model.actuator_trnid[aid, 0]), aid)

    joint_acts: list[tuple[int, int]] = []
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}")
        if jid < 0:
            raise RuntimeError(f"no joint '{prefix}{name}'")
        aid = by_joint.get(jid)
        if aid is None:
            raise RuntimeError(
                f"joint '{prefix}{name}' has no position actuator, so it cannot be commanded; "
                f"list only actuated joints in `joints:`"
            )
        joint_acts.append((aid, jid))

    aux_acts: list[int] = []
    if gripper_actuator:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{gripper_actuator}")
        if aid < 0:
            raise RuntimeError(f"no actuator '{prefix}{gripper_actuator}'")
        aux_acts.append(aid)
    return joint_acts, aux_acts


def named_joints(model, prefix: str, names: list[str]) -> list[int]:
    """Joint ids for *names*, in the given order. Raises on an unresolvable name."""
    out = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}")
        if jid < 0:
            raise RuntimeError(f"no joint '{prefix}{name}'")
        out.append(jid)
    return out
