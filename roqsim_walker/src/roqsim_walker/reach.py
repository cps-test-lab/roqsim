"""A walker's hands on world points, one or both: each arm by two-bone IK, the spine bent forward
as far as the points need.

A walker reaching for something -- a box by its handles, a part on a bench, a door handle with one
hand -- has those hands where the thing is, whatever its gait does. :func:`reach_pose` gives the
local rotations of the pelvis, the hips, the spine and each reaching arm that put those wrists on
their world points, for the skeleton's own bone lengths: the body bends forward (up to
``MAX_BEND``, most of it at the hips -- the pelvis turns, the legs turned back by as much so they
stand as they did -- the rest along the spine) only as far as the reaching hands need
(:func:`needed_bend`, :func:`trunk_pose`), the elbows bend down and outward, the hands carry on the
forearms' line (:func:`arm_pose`). An arm with no point is not posed. A point out of reach even
bent over is reached for: the arm points at it, straight. :mod:`roqsim_walker.animation` blends
each arm's pose over the gait's (``AnimState.reach``), and the walker's handle sets the points
(``WalkerHandle.reach``).

Targets are a mapping of arm (``"l"``, ``"r"``) to a world point or ``None``; a missing arm is
``None``.
"""

from __future__ import annotations

import math

import numpy as np

from roqsim_walker.humanoid import quat_mul, quat_rotate

#: How far the body bends forward at most (rad), how finely the bend a reach needs is found (rad),
#: and the share of it at the hips (the rest along the spine).
MAX_BEND = math.radians(90.0)
BEND_TOLERANCE = math.radians(0.2)
AT_HIPS = 0.7
#: The share of the arm's length a reach may use, so the elbow keeps a bend.
REACH = 0.97
#: Where an elbow points, in the walker's frame (x forward, y left, z up): down, a little out and
#: back -- the y component's sign is the arm's side.
ELBOW_POLE = (-0.3, 0.5, -1.0)

#: The arms, each with the sign of its side (y left).
SIDES = {"l": 1.0, "r": -1.0}


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _axis_angle(axis, angle: float):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), *(axis * s)])


def _between(a, b):
    """The shortest rotation taking direction ``a`` to direction ``b``."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        # Half a turn about any axis normal to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return _axis_angle(axis, math.pi)
    q = np.array([1.0 + c, *np.cross(a, b)])
    return q / np.linalg.norm(q)


def _lengths(skeleton) -> dict:
    return {
        s: (
            np.linalg.norm(skeleton.offset(f"elbow_{s}")),
            np.linalg.norm(skeleton.offset(f"wrist_{s}")),
        )
        for s in SIDES
    }


def _reaching(targets) -> dict:
    """The arms of ``targets`` that have a point, as float arrays; refuses an arm it does not
    know."""
    unknown = set(targets) - set(SIDES)
    if unknown:
        raise ValueError(f"no arm {sorted(unknown)!r}: the arms are {list(SIDES)!r}")
    return {s: np.asarray(t, float) for s, t in targets.items() if t is not None}


def _trunk(skeleton, root, heading_q, local: dict, bend: float):
    """The pelvis and spine bent forward by ``bend``: their local rotations, the hips' turn, and the
    spine's world position and rotation."""
    hips = _axis_angle((0.0, 1.0, 0.0), AT_HIPS * bend)
    pelvis_new = quat_mul(local["pelvis"], hips)
    pelvis_world = quat_mul(heading_q, pelvis_new)
    spine_pos = root + quat_rotate(pelvis_world, np.array(skeleton.offset("spine")))
    spine_new = quat_mul(local["spine"], _axis_angle((0.0, 1.0, 0.0), (1.0 - AT_HIPS) * bend))
    return pelvis_new, hips, spine_new, spine_pos, quat_mul(pelvis_world, spine_new)


def trunk_pose(skeleton, root_xyz, heading_q, local: dict, bend: float):
    """The trunk bent forward by ``bend`` (rad): ``(rotations, spine_pos, spine_world)``, where
    ``rotations`` are the local rotations of ``pelvis``, ``spine``, ``hip_l`` and ``hip_r`` -- the
    hips turned back by the pelvis's bend, so the legs stand as they did -- and ``spine_pos`` and
    ``spine_world`` the spine's world position and rotation the arms hang from. ``local`` holds the
    clip's local rotations of those four joints; the walker's pelvis is at ``root_xyz``, heading
    ``heading_q`` (world)."""
    root = np.asarray(root_xyz, float)
    pelvis_new, hips, spine_new, spine_pos, spine_world = _trunk(
        skeleton, root, heading_q, local, bend
    )
    rotations = {
        "pelvis": pelvis_new,
        "spine": spine_new,
        "hip_l": quat_mul(_qinv(hips), local["hip_l"]),
        "hip_r": quat_mul(_qinv(hips), local["hip_r"]),
    }
    return rotations, spine_pos, spine_world


def _shoulder(skeleton, side: str, spine_pos, spine_world):
    return spine_pos + quat_rotate(spine_world, np.array(skeleton.offset(f"shoulder_{side}")))


def needed_bend(skeleton, root_xyz, heading_q, local: dict, targets) -> float:
    """The least forward bend (rad) that brings every arm's target in ``targets`` (arm -> world
    point or ``None``) within that arm's reach: found to ``BEND_TOLERANCE``, ``MAX_BEND`` when none
    does, and 0 when no arm has a point."""
    root = np.asarray(root_xyz, float)
    reaching = _reaching(targets)
    lengths = _lengths(skeleton)

    def reached(bend: float) -> bool:
        *_, spine_pos, spine_world = _trunk(skeleton, root, heading_q, local, bend)
        return all(
            np.linalg.norm(target - _shoulder(skeleton, side, spine_pos, spine_world))
            <= REACH * sum(lengths[side])
            for side, target in reaching.items()
        )

    if reached(0.0):
        return 0.0
    if not reached(MAX_BEND):
        return MAX_BEND
    lo, hi = 0.0, MAX_BEND
    while hi - lo > BEND_TOLERANCE:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if reached(mid) else (mid, hi)
    return hi


def arm_pose(skeleton, side: str, spine_pos, spine_world, target) -> dict:
    """Local rotations of ``shoulder_<side>``, ``elbow_<side>`` and ``wrist_<side>`` that put that
    wrist on ``target`` (a world point), for an arm hanging from a spine at ``spine_pos`` turned by
    ``spine_world`` (both world; :func:`trunk_pose`). A target out of reach is pointed at, the arm
    straight."""
    if side not in SIDES:
        raise ValueError(f"no arm {side!r}: the arms are {list(SIDES)!r}")
    target = np.asarray(target, float)
    sign = SIDES[side]
    elbow_off = np.array(skeleton.offset(f"elbow_{side}"))
    wrist_off = np.array(skeleton.offset(f"wrist_{side}"))
    a, b = _lengths(skeleton)[side]
    shoulder = _shoulder(skeleton, side, spine_pos, spine_world)
    d = target - shoulder
    dist = float(np.linalg.norm(d))
    dhat = d / dist if dist > 1e-9 else quat_rotate(spine_world, np.array([1.0, 0.0, 0.0]))
    reach = min(max(dist, abs(a - b) + 1e-6), (a + b) * (1.0 - 1e-6))
    # The angle at the shoulder between the reach and the upper arm (law of cosines).
    alpha = math.acos(max(-1.0, min(1.0, (a * a + reach * reach - b * b) / (2 * a * reach))))
    pole = quat_rotate(spine_world, np.array([ELBOW_POLE[0], sign * ELBOW_POLE[1], ELBOW_POLE[2]]))
    pole = pole - np.dot(pole, dhat) * dhat
    pole = (
        pole / np.linalg.norm(pole) if np.linalg.norm(pole) > 1e-9 else np.array([0.0, 0.0, -1.0])
    )
    upper = math.cos(alpha) * dhat + math.sin(alpha) * pole
    elbow = shoulder + a * upper
    fore = shoulder + reach * dhat - elbow
    fore /= np.linalg.norm(fore)
    # The shoulder turns the upper arm's rest direction onto the reach's; the elbow then turns
    # the forearm from where the shoulder left it, so the arm does not twist about itself.
    sh_world = quat_mul(_between(quat_rotate(spine_world, elbow_off), upper), spine_world)
    el_world = quat_mul(_between(quat_rotate(sh_world, wrist_off), fore), sh_world)
    return {
        f"shoulder_{side}": quat_mul(_qinv(spine_world), sh_world),
        f"elbow_{side}": quat_mul(_qinv(sh_world), el_world),
        f"wrist_{side}": np.array([1.0, 0.0, 0.0, 0.0]),
    }


def reach_pose(
    skeleton, root_xyz, heading_q, local: dict, targets, bend: float | None = None
) -> dict:
    """Local rotations of ``pelvis``, ``hip_*``, ``spine`` (:func:`trunk_pose`) and of the
    shoulder, elbow and wrist of each arm with a point in ``targets`` (arm -> world point or
    ``None``; :func:`arm_pose`) that put those wrists on their points, for a walker whose pelvis is
    at ``root_xyz``, heading ``heading_q`` (world), its clip's local rotations ``local`` (by joint);
    bent forward by ``bend`` (default: :func:`needed_bend`). An arm without a point has no entry,
    so it keeps the clip's pose."""
    reaching = _reaching(targets)
    if bend is None:
        bend = needed_bend(skeleton, root_xyz, heading_q, local, reaching)
    out, spine_pos, spine_world = trunk_pose(skeleton, root_xyz, heading_q, local, bend)
    for side, target in reaching.items():
        out.update(arm_pose(skeleton, side, spine_pos, spine_world, target))
    return out
