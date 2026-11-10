# SPDX-License-Identifier: Apache-2.0
"""The whole-body solver: a motion clip's pose and targets in, joint rotations out.

Analytic first, because it runs in Python on every update of every handling humanoid (the budget is
in ``roqsim_walker/README.md``, "Hands"): the clip's pose is the start; the trunk turns toward the
targets, crouches for a low one, bends forward as far as they need and leans for one beside the
body; each arm with a target is solved by two-bone IK on the description's own bone lengths, its
elbow pointing down and out; each wrist takes the target's rotation, inside its limits. Gaze turns
the head. A target left out of reach after all of that is reported by its residual, never stretched
to.

Every output is a pure function of the inputs: no iteration count depends on time and nothing is
drawn, so a run is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import quat as Q
from .description import SIDES, Description
from .hands import HandLayout

#: The largest forward bend (rad), how finely the bend a reach needs is found, and its share at the
#: hips (the rest along the spine) -- the walker's own forward bend, kept as it is.
MAX_BEND = math.radians(90.0)
BEND_TOLERANCE = math.radians(0.2)
AT_HIPS = 0.7
#: The share of the arm's length a reach may use, so the elbow keeps a bend.
REACH = 0.97
#: Where an elbow points, in the body's frame (x forward, y left, z up); y's sign is the side.
ELBOW_POLE = (-0.3, 0.5, -1.0)
#: The turn's share at the pelvis (the rest in the spine), and the largest turn and lean (rad).
TURN_AT_PELVIS = 0.4
MAX_TURN = math.radians(60.0)
MAX_LEAN = math.radians(25.0)
#: A crouch lowers the pelvis by at most this share of its standing height, tried in this many steps
#: where bending over does not reach.
MAX_CROUCH = 0.45
CROUCH_STEPS = 8
#: The share of a hand's twist about the forearm's line the forearm takes (pronation and supination
#: turn the forearm's bones along it); the wrist keeps the rest.
FOREARM_TWIST = 0.5


@dataclass
class HandGoal:
    """Where one hand's grip frame goes: a world position and, optionally, a world rotation."""

    pos: np.ndarray
    quat: np.ndarray | None = None


@dataclass
class Posture:
    """The trunk's posture. ``None`` lets the solver choose what the targets need; a number
    states it: ``turn`` (rad, left positive), ``lean`` (rad, toward the left positive), ``crouch``
    (m the pelvis is lowered), ``bend`` (rad forward)."""

    turn: float | None = None
    lean: float | None = None
    crouch: float | None = None
    bend: float | None = None


@dataclass
class Solution:
    rotations: dict  # joint -> local rotation, every joint the solver set
    root_drop: float = 0.0  # m the pelvis was lowered (a crouch)
    residual: dict = field(default_factory=dict)  # side -> m between the grip and its goal
    posture: Posture = field(default_factory=Posture)


def _trunk(desc: Description, root, heading, local, turn, lean, bend):
    turn_p = Q.axis_angle("z", TURN_AT_PELVIS * turn)
    hips = Q.mul(
        turn_p, Q.mul(Q.axis_angle("x", -AT_HIPS * lean), Q.axis_angle("y", AT_HIPS * bend))
    )
    pelvis = Q.mul(local["pelvis"], hips)
    pelvis_w = Q.mul(heading, pelvis)
    spine = Q.mul(
        local["spine"],
        Q.mul(
            Q.axis_angle("z", (1 - TURN_AT_PELVIS) * turn),
            Q.mul(
                Q.axis_angle("x", -(1 - AT_HIPS) * lean), Q.axis_angle("y", (1 - AT_HIPS) * bend)
            ),
        ),
    )
    spine_pos = root + Q.rotate(pelvis_w, desc.offset("spine"))
    spine_w = Q.mul(pelvis_w, spine)
    return pelvis, hips, spine, pelvis_w, spine_pos, spine_w


def _shoulder(desc, side, spine_pos, spine_w):
    return spine_pos + Q.rotate(spine_w, desc.offset(f"shoulder_{side}"))


def _reaches(desc, root, heading, local, wrists, turn, lean, bend) -> bool:
    *_, spine_pos, spine_w = _trunk(desc, root, heading, local, turn, lean, bend)
    return all(
        np.linalg.norm(t - _shoulder(desc, s, spine_pos, spine_w)) <= REACH * desc.arm_length(s)
        for s, t in wrists.items()
    )


def _bend_for(desc, root, heading, local, wrists, turn, lean) -> tuple[float, bool]:
    if _reaches(desc, root, heading, local, wrists, turn, lean, 0.0):
        return 0.0, True
    if not _reaches(desc, root, heading, local, wrists, turn, lean, MAX_BEND):
        return MAX_BEND, False
    lo, hi = 0.0, MAX_BEND
    while hi - lo > BEND_TOLERANCE:
        mid = (lo + hi) / 2
        lo, hi = (
            (lo, mid)
            if _reaches(desc, root, heading, local, wrists, turn, lean, mid)
            else (mid, hi)
        )
    return hi, True


def two_bone(root, a: float, b: float, target, pole):
    """Upper and lower directions (unit) of a two-bone chain of lengths ``a``, ``b`` from ``root``
    toward ``target``, bending toward ``pole``; out of reach, straight at it."""
    d = np.asarray(target, float) - root
    dist = float(np.linalg.norm(d))
    dhat = d / dist if dist > 1e-9 else np.array([0.0, 0.0, -1.0])
    reach = min(max(dist, abs(a - b) + 1e-6), (a + b) * (1.0 - 1e-6))
    alpha = math.acos(max(-1.0, min(1.0, (a * a + reach * reach - b * b) / (2 * a * reach))))
    pole = np.asarray(pole, float) - np.dot(pole, dhat) * dhat
    n = np.linalg.norm(pole)
    pole = pole / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    upper = math.cos(alpha) * dhat + math.sin(alpha) * pole
    elbow = root + a * upper
    lower = root + reach * dhat - elbow
    return upper, lower / np.linalg.norm(lower)


def _limb(desc, upper_joint, lower_joint, parent_w, root, target, pole):
    """Local rotations of a limb's two upper joints that put its end on ``target``."""
    off_u = desc.offset(lower_joint)  # the upper bone ends where the lower joint is
    child = {"elbow": "wrist", "knee": "ankle"}[lower_joint.split("_")[0]]
    off_l = desc.offset(f"{child}_{lower_joint.split('_')[1]}")
    up, lo = two_bone(root, np.linalg.norm(off_u), np.linalg.norm(off_l), target, pole)
    u_w = Q.mul(Q.between(Q.rotate(parent_w, off_u), up), parent_w)
    l_w = Q.mul(Q.between(Q.rotate(u_w, off_l), lo), u_w)
    return Q.mul(Q.inv(parent_w), u_w), Q.mul(Q.inv(u_w), l_w), u_w, l_w


def solve(
    desc: Description,
    root_xyz,
    heading,
    local: dict,
    goals: dict,
    layouts: dict,
    posture: Posture | None = None,
    gaze=None,
) -> Solution:
    """Rotations that put each hand's grip frame on its goal (``goals``: side -> :class:`HandGoal`
    or ``None``) over the clip pose ``local`` (joint -> local rotation), for a pelvis at
    ``root_xyz`` turned by ``heading`` (world). ``layouts`` holds each hand's
    :class:`~roqsim_motion.hands.HandLayout` (where its grip frame is on the wrist). A goal without a
    rotation places the wrist, as ``WalkerHandle.reach`` does. ``gaze`` is a world point the
    head turns to at once, inside its limits; no body passes one -- a walker turns its head itself,
    at a bounded rate, with :func:`head_toward`, and a robot body turns none."""
    posture = posture or Posture()
    root = np.asarray(root_xyz, float)
    heading = np.asarray(heading, float)
    goals = {s: g for s, g in goals.items() if g is not None}
    unknown = set(goals) - set(SIDES)
    if unknown:
        raise ValueError(f"no arm {sorted(unknown)!r}: the arms are {list(SIDES)!r}")
    wrist_goal = {}
    for s, g in goals.items():
        if g.quat is None:
            wrist_goal[s] = (np.asarray(g.pos, float), None)
        else:
            lay: HandLayout = layouts[s]
            wq = Q.mul(g.quat, Q.inv(lay.grip_quat))
            wrist_goal[s] = (np.asarray(g.pos, float) - Q.rotate(wq, lay.grip_pos), wq)
    wrists = {s: p for s, (p, _) in wrist_goal.items()}

    # The posture: turn toward the targets, crouch for a low one, bend, lean for one to the side.
    hx = Q.rotate(heading, [1.0, 0.0, 0.0])
    hy = Q.rotate(heading, [0.0, 1.0, 0.0])
    turn = posture.turn
    if turn is None:
        turn = 0.0
        if wrists:
            mean = np.mean(list(wrists.values()), axis=0) - root
            fwd, side = float(np.dot(mean, hx)), float(np.dot(mean, hy))
            if math.hypot(fwd, side) > 0.15:
                turn = max(-MAX_TURN, min(MAX_TURN, math.atan2(side, max(fwd, 1e-3))))
    crouch = posture.crouch
    if crouch is None:
        crouch = 0.0
        if wrists:
            low = min(p[2] for p in wrists.values())
            crouch = max(0.0, min(MAX_CROUCH * desc.root_height, 0.6 * desc.root_height - low))
    lowered = root - np.array([0.0, 0.0, crouch])
    lean = posture.lean if posture.lean is not None else 0.0
    if posture.bend is not None:
        bend = posture.bend
    else:
        bend, ok = _bend_for(desc, lowered, heading, local, wrists, turn, lean)
        low_target = bool(wrists) and min(p[2] for p in wrists.values()) < root[2]
        if (
            not ok
            and posture.crouch is None
            and low_target
            and crouch < MAX_CROUCH * desc.root_height
        ):
            # Bent over as far as it goes and still short: crouch deeper first, no deeper than it
            # takes -- a hand a few centimetres short of a handle is no reason to kneel.
            deepest = MAX_CROUCH * desc.root_height
            for k in range(1, CROUCH_STEPS + 1):
                crouch_k = crouch + (deepest - crouch) * k / CROUCH_STEPS
                lowered_k = root - np.array([0.0, 0.0, crouch_k])
                bend_k, ok = _bend_for(desc, lowered_k, heading, local, wrists, turn, lean)
                if ok or k == CROUCH_STEPS:
                    crouch, lowered, bend = crouch_k, lowered_k, bend_k
                    break
        if not ok and posture.lean is None and wrists:
            # One to the side: lean toward it, as little as reaches.
            far = max(wrists.items(), key=lambda kv: abs(float(np.dot(kv[1] - root, hy))))
            sign = 1.0 if float(np.dot(far[1] - root, hy)) > 0 else -1.0
            for step in (0.1, 0.2, 0.3, 0.4):
                cand = sign * step / 0.4 * MAX_LEAN
                b, ok = _bend_for(desc, lowered, heading, local, wrists, turn, cand)
                if ok:
                    lean, bend = cand, b
                    break
            else:
                lean = sign * MAX_LEAN
                bend, _ = _bend_for(desc, lowered, heading, local, wrists, turn, lean)
    pelvis, hips, spine, pelvis_w, spine_pos, spine_w = _trunk(
        desc, lowered, heading, local, turn, lean, bend
    )
    out = {
        "pelvis": pelvis,
        "spine": spine,
        "hip_l": Q.mul(Q.inv(hips), local["hip_l"]),
        "hip_r": Q.mul(Q.inv(hips), local["hip_r"]),
    }

    # Legs: a crouch keeps each foot where the clip put it.
    if crouch > 1e-6:
        stand = desc.fk(root, heading, local)
        for s in SIDES:
            hip_pos = lowered + Q.rotate(pelvis_w, desc.offset(f"hip_{s}"))
            ankle_pos, ankle_w = stand[f"ankle_{s}"]
            hip_l, knee_l, _, knee_w = _limb(
                desc, f"hip_{s}", f"knee_{s}", pelvis_w, hip_pos, ankle_pos, hx
            )
            out[f"hip_{s}"] = hip_l
            out[f"knee_{s}"] = knee_l
            out[f"ankle_{s}"] = Q.mul(Q.inv(knee_w), ankle_w)

    # Arms: two-bone onto the wrist goal, then the wrist's rotation.
    for s, (target, wq) in wrist_goal.items():
        sign = SIDES[s]
        shoulder = _shoulder(desc, s, spine_pos, spine_w)
        pole = Q.rotate(spine_w, [ELBOW_POLE[0], sign * ELBOW_POLE[1], ELBOW_POLE[2]])
        wrist_l = Q.IDENTITY.copy()
        # A wrist clamped short of the goal's rotation holds the grip elsewhere than the goal
        # assumed: the wrist's own goal is moved by what the clamp left, and the arm solved again.
        for _ in range(4 if wq is not None else 1):
            sh_l, el_l, _, el_w = _limb(
                desc, f"shoulder_{s}", f"elbow_{s}", spine_w, shoulder, target, pole
            )
            if wq is None:
                break
            # The forearm turns with the hand: part of the twist the wrist's rotation asks for is the
            # forearm's, about its own line (which leaves the wrist where it is), so a hand turned
            # palm down twists the forearm along its length, not the skin at the wrist alone.
            pronation = _twist_part(
                Q.mul(Q.inv(el_w), wq), desc.offset(f"wrist_{s}"), FOREARM_TWIST
            )
            el_l, el_w = Q.mul(el_l, pronation), Q.mul(el_w, pronation)
            wrist_l = desc.clamp(f"wrist_{s}", Q.mul(Q.inv(el_w), wq))
            reached = Q.mul(el_w, wrist_l)
            if Q.angle_between(reached, wq) < 1e-4:
                break
            goal = goals[s]
            target = np.asarray(goal.pos, float) - Q.rotate(reached, layouts[s].grip_pos)
        out[f"shoulder_{s}"] = sh_l
        out[f"elbow_{s}"] = el_l
        out[f"wrist_{s}"] = wrist_l

    if gaze is not None:
        out["head"] = head_toward(desc, spine_pos, spine_w, gaze)

    sol = Solution(out, crouch, {}, Posture(turn, lean, crouch, bend))
    if goals:
        merged = {**local, **out}
        poses = desc.fk(lowered, heading, merged)
        for s, g in goals.items():
            wp, wqq = poses[f"wrist_{s}"]
            grip = wp if g.quat is None else wp + Q.rotate(wqq, layouts[s].grip_pos)
            sol.residual[s] = float(np.linalg.norm(grip - np.asarray(g.pos, float)))
    return sol


def head_toward(desc: Description, spine_pos, spine_w, point) -> np.ndarray:
    """The head's local rotation that turns it toward the world ``point``, inside its limits, for a
    spine at world ``spine_pos`` turned by ``spine_w``."""
    head_pos = np.asarray(spine_pos, float) + Q.rotate(spine_w, desc.offset("head"))
    d = Q.rotate(Q.inv(spine_w), np.asarray(point, float) - head_pos)
    yaw = math.atan2(d[1], d[0])
    pitch = -math.atan2(d[2], math.hypot(d[0], d[1]))
    return desc.clamp("head", Q.from_euler("zy", [yaw, pitch]))


def _twist_part(rot, axis, share: float) -> np.ndarray:
    """``share`` of ``rot``'s twist about ``axis`` (its swing-twist decomposition), as a rotation."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    w, v = float(rot[0]), np.asarray(rot[1:], float)
    along = float(v @ a)
    if abs(w) < 1e-12 and abs(along) < 1e-12:
        return Q.IDENTITY.copy()
    angle = 2.0 * math.atan2(along, w)
    angle = math.atan2(math.sin(angle), math.cos(angle))  # the short way round
    return Q.axis_angle(a, share * angle)
