# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Collision-free paths in joint space, and straight lines of the arm's site.

:func:`plan_path` tries the straight joint-space segment first -- in an open cell most moves are
one -- and otherwise grows two trees, one from each end, toward random configurations and toward
each other (RRT-Connect, Kuffner and LaValle 2000) until they meet. The path found is then
shortcut: random pairs of its points joined directly wherever that segment is free, which removes
the detours a random tree takes. Every segment is checked at ``step`` rad of the largest joint
change, so a path is free at that resolution.

:func:`cartesian_path` moves the site along a straight line, its rotation turning evenly, by
inverse kinematics at small steps from the configuration before -- the approach into a die or a
slot, where a joint-space path could swing the part through the tooling. It refuses a line it cannot
follow (a step that would jump the arm to another branch, or a configuration in collision).

Both draw every random number from the generator they are given.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .collision import CollisionChecker
from .kinematics import IKError


class PlanningError(RuntimeError):
    """No path was found."""


def _nearest(tree: list[np.ndarray], q: np.ndarray) -> int:
    return int(np.argmin([np.max(np.abs(n - q)) for n in tree]))


def _extend(checker, tree, parents, target, extend, step):
    """Grow ``tree`` from its node nearest ``target`` toward it by at most ``extend``; the new node."""
    i = _nearest(tree, target)
    near = tree[i]
    delta = target - near
    span = np.max(np.abs(delta))
    new = target if span <= extend else near + delta * (extend / span)
    if not checker.segment_free(near, new, step):
        return None
    tree.append(new)
    parents.append(i)
    return len(tree) - 1


def _connect(checker, tree, parents, target, extend, step):
    """Grow ``tree`` toward ``target`` until it gets there or is blocked; whether it got there."""
    while True:
        k = _extend(checker, tree, parents, target, extend, step)
        if k is None:
            return None
        if np.max(np.abs(tree[k] - target)) < 1e-9:
            return k


def _trace(tree, parents, k):
    out = []
    while k >= 0:
        out.append(tree[k])
        k = parents[k]
    return out[::-1]


def shortcut(checker: CollisionChecker, path, rng, *, iterations=150, step=0.02):
    """Join random pairs of ``path``'s points directly wherever the segment is free."""
    path = [np.asarray(p, float) for p in path]
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        if checker.segment_free(path[i], path[j], step):
            path = path[: i + 1] + path[j:]
    return path


def plan_path(
    checker: CollisionChecker,
    start,
    goal,
    rng: np.random.Generator,
    *,
    step: float = 0.02,
    extend: float = 0.4,
    iterations: int = 4000,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    shortcuts: int = 150,
) -> list[np.ndarray]:
    """A free joint-space path from ``start`` to ``goal`` as waypoints, both ends included.

    ``bounds`` (lower, upper) limits where random configurations are drawn (default the joint
    limits; an arm with ±2π joints plans faster in a smaller box). Raises :class:`PlanningError`
    when an end is in collision or no path is found within ``iterations``.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    for name, q in (("start", start), ("goal", goal)):
        if not checker.free(q):
            raise PlanningError(f"the {name} configuration is in collision: {checker.describe(q)}")
    if checker.segment_free(start, goal, step):
        return [start, goal]
    lo, hi = bounds if bounds is not None else (checker.kin.lower, checker.kin.upper)
    lo = np.maximum(lo, checker.kin.lower)
    hi = np.minimum(hi, checker.kin.upper)
    a, pa = [start], [-1]
    b, pb = [goal], [-1]
    for _ in range(iterations):
        target = rng.uniform(lo, hi)
        k = _extend(checker, a, pa, target, extend, step)
        if k is not None:
            m = _connect(checker, b, pb, a[k], extend, step)
            if m is not None:
                # From the root of one tree to the root of the other; the trees swap roles every
                # iteration, so the path may run from the goal, and is turned round if it does.
                left, right = _trace(a, pa, k), _trace(b, pb, m)
                path = left + right[::-1][1:]
                if np.max(np.abs(path[0] - start)) > 1e-9:
                    path = path[::-1]
                return shortcut(checker, path, rng, iterations=shortcuts, step=step)
        a, pa, b, pb = b, pb, a, pa
    raise PlanningError(
        f"no free path found in {iterations} iterations from {np.round(start, 3).tolist()} "
        f"to {np.round(goal, 3).tolist()}"
    )


def _slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, r0.reshape(-1))
    mujoco.mju_mat2Quat(q1, r1.reshape(-1))
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-9:
        q = q0
    else:
        q = (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q / np.linalg.norm(q))
    return out.reshape(3, 3)


def cartesian_path(
    checker: CollisionChecker | None,
    kin,
    start,
    pos,
    rot=None,
    *,
    step: float = 0.005,
    max_joint_step: float = 0.15,
    check: bool = True,
) -> list[np.ndarray]:
    """Joint configurations moving the site in a straight line from where ``start`` puts it to
    ``pos`` (rotation turning evenly to ``rot``), one every ``step`` m.

    Each is solved from the one before, so the arm stays on its branch; a step that changes a joint
    by more than ``max_joint_step`` rad is a jump to another branch and refused, as is a
    configuration in collision when ``check`` (with the checker's allowed pairs).
    """
    start = np.asarray(start, float)
    p0, r0 = kin.fk(start)
    pos = np.asarray(pos, float)
    r1 = r0 if rot is None else np.asarray(rot, float).reshape(3, 3)
    angle = np.linalg.norm(np.asarray(_rotvec(r0, r1)))
    n = max(1, int(np.ceil(max(np.linalg.norm(pos - p0) / step, angle / 0.05))))
    out = [start]
    q = start
    for k in range(1, n + 1):
        s = k / n
        try:
            q_next = kin.ik(p0 + (pos - p0) * s, _slerp(r0, r1, s), q, restarts=0)
        except IKError as err:
            raise PlanningError(
                f"the straight line leaves the arm's reach at {s:.0%}: {err}"
            ) from err
        if np.max(np.abs(q_next - q)) > max_joint_step:
            raise PlanningError(
                f"the straight line needs a jump of {np.max(np.abs(q_next - q)):.2f} rad at {s:.0%}: "
                "a branch change or a singularity on the way"
            )
        if check and checker is not None and not checker.free(q_next):
            raise PlanningError(
                f"the straight line is in collision at {s:.0%}: {checker.describe(q_next)}"
            )
        out.append(q_next)
        q = q_next
    return out


def _rotvec(r0, r1):
    from .kinematics import rotation_error

    return rotation_error(r0, r1)
