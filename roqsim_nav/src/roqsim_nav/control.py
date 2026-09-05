"""Turning a planner's preferred velocity into something a body can actually do.

:class:`~roqsim_nav.behavior.NavCore` produces a **world-frame 2-vector whose magnitude is the
desired speed**. That is the right output for a point-mass planner and the wrong input for most
bases: it carries no orientation, and a differential-drive robot cannot move sideways to follow it.

The three laws below close that gap, one per base geometry the substrate has. They are pure
functions of their arguments -- no MuJoCo, no state -- so they are cheap to test exhaustively and
cannot drift with the rest of the system.

**None of them clamps to the robot's own limits.** ``diff_drive.drive()`` already clips to the
platform's ``max_linear_vel`` / ``max_angular_vel``, and it owns the wheel acceleration ramp and the
inverse kinematics. Re-clamping here would put the robot's physical limits in two places and make the
navigator's ``max_angular_vel`` silently override a platform's.
"""

from __future__ import annotations

import math

import numpy as np

#: Below this the preferred velocity is treated as "no command": its direction is numerically
#: meaningless, so steering toward it would spin a stopped body on rounding error.
STOPPED = 1e-3


def wrap_angle(a: float) -> float:
    """``a`` folded into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def approach_angle(cur: float, target: float, max_step: float) -> float:
    """Move ``cur`` toward ``target`` by at most ``max_step``, the shortest way round."""
    d = wrap_angle(target - cur)
    if abs(d) <= max_step:
        return target
    return cur + math.copysign(max_step, d)


def yaw_to_quat(yaw: float) -> list[float]:
    """MuJoCo ``(w, x, y, z)`` for a rotation about +Z."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def yaw_from_quat(q) -> float:
    """Yaw (rad) from a MuJoCo ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def unicycle(
    pref_vel, yaw: float, *, gain: float, max_w: float, turn_in_place: float
) -> tuple[float, float, float]:
    """A differential-drive base: drive forward, turn, never strafe.

    ``v = speed * cos(heading error)`` rather than ``speed``: the base slows *into* a turn instead of
    arcing wide past the corner, and the projection is what keeps forward speed and yaw rate
    consistent with each other. Past ``turn_in_place`` the error is too large to drive out of, so it
    pivots on the spot instead of carving a long arc through whatever is beside it.

    Asked to go almost exactly backwards, it **reverses** rather than pivoting through 180 degrees.
    That is the mirror of the pivot rule and uses the same threshold: within ``turn_in_place`` of
    straight ahead, drive; within ``turn_in_place`` of straight back, reverse; in between, pivot.

    Reversing is not a special case bolted on for one caller -- it is the only way a differential
    base can honour a command to go backwards at all. Recovery expresses "back away from what stopped
    you" as a world-frame velocity pointing away from the blocker, and a law that always turned to
    face it would spend the whole backup window spinning on the spot and never actually retreat,
    which makes recovery inert on exactly the base that most needs it. Ordinary navigation cannot
    trigger it: a path point is never that far behind unless the mover has been sent back the way it
    came, where reversing is also the right answer.
    """
    speed = float(math.hypot(pref_vel[0], pref_vel[1]))
    if speed < STOPPED:
        return 0.0, 0.0, 0.0
    err = wrap_angle(math.atan2(float(pref_vel[1]), float(pref_vel[0])) - yaw)
    if abs(err) <= turn_in_place:
        w = float(np.clip(gain * err, -max_w, max_w))
        return speed * math.cos(err), 0.0, w
    back = wrap_angle(err - math.pi)
    if abs(back) <= turn_in_place:
        # Reverse: steer on the error to the REVERSED heading and drive backwards along it. Forward
        # is tested first, so a `turn_in_place` above 90 degrees -- where the two windows overlap --
        # keeps driving forwards rather than flipping to reverse.
        w = float(np.clip(gain * back, -max_w, max_w))
        return -speed * math.cos(back), 0.0, w
    return 0.0, 0.0, float(np.clip(gain * err, -max_w, max_w))


def holonomic(
    pref_vel, yaw: float, *, gain: float, max_w: float, face: str = "travel"
) -> tuple[float, float, float]:
    """An omnidirectional base: execute the preferred velocity, rotated into the body frame.

    ``face: travel`` turns to look where it is going, like a differential base, which is what makes
    a mounted sensor point along the path. ``face: hold`` keeps the current heading and crabs
    sideways -- right for a cart that must stay square to the aisle.
    """
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    vx = float(pref_vel[0]) * cos_y + float(pref_vel[1]) * sin_y
    vy = -float(pref_vel[0]) * sin_y + float(pref_vel[1]) * cos_y
    speed = float(math.hypot(pref_vel[0], pref_vel[1]))
    if face == "hold" or speed < STOPPED:
        return vx, vy, 0.0
    err = wrap_angle(math.atan2(float(pref_vel[1]), float(pref_vel[0])) - yaw)
    return vx, vy, float(np.clip(gain * err, -max_w, max_w))


def ackermann(
    pref_vel, yaw: float, *, gain: float, max_w: float, min_speed: float
) -> tuple[float, float, float]:
    """A car-like base: it cannot turn in place, and its twist states a *curvature*.

    ``v`` is floored at ``min_speed`` whenever there is anywhere to go, because a stationary car has
    no curvature: ``ackermann_drive`` derives the rack angle from ``w / v``, so a zero ``v`` steers
    the wheels nowhere however large ``w`` is. Commanding ``v = 0`` to turn -- which is exactly what
    the unicycle law does -- would leave a car sitting still with its wheels straight.

    The steering limit is the drive plugin's, not ours: it clips the rack angle to the platform's
    ``max_steer_angle``, so this needs no wheelbase and no steering geometry.
    """
    speed = float(math.hypot(pref_vel[0], pref_vel[1]))
    if speed < STOPPED:
        return 0.0, 0.0, 0.0
    err = wrap_angle(math.atan2(float(pref_vel[1]), float(pref_vel[0])) - yaw)
    return max(speed, min_speed), 0.0, float(np.clip(gain * err, -max_w, max_w))


#: The law for each declared base geometry. A `RobotHandle` states which it is, so nothing here is
#: keyed on a robot's name and an out-of-tree drive gets the right law by declaring one of these.
LAWS = {"unicycle": unicycle, "holonomic": holonomic, "ackermann": ackermann}


# -- path tracking -----------------------------------------------------------------------------
def closest_point_on_polyline(path, pos) -> tuple[int, float, np.ndarray]:
    """``(segment index, fraction along it, point)`` for the nearest point on ``path`` to ``pos``."""
    pos = np.asarray(pos, dtype=float)
    best = (0, 0.0, np.asarray(path[0], dtype=float), float("inf"))
    for i in range(len(path) - 1):
        a = np.asarray(path[i], dtype=float)
        b = np.asarray(path[i + 1], dtype=float)
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-12 else float(np.clip((pos - a) @ ab / denom, 0.0, 1.0))
        point = a + t * ab
        d = float(np.linalg.norm(pos - point))
        if d < best[3]:
            best = (i, t, point, d)
    return best[0], best[1], best[2]


def lookahead_point(path, pos, lookahead: float) -> np.ndarray:
    """The point ``lookahead`` metres along ``path`` from the point nearest ``pos``.

    Pure pursuit's target selection: the carrot is on the **path**, not at the end of the current
    leg. That is the whole difference from chasing waypoints, and it is why corner error is set by
    the lookahead rather than by how close the mover has to get before it gives up on a waypoint.
    Past the end of the path the last point is returned, so the mover converges on the goal instead
    of orbiting a carrot that ran out.
    """
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) == 1:
        return path[0]
    i, t, point = closest_point_on_polyline(path, pos)
    remaining = float(lookahead)
    # Walk forward from the projection, consuming segment lengths.
    seg_end = path[i + 1]
    step = float(np.linalg.norm(seg_end - point))
    if step >= remaining:
        direction = seg_end - point
        n = float(np.linalg.norm(direction))
        return point if n < 1e-12 else point + direction / n * remaining
    remaining -= step
    for j in range(i + 1, len(path) - 1):
        a, b = path[j], path[j + 1]
        step = float(np.linalg.norm(b - a))
        if step >= remaining:
            return a + (b - a) / step * remaining
        remaining -= step
    return path[-1]


def path_remaining(path, pos) -> float:
    """Arc length from the point on ``path`` nearest ``pos`` to its end."""
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) == 1:
        return float(np.linalg.norm(path[0] - np.asarray(pos, dtype=float)))
    i, _t, point = closest_point_on_polyline(path, pos)
    total = float(np.linalg.norm(path[i + 1] - point))
    for j in range(i + 1, len(path) - 1):
        total += float(np.linalg.norm(path[j + 1] - path[j]))
    return total


def pure_pursuit(path, pos, yaw: float, *, lookahead: float, speed: float):
    """A preferred velocity aimed at the pure-pursuit carrot, plus the arc curvature to it.

    Returns ``(pref_vel, curvature)``. ``pref_vel`` feeds the same base-geometry laws above, so an
    omnidirectional base simply executes it; ``curvature`` is the geometric quantity a car-like base
    actually needs -- ``2 * y_body / L^2``, the arc through the carrot -- and is what makes this pure
    pursuit rather than carrot-chasing with a heading gain bolted on.

    Near the end of the path the carrot stops receding, so the commanded speed is eased off by the
    **remaining arc length**. Without that a mover circles its final goal forever: the carrot sits on
    top of it, the heading error flips every step, and it never converges.

    Arc length rather than straight-line distance to the carrot, and the difference is not academic:
    a path that bends puts the carrot closer *as the crow flies* than it is along the path, so easing
    on the chord slows the mover at every corner. Slowing into corners may well be wanted -- nav2's
    Regulated Pure Pursuit does exactly that -- but it should be an explicit rule about curvature,
    not a side effect of measuring the wrong thing.
    """
    pos = np.asarray(pos, dtype=float)
    target = lookahead_point(path, pos, lookahead)
    delta = target - pos
    dist = float(np.linalg.norm(delta))
    if dist < STOPPED:
        return np.zeros(2), 0.0
    # Body-frame lateral offset of the carrot: the y in the pure-pursuit curvature formula.
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    y_body = -delta[0] * sin_y + delta[1] * cos_y
    curvature = 2.0 * y_body / (dist * dist)
    scale = min(1.0, path_remaining(path, pos) / max(lookahead, 1e-6))
    return delta / dist * speed * scale, curvature
