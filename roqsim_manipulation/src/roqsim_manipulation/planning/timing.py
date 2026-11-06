# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The fastest timing of a joint-space path under per-joint velocity and acceleration limits.

**The path.** The waypoints a planner returns are joined by a smooth curve ``q(s)``: the polyline is
divided into pieces of at most ``densify`` rad and a cubic spline drawn through their ends, with
``s`` the chord length, so the curve stays within a fraction of the piece length of the polyline and
has a continuous tangent through each waypoint, where a robot blends rather than stops. With a
collision checker given, the curve is checked at ``step`` rad and, if it cuts a corner into
something, drawn again through finer pieces; a curve that still does not clear is replaced by the
polyline itself, stopping at each waypoint.

**The timing** is time-optimal path parameterisation by reachability analysis (Pham and Pham,
"A new approach to time-optimal path parameterization based on reachability analysis", IEEE T-RO
2018), on a grid of ``s``. With ``x = ṡ²`` and ``u = s̈``, joint ``j``'s velocity is ``q'_j ṡ`` and
its acceleration ``q'_j u + q''_j x``; each limit is a linear constraint on ``(x, u)`` at each grid
point, and ``x`` at the next grid point is ``x + 2 Δs u``. A backward pass finds, at every grid
point, the largest ``x`` from which the path can still be brought to rest at its end; a forward
pass from rest then takes the largest acceleration that stays within it. The result is the
trajectory as knots with positions, velocities and accelerations, which
``ArmHandle.follow_trajectory`` interpolates as quintics.

**Torque.** UR, like other vendors, publishes no joint acceleration limit; what bounds a real arm is
its drives' torque. With a ``torque`` function given (inverse dynamics, the payload included), every
knot's torque is compared with ``torque_limit`` times ``torque_fraction``; a trajectory that exceeds
it is timed again with every acceleration limit scaled down until it does not, and the scale that
held is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline


class TimingError(RuntimeError):
    """A path cannot be timed within the limits."""


@dataclass
class TimedPath:
    """A trajectory as knots, from rest to rest."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    #: the factor the acceleration limits were scaled by to keep the torques within their limits
    accel_scale: float = 1.0
    #: the largest |torque| / limit over the knots, when a torque function was given
    torque_ratio: float | None = None
    #: whether the path is the waypoint polyline with a stop at each waypoint
    stops_at_waypoints: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def _polyline(waypoints, densify):
    pts = [np.asarray(waypoints[0], float)]
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / densify)))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * k / n)
    pts = np.array(pts)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
            keep.append(i)
    return pts[keep]


def _spline(points):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if len(points) == 2:
        # A straight segment: the spline of two points is the line.
        return s[-1], lambda x, nu=0: _line(points, s[-1], x, nu)
    return s[-1], CubicSpline(s, points, axis=0, bc_type="natural")


def _line(points, length, x, nu):
    x = np.atleast_1d(x)
    a, b = points[0], points[1]
    if nu == 0:
        return a + np.outer(x / length, b - a)
    if nu == 1:
        return np.tile((b - a) / length, (len(x), 1))
    return np.zeros((len(x), len(a)))


def _max_x(A, b):
    """The largest x over {(x, u): A @ (x, u) <= b}, by the polygon's vertices (2-D LP)."""
    m = len(b)
    i, j = np.triu_indices(m, 1)
    a1, a2 = A[i], A[j]
    det = a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0]
    ok = np.abs(det) > 1e-14
    if not ok.any():
        return None
    i, j, det = i[ok], j[ok], det[ok]
    a1, a2, b1, b2 = A[i], A[j], b[i], b[j]
    x = (b1 * a2[:, 1] - b2 * a1[:, 1]) / det
    u = (a1[:, 0] * b2 - a2[:, 0] * b1) / det
    pts = np.stack([x, u], axis=1)
    slack = A @ pts.T - b[:, None]
    tol = 1e-9 * (1.0 + np.abs(b)[:, None])
    feasible = np.all(slack <= tol, axis=0)
    if not feasible.any():
        return None
    return float(np.max(pts[feasible, 0]))


def _reachability(qs, qss, ds, vmax, amax):
    """The forward-backward pass; ``(x, u)`` at each grid point."""
    n_pts, n = qs.shape
    big = 1e12
    xvel = np.full(n_pts, big)
    with np.errstate(divide="ignore"):
        for j in range(n):
            lim = np.where(np.abs(qs[:, j]) > 1e-12, (vmax[j] / np.abs(qs[:, j])) ** 2, big)
            xvel = np.minimum(xvel, lim)
    # Joint accelerations at a point where the path's tangent vanishes but its curvature does not
    # bound x too: |q''_j| x <= a_j.
    with np.errstate(divide="ignore"):
        for j in range(n):
            curv = np.where(np.abs(qs[:, j]) <= 1e-12, np.abs(qss[:, j]), 0.0)
            xvel = np.minimum(xvel, np.where(curv > 1e-12, amax[j] / np.maximum(curv, 1e-300), big))
    hi = np.zeros(n_pts)
    hi[-1] = 0.0
    for i in range(n_pts - 2, -1, -1):
        rows, rhs = [], []
        for j in range(n):
            rows += [[qss[i, j], qs[i, j]], [-qss[i, j], -qs[i, j]]]
            rhs += [amax[j], amax[j]]
        rows += [[1.0, 0.0], [-1.0, 0.0], [1.0, 2 * ds[i]], [-1.0, -2 * ds[i]]]
        rhs += [xvel[i], 0.0, hi[i + 1], 0.0]
        best = _max_x(np.array(rows), np.array(rhs))
        hi[i] = 0.0 if best is None else max(0.0, best)
    x = np.zeros(n_pts)
    u = np.zeros(n_pts)
    for i in range(n_pts - 1):
        lo_u, hi_u = -np.inf, np.inf
        for j in range(n):
            a, c = qs[i, j], qss[i, j] * x[i]
            if abs(a) > 1e-12:
                b1, b2 = (-amax[j] - c) / a, (amax[j] - c) / a
                lo_u, hi_u = max(lo_u, min(b1, b2)), min(hi_u, max(b1, b2))
        lo_u = max(lo_u, -x[i] / (2 * ds[i]))
        hi_u = min(hi_u, (hi[i + 1] - x[i]) / (2 * ds[i]))
        ui = hi_u if hi_u >= lo_u else lo_u
        u[i] = ui
        x[i + 1] = min(max(x[i] + 2 * ds[i] * ui, 0.0), hi[i + 1])
    u[-1] = u[-2] if n_pts > 1 else 0.0
    return x, u


def _time_curve(length, curve, vmax, amax, grid):
    n_pts = max(int(np.ceil(length / grid)) + 1, 30)
    s = np.linspace(0.0, length, n_pts)
    q = np.asarray(curve(s))
    qs = np.asarray(curve(s, 1))
    qss = np.asarray(curve(s, 2))
    ds = np.diff(s)
    x, u = _reachability(qs, qss, ds, vmax, amax)
    sd = np.sqrt(np.maximum(x, 0.0))
    dt = np.empty(n_pts - 1)
    for i in range(n_pts - 1):
        v = sd[i] + sd[i + 1]
        if v > 1e-9:
            dt[i] = 2 * ds[i] / v
        elif abs(u[i]) > 1e-12:
            dt[i] = np.sqrt(2 * ds[i] / abs(u[i]))
        else:
            raise TimingError("the path cannot be moved along: a grid point admits no speed")
    t = np.concatenate([[0.0], np.cumsum(dt)])
    qd = qs * sd[:, None]
    qdd = qs * u[:, None] + qss * x[:, None]
    qd[0] = qd[-1] = 0.0
    return t, q, qd, qdd


def _concat(parts):
    t, q, qd, qdd = [], [], [], []
    offset = 0.0
    for k, (tk, qk, qdk, qddk) in enumerate(parts):
        sl = slice(0 if k == 0 else 1, None)
        t.append(tk[sl] + offset)
        q.append(qk[sl])
        qd.append(qdk[sl])
        qdd.append(qddk[sl])
        offset += tk[-1]
    return np.concatenate(t), np.vstack(q), np.vstack(qd), np.vstack(qdd)


def time_parameterize(
    waypoints,
    vmax,
    amax,
    *,
    checker=None,
    step: float = 0.02,
    densify: float = 0.15,
    grid: float = 0.01,
    torque=None,
    torque_limit=None,
    torque_fraction: float = 0.95,
    max_retimes: int = 12,
) -> TimedPath:
    """Time ``waypoints`` (a list of joint vectors) from rest to rest, as fast as the limits allow.

    Args:
        vmax, amax: per-joint velocity (rad/s) and acceleration (rad/s²) limits
        checker: a :class:`~.collision.CollisionChecker` the smoothed curve must clear
        step: the resolution the curve is checked at (rad)
        densify: the longest polyline piece the spline is drawn through (rad)
        grid: the spacing of the timing grid along the path (rad of chord length)
        torque: ``torque(q, qd, qdd) -> tau`` per joint (inverse dynamics with the payload)
        torque_limit: per-joint |torque| limits the trajectory must stay within, times
            ``torque_fraction``
    """
    pts = [np.asarray(w, float) for w in waypoints]
    if len(pts) < 2 or all(np.max(np.abs(p - pts[0])) < 1e-12 for p in pts[1:]):
        q0 = pts[0]
        z = np.zeros_like(q0)
        return TimedPath(np.array([0.0]), q0[None], z[None], z[None])
    vmax = np.asarray(vmax, float)
    base_amax = np.asarray(amax, float)

    curves = None
    stops = False
    for factor in (1.0, 1 / 3, 1 / 9):
        poly = _polyline(pts, densify * factor)
        length, curve = _spline(poly)
        if checker is None:
            curves = [(length, curve)]
            break
        n = max(2, int(np.ceil(length / (step * 0.5))))
        samples = np.asarray(curve(np.linspace(0.0, length, n)))
        if all(checker.free(q) for q in samples):
            curves = [(length, curve)]
            break
    if curves is None:
        stops = True
        curves = [
            _spline(_polyline([a, b], densify)) for a, b in zip(pts[:-1], pts[1:], strict=True)
        ]

    scale = 1.0
    ratio = None
    for _ in range(max_retimes + 1):
        amax_now = base_amax * scale
        t, q, qd, qdd = _concat([_time_curve(L, c, vmax, amax_now, grid) for L, c in curves])
        if torque is None or torque_limit is None:
            break
        limit = np.asarray(torque_limit, float) * torque_fraction
        worst = max(
            float(np.max(np.abs(torque(q[i], qd[i], qdd[i])) / limit)) for i in range(len(t))
        )
        ratio = worst * torque_fraction
        if worst <= 1.0:
            break
        scale *= max(0.5, min(0.9, 1.0 / worst))
    else:
        raise TimingError(
            f"the path needs {ratio:.0%} of a joint's torque limit even with the accelerations "
            f"scaled to {scale:.2f}; the payload or the pose is beyond the arm"
        )
    return TimedPath(t, q, qd, qdd, accel_scale=scale, torque_ratio=ratio, stops_at_waypoints=stops)
