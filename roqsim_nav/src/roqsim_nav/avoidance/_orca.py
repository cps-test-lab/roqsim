"""The ORCA solve for one agent: its half-planes and the linear programs that pick its velocity.

A port of the RVO2 Library's ``Agent::computeNewVelocity`` and ``linearProgram1``-``3`` (van den
Berg, Guy, Lin, Manocha, "Reciprocal n-body collision avoidance", ISRR 2009; Copyright 2008
University of North Carolina at Chapel Hill, Apache License 2.0), in plain Python floats. Its k-d
trees are not ported: a scan finds the agents and edges near an agent, for the handful of movers a
world has, and can leave out what an agent ignores. RVO2's obstacle tree also splits every edge
another edge's line crosses, at a vertex it takes as convex; here an edge stays whole, so where
walls meet or overlap an agent can choose another velocity than RVO2's, as clear of the walls.

A velocity constraint is a directed line ``(px, py, dx, dy)``; the permitted half-plane is to its
left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: RVO2's tolerance for parallel lines and covered edges.
EPSILON = 1e-5


def _det(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _normalize(x: float, y: float) -> tuple[float, float]:
    n = math.hypot(x, y)
    return x / n, y / n


@dataclass
class Edges:
    """Every edge of the static obstacles, as RVO2 links them: an edge runs from its vertex to the
    next one of its polygon, counter-clockwise, so the free side is to its right."""

    point: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    unit: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    nxt: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    prev: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    convex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def add(self, vertices) -> None:
        """One polygon, or a segment with two vertices, as ``RVOSimulator::addObstacle`` adds it."""
        pts = np.asarray(vertices, dtype=float).reshape(-1, 2)
        n = len(pts)
        if n < 2:
            return
        base = len(self.point)
        idx = np.arange(n)
        nxt = (idx + 1) % n
        prv = (idx - 1) % n
        d = pts[nxt] - pts
        unit = d / np.linalg.norm(d, axis=1)[:, None]
        if n == 2:
            convex = np.ones(n, dtype=bool)
        else:
            # leftOf(prev, this, next) >= 0: det(prev - next, this - prev)
            a, b, c = pts[prv], pts, pts[nxt]
            convex = (
                (a[:, 0] - c[:, 0]) * (b[:, 1] - a[:, 1])
                - (a[:, 1] - c[:, 1]) * (b[:, 0] - a[:, 0])
            ) >= 0.0
        self.point = np.vstack([self.point, pts])
        self.unit = np.vstack([self.unit, unit])
        self.nxt = np.concatenate([self.nxt, base + nxt])
        self.prev = np.concatenate([self.prev, base + prv])
        self.convex = np.concatenate([self.convex, convex])

    def near(self, pos, range_sq: float) -> list[int]:
        """The edges an agent at ``pos`` sees within ``sqrt(range_sq)``, nearest first.

        What RVO2's obstacle tree returns: an edge counts when the agent is on its free side and the
        segment is nearer than the range; ties keep the order the edges were added in.
        """
        if not len(self.point):
            return []
        a = self.point
        b = self.point[self.nxt]
        ab = b - a
        ap = np.asarray(pos, dtype=float) - a
        # leftOf(a, b, pos) = det(a - pos, b - a); the agent sees the edge from its right.
        left = (-ap[:, 0]) * ab[:, 1] - (-ap[:, 1]) * ab[:, 0]
        r = np.clip(np.einsum("ij,ij->i", ap, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
        closest = a + r[:, None] * ab
        dist_sq = np.einsum("ij,ij->i", closest - pos, closest - pos)
        hit = np.flatnonzero((left < 0.0) & (dist_sq < range_sq))
        return [int(i) for i in hit[np.argsort(dist_sq[hit], kind="stable")]]


def obstacle_lines(edges: Edges, near, pos, vel, radius: float, time_horizon_obst: float):
    """The half-planes that keep an agent out of the walls within ``time_horizon_obst``."""
    lines: list[tuple[float, float, float, float]] = []
    inv = 1.0 / time_horizon_obst
    px, py = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    radius_sq = radius * radius
    point, unit, nxt, prev, convex = edges.point, edges.unit, edges.nxt, edges.prev, edges.convex
    for o1 in near:
        o2 = int(nxt[o1])
        r1x, r1y = point[o1, 0] - px, point[o1, 1] - py
        r2x, r2y = point[o2, 0] - px, point[o2, 1] - py

        # Already taken care of by a line built for an earlier edge?
        covered = False
        for lpx, lpy, ldx, ldy in lines:
            if (
                _det(inv * r1x - lpx, inv * r1y - lpy, ldx, ldy) - inv * radius >= -EPSILON
                and _det(inv * r2x - lpx, inv * r2y - lpy, ldx, ldy) - inv * radius >= -EPSILON
            ):
                covered = True
                break
        if covered:
            continue

        dist_sq1 = r1x * r1x + r1y * r1y
        dist_sq2 = r2x * r2x + r2y * r2y
        ovx, ovy = point[o2, 0] - point[o1, 0], point[o2, 1] - point[o1, 1]
        s = (-r1x * ovx - r1y * ovy) / (ovx * ovx + ovy * ovy)
        lx, ly = -r1x - s * ovx, -r1y - s * ovy
        dist_sq_line = lx * lx + ly * ly

        if s < 0.0 and dist_sq1 <= radius_sq:
            # In collision with the left vertex; ignored where it is not convex.
            if convex[o1]:
                dx, dy = _normalize(-r1y, r1x)
                lines.append((0.0, 0.0, dx, dy))
            continue
        if s > 1.0 and dist_sq2 <= radius_sq:
            # In collision with the right vertex; ignored where it is not convex, or where the next
            # edge takes care of it.
            if convex[o2] and _det(r2x, r2y, unit[o2, 0], unit[o2, 1]) >= 0.0:
                dx, dy = _normalize(-r2y, r2x)
                lines.append((0.0, 0.0, dx, dy))
            continue
        if 0.0 <= s < 1.0 and dist_sq_line <= radius_sq:
            # In collision with the edge.
            lines.append((0.0, 0.0, -unit[o1, 0], -unit[o1, 1]))
            continue

        # No collision: the legs of the velocity obstacle. Seen obliquely both come from one
        # vertex; at a vertex that is not convex a leg extends the cut-off line.
        if s < 0.0 and dist_sq_line <= radius_sq:
            if not convex[o1]:
                continue
            o2 = o1
            leg = math.sqrt(dist_sq1 - radius_sq)
            llx = (r1x * leg - r1y * radius) / dist_sq1
            lly = (r1x * radius + r1y * leg) / dist_sq1
            rlx = (r1x * leg + r1y * radius) / dist_sq1
            rly = (-r1x * radius + r1y * leg) / dist_sq1
        elif s > 1.0 and dist_sq_line <= radius_sq:
            if not convex[o2]:
                continue
            o1 = o2
            leg = math.sqrt(dist_sq2 - radius_sq)
            llx = (r2x * leg - r2y * radius) / dist_sq2
            lly = (r2x * radius + r2y * leg) / dist_sq2
            rlx = (r2x * leg + r2y * radius) / dist_sq2
            rly = (-r2x * radius + r2y * leg) / dist_sq2
        else:
            if convex[o1]:
                leg = math.sqrt(dist_sq1 - radius_sq)
                llx = (r1x * leg - r1y * radius) / dist_sq1
                lly = (r1x * radius + r1y * leg) / dist_sq1
            else:
                llx, lly = -unit[o1, 0], -unit[o1, 1]
            if convex[o2]:
                leg = math.sqrt(dist_sq2 - radius_sq)
                rlx = (r2x * leg + r2y * radius) / dist_sq2
                rly = (-r2x * radius + r2y * leg) / dist_sq2
            else:
                rlx, rly = unit[o1, 0], unit[o1, 1]

        # A leg at a convex vertex never points into the neighbouring edge: that edge's cut-off
        # line is taken instead, and a velocity projected onto such a "foreign" leg adds nothing.
        left_neighbor = int(prev[o1])
        left_foreign = right_foreign = False
        if convex[o1] and _det(llx, lly, -unit[left_neighbor, 0], -unit[left_neighbor, 1]) >= 0.0:
            llx, lly = -unit[left_neighbor, 0], -unit[left_neighbor, 1]
            left_foreign = True
        if convex[o2] and _det(rlx, rly, unit[o2, 0], unit[o2, 1]) <= 0.0:
            rlx, rly = unit[o2, 0], unit[o2, 1]
            right_foreign = True

        lcx, lcy = inv * (point[o1, 0] - px), inv * (point[o1, 1] - py)
        rcx, rcy = inv * (point[o2, 0] - px), inv * (point[o2, 1] - py)
        cvx, cvy = rcx - lcx, rcy - lcy
        same = o1 == o2
        t = 0.5 if same else ((vx - lcx) * cvx + (vy - lcy) * cvy) / (cvx * cvx + cvy * cvy)
        t_left = (vx - lcx) * llx + (vy - lcy) * lly
        t_right = (vx - rcx) * rlx + (vy - rcy) * rly

        if (t < 0.0 and t_left < 0.0) or (same and t_left < 0.0 and t_right < 0.0):
            # Onto the left cut-off circle.
            ux, uy = _normalize(vx - lcx, vy - lcy)
            lines.append((lcx + radius * inv * ux, lcy + radius * inv * uy, uy, -ux))
            continue
        if t > 1.0 and t_right < 0.0:
            # Onto the right cut-off circle.
            ux, uy = _normalize(vx - rcx, vy - rcy)
            lines.append((rcx + radius * inv * ux, rcy + radius * inv * uy, uy, -ux))
            continue

        # Onto the left leg, the right leg or the cut-off line, whichever is nearest.
        inf = math.inf
        d_cut = (
            inf
            if (t < 0.0 or t > 1.0 or same)
            else (vx - (lcx + t * cvx)) ** 2 + (vy - (lcy + t * cvy)) ** 2
        )
        d_left = (
            inf
            if t_left < 0.0
            else (vx - (lcx + t_left * llx)) ** 2 + (vy - (lcy + t_left * lly)) ** 2
        )
        d_right = (
            inf
            if t_right < 0.0
            else (vx - (rcx + t_right * rlx)) ** 2 + (vy - (rcy + t_right * rly)) ** 2
        )
        if d_cut <= d_left and d_cut <= d_right:
            dx, dy = -unit[o1, 0], -unit[o1, 1]
            lines.append((lcx + radius * inv * -dy, lcy + radius * inv * dx, dx, dy))
        elif d_left <= d_right:
            if left_foreign:
                continue
            lines.append((lcx + radius * inv * -lly, lcy + radius * inv * llx, llx, lly))
        else:
            if right_foreign:
                continue
            dx, dy = -rlx, -rly
            lines.append((rcx + radius * inv * -dy, rcy + radius * inv * dx, dx, dy))
    return lines


def agent_line(pos, vel, radius, other_pos, other_vel, other_radius, time_horizon, time_step):
    """The half-plane that leaves half of avoiding ``other`` to this agent, or ``None`` where two
    agents share a place and a velocity and nothing says which way either should leave."""
    rpx, rpy = other_pos[0] - pos[0], other_pos[1] - pos[1]
    rvx, rvy = vel[0] - other_vel[0], vel[1] - other_vel[1]
    dist_sq = rpx * rpx + rpy * rpy
    combined = radius + other_radius
    combined_sq = combined * combined
    if dist_sq > combined_sq:
        inv = 1.0 / time_horizon
        wx, wy = rvx - inv * rpx, rvy - inv * rpy
        w_len_sq = wx * wx + wy * wy
        dot1 = wx * rpx + wy * rpy
        if dot1 < 0.0 and dot1 * dot1 > combined_sq * w_len_sq:
            # Onto the cut-off circle.
            w_len = math.sqrt(w_len_sq)
            ux, uy = wx / w_len, wy / w_len
            dx, dy = uy, -ux
            ex, ey = (combined * inv - w_len) * ux, (combined * inv - w_len) * uy
        else:
            # Onto a leg.
            leg = math.sqrt(dist_sq - combined_sq)
            if _det(rpx, rpy, wx, wy) > 0.0:
                dx = (rpx * leg - rpy * combined) / dist_sq
                dy = (rpx * combined + rpy * leg) / dist_sq
            else:
                dx = -(rpx * leg + rpy * combined) / dist_sq
                dy = -(-rpx * combined + rpy * leg) / dist_sq
            dot2 = rvx * dx + rvy * dy
            ex, ey = dot2 * dx - rvx, dot2 * dy - rvy
    else:
        # Already overlapping: out within one time step.
        inv = 1.0 / time_step
        wx, wy = rvx - inv * rpx, rvy - inv * rpy
        w_len = math.hypot(wx, wy)
        if w_len == 0.0:
            return None  # the same place at the same velocity: no side to leave it by
        ux, uy = wx / w_len, wy / w_len
        dx, dy = uy, -ux
        ex, ey = (combined * inv - w_len) * ux, (combined * inv - w_len) * uy
    return (vel[0] + 0.5 * ex, vel[1] + 0.5 * ey, dx, dy)


def _lp1(lines, n, radius, opt, direction_opt):
    """The best point on line ``n`` that satisfies lines ``0..n-1``, or ``None``."""
    px, py, dx, dy = lines[n]
    dot = px * dx + py * dy
    disc = dot * dot + radius * radius - (px * px + py * py)
    if disc < 0.0:
        return None  # the speed circle leaves nothing of line n
    sq = math.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq
    for i in range(n):
        ipx, ipy, idx_, idy = lines[i]
        denominator = _det(dx, dy, idx_, idy)
        numerator = _det(idx_, idy, px - ipx, py - ipy)
        if abs(denominator) <= EPSILON:
            if numerator < 0.0:
                return None  # parallel, and on the forbidden side
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        t = t_right if opt[0] * dx + opt[1] * dy > 0.0 else t_left
    else:
        t = dx * (opt[0] - px) + dy * (opt[1] - py)
        t = min(max(t, t_left), t_right)
    return (px + t * dx, py + t * dy)


def _lp2(lines, radius, opt, direction_opt):
    """The velocity nearest ``opt`` inside the speed circle and every half-plane.

    Returns ``(velocity, failed)``: ``failed`` is the first line it could not satisfy, or
    ``len(lines)``.
    """
    if direction_opt:
        result = (opt[0] * radius, opt[1] * radius)
    elif opt[0] * opt[0] + opt[1] * opt[1] > radius * radius:
        nx, ny = _normalize(opt[0], opt[1])
        result = (nx * radius, ny * radius)
    else:
        result = (opt[0], opt[1])
    for i, (px, py, dx, dy) in enumerate(lines):
        if _det(dx, dy, px - result[0], py - result[1]) > 0.0:
            new = _lp1(lines, i, radius, opt, direction_opt)
            if new is None:
                return result, i
            result = new
    return result, len(lines)


def _lp3(lines, num_obst_lines, begin, radius, result):
    """Where no velocity satisfies every line: the one that violates the agents' lines least,
    the walls' lines kept."""
    distance = 0.0
    for i in range(begin, len(lines)):
        ipx, ipy, idx_, idy = lines[i]
        if _det(idx_, idy, ipx - result[0], ipy - result[1]) > distance:
            proj = list(lines[:num_obst_lines])
            for j in range(num_obst_lines, i):
                jpx, jpy, jdx, jdy = lines[j]
                determinant = _det(idx_, idy, jdx, jdy)
                if abs(determinant) <= EPSILON:
                    if idx_ * jdx + idy * jdy > 0.0:
                        continue  # parallel, the same way
                    point = (0.5 * (ipx + jpx), 0.5 * (ipy + jpy))
                else:
                    s = _det(jdx, jdy, ipx - jpx, ipy - jpy) / determinant
                    point = (ipx + s * idx_, ipy + s * idy)
                ndx, ndy = _normalize(jdx - idx_, jdy - idy)
                proj.append((point[0], point[1], ndx, ndy))
            new, failed = _lp2(proj, radius, (-idy, idx_), True)
            if failed >= len(proj):
                result = new
            # else: floating-point error on a result that is feasible by construction; keep it
            distance = _det(idx_, idy, ipx - result[0], ipy - result[1])
    return result


def new_velocity(obst_lines, agent_lines, max_speed: float, pref) -> tuple[float, float]:
    """The velocity ORCA chooses: nearest ``pref`` within ``max_speed`` and every half-plane, or,
    where none is, the one that violates the agents' half-planes least."""
    lines = list(obst_lines) + [line for line in agent_lines if line is not None]
    result, failed = _lp2(lines, max_speed, (float(pref[0]), float(pref[1])), False)
    if failed < len(lines):
        result = _lp3(lines, len(obst_lines), failed, max_speed, result)
    return result
