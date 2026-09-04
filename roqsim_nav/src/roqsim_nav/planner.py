"""Grid-based global path planning on an :class:`~roqsim_nav.occupancy.OccupancyGrid`.

Ported from an earlier in-house navigation prototype.

ORCA only avoids *imminent* collisions; it cannot route around a wall. This is the
global layer: 8-connected A* over the (inflated) occupancy grid finds a wall-safe
cell path to a goal, then line-of-sight string-pulling collapses it to a handful
of sparse world waypoints the behaviour tree / ORCA can chase. The two layers are
the textbook "global planner + local controller" split.
"""

from __future__ import annotations

import heapq
import math

_SQRT2 = math.sqrt(2.0)
# 8-connected neighbourhood with step costs (diagonals cost sqrt(2)).
_NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
]


class GridPlanner:
    """Plans world-frame paths on an inflated occupancy grid.

    The grid is inflated once by ``inflation_radius`` (the agent's body radius)
    so the returned path keeps that clearance from walls and A* can treat the
    agent as a point."""

    def __init__(self, grid, inflation_radius: float = 0.0):
        self.grid = grid
        self.inflation_radius = float(inflation_radius)
        self.height, self.width = grid.occupied.shape
        self._occ_cache = {}
        self.occ = self._occ_for(self.inflation_radius)

    def _occ_for(self, radius: float):
        key = round(max(radius, 0.0), 4)
        if key not in self._occ_cache:
            self._occ_cache[key] = self.grid.inflate(key)
        return self._occ_cache[key]

    # -- planning ----------------------------------------------------------
    def plan(self, start_xy, goal_xy):
        """A* from ``start_xy`` to ``goal_xy`` (world metres).

        Returns a simplified list of ``(x, y)`` world waypoints (start excluded,
        goal last), or ``None`` if the goal is unreachable. Plans at the full
        inflation first; if that disconnects the goal (a narrow doorway sealed by
        inflation), retries with progressively less clearance -- ORCA still keeps
        the body off the walls -- before giving up."""
        radii, seen = [], set()
        for cand in (self.inflation_radius, self.inflation_radius * 0.5, 0.0):
            key = round(max(cand, 0.0), 4)
            if key not in seen:
                seen.add(key)
                radii.append(key)
        for radius in radii:
            self.occ = self._occ_for(radius)
            path = self._plan_once(start_xy, goal_xy)
            if path is not None:
                return path
        return None

    def _plan_once(self, start_xy, goal_xy):
        start = self._nearest_free(self.grid.world_to_cell(*start_xy))
        goal = self._nearest_free(self.grid.world_to_cell(*goal_xy))
        if start is None or goal is None:
            return None
        if start == goal:
            return [tuple(goal_xy)]

        cells = self._astar(start, goal)
        if cells is None:
            return None
        pts = self.simplify(cells)
        # Pin the true goal as the final waypoint (cell centre != exact goal).
        pts[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return pts

    def _astar(self, start, goal):
        h = lambda rc: _octile(rc, goal)  # noqa: E731
        open_heap = [(h(start), 0.0, start)]
        came = {start: None}
        g = {start: 0.0}
        occ = self.occ
        H, W = self.height, self.width
        while open_heap:
            _, gc, cur = heapq.heappop(open_heap)
            if cur == goal:
                return _reconstruct(came, goal)
            if gc > g.get(cur, math.inf):
                continue  # stale heap entry
            r, c = cur
            for dr, dc, step in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or occ[nr, nc]:
                    continue
                if dr and dc and (occ[r, nc] or occ[nr, c]):
                    continue  # no corner-cutting
                ng = gc + step
                nxt = (nr, nc)
                if ng < g.get(nxt, math.inf):
                    g[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
        return None

    # -- helpers -----------------------------------------------------------
    def simplify(self, cells):
        """Line-of-sight string-pulling: keep a cell only when the straight
        segment from the last kept cell to the next would clip an obstacle."""
        if len(cells) <= 2:
            return [self.grid.cell_to_world(*c) for c in cells]
        kept = [cells[0]]
        anchor = 0
        for i in range(1, len(cells)):
            if not self._line_of_sight(cells[anchor], cells[i]):
                kept.append(cells[i - 1])
                anchor = i - 1
        kept.append(cells[-1])
        return [self.grid.cell_to_world(*c) for c in kept]

    def _line_of_sight(self, a, b) -> bool:
        """True if every cell on the Bresenham line a->b is free."""
        (r0, c0), (r1, c1) = a, b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        occ = self.occ
        while True:
            if occ[r, c]:
                return False
            if (r, c) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _nearest_free(self, cell):
        """Nearest non-occupied cell to ``cell`` by expanding-ring search; the
        cell itself if already free, or ``None`` within a small search budget."""
        r0, c0 = cell
        H, W = self.height, self.width
        if 0 <= r0 < H and 0 <= c0 < W and not self.occ[r0, c0]:
            return cell
        for rad in range(1, max(H, W)):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue  # ring perimeter only
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < H and 0 <= c < W and not self.occ[r, c]:
                        return (r, c)
            if rad > 40:  # ~2 m at 0.05 m/cell
                break
        return None


def _octile(a, b) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)


def _reconstruct(came, goal):
    path = [goal]
    cur = came[goal]
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path
