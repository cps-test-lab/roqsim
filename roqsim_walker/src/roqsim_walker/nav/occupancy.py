"""A 2D occupancy grid for the pedestrian global planner, built by rasterizing
the model's wall footprints (:mod:`~roqsim_walker.nav.obstacles`).

The grid is just the raster the :mod:`~roqsim_walker.nav.planner` A* searches:
``occupied[r, c]`` is ``True`` where a wall covers that cell. It is derived from
the same polygons fed to ORCA, so the planner avoids exactly what ORCA does.
Row 0 is the top; ``origin`` is the world position of the grid's lower-left corner.
"""

from __future__ import annotations

import math

import numpy as np


class OccupancyGrid:
    def __init__(self, occupied: np.ndarray, resolution: float, origin):
        self.occupied = np.ascontiguousarray(occupied, dtype=bool)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.height, self.width = self.occupied.shape

    # -- construction ------------------------------------------------------
    @classmethod
    def from_polygons(
        cls, polygons, resolution: float = 0.05, bounds=None, pad: float = 1.0
    ) -> OccupancyGrid | None:
        """Build a grid covering ``bounds`` (``(minx, miny, maxx, maxy)``, or the
        polygons' extent when ``None``), padded by ``pad`` metres, with every
        polygon stamped occupied. Returns ``None`` when there is nothing to cover."""
        if bounds is None:
            pts = [p for poly in polygons for p in poly]
            if not pts:
                return None
            arr = np.asarray(pts, dtype=float)
            minx, miny = arr.min(0)
            maxx, maxy = arr.max(0)
        else:
            minx, miny, maxx, maxy = bounds
        minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
        w = int(math.ceil((maxx - minx) / resolution))
        h = int(math.ceil((maxy - miny) / resolution))
        if w <= 0 or h <= 0:
            return None
        grid = cls(np.zeros((h, w), dtype=bool), resolution, (minx, miny))
        grid.mark_polygons(polygons)
        return grid

    # -- transforms --------------------------------------------------------
    def world_to_cell(self, x: float, y: float):
        c = int(math.floor((x - self.origin[0]) / self.resolution))
        r = self.height - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        return r, c

    def cell_to_world(self, r: int, c: int):
        x = self.origin[0] + (c + 0.5) * self.resolution
        y = self.origin[1] + (self.height - 1 - r + 0.5) * self.resolution
        return x, y

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, x: float, y: float) -> bool:
        r, c = self.world_to_cell(x, y)
        return self.in_bounds(r, c) and not self.occupied[r, c]

    # -- editing -----------------------------------------------------------
    def mark_polygons(self, polygons) -> int:
        """Stamp filled world-space ``polygons`` (each ``[(x, y), ...]``) as
        occupied. Returns the polygon count applied."""
        n = 0
        for poly in polygons:
            if len(poly) < 3:
                continue
            self._fill_polygon(poly)
            self._stamp_edges(poly)  # guard thin walls < 1 cell wide
            n += 1
        return n

    def _fill_polygon(self, poly):
        cells = [self.world_to_cell(x, y) for x, y in poly]
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = max(min(rs), 0), min(max(rs), self.height - 1)
        c0, c1 = max(min(cs), 0), min(max(cs), self.width - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                x, y = self.cell_to_world(r, c)
                if _point_in_poly(x, y, poly):
                    self.occupied[r, c] = True

    def _stamp_edges(self, poly):
        n = len(poly)
        for i in range(n):
            a = self.world_to_cell(*poly[i])
            b = self.world_to_cell(*poly[(i + 1) % n])
            for r, c in _bresenham(a, b):
                if self.in_bounds(r, c):
                    self.occupied[r, c] = True

    # -- inflation ---------------------------------------------------------
    def inflate(self, radius: float) -> np.ndarray:
        """Occupancy grown by ``radius`` metres (square/Chebyshev dilation) so a
        point-robot A* keeps a body of that radius clear of walls. No SciPy."""
        cells = int(math.ceil(max(radius, 0.0) / self.resolution))
        if cells <= 0:
            return self.occupied.copy()
        occ = self.occupied
        out = occ.copy()
        for _ in range(cells):
            out[:, :-1] |= occ[:, 1:]
            out[:, 1:] |= occ[:, :-1]
            occ = out.copy()
        for _ in range(cells):
            out[:-1, :] |= occ[1:, :]
            out[1:, :] |= occ[:-1, :]
            occ = out.copy()
        return out


def _point_in_poly(x: float, y: float, poly) -> bool:
    """Even-odd ray cast: is ``(x, y)`` inside the polygon ``[(x, y), ...]``?"""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _bresenham(a, b):
    """Integer cells on the line from cell ``a`` to cell ``b`` (inclusive)."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    out = []
    r, c = r0, c0
    while True:
        out.append((r, c))
        if (r, c) == (r1, c1):
            return out
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
