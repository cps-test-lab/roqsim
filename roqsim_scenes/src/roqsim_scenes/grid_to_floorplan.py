"""Occupancy grid → axis-aligned wall segments → floorplan JSON.

The shared core of the two grid-shaped front-ends, holding everything between *a boolean grid* and
*the floorplan schema* and nothing on either side of that:

* ``mapimage_to_floorplan.py`` reaches it through a picture — decode, de-rotate, rasterise, and the
  grid it lands on is a *measurement of a figure*, walls 2-3 cells thick and speckled with SLAM noise.
* ``gridmap_to_floorplan.py`` reaches it with the grid already in hand, crisp and 1 cell thick.

Both then produce exactly the JSON ``sketch_floorplan_by_human`` returns, so ``floorplan_to_world.py``
downstream cannot tell either of them from a hand-drawn plan.

It lives in its own module rather than in the image tool because the grid route needs no image
library at all: importing the tracer to reach ``segments`` would make Pillow a hard requirement of a
tool that never opens a file it did not already parse as numbers.

The segmentation is greedy and deliberately biased toward long walls: the globally longest run — row
or column — is claimed whole before the shorter walls that touch it, so an L-shape comes out as two
segments rather than four fragments.
"""

from __future__ import annotations

import numpy as np


def _runs(line: np.ndarray, min_cells: int, gap: int) -> list[tuple[int, int]]:
    """Maximal runs of True in `line`, bridging holes of up to `gap` cells."""
    out: list[tuple[int, int]] = []
    idx = np.nonzero(line)[0]
    if idx.size == 0:
        return out
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev <= gap + 1:
            prev = i
            continue
        if prev - start + 1 >= min_cells:
            out.append((int(start), int(prev)))
        start = prev = i
    if prev - start + 1 >= min_cells:
        out.append((int(start), int(prev)))
    return out


def segments(grid: np.ndarray, min_cells: int, gap: int) -> list[tuple[str, int, int, int]]:
    """Greedily extract axis-aligned wall segments: ('h'|'v', fixed, lo, hi) in grid cells.

    Always take the globally longest run — row or column — so a long wall is claimed whole before
    the shorter walls that touch it, and an L-shape comes out as two segments rather than four.
    """
    g = grid.copy()
    found: list[tuple[str, int, int, int]] = []
    while True:
        best = None  # (length, kind, fixed, lo, hi)
        for kind, n in (("h", g.shape[0]), ("v", g.shape[1])):
            for fixed in range(n):
                line = g[fixed, :] if kind == "h" else g[:, fixed]
                for lo, hi in _runs(line, min_cells, gap):
                    length = hi - lo + 1
                    if best is None or length > best[0]:
                        best = (length, kind, fixed, lo, hi)
        if best is None:
            return found
        _, kind, fixed, lo, hi = best
        found.append((kind, fixed, lo, hi))
        # Consume the claimed cells plus one cell to each side across the wall: a traced wall is 2-3
        # cells thick, and leaving the neighbours behind would re-emit the same wall shifted by one.
        for d in (-1, 0, 1):
            f = fixed + d
            if kind == "h" and 0 <= f < g.shape[0]:
                g[f, lo : hi + 1] = False
            elif kind == "v" and 0 <= f < g.shape[1]:
                g[lo : hi + 1, f] = False


def to_floorplan(
    segs: list[tuple[str, int, int, int]],
    rows: int,
    cell_m: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Grid segments -> floorplan JSON (`lines` with x0_m/y0_m/x1_m/y1_m, metres, y up).

    ``rows`` is the grid's row count and must come from the grid itself: it is the only thing that
    turns image rows (down) into floorplan y (up), and a segment's own indices cannot supply it --
    for a vertical segment the `fixed` field is a COLUMN, so deriving it from the segments makes
    every y depend on the image's width and shifts the whole plan on any non-square input (silently,
    since the shape still looks right).

    ``origin`` is the world-metre position of the grid's BOTTOM-LEFT corner, the same ROS map
    convention ``gridmap_to_world.py`` uses, so a floorplan and a map built from one grid land in one
    frame. A traced figure has no such frame and leaves it at (0, 0).
    """
    ox, oy = origin
    lines = []
    for i, (kind, fixed, lo, hi) in enumerate(segs, start=1):
        if kind == "h":
            y = oy + (rows - 1 - fixed + 0.5) * cell_m  # image rows go down; floorplan y goes up
            x0, x1 = ox + lo * cell_m, ox + (hi + 1) * cell_m
            lines.append(
                {
                    "id": i,
                    "x0_m": round(x0, 3),
                    "y0_m": round(y, 3),
                    "x1_m": round(x1, 3),
                    "y1_m": round(y, 3),
                }
            )
        else:
            x = ox + (fixed + 0.5) * cell_m
            y0, y1 = oy + (rows - 1 - hi + 0.5) * cell_m, oy + (rows - 1 - lo + 0.5) * cell_m
            lines.append(
                {
                    "id": i,
                    "x0_m": round(x, 3),
                    "y0_m": round(y0, 3),
                    "x1_m": round(x, 3),
                    "y1_m": round(y1, 3),
                }
            )
    # The same keys the sketch window returns, so floorplan_to_world.py cannot tell the two apart.
    # Wall THICKNESS is deliberately absent: it is not in that schema and the generator takes it as
    # its own --wall-thickness, so a thickness written here would be silently ignored.
    return {"lines": lines, "doors": [], "rooms": [], "markers": []}
