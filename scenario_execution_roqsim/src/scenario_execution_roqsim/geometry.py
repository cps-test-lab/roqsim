# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Where an entity is, as arithmetic: how far from a point, and whether inside a region.

No MuJoCo, no ROS, no scenario-execution, like :mod:`~scenario_execution_roqsim.displacement`, so the
part with a right and a wrong answer is tested as a table.

Every point here is an entity's **reference point**, the origin of its body in the world frame (what
the core's ``sim/entities/<name>/pose`` reports), not the nearest point of its geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How a separation is measured. ``planar`` ignores z: a robot "at" a shelf, a person within a metre
#: of a robot, are statements about the floor plan, and the reference points of two entities of
#: different heights never coincide in z. ``spatial`` is the full 3D distance.
DISTANCE_MODES = ("planar", "spatial")


def separation(mode: str, a, b) -> float:
    """The distance between points *a* and *b* in metres, measured as *mode* says."""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if mode == "planar":
        return float(np.linalg.norm(delta[:2]))
    if mode == "spatial":
        return float(np.linalg.norm(delta))
    raise ValueError(f"unknown distance mode {mode!r}; known: {', '.join(DISTANCE_MODES)}")


def point_of(value) -> tuple[float, float, float]:
    """A ``position_3d`` argument (a mapping of ``x``, ``y``, ``z`` in metres) as a tuple."""
    value = value or {}
    return (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))


class RegionError(ValueError):
    """The points given do not describe a region."""


@dataclass(frozen=True)
class Region:
    """An area of the floor plan: an axis-aligned box or a polygon, in x and y.

    Planar because the regions a scenario names (a zone of the floor, a doorway, a room) are areas
    of the floor plan; a point's z is ignored. A point on the boundary is inside.
    """

    kind: str  # "box" or "polygon"
    vertices: tuple[tuple[float, float], ...]

    @classmethod
    def from_points(cls, points) -> Region:
        """Two points are a box's opposite corners, three or more a polygon's vertices in order."""
        xy = tuple((float(p[0]), float(p[1])) for p in points)
        if len(xy) == 2:
            (x0, y0), (x1, y1) = xy
            if x0 == x1 or y0 == y1:
                raise RegionError(
                    f"a box's two corners {xy[0]} and {xy[1]} must differ in both x and y, or the "
                    "box has no area."
                )
            lo, hi = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
            return cls("box", ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])))
        if len(xy) < 2:
            raise RegionError(
                f"a region needs two points (a box's opposite corners) or three or more (a "
                f"polygon's vertices in order); got {len(xy)}."
            )
        if _area(xy) == 0.0:
            raise RegionError(f"the polygon {list(xy)} has no area: its vertices lie on one line.")
        return cls("polygon", xy)

    def contains(self, point) -> bool:
        """Is *point*'s ``(x, y)`` inside, or on the boundary? Even-odd rule for a polygon."""
        x, y = float(point[0]), float(point[1])
        if self.kind == "box":
            (x0, y0), _, (x1, y1), _ = self.vertices
            return x0 <= x <= x1 and y0 <= y <= y1
        inside = False
        n = len(self.vertices)
        for i in range(n):
            (xa, ya), (xb, yb) = self.vertices[i], self.vertices[(i + 1) % n]
            if _on_segment(x, y, xa, ya, xb, yb):
                return True
            if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
                inside = not inside
        return inside

    def describe(self) -> str:
        if self.kind == "box":
            (x0, y0), _, (x1, y1), _ = self.vertices
            return f"box x {x0:g}..{x1:g}, y {y0:g}..{y1:g}"
        return f"{len(self.vertices)}-gon"


def _area(xy) -> float:
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1], strict=True):
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2.0


def _on_segment(x, y, xa, ya, xb, yb, eps: float = 1e-9) -> bool:
    cross = (xb - xa) * (y - ya) - (yb - ya) * (x - xa)
    if abs(cross) > eps * max(1.0, abs(xb - xa) + abs(yb - ya)):
        return False
    return (
        min(xa, xb) - eps <= x <= max(xa, xb) + eps and min(ya, yb) - eps <= y <= max(ya, yb) + eps
    )
