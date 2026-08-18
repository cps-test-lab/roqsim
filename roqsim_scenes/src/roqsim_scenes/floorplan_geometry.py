# SPDX-License-Identifier: Apache-2.0
"""The 2D geometry of a floorplan sketch -- shared by everything that reads one.

A floorplan JSON (metres, y-up: ``{comment, description, rooms, lines, doors, markers}``, written by
:mod:`roqsim_scenes.dxf_to_floorplan` or the scene-builder's sketch window) has two consumers that must
agree: :mod:`roqsim_scenes.cli.floorplan_to_world` builds the walls from it, and :mod:`roqsim_scenes.floorplan_to_png`
draws them. The wall/opening arithmetic therefore lives here rather than in either one -- an opening the
plan draws but the bake does not cut (or the reverse) would make the preview lie about the world it
claims to show.

Pure metric geometry over the JSON: no MuJoCo, no meshes, no plotting, no numpy.

The room reconstruction (:func:`room_polygons`) chains a room's unordered ``line_ids`` into a closed
loop; ``roqsim_sensors.coverage.regions`` does the same thing for coverage regions, with its own z-banded
``Region`` type on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- walls and openings --------------------------------------------------------------------------

_MIN_WALL_STUB_M = 0.05  # solid wall pieces shorter than this are dropped rather than built


def line_segments(lines: list[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Each structured wall line as an ``((x0,y0),(x1,y1))`` segment -- one wall box each.

    Lines are independent (corners are not shared), so this is a straight unpacking, not a polygon.
    """
    return [
        ((float(x["x0_m"]), float(x["y0_m"])), (float(x["x1_m"]), float(x["y1_m"]))) for x in lines
    ]


def cut_openings(length: float, openings: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Solid sub-intervals ``(t0, t1)`` of a wall of ``length`` after cutting door openings.

    ``openings`` are ``(centre_t_m, width_m)`` along the wall. Overlapping openings merge; stubs
    shorter than ``_MIN_WALL_STUB_M`` vanish (a 3 cm sliver of wall is a mesh artifact, not a wall).
    """
    spans = sorted((t - w / 2, t + w / 2) for t, w in openings)
    out, cur = [], 0.0
    for s0, s1 in spans:
        if s0 - cur >= _MIN_WALL_STUB_M:
            out.append((cur, s0))
        cur = max(cur, s1)
    if length - cur >= _MIN_WALL_STUB_M:
        out.append((cur, length))
    return out


def assign_doors(
    lines: list[dict], doors: list[dict], opening_h: float
) -> dict[int, list[tuple[float, float, float]]]:
    """Group each door's opening onto the wall (line): ``{line index: [(t_m, width, height)]}``.

    A door names its line by ``line_id`` and its position by ``t`` (0..1) along it -- no nearest-wall
    guessing. Its ``height`` is ``height_m`` if the door sets one, else the global ``opening_h`` (the
    wall above it becomes a lintel). An unknown ``line_id``, or a line too short for the opening, is a
    hard error (a door that cannot be cut must not silently vanish).
    """
    by_id = {int(x["id"]): i for i, x in enumerate(lines)}
    per_seg: dict[int, list[tuple[float, float, float]]] = {}
    for d in doors:
        line_id, t, width = int(d["line_id"]), float(d["t"]), float(d.get("width_m", 0.9))
        height = float(d.get("height_m", opening_h))
        if line_id not in by_id:
            raise ValueError(
                f"door {d.get('id', '?')} references line {line_id}, which does not exist"
            )
        idx = by_id[line_id]
        (x0, y0), (x1, y1) = (
            (lines[idx]["x0_m"], lines[idx]["y0_m"]),
            (lines[idx]["x1_m"], lines[idx]["y1_m"]),
        )
        length = math.hypot(x1 - x0, y1 - y0)
        if length < width:
            raise ValueError(
                f"door {d.get('id', '?')} needs a {width:g} m wall but line {line_id} is {length:.2f} m"
            )
        t_m = min(
            max(t * length, width / 2), length - width / 2
        )  # keep the opening inside the wall
        per_seg.setdefault(idx, []).append((t_m, width, height))
    return per_seg


@dataclass
class Opening:
    """One door/window opening, resolved onto its wall: metres along the wall, not a fraction."""

    door_id: int | None
    t_m: float  # centre, measured from the wall's (x0,y0) end
    width_m: float
    height_m: float

    @property
    def span(self) -> tuple[float, float]:
        return (self.t_m - self.width_m / 2, self.t_m + self.width_m / 2)


@dataclass
class WallPlan:
    """A drawn wall line in plan view: its solid pieces and the openings cut out of it."""

    line_id: int
    p0: tuple[float, float]
    p1: tuple[float, float]
    length: float
    solid: list[tuple[float, float]] = field(default_factory=list)  # (t0, t1) metres along the wall
    openings: list[Opening] = field(default_factory=list)

    def point_at(self, t: float) -> tuple[float, float]:
        """The world point ``t`` metres along the wall from ``p0``."""
        ux, uy = self.direction
        return (self.p0[0] + ux * t, self.p0[1] + uy * t)

    @property
    def direction(self) -> tuple[float, float]:
        (x0, y0), (x1, y1) = self.p0, self.p1
        return ((x1 - x0) / self.length, (y1 - y0) / self.length)


def plan_walls(lines: list[dict], doors: list[dict], opening_h: float = 2.0) -> list[WallPlan]:
    """Every wall line with its door openings cut out -- the plan-view counterpart of ``wall_pieces``.

    Same arithmetic the world generator bakes (:func:`assign_doors` + :func:`cut_openings`), reduced to
    2D: no lintels, no z. ``opening_h`` is only the fallback height for an opening that does not set
    ``height_m``, and is carried so a caller can tell a door-height opening from a full-height one.
    """
    per_seg = assign_doors(lines, doors, opening_h)
    # assign_doors appends per line in door order, so the ids of a line's doors -- in that same order --
    # line up with its tuples. This is how an opening keeps the door id it came from without a second,
    # possibly divergent, assignment pass.
    ids_by_seg: dict[int, list[int | None]] = {}
    by_id = {int(x["id"]): i for i, x in enumerate(lines)}
    for d in doors:
        idx = by_id.get(int(d["line_id"]))
        if idx is not None:
            ids_by_seg.setdefault(idx, []).append(d.get("id"))

    out: list[WallPlan] = []
    for idx, ((x0, y0), (x1, y1)) in enumerate(line_segments(lines)):
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0:
            raise ValueError(f"degenerate wall line {lines[idx].get('id', '?')} (zero length)")
        tuples = per_seg.get(idx, [])
        door_ids = ids_by_seg.get(idx, [None] * len(tuples))
        openings = [
            Opening(door_id=did, t_m=t_m, width_m=w, height_m=h)
            for (t_m, w, h), did in zip(tuples, door_ids, strict=True)
        ]
        out.append(
            WallPlan(
                line_id=int(lines[idx]["id"]),
                p0=(float(x0), float(y0)),
                p1=(float(x1), float(y1)),
                length=length,
                solid=cut_openings(length, [(o.t_m, o.width_m) for o in openings]),
                openings=openings,
            )
        )
    return out


def bounds(lines: list[dict]) -> tuple[float, float, float, float]:
    """``(xmin, ymin, xmax, ymax)`` over every wall endpoint. Raises on an empty floorplan."""
    segs = line_segments(lines)
    if not segs:
        raise ValueError("floorplan has no wall lines, so it has no extent")
    xs = [p[0] for seg in segs for p in seg]
    ys = [p[1] for seg in segs for p in seg]
    return (min(xs), min(ys), max(xs), max(ys))


# --- rooms ---------------------------------------------------------------------------------------


@dataclass
class Room:
    """A room recovered as a closed polygon, with its name and floor area."""

    id: int | None
    name: str
    polygon: list[tuple[float, float]]
    description: str = ""

    @property
    def area_m2(self) -> float:
        return polygon_area(self.polygon)


def room_polygons(floorplan: dict, bridge: float = 1.6) -> list[Room]:
    """One closed polygon per room in the floorplan.

    Rooms are stored as *unordered* wall-line id lists, so this chains the segments end-to-end,
    bridging gaps up to ``bridge`` metres (walls meet only approximately, and a room's boundary can run
    across a doorway). A room whose walls do not chain into >= 3 vertices is skipped: an open loop has
    no area, and a polygon closed through a guess would report a wrong one.
    """
    by_id = {int(line["id"]): line for line in floorplan.get("lines", [])}
    out: list[Room] = []
    for room in floorplan.get("rooms", []):
        segs = [
            (
                (float(line["x0_m"]), float(line["y0_m"])),
                (float(line["x1_m"]), float(line["y1_m"])),
            )
            for lid in room.get("line_ids", [])
            if (line := by_id.get(int(lid))) is not None
        ]
        if len(segs) < 3:
            continue
        used = [False] * len(segs)
        chain = [segs[0][0], segs[0][1]]
        used[0] = True
        for _ in range(len(segs) - 1):
            cur = chain[-1]
            best_i, best_end, best_d = -1, None, bridge
            for i, (a, b) in enumerate(segs):
                if used[i]:
                    continue
                for p, q in ((a, b), (b, a)):
                    d = math.dist(cur, p)
                    if d < best_d:
                        best_i, best_end, best_d = i, q, d
            if best_i < 0:
                break
            used[best_i] = True
            chain.append(best_end)
        if len(chain) >= 3:
            out.append(
                Room(
                    id=room.get("id"),
                    name=str(room.get("name", room.get("id", "room"))),
                    polygon=chain,
                    description=str(room.get("description", "")),
                )
            )
    return out


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace area in m^2, unsigned (winding does not matter here)."""
    if len(polygon) < 3:
        return 0.0
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2.0


def point_in_polygon(polygon: list[tuple[float, float]], x: float, y: float) -> bool:
    """Even-odd ray-casting test. Boundary membership is unspecified."""
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[i - 1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
    return inside


def _distance_to_edges(polygon: list[tuple[float, float]], x: float, y: float) -> float:
    """Shortest distance from ``(x, y)`` to any polygon edge."""
    best = math.inf
    for i in range(len(polygon)):
        (x0, y0), (x1, y1) = polygon[i - 1], polygon[i]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg_len2))
        best = min(best, math.dist((x, y), (x0 + t * dx, y0 + t * dy)))
    return best


_SPOT_TOL = 0.02  # samples this close to the best clearance count as equally clear


def label_point(polygon: list[tuple[float, float]], samples: int = 48) -> tuple[float, float]:
    """A point well inside ``polygon`` -- where a room label can go. See :func:`label_spot`."""
    x, y, _ = label_spot(polygon, samples)
    return (x, y)


def label_spot(polygon: list[tuple[float, float]], samples: int = 48) -> tuple[float, float, float]:
    """``(x, y, clearance)``: a point well inside ``polygon``, and its distance to the nearest wall.

    The area centroid falls outside an L-shaped or diagonally-cut room, so this takes the sampled
    interior point furthest from any wall instead (a coarse pole of inaccessibility). In a rectangular
    room that maximum is not a point but a whole segment of the medial axis -- every point down the middle
    of a 6 x 8 m room is 3 m from a wall -- so the *mean* of the equally-clear samples is returned, which
    is the room's middle rather than an arbitrary end of that segment. Falls back to the centroid only
    when no sample lands inside, i.e. for a polygon too thin to sample at this density. The clearance is
    how much room a label has there -- what a renderer needs to size text that fits.
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    inside: list[tuple[float, float, float]] = []
    best_d = 0.0
    for i in range(samples):
        x = min(xs) + (max(xs) - min(xs)) * (i + 0.5) / samples
        for j in range(samples):
            y = min(ys) + (max(ys) - min(ys)) * (j + 0.5) / samples
            if not point_in_polygon(polygon, x, y):
                continue
            d = _distance_to_edges(polygon, x, y)
            inside.append((x, y, d))
            best_d = max(best_d, d)
    clearest = [(x, y) for x, y, d in inside if d >= best_d * (1.0 - _SPOT_TOL)]
    if clearest:
        n = len(clearest)
        return (sum(p[0] for p in clearest) / n, sum(p[1] for p in clearest) / n, best_d)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    return (cx, cy, _distance_to_edges(polygon, cx, cy))
