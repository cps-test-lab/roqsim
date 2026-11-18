# SPDX-License-Identifier: Apache-2.0
"""The 2D geometry of a floorplan sketch -- one answer, for everything that reads one.

A floorplan JSON (metres, y-up: ``{version, comment, description, rooms, lines, doors, markers}``,
checked by :func:`check_sketch`) is read by several things that must agree about it: what builds the walls, what draws the plan view, what bakes
a mesh. Where a doorway is has one answer, so the arithmetic lives here rather than in any of them
-- an opening one cuts and another draws differently would make a preview lie about the world it
claims to show, and neither would look wrong on its own.

In core for that reason, beside :mod:`roqsim.floorplan_collision`: a fact about a layout is not the
property of whichever package happened to need it first.

Pure metric geometry over the JSON: no MuJoCo, no meshes, no plotting, no numpy.

The room reconstruction (:func:`room_polygons`) chains a room's unordered ``line_ids`` into a closed
loop. Coverage regions are reconstructed the same way, with a z-banded region type on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .document import check_version, refuse_unknown_keys

# --- the document ------------------------------------------------------------------------------

#: The sketch version this code reads, stated with a top-level ``version:`` (absent is 1). Bumped
#: when a key is renamed or changes meaning.
SKETCH_VERSION = 1

#: Every key a sketch carries, by level. A key outside these is refused: nothing reads it, so a
#: misspelt ``width_m`` would otherwise build the default door.
SKETCH_KEYS = frozenset({"version", "comment", "description", "rooms", "lines", "doors", "markers"})
SKETCH_ITEM_KEYS = {
    "rooms": frozenset({"id", "name", "line_ids", "description"}),
    "lines": frozenset({"id", "x0_m", "y0_m", "x1_m", "y1_m"}),
    "doors": frozenset({"id", "line_id", "t", "width_m", "height_m"}),
    "markers": frozenset({"id", "x_m", "y_m", "comment", "in_room", "yaw_deg"}),
}


def stamp_sketch(sketch: dict) -> dict:
    """*sketch* with ``version`` first, as every writer emits it."""
    return {"version": SKETCH_VERSION, **{k: v for k, v in sketch.items() if k != "version"}}


def check_sketch(sketch: dict, where: str) -> int:
    """Return the sketch's version, refusing a newer one or a key no reader reads, at every level."""
    if not isinstance(sketch, dict):
        raise ValueError(f"{where}: a floorplan sketch is a mapping, not {type(sketch).__name__}")
    # The version first: a newer sketch's new keys are its version's, not typos.
    version = check_version(
        sketch, "version", reads=SKETCH_VERSION, document="floorplan sketch", where=where
    )
    refuse_unknown_keys(sketch, SKETCH_KEYS, where)
    for level, known in SKETCH_ITEM_KEYS.items():
        for i, item in enumerate(sketch.get(level) or []):
            refuse_unknown_keys(item, known, f"{where}: {level}[{i}]")
    return version


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


def wall_pieces(
    lines: list[dict], doors: list[dict], ceiling_h: float, opening_h: float, *, thickness: float
):
    """The wall boxes to build as ``(p0, p1, z0, z1)``, with each line's door openings cut out.

    A drawn line becomes: full-height solid pieces ``(0, ceiling_h)`` around its openings, plus one
    lintel per opening spanning ``(height, ceiling_h)`` over the opening width (skipped when the
    opening reaches the ceiling). So a door is a 2 m-high hole with a beam above it, not a full gap.
    Each box is ``thickness`` wide, centred on the segment ``p0``->``p1``.

    Where walls meet, no two boxes show a face in one plane, which would flicker in every render as
    the camera moves (:func:`_wall_joints`): a box overlapping another stands ``JOINT_INSET_M`` lower
    (and a lintel's underside that much higher) per step, and at a corner of two walls each runs on
    past the corner point until its end face lies ``JOINT_INSET_M`` inside the other wall.
    """
    if thickness <= 0:
        raise ValueError(f"wall thickness must be positive, not {thickness:g}")
    per_seg = assign_doors(lines, doors, opening_h)
    segs = line_segments(lines)
    reach = _corner_extensions(segs, thickness)
    pieces = []
    for i, ((x0, y0), (x1, y1)) in enumerate(segs):
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        # A piece ending at the line's end runs on past it by that corner's extension.
        ends = {0.0: -reach[i][0], length: length + reach[i][1]}

        def at(t: float, x0=x0, y0=y0, ux=ux, uy=uy, ends=ends) -> tuple[float, float]:
            t = ends.get(t, t)
            return (x0 + ux * t, y0 + uy * t)

        openings = per_seg.get(i, [])
        for t0, t1 in cut_openings(length, [(t, w) for t, w, _ in openings]):
            pieces.append((at(t0), at(t1), 0.0, ceiling_h))
        for t_m, width, height in openings:
            if height >= ceiling_h:
                continue  # a full-height opening keeps a true doorway -- no lintel above it
            g0, g1 = max(0.0, t_m - width / 2), min(length, t_m + width / 2)
            pieces.append((at(g0), at(g1), height, ceiling_h))
    return _wall_joints(pieces, thickness)


#: m: how far apart two wall boxes keep their faces where they meet. A render cannot show 1 mm, and
#: it keeps the depth test from having to choose between two faces in one plane.
JOINT_INSET_M = 0.001
#: m: wall ends this close are one corner, and a wall end this close to another wall's line meets it.
_JOINT_TOL_M = 1e-6
#: m²: an overlap of two boxes' footprints smaller than this is two boxes touching, not overlapping.
_OVERLAP_M2 = 1e-9


def _corner_extensions(
    segs: list[tuple[tuple[float, float], tuple[float, float]]], thickness: float
) -> list[tuple[float, float]]:
    """How far each wall runs on past its start and its end, in metres: ``[(before, after)]``.

    A box ending at the corner point leaves the outside of the corner open, a notch half the
    thickness square at a right angle. Where exactly two walls end at one point and no other wall
    passes through it, each runs on past that point as far as its end face stays ``JOINT_INSET_M``
    inside the other wall's sides: ``((t/2)(1 - |cos a|) - inset) / sin a`` for the angle ``a``
    between them, ``t/2 - inset`` at a right angle. The corner is closed but for a sliver of that
    inset, the inner faces end where they did and nothing reaches past the other wall's outer face.
    Walls in line (``a`` = 180 degrees) meet end to end and run on by nothing; so does any end at a
    point three or more walls share, or on another wall's length (a T), which already lies inside
    a wall.
    """
    half = thickness / 2
    ends = [(i, k, seg[k]) for i, seg in enumerate(segs) for k in (0, 1)]
    near = _near_pairs([(p, p) for _, _, p in ends] + [_aabb(seg) for seg in segs])
    meets: dict[int, list[int]] = {}  # end -> the other ends at its point
    through: set[int] = set()  # ends lying on another wall's length
    for e, f in near:
        if f < len(ends):
            if ends[e][0] != ends[f][0] and math.dist(ends[e][2], ends[f][2]) <= _JOINT_TOL_M:
                meets.setdefault(e, []).append(f)
                meets.setdefault(f, []).append(e)
        elif e < len(ends) and ends[e][0] != f - len(ends):
            if _on_interior(ends[e][2], segs[f - len(ends)]):
                through.add(e)
    out = [[0.0, 0.0] for _ in segs]
    for e, (i, k, _) in enumerate(ends):
        others = meets.get(e, [])
        if len(others) != 1 or e in through:
            continue
        j, m, _ = ends[others[0]]
        a, b = _away(segs[i], k), _away(segs[j], m)
        cos_a = a[0] * b[0] + a[1] * b[1]
        sin_a = abs(a[0] * b[1] - a[1] * b[0])
        if sin_a <= 1e-9:
            continue
        out[i][k] = max(0.0, (half * (1.0 - abs(cos_a)) - JOINT_INSET_M) / sin_a)
    return [(before, after) for before, after in out]


def _aabb(points) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def _near_pairs(boxes: list) -> list[tuple[int, int]]:
    """The pairs ``(i, j)``, ``i < j``, of ``((xmin, ymin), (xmax, ymax))`` boxes that touch, within
    ``_JOINT_TOL_M``: hashed into a grid of cells, so a plan of many walls is not compared all to all.
    """
    if not boxes:
        return []
    cell = max(
        1.0, max(max(hi[0] - lo[0], hi[1] - lo[1]) for lo, hi in boxes) / 4
    )  # a long wall spans a few cells, not thousands
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (lo, hi) in enumerate(boxes):
        for cx in range(
            math.floor((lo[0] - _JOINT_TOL_M) / cell), math.floor((hi[0] + _JOINT_TOL_M) / cell) + 1
        ):
            for cy in range(
                math.floor((lo[1] - _JOINT_TOL_M) / cell),
                math.floor((hi[1] + _JOINT_TOL_M) / cell) + 1,
            ):
                grid.setdefault((cx, cy), []).append(i)
    pairs = set()
    for members in grid.values():
        for a, i in enumerate(members):
            for j in members[a + 1 :]:
                (alo, ahi), (blo, bhi) = boxes[i], boxes[j]
                if all(
                    alo[d] <= bhi[d] + _JOINT_TOL_M and blo[d] <= ahi[d] + _JOINT_TOL_M
                    for d in (0, 1)
                ):
                    pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _away(seg, end: int) -> tuple[float, float]:
    """The unit direction from a segment's ``end`` along the segment."""
    (x0, y0), (x1, y1) = seg if end == 0 else (seg[1], seg[0])
    n = math.hypot(x1 - x0, y1 - y0)
    return ((x1 - x0) / n, (y1 - y0) / n)


def _on_interior(p, seg) -> bool:
    """Whether point ``p`` lies on ``seg`` away from both its ends."""
    (x0, y0), (x1, y1) = seg
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    t = ((p[0] - x0) * dx + (p[1] - y0) * dy) / length
    if t <= _JOINT_TOL_M or t >= length - _JOINT_TOL_M:
        return False
    return abs((p[0] - x0) * dy - (p[1] - y0) * dx) / length <= _JOINT_TOL_M


def _wall_joints(pieces: list, thickness: float) -> list:
    """``pieces`` with no two overlapping boxes at one height: each steps ``JOINT_INSET_M`` down.

    Two boxes whose footprints overlap and whose heights overlap show their tops in one plane where
    they overlap (and two lintels their undersides). Taken in order, each box takes the lowest step
    no earlier box it overlaps holds -- at a corner one wall stays at full height and the other
    stands 1 mm lower -- and a step lowers its top and raises a floating box's underside by
    ``JOINT_INSET_M`` each. Boxes that only touch, end to end or side to side, keep their heights.
    """
    feet = [_footprint(p0, p1, thickness) for p0, p1, _, _ in pieces]
    overlaps: dict[int, list[int]] = {}  # piece -> the earlier pieces it overlaps
    for i, j in _near_pairs([_aabb(f) for f in feet]):
        (_, _, zi0, zi1), (_, _, zj0, zj1) = pieces[i], pieces[j]
        if zi0 < zj1 and zj0 < zi1 and _overlap_area(feet[i], feet[j]) > _OVERLAP_M2:
            overlaps.setdefault(j, []).append(i)
    steps: list[int] = []
    for i in range(len(pieces)):
        taken = {steps[j] for j in overlaps.get(i, [])}
        steps.append(next(k for k in range(len(taken) + 1) if k not in taken))
    out = []
    for (p0, p1, z0, z1), k in zip(pieces, steps, strict=True):
        lift = k * JOINT_INSET_M
        out.append((p0, p1, z0 + lift if z0 > 0.0 else z0, z1 - lift))
    return out


def _footprint(p0, p1, thickness: float) -> list[tuple[float, float]]:
    """A wall box's footprint: the rectangle ``thickness`` wide round ``p0``->``p1``, counter-clockwise."""
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0) / length * thickness / 2, (x1 - x0) / length * thickness / 2
    return [(x0 - nx, y0 - ny), (x1 - nx, y1 - ny), (x1 + nx, y1 + ny), (x0 + nx, y0 + ny)]


def _overlap_area(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """The area two convex counter-clockwise polygons share (Sutherland-Hodgman)."""
    out = a
    for i in range(len(b)):
        (ax, ay), (bx, by) = b[i], b[(i + 1) % len(b)]

        def side(p, ax=ax, ay=ay, bx=bx, by=by):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

        kept = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                kept.append(p)
            if (sp >= 0) != (sq >= 0):
                f = sp / (sp - sq)
                kept.append((p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f))
        out = kept
        if not out:
            return 0.0
    return polygon_area(out)


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
