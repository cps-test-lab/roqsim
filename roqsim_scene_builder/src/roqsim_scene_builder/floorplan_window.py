# SPDX-License-Identifier: Apache-2.0
"""The native 2D floorplan-sketch window (tkinter, no MuJoCo).

The second annotation window of the scene builder. Where :mod:`scene_window` reviews a *3D* scene,
this one *authors the walls of a 2D top-view floorplan* directly as **lines** and returns them in
metres. It is the human input the deterministic world generator
(``roqsim scenes floorplan-to-world``) turns into a world -- the LLM never hand-draws it.

There are five exclusive **modes** (default *draw*):

* **draw** -- draw a wall **freehand** (press-drag-release), **lineified immediately** on release
  (the raw pencil is discarded), leaving straight wall lines (near-axis ones snapped level/plumb).
  Every wall end connects to what is there: near an existing **point** it reuses it exactly; on a
  line's **interior** it splits that line; and **crossing** a line creates a point at the crossing
  and splits both lines through it.
* **move** -- drag an existing **point** (a wall endpoint, or a marker); wall endpoints that share a
  point move together (so connected walls stay connected), and dropping near another point joins them.
* **door** -- place standard-width door *openings* on wall lines (hover previews, click places; the
  generator cuts the opening out; a door rides along its line and dies with it).
* **delete** -- a click removes what is under it: a marker, else a door (only the door, never its
  wall), else the line.
* **mark** -- click to drop a **prop marker**; type what goes there in its comment ("office table").
  Keep the button held and **drag a direction** out of the point to set the prop's heading (a small
  arrow shows it); a plain click leaves it headingless. The generator maps each marker's comment to a
  model and its heading to the prop's yaw. Markers can also come from 3D-review dots.

Lines are independent segments with **stable, monotonic ids** (a split keeps the first half's id and
mints a new id for the second) so "move line 3 two metres left" always means the same wall. Walls
that enclose a closed loop are reported as **rooms** (nameable), listed above the lines. **↶/↷
undo/redo** the last edit; **the wheel zooms** (about the cursor) and **the right button pans** (a
no-op on an empty canvas).

As in :mod:`scene_window`, the non-GUI core (the model, the mapping, snapping/splitting, result
assembly) is kept free of tkinter so it can be tested headless.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from roqsim_scene_builder.annotate_ui import (
    BG,
    BORDER,
    ENTRY_BG,
    FG,
    MUTED,
    PANEL,
    SEND_BG,
    build_comment_box,
    build_point_rows,
    build_scrollable,
    color_for,
    enable_edit_shortcuts,
    renumber,
    rgba_hex,
)

_DEFAULT_SIZE = (760, 760)
_DEFAULT_DIMS = (10.0, 10.0)  # width_m, height_m of a fresh floorplan
_DOOR_WIDTH_M = 0.9  # standard door(-opening) width; fixed for now (used as a dataclass default)
_SPLIT_END_MARGIN_M = (
    0.1  # a wall end this close to a line's end snaps to it, not splits (method default)
)


# --- non-GUI core (testable without a display) ---------------------------------------------------


@dataclass
class Line:
    """One wall -- an independent segment with its own two endpoints, in metres, and a stable id."""

    id: int
    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float

    @property
    def label(self) -> str:
        return f"({self.x0_m:.1f},{self.y0_m:.1f}) → ({self.x1_m:.1f},{self.y1_m:.1f})"

    @property
    def length_m(self) -> float:
        return math.hypot(self.x1_m - self.x0_m, self.y1_m - self.y0_m)

    def endpoint(self, end: int) -> tuple[float, float]:
        return (self.x0_m, self.y0_m) if end == 0 else (self.x1_m, self.y1_m)

    def set_endpoint(self, end: int, x_m: float, y_m: float) -> None:
        if end == 0:
            self.x0_m, self.y0_m = x_m, y_m
        else:
            self.x1_m, self.y1_m = x_m, y_m


@dataclass
class Door:
    """A door -- for now purely an *opening* of standard width cut into a wall.

    Attached to a line: ``line_id`` names the wall and ``t`` (0..1) the fraction along it where the
    opening is centred, so the door follows the line when it moves and dies with it.
    """

    id: int
    line_id: int
    t: float
    width_m: float = _DOOR_WIDTH_M
    comment: str = ""  # unused; present so door rows reuse build_point_rows

    @property
    def label(self) -> str:
        return f"on line {self.line_id}, {self.width_m:g} m wide"


@dataclass
class Marker:
    """A prop marker -- a point (metres) with a ``comment`` naming what goes there (e.g. "office
    table"). The comment feeds the generator's marker -> model mapping.

    ``yaw_deg`` is the prop's optional heading about +Z (degrees, 0 = +x, CCW), drawn by dragging a
    direction out of the point in Mark mode. ``None`` means no heading was drawn -- the prop is placed
    axis-aligned unless the markers-map overrides it.
    """

    id: int
    x_m: float
    y_m: float
    comment: str = ""
    yaw_deg: float | None = None

    @property
    def label(self) -> str:
        arrow = f" ∠{self.yaw_deg:.0f}°" if self.yaw_deg is not None else ""
        return f"({self.x_m:.1f}, {self.y_m:.1f}) m{arrow}"


def axis_snap(
    points: list[tuple[float, float]], threshold_deg: float = 10.0
) -> list[tuple[float, float]]:
    """Make nearly-horizontal/vertical segments exactly so (a freehand wall is rarely level).

    Walks the polyline snapping each segment against its (already snapped) predecessor; segments
    further than ``threshold_deg`` from both axes (a real diagonal) are left alone.
    """
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for p in points[1:]:
        x0, y0 = out[-1]
        ang = math.degrees(math.atan2(p[1] - y0, p[0] - x0)) % 180.0
        if min(ang, 180.0 - ang) <= threshold_deg:
            out.append((p[0], y0))
        elif abs(ang - 90.0) <= threshold_deg:
            out.append((x0, p[1]))
        else:
            out.append(p)
    return out


def segment_intersection(p0, p1, q0, q1, eps: float = 1e-9):
    """The point where segments ``p0-p1`` and ``q0-q1`` properly cross, or ``None``.

    Only a strict interior crossing counts -- segments that merely share an endpoint (or are
    parallel) return ``None``, so touching at a shared point is not treated as a crossing.
    """
    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < eps:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if eps < t < 1 - eps and eps < u < 1 - eps:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def detect_rooms(lines: list[Line]) -> list[dict]:
    """Find the closed loops (rooms) the walls enclose, as a list of ``{"line_ids", "points"}``.

    Treats the walls as a planar graph (endpoints that coincide are the same node) and walks its
    bounded faces. Each bounded face is a room; the unbounded outer face and any dangling walls are
    not rooms. Returned in descending area so the biggest room lists first.
    """

    def key(p):
        return (round(p[0], 3), round(p[1], 3))

    coords, adj = {}, {}
    for line in lines:
        a, b = key(line.endpoint(0)), key(line.endpoint(1))
        if a == b:
            continue
        coords[a], coords[b] = line.endpoint(0), line.endpoint(1)
        adj.setdefault(a, []).append((b, line.id))
        adj.setdefault(b, []).append((a, line.id))
    for node, nbrs in adj.items():
        (nx, ny) = coords[node]
        nbrs.sort(key=lambda nb: math.atan2(coords[nb[0]][1] - ny, coords[nb[0]][0] - nx))

    visited: set = set()
    rooms = []
    for start in list(adj):
        for he in [(start, nb, lid) for nb, lid in adj[start]]:
            if he in visited:
                continue
            nodes, line_ids, cur = [], [], he
            while cur not in visited:
                visited.add(cur)
                cu, cv, clid = cur
                nodes.append(cu)
                line_ids.append(clid)
                nbrs = adj[cv]
                idx = next(i for i, (nb, lid) in enumerate(nbrs) if nb == cu and lid == clid)
                nb_node, nb_lid = nbrs[
                    (idx - 1) % len(nbrs)
                ]  # clockwise-most turn -> CCW inner faces
                cur = (cv, nb_node, nb_lid)
            pts = [coords[n] for n in nodes]
            area = 0.5 * sum(
                pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                for i in range(len(pts))
            )
            if area > 1e-6:  # a bounded interior face (CCW); the outer face is CW / negative
                rooms.append({"line_ids": sorted(set(line_ids)), "points": pts, "area": area})
    rooms.sort(key=lambda r: r["area"], reverse=True)
    return [{"line_ids": r["line_ids"], "points": r["points"]} for r in rooms]


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (poly is a list of (x, y) vertices).

    The same test as ``roqsim_scenes.floorplan_geometry.point_in_polygon`` (argument order differs).
    Duplicated on purpose: importing it would make this MuJoCo-free 2D window depend on all of
    ``roqsim_scenes`` -- scipy, lxml, requests -- for eight lines of arithmetic that cannot drift,
    having no tuning constant to disagree about. The door clamp below is the one that *can*; see
    :func:`door_interval`.
    """
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def rooms_with_names(lines: list[Line], room_names: dict) -> list[dict]:
    """:func:`detect_rooms` plus a 1-based id and a name (the stored one, else default ``room N``)."""
    out = []
    for i, room in enumerate(detect_rooms(lines), start=1):
        name = room_names.get(frozenset(room["line_ids"])) or f"room {i}"
        out.append({"id": i, "name": name, "line_ids": room["line_ids"], "points": room["points"]})
    return out


def shy_hex(item_id: int, alpha: float = 0.22) -> str:
    """A faint fill colour for a room -- ``color_for(item_id)`` blended toward the dark background."""
    r, g, b, _ = color_for(item_id)
    bg = 0x1E / 255
    return "#{:02x}{:02x}{:02x}".format(
        *(int((c * alpha + bg * (1 - alpha)) * 255) for c in (r, g, b))
    )


@dataclass
class SketchModel:
    """Independent wall lines + doors + prop markers, plus room names keyed by their wall-id set.

    ``description`` (floorplan-level) and ``room_descriptions`` (per room, keyed by the room's
    wall-id set like ``room_names``) are free-text intent for the object placer -- what the space is
    for, so an agent knows which props belong. Unlike the feedback ``comment``, they are seeded from
    ``initial`` and round-trip through the scene.
    """

    lines: list[Line] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    room_names: dict = field(default_factory=dict)  # frozenset(line_ids) -> name
    description: str = ""  # floorplan-level object-placement intent
    room_descriptions: dict = field(default_factory=dict)  # frozenset(line_ids) -> description

    def _next_line_id(self) -> int:
        """Monotonic line id -- ids never shift, so "line 3" stays valid and a split mints a new id."""
        return max((line.id for line in self.lines), default=0) + 1

    def add_line(
        self, x0_m: float, y0_m: float, x1_m: float, y1_m: float, line_id: int | None = None
    ) -> Line:
        if (
            line_id is None
        ):  # interactive add: a wall between two points that already have one is a no-op
            existing = self._line_between(x0_m, y0_m, x1_m, y1_m)
            if existing is not None:
                return existing
        line = Line(
            id=self._next_line_id() if line_id is None else int(line_id),
            x0_m=float(x0_m),
            y0_m=float(y0_m),
            x1_m=float(x1_m),
            y1_m=float(y1_m),
        )
        self.lines.append(line)
        return line

    def _line_between(
        self, x0_m: float, y0_m: float, x1_m: float, y1_m: float, eps_m: float = 1e-6
    ) -> Line | None:
        """An existing line with these two endpoints (in either direction), or ``None``."""

        def same(ax, ay, bx, by) -> bool:
            return math.hypot(ax - bx, ay - by) <= eps_m

        for ln in self.lines:
            (a, b) = ln.endpoint(0), ln.endpoint(1)
            if (same(*a, x0_m, y0_m) and same(*b, x1_m, y1_m)) or (
                same(*a, x1_m, y1_m) and same(*b, x0_m, y0_m)
            ):
                return ln
        return None

    def prune_lines(self, eps_m: float = 1e-6) -> None:
        """Clean up after a point is dragged onto another: drop any wall that collapsed to zero
        length, then merge walls that now share both endpoints (either direction) into the earliest.

        Doors follow the surviving wall (``t`` flips to ``1 - t`` when the survivor runs the opposite
        way); doors on a vanished zero-length wall go with it.
        """
        for ln in [x for x in self.lines if x.length_m <= eps_m]:
            self.delete_line(ln.id)  # cascades this wall's doors, keeps door ids 1..N
        kept: list[Line] = []
        for ln in self.lines:
            dup = self._matching(kept, ln, eps_m)
            if dup is None:
                kept.append(ln)
                continue
            flip = (
                math.hypot(*(a - b for a, b in zip(dup.endpoint(0), ln.endpoint(0), strict=True)))
                > eps_m
            )
            for d in self.doors:
                if d.line_id == ln.id:
                    d.line_id, d.t = dup.id, (1.0 - d.t if flip else d.t)
        self.lines = kept
        renumber(self.doors)

    @staticmethod
    def _matching(lines: list[Line], target: Line, eps_m: float) -> Line | None:
        """A line in ``lines`` sharing ``target``'s two endpoints (either direction), or ``None``."""

        def same(p, q) -> bool:
            return math.hypot(p[0] - q[0], p[1] - q[1]) <= eps_m

        ta, tb = target.endpoint(0), target.endpoint(1)
        for ln in lines:
            a, b = ln.endpoint(0), ln.endpoint(1)
            if (same(a, ta) and same(b, tb)) or (same(a, tb) and same(b, ta)):
                return ln
        return None

    def delete_line(self, line_id: int) -> None:
        # A door is attached to its line: removing the line removes its doors too. Line ids do NOT
        # renumber (they are stable references); door ids are labels, kept 1..N.
        self.lines = [line for line in self.lines if line.id != line_id]
        self.doors = [d for d in self.doors if d.line_id != line_id]
        renumber(self.doors)

    def add_door(self, line_id: int, t: float, width_m: float = _DOOR_WIDTH_M) -> Door:
        d = Door(id=len(self.doors) + 1, line_id=int(line_id), t=float(t), width_m=float(width_m))
        self.doors.append(d)
        return d

    def delete_door(self, door_id: int) -> None:
        self.doors = [d for d in self.doors if d.id != door_id]
        renumber(self.doors)

    def add_marker(self, x_m: float, y_m: float, marker_id: int | None = None) -> Marker:
        m = Marker(
            id=(len(self.markers) + 1 if marker_id is None else int(marker_id)),
            x_m=float(x_m),
            y_m=float(y_m),
        )
        self.markers.append(m)
        return m

    def delete_marker(self, marker_id: int) -> None:
        self.markers = [m for m in self.markers if m.id != marker_id]
        renumber(self.markers)

    def line_by_id(self, line_id: int) -> Line | None:
        return next((line for line in self.lines if line.id == line_id), None)

    def snap_or_split(
        self,
        x_m: float,
        y_m: float,
        tol_m: float,
        exclude: Line | None = None,
        end_margin_m: float = _SPLIT_END_MARGIN_M,
    ) -> tuple[float, float]:
        """Connect a wall end at ``(x_m, y_m)`` to existing geometry; return the point to use.

        Within ``tol_m`` of an existing **endpoint** -> snap to it exactly (reuse, no new point).
        Else on a line's **interior** -> split that line at the projection, creating a shared point.
        Else unchanged. ``exclude`` skips one line (the one being dragged) so it never snaps to self.
        """
        for line in self.lines:
            if line is exclude:
                continue
            for end in (0, 1):
                ex, ey = line.endpoint(end)
                if math.hypot(x_m - ex, y_m - ey) <= tol_m:
                    return ex, ey
        for line in self.lines:
            if line is exclude:
                continue
            (x0, y0), (x1, y1) = line.endpoint(0), line.endpoint(1)
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            t = ((x_m - x0) * dx + (y_m - y0) * dy) / (length * length)
            if t * length < end_margin_m or (1 - t) * length < end_margin_m:
                continue
            px, py = x0 + t * dx, y0 + t * dy
            if math.hypot(x_m - px, y_m - py) <= tol_m:
                self._split_line(line, t, px, py)
                return px, py
        return x_m, y_m

    def _split_line(self, line: Line, t: float, px: float, py: float) -> None:
        """Replace ``line`` with two: the first keeps ``line``'s id, the second gets a NEW id.

        A door on the split line follows by fraction (stays on the first half or moves to the new).
        """
        idx = self.lines.index(line)
        a = Line(id=line.id, x0_m=line.x0_m, y0_m=line.y0_m, x1_m=px, y1_m=py)
        b = Line(id=self._next_line_id(), x0_m=px, y0_m=py, x1_m=line.x1_m, y1_m=line.y1_m)
        for d in self.doors:
            if d.line_id != line.id:
                continue
            if d.t <= t:
                d.t = d.t / t if t > 0 else 0.0
            else:
                d.line_id, d.t = b.id, ((d.t - t) / (1 - t) if t < 1 else 1.0)
        self.lines[idx : idx + 1] = [a, b]

    def _split_line_at_point(self, line: Line, px: float, py: float) -> None:
        (x0, y0), (x1, y1) = line.endpoint(0), line.endpoint(1)
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        if length2 == 0:
            return
        self._split_line(line, ((px - x0) * dx + (py - y0) * dy) / length2, px, py)

    def _segment_with_crossings(self, p0, p1) -> list[Line]:
        """Add the wall ``p0-p1``, splitting every existing line it crosses at a shared point (and
        the new wall there too), so walls meeting at a point actually share it."""
        crossings = []
        for line in list(self.lines):
            c = segment_intersection(p0, p1, line.endpoint(0), line.endpoint(1))
            if c is not None:
                self._split_line_at_point(line, *c)
                crossings.append(c)
        ordered = [p0, *sorted(crossings, key=lambda c: math.hypot(c[0] - p0[0], c[1] - p0[1])), p1]
        made = []
        for a, b in zip(ordered, ordered[1:], strict=False):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6:
                made.append(self.add_line(*a, *b))
        return made

    def add_wall(
        self,
        x0_m: float,
        y0_m: float,
        x1_m: float,
        y1_m: float,
        tol_m: float,
        axis_snap_deg: float = 10.0,
    ) -> list[Line]:
        """Add one straight wall: snap/split both ends, axis-snap it, split any lines it crosses."""
        p0 = self.snap_or_split(x0_m, y0_m, tol_m)
        (_, p1_aligned) = axis_snap([p0, (x1_m, y1_m)], axis_snap_deg)
        p1 = self.snap_or_split(*p1_aligned, tol_m)
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < tol_m:
            return []  # degenerate -- a click, not a wall
        return self._segment_with_crossings(p0, p1)

    def add_freehand(
        self,
        points: list[tuple[float, float]],
        tol_m: float,
        eps_m: float,
        axis_snap_deg: float = 10.0,
    ) -> list[Line]:
        """Turn a freehand pencil path into walls **immediately** (the raw path is not kept).

        The path is Ramer-Douglas-Peucker-simplified and axis-snapped; its two open ends snap to
        existing geometry (reuse a point / split a line), its interior corners stay as shared points
        between consecutive walls, and every segment splits any existing line it crosses.
        """
        pts = axis_snap(lineify(points, eps_m), axis_snap_deg)
        if len(pts) < 2:
            return []
        pts[0] = self.snap_or_split(*pts[0], tol_m)
        pts[-1] = self.snap_or_split(*pts[-1], tol_m)
        made = []
        for a, b in zip(pts, pts[1:], strict=False):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6:
                made += self._segment_with_crossings(a, b)
        return made


def px_to_m(
    px: float, py: float, w_px: int, h_px: int, width_m: float, height_m: float
) -> tuple[float, float]:
    """Canvas pixel -> world metre. Pixel y is top-down; metre y is bottom-up (so y is flipped)."""
    return px / w_px * width_m, (1.0 - py / h_px) * height_m


def m_to_px(
    x_m: float, y_m: float, w_px: int, h_px: int, width_m: float, height_m: float
) -> tuple[float, float]:
    """World metre -> canvas pixel (inverse of :func:`px_to_m`)."""
    return x_m / width_m * w_px, (1.0 - y_m / height_m) * h_px


def clamp_to_room(x_m: float, y_m: float, width_m: float, height_m: float) -> tuple[float, float]:
    """Clamp a point into the room box -- a drawn or dragged point cannot leave the floorplan."""
    return min(max(x_m, 0.0), width_m), min(max(y_m, 0.0), height_m)


_HIT_TOL_PX = 10
_SNAP_TOL_PX = 24  # generous radius for reusing a nearby point / splitting a nearby line to connect
_DOOR_HOVER_TOL_PX = 20  # how close the cursor must hover to a wall line to preview a door
_STROKE_SAMPLE_PX = 4  # minimum pencil movement between sampled freehand points
_MIN_STROKE_POINTS = 3  # fewer sampled points than this is a click, not a drawing
_CLICK_TOL_PX = (
    5  # press+release within this radius is a click (two-click draw), not a freehand drag
)
_LINEIFY_EPS_FRAC = 0.015  # RDP tolerance as a fraction of the room's larger dimension


def _perp_dist(pt, a, b) -> float:
    """Distance from ``pt`` to the segment ``a``-``b``."""
    (px, py), (ax, ay), (bx, by) = pt, a, b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def lineify(points: list[tuple[float, float]], epsilon_m: float) -> list[tuple[float, float]]:
    """Collapse a dense freehand polyline into its straight sections (Ramer-Douglas-Peucker).

    Keeps a point only where the pencil path strays more than ``epsilon_m`` from the straight line
    between its kept neighbours; endpoints always survive. A polyline of <3 points is returned as-is.
    """
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dmax, idx = 0.0, None
        for k in range(i + 1, j):
            d = _perp_dist(points[k], points[i], points[j])
            if d > dmax:
                dmax, idx = d, k
        if dmax > epsilon_m:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(points, keep, strict=True) if k]


def door_geom(seg, t: float, width_m: float) -> tuple[float, float, float]:
    """A door's centre + angle on its wall: ``(cx, cy, angle_deg)`` for fraction ``t`` along ``seg``
    (``t`` clamped so the whole opening fits between the wall's ends)."""
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    t_m = min(max(t * length, width_m / 2), max(width_m / 2, length - width_m / 2))
    ux, uy = (dx / length, dy / length) if length else (1.0, 0.0)
    return ax + ux * t_m, ay + uy * t_m, math.degrees(math.atan2(dy, dx))


def door_interval(t: float, length: float, width_m: float) -> tuple[float, float]:
    """The ``[start, end]`` metres a door occupies along a wall of length ``length`` at fraction ``t``
    (centre clamped exactly as :func:`door_geom` does so the opening stays within the wall).

    **This clamp must agree with the bake.** ``roqsim_scenes.floorplan_geometry.assign_doors`` computes
    the same ``t_m`` when it cuts the wall, so if the two ever disagree the preview drawn here stops
    describing the world that gets built. The one intended difference is at the edges: a wall shorter
    than the door is a hard error there and merely a degenerate preview here, and two openings that
    overlap are merged by ``cut_openings`` but *refused* by :func:`door_fits` -- the window is
    deliberately the stricter of the two, so nothing the human can draw surprises the generator.
    """
    half = width_m / 2
    center = min(max(t * length, half), max(half, length - half))
    return center - half, center + half


def door_fits(existing_ts, length: float, t: float, width_m: float = _DOOR_WIDTH_M) -> bool:
    """Whether a door at fraction ``t`` clears every existing door (their ``t`` values) on the same
    wall -- i.e. their occupied intervals do not overlap. Openings may abut but not overlap."""
    a0, a1 = door_interval(t, length, width_m)
    for et in existing_ts:
        b0, b1 = door_interval(et, length, width_m)
        if a1 > b0 and b1 > a0:  # intervals overlap
            return False
    return True


def place_door(
    segments, x_m: float, y_m: float, tol_m: float, width_m: float = _DOOR_WIDTH_M
) -> tuple[int, float] | None:
    """Which wall a door would attach to at a cursor position: ``(segment index, t)`` or ``None``.

    Picks the nearest segment within ``tol_m`` that is at least a door wide; ``t`` (0..1) is the
    cursor's projection along it. Too-short segments are skipped.
    """
    best = None  # (dist, index, t)
    for idx, ((ax, ay), (bx, by)) in enumerate(segments):
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < width_m:
            continue
        t = max(0.0, min(1.0, ((x_m - ax) * dx + (y_m - ay) * dy) / (length * length)))
        dist = math.hypot(x_m - (ax + t * dx), y_m - (ay + t * dy))
        if dist <= tol_m and (best is None or dist < best[0]):
            best = (dist, idx, t)
    return None if best is None else (best[1], best[2])


def erase_line_at(lines: list[Line], x_m: float, y_m: float, tol_m: float) -> list[Line] | None:
    """The lines list without the line whose segment passes within ``tol_m`` of a click, or
    ``None`` when none is hit. Later-added lines win when they overlap. Ids are NOT renumbered."""
    for line in reversed(lines):
        if _perp_dist((x_m, y_m), line.endpoint(0), line.endpoint(1)) <= tol_m:
            return [x for x in lines if x is not line]
    return None


def hit_line_end(
    model: SketchModel, x_m: float, y_m: float, tol_m: float
) -> tuple[Line, int] | None:
    """The topmost line endpoint within ``tol_m`` of a point (metres): ``(line, end)`` or ``None``
    (``end`` is 0 or 1). Later-added lines win when endpoints overlap."""
    for line in reversed(model.lines):
        for end in (1, 0):
            ex, ey = line.endpoint(end)
            if (x_m - ex) ** 2 + (y_m - ey) ** 2 <= tol_m**2:
                return line, end
    return None


def load_sketch(initial: dict | None) -> tuple[SketchModel, str]:
    """Turn an ``initial`` sketch dict into ``(SketchModel, comment)``.

    Reads ``lines`` (ids preserved), ``doors``, ``markers`` (their comments = prop names), room
    names, and the object-placement ``description`` (floorplan-level) + per-room ``description``.
    The canvas is unbounded; there are no overall dimensions. The top-level ``comment`` is **not**
    loaded: that box is the human's feedback channel back to the agent, so it always starts empty
    rather than pre-filled with the agent's own note (which belongs in ``message``). The
    descriptions are the opposite -- persistent intent, so they *are* seeded from ``initial``.
    """
    initial = initial or {}
    model = SketchModel()
    model.description = initial.get("description", "") or ""
    for entry in initial.get("lines") or []:
        model.add_line(
            entry["x0_m"], entry["y0_m"], entry["x1_m"], entry["y1_m"], line_id=entry.get("id")
        )
    for entry in initial.get("doors") or []:
        model.add_door(entry["line_id"], entry["t"], entry.get("width_m", _DOOR_WIDTH_M))
    for entry in initial.get("markers") or []:
        m = model.add_marker(entry["x_m"], entry["y_m"], marker_id=entry.get("id"))
        m.comment = entry.get("comment", "")
        yaw = entry.get("yaw_deg")
        m.yaw_deg = float(yaw) if yaw is not None else None
    for entry in (
        initial.get("rooms") or []
    ):  # restore names + descriptions keyed by the wall-id set
        key = frozenset(entry["line_ids"])
        if entry.get("name"):
            model.room_names[key] = entry["name"]
        if entry.get("description"):
            model.room_descriptions[key] = entry["description"]
    return model, ""


def write_result(json_out: str | None, comment: str, sketch: SketchModel) -> dict:
    """Assemble the sketch result dict and, if ``json_out`` is given, write it there.

    Structured **rooms first, then lines** (no overall dimensions -- the canvas is unbounded and the
    generator sizes the floor from the walls): ``rooms`` are the closed loops the walls enclose
    (``line_ids`` + optional ``name``); ``lines`` are the walls (each independent, stable id); each
    door is attached to a line by ``line_id`` + ``t``; each marker is a prop point
    ``{id, x_m, y_m, comment, in_room}`` where ``in_room`` is the id of the room that contains it
    (computed, ``null`` if outside every room), plus ``yaw_deg`` **only when the human drew a heading**
    (degrees about +Z, 0 = +x, CCW) -- a headingless marker omits it and the prop is placed
    axis-aligned.
    """
    rooms = rooms_with_names(sketch.lines, sketch.room_names)

    def _in_room(mx: float, my: float):
        return next((r["id"] for r in rooms if point_in_polygon(mx, my, r["points"])), None)

    def _room_desc(line_ids):
        return sketch.room_descriptions.get(frozenset(line_ids), "").strip()

    result = {
        "comment": comment,
        # floorplan-level placement intent -- omitted when empty so old sketches stay unchanged
        **({"description": desc} if (desc := sketch.description.strip()) else {}),
        "rooms": [
            {
                "id": r["id"],
                "name": r["name"],
                "line_ids": r["line_ids"],
                **({"description": d} if (d := _room_desc(r["line_ids"])) else {}),
            }
            for r in rooms
        ],
        "lines": [
            {
                "id": line.id,
                "x0_m": round(line.x0_m, 2),
                "y0_m": round(line.y0_m, 2),
                "x1_m": round(line.x1_m, 2),
                "y1_m": round(line.y1_m, 2),
            }
            for line in sketch.lines
        ],
        "doors": [
            {"id": d.id, "line_id": d.line_id, "t": round(d.t, 2), "width_m": d.width_m}
            for d in sketch.doors
        ],
        "markers": [
            {
                "id": m.id,
                "x_m": round(m.x_m, 2),
                "y_m": round(m.y_m, 2),
                "comment": m.comment,
                "in_room": _in_room(m.x_m, m.y_m),
                # only present when the human drew a heading -- keeps headingless markers unchanged
                **({"yaw_deg": round(m.yaw_deg, 1)} if m.yaw_deg is not None else {}),
            }
            for m in sketch.markers
        ],
    }
    if json_out:
        Path(json_out).write_text(json.dumps(result), encoding="utf-8")
    return result


# --- GUI ------------------------------------------------------------------------------------------


def run_window(
    message: str = "",
    initial: dict | None = None,
    json_out: str | None = None,
    size: tuple[int, int] = _DEFAULT_SIZE,
    title: str = "",
) -> int:
    """Open the floorplan-sketch window and block until the human sends or closes it.

    Returns an exit code: 0 (sent), 2 (no display), 3 (closed without sending). On send the sketch
    JSON is written to ``json_out`` (when given). ``initial`` pre-seeds the window; ``title`` (larger
    font) and ``message`` head the panel.
    """
    # The same predicate as :func:`roqsim.viewer.has_display`, which the 3D window uses -- spelled out
    # here rather than imported because ``roqsim.viewer`` imports MuJoCo at module level, and this is
    # the one window that never needs it. A one-line env check is not worth that import.
    if not os.environ.get("DISPLAY"):
        print(
            "roqsim-scene-builder: no DISPLAY -- the floorplan-sketch window needs a graphical session.",
            flush=True,
        )
        return 2

    import tkinter as tk  # imported here so the module stays importable headless

    model, comment = load_sketch(initial) if initial else (SketchModel(), "")
    app = _SketchApp(tk, title, message, model, comment, json_out, size)
    app.root.mainloop()
    return app.exit_code


class _SketchApp:
    """The tkinter window: a metric top-view canvas on the left, the sketch panel on the right."""

    _BTN_IPADY = 4  # one button height for every button in the form

    _DEFAULT_VIEW_WIDTH_M = 10.0  # metres visible across the canvas width before any zoom

    def __init__(self, tk, title, message, model, comment, json_out, size):
        self.tk = tk
        self.json_out = json_out
        self.width, self.height = size
        self.model = model
        self.exit_code = 3  # closed-without-sending unless Send sets it
        self._draw_pts = None  # freehand points (metres) while the pencil is down
        self._draw_last_px = None  # last sampled canvas position of that pencil path
        self._press_px = (
            None  # canvas position where the left button went down (click-vs-drag test)
        )
        self._pending_start = (
            None  # first click of a two-click straight wall (metres), awaiting the end
        )
        self._move_ends = None  # (line, end) list sharing the point being dragged in move mode
        self._move_marker = None  # the Marker being dragged in move mode
        self._mark_drag = None  # (Marker, press_px) while dragging a heading out of a fresh marker
        self._marker_entries = {}  # marker id -> its comment Entry (for focus-on-place)
        self._undo = []  # (lines, doors) snapshots for ↶
        self._redo = []  # snapshots for ↷
        # View: metres map to pixels at 1 px/m (y-up) then scale (zoom) + offset (pan). The initial
        # framing is computed once the real canvas size is known (first <Configure>).
        self._scale = 1.0
        self._off_x = 0.0
        self._off_y = 0.0
        self._fitted = False  # has the initial fit-to-content run yet?
        self._pan_last = None  # last right-button position while panning
        self._INDEF_EPS_M = 0.1  # RDP tolerance for freehand on an indefinite canvas

        root = tk.Tk()
        root.title("roqsim floorplan sketch")
        root.configure(bg=BG)
        root.geometry(f"900x{self.height + 24}")  # start at a sensible width; resize keeps the 2:1
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root = root
        enable_edit_shortcuts(root)  # Ctrl+A select-all (Ctrl+C/V/X are Tk defaults) in text fields

        self.mode_var = tk.StringVar(
            value="draw"
        )  # draw | move | door | mark | delete (one at a time)
        self._build(title, message, comment)
        self._redraw()

    def _panel_text(self, panel, text: str, font, pady) -> None:
        """A left-aligned panel label that wraps to the panel's width (matching the comment box, which
        fills ``x`` at ``padx=12``): its ``wraplength`` tracks the label's real width on resize."""
        lbl = self.tk.Label(
            panel, text=text, bg=PANEL, fg=FG, justify="left", anchor="w", font=font
        )
        lbl.pack(fill="x", padx=12, pady=pady)
        lbl.bind("<Configure>", lambda e: e.widget.config(wraplength=e.width))

    # -- layout --
    def _build(self, title: str, message: str, comment: str) -> None:
        tk = self.tk

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        # Drawing : form = 2 : 1, total ~self.width. A uniform group pins the ratio; the form's
        # widgets are all kept narrow (below) so they never drive the columns wider.
        self._panel_w = self.width // 3
        body.grid_columnconfigure(0, weight=2, uniform="split")
        body.grid_columnconfigure(1, weight=1, uniform="split")
        body.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            body, width=self.width * 2 // 3, height=self.height, bg="#141414", highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)  # track the real canvas size
        self.canvas.bind("<ButtonPress-1>", self._on_press)  # start drawing a wall
        self.canvas.bind("<B1-Motion>", self._on_motion)  # sample the freehand pencil
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_hover)  # door-placement preview
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)  # right-drag pans the view
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_wheel)  # zoom (Windows / macOS)
        self.canvas.bind("<Button-4>", self._on_wheel)  # zoom in (Linux)
        self.canvas.bind("<Button-5>", self._on_wheel)  # zoom out (Linux)

        panel = tk.Frame(body, bg=PANEL, width=self._panel_w)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        panel.grid_propagate(False)  # hold the 1/3 column width; clip content rather than grow

        # Fixed footer (packed to the bottom, always visible): description, comment, status, Send.
        # Packed FIRST so it claims its parcel before anything variable above it can eat the cavity.
        footer = tk.Frame(panel, bg=PANEL)
        footer.pack(side="bottom", fill="x")
        # Persistent scene description (object-placement intent) -- seeded from the model and sent
        # back, unlike the feedback Comment box below it which always opens empty.
        self.description = build_comment_box(
            tk,
            footer,
            height=3,
            label="Scene description (what is this space, for the object placer)",
        )
        self.description.insert("1.0", self.model.description)
        self.comment = build_comment_box(tk, footer)
        self.comment.insert("1.0", comment)
        self.status = tk.Label(
            footer, text="", bg=PANEL, fg=MUTED, anchor="w", font=("TkDefaultFont", 9)
        )
        self.status.pack(fill="x", padx=12, pady=(6, 0))
        tk.Button(
            footer,
            text="Send",
            command=self._submit,
            bg=SEND_BG,
            fg="#ffffff",
            relief="flat",
            activebackground=SEND_BG,
            activeforeground="#ffffff",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(fill="x", padx=8, pady=10, ipady=self._BTN_IPADY)

        # Everything above the footer scrolls as one region: the title and message (which the caller
        # sizes, and which used to push the tool row and lists down), the tool row, and the lists.
        inner, self._sync_panel = build_scrollable(tk, panel, padx=0)

        if title:
            self._panel_text(
                inner, title, ("TkDefaultFont", 15, "bold"), pady=(12, 2 if message else 6)
            )
        if message:
            self._panel_text(inner, message, ("TkDefaultFont", 10), pady=((2 if title else 12), 6))

        self._build_tool_row(inner)

        for title, attr in (
            ("Rooms", "room_frame"),
            ("Walls", "line_frame"),
            ("Doors", "door_frame"),
            ("Markers", "marker_frame"),
        ):
            tk.Label(inner, text=title, bg=PANEL, fg=MUTED, anchor="w").pack(
                fill="x", padx=12, pady=(8, 0)
            )
            frame = tk.Frame(inner, bg=PANEL)
            frame.pack(fill="x", padx=12)
            setattr(self, attr, frame)
        self._rebuild_rows()

    def _build_tool_row(self, panel) -> None:
        # Exactly one mode at a time (default draw); ↶/↷ undo/redo the last edit -- all one row,
        # all narrow and the same height so they fit the 1/3-width form.
        tk = self.tk
        row = tk.Frame(panel, bg=PANEL)
        row.pack(fill="x", padx=8, pady=(8, 0))
        for text, value in (
            ("Draw", "draw"),
            ("Move", "move"),
            ("Door", "door"),
            ("Mark", "mark"),
            ("Del", "delete"),
        ):
            tk.Radiobutton(
                row,
                text=text,
                value=value,
                variable=self.mode_var,
                command=self._on_mode_change,
                indicatoron=False,
                bg="#2a2a2a",
                fg=FG,
                selectcolor=SEND_BG,
                width=1,
                activebackground="#333333",
                activeforeground=FG,
                relief="flat",
                font=("TkDefaultFont", 8),
                padx=0,
                bd=0,
            ).pack(side="left", expand=True, fill="x", padx=1, ipady=self._BTN_IPADY)
        for icon, cmd in (("↶", self._undo_edit), ("↷", self._redo_edit)):
            tk.Button(  # keep the default (white) focus border; trim internal padding to stay short
                row,
                text=icon,
                command=cmd,
                bg="#2a2a2a",
                fg=FG,
                width=1,
                activebackground="#333333",
                activeforeground=FG,
                relief="flat",
                font=("TkDefaultFont", 8),
                padx=0,
                pady=0,
                bd=0,
            ).pack(side="left", expand=True, fill="x", padx=1, ipady=self._BTN_IPADY)

    def _on_mode_change(self) -> None:
        self.canvas.delete("door_preview")  # clear a stale door hover when leaving door mode
        self.canvas.delete("wall_preview")  # and a pending two-click wall
        self._pending_start = None  # switching modes cancels a half-drawn wall
        self.status.config(text="")

    def _on_canvas_resize(self, event) -> None:
        # The canvas fills its 2/3 column; track its real pixel size. The first real size triggers the
        # initial framing (fit the canvas box, or auto-zoom to the drawn geometry).
        if event.width > 1 and event.height > 1:
            self.width, self.height = event.width, event.height
            if not self._fitted:
                self._fit_initial()
                self._fitted = True
            self._redraw()

    # -- undo/redo (snapshots of the whole model geometry) --
    def _snapshot(self):
        return copy.deepcopy(self.model.lines), copy.deepcopy(self.model.doors)

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _undo_edit(self) -> None:
        if not self._undo:
            self.status.config(text="Nothing to undo.")
            return
        self._redo.append(self._snapshot())
        self.model.lines, self.model.doors = self._undo.pop()
        self._after_history()

    def _redo_edit(self) -> None:
        if not self._redo:
            self.status.config(text="Nothing to redo.")
            return
        self._undo.append(self._snapshot())
        self.model.lines, self.model.doors = self._redo.pop()
        self._after_history()

    def _after_history(self) -> None:
        self.status.config(text="")
        self._rebuild_rows()
        self._redraw()

    # -- view transform: metres -> pixels at one uniform scale (y-up), then pan offset. The scale is
    # shared by x and y so the floorplan keeps its real aspect (a door is the same pixel length
    # whichever way its wall runs). The canvas is unbounded; framing lives entirely in scale/offset.
    def _to_px(self, x_m: float, y_m: float) -> tuple[float, float]:
        return self._off_x + self._scale * x_m, self._off_y - self._scale * y_m

    def _to_m(self, px: float, py: float) -> tuple[float, float]:
        return (px - self._off_x) / self._scale, (self._off_y - py) / self._scale

    def _px_tol_m(self, px_tol: float) -> float:
        return px_tol / self._scale

    def _fit_view(self, x0, y0, x1, y1, margin: float = 0.9) -> None:
        """Set scale + offset so the metre rectangle (x0,y0)-(x1,y1) is centred in the canvas."""
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        self._scale = margin * min(self.width / w, self.height / h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self._off_x = self.width / 2 - self._scale * cx
        self._off_y = self.height / 2 + self._scale * cy

    def _geometry_bbox(self):
        xs, ys = [], []
        for ln in self.model.lines:
            xs += [ln.x0_m, ln.x1_m]
            ys += [ln.y0_m, ln.y1_m]
        for m in self.model.markers:
            xs.append(m.x_m)
            ys.append(m.y_m)
        return (min(xs), min(ys), max(xs), max(ys)) if xs else None

    def _fit_initial(self) -> None:
        """Auto-zoom to the drawn geometry if any; else a default view (~10 m across the width)."""
        if (bb := self._geometry_bbox()) is not None:
            x0, y0, x1, y1 = bb
            self._fit_view(x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5)
        else:
            self._scale = self.width / self._DEFAULT_VIEW_WIDTH_M
            self._off_x, self._off_y = self.width / 2, self.height / 2  # origin at canvas centre

    def _has_content(self) -> bool:
        return bool(self.model.lines or self.model.doors or self.model.markers)

    # -- zoom / pan --
    def _on_wheel(self, event) -> None:
        up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        factor = 1.1 if up else 1 / 1.1
        self._off_x = event.x - factor * (event.x - self._off_x)
        self._off_y = event.y - factor * (event.y - self._off_y)
        self._scale *= factor
        self._redraw()

    def _on_pan_start(self, event) -> None:
        self._pan_last = (event.x, event.y) if self._has_content() else None

    def _on_pan_move(self, event) -> None:
        if self._pan_last is None:
            return
        self._off_x += event.x - self._pan_last[0]
        self._off_y += event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)
        self._redraw()

    def _on_pan_end(self, event) -> None:
        self._pan_last = None

    # -- left button: dispatched by mode (draw / move / door / mark / delete) --
    def _on_press(self, event) -> None:
        mode = self.mode_var.get()
        if mode == "delete":
            self._delete_at(event)
        elif mode == "door":
            self._place_door(event)
        elif mode == "mark":
            self._begin_marker(event)
        elif mode == "move":
            self._grab_point(event)
        else:  # draw: begin a stroke -- a drag becomes freehand, a click drives the two-click path
            self._draw_pts = [self._to_m(event.x, event.y)]
            self._draw_last_px = (event.x, event.y)
            self._press_px = (event.x, event.y)

    def _on_motion(self, event) -> None:
        if self._mark_drag is not None:  # mark mode: drag a heading out of the fresh marker
            self._drag_marker_heading(event, *self._mark_drag)
            return
        if self._move_marker is not None:  # move mode: drag a marker
            self._move_marker.x_m, self._move_marker.y_m = self._to_m(event.x, event.y)
            self._redraw()
            return
        if self._move_ends is not None:  # move mode: drag the shared point (all its endpoints)
            nx, ny = self._to_m(event.x, event.y)
            for line, end in self._move_ends:
                line.set_endpoint(end, nx, ny)
            self._redraw()
            return
        if self._draw_pts is not None:
            lx, ly = self._draw_last_px
            if (event.x - lx) ** 2 + (event.y - ly) ** 2 < _STROKE_SAMPLE_PX**2:
                return  # sample the pencil sparsely; sub-pixel jitter is noise, not shape
            self._draw_pts.append(self._to_m(event.x, event.y))
            self.canvas.create_line(
                lx, ly, event.x, event.y, fill="#8a8a8a", width=2, tags="wall_preview"
            )  # live pencil; discarded on release
            self._draw_last_px = (event.x, event.y)

    def _on_release(self, event) -> None:
        if self._mark_drag is not None:  # finished dragging the marker's heading
            self._finish_marker()
            return
        if self._move_marker is not None:
            self._move_marker = None
            self._rebuild_rows()
            self._redraw()
            return
        if self._move_ends is not None:
            self._join_moved_point()  # drop it onto a nearby point to connect
            self._move_ends = None
            self._rebuild_rows()
            self._redraw()
            return
        if self._draw_pts is not None:
            self.canvas.delete("wall_preview")  # forget the raw freehand -- only lines remain
            pts, self._draw_pts, self._draw_last_px = self._draw_pts, None, None
            px = self._press_px
            self._press_px = None
            moved = (
                px is not None
                and (event.x - px[0]) ** 2 + (event.y - px[1]) ** 2 > _CLICK_TOL_PX**2
            )
            if moved and len(pts) >= _MIN_STROKE_POINTS:  # a freehand drag
                self._pending_start = None
                self._push_undo()
                self.model.add_freehand(pts, self._px_tol_m(_SNAP_TOL_PX), self._INDEF_EPS_M)
                self._rebuild_rows()
            elif not moved:  # a click: two-click straight-wall drawing
                self._on_draw_click(event)
            self._redraw()

    def _on_draw_click(self, event) -> None:
        """Two-click drawing: the first click sets the wall's start, the second draws to the end."""
        p = self._to_m(event.x, event.y)
        if self._pending_start is None:
            self._pending_start = p
            self.status.config(text="Click again to finish the wall (or switch modes to cancel).")
            return
        start, self._pending_start = self._pending_start, None
        self.status.config(text="")
        self._push_undo()
        self.model.add_wall(start[0], start[1], p[0], p[1], self._px_tol_m(_SNAP_TOL_PX))
        self._rebuild_rows()

    # -- move mode: drag an existing point (all endpoints that share it move together), or a marker --
    def _grab_point(self, event) -> None:
        x_m, y_m = self._to_m(event.x, event.y)
        tol = self._px_tol_m(_SNAP_TOL_PX)
        hit = hit_line_end(self.model, x_m, y_m, tol)
        if hit is not None:
            line, end = hit
            cx, cy = line.endpoint(end)
            self._push_undo()
            self._move_ends = [
                (ln, e)
                for ln in self.model.lines
                for e in (0, 1)
                if math.hypot(ln.endpoint(e)[0] - cx, ln.endpoint(e)[1] - cy) < 1e-6
            ]
            return
        marker = self._marker_at(x_m, y_m, tol)
        if marker is not None:
            self._push_undo()
            self._move_marker = marker

    def _marker_at(self, x_m: float, y_m: float, tol_m: float) -> Marker | None:
        for m in reversed(self.model.markers):  # topmost wins
            if math.hypot(m.x_m - x_m, m.y_m - y_m) <= tol_m:
                return m
        return None

    # -- mark mode: drop a prop marker; a press-drag-release draws its heading, then type its comment --
    def _begin_marker(self, event) -> None:
        """Press in Mark mode: drop the marker at the press point. If the button is then dragged, the
        drag direction becomes the prop's heading (see :meth:`_drag_marker_heading`); a plain click
        (no drag) leaves it headingless."""
        self._push_undo()
        m = self.model.add_marker(*self._to_m(event.x, event.y))
        self._mark_drag = (m, (event.x, event.y))
        self._rebuild_rows()
        self._redraw()

    def _drag_marker_heading(self, event, marker: Marker, press_px) -> None:
        """While the button is held after dropping a marker, the drag direction sets its ``yaw_deg``
        (world frame: 0 = +x, CCW). Within the click tolerance there is no heading yet."""
        if (event.x - press_px[0]) ** 2 + (event.y - press_px[1]) ** 2 <= _CLICK_TOL_PX**2:
            marker.yaw_deg = None
        else:
            mx, my = self._to_m(event.x, event.y)
            marker.yaw_deg = round(math.degrees(math.atan2(my - marker.y_m, mx - marker.x_m)), 1)
        self._redraw()

    def _finish_marker(self) -> None:
        marker, _ = self._mark_drag
        self._mark_drag = None
        self._rebuild_rows()
        self._redraw()
        entry = self._marker_entries.get(marker.id)  # let the human type the prop name immediately
        if entry is not None:
            entry.focus_set()
            entry.icursor("end")

    def _delete_marker(self, marker_id: int) -> None:
        self._push_undo()
        self.model.delete_marker(marker_id)
        self._rebuild_rows()
        self._redraw()

    def _join_moved_point(self) -> None:
        """After a move, if the point landed on another point, merge them: snap all moved ends onto
        it (so lines that met at the dragged point now meet at the target), then prune the collapsed
        edge and any duplicate wall the merge produced."""
        if not self._move_ends:
            return
        cx, cy = self._move_ends[0][0].endpoint(self._move_ends[0][1])
        tol = self._px_tol_m(_SNAP_TOL_PX)
        moved = {(id(ln), e) for ln, e in self._move_ends}
        best = None
        for ln in self.model.lines:
            for e in (0, 1):
                if (id(ln), e) in moved:
                    continue
                p = ln.endpoint(e)
                d = math.hypot(cx - p[0], cy - p[1])
                if d <= tol and (best is None or d < best[0]):
                    best = (d, p)
        if best is not None:
            for ln, e in self._move_ends:
                ln.set_endpoint(e, *best[1])
            self.model.prune_lines()  # remove the point's collapsed/duplicate walls

    # -- doors --
    def _line_segments(self):
        return [(line.endpoint(0), line.endpoint(1)) for line in self.model.lines]

    def _door_at(self, event):
        x_m, y_m = self._to_m(event.x, event.y)
        spot = place_door(self._line_segments(), x_m, y_m, self._px_tol_m(_DOOR_HOVER_TOL_PX))
        return None if spot is None else (self.model.lines[spot[0]], spot[1])

    def _door_center(self, door: Door):
        line = self.model.line_by_id(door.line_id)
        if line is None:
            return None
        return door_geom((line.endpoint(0), line.endpoint(1)), door.t, door.width_m)

    def _door_ok(self, line: Line, t: float) -> bool:
        """Whether a door at fraction ``t`` on ``line`` clears every other door already on it."""
        existing = [d.t for d in self.model.doors if d.line_id == line.id]
        return door_fits(existing, line.length_m, t, _DOOR_WIDTH_M)

    def _on_hover(self, event) -> None:
        mode = self.mode_var.get()
        if mode == "draw":  # rubber-band the pending two-click wall to the cursor
            self.canvas.delete("wall_preview")
            if self._pending_start is not None:
                sx, sy = self._to_px(*self._pending_start)
                self.canvas.create_line(
                    sx,
                    sy,
                    event.x,
                    event.y,
                    fill="#8a8a8a",
                    width=2,
                    dash=(4, 3),
                    tags="wall_preview",
                )
            return
        if mode != "door":
            return
        self.canvas.delete("door_preview")
        spot = self._door_at(event)
        if spot is not None:
            line, t = spot
            cx, cy, ang = door_geom((line.endpoint(0), line.endpoint(1)), t, _DOOR_WIDTH_M)
            # amber where the opening can go, red where it would overlap an existing door
            color = "#e8c56a" if self._door_ok(line, t) else "#e07070"
            self._draw_door(cx, cy, ang, _DOOR_WIDTH_M, color=color, tag="door_preview")

    def _place_door(self, event) -> None:
        spot = self._door_at(event)
        if spot is None:
            self.status.config(text="No wall line there (or the line is too short for a door).")
            return
        line, t = spot
        if not self._door_ok(line, t):
            self.status.config(text="Doors cannot overlap on a wall.")
            return
        self._push_undo()
        self.model.add_door(line.id, t)
        self.status.config(text="")
        self._rebuild_rows()
        self._redraw()

    def _draw_door(self, cx, cy, angle_deg, width_m, color, tag="") -> None:
        rad = math.radians(angle_deg)
        hx, hy = math.cos(rad) * width_m / 2, math.sin(rad) * width_m / 2
        x0, y0 = self._to_px(cx - hx, cy - hy)
        x1, y1 = self._to_px(cx + hx, cy + hy)
        self.canvas.create_line(x0, y0, x1, y1, fill=color, width=7, tags=tag)

    def _draw_doors(self) -> None:
        for d in self.model.doors:
            center = self._door_center(d)
            if center is None:
                continue
            cx, cy, ang = center
            self._draw_door(cx, cy, ang, d.width_m, color="#b07830")
            x, y = self._to_px(cx, cy)
            self.canvas.create_text(
                x + 6,
                y - 8,
                text=f"D{d.id}",
                fill="#b07830",
                anchor="sw",
                font=("TkDefaultFont", 9, "bold"),
            )

    def _delete_door(self, door_id: int) -> None:
        self._push_undo()
        self.model.delete_door(door_id)
        self._rebuild_rows()
        self._redraw()

    # -- delete: a click removes the marker / door / line under the cursor (in that order) --
    def _delete_at(self, event) -> None:
        x_m, y_m = self._to_m(event.x, event.y)
        tol = self._px_tol_m(_HIT_TOL_PX)
        marker = self._marker_at(x_m, y_m, tol)
        if marker is not None:
            self._push_undo()
            self.model.delete_marker(marker.id)
            self.status.config(text="")
            self._rebuild_rows()
            self._redraw()
            return
        for d in reversed(
            self.model.doors
        ):  # a door sits on top of its wall -> erase only the door
            center = self._door_center(d)
            if (
                center is not None
                and math.hypot(center[0] - x_m, center[1] - y_m) <= tol + d.width_m / 2
            ):
                self._push_undo()
                self.model.delete_door(d.id)
                self.status.config(text="")
                self._rebuild_rows()
                self._redraw()
                return
        lines = erase_line_at(self.model.lines, x_m, y_m, tol)
        if lines is None:
            self.status.config(text="No line there.")
            return
        gone = {line.id for line in self.model.lines} - {line.id for line in lines}
        self._push_undo()
        for line_id in gone:
            self.model.delete_line(line_id)  # cascades to that line's doors
        self.status.config(text="")
        self._rebuild_rows()
        self._redraw()

    # -- rendering --
    def _redraw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_rooms()
        self._draw_lines()
        self._draw_doors()
        self._draw_markers()
        self._draw_legend()

    def _draw_rooms(self) -> None:
        # A shy background fill + the room's name at its centroid, matching the Rooms list colours.
        for room in rooms_with_names(self.model.lines, self.model.room_names):
            pts = room["points"]
            flat = [c for p in pts for c in self._to_px(*p)]
            self.canvas.create_polygon(*flat, fill=shy_hex(room["id"]), outline="")
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            x, y = self._to_px(cx, cy)
            self.canvas.create_text(
                x, y, text=room["name"], fill="#9a9a9a", font=("TkDefaultFont", 9, "italic")
            )

    def _draw_grid(self) -> None:
        # A metre grid across whatever the view currently shows (coarsens when zoomed out).
        xa, xb = sorted((self._to_m(0, 0)[0], self._to_m(self.width, 0)[0]))
        ya, yb = sorted((self._to_m(0, 0)[1], self._to_m(0, self.height)[1]))
        step = 1.0
        while self._scale * step < 8:
            step *= 5
        i = math.ceil(xa / step) * step
        while i <= xb:
            self.canvas.create_line(*self._to_px(i, ya), *self._to_px(i, yb), fill="#242424")
            i += step
        j = math.ceil(ya / step) * step
        while j <= yb:
            self.canvas.create_line(*self._to_px(xa, j), *self._to_px(xb, j), fill="#242424")
            j += step

    def _draw_legend(self) -> None:
        # A scale bar bottom-right: a "nice" round length whose pixel width is ~80 px.
        raw = 80 / self._scale
        exp = math.floor(math.log10(raw)) if raw > 0 else 0
        f = raw / 10**exp
        nice = (1 if f < 1.5 else 2 if f < 3.5 else 5 if f < 7.5 else 10) * 10**exp
        px_len = nice * self._scale
        x1, y = self.width - 16, self.height - 16
        x0 = x1 - px_len
        self.canvas.create_line(x0, y, x1, y, fill="#c8c8c8", width=2)
        for x in (x0, x1):
            self.canvas.create_line(x, y - 4, x, y + 4, fill="#c8c8c8", width=2)
        self.canvas.create_text(
            (x0 + x1) / 2,
            y - 6,
            text=f"{nice:g} m",
            fill="#c8c8c8",
            anchor="s",
            font=("TkDefaultFont", 8),
        )

    def _draw_lines(self) -> None:
        for line in self.model.lines:
            hexc = rgba_hex(color_for(line.id))
            x0, y0 = self._to_px(*line.endpoint(0))
            x1, y1 = self._to_px(*line.endpoint(1))
            self.canvas.create_line(x0, y0, x1, y1, fill=hexc, width=2)
            self._dot(x0, y0, hexc)
            self._dot(x1, y1, hexc)
            self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 - 8,
                text=str(line.id),
                fill=hexc,
                font=("TkDefaultFont", 9, "bold"),
            )

    def _draw_markers(self) -> None:
        for m in self.model.markers:
            hexc = rgba_hex(color_for(m.id))
            x, y = self._to_px(m.x_m, m.y_m)
            if (
                m.yaw_deg is not None
            ):  # heading arrow: a fixed-length pixel arrow along the yaw (y-up)
                rad = math.radians(m.yaw_deg)
                self.canvas.create_line(
                    x,
                    y,
                    x + 26 * math.cos(rad),
                    y - 26 * math.sin(rad),
                    fill=hexc,
                    width=2,
                    arrow="last",
                    arrowshape=(9, 11, 4),
                )
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=hexc, outline="#141414")
            label = f"M{m.id}" + (f" {m.comment}" if m.comment else "")
            self.canvas.create_text(
                x + 8, y + 8, text=label, fill=hexc, anchor="w", font=("TkDefaultFont", 8, "bold")
            )

    def _dot(self, x: float, y: float, hexc: str, r: int = 5) -> None:
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=hexc, outline="#141414")

    # -- rows (rooms and lines share the same shape: colour chip on the left, then content) --
    def _rebuild_rows(self) -> None:
        self._rebuild_room_rows()
        self._rebuild_line_rows()
        build_point_rows(
            self.tk, self.door_frame, self.model.doors, self._delete_door, with_comment=False
        )
        # markers carry an editable comment (the prop name); keep the entries for focus-on-place.
        self._marker_entries = build_point_rows(
            self.tk, self.marker_frame, self.model.markers, self._delete_marker
        )
        # Rows packed in code produce no <Configure> on the scroll canvas, so re-measure explicitly
        # or the scrollbar would not appear until something else resized the window.
        sync = getattr(self, "_sync_panel", None)
        if sync is not None:
            self.marker_frame.after_idle(sync)

    def _rebuild_room_rows(self) -> None:
        # A colour chip with the id + a plain (background-less) name entry on the first line, then a
        # boxed description entry (object-placement intent) below it. The chip hue matches the room's
        # shy fill on the canvas.
        tk = self.tk
        for child in self.room_frame.winfo_children():
            child.destroy()
        for room in rooms_with_names(self.model.lines, self.model.room_names):
            key = frozenset(room["line_ids"])
            row = tk.Frame(self.room_frame, bg=PANEL)
            row.pack(fill="x", pady=1)
            head = tk.Frame(row, bg=PANEL)
            head.pack(fill="x")
            # chip fill = the room's canvas fill (shy_hex), so list and drawing read as the same colour;
            # light text since that fill is a faint dark tint.
            tk.Label(
                head,
                text=f" {room['id']} ",
                bg=shy_hex(room["id"]),
                fg=FG,
                font=("TkDefaultFont", 9, "bold"),
            ).pack(side="left")
            name_entry = tk.Entry(
                head, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", highlightthickness=0
            )
            name_entry.insert(0, room["name"])  # default "room N"
            name_entry.pack(side="left", fill="x", expand=True, ipady=1)
            name_entry.bind(
                "<KeyRelease>", lambda e, k=key, ent=name_entry: self._set_room_name(k, ent.get())
            )
            # per-room description: what belongs here, for the object placer (optional).
            desc_entry = tk.Entry(
                row,
                bg=ENTRY_BG,
                fg=FG,
                insertbackground=FG,
                relief="flat",
                highlightthickness=1,
                highlightbackground=BORDER,
                font=("TkDefaultFont", 8),
            )
            desc_entry.insert(0, self.model.room_descriptions.get(key, ""))
            desc_entry.pack(fill="x", pady=(2, 0))
            desc_entry.bind(
                "<KeyRelease>",
                lambda e, k=key, ent=desc_entry: self._set_room_description(k, ent.get()),
            )

    def _rebuild_line_rows(self) -> None:
        tk = self.tk
        for child in self.line_frame.winfo_children():
            child.destroy()
        for line in self.model.lines:
            hexc = rgba_hex(color_for(line.id))
            row = tk.Frame(self.line_frame, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(
                row, text=f" {line.id} ", bg=hexc, fg="#1e1e1e", font=("TkDefaultFont", 9, "bold")
            ).pack(side="left")
            tk.Button(
                row,
                text="✕",
                command=lambda i=line.id: self._delete_line(i),
                bg=PANEL,
                fg="#ff8888",
                activebackground=PANEL,
                activeforeground="#ffaaaa",
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=2,
                pady=0,
                font=("TkDefaultFont", 9),
            ).pack(side="right")
            # the wall's length in metres, right after its id.
            tk.Label(
                row,
                text=f"· {line.length_m:.2f} m",
                bg=PANEL,
                fg=MUTED,
                anchor="w",
                font=("TkDefaultFont", 8),
            ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    def _set_room_name(self, key, name: str) -> None:
        if name.strip():
            self.model.room_names[key] = name
        else:
            self.model.room_names.pop(key, None)
        self._redraw()  # reflect the new name at the room's centroid immediately

    def _set_room_description(self, key, text: str) -> None:
        # Not drawn on the canvas, so no redraw; just keep the intent keyed to the room's wall set.
        if text.strip():
            self.model.room_descriptions[key] = text
        else:
            self.model.room_descriptions.pop(key, None)

    def _delete_line(self, line_id: int) -> None:
        self._push_undo()
        self.model.delete_line(line_id)
        self._rebuild_rows()
        self._redraw()

    # -- submit / close --
    def _submit(self) -> None:
        self.model.description = self.description.get("1.0", "end-1c").strip()
        write_result(self.json_out, self.comment.get("1.0", "end-1c"), self.model)
        self.exit_code = 0
        self.root.destroy()

    def _on_close(self) -> None:
        self.exit_code = 3
        self.root.destroy()
