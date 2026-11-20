"""A way over position and heading for a body holding something, checked piece by piece in height
layers: what is checked is what the body will walk.

:class:`~roqsim_nav.planner.GridPlanner` plans for a disc, by position only. That is the right
question for a body that is round and empty-handed, and the wrong one for a body carrying a tote
in front of it or a profile along its side: the load sweeps far past the disc when the body turns,
and it is at a height of its own. This planner answers the second question, for any mover that
holds something -- nothing here knows a walker, a robot or a task.

**State.** ``(x, y, heading)``, keyed on the cells of a :data:`~roqsim_nav.footprint.RESOLUTION`
grid and :data:`~roqsim_nav.footprint.BINS` heading bins, plus whether the body has turned on the
spot and whether it has stepped back since its last move forward.

**Footprint and layers.** A :class:`~roqsim_nav.footprint.Footprint` -- pieces in the body frame,
each with its height band -- per heading bin. Each band has an obstacle raster of its own, built
from :func:`~roqsim_nav.obstacles.wall_polygons` over that band: the static walls, and the bodies
with degrees of freedom that stand still (a door's leaf as it stands, a part on a rack, a cart),
except what is held (``skip_roots``). A piece meets only its band: the load at 1.0 m meets a shelf
at that height and passes over a bench below it; the legs meet what stands on the floor. Movers --
mocap bodies, a walker, a navigated prop -- are not in it: they move, and a carry among them stops
or gives way as its navigator's avoidance says.

Where the body may stand at a bin is the band rasters correlated with that bin's cells, by FFT, over
the corridor's box; it is computed the first time the search reaches that bin. A check is then one
lookup.

**Moves**, each checked at every :data:`~roqsim_nav.footprint.RESOLUTION` of travel and every bin it
passes through:

======================  ===============================================================  ========
move                    what                                                             cost
======================  ===============================================================  ========
arc forward             :data:`STEP` m, the heading changing by 0 or one bin             time + :data:`ARC_COST` a bin
turn on the spot        to any heading it sweeps clear, as one move; not twice running   time + :data:`TURN_COST`
step straight back      steps of :data:`BACK_STEP` up to ``back_max``; once between moves forward  2 x time + :data:`BACK_COST`
turn past half a turn   only with ``long_way``                                           + :data:`LONG_COST`
shot onto the goal      within :data:`SHOT` m: turn to it, straight to it, turn to its heading  as its parts
near something          a sample within :data:`NEAR_CELLS` cells of what it would meet   + :data:`NEAR_COST`
======================  ===============================================================  ========

Time is at :data:`SPEED` forward and :data:`TURN_RATE` turning. Which side a half turn goes round on
is the search's: the side that is clear, and of two, the one with more room.

**Corridor.** The search keeps to the disc's way (a ``GridPlanner`` at :data:`DISC_RADIUS`), widened
by the footprint's reach and :data:`WIDEN` m; where nothing is found in it, once more at twice that,
and the way says so (``widened``). Without a disc's way there is no way.

**Where no way exists** it raises :class:`NoWay`, naming the farthest it got along the disc's way,
how wide the way is there and between what, and what the footprint meets there at every heading.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .footprint import Footprint
from .footprint import _grown as _grown_poly
from .obstacles import wall_polygons
from .occupancy import OccupancyGrid
from .planner import GridPlanner

#: The length of an arc forward (m).
STEP = 0.1
#: How far one step back goes (m); a step back is a multiple of it, up to ``back_max``.
BACK_STEP = 0.05
#: The pace the costs are reckoned in (m/s forward, rad/s turning on the spot).
SPEED = 0.8
TURN_RATE = 1.0
#: Seconds a move costs on top of its time: an arc for each bin it turns, a turn on the spot (to
#: stop and set off again), a step back (twice its time as well), a turn past half a turn, and a
#: sample within NEAR_CELLS cells of what it would meet.
ARC_COST = 0.05
TURN_COST = 0.5
BACK_COST = 1.0
LONG_COST = 2.0
NEAR_COST = 0.1
NEAR_CELLS = 2
#: Within SHOT of the goal (m) the search tries the shot onto it. Within GOAL_ZONE of the goal and
#: ZONE of the start it checks a move the raster says is blocked exactly, across its sweep: where a
#: body stands close to what it works at, the raster's slack would keep it from standing there.
SHOT = 1.5
ZONE = 0.15
GOAL_ZONE = 0.6
#: How far any point of the footprint moves between two samples of an exact check of a move (m):
#: each sample is checked with half of it to spare, so every point the move sweeps is checked.
SWEEP_STEP = 0.01
#: How near the goal a state must come (m), and within half a bin of its heading.
ARRIVE = 0.03
#: The disc the corridor's way is planned for (m), and how much wider than the footprint's reach the
#: corridor is (m) before it widens.
DISC_RADIUS = 0.15
WIDEN = 0.5
#: The cell (m) and the turn (rad) on which two states count as one: the search's closed set.
KEY_CELL = 0.1
KEY_TURN = math.radians(10)
#: The most states a search expands before it gives up.
MAX_STATES = 60_000
#: How many states a search expands between the points where :func:`plan_way_steps` hands back,
#: so a caller on the physics thread steps the simulation on between them.
SLICE_STATES = 100
#: How many heading bins' rasters it builds between those points, before its search.
RASTERS_PER_STEP = 4

_TURNED, _BACKED = 1, 2


class NoWay(Exception):
    """No way exists for the footprint: the message says where it got to and what is in the way."""


@dataclass
class Segment:
    """One part of a :class:`Way`: ``walk`` (``poses`` walked through, facing where it walks),
    ``turn`` (on the spot by ``angle`` rad, signed, left positive) or ``back`` (straight back by
    ``distance`` m, keeping the heading)."""

    kind: str
    poses: list
    angle: float = 0.0
    distance: float = 0.0


@dataclass
class Way:
    """A planned way: its segments, and what it cost to find."""

    segments: list
    seconds: float
    expanded: int
    margin: float
    slack: float
    widened: bool = False
    poses: list = field(default_factory=list)

    @property
    def length(self) -> float:
        n = 0.0
        for s in self.segments:
            if s.kind == "walk":
                p = np.asarray([q[:2] for q in s.poses])
                n += (
                    float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0
                )
            elif s.kind == "back":
                n += s.distance
        return n

    @property
    def reverses(self) -> int:
        return sum(1 for s in self.segments if s.kind == "back")

    @property
    def long_turns(self) -> int:
        return sum(1 for s in self.segments if s.kind == "turn" and abs(s.angle) > math.pi + 1e-6)

    def summary(self) -> dict:
        """The numbers a move reports of its way."""
        return {
            "seconds": round(self.seconds, 3),
            "expanded": int(self.expanded),
            "length": round(self.length, 3),
            "reverses": self.reverses,
            "long_turns": self.long_turns,
            "margin": round(self.margin, 3),
            "widened": bool(self.widened),
        }


# -- obstacles in layers --------------------------------------------------------------------------
def standing_roots(model, data, skip_roots=()) -> set:
    """The weld roots of the bodies with degrees of freedom that are not mocap and not in
    ``skip_roots`` -- what :func:`wall_polygons` counts as a wall while it stands still."""
    skip = {int(r) for r in skip_roots}
    out = set()
    for b in range(1, model.nbody):
        root = int(model.body_weldid[b])
        if root != 0 and model.body_mocapid[b] < 0 and root not in skip:
            out.add(root)
    return out


def _fill(raster, polygon, origin, res, within):
    """Mark the cells of ``raster`` (indexed [i(x), j(y)]) whose centre is within ``within`` of the
    convex ``polygon``."""
    poly = np.asarray(polygon, float)
    lo = np.floor((poly.min(0) - within - origin) / res).astype(int)
    hi = np.floor((poly.max(0) + within - origin) / res).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(raster.shape) - 1)
    if np.any(hi < lo):
        return
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    cx = origin[0] + (ii + 0.5) * res
    cy = origin[1] + (jj + 0.5) * res
    dmax = np.full(ii.shape, -np.inf)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        e = b - a
        n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)
        dmax = np.maximum(dmax, (cx - a[0]) * n[0] + (cy - a[1]) * n[1])
    raster[ii, jj] |= dmax <= within


class Layers:
    """One obstacle raster per height band, on one grid: ``occ[band][i, j]`` (i along x, j along y)
    is true where something in that band touches the cell; ``polygons[band]`` are its footprints
    with the body each belongs to."""

    def __init__(self, model, data, bands, bounds, res, skip_roots=()):
        self.res = float(res)
        minx, miny, maxx, maxy = bounds
        self.origin = np.array([minx, miny], float)
        self.shape = (int(math.ceil((maxx - minx) / res)), int(math.ceil((maxy - miny) / res)))
        resting = standing_roots(model, data, skip_roots)
        self.occ, self.polygons = {}, {}
        for band in bands:
            polys = wall_polygons(model, data, band[0], band[1], resting_roots=resting, labels=True)
            raster = np.zeros(self.shape, bool)
            for poly, _name in polys:
                _fill(raster, poly, self.origin, self.res, 0.5 * math.sqrt(2.0) * self.res)
            self.occ[band] = raster
            self.polygons[band] = [(np.asarray(p, float), n) for p, n in polys]
        self.union = np.zeros(self.shape, bool)
        for raster in self.occ.values():
            self.union |= raster

    def cell(self, x, y):
        return int(math.floor((x - self.origin[0]) / self.res)), int(
            math.floor((y - self.origin[1]) / self.res)
        )

    def centre(self, i, j):
        return self.origin[0] + (i + 0.5) * self.res, self.origin[1] + (j + 0.5) * self.res


def _bounds(model, data, points, reach):
    """A box covering the world's footprints and ``points``, padded by ``reach`` and a metre."""
    pts = [np.asarray(p[:2], float) for p in points]
    polys = wall_polygons(model, data, 0.0, 10.0)
    for p in polys:
        pts.extend(np.asarray(p, float))
    arr = np.asarray(pts)
    pad = reach + 1.0
    return (*(arr.min(0) - pad), *(arr.max(0) + pad))


# -- the configuration space ----------------------------------------------------------------------
class _CSpace:
    """Where the footprint's root may stand, per heading bin, over a box of the grid: the union
    over bands of the band's raster correlated with the footprint's cells at that bin."""

    def __init__(self, layers: Layers, fp: Footprint, box, corridor):
        self.L, self.fp = layers, fp
        self.i0, self.j0, self.i1, self.j1 = box
        self.corridor = corridor[self.i0 : self.i1, self.j0 : self.j1]
        self.pad = int(math.ceil(fp.radius / layers.res)) + 3
        p = self.pad
        h, w = self.i1 - self.i0 + 2 * p, self.j1 - self.j0 + 2 * p
        self.size = (1 << (h + 2 * p).bit_length(), 1 << (w + 2 * p).bit_length())
        self._occ_f = {}
        for band in fp.bands:
            region = np.zeros((h, w))
            src = layers.occ[band]
            a0, b0 = self.i0 - p, self.j0 - p
            sa0, sb0 = max(a0, 0), max(b0, 0)
            sa1, sb1 = min(a0 + h, src.shape[0]), min(b0 + w, src.shape[1])
            if sa1 > sa0 and sb1 > sb0:
                region[sa0 - a0 : sa1 - a0, sb0 - b0 : sb1 - b0] = src[sa0:sa1, sb0:sb1]
            self._occ_f[band] = np.fft.rfft2(region, self.size) if region.any() else None
        self._blocked = {}
        self._near = {}
        self._setup_lookup()

    def blocked(self, k: int) -> np.ndarray:
        if k not in self._blocked:
            p = self.pad
            h, w = self.i1 - self.i0, self.j1 - self.j0
            out = ~self.corridor.copy()
            for band, cells in self.fp.cells(k).items():
                occ_f = self._occ_f.get(band)
                if occ_f is None or len(cells) == 0:
                    continue
                # blocked[i, j] = any occ[i + di, j + dj]: a correlation, as a convolution with the
                # cells flipped, whose result is read at an offset of the cells' upper corner
                a1, b1 = int(cells[:, 0].max()), int(cells[:, 1].max())
                a0, b0 = int(cells[:, 0].min()), int(cells[:, 1].min())
                if -a0 > p or a1 > p or -b0 > p or b1 > p:
                    raise ValueError("a footprint's cells reach past the box's pad")
                kern = np.zeros((a1 - a0 + 1, b1 - b0 + 1))
                kern[a1 - cells[:, 0], b1 - cells[:, 1]] = 1.0
                conv = np.fft.irfft2(occ_f * np.fft.rfft2(kern, self.size), self.size)
                # region index of box cell (0, 0) is (p, p); conv index of (i + di) is (i + a1)
                out |= conv[p + a1 : p + a1 + h, p + b1 : p + b1 + w] > 0.5
            self._blocked[k] = out
        return self._blocked[k]

    def near(self, k: int) -> np.ndarray:
        if k not in self._near:
            b = self.blocked(k).copy()
            for _ in range(NEAR_CELLS):
                g = b.copy()
                g[1:, :] |= b[:-1, :]
                g[:-1, :] |= b[1:, :]
                g[:, 1:] |= b[:, :-1]
                g[:, :-1] |= b[:, 1:]
                b = g
            self._near[k] = b
        return self._near[k]

    def _setup_lookup(self):
        self._ox, self._oy = float(self.L.origin[0]), float(self.L.origin[1])
        self._inv = 1.0 / self.L.res
        self._h, self._w = self.i1 - self.i0, self.j1 - self.j0
        self._bins = self.fp.bins
        self._per = self.fp.bins / (2 * math.pi)
        self._b = [None] * self.fp.bins
        self._n = [None] * self.fp.bins
        self._last_free = None

    def free(self, x, y, th) -> bool:
        """Whether the root may stand at ``(x, y)`` facing ``th``: one lookup (the hot path)."""
        u = (x - self._ox) * self._inv
        v = (y - self._oy) * self._inv
        if u < 0.0 or v < 0.0:
            return False
        i = int(u) - self.i0
        j = int(v) - self.j0
        if not (0 <= i < self._h and 0 <= j < self._w):
            return False
        k = int(math.floor(th * self._per + 0.5)) % self._bins
        b = self._b[k]
        if b is None:
            b = self._b[k] = self.blocked(k)
        return not b[i, j]

    def is_near(self, x, y, th) -> bool:
        i = int((x - self._ox) * self._inv) - self.i0
        j = int((y - self._oy) * self._inv) - self.j0
        if not (0 <= i < self._h and 0 <= j < self._w):
            return True  # off the box: as near as can be
        k = int(math.floor(th * self._per + 0.5)) % self._bins
        b = self._n[k]
        if b is None:
            b = self._n[k] = self.near(k)
        return bool(b[i, j])

    def _exact_setup(self):
        grown = []
        for piece in self.fp.pieces:
            poly = _grown_poly(piece.polygon, self.fp.margin)
            grown.append(
                (
                    piece.z,
                    poly,
                    float(np.max(np.linalg.norm(poly, axis=1))),
                    piece.body,
                    _normals(poly),
                )
            )
        self._grown = grown
        # each piece's own centre (body frame) and the radius about it: a piece carried off the root
        # (a tote ahead of the body) meets far fewer obstacles' circles than its reach from the root
        self._centre = []
        for _band, poly, _r, _own, _axes in grown:
            mid = (poly.min(0) + poly.max(0)) / 2.0
            self._centre.append((mid, float(np.max(np.linalg.norm(poly - mid, axis=1)))))
        self._obst = {}
        for band, polys in self.L.polygons.items():
            out = []
            for q, name in polys:
                c = q.mean(0)
                axes = _normals(q)
                proj = q @ axes.T
                out.append(
                    (
                        q,
                        name,
                        c,
                        float(np.max(np.linalg.norm(q - c, axis=1))),
                        axes,
                        proj.min(0),
                        proj.max(0),
                        q.min(0),
                        q.max(0),
                    )
                )
            self._obst[band] = out
        self.zones = []
        self._memo = {}
        self._sweeps = {}
        self._turned = {}
        self._sure = {}
        self._inner = None

    def exact_free(self, x, y, th, allowed=()) -> bool:
        """Whether the footprint, grown by its margin, overlaps nothing at ``(x, y, th)``: the
        polygons themselves, without the raster's slack. The body's own pieces may overlap the
        bodies named in ``allowed``."""
        if not hasattr(self, "_grown"):
            self._exact_setup()
        # a search asks of one pose many times (once for each way it reached it): kept, to the
        # millimetre and the thousandth of a radian
        key = (round(x * 1000.0), round(y * 1000.0), round(th * 1000.0), frozenset(allowed))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        free = self._exact(x, y, th, allowed)
        self._memo[key] = free
        return free

    def _exact(self, x, y, th, allowed) -> bool:
        for i, (band, _poly, r, own, _axes) in enumerate(self._grown):
            near = self._near_piece(band, own, allowed, x, y, r)
            if near and self._hits(i, x, y, th, near, 0.0):
                return False
        return True

    def _near_piece(self, band, own, allowed, x, y, reach) -> list:
        """The obstacles of ``band`` whose bounding circle comes within ``reach`` of ``(x, y)``,
        less what the body's own pieces (``own``) may stand against: the cheap check before the
        polygons'."""
        return [
            o
            for o in self._obst.get(band, ())
            if (o[2][0] - x) ** 2 + (o[2][1] - y) ** 2 <= (o[3] + reach) ** 2
            and not (own and o[1] in allowed)
        ]

    def _turned_piece(self, i: int, th: float):
        """Piece ``i``'s grown polygon and its axes turned to ``th``, and its extent along them:
        kept per heading (to the thousandth of a radian), as a search turns to few of them."""
        key = (i, round(th * 1000.0))
        hit = self._turned.get(key)
        if hit is None:
            c, s_ = math.cos(th), math.sin(th)
            rot = np.array([[c, s_], [-s_, c]])
            poly = self._grown[i][1] @ rot
            axes = self._grown[i][4] @ rot
            proj = poly @ axes.T
            hit = self._turned[key] = (poly, axes, proj.min(0), proj.max(0))
        return hit

    def _hits(self, i: int, x, y, th, near, gap: float) -> bool:
        """Whether piece ``i`` at ``(x, y, th)`` comes within ``gap`` of one of ``near`` -- not
        separated by more than ``gap`` along any axis of either (a gap of 0: overlaps)."""
        poly, axes, pmin, pmax = self._turned_piece(i, th)
        posed = poly + (x, y)
        shift = axes @ (x, y)
        pmin, pmax = pmin + shift, pmax + shift
        for q, _name, _qc, _qr, qaxes, qmin, qmax, *_box in near:
            on_q = posed @ qaxes.T
            if np.any((on_q.max(0) + gap < qmin) | (qmax + gap < on_q.min(0))):
                continue
            on_p = q @ axes.T
            if np.any((on_p.max(0) + gap < pmin) | (pmax + gap < on_p.min(0))):
                continue
            return True
        return False

    def surely_blocked(self, x, y, th) -> bool:
        """Whether the footprint certainly overlaps something at ``(x, y, th)``: a cell wholly
        inside one of its pieces (wherever the root is in its cell, at any heading in the bin) is a
        cell wholly inside an obstacle of that piece's band. What the body may stand against in a
        zone (any zone's ``allowed``) is left out of the bands the body's pieces are in."""
        i = int((x - self._ox) * self._inv) - self.i0
        j = int((y - self._oy) * self._inv) - self.j0
        if not (0 <= i < self._h and 0 <= j < self._w):
            return False
        k = int(math.floor(th * self._per + 0.5)) % self._bins
        b = self._sure.get(k)
        if b is None:
            b = self._sure[k] = self._surely(k)
        return bool(b[i, j])

    def _surely(self, k: int) -> np.ndarray:
        L, fp = self.L, self.fp
        res = L.res
        if self._inner is None:
            allowed = set().union(*(z[3] for z in self.zones)) if self.zones else set()
            body_bands = {p.z for p in fp.pieces if p.body}
            self._inner = {}
            for band in fp.bands:
                raster = np.zeros(L.shape, bool)
                for poly, name in L.polygons.get(band, ()):
                    if band in body_bands and name in allowed:
                        continue
                    _fill(raster, poly, L.origin, res, -0.5 * math.sqrt(2.0) * res)
                self._inner[band] = raster
        dth = 2 * math.pi / fp.bins
        h, w = self.i1 - self.i0, self.j1 - self.j0
        out = np.zeros((h, w), bool)
        p = self.pad
        for band in fp.bands:
            polys = [pc.polygon for pc in fp.pieces if pc.z == band]
            cells = None
            for sub in (-0.5, 0.0, 0.5):
                th = (k + sub) * dth
                c, s_ = math.cos(th), math.sin(th)
                # cells wholly inside a piece wherever the root is in its cell: a cell's centre
                # a cell's diagonal inside it
                got = {
                    tuple(v)
                    for v in _cells_inside([q @ np.array([[c, s_], [-s_, c]]) for q in polys], res)
                }
                cells = got if cells is None else cells & got
            if not cells:
                continue
            cells = np.array(sorted(cells))
            src = self._inner[band]
            for di, dj in cells:
                a0, b0 = self.i0 + di, self.j0 + dj
                sa0, sb0 = max(a0, 0), max(b0, 0)
                sa1, sb1 = min(a0 + h, src.shape[0]), min(b0 + w, src.shape[1])
                if sa1 > sa0 and sb1 > sb0:
                    out[sa0 - a0 : sa1 - a0, sb0 - b0 : sb1 - b0] |= src[sa0:sa1, sb0:sb1]
        del p
        return out

    def ok(self, x, y, th) -> bool:
        """The search's check: the raster's, and where the raster says blocked within one of
        ``zones`` (round the start and the goal, where a body stands close to what it works at),
        the exact one."""
        if self.free(x, y, th):
            return True
        i = int((x - self._ox) * self._inv) - self.i0
        j = int((y - self._oy) * self._inv) - self.j0
        if not (0 <= i < self._h and 0 <= j < self._w) or x < self._ox or y < self._oy:
            return False
        # what it may stand against where zones overlap (a goal near the start): each zone's
        allowed = self._zone_of((x, y))
        if allowed is None or self.surely_blocked(x, y, th):
            return False
        return self.exact_free(x, y, th, frozenset(allowed))

    def move_ok(self, a, b) -> bool:
        """The search's check of a move from pose ``a`` to the next sample ``b`` of it. The
        raster's cells and bins cover what a move sweeps between samples; where the raster says
        blocked within a zone, the exact check covers it too, across the move's sweep
        (:meth:`sweep_free`), not at its ends alone -- a thin post between two samples' headings
        is met by the piece that sweeps across it."""
        # a move's start is most often the end of the move checked just before it
        last = self._last_free
        fa = last[1] if last is not None and last[0] == a else self.free(*a)
        fb = self.free(*b)
        self._last_free = (b, fb)
        if fa and fb:
            return True
        allowed = None
        for p, f in ((a, fa), (b, fb)):
            if f:
                continue
            i = int((p[0] - self._ox) * self._inv) - self.i0
            j = int((p[1] - self._oy) * self._inv) - self.j0
            if not (0 <= i < self._h and 0 <= j < self._w) or p[0] < self._ox or p[1] < self._oy:
                return False
            zone = self._zone_of(p)
            if zone is None or self.surely_blocked(*p):
                return False
            allowed = zone if allowed is None else allowed & zone
        # the cheap exact check first, of the ends (kept), then the sweep between them
        allowed = frozenset(allowed)
        if not (self.exact_free(*a, allowed) and self.exact_free(*b, allowed)):
            return False
        return self.sweep_free(a, b, allowed)

    def _zone_of(self, p):
        """The bodies the body may stand against at ``p`` (the zones' it is in), or ``None`` outside
        every zone."""
        out = None
        for zx, zy, zr, allowed in self.zones:
            if (p[0] - zx) ** 2 + (p[1] - zy) ** 2 <= zr * zr:
                out = set(allowed) if out is None else out | set(allowed)
        return out

    def sweep_free(self, a, b, allowed=frozenset()) -> bool:
        """Whether the footprint, grown by its margin, overlaps nothing as its root moves from pose
        ``a`` to pose ``b`` (straight, turning evenly by ``b``'s heading less ``a``'s, unwrapped).
        First cheaply: only the obstacles whose bounding circle comes within a piece's reach of the
        move are looked at, and a move near none is free. Then at samples between which no point of
        the footprint moves more than :data:`SWEEP_STEP`, each checked with half of it to spare,
        so what the move sweeps between them is checked too. The body's own pieces may overlap the
        bodies named in ``allowed``."""
        if not hasattr(self, "_grown"):
            self._exact_setup()
        allowed = frozenset(allowed)
        key = tuple(round(v * 1000.0) for v in (*a[:3], *b[:3])) + (allowed,)
        hit = self._sweeps.get(key)
        if hit is not None:
            return hit
        turn = b[2] - a[2]  # as the move turns: a turn past half a turn is not its short way
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        pieces, reach = [], 0.0
        for i, (band, _poly, r, own, _axes) in enumerate(self._grown):
            # round the piece's own centre, at both ends of the move: what it may come near
            mid, rho = self._centre[i]
            ca, cb = _turn(mid, a[2]), _turn(mid, b[2])
            ax_, ay_ = a[0] + ca[0], a[1] + ca[1]
            bx_, by_ = b[0] + cb[0], b[1] + cb[1]
            # the centre's path bows out of its chord by at most its reach times the turn
            bow = math.hypot(*mid) * min(abs(turn), math.pi) / 2.0
            spread = math.hypot(bx_ - ax_, by_ - ay_) / 2.0 + bow + SWEEP_STEP
            near = self._near_piece(
                band, own, allowed, (ax_ + bx_) / 2.0, (ay_ + by_) / 2.0, rho + spread
            )
            if near:
                pieces.append((i, near))
                reach = max(reach, r)
        free = True
        if pieces:
            n = max(1, int(math.ceil((span + reach * abs(turn)) / SWEEP_STEP)))
            k = np.arange(n + 1) / n
            xs, ys = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
            ths = a[2] + turn * k
            c, s_ = np.cos(ths)[:, None], np.sin(ths)[:, None]
            free = not any(
                self._sweep_hits(i, near, xs, ys, c, s_, SWEEP_STEP / 2.0) for i, near in pieces
            )
        self._sweeps[key] = free
        return free

    def _sweep_hits(self, i: int, near, xs, ys, c, s_, gap: float) -> bool:
        """Whether piece ``i`` comes within ``gap`` of one of ``near`` at any of the samples
        ``(xs, ys)`` turned by ``(c, s_)`` (cosines and sines, column vectors): the separating-axis
        check of :meth:`_hits`, every sample at once."""
        _band, poly, _r, _own, paxes = self._grown[i]
        # every sample's polygon and axes, (n, v, 2) and (n, a, 2)
        px, py = poly[:, 0][None, :], poly[:, 1][None, :]
        posed = np.stack([px * c - py * s_ + xs[:, None], px * s_ + py * c + ys[:, None]], -1)
        ax, ay = paxes[:, 0][None, :], paxes[:, 1][None, :]
        axes = np.stack([ax * c - ay * s_, ax * s_ + ay * c], -1)
        lo, hi = posed.min((0, 1)), posed.max((0, 1))
        own = None
        for q, _name, _qc, _qr, qaxes, qmin, qmax, qlo, qhi in near:
            # the cheap check first: the boxes round all its samples and round the obstacle
            if np.any(hi + gap < qlo) or np.any(qhi + gap < lo):
                continue
            if own is None:
                own = np.einsum("nvd,nad->nva", posed, axes)
                pmin, pmax = own.min(1), own.max(1)
            on_q = posed @ qaxes.T  # (n, v, b)
            apart = np.any((on_q.max(1) + gap < qmin) | (qmax + gap < on_q.min(1)), axis=1)
            on_p = np.einsum("qd,nad->nqa", q, axes)  # (n, q, a)
            apart |= np.any((on_p.max(1) + gap < pmin) | (pmax + gap < on_p.min(1)), axis=1)
            if not apart.all():
                return True
        return False

    def body_meets(self, x, y, th) -> set:
        """The bodies the mover's own pieces overlap at ``(x, y, th)``, exactly."""
        if not hasattr(self, "_grown"):
            self._exact_setup()
        c, s = math.cos(th), math.sin(th)
        rot = np.array([[c, s], [-s, c]])
        out = set()
        for band, poly, _r, own, _axes in self._grown:
            if not own:
                continue
            posed = poly @ rot + [x, y]
            for q, name in self.L.polygons.get(band, ()):
                if name not in out and _overlap(posed, q):
                    out.add(name)
        return out

    def meets(self, x, y, th) -> list:
        """The bodies the footprint at ``(x, y, th)`` overlaps, each once, by name."""
        names = []
        for piece in self.fp.pieces:
            poly = piece.posed(x, y, th)
            for q, name in self.L.polygons.get(piece.z, ()):
                if name not in names and _overlap(poly, q):
                    names.append(name)
        return names


def _turn(p, th: float):
    """The body-frame point ``p`` turned to heading ``th`` (as a posed footprint is)."""
    c, s = math.cos(th), math.sin(th)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _cells_inside(polygons, res: float) -> np.ndarray:
    """Cell offsets whose centre lies a cell's diagonal inside one of the convex ``polygons``:
    wholly inside it wherever the root is in its own cell."""
    allp = np.concatenate(polygons)
    lo = np.floor(allp.min(0) / res).astype(int)
    hi = np.ceil(allp.max(0) / res).astype(int)
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    centres = np.stack([ii.ravel() * res, jj.ravel() * res], 1)
    hit = np.zeros(len(centres), bool)
    for poly in polygons:
        dmax = np.full(len(centres), -np.inf)
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            e = b - a
            n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)
            dmax = np.maximum(dmax, (centres - a) @ n)
        hit |= dmax <= -math.sqrt(2.0) * res
    return np.stack([ii.ravel()[hit], jj.ravel()[hit]], 1)


def _normals(poly) -> np.ndarray:
    """A convex polygon's edge normals, one a row."""
    e = np.roll(poly, -1, axis=0) - poly
    return np.stack([e[:, 1], -e[:, 0]], 1)


def _overlap(a, b) -> bool:
    """Whether two convex polygons overlap (separating axes, every edge normal of both at once)."""
    axes = np.concatenate([_normals(a), _normals(b)])
    pa = a @ axes.T
    pb = b @ axes.T
    return not bool(np.any((pa.max(0) < pb.min(0)) | (pb.max(0) < pa.min(0))))


# -- the disc's way, the corridor and the heuristic -----------------------------------------------
def _coarse(raster, f):
    h, w = raster.shape
    H, W = (h + f - 1) // f, (w + f - 1) // f
    pad = np.zeros((H * f, W * f), bool)
    pad[:h, :w] = raster
    return pad.reshape(H, f, W, f).any(axis=(1, 3))


def _disc_way(layers: Layers, start, goal, f: int):
    """The disc's way at :data:`DISC_RADIUS` on the union of the bands, on a grid ``f`` times
    coarser: ``[(x, y), ...]`` from start to goal, or ``None``."""
    coarse = _coarse(layers.union, f)
    # OccupancyGrid rows run top first, y down; ours run along x
    grid = OccupancyGrid(coarse.T[::-1].copy(), layers.res * f, tuple(layers.origin))
    path = GridPlanner(grid, inflation_radius=DISC_RADIUS).plan(start[:2], goal[:2])
    return [tuple(start[:2]), *path] if path else None


def _corridor(layers: Layers, way, width: float) -> np.ndarray:
    mask = np.zeros(layers.shape, bool)
    for (x0, y0), (x1, y1) in zip(way[:-1], way[1:], strict=False):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (layers.res / 2)))
        t = np.linspace(0.0, 1.0, n)
        ii = np.floor((x0 + t * (x1 - x0) - layers.origin[0]) / layers.res).astype(int)
        jj = np.floor((y0 + t * (y1 - y0) - layers.origin[1]) / layers.res).astype(int)
        ok = (ii >= 0) & (ii < layers.shape[0]) & (jj >= 0) & (jj < layers.shape[1])
        mask[ii[ok], jj[ok]] = True
    r = int(math.ceil(width / layers.res))
    rr = np.arange(-r, r + 1)
    kern = (rr[:, None] ** 2 + rr[None, :] ** 2 <= r * r).astype(float)
    size = (1 << (mask.shape[0] + 2 * r).bit_length(), 1 << (mask.shape[1] + 2 * r).bit_length())
    conv = np.fft.irfft2(np.fft.rfft2(mask.astype(float), size) * np.fft.rfft2(kern, size), size)
    return conv[r : r + mask.shape[0], r : r + mask.shape[1]] > 0.5


def _cost_to_go(free: np.ndarray, goal_ij) -> np.ndarray:
    """Cells to the goal over 8-connected ``free`` cells (Dijkstra): the search's heuristic."""
    h, w = free.shape
    dist = np.full((h, w), np.inf)
    gi, gj = goal_ij
    if not (0 <= gi < h and 0 <= gj < w):
        return dist
    dist[gi, gj] = 0.0
    heap = [(0.0, gi, gj)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)] + [
        (a, b, math.sqrt(2.0)) for a in (1, -1) for b in (1, -1)
    ]
    while heap:
        dc, i, j = heapq.heappop(heap)
        if dc > dist[i, j]:
            continue
        for a, b, c in steps:
            p, q = i + a, j + b
            if 0 <= p < h and 0 <= q < w and free[p, q] and dc + c < dist[p, q]:
                dist[p, q] = dc + c
                heapq.heappush(heap, (dc + c, p, q))
    return dist


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# -- the search -----------------------------------------------------------------------------------
class _Search:
    def __init__(self, cs: _CSpace, h2d, f, layers, back_max, long_way, max_states):
        self.cs, self.h2d, self.f, self.L = cs, h2d, f, layers
        self.back_max, self.long_way = back_max, long_way
        self.max_states = max_states
        self.dth = 2 * math.pi / cs.fp.bins
        self.expanded = 0

    def h(self, x, y, th, goal):
        i, j = self.L.cell(x, y)
        ci, cj = i // self.f, j // self.f
        h = self.h2d
        v = h[ci, cj] if 0 <= ci < h.shape[0] and 0 <= cj < h.shape[1] else math.inf
        if not math.isfinite(v):
            v = math.hypot(x - goal[0], y - goal[1]) / SPEED + 10.0
        else:
            v = v * self.L.res * self.f / SPEED
        if goal[2] is not None:
            v = max(v, abs(_wrap(goal[2] - th)) / TURN_RATE)
        return v

    def turn_way(self, x, y, th0, th1):
        """The signed angle a turn on the spot from ``th0`` to ``th1`` takes if every bin it passes
        is clear -- the short way, else (``long_way``) the long way -- or ``None``."""
        short = _wrap(th1 - th0)
        ways = [short]
        if self.long_way and abs(short) > 1e-9:
            ways.append(short - math.copysign(2 * math.pi, short))
        for way in ways:
            n = max(1, int(math.ceil(abs(way) / (self.dth / 2))))
            at = [(x, y, th0 + way * s / n) for s in range(n + 1)]
            if self.cs.ok(*at[0]) and all(
                self.cs.move_ok(p, q) for p, q in zip(at[:-1], at[1:], strict=False)
            ):
                return way
        return None

    def straight_ok(self, x0, y0, x1, y1, th):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(d / self.L.res)))
        at = [(x0 + (x1 - x0) * s / n, y0 + (y1 - y0) * s / n, th) for s in range(n + 1)]
        return self.cs.ok(*at[0]) and all(
            self.cs.move_ok(p, q) for p, q in zip(at[:-1], at[1:], strict=False)
        )

    def shot(self, s, goal):
        """turn - straight - turn onto the goal from ``s``, every part checked; its moves or None."""
        x, y, th, flags = s
        moves = []
        d = math.hypot(goal[0] - x, goal[1] - y)
        if d > 1e-3:
            face = math.atan2(goal[1] - y, goal[0] - x)
            if abs(_wrap(face - th)) > 1e-6:
                if flags & _TURNED:
                    return None
                way = self.turn_way(x, y, th, face)
                if way is None:
                    return None
                moves.append(("turn", (x, y, th + way), way))
                th = th + way
            if not self.straight_ok(x, y, goal[0], goal[1], th):
                return None
            moves.append(("arc", (goal[0], goal[1], th), 0.0))
            x, y = goal[0], goal[1]
        if goal[2] is not None and abs(_wrap(goal[2] - th)) > 1e-6:
            if not moves and flags & _TURNED:
                return None  # two turns on the spot running are one, and checked as one
            way = self.turn_way(x, y, th, goal[2])
            if way is None:
                return None
            moves.append(("turn", (x, y, th + way), way))
        return moves

    def run(self, start, goal, flags=0):
        """A* from ``start`` ``(x, y, yaw)`` to ``goal`` ``(x, y, yaw or None)``: the moves, as
        ``[(kind, (x, y, th), angle), ...]``, or ``None`` and the state nearest the goal. ``flags``:
        the turn on the spot and the step back the way made since its last move forward, when it
        comes from a via point."""
        return _drain(self.steps(start, goal, flags))

    def steps(self, start, goal, flags=0):
        """:meth:`run`, handing back (yielding) every :data:`SLICE_STATES` states it expands; it
        returns what :meth:`run` does."""
        cs, dth = self.cs, self.dth
        fp = cs.fp
        s0 = (float(start[0]), float(start[1]), float(start[2]), int(flags))

        # states are kept apart on a coarser grid than the checks (Hybrid-A*'s closed set): a
        # state's pose stays where its moves took it, and is checked there
        ox, oy, inv = cs._ox, cs._oy, 1.0 / KEY_CELL
        bins = int(round(2 * math.pi / KEY_TURN))
        per = bins / (2 * math.pi)
        floor = math.floor

        def key(s):
            return (
                int(floor((s[0] - ox) * inv)),
                int(floor((s[1] - oy) * inv)),
                int(floor(s[2] * per + 0.5)) % bins,
                s[3],
            )

        # each state's way back is its own chain of nodes (parent node, pose, move): the closed set
        # keeps the latest pose of a cell, and a way joined through it to that pose would join a
        # move to a pose it was not made from
        root = (None, s0, "start", 0.0)
        g = {key(s0): 0.0}
        heap = [
            (self.h(*s0[:3], goal), 0.0, 0, root)
        ]  # (f, -g: deeper first among equals, n, node)
        best = (heap[0][0], s0)
        tried = set()
        n = 0
        while heap:
            if self.expanded >= self.max_states:
                break
            if self.expanded and self.expanded % SLICE_STATES == 0:
                yield
            _f, ngc, _, node = heapq.heappop(heap)
            s = node[1]
            gc = -ngc
            ks = key(s)
            if gc > g.get(ks, math.inf) + 1e-9:
                continue
            self.expanded += 1
            x, y, th, flags = s
            hv = self.h(x, y, th, goal)
            if hv < best[0]:
                best = (hv, s)
            d = math.hypot(goal[0] - x, goal[1] - y)
            if d <= ARRIVE and (goal[2] is None or abs(_wrap(goal[2] - th)) <= dth / 2):
                return self._moves(node), best[1]
            if d < SHOT and ks[:3] not in tried:
                tried.add(ks[:3])
                tail = self.shot(s, goal)
                if tail is not None:
                    return self._moves(node) + tail, best[1]
            succ = []
            # arcs forward
            for db in (0, 1, -1):
                ok, px, py, near = True, x, y, 0
                sub = 4
                prev = (x, y, th)
                for q in range(1, sub + 1):
                    tq = th + db * dth * (q - 0.5) / sub
                    px += STEP / sub * math.cos(tq)
                    py += STEP / sub * math.sin(tq)
                    cur = (px, py, th + db * dth * q / sub)
                    if not cs.move_ok(prev, cur):
                        ok = False
                        break
                    prev = cur
                if ok:
                    near = cs.is_near(px, py, th + db * dth)
                    cost = STEP / SPEED + ARC_COST * abs(db) + NEAR_COST * near
                    succ.append(((px, py, th + db * dth, 0), cost, "arc", 0.0))
            # a turn on the spot, to every heading it sweeps clear
            if not flags & _TURNED:
                k0 = fp.bin_of(th)
                base = _wrap(fp.heading(k0) - th)
                for sgn in (1, -1):
                    last = 0.0
                    for m in range(1, fp.bins):
                        way = base + sgn * m * dth
                        # a bin's cells cover its whole width, so its centre's check is the
                        # check of every heading between it and the last; an exact one covers
                        # the turn from the last (move_ok)
                        if not cs.move_ok((x, y, th + last), (x, y, th + way)):
                            break
                        last = way
                        if abs(way) > math.pi + 1e-6 and not self.long_way:
                            break
                        cost = abs(way) / TURN_RATE + TURN_COST
                        if abs(way) > math.pi + 1e-6:
                            cost += LONG_COST
                        cost += NEAR_COST * cs.is_near(x, y, th + way)
                        succ.append(((x, y, th + way, flags | _TURNED), cost, "turn", way))
            # a step straight back
            if not flags & _BACKED and self.back_max >= BACK_STEP - 1e-9:
                cth, sth = math.cos(th), math.sin(th)
                steps = int(math.floor(self.back_max / BACK_STEP + 1e-9))
                for m in range(1, steps + 1):
                    dist = m * BACK_STEP
                    p0 = (x - (dist - BACK_STEP) * cth, y - (dist - BACK_STEP) * sth, th)
                    mid = (x - (dist - BACK_STEP / 2) * cth, y - (dist - BACK_STEP / 2) * sth, th)
                    p1 = (x - dist * cth, y - dist * sth, th)
                    if not (cs.move_ok(p0, mid) and cs.move_ok(mid, p1)):
                        break
                    cost = 2 * dist / SPEED + BACK_COST
                    succ.append(
                        ((x - dist * cth, y - dist * sth, th, flags | _BACKED), cost, "back", dist)
                    )
            for s2, cost, kind, amount in succ:
                k2 = key(s2)
                ng = gc + cost
                if ng < g.get(k2, math.inf) - 1e-9:
                    g[k2] = ng
                    n += 1
                    heapq.heappush(
                        heap, (ng + self.h(*s2[:3], goal), -ng, n, (node, s2, kind, amount))
                    )
        return None, best[1]

    @staticmethod
    def _moves(node):
        out = []
        while node is not None:
            parent, s, kind, amount = node
            if kind != "start":
                out.append((kind, (s[0], s[1], s[2]), amount))
            node = parent
        out.reverse()
        return out


def _verify(cs: _CSpace, start, moves, what: str) -> None:
    """Every move of a way found, checked exactly across its sweep (:meth:`_CSpace.sweep_free`) in
    the samples the search took of it, with what the search let the body stand against there. The
    search's own checks already cover it; one that does not is a fault of the planner's, raised
    rather than walked."""
    dth = 2 * math.pi / cs.fp.bins
    res = cs.L.res
    pose = (float(start[0]), float(start[1]), float(start[2]))
    for kind, p, _amount in moves:
        p = (float(p[0]), float(p[1]), float(p[2]))
        d = math.hypot(p[0] - pose[0], p[1] - pose[1])
        if kind == "turn":
            n = max(1, int(math.ceil(abs(p[2] - pose[2]) / (dth / 2))))
            steps = [(pose[0], pose[1], pose[2] + (p[2] - pose[2]) * k / n) for k in range(n + 1)]
        elif kind == "arc" and d <= STEP + 1e-6 and abs(_wrap(p[2] - pose[2])) <= dth + 1e-6:
            # an arc of the search's, a quarter of it at a time as the search took it
            db = int(round(_wrap(p[2] - pose[2]) / dth))
            steps, x, y = [pose], pose[0], pose[1]
            for q in range(1, 5):
                tq = pose[2] + db * dth * (q - 0.5) / 4
                x, y = x + STEP / 4 * math.cos(tq), y + STEP / 4 * math.sin(tq)
                steps.append((x, y, pose[2] + db * dth * q / 4))
        else:
            # a straight shot or a step back, a cell at a time
            n = max(1, int(math.ceil(d / res)))
            steps = [
                (pose[0] + (p[0] - pose[0]) * k / n, pose[1] + (p[1] - pose[1]) * k / n, p[2])
                for k in range(n + 1)
            ]
        for a, b in zip(steps[:-1], steps[1:], strict=False):
            # where the raster says blocked, the body may stand against what the zones there
            # allow, as the search's move_ok lets it
            allowed = None
            for q in (a, b):
                if not cs.free(*q):
                    zone = cs._zone_of(q) or set()
                    allowed = zone if allowed is None else allowed & zone
            allowed = allowed or set()
            if not cs.sweep_free(a, b, allowed):
                names = [n for n in cs.meets(*b) if n not in allowed] or cs.meets(*a)
                raise RuntimeError(
                    f"planned a way {what} that meets {', '.join(names) or 'something'} between "
                    f"({a[0]:.2f}, {a[1]:.2f}, {math.degrees(a[2]):.0f} deg) and "
                    f"({b[0]:.2f}, {b[1]:.2f}, {math.degrees(b[2]):.0f} deg): a fault of the planner's"
                )
        pose = p


def _segments(start, moves) -> list:
    """The search's moves as segments: consecutive arcs one walk."""
    segs, pose = [], (float(start[0]), float(start[1]), float(start[2]))
    for kind, p, amount in moves:
        if kind == "arc":
            if segs and segs[-1].kind == "walk":
                segs[-1].poses.append(p)
            else:
                segs.append(Segment("walk", [pose, p]))
        elif kind == "turn":
            if abs(amount) > 1e-9:
                segs.append(Segment("turn", [pose, p], angle=float(amount)))
        elif kind == "back":
            segs.append(Segment("back", [pose, p], distance=float(amount)))
        pose = p
    return segs


def _width_at(layers: Layers, point, direction) -> tuple:
    """How wide the way is at ``point`` across ``direction`` (m), on the union of the bands, and
    the bodies on either side."""
    nx, ny = -direction[1], direction[0]
    out, names = 0.0, []
    for sgn in (1, -1):
        d, name = 0.0, "nothing"
        while d < 5.0:
            d += layers.res / 2
            x, y = point[0] + sgn * d * nx, point[1] + sgn * d * ny
            i, j = layers.cell(x, y)
            if not (0 <= i < layers.shape[0] and 0 <= j < layers.shape[1]):
                break
            if layers.union[i, j]:
                for band in layers.polygons:
                    for poly, n in layers.polygons[band]:
                        if _inside(poly, (x, y), layers.res):
                            name = n
                            break
                    if name != "nothing":
                        break
                break
        out += d
        names.append(name)
    return out, names


def _inside(poly, p, tol) -> bool:
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        e = b - a
        n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)
        if (p[0] - a[0]) * n[0] + (p[1] - a[1]) * n[1] > tol:
            return False
    return True


def _drain(steps):
    """Run a generator of steps to its end; what it returns."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def plan_way(model, data, footprint: Footprint, start, goal, **kw) -> Way:
    """:func:`plan_way_steps`, in one go."""
    return _drain(plan_way_steps(model, data, footprint, start, goal, **kw))


def plan_way_steps(
    model,
    data,
    footprint: Footprint,
    start,
    goal,
    *,
    via=(),
    back_max: float = 0.5,
    long_way: bool = False,
    skip_roots=(),
    what: str = "what it holds",
    to: str = "",
    dock=(),
    max_states: int = MAX_STATES,
):
    """A way for ``footprint`` from ``start`` ``(x, y, yaw)`` through the floor points ``via`` (any
    heading) to ``goal`` ``(x, y, yaw)`` (``yaw`` ``None``: any). ``skip_roots``: the weld roots of
    what is held and what rides in it, which are part of the footprint and not in the world.
    Raises :class:`NoWay` naming ``what`` and ``to`` where there is none. Call it with ``data``
    forwarded (positions current); it reads, never writes.

    Near the start and the goal a move the raster says is blocked is checked exactly, across its
    sweep and not at its samples alone, and two
    things are not in the way of the mover's own body pieces (:attr:`Piece.body`) there: what it
    already stands against at the start, and at the goal the bodies named in ``dock`` -- the target
    it goes to work at and what that stands on, which a worker stands against. What it holds
    meets them as anything else. Every way found is checked exactly so once more before it is
    returned; one that meets something is a fault of the planner's, raised as a ``RuntimeError``.

    A generator: it hands back (yields) every :data:`SLICE_STATES` states its search expands, and
    returns the :class:`Way`; :func:`plan_way` runs it in one go. The world is read before it
    first hands back, so a caller may step the simulation between its steps. The way's
    ``seconds`` are the time it ran, not the time it was handed back for."""
    ran, t0 = 0.0, time.perf_counter()
    res = footprint.resolution
    if not (math.isfinite(float(back_max)) and back_max >= 0.0):
        raise ValueError(f"back_max is a distance of 0 m or more, not {back_max!r}")
    goals = [(float(p[0]), float(p[1]), None) for p in via] + [
        (float(goal[0]), float(goal[1]), None if goal[2] is None else float(goal[2]))
    ]
    reach = footprint.radius
    bounds = _bounds(model, data, [start, *goals], reach)
    layers = Layers(model, data, footprint.bands, bounds, res, skip_roots)
    f = max(1, int(round(0.05 / res)))
    segments, expanded, widened, poses = [], 0, False, []
    # a turn on the spot and a step back since the last move forward, carried across a via point:
    # a leg that ends stepping back does not let the next one start with another
    flags = 0
    pose = (float(start[0]), float(start[1]), float(start[2]))
    limits = f"back_max {back_max:.2f} m, long_way {'true' if long_way else 'false'}"
    target = to or f"({goals[-1][0]:.2f}, {goals[-1][1]:.2f})"
    for gl in goals:
        way = _disc_way(layers, pose, gl, f)
        if way is None:
            raise NoWay(
                f"no way holding {what} to {target}: no route for a disc of "
                f"{DISC_RADIUS:.2f} m from ({pose[0]:.2f}, {pose[1]:.2f}) to "
                f"({gl[0]:.2f}, {gl[1]:.2f})"
            )
        moves, nearest = None, None
        for widen in (1.0, 2.0):
            if expanded >= max_states:
                break
            corridor = _corridor(layers, way, widen * (reach + WIDEN))
            ii, jj = np.nonzero(corridor)
            box = (int(ii.min()), int(jj.min()), int(ii.max()) + 1, int(jj.max()) + 1)
            cs = _CSpace(layers, footprint, box, corridor)
            cs._exact_setup()
            # what the body already stands against, it may go on standing against as it leaves;
            # at the goal, it may stand against what it goes to work at
            standing = cs.body_meets(*pose)
            at_goal = set(dock) if gl is goals[-1] else set()
            cs.zones = [(gl[0], gl[1], GOAL_ZONE, at_goal), (pose[0], pose[1], ZONE, standing)]
            if not cs.ok(*pose):
                names = cs.meets(*pose)
                raise NoWay(
                    f"no way holding {what} to {target}: where it stands at "
                    f"({pose[0]:.2f}, {pose[1]:.2f}) it already meets "
                    f"{', '.join(names) if names else 'something within a few centimetres'}"
                )
            if gl[2] is not None and not cs.exact_free(*gl, allowed=at_goal):
                names = [n for n in cs.meets(*gl) if n not in at_goal]
                raise NoWay(
                    f"no way holding {what} to {target}: where it would stand at "
                    f"({gl[0]:.2f}, {gl[1]:.2f}) facing {math.degrees(gl[2]):.0f} deg it meets "
                    f"{', '.join(names) if names else 'something'}"
                )
            # the heuristic: a disc's cost to go over what stands on the floor (the lowest band),
            # the goal's zone left free, as the body may stand against what is there
            floor = layers.occ[footprint.bands[0]]
            free = _coarse(corridor & ~floor, f)
            gi, gj = layers.cell(gl[0], gl[1])
            r = int(math.ceil(GOAL_ZONE / (layers.res * f)))
            ci, cj = gi // f, gj // f
            ii_, jj_ = np.ogrid[: free.shape[0], : free.shape[1]]
            free |= ((ii_ - ci) ** 2 + (jj_ - cj) ** 2 <= r * r) & _coarse(corridor, f)
            free[min(ci, free.shape[0] - 1), min(cj, free.shape[1] - 1)] = True
            h2d = _cost_to_go(free, (gi // f, gj // f))
            search = _Search(
                cs, h2d, f, layers, float(back_max), bool(long_way), max_states - expanded
            )
            # the heading bins' rasters first, a few at a time between hand-backs: a search's first
            # turn on the spot would ask for every one of them at once
            for k in range(footprint.bins):
                if cs._b[k] is None:
                    cs._b[k] = cs.blocked(k)
                if k % RASTERS_PER_STEP == RASTERS_PER_STEP - 1:
                    ran += time.perf_counter() - t0
                    yield
                    t0 = time.perf_counter()
            steps = search.steps(pose, gl, flags)
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    moves, best = done.value
                    break
                ran += time.perf_counter() - t0
                yield
                t0 = time.perf_counter()
            expanded += search.expanded
            if nearest is None:
                # where the search in the disc's own corridor got to is where a failure is
                # reported: a wider corridor may wander off round the far end of a wall
                nearest = best
            if moves is not None:
                widened = widened or widen > 1.0
                break
        if moves is None:
            x, y, th, _ = nearest
            k = int(np.argmin([math.hypot(px - x, py - y) for px, py in way]))
            a = way[max(k - 1, 0)]
            b = way[min(k + 1, len(way) - 1)]
            dvec = np.array([b[0] - a[0], b[1] - a[1]], float)
            if np.linalg.norm(dvec) < 1e-6:  # no way to speak of: across where it faces
                dvec = np.array([math.cos(th), math.sin(th)])
            dvec /= np.linalg.norm(dvec)
            width, sides = _width_at(layers, (x, y), dvec)
            met = []
            for kk in range(footprint.bins):
                tt = footprint.heading(kk)
                nx, ny = x + STEP * math.cos(tt), y + STEP * math.sin(tt)
                if not cs.free(nx, ny, tt):
                    for n in cs.meets(nx, ny, tt):
                        if n not in met:
                            met.append(n)
            capped = f" (it gave up after {expanded} states)" if expanded >= max_states else ""
            raise NoWay(
                f"no way holding {what} to {target}: it gets no further than ({x:.2f}, {y:.2f}), "
                f"where the way is {width:.2f} m wide between {sides[0]} and {sides[1]}; the body and "
                f"what it holds, reaching {reach:.2f} m from where it stands, meet "
                f"{', '.join(met) if met else 'what is round them'} at every heading there; "
                f"{limits}{capped}"
            )
        _verify(cs, pose, moves, f"holding {what} to {target}")
        segs = _segments(pose, moves)
        segments.extend(segs)
        poses.append(pose)
        for _k, p, _a in moves:
            poses.append(p)
        for kind, _p, _a in moves:
            flags = 0 if kind == "arc" else flags | (_TURNED if kind == "turn" else _BACKED)
        if moves:
            pose = moves[-1][1]
    return Way(
        segments,
        ran + time.perf_counter() - t0,
        expanded,
        footprint.margin,
        footprint.slack,
        widened,
        poses,
    )


def check_way(model, data, footprint: Footprint, way: Way, skip_roots=()) -> list:
    """Every pose along ``way`` (at every centimetre and every degree) whose exact footprint
    overlaps a body in its band: ``[(x, y, yaw, [names])]``. For tests and for a caller that wants
    to know a way's exact clearance rather than the search's conservative one."""
    bounds = _bounds(model, data, way.poses or [(0, 0)], footprint.radius)
    layers = Layers(model, data, footprint.bands, bounds, footprint.resolution, skip_roots)
    out = []
    for seg in way.segments:
        samples = []
        if seg.kind == "turn":
            x, y, th = seg.poses[0]
            n = max(1, int(abs(seg.angle) / math.radians(1)))
            samples = [(x, y, th + seg.angle * s / n) for s in range(n + 1)]
        else:
            for a, b in zip(seg.poses[:-1], seg.poses[1:], strict=False):
                n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.01))
                samples += [
                    (
                        a[0] + (b[0] - a[0]) * s / n,
                        a[1] + (b[1] - a[1]) * s / n,
                        a[2] + _wrap(b[2] - a[2]) * s / n,
                    )
                    for s in range(n + 1)
                ]
        for x, y, th in samples:
            names = []
            for piece in footprint.pieces:
                poly = piece.posed(x, y, th)
                for q, name in layers.polygons.get(piece.z, ()):
                    if name not in names and _overlap(poly, q):
                        names.append(name)
            if names:
                out.append((x, y, th, names))
    return out
