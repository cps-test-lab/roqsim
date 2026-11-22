"""The footprint of a body and what it holds: convex pieces in the body's frame, each with the height
band it occupies, rasterized per heading for :mod:`~roqsim_nav.lattice`.

A disc answers "can this body stand here" only for a body that is round and holds nothing. A worker
carrying a tote in front of it, or a profile along its side, is long in one direction and short in
another, and what it holds is at one height while its legs are at another: the tote passes over a
bench its legs must walk round, and meets a shelf the legs pass under. So the footprint here is a
set of **pieces**, each a convex polygon in the body frame (x forward, y left, origin at the root
the navigator moves) with the band ``[z_lo, z_hi]`` it occupies, and each piece is checked only
against what stands in its own band.

The caller builds the pieces -- from a body's collision geoms (:func:`geom_piece`), a held object's,
and anything else it knows the body carries (:func:`capsule_piece`, an arm). Nothing here knows
what a body is: a walker, a humanoid robot and a wheeled base with a fork are all pieces.

**Per heading, once.** :meth:`Footprint.cells` is a piece set's cells for one of :data:`BINS`
heading bins, swept over the bin's whole width and grown by enough that a root anywhere in a cell
is covered (below). It is computed the first time a bin is asked for and kept, so a footprint is
rasterized at most once per bin per carry.

**Conservative, and by how much.** A root anywhere in a cell, at any heading in a bin, is covered by
the cells of that bin for that cell: a piece's cells are those whose centre comes within ``sqrt(2)``
cells of the piece swept over the bin (half a cell's diagonal for where the root lies in its cell,
half again for where the obstacle lies in its own). A check therefore errs only toward "blocked",
by at most about two cells at the root plus the bin's sweep at the piece's reach -- 2.2 cm at the
end of a piece 0.5 m out, 4.4 cm at 1 m -- which :attr:`Footprint.slack` states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .obstacles import _ccw, _convex_hull_2d, _footprint

#: Heading bins of a turn (72: 5 degrees, as Nav2's Smac planners default to).
BINS = 72
#: The carry grid's cell (m): finer than the navigator's 5 cm, since a carried load clears a post
#: or a door jamb by centimetres.
RESOLUTION = 0.025
#: How far, in cells, a cell centre may lie from a piece and still count as covered by it:
#: half a diagonal for the root, half for the obstacle.
_COVER = math.sqrt(2.0)


@dataclass
class Piece:
    """One convex part of a footprint: ``polygon`` (n, 2) in the body frame, counter-clockwise, the
    band ``z`` = ``(z_lo, z_hi)`` it occupies (m above the floor), and ``what`` it is (a name, for a
    failure to say)."""

    polygon: np.ndarray
    z: tuple
    what: str = ""
    #: Whether the piece is the mover's own body (not what it holds): a body may stand against the
    #: target it works at (``plan_way``'s ``dock``), what it holds may not.
    body: bool = False

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, dtype=float).reshape(-1, 2)
        self.z = (float(self.z[0]), float(self.z[1]))
        if len(self.polygon) < 3:
            raise ValueError(f"a footprint piece is a polygon of 3 points or more: {self.what!r}")
        if not self.z[1] > self.z[0]:
            raise ValueError(f"a footprint piece's band is (low, high), low below high: {self.z}")

    def posed(self, x: float, y: float, yaw: float) -> np.ndarray:
        """The polygon in the world, the body's root at ``(x, y)`` facing ``yaw``."""
        c, s = math.cos(yaw), math.sin(yaw)
        return self.polygon @ np.array([[c, s], [-s, c]]) + [x, y]


def geom_piece(model, data, geom: int, root, what: str = "") -> Piece | None:
    """A piece for MuJoCo geom ``geom`` as it stands now, in the frame of the body at ``root``
    ``(x, y, yaw)``: its convex floor footprint and its height. ``None`` for a geom type that has no
    footprint (a plane, a height field), or one wholly below the floor."""
    pts, z_lo, z_hi = _footprint(model, data, int(geom))
    if pts is None or z_hi <= 0.0:
        return None  # no footprint, or all of it below the floor (a part parked there)
    x, y, yaw = root
    c, s = math.cos(-yaw), math.sin(-yaw)
    rel = (np.asarray(pts, float) - [x, y]) @ np.array([[c, s], [-s, c]])
    hull = _convex_hull_2d(rel)
    if len(hull) < 3:
        return None
    z_hi = max(z_hi, z_lo + 1e-3)
    return Piece(_ccw(hull), (max(0.0, z_lo), z_hi), what)


def convex_hull(points) -> np.ndarray:
    """The counter-clockwise convex hull of ``points`` (n, 2): a piece for several at once."""
    return _ccw(_convex_hull_2d(np.asarray(points, float)))


def capsule_piece(a, b, radius: float, what: str = "", segments: int = 8) -> Piece:
    """A piece for a capsule from ``a`` to ``b`` (each ``(x, y, z)`` in the body frame) of
    ``radius``: an arm from shoulder to hand, a handle."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ang = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    disc = np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1)
    hull = _convex_hull_2d(np.concatenate([disc + a[:2], disc + b[:2]]))
    z = (max(0.0, min(a[2], b[2]) - radius), max(a[2], b[2]) + radius)
    return Piece(_ccw(hull), z, what)


def _grown(polygon: np.ndarray, by: float) -> np.ndarray:
    """A convex polygon grown outward by ``by`` (m): each vertex pushed out along its two edges'
    normals, so every edge moves out by ``by``."""
    if by <= 0.0:
        return polygon
    n = len(polygon)
    out = np.empty_like(polygon)
    for i in range(n):
        p0, p1, p2 = polygon[i - 1], polygon[i], polygon[(i + 1) % n]
        e0, e1 = p1 - p0, p2 - p1
        n0 = np.array([e0[1], -e0[0]]) / (np.linalg.norm(e0) + 1e-12)
        n1 = np.array([e1[1], -e1[0]]) / (np.linalg.norm(e1) + 1e-12)
        bis = n0 + n1
        k = float(np.dot(bis, n1))
        out[i] = p1 + bis * (by / k if abs(k) > 1e-6 else by)
    return out


def _cells_near(polygons, res: float, within: float) -> np.ndarray:
    """``(k, 2)`` integer cell offsets ``(di, dj)`` (x, y) whose centre ``(di, dj) * res`` lies within
    ``within`` (m) of one of the convex ``polygons`` (inside counts). The half-plane test makes it
    conservative near a corner, never short."""
    allp = np.concatenate(polygons)
    lo = np.floor((allp.min(0) - within) / res).astype(int) - 1
    hi = np.ceil((allp.max(0) + within) / res).astype(int) + 1
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    centres = np.stack([ii.ravel() * res, jj.ravel() * res], 1)
    hit = np.zeros(len(centres), bool)
    for poly in polygons:
        # every edge's outward normal (for CCW) and offset at once: a centre's distance past each
        e = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([e[:, 1], -e[:, 0]], 1) / (np.linalg.norm(e, axis=1)[:, None] + 1e-12)
        dmax = (centres @ normals.T - np.einsum("ed,ed->e", poly, normals)).max(1)
        hit |= dmax <= within
    return np.stack([ii.ravel()[hit], jj.ravel()[hit]], 1)


@dataclass
class Footprint:
    """A body and what it holds, as :class:`Piece` s, for one carry.

    ``margin`` (m) grows every piece: what a body that follows its way with an error (a robot's
    base on its anchor) may stray by, so the way keeps clear of that too. ``band_step`` (m) rounds
    the pieces' bands outward to a step, so that bands a float apart share a raster; each distinct
    band costs a raster, so a caller gives pieces of one part one band (a body's capsules, the
    geoms of what it holds).
    """

    pieces: list
    margin: float = 0.0
    resolution: float = RESOLUTION
    bins: int = BINS
    band_step: float = 0.01
    _cells: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.pieces:
            raise ValueError("a footprint has at least one piece")
        if not (math.isfinite(self.margin) and self.margin >= 0.0):
            raise ValueError(
                f"a footprint's margin is a distance of 0 m or more, not {self.margin}"
            )
        step = self.band_step
        for p in self.pieces:
            if step > 0:
                p.z = (
                    math.floor(p.z[0] / step + 1e-9) * step,
                    math.ceil(p.z[1] / step - 1e-9) * step,
                )

    @property
    def bands(self) -> list:
        """The height bands the pieces occupy, each once."""
        return sorted({p.z for p in self.pieces})

    @property
    def radius(self) -> float:
        """How far the footprint reaches from the root at most (m), its margin included."""
        return max(float(np.max(np.linalg.norm(p.polygon, axis=1))) for p in self.pieces) + (
            self.margin
        )

    @property
    def slack(self) -> float:
        """How far a check may err toward "blocked" at the root (m): see the module's note."""
        return (_COVER + 0.5 * math.sqrt(2.0)) * self.resolution

    def bin_of(self, yaw: float) -> int:
        return int(math.floor(yaw * self.bins / (2 * math.pi) + 0.5)) % self.bins

    def heading(self, k: int) -> float:
        return k * 2 * math.pi / self.bins

    def cells(self, k: int) -> dict:
        """``{band: (n, 2) cell offsets}`` the footprint covers at heading bin ``k``, with its root at
        a cell's centre: every piece swept over the bin, grown by the margin and by the cover a cell
        needs."""
        k %= self.bins
        if k not in self._cells:
            dth = 2 * math.pi / self.bins
            out = {}
            for band in self.bands:
                polys = []
                for p in self.pieces:
                    if p.z != band:
                        continue
                    grown = _grown(p.polygon, self.margin)
                    for sub in (-0.5, -0.25, 0.0, 0.25, 0.5):
                        th = (k + sub) * dth
                        c, s = math.cos(th), math.sin(th)
                        polys.append(grown @ np.array([[c, s], [-s, c]]))
                # a sweep of a quarter bin between the five samples moves a point at most
                # r * dth / 8 off the polygons sampled; cover that too
                reach = max(float(np.max(np.linalg.norm(q, axis=1))) for q in polys)
                within = _COVER * self.resolution + reach * dth / 8.0
                out[band] = _cells_near(polys, self.resolution, within)
            self._cells[k] = out
        return self._cells[k]
