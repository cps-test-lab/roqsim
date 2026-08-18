"""Did the convex hulls close a way through? The check a picture of a scene cannot make.

MuJoCo collides a mesh by its **convex hull**. A wall with a door cut out of it is not convex, so its
hull is the solid wall -- and the doorway then behaves differently depending on who asks:

===================  ====================================================
renderers            see the triangles: the doorway is open
``mj_ray``/lidar     tests the triangles too: the doorway is open
physics              collides the hull: the doorway is a solid slab
===================  ====================================================

Those three are the axes a scene's geometry is judged on, and they are set by different fields, which
is worth knowing before assuming one implies another:

===================  ====================================================
a renderer draws     geom group in ``MjvOption.geomgroup`` (default 0-2), and alpha > 0
a raycast returns    resolved alpha > 0, **and** the group is in the caster's mask
physics collides     ``contype``/``conaffinity`` -- appearance is irrelevant
===================  ====================================================

So "invisible" and "unsensed" are one setting (alpha), not two: hiding a geom takes it away from every
lidar as well. A baked scene's ground uses that deliberately -- ``floor`` is alpha 0 and collides,
``floor_visual`` is opaque and does not.

So the world looks right in every render, reads as passable on ``/scan``, and stops the robot dead.
That is not a defect a human or an agent can see; it has to be measured. secorolab shipped this way
and 87 % of the building was unreachable through eight walled doorways, through four campaigns.

What is measured, and why it is not per-part
--------------------------------------------

The obvious per-part test -- "does this part's hull add material wider than X?" -- does not work. The
phantom slab filling a secorolab doorway is 0.05 m across the wall by 0.79 m along it: its inscribed
width is the *wall thickness*, so any threshold catching it also fires on every ordinary wall. **The
quantity that matters is the opening closed, not the girth of the thing closing it**, and the two are
perpendicular.

So the test is about connectivity, over the whole scene at once:

1. rasterise the collidable parts twice at a probe height -- once from their triangles (what is really
   there), once from their convex hulls (what MuJoCo will collide);
2. erode the free space by the probe radius;
3. if two cells connected in the triangle rasterisation are disconnected in the hull one, a hull closed
   a way through.

Scene-level because a passage can be closed *jointly* by parts none of which closes it alone -- in
secorolab every doorway is covered by two overlapping wall parts.
"""

from __future__ import annotations

import numpy as np

#: Half the narrowest passage worth defending, in metres. A gap this wide that the geometry shows open
#: and the hulls close is a doorway, not modelling slop: it is under every robot the substrate runs
#: (the TurtleBot 4 is 0.35 m across) and far over the millimetre scale of a chamfer or a mesh seam.
DEFAULT_PROBE_RADIUS = 0.15

#: Rasterisation cell, in metres. Matches the 0.05 m of a ROS ``map_server`` occupancy grid, so a
#: verdict here is at the resolution the navigation stack plans at.
DEFAULT_CELL = 0.05


class Passage:
    """A way through that the triangles leave open and the hulls close."""

    def __init__(self, centre: tuple[float, float], cells: int, cell: float):
        self.centre = centre
        self.cells = cells
        self.area = cells * cell * cell

    def __str__(self) -> str:
        return (
            f"({self.centre[0]:.2f}, {self.centre[1]:.2f}) -- {self.area:.2f} m^2 of floor cut off"
        )


def _hull_2d(pts: np.ndarray) -> np.ndarray | None:
    from scipy.spatial import ConvexHull

    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:  # degenerate/collinear: no area, so it closes nothing
        return None


def _slice_segments(verts, faces, z):
    """Where the plane ``Z = z`` cuts the part's surface: one 2D segment per crossing triangle.

    Not the triangles' own projection -- a wall's faces are vertical, so their projection is a
    zero-area line and rasterising that leaves a one-cell-wide wall the erosion then eats, which reads
    as "you can walk through walls". These segments are the *outline* of the cross-section; the region
    they enclose is filled by :func:`_fill_parity`.
    """
    tri = verts[faces]
    out = []
    for t in tri[(tri[:, :, 2].min(axis=1) <= z) & (tri[:, :, 2].max(axis=1) >= z)]:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = t[a, 2], t[b, 2]
            if (za - z) * (zb - z) < 0:
                s = (z - za) / (zb - za)
                pts.append(t[a, :2] + s * (t[b, :2] - t[a, :2]))
            elif za == z:
                pts.append(t[a, :2])
        if len(pts) >= 2:
            out.append([pts[0], pts[1]])
    return np.array(out) if out else np.empty((0, 2, 2))


def _fill_parity(segs, grid):
    """Even-odd fill of the region enclosed by *segs*, scanline by scanline.

    A closed mesh gives closed loops here, and so does an open one that merely lacks top/bottom caps
    (a wall skin still encloses its own thickness at mid-height). A genuinely zero-thickness surface
    gives a single unpaired segment and so fills nothing -- which is right: it collides nothing, and
    its entire hull is phantom.
    """
    X, Y, res, ox, oy, h, w = grid
    occ = np.zeros((h, w), bool)
    if len(segs) == 0:
        return occ
    y0, y1 = segs[:, 0, 1], segs[:, 1, 1]
    x0, x1 = segs[:, 0, 0], segs[:, 1, 0]
    ys = oy + (np.arange(h) + 0.5) * res
    for j, y in enumerate(ys):
        hit = (y0 - y) * (y1 - y) < 0
        if not hit.any():
            continue
        t = (y - y0[hit]) / (y1[hit] - y0[hit])
        xs_hit = np.sort(x0[hit] + t * (x1[hit] - x0[hit]))
        # strict=False on purpose: an open or non-manifold surface can leave an odd number of
        # crossings on a scanline, and the unpaired last one has no interior to fill. Dropping it
        # under-fills that row, which is the safe direction -- the check then reports less, never more.
        for a, b in zip(xs_hit[0::2], xs_hit[1::2], strict=False):
            ca = max(0, int(np.ceil((a - ox) / res - 0.5)))
            cb = min(w, int(np.floor((b - ox) / res - 0.5)) + 1)
            if cb > ca:
                occ[j, ca:cb] = True
    return occ


def _rasterise(polys, grid):
    from matplotlib.path import Path as MPath

    X, Y, res, ox, oy, h, w = grid
    occ = np.zeros((h, w), bool)
    for poly in polys:
        if poly is None or len(poly) < 3:
            continue
        c0 = max(0, int((poly[:, 0].min() - ox) / res) - 1)
        c1 = min(w, int((poly[:, 0].max() - ox) / res) + 2)
        r0 = max(0, int((poly[:, 1].min() - oy) / res) - 1)
        r1 = min(h, int((poly[:, 1].max() - oy) / res) + 2)
        if c1 <= c0 or r1 <= r0:
            continue
        sub = np.column_stack([X[r0:r1, c0:c1].ravel(), Y[r0:r1, c0:c1].ravel()])
        occ[r0:r1, c0:c1] |= MPath(poly).contains_points(sub).reshape(r1 - r0, c1 - c0)
    return occ


def closed_passages(
    parts,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    cell: float = DEFAULT_CELL,
    z: float | None = None,
) -> list[Passage]:
    """Ways through that *parts*' convex hulls close but their triangles leave open.

    *parts* is an iterable of ``(verts, faces)`` for the **collidable** geometry only -- a visual-only
    mesh has no hull in physics and must not be judged. Empty list means the hulls are faithful at
    this height.

    *z* defaults to the probe radius above the parts' floor, i.e. where a robot's widest part is.
    """
    from scipy import ndimage

    parts = [(np.asarray(v, float), np.asarray(f, int)) for v, f in parts if len(v) >= 3]
    if not parts:
        return []

    lo = np.min([v.min(axis=0) for v, _ in parts], axis=0)
    hi = np.max([v.max(axis=0) for v, _ in parts], axis=0)
    if z is None:
        z = float(lo[2]) + probe_radius

    pad = probe_radius * 2 + cell * 2  # room for the erosion, and a free border to flood from
    ox, oy = float(lo[0]) - pad, float(lo[1]) - pad
    w = int(np.ceil((hi[0] - lo[0] + 2 * pad) / cell))
    h = int(np.ceil((hi[1] - lo[1] + 2 * pad) / cell))
    if w < 3 or h < 3:
        return []
    xs = ox + (np.arange(w) + 0.5) * cell
    ys = oy + (np.arange(h) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    grid = (X, Y, cell, ox, oy, h, w)

    true_occ = np.zeros((h, w), bool)
    for v, f in parts:
        true_occ |= _fill_parity(_slice_segments(v, f, z), grid)
    # A part is hulled in 3D, so its footprint at any height it spans is the hull of all its vertices.
    # Parts that do not reach this height collide nothing here and are skipped in both rasterisations.
    hull_occ = _rasterise(
        [_hull_2d(v[:, :2]) for v, _ in parts if v[:, 2].min() <= z <= v[:, 2].max()], grid
    )

    r = max(1, int(round(probe_radius / cell)))
    disc = np.zeros((2 * r + 1, 2 * r + 1), bool)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disc[yy * yy + xx * xx <= r * r] = True

    # border_value=1: beyond the grid is open ground. The default treats it as solid, which erodes the
    # padded border away -- and the border is exactly the "outside" both floods start from.
    true_free = ndimage.binary_erosion(~true_occ, disc, border_value=1)
    hull_free = ndimage.binary_erosion(~hull_occ, disc, border_value=1)

    # The outside border is one region in both, so label from there: anything the triangles reach from
    # outside but the hulls do not is a region the hulls sealed off.
    true_lab, _ = ndimage.label(true_free)
    hull_lab, _ = ndimage.label(hull_free)
    outside_true = true_lab[0, 0]
    outside_hull = hull_lab[0, 0]
    if outside_true == 0 or outside_hull == 0:
        return []  # the border is not free: nothing to compare against

    sealed = (true_lab == outside_true) & (hull_lab != outside_hull)
    if not sealed.any():
        return []

    lab, n = ndimage.label(sealed)
    out = []
    for i in range(1, n + 1):
        rs, cs = np.where(lab == i)
        # One cell here and there is rasterisation noise on a wall face, not a way through.
        if len(rs) * cell * cell < (2 * probe_radius) ** 2:
            continue
        out.append(Passage((float(xs[cs].mean()), float(ys[rs].mean())), len(rs), cell))
    return sorted(out, key=lambda p: -p.cells)
