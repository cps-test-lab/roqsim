"""Static wall footprints for the walker planner and ORCA, read from the compiled MuJoCo model --
the single source of truth.

Ported from an earlier in-house navigation prototype.

Walls live in the model as *collidable, static, non-floor* geoms: the ``floorplan`` plugin's wall
colliders, and any environment MJCF attached as welded bodies -- so reading the model needs no
external map files. "Static" means welded to the world (``body_weldid == 0``); the robot has DOFs
(excluded) and walkers are mocap bodies (excluded).

A body with DOFs that is **standing still and nobody is driving** is the one exception, and the
caller opts into it by naming those bodies in ``resting_roots``: a crate someone parked in a doorway
is a wall for as long as it stays there, and planning through it and then stopping in front of it is
worse than planning around it. The caller decides which bodies qualify -- see
:func:`~roqsim_nav.grid.build_grid` -- because "nobody is driving it" is knowledge about the world's
components, not about the model.

Each such geom is reduced to a 2D convex **footprint** (counter-clockwise, solid inside -- ORCA's
convention for an obstacle agents stay outside). The same polygons rasterize the planner grid and
feed ORCA, so global plan and local avoidance always agree. Geoms whose height doesn't overlap a
walking body are skipped (e.g. ceiling beams).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

_CIRCLE_SEG = 12  # polygon segments approximating a round footprint


#: Linear speed (m/s) below which a body in ``resting_roots`` counts as parked rather than moving.
#: Well under a walking pace, and above the jitter of a settled contact.
RESTING_SPEED = 0.05


def wall_polygons(
    model,
    data,
    z_lo: float = 0.1,
    z_hi: float = 1.8,
    *,
    resting_roots=(),
):
    """CCW ``[(x, y), ...]`` footprints for every static collidable wall geom
    whose vertical extent overlaps ``[z_lo, z_hi]`` (a walking body).

    ``resting_roots`` are weld-root body ids the caller has judged to be undriven props; each is
    included as a wall for this call only while it is moving slower than :data:`RESTING_SPEED`. The
    test is per call, so the same prop is a wall in one episode's grid and not in the next.
    """
    resting = frozenset(int(b) for b in resting_roots)
    polys = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if model.body_weldid[b] != 0:  # has DOFs (e.g. robot)
            if int(model.body_weldid[b]) not in resting or not _at_rest(model, data, b):
                continue
        if model.body_mocapid[b] >= 0:  # mocap (a walker, a navigated prop)
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue  # the floor
        if (model.geom_contype[g] | model.geom_conaffinity[g]) == 0:
            continue  # visual-only
        pts, zlo, zhi = _footprint(model, data, g)
        if pts is None or zhi < z_lo or zlo > z_hi:
            continue
        hull = _convex_hull_2d(pts)
        if len(hull) >= 3:
            polys.append([(float(x), float(y)) for x, y in _ccw(hull)])
    return polys


def _at_rest(model, data, body_id: int) -> bool:
    """Whether ``body_id`` is moving slower than :data:`RESTING_SPEED`.

    Reads ``cvel``, which is only meaningful after a forward pass -- every caller here runs one
    before rasterizing, for the same reason ``geom_xpos`` needs it.
    """
    return float(np.linalg.norm(data.cvel[int(body_id)][3:6])) < RESTING_SPEED


def _footprint(model, data, g):
    """``(xy_points (n,2), z_min, z_max)`` for geom ``g`` in world frame, or
    ``(None, 0, 0)`` for an unsupported type."""
    t = int(model.geom_type[g])
    R = data.geom_xmat[g].reshape(3, 3)
    p = np.asarray(data.geom_xpos[g], dtype=float)
    s = model.geom_size[g]
    G = mujoco.mjtGeom

    if t == G.mjGEOM_MESH:
        did = int(model.geom_dataid[g])
        adr = int(model.mesh_vertadr[did])
        n = int(model.mesh_vertnum[did])
        V = model.mesh_vert[adr : adr + n].reshape(-1, 3)
        W = V @ R.T + p
    elif t == G.mjGEOM_BOX:
        corners = np.array(
            [[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])]
        )
        W = corners @ R.T + p
    elif t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE, G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID):
        r = float(max(s[0], s[1])) if t == G.mjGEOM_ELLIPSOID else float(s[0])
        if t == G.mjGEOM_CYLINDER:
            half = float(s[1])
        elif t == G.mjGEOM_CAPSULE:
            half = float(s[1]) + r
        elif t == G.mjGEOM_ELLIPSOID:
            half = float(s[2])
        else:  # sphere
            half = r
        ang = np.linspace(0.0, 2 * np.pi, _CIRCLE_SEG, endpoint=False)
        circ = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], 1)
        W = circ @ R.T + p
        return W[:, :2], float(p[2] - half), float(p[2] + half)
    else:
        return None, 0.0, 0.0
    return W[:, :2], float(W[:, 2].min()), float(W[:, 2].max())


def dynamic_obstacle_bodies(model, exclude_mocapids):
    """``[(body_id, radius), ...]`` for every collidable **mocap** body that is
    not one of ``exclude_mocapids`` (the caller's own articulated parts).

    These are the things that move without having DOFs: a navigated prop, a
    walker's peers, a prop teleported in at runtime via ``/spawn_entity``.
    :func:`wall_polygons` deliberately skips them because they are not walls,
    and a local-avoidance model's *static* geometry cannot represent them
    because they move. A caller instead refreshes each one's xy into the model
    as an immovable agent every step -- the same ground-truth-overwrite a
    non-yielding participant gets -- so navigating agents steer around them too.

    ``radius`` is the geom's circumscribed xy radius, which is yaw-invariant, so a
    rotating box keeps a footprint that never shrinks below its true extent."""
    out = []
    for b in range(model.nbody):
        mid = int(model.body_mocapid[b])
        if mid < 0 or mid in exclude_mocapids:
            continue
        r = 0.0
        adr, num = int(model.body_geomadr[b]), int(model.body_geomnum[b])
        for g in range(adr, adr + num):
            if (model.geom_contype[g] | model.geom_conaffinity[g]) == 0:
                continue  # visual-only geom
            r = max(r, _geom_xy_radius(model, g))
        if r > 0.0:
            out.append((b, r))
    return out


def _geom_xy_radius(model, g) -> float:
    """Circumscribed radius of geom ``g`` in its xy plane (independent of yaw)."""
    t = int(model.geom_type[g])
    s = model.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_BOX:
        return float(math.hypot(s[0], s[1]))
    if t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE, G.mjGEOM_SPHERE):
        return float(s[0])
    if t == G.mjGEOM_ELLIPSOID:
        return float(max(s[0], s[1]))
    if t == G.mjGEOM_MESH:
        did = int(model.geom_dataid[g])
        adr, n = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
        if n <= 0:
            return 0.0
        V = model.mesh_vert[adr : adr + n].reshape(-1, 3)
        return float(np.max(np.hypot(V[:, 0], V[:, 1])))
    return 0.0


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain 2D convex hull (no SciPy). Returns the hull
    vertices CCW; fewer than 3 unique points -> the unique points as-is."""
    pts = np.unique(np.round(np.asarray(points, dtype=float), 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(chain_pts):
        out = []
        for q in chain_pts:
            while len(out) >= 2 and _cross(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _ccw(poly: np.ndarray) -> np.ndarray:
    """Ensure counter-clockwise winding (positive signed area)."""
    x, y = poly[:, 0], poly[:, 1]
    area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return poly if area > 0 else poly[::-1]
