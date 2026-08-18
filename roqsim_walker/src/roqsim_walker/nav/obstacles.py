"""Static wall footprints for the walker planner and ORCA, read from the compiled MuJoCo model --
the single source of truth.

Ported from our earlier in-house nav prototype's ``mujoco_nav.pedestrian.obstacles``.

Walls live in the model as *collidable, static, non-floor* geoms: the ``floorplan`` plugin's wall
colliders, and any environment MJCF attached as welded bodies -- so reading the model needs no
external map files. "Static" means welded to the world (``body_weldid == 0``); the robot has DOFs
(excluded) and walkers are mocap bodies (excluded).

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


def wall_polygons(model, data, z_lo: float = 0.1, z_hi: float = 1.8):
    """CCW ``[(x, y), ...]`` footprints for every static collidable wall geom
    whose vertical extent overlaps ``[z_lo, z_hi]`` (a walking body)."""
    polys = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if model.body_weldid[b] != 0:  # has DOFs (e.g. robot)
            continue
        if model.body_mocapid[b] >= 0:  # mocap (e.g. pedestrian)
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
    not one of ``exclude_mocapids`` (the pedestrians' own articulated parts).

    These are the runtime-teleported pool obstacles -- the ``spawnable_objects``
    spawned via ``/spawn_entity`` and the ``initial_objects`` placed at startup
    (see :mod:`roqsim_walker.pool`). Being mocap bodies, :func:`wall_polygons`
    deliberately skips them, and ORCA's *static* obstacles can't represent them
    because they move (spawned in/out, parked off-map when idle). The controller
    instead refreshes each one's xy into ORCA as an immovable agent every step --
    the same ground-truth-overwrite trick it uses for the robot -- so pedestrians
    walk around spawned/initial props too.

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
