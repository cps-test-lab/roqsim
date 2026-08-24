"""Where to put the sample points that coverage is measured over.

Two targets, both derived from the *compiled* MuJoCo world (so they work for any world source -- a
baked scene, a floorplan STL, a hand-written MJCF) using only numpy and MuJoCo raycasts:

* :func:`object_surface_points` -- points on the surfaces of the world's objects (every collidable
  geom that is not obvious structure: floor/walls/ceiling), labelled by geom name. Answers "are these
  objects observed, and by how many sensors."
* :func:`room_volume_points` -- a 3D grid of points in the room's free interior at a few heights.
  Answers "how much of the room can be observed."

Free-space classification (for the volume grid) is raycast-based and deliberately conservative: a point
is kept only if it is *enclosed* (an axis ray hits geometry on all four horizontal sides -- so points
outside the building footprint, over the infinite floor plane, are dropped) and not *embedded* in a
geom (most axis rays hit a back-face). Points we wrongly drop only make coverage look worse, never
better, so the error is safe. There is no scipy/trimesh dependency anywhere here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim import raycast

# Geoms whose name contains one of these are treated as structure, not "objects", for surface sampling.
_STRUCTURE_SUBSTRINGS = ("floor", "ground", "wall", "ceiling", "roof")

_AXES6 = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
)
_HORIZONTAL = slice(0, 4)  # the first four axes are horizontal


def _geomgroup_mask(include_groups=(0, 1, 2, 3)) -> np.ndarray:
    mask = np.zeros(6, dtype=np.uint8)
    for g in include_groups:
        if 0 <= int(g) < 6:
            mask[int(g)] = 1
    return mask


def world_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds over finite (non-plane) geoms, using each geom's bounding radius."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c = data.geom_xpos[g]
        r = float(model.geom_rbound[g]) or 0.0
        lo = np.minimum(lo, c - r)
        hi = np.maximum(hi, c + r)
    if not np.isfinite(lo).all():  # no finite geoms -> fall back to the model's stat extent
        ext = float(model.stat.extent)
        center = np.array(model.stat.center, dtype=np.float64)
        lo, hi = center - ext, center + ext
    return lo, hi


# -- object surfaces ---------------------------------------------------------------------------------


def _geom_local_surface(model: mujoco.MjModel, g: int, per_object: int) -> np.ndarray:
    """Surface sample points in the geom's local frame."""
    # int(): mjt* enums compare False against a numpy scalar (`x in (...)` puts the enum on the
    # left of `==`), so the CYLINDER/CAPSULE match below would silently never fire.
    gtype = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        adr = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        verts = np.array(model.mesh_vert[adr : adr + n], dtype=np.float64).reshape(-1, 3)
        if len(verts) > per_object:  # deterministic even stride, no RNG
            verts = verts[np.linspace(0, len(verts) - 1, per_object).astype(int)]
        return verts
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = size[:3]
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        corners = signs * size[:3]
        faces = np.array(
            [[sx, 0, 0], [-sx, 0, 0], [0, sy, 0], [0, -sy, 0], [0, 0, sz], [0, 0, -sz]], float
        )
        return np.vstack([corners, faces])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _fibonacci_sphere(per_object) * size[0]
    if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, half = size[0], size[1]
        ang = np.linspace(0, 2 * np.pi, max(8, per_object // 2), endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], axis=1)
        return np.vstack([ring + [0, 0, half], ring - [0, 0, half]])
    # ellipsoid / other: sample the bounding sphere scaled by size
    return _fibonacci_sphere(per_object) * float(model.geom_rbound[g] or size[0])


def _fibonacci_sphere(n: int) -> np.ndarray:
    n = max(8, int(n))
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def object_surface_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    per_object: int = 64,
    offset: float = 0.02,
    exclude_substrings=_STRUCTURE_SUBSTRINGS,
    include_groups=(0, 1, 2, 3),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Points on the world's object surfaces, offset outward along the normal.

    Returns ``(points (P,3), labels (P,) int, label_names)`` where ``labels[i]`` indexes
    ``label_names`` (the geom name). Structure geoms (floor/walls/ceiling by name) and geoms outside
    ``include_groups`` are skipped.
    """
    groups = set(int(g) for g in include_groups)
    pts: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for g in range(model.ngeom):
        if int(model.geom_group[g]) not in groups:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        low = name.lower()
        if any(sub in low for sub in exclude_substrings):
            continue
        local = _geom_local_surface(model, g, per_object)
        if len(local) == 0:
            continue
        R = np.array(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        t = np.array(data.geom_xpos[g], dtype=np.float64)
        world = local @ R.T + t
        # Offset outward from the geom centre so the target's own surface is not read as an occluder.
        out = world - t
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-9)
        world = world + offset * out
        idx = len(names)
        names.append(name)
        pts.append(world)
        labels.append(np.full(len(world), idx))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=int), []
    return np.vstack(pts), np.concatenate(labels), names


# -- room volume -------------------------------------------------------------------------------------


def _classify_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points: np.ndarray,
    *,
    max_dist: float,
    geomgroup: np.ndarray,
) -> np.ndarray:
    """Boolean mask: which points are enclosed (inside the building) and not embedded in a geom.

    One :func:`roqsim.raycast.cast_many` for the whole grid rather than a call per point. Not a single
    batch -- ``mj_multiRay`` casts from one origin, and here every point *is* an origin -- but the
    calls are independent, so the seam threads them and the classification below vectorises over all
    points at once instead of once per point.
    """
    hits = raycast.cast_many(
        model,
        data,
        points,
        _AXES6,
        cutoff=max_dist,
        geomgroup=geomgroup,
        flg_static=True,
        normals=True,
    )
    hit = hits.dist >= 0.0  # (P, 6)
    # Enclosed: every horizontal axis meets something, i.e. the point is inside the building.
    enclosed = hit[:, _HORIZONTAL].all(axis=1)
    # Embedded: a hit whose normal points the same way as the ray is a backface, so the point is
    # inside that geom rather than in free space beside it. Four of six is the majority test.
    facing = np.einsum("pij,ij->pi", hits.normal, _AXES6) > 0.0
    embedded = (hit & facing).sum(axis=1) >= 4
    return enclosed & ~embedded


def room_volume_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    resolution: float = 0.25,
    heights=(0.3, 1.0, 1.7),
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    include_groups=(0, 1, 2, 3),
) -> np.ndarray:
    """A 3D grid of free-interior room points at the given heights (world z)."""
    lo, hi = bounds if bounds is not None else world_bounds(model, data)
    xs = np.arange(lo[0], hi[0] + resolution, resolution)
    ys = np.arange(lo[1], hi[1] + resolution, resolution)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    flat = np.stack([gx.ravel(), gy.ravel()], axis=1)
    max_dist = float(np.linalg.norm(hi - lo)) + 1.0
    geomgroup = _geomgroup_mask(include_groups)
    kept: list[np.ndarray] = []
    for h in heights:
        pts = np.column_stack([flat, np.full(len(flat), float(h))])
        mask = _classify_points(model, data, pts, max_dist=max_dist, geomgroup=geomgroup)
        kept.append(pts[mask])
    if not kept:
        return np.zeros((0, 3))
    return np.vstack(kept)


# -- gap clustering ----------------------------------------------------------------------------------


@dataclass
class Gap:
    centroid: list[float]
    n_points: int
    bbox_min: list[float]
    bbox_max: list[float]


def cluster_gaps(uncovered_points: np.ndarray, *, resolution: float = 0.25) -> list[Gap]:
    """Cluster uncovered points into connected regions via a voxel-grid BFS (26-connectivity)."""
    pts = np.asarray(uncovered_points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return []
    origin = pts.min(axis=0)
    vox = np.floor((pts - origin) / resolution).astype(int)
    occupied: dict[tuple[int, int, int], list[int]] = {}
    for i, v in enumerate(map(tuple, vox)):
        occupied.setdefault(v, []).append(i)

    seen: set[tuple[int, int, int]] = set()
    neigh = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]
    gaps: list[Gap] = []
    for start in occupied:
        if start in seen:
            continue
        cluster_idx: list[int] = []
        q = deque([start])
        seen.add(start)
        while q:
            cell = q.popleft()
            cluster_idx.extend(occupied[cell])
            cx, cy, cz = cell
            for dx, dy, dz in neigh:
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in occupied and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        cluster_pts = pts[cluster_idx]
        gaps.append(
            Gap(
                centroid=[float(v) for v in cluster_pts.mean(axis=0)],
                n_points=len(cluster_idx),
                bbox_min=[float(v) for v in cluster_pts.min(axis=0)],
                bbox_max=[float(v) for v in cluster_pts.max(axis=0)],
            )
        )
    gaps.sort(key=lambda gp: gp.n_points, reverse=True)
    return gaps
