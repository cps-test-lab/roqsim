"""Measure what a prop collides as against what it looks like.

A prop the import pipeline produces is one mesh geom, and MuJoCo collides a mesh geom as its
**convex hull**. For anything with a span under it -- a trestle desk, a shelf, a chair -- the hull is
a solid block filling the space the shape exists to leave open, so a robot crashes into geometry that
is not there, and the contact it reports names a geom the model never named. No other check in the
pipeline looks at this, so a prop reaches a scene that way unless someone measures it.

Two subcommands, one measurement:

``audit`` -- which props are wrong. Samples each collidable mesh geom's convex hull and reports how
far that hull stands from the surface it is meant to represent. Corpus triage: run it over a
provider and it ranks what needs work.

``diff`` -- how wrong one candidate is. Compares the collision geoms against the visual geoms in both
directions, because each direction is a different defect and one alone hides the other:

  coverage   visual surface -> nearest collider   geometry a robot will pass straight through
  overreach  collider surface -> nearest visual   obstacle standing where the prop is not

**Both measurements are surface-to-surface, and that is not a detail.** The obvious metric -- the
ratio of a hull's volume to its mesh's -- needs a watertight surface to have a volume at all, and not
one prop in this library has one; they are artwork, open shells. On such a mesh the signed volume is
whatever the unclosed faces happen to sum to, and the ratio ranges over seven orders of magnitude on
geometry that differs by a factor of two. Distances between surfaces need no interior, and they come
out in millimetres a reader can act on.

Everything is read off the **compiled** model, not the source OBJ, so a baked ``<mesh scale>``, a
prop built from several meshes, meshes in a ``meshes/`` subfolder and ``assets:``-borrowed geometry
are all handled without a special case -- and a model is named the way a world names it
(``office_table``, ``roqsim_assets:office_table``, or a path), so this works on anything with an
MJCF, a robot link included, not only on a prop folder.

Usage::

    roqsim assets collision audit                          # every model of every provider
    roqsim assets collision audit --provider roqsim_assets # one provider
    roqsim assets collision audit office_table desk_diy    # named models
    roqsim assets collision diff office_table              # one model, in detail
    roqsim assets collision diff office_table --tol 5      # a tighter question
    roqsim assets collision diff office_table --json       # machine-readable

Exits non-zero when anything is FAIL, so an agent or a CI step fails loudly rather than reading past
it. To SEE what a verdict is talking about, render the two halves against each other::

    roqsim render office_table --geomgroup 2,3 --out check.png

Needs the ``collision`` extra (``pip install 'roqsim_assets[collision]'``) for trimesh + scipy:
surface sampling, convex hulls and the nearest-point queries. It is an extra rather than a
dependency because placing a prop needs none of it -- this is a tool for the person adding one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

#: Default agreement required between the two surfaces, in metres. A costmap cell in the navigation
#: worlds here is 50 mm, so 10 mm is well inside the resolution any consumer of this geometry has.
DEFAULT_TOL = 0.010

#: Fixed so two runs of the same question give the same answer; ``--seed`` varies it to check that
#: a verdict is not an artefact of one point set.
DEFAULT_SEED = 0

#: Points sampled on each surface for a query. The reported resolution is derived from this and the
#: surface area, so a caller can see how sharp the answer is rather than assume it.
QUERY_SAMPLES = 60_000
CLOUD_SAMPLES = 600_000

_EXTRA_HELP = (
    "roqsim assets collision needs trimesh + scipy: pip install 'roqsim_assets[collision]'"
)


def _deps():
    """trimesh + scipy, or a clear instruction. Never a silent degradation to a weaker metric."""
    try:
        import trimesh
        from scipy.spatial import cKDTree
    except ImportError as exc:  # noqa: TRY003 - the message IS the handling
        raise SystemExit(f"{_EXTRA_HELP}\n  ({exc})") from exc
    return trimesh, cKDTree


# -- geometry ------------------------------------------------------------------------------------


def compile_model(ref: str):
    """Compile the model ``ref`` names and return ``(model, data)`` with poses resolved.

    Compiling is what makes this work on any model rather than on a prop folder: MuJoCo applies the
    mesh scale, the geom offsets and the compiler's own mesh recentring, so ``geom_xpos``/``geom_xmat``
    are where the geometry actually is.
    """
    from roqsim.models import apply_assets, resolve_model

    asset = resolve_model(ref)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def geom_surface(trimesh, model, data, gid: int):
    """A geom as a surface mesh in world coordinates, or ``None`` for a shape with no closed form.

    A plane and a heightfield are unbounded, so there is no surface to sample and no useful distance
    to anything; they are scenery a prop is placed on, never part of the prop.
    """
    kind, size = model.geom_type[gid], model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size[:3])
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]))
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(radius=1.0)
        mesh.apply_scale(size[:3])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]))
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]))
        mesh.apply_translation([0, 0, -float(size[1])])  # trimesh builds it from the origin up
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        i = model.geom_dataid[gid]
        verts = model.mesh_vert[
            model.mesh_vertadr[i] : model.mesh_vertadr[i] + model.mesh_vertnum[i]
        ]
        faces = model.mesh_face[
            model.mesh_faceadr[i] : model.mesh_faceadr[i] + model.mesh_facenum[i]
        ]
        mesh = trimesh.Trimesh(
            vertices=np.asarray(verts, float), faces=np.asarray(faces), process=False
        )
    else:
        return None
    pose = np.eye(4)
    pose[:3, :3] = data.geom_xmat[gid].reshape(3, 3)
    pose[:3, 3] = data.geom_xpos[gid]
    mesh.apply_transform(pose)
    return mesh


def split_geoms(model) -> tuple[list[int], list[int]]:
    """Geom ids as ``(visual, collision)``.

    The split is MuJoCo's own: a geom with neither ``contype`` nor ``conaffinity`` cannot pair with
    anything, so it is decoration whatever group it is drawn in. Reading the masks rather than the
    group number means a model that uses the group convention loosely still measures correctly.
    """
    visual, collision = [], []
    for gid in range(model.ngeom):
        solid = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
        (collision if solid else visual).append(gid)
    return visual, collision


#: A face this close to the model's lowest point, pointing down, is where the prop meets the floor.
BASE_EPS = 0.002


def drop_base_faces(trimesh, meshes: list, points, owners, normals, z_floor: float):
    """Discard samples on a downward face at the prop's base -- the patch it stands on.

    Artwork is an open shell far more often than not: of the props here, most have NO downward area
    at the base at all, the underside simply not being modelled. A collision primitive resting on the
    floor does have a bottom face, so measured naively every such box reports its own footprint as
    obstacle standing where the prop is not -- on one table that alone was a third of the overreach.
    The absence is a modelling convention, not a claim about the shape, so neither surface is scored
    on it.
    """
    keep = ~((points[:, 2] <= z_floor + BASE_EPS) & (normals[:, 2] < -0.9))
    return points[keep], owners[keep]


def sample_surfaces(
    trimesh, meshes: list, total: int, seed: int = DEFAULT_SEED, normals: bool = False
):
    """``total`` points over ``meshes`` in proportion to area, plus which mesh each point came from.

    Seeded, because the numbers are meant to be COMPARED: an unseeded sampling moves the reported
    maximum by a few tenths of a millimetre between runs, and an agent editing a collision model
    cannot then tell a real improvement from the noise of having asked twice. The seed is offset per
    mesh so two identical parts do not receive identical point sets.

    The owner index is what turns a score into a work order -- it is how a millimetre figure becomes
    the NAME of the geom responsible for it.
    """
    if not meshes:
        empty = (np.zeros((0, 3)), np.zeros(0, int))
        return (*empty, np.zeros((0, 3))) if normals else empty
    areas = np.array([max(float(m.area), 1e-12) for m in meshes])
    share = areas / areas.sum()
    points, owners, face_normals = [], [], []
    for i, (mesh, frac) in enumerate(zip(meshes, share, strict=True)):
        count = max(int(total * frac), 1)
        if normals:
            pts, faces = mesh.sample(count, return_index=True, seed=seed + i)
            face_normals.append(np.asarray(mesh.face_normals)[faces])
        else:
            pts = mesh.sample(count, seed=seed + i)
        points.append(pts)
        owners.append(np.full(len(pts), i))
    if normals:
        return np.vstack(points), np.concatenate(owners), np.vstack(face_normals)
    return np.vstack(points), np.concatenate(owners)


def distance_to(
    trimesh, cKDTree, meshes: list, points: np.ndarray, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, float]:
    """Distance from each point to the nearest surface in ``meshes``, and the resolution of that answer.

    A KD-tree over a dense sampling of the target surfaces rather than an exact point-to-triangle
    query: the exact one needs ``rtree``, a binding to a native spatial index, and it would buy
    accuracy far below the tolerance anyone asks of this. The cost is that a distance is never
    under-reported by more than the sample spacing, which is returned so the caller can print it
    instead of trusting it.
    """
    if not meshes or len(points) == 0:
        return np.full(len(points), np.inf), 0.0
    cloud, _ = sample_surfaces(trimesh, meshes, CLOUD_SAMPLES, seed)
    area = sum(float(m.area) for m in meshes)
    resolution = float(np.sqrt(area / max(len(cloud), 1)))
    return cKDTree(cloud).query(points)[0], resolution


def inside_any(model, data, gids: list[int], points: np.ndarray) -> np.ndarray:
    """Which points lie INSIDE one of the collision primitives.

    Coverage asks whether a bit of the prop is backed by something solid, and a distance to the
    collider's outer SURFACE cannot answer that: a hollow shell standing in for a solid panel has its
    inner faces enclosed by the primitive, metres of them, every one reading as uncovered. Enclosed
    is covered, so containment is the test and the distance is only the fallback outside.

    Analytic per shape rather than a mesh query, since the primitives are exactly the shapes MuJoCo
    collides. A mesh collision geom has no cheap inside, so it contributes none -- and a mesh geom
    that collides is the FAIL this tool exists to report anyway.
    """
    hit = np.zeros(len(points), bool)
    for gid in gids:
        kind, size = model.geom_type[gid], model.geom_size[gid]
        local = (points - data.geom_xpos[gid]) @ data.geom_xmat[gid].reshape(3, 3)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            hit |= np.all(np.abs(local) <= size[:3], axis=1)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            hit |= np.linalg.norm(local, axis=1) <= size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            hit |= np.sum((local / size[:3]) ** 2, axis=1) <= 1.0
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            hit |= (np.linalg.norm(local[:, :2], axis=1) <= size[0]) & (
                np.abs(local[:, 2]) <= size[1]
            )
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axial = np.clip(local[:, 2], -size[1], size[1])
            hit |= (
                np.linalg.norm(local - np.c_[np.zeros((len(local), 2)), axial], axis=1) <= size[0]
            )
    return hit


def _stats(distance: np.ndarray, tol: float) -> dict:
    if len(distance) == 0:
        return {"mean_mm": 0.0, "p95_mm": 0.0, "max_mm": 0.0, "beyond_tol": 0.0}
    return {
        "mean_mm": float(distance.mean() * 1000),
        "p95_mm": float(np.percentile(distance, 95) * 1000),
        "max_mm": float(distance.max() * 1000),
        "beyond_tol": float((distance > tol).mean()),
    }


def effort(trimesh, meshes: list, tol: float) -> dict:
    """How much work this prop's skeleton is, before anyone starts writing it.

    Two deterministic numbers, because "which of these 22 do I do first" is a planning question and
    guessing it from a photograph is how a half-hour job turns out to be an afternoon:

    ``boxes_upper`` what a greedy voxel decomposition needs at this tolerance -- an upper bound on
                    the primitive count, since a person reading the shape does far better (the
                    trestle desk: 31 by voxel at 50 mm, 11 by hand). Monotone in complexity, which
                    is what makes it orderable even though it is not the answer. It counts the
                    SURFACE, these meshes being shells: a solid cube reads 6, one slab per face.
    ``axis_frac``   the fraction of surface area whose normal points along x, y or z. It separates
                    HOW MANY primitives from WHAT KIND: near 1.0 the prop is slabs and posts and
                    plain boxes will land on it, near 0 every part is oblique or round and each one
                    costs a rotation to place or a capsule to approximate.

    Not the count of connected components, which on artwork counts triangle islands -- a server rack
    reads 3447 of them -- and so measures how the model was built rather than what it is.

    Reported rather than acted on. A high count can still be the right prop to do first if a world
    places it fifty times, and that is not something this file can know.
    """
    pitch = max(2 * tol, 0.02)
    boxes = 0
    axis_area = total_area = 0.0
    for mesh in meshes:
        try:
            normals = np.asarray(mesh.face_normals)
            areas = np.asarray(mesh.area_faces)
            # within ~10 degrees of an axis: |n . axis| >= cos(10 deg)
            aligned = (np.abs(normals) >= 0.985).any(axis=1)
            axis_area += float(areas[aligned].sum())
            total_area += float(areas.sum())
        except Exception:  # noqa: BLE001 - a degenerate mesh contributes no area either way
            pass
        try:
            voxels = mesh.voxelized(pitch=pitch)
            occupied = np.zeros(voxels.shape, bool)
            occupied[tuple(np.array(voxels.sparse_indices).T)] = True
        except Exception:  # noqa: BLE001 - nothing to voxelize is nothing to build
            continue
        boxes += _greedy_boxes(occupied)
    return {
        "boxes_upper": boxes,
        "axis_frac": round(axis_area / total_area, 2) if total_area else 0.0,
        "effort": "small" if boxes <= 12 else "moderate" if boxes <= 40 else "large",
    }


def _greedy_boxes(occupied: np.ndarray) -> int:
    """Merge an occupancy grid into axis-aligned boxes, greedily. Only the count is wanted."""
    grid = occupied.copy()
    count = 0
    while grid.any():
        i, j, k = np.argwhere(grid)[0]
        di = 1
        while i + di < grid.shape[0] and grid[i + di, j, k]:
            di += 1
        dj = 1
        while j + dj < grid.shape[1] and grid[i : i + di, j + dj, k].all():
            dj += 1
        dk = 1
        while k + dk < grid.shape[2] and grid[i : i + di, j : j + dj, k + dk].all():
            dk += 1
        grid[i : i + di, j : j + dj, k : k + dk] = False
        count += 1
    return count


def _region(points: np.ndarray, distance: np.ndarray, tol: float) -> dict | None:
    """Where the WORST of the disagreement is: the box holding the worst tenth of the failures.

    An agent fixing a collision model needs a place to look, not only a magnitude: a coverage hole
    spanning z 0.10..0.35 at |x| ~ 0.55 names the missing part on sight. The worst tenth rather than
    everything past tolerance, because a prop always disagrees a little all over -- at a rounded
    corner, a castor, a chamfer -- and the box around ALL of that is just the prop's own bounding
    box, which locates nothing. Reported as an extent, since a defect is a part and not a point.
    """
    beyond = distance > tol
    if not beyond.any():
        return None
    cut = max(tol, float(np.percentile(distance[beyond], 90)))
    worst = points[distance >= cut]
    if len(worst) == 0:
        return None
    lo, hi = worst.min(axis=0), worst.max(axis=0)
    return {
        "count": int(beyond.sum()),
        "worst_beyond_mm": round(cut * 1000, 1),
        "min": [round(float(v), 3) for v in lo],
        "max": [round(float(v), 3) for v in hi],
    }


def _worst_geoms(
    model, ids: list[int], owners: np.ndarray, distance: np.ndarray, tol: float
) -> list:
    """Per-geom blame for the samples beyond tolerance, worst first, so the fix has an address."""
    beyond = distance > tol
    if not beyond.any():
        return []
    out = []
    for index, gid in enumerate(ids):
        mine = owners == index
        if not mine.any() or not (mine & beyond).any():
            continue
        out.append(
            {
                "geom": _name(model, gid),
                "share": float((mine & beyond).sum() / beyond.sum()),
                "max_mm": float(distance[mine].max() * 1000),
            }
        )
    return sorted(out, key=lambda r: -r["share"])


def _name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<unnamed geom {gid}>"


# -- diff ----------------------------------------------------------------------------------------


def diff(
    ref: str,
    tol: float = DEFAULT_TOL,
    samples: int = QUERY_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compare one model's collision geometry against its visual geometry, both ways."""
    trimesh, cKDTree = _deps()
    model, data = compile_model(ref)
    visual_ids, collision_ids = split_geoms(model)
    hulls = [g for g in collision_ids if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

    report = {
        "model": ref,
        "tolerance_mm": tol * 1000,
        "visual_geoms": len(visual_ids),
        "collision_geoms": len(collision_ids),
        "hull_colliders": [_name(model, g) for g in hulls],
    }

    visual = [
        m for m in (geom_surface(trimesh, model, data, g) for g in visual_ids) if m is not None
    ]
    collision = [
        m for m in (geom_surface(trimesh, model, data, g) for g in collision_ids) if m is not None
    ]

    if not visual:
        # Nothing to measure against. With a hull collider that is the FAIL below; without one it
        # means the model draws nothing, which is a different problem and not this tool's to judge.
        report["verdict"] = "FAIL" if hulls else "WARN"
        report["reason"] = (
            "the mesh collides, so physics sees its convex hull"
            if hulls
            else "no visual geometry to compare the collider against"
        )
        return report

    z_floor = min(float(m.vertices[:, 2].min()) for m in visual + collision)
    visual_pts, visual_owner, visual_n = sample_surfaces(
        trimesh, visual, samples, seed, normals=True
    )
    collision_pts, collision_owner, collision_n = sample_surfaces(
        trimesh, collision, samples, seed, normals=True
    )
    visual_pts, visual_owner = drop_base_faces(
        trimesh, visual, visual_pts, visual_owner, visual_n, z_floor
    )
    collision_pts, collision_owner = drop_base_faces(
        trimesh, collision, collision_pts, collision_owner, collision_n, z_floor
    )
    coverage, res_a = distance_to(trimesh, cKDTree, collision, visual_pts, seed)
    coverage = np.where(inside_any(model, data, collision_ids, visual_pts), 0.0, coverage)
    overreach, res_b = distance_to(trimesh, cKDTree, visual, collision_pts, seed)
    report["coverage"] = _stats(coverage, tol)
    report["overreach"] = _stats(overreach, tol)
    # Where, and whose fault. Coverage is blamed on a REGION (the missing part has no geom to name)
    # and overreach on the GEOMS that stand clear of the prop, which do have names.
    report["coverage"]["region"] = _region(visual_pts, coverage, tol)
    report["coverage"]["worst_visual"] = _worst_geoms(
        model, visual_ids, visual_owner, coverage, tol
    )
    report["overreach"]["region"] = _region(collision_pts, overreach, tol)
    report["overreach"]["worst_geoms"] = _worst_geoms(
        model, collision_ids, collision_owner, overreach, tol
    )
    report["resolution_mm"] = max(res_a, res_b) * 1000
    worst = max(report["coverage"]["beyond_tol"], report["overreach"]["beyond_tol"])
    report["agreement"] = 1.0 - worst
    if hulls:
        report["verdict"], report["reason"] = (
            "FAIL",
            "the mesh collides, so physics sees its convex hull",
        )
    elif worst > 0.05:
        report["verdict"] = "WARN"
        report["reason"] = f"the two surfaces disagree over {worst * 100:.1f}% of themselves"
    else:
        report["verdict"] = "ok"
        report["reason"] = (
            f"within {tol * 1000:.0f} mm over {(1 - worst) * 100:.1f}% of both surfaces"
        )
    return report


def print_diff(report: dict) -> None:
    print(f"collision diff: {report['model']}")
    print(
        f"  visual {report['visual_geoms']:4d} geoms   collision {report['collision_geoms']:4d} geoms"
        f"   tolerance {report['tolerance_mm']:.0f} mm"
    )
    for name in report["hull_colliders"]:
        print(f"  [FAIL] {name} is a mesh and collides -- physics sees its CONVEX HULL")
    for key, arrow in (
        ("coverage", "visual surface  -> nearest collider"),
        ("overreach", "collider surface -> nearest visual "),
    ):
        s = report.get(key)
        if not s:
            continue
        print(
            f"  {key:9s} {arrow}   mean {s['mean_mm']:5.1f} mm"
            f"  p95 {s['p95_mm']:6.1f} mm  max {s['max_mm']:6.1f} mm"
        )
        what = (
            "of the prop has no collider within tolerance"
            if key == "coverage"
            else "of the collider stands where the prop does not"
        )
        print(f"  {'':9s} {s['beyond_tol'] * 100:5.1f}% {what}")
        region = s.get("region")
        if region:
            lo, hi = region["min"], region["max"]
            print(
                f"  {'':9s} worst of it (>{region['worst_beyond_mm']:.0f} mm) in"
                f"  x {lo[0]:+.2f}..{hi[0]:+.2f}  y {lo[1]:+.2f}..{hi[1]:+.2f}"
                f"  z {lo[2]:+.2f}..{hi[2]:+.2f}"
            )
        for row in s.get("worst_geoms", [])[:4]:
            print(
                f"  {'':9s} {row['share'] * 100:5.1f}% of it is {row['geom']} (max {row['max_mm']:.0f} mm)"
            )
    if "resolution_mm" in report:
        print(f"  resolution ~{report['resolution_mm']:.1f} mm (surfaces are sampled, not solved)")
    print(f"  [{report['verdict']:4s}] {report['reason']}")


# -- propose -------------------------------------------------------------------------------------


def bands(trimesh, meshes: list, res: float, samples: int = 900_000, seed: int = DEFAULT_SEED):
    """Slice the prop into z-bands of constant plan footprint, each as axis-aligned rectangles.

    This is the shape of the MJCF somebody has to write: between these two heights the prop occupies
    these boxes. Bands are merged while their rectangle count and total width hold, so a post that
    runs the height of a desk comes out as one band rather than forty.

    ``res`` is the plan grid, and it is the whole accuracy story -- a 20 mm grid rounds a 25 mm post
    up to 40 and a tabletop out by 20, which is exactly how a first draft ends up 26 mm proud. Read
    the numbers, then measure the parts that matter; that is why this prints a draft and a warning
    rather than writing the file.
    """
    points, _ = sample_surfaces(trimesh, meshes, samples, seed)
    if len(points) == 0:
        return []
    lo, hi = points.min(axis=0), points.max(axis=0)
    steps = max(int((hi[2] - lo[2]) / res), 1)
    out, run, start = [], None, lo[2]
    for k in range(steps + 1):
        z = lo[2] + k * (hi[2] - lo[2]) / steps
        slab = points[(points[:, 2] >= z) & (points[:, 2] < z + (hi[2] - lo[2]) / steps)]
        rect = _plan_rectangles(slab, lo, res) if len(slab) > 5 else []
        key = (len(rect), round(sum(r[1] - r[0] for r in rect), 2))
        if run is not None and key != run[0]:
            out.append((start, z, run[1]))
            start = z
        run = (key, rect)
    if run is not None:
        out.append((start, hi[2], run[1]))
    return [b for b in out if b[2]]


def _plan_rectangles(points: np.ndarray, origin, res: float) -> list:
    """Greedy axis-aligned rectangles covering a slab's plan occupancy."""
    nx = int((points[:, 0].max() - origin[0]) / res) + 2
    ny = int((points[:, 1].max() - origin[1]) / res) + 2
    grid = np.zeros((nx, ny), bool)
    grid[
        ((points[:, 0] - origin[0]) / res).astype(int).clip(0, nx - 1),
        ((points[:, 1] - origin[1]) / res).astype(int).clip(0, ny - 1),
    ] = True
    out = []
    while grid.any():
        i, j = np.argwhere(grid)[0]
        di = 1
        while i + di < nx and grid[i + di, j]:
            di += 1
        dj = 1
        while j + dj < ny and grid[i : i + di, j + dj].all():
            dj += 1
        grid[i : i + di, j : j + dj] = False
        out.append(
            (
                origin[0] + i * res,
                origin[0] + (i + di) * res,
                origin[1] + j * res,
                origin[1] + (j + dj) * res,
            )
        )
    return out


def propose(ref: str, res: float = 0.02, max_geoms: int = 40, seed: int = DEFAULT_SEED) -> dict:
    """A first-draft skeleton for ``ref`` -- boxes, named by band, ready to be corrected."""
    trimesh, _ = _deps()
    model, data = compile_model(ref)
    visual_ids, collision_ids = split_geoms(model)
    ids = visual_ids + [
        g for g in collision_ids if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    meshes = [m for m in (geom_surface(trimesh, model, data, g) for g in ids) if m is not None]
    if not meshes:
        return {"model": ref, "bands": [], "geoms": [], "reason": "no mesh geometry to read"}
    found = bands(trimesh, meshes, res, seed=seed)
    points, _ = sample_surfaces(trimesh, meshes, 900_000, seed)
    name = Path(ref).stem
    geoms, count = [], 0
    for lo, hi, rects in found:
        for r in rects:
            count += 1
            if count > max_geoms:
                break
            box = _refine(points, r, lo, hi)
            if box is None:
                continue
            (x0, x1), (y0, y1), (z0, z1) = box
            geoms.append(
                {
                    "name": f"{name}_b{len(geoms):02d}",
                    "pos": [
                        round((x0 + x1) / 2, 4),
                        round((y0 + y1) / 2, 4),
                        round((z0 + z1) / 2, 4),
                    ],
                    "size": [
                        round((x1 - x0) / 2, 4),
                        round((y1 - y0) / 2, 4),
                        round((z1 - z0) / 2, 4),
                    ],
                }
            )
    return {
        "model": ref,
        "resolution_mm": res * 1000,
        "bands": len(found),
        "geoms": geoms,
        "truncated": count > max_geoms,
    }


def _refine(points: np.ndarray, rect, lo: float, hi: float):
    """Shrink a grid rectangle onto the geometry actually inside it.

    The grid is what makes the decomposition possible and what makes it wrong: at 20 mm it rounds a
    25 mm post up to 40 and a tabletop out by 20, and a draft carrying those numbers is a collision
    model that stands proud of the prop everywhere. Re-reading the samples inside each cell costs
    nothing and removes the rounding entirely, so the draft's numbers are measurements.
    """
    inside = points[
        (points[:, 0] >= rect[0])
        & (points[:, 0] < rect[1])
        & (points[:, 1] >= rect[2])
        & (points[:, 1] < rect[3])
        & (points[:, 2] >= lo)
        & (points[:, 2] <= hi)
    ]
    if len(inside) < 8:
        return None
    return [(float(inside[:, i].min()), float(inside[:, i].max())) for i in range(3)]


def print_propose(draft: dict) -> None:
    if not draft["geoms"]:
        print(f"collision propose: {draft['model']} -- {draft.get('reason', 'nothing to propose')}")
        return
    print(
        f"<!-- draft skeleton for {draft['model']}: {draft['bands']} bands found on a "
        f"{draft['resolution_mm']:.0f} mm plan grid,"
    )
    print("     each box then shrunk onto the geometry inside it, so these numbers are measured")
    print(
        "     rather than rounded. The DECOMPOSITION is still a guess -- a curved or diagonal part"
    )
    print("     comes out as a staircase of slivers -- so read it, do not paste it, and settle the")
    print("     result with `collision diff`. -->")
    for g in draft["geoms"]:
        print(f'<geom type="box" name="{g["name"]}" group="3"')
        print(
            f'      pos="{g["pos"][0]} {g["pos"][1]} {g["pos"][2]}"'
            f' size="{g["size"][0]} {g["size"][1]} {g["size"][2]}"/>'
        )
    if draft["truncated"]:
        print(f"<!-- truncated at {len(draft['geoms'])} geoms; a coarser --res gives fewer -->")


# -- audit ---------------------------------------------------------------------------------------


def audit_one(ref: str, tol: float, seed: int = DEFAULT_SEED) -> dict:
    """How far one model's collidable hulls stand from the surface they stand in for."""
    trimesh, cKDTree = _deps()
    try:
        model, data = compile_model(ref)
    except Exception as exc:  # noqa: BLE001 - a model that will not compile is a result, not a crash
        return {"model": ref, "verdict": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}

    _, collision_ids = split_geoms(model)
    meshes = [g for g in collision_ids if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if not meshes:
        return {
            "model": ref,
            "collider": "primitive",
            "verdict": "ok",
            "reason": "collision is primitives, not a hull",
        }

    surfaces = [m for m in (geom_surface(trimesh, model, data, g) for g in meshes) if m is not None]
    surfaces = [m for m in surfaces if len(m.vertices) >= 4]
    hulls = []
    for mesh in surfaces:
        try:
            hulls.append(mesh.convex_hull)
        except Exception:  # noqa: BLE001, S112 - a degenerate mesh has no hull; say so, don't crash
            continue
    if not hulls:
        return {"model": ref, "collider": "hull", "verdict": "ERROR", "reason": "degenerate mesh"}

    hull_pts, _hull_owner = sample_surfaces(trimesh, hulls, 40_000, seed)
    distance, _ = distance_to(trimesh, cKDTree, surfaces, hull_pts, seed)
    p95 = float(np.percentile(distance, 95) * 1000)
    peak = float(distance.max() * 1000)
    verdict = "FAIL" if p95 > tol * 1000 else "WARN" if peak > tol * 2000 else "ok"
    return {
        "model": ref,
        "collider": "hull",
        "phantom_p95_mm": p95,
        "phantom_max_mm": peak,
        "geoms": [_name(model, g) for g in meshes],
        **effort(trimesh, surfaces, tol),
        "verdict": verdict,
        "reason": (
            f"the hull stands up to {peak:.0f} mm off the prop"
            if verdict != "ok"
            else "the hull follows the prop"
        ),
    }


def provider_models(provider: str | None) -> list[str]:
    """Every model name a provider offers -- the nested ``<name>/<name>.xml`` layout props use."""
    from roqsim.models import providers

    found: list[str] = []
    for name, models_dir, _mesh, _tex in providers():
        if provider and name != provider:
            continue
        for path in sorted(Path(models_dir).glob("*/*.xml")):
            if path.stem == path.parent.name:
                found.append(path.stem)
        found += [p.stem for p in sorted(Path(models_dir).glob("*.xml"))]
    return sorted(dict.fromkeys(found))


def print_audit(rows: list[dict], tol: float) -> None:
    print(
        f"collision audit -- how far a hull collider stands from the prop (tolerance {tol * 1000:.0f} mm)"
    )
    print(
        f"{'model':32s} {'collider':>10s} {'phantom p95':>12s} {'max':>9s}"
        f" {'boxes':>6s} {'boxy':>5s} {'effort':>9s}  verdict"
    )
    for row in sorted(rows, key=lambda r: -r.get("phantom_p95_mm", -1)):
        p95 = f"{row['phantom_p95_mm']:9.0f} mm" if "phantom_p95_mm" in row else f"{'-':>12s}"
        peak = f"{row['phantom_max_mm']:6.0f} mm" if "phantom_max_mm" in row else f"{'-':>9s}"
        boxes = f"{row['boxes_upper']:6d}" if "boxes_upper" in row else f"{'-':>6s}"
        boxy = f"{row['axis_frac']:5.2f}" if "axis_frac" in row else f"{'-':>5s}"
        note = f" -- {row['reason']}" if row["verdict"] == "ERROR" else ""
        print(
            f"{row['model']:32s} {row.get('collider', '-'):>10s} {p95} {peak}"
            f" {boxes} {boxy} {row.get('effort', '-'):>9s}  {row['verdict']}{note}"
        )
    bad = sum(1 for r in rows if r["verdict"] == "FAIL")
    print(f"\n{bad} of {len(rows)} collide as a convex hull that is not the shape.")


# -- cli -----------------------------------------------------------------------------------------


def main(argv: list | None = None) -> int:
    # The shared flags are on a parent parser, so they are accepted on EITHER side of the
    # subcommand. argparse otherwise binds them to the top level only, and `collision diff x --json`
    # -- where anyone would naturally put it -- fails with "unrecognized arguments".
    # SUPPRESS, not a real default: a parent's default is re-applied by the subparser, so a value
    # given BEFORE the subcommand would be silently overwritten by the default given after it --
    # `--tol 30 audit` would quietly measure at 10. With SUPPRESS the attribute exists only where the
    # flag was actually typed, and the defaults are applied once, below.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="machine-readable output only",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help=f"sampling seed (default: {DEFAULT_SEED}); vary it to confirm a verdict is not one "
        "unlucky point set",
    )
    common.add_argument(
        "--tol",
        type=float,
        default=argparse.SUPPRESS,
        metavar="MM",
        help=f"agreement required between the surfaces, in mm (default: {DEFAULT_TOL * 1000:.0f})",
    )
    parser = argparse.ArgumentParser(
        prog="roqsim assets collision",
        description=__doc__.split("\n")[0],
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diff = sub.add_parser(
        "diff", parents=[common], help="compare one model's collider against its visual geometry"
    )
    p_diff.add_argument("model", help="model reference: <name>, <provider>:<name>, or a path")
    p_diff.add_argument(
        "--samples", type=int, default=QUERY_SAMPLES, metavar="N", help="points sampled per surface"
    )

    p_propose = sub.add_parser(
        "propose", parents=[common], help="a first-draft skeleton to correct, not to trust"
    )
    p_propose.add_argument("model", help="model reference: <name>, <provider>:<name>, or a path")
    p_propose.add_argument(
        "--res", type=float, default=20.0, metavar="MM", help="plan grid, in mm (default: 20)"
    )
    p_propose.add_argument(
        "--max-geoms", type=int, default=40, metavar="N", help="stop after N boxes (default: 40)"
    )

    p_audit = sub.add_parser(
        "audit",
        parents=[common],
        help="which models collide as a convex hull that is not the shape",
    )
    p_audit.add_argument(
        "models", nargs="*", help="model references (default: every one registered)"
    )
    p_audit.add_argument("--provider", help="limit the default set to one roqsim.models provider")

    args = parser.parse_args(argv)
    tol = getattr(args, "tol", DEFAULT_TOL * 1000) / 1000
    as_json = getattr(args, "json", False)

    if args.command == "diff":
        report = diff(args.model, tol=tol, samples=args.samples, seed=args.seed)
        print(json.dumps(report, indent=2)) if as_json else print_diff(report)
        return 1 if report["verdict"] in ("FAIL", "ERROR") else 0

    if args.command == "propose":
        draft = propose(args.model, res=args.res / 1000, max_geoms=args.max_geoms, seed=args.seed)
        print(json.dumps(draft, indent=2)) if as_json else print_propose(draft)
        return 0

    models = args.models or provider_models(args.provider)
    if not models:
        parser.error(
            "no models to audit" + (f" for provider {args.provider!r}" if args.provider else "")
        )
    rows = [audit_one(ref, tol, args.seed) for ref in models]
    print(json.dumps(rows, indent=2)) if as_json else print_audit(rows, tol)
    return 1 if any(r["verdict"] in ("FAIL", "ERROR") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
