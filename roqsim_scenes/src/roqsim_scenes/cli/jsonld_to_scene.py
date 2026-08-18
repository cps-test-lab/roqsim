"""Floorplan-DSL json-ld + its mesh -> scene.json (+ world-space OBJs), collision from the structure.

A scenery_builder / Floorplan-DSL environment ships two views of one building::

    <env>/3d-mesh/<name>.stl     the fused surface mesh -- what a room LOOKS like
    <env>/json-ld/*.json         every wall and column as an exact convex polyhedron

Importing only the mesh is the failure this tool exists to prevent. **MuJoCo collides a mesh by its
convex hull**, and a wall with a door cut out of it is not convex: its hull is the solid wall. The
doorway then stays open to every renderer and to ``mj_ray``/``mj_multiRay`` (both test the real
triangles) while being a solid slab to physics -- so the world looks right in every picture, reads as
passable on ``/scan``, and stops the robot dead. secorolab was imported that way and 87% of the
building was unreachable through eight walled doorways; four campaigns ran against it before anyone
noticed, because nothing anyone looked at could show it.

So the two views are split by role, which is what ``scene.json`` already expresses per object:

* the **mesh** becomes one visual object (``collide: false``) -- the room's real appearance, and the
  surface a lidar sees, doorways included;
* the **json-ld** becomes the collision set (``render: false``): one convex box per wall segment with
  door openings subtracted, plus columns as-is, via :func:`roqsim.floorplan_collision.wall_colliders`.

Doorways are therefore the *absence* of a collider rather than a hole in one, and no hull can close
them. Windows stay solid (their sill is above the floor, so the wall still blocks a ground robot).

Typical use, then the shared stage 2::

    roqsim scenes jsonld-to-scene --mesh <env>/3d-mesh/<name>.stl --out-dir <env>/mujoco
    roqsim scenes scene-to-mjcf --scene <env>/mujoco/scene.json --out <env>/mujoco/<name>.xml

The alternative to baking is the ``floorplan`` plugin (``roqsim_mobile.plugins.floorplan``), which reads
the same json-ld at run time and needs no scene at all. Bake when the world should be one
self-contained artifact -- a campaign staging files into a container, a run view compiling the scene,
``roqsim scenes describe`` answering without the source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from roqsim.floorplan_collision import wall_colliders
from roqsim_scenes import scene_mesh_io as mio

#: Collider colour. Never rendered (``render: false`` puts them in group 3), but a debugger toggling
#: the group on wants them distinguishable from the visual mesh.
_COLLIDER_RGBA = [0.8, 0.35, 0.35, 1.0]
_VISUAL_RGBA = [0.7, 0.7, 0.72, 1.0]


def _box_faces() -> np.ndarray:
    """Triangles for the 8 corners :func:`wall_colliders` emits, ordered ``x`` outer, ``y``, ``z``.

    The vertex order is fixed by ``floorplan_collision._wall_boxes`` (``for x for y for z``), so the
    faces can be stated once rather than hulled per box.
    """
    # bit 2 = x, bit 1 = y, bit 0 = z
    quads = [
        (0, 1, 3, 2),  # x = lo
        (4, 6, 7, 5),  # x = hi
        (0, 4, 5, 1),  # y = lo
        (2, 3, 7, 6),  # y = hi
        (0, 2, 6, 4),  # z = lo
        (1, 5, 7, 3),  # z = hi
    ]
    return np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)


def _faces_for(verts: np.ndarray) -> np.ndarray:
    """Triangles enclosing *verts*: the fixed box winding for 8 corners, else a convex hull.

    A column is an arbitrary convex polyhedron, so it needs the hull. MuJoCo rebuilds the hull at
    compile time either way -- these faces only have to make the OBJ a closed solid, so that anything
    reading it back (a viewer, ``scene-to-map``, the passability check in ``sdf-to-scene``) sees the
    same volume MuJoCo will collide.
    """
    if len(verts) == 8:
        return _box_faces()
    from scipy.spatial import ConvexHull  # local: only the non-box case needs scipy

    return ConvexHull(verts).simplices.astype(np.int64)


def build(mesh: Path, out_dir: Path, scene_name: str | None = None) -> dict:
    """Write ``out_dir/scene.json`` + ``out_dir/meshes/`` for the floorplan whose mesh is *mesh*."""
    colliders = wall_colliders(str(mesh))
    if not colliders:
        # Loudly, not as visual-only walls: a scene with no collision geometry is a building a robot
        # drives straight through, which no downstream check would flag as an import failure.
        raise SystemExit(
            f"{mesh}: no wall colliders came out of the json-ld beside it (expected "
            f"{mesh.parent.parent / 'json-ld'}/*.json).\n"
            f"That directory is the whole point of this importer -- without it there is nothing to "
            f"collide with, and baking the mesh instead would hull every doorway shut.\n"
            f"If this environment genuinely has no json-ld, it is not a Floorplan-DSL room: import it "
            f"with `roqsim scenes sdf-to-scene` and read that tool's hull warnings."
        )

    name = scene_name or mesh.stem
    meshes = out_dir / "meshes"
    objects = []

    visual = f"{name}_visual"
    mio.write_obj(meshes / f"{visual}.obj", *_read_visual(mesh))
    objects.append(
        {
            "name": visual,
            "mesh": f"meshes/{visual}.obj",
            "rgba": _VISUAL_RGBA,
            "collide": False,
            "render": True,
        }
    )

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i, verts in enumerate(colliders):
        part = f"{name}_collider_{i:03d}"
        mio.write_obj(meshes / f"{part}.obj", verts, _faces_for(verts))
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
        objects.append(
            {
                "name": part,
                "mesh": f"meshes/{part}.obj",
                "rgba": _COLLIDER_RGBA,
                "collide": True,
                "render": False,
            }
        )

    manifest = {
        "name": name,
        "source": os.path.relpath(mesh, out_dir),
        "unit_scale": 1.0,
        # From the COLLIDERS, not the visual mesh: the bounds size the ground plane and a floorplan
        # mesh may carry a base slab or overhang that the walls do not, which would grow the floor
        # past the building.
        "bounds_min": [float(v) for v in lo],
        "bounds_max": [float(v) for v in hi],
        "objects": objects,
        "ground_z": float(lo[2]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scene.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _read_visual(mesh: Path) -> tuple[np.ndarray, np.ndarray]:
    """The floorplan mesh as one world-space vertex/face pair.

    Re-written rather than referenced so the scene dir is self-contained in the units and frame stage
    2 expects, like every other stage-1 front-end.
    """
    subs = mio.read_mesh(mesh)
    if not subs:
        raise SystemExit(f"{mesh}: no geometry")
    verts = np.vstack([s.verts for s in subs])
    faces, offset = [], 0
    for s in subs:
        faces.append(np.asarray(s.faces) + offset)
        offset += len(s.verts)
    return verts, np.vstack(faces)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="floorplan mesh (<env>/3d-mesh/<name>.stl); the json-ld is found beside it",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="scene dir to write (scene.json + meshes/)"
    )
    ap.add_argument("--scene-name", default=None, help="scene name (default: the mesh's stem)")
    args = ap.parse_args(argv)

    if not args.mesh.exists():
        raise SystemExit(f"{args.mesh}: no such file")
    manifest = build(args.mesh, args.out_dir, args.scene_name)

    n_col = sum(1 for o in manifest["objects"] if o["collide"])
    lo, hi = manifest["bounds_min"], manifest["bounds_max"]
    print(
        f"SCENE_OK {manifest['name']}: 1 visual mesh + {n_col} convex colliders, "
        f"bounds [{lo[0]:.2f}, {lo[1]:.2f}] .. [{hi[0]:.2f}, {hi[1]:.2f}] m -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
