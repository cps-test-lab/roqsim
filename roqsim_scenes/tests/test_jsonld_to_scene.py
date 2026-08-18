"""Importing a Floorplan-DSL room: the doorway must survive as absent geometry.

The whole point of this front-end is that it does NOT bake the fused mesh. A wall with a door cut out
of it is non-convex, and MuJoCo collides a mesh by its convex hull, so baking it makes the doorway a
solid slab that every renderer and ``mj_ray`` still draw as open. Here the walls come from the
json-ld instead -- one convex box per wall segment, the doorway being the gap between two of them.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from roqsim_scenes.cli import jsonld_to_scene
from roqsim_scenes.passability import closed_passages


def _corners(prefix, frame, lo, hi):
    """Eight ``PositionCoordinate`` nodes for an axis-aligned box, plus the ids that name them."""
    ids, nodes = [], []
    for i, (x, y, z) in enumerate(
        [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    ):
        cid = f"w:{prefix}-corner-{i}"
        ids.append(cid)
        nodes.append(
            {
                "@id": f"w:coord-position-{prefix}-corner-{i}-wrt-{frame}",
                "@type": ["PositionReference", "PositionCoordinate", "VectorXYZ"],
                "as-seen-by": f"w:{frame}",
                "of-position": f"w:position-{prefix}-corner-{i}-wrt-{frame}",
                "x": x,
                "y": y,
                "z": z,
            }
        )
    return ids, nodes


def _part(prefix, lo, hi):
    """A ``Polyhedron`` plus the frame pose and corner coordinates it needs, all wrt the world."""
    frame = f"{prefix}-frame"
    ids, nodes = _corners(prefix, frame, lo, hi)
    nodes.append(
        {
            "@id": f"w:pose-coord-{prefix}",
            "@type": ["PoseReference", "PoseCoordinate", "VectorXYZ", "EulerAngles"],
            "as-seen-by": "w:world-frame",
            "of-pose": f"w:pose-{frame}-wrt-world-frame",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "alpha": 0.0,
        }
    )
    nodes.append({"@id": f"w:{prefix}-polyhedron", "@type": "Polyhedron", "points": ids})
    return nodes


def _floorplan(env, gap=0.9, s=4.0, t=0.1, h=2.2):
    """A four-wall room with one doorway on the north wall, written as ``<env>/json-ld/``."""
    graph = []
    graph += _part("north-wall", (-s, s - t, 0), (s, s, h))
    graph += _part("south-wall", (-s, -s, 0), (s, -s + t, h))
    graph += _part("west-wall", (-s, -s, 0), (-s + t, s, h))
    graph += _part("east-wall", (s - t, -s, 0), (s, s, h))
    # An opening through the north wall. `door` in the id is how the reader classifies it, and its
    # polyhedron is the void to subtract -- so it must span the wall's thickness and height.
    graph += _part("north-door", (-gap / 2, s - t, 0), (gap / 2, s, h))
    jdir = env / "json-ld"
    jdir.mkdir(parents=True)
    (jdir / "floorplan.fpm.json").write_text(json.dumps({"@context": {}, "@graph": graph}))


def _stl(path, tris):
    """A minimal binary STL, so the importer has a visual mesh to carry through."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0, 0, 1))
            for v in t:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))


def _room(tmp_path, **kw):
    env = tmp_path / "secorolab_ish"
    _floorplan(env, **kw)
    mesh = env / "3d-mesh" / "room.stl"
    _stl(mesh, [[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0), (0, 1, 0)]])
    return mesh, env


def _collidable(manifest, out):
    from roqsim_scenes import scene_mesh_io as mio

    parts = []
    for o in manifest["objects"]:
        if not o["collide"]:
            continue
        subs = mio.read_mesh(out / o["mesh"])
        verts = np.vstack([s.verts for s in subs])
        faces, off = [], 0
        for s in subs:
            faces.append(np.asarray(s.faces) + off)
            off += len(s.verts)
        parts.append((verts, np.vstack(faces)))
    return parts


def test_the_doorway_is_a_gap_between_colliders_not_a_hole_in_one(tmp_path):
    mesh, _ = _room(tmp_path)
    out = tmp_path / "mujoco"
    manifest = jsonld_to_scene.build(mesh, out)

    # The north wall came apart at the doorway: two colliders where the json-ld had one wall.
    assert len(_collidable(manifest, out)) == 5, "4 walls, the north one split in two by the door"
    assert closed_passages(_collidable(manifest, out)) == []


def test_the_mesh_is_carried_as_visual_only(tmp_path):
    """The fused mesh must never collide -- its hull is the building.

    This is the one flag that separates this route from the failure it exists to prevent, so it is
    asserted rather than left to the reader of scene.json.
    """
    mesh, _ = _room(tmp_path)
    manifest = jsonld_to_scene.build(mesh, tmp_path / "mujoco")

    visual = [o for o in manifest["objects"] if o["render"] and not o["collide"]]
    assert len(visual) == 1
    assert all(not o["render"] for o in manifest["objects"] if o["collide"])


def test_a_mesh_with_no_json_ld_is_refused_not_degraded(tmp_path):
    """Silence here would be a building a robot drives straight through.

    ``wall_colliders`` returns [] and warns when the json-ld is missing, which suits a run-time plugin
    that can fall back to visual-only walls. A baked scene has no such fallback: it would simply have
    no collision geometry, and nothing downstream would call that an import failure.
    """
    import pytest

    mesh = tmp_path / "3d-mesh" / "room.stl"
    _stl(mesh, [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]])
    with pytest.raises(SystemExit, match="json-ld"):
        jsonld_to_scene.build(mesh, tmp_path / "mujoco")
