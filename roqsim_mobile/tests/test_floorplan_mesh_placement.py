# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A floorplan mesh and its json-ld colliders are placed together.

The mesh is what the renderer draws and the lidar sees; the colliders, read from the json-ld beside
it, are what the robot hits. Both state the walls in one frame, so ``mesh_scale`` and ``mesh_pos``
have to move both, or a world that sets either has walls the lidar sees in one place and walls the
robot bumps into in another.
"""

from __future__ import annotations

import json
import struct

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_mobile.plugins.floorplan import FloorplanPlugin

#: One wall, as an axis-aligned box (lo, hi): the json-ld collider and the mesh are the same box.
LO, HI = (0.0, 0.0, 0.0), (2.0, 0.1, 1.0)


def _box_corners(lo, hi):
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def _jsonld(env):
    """A one-wall json-ld graph: a frame at the world origin and the wall's eight corners in it."""
    frame, ids, graph = "wall-frame", [], []
    for i, (x, y, z) in enumerate(_box_corners(LO, HI)):
        ids.append(f"w:wall-corner-{i}")
        graph.append(
            {
                "@id": f"w:coord-position-wall-corner-{i}-wrt-{frame}",
                "@type": ["PositionReference", "PositionCoordinate", "VectorXYZ"],
                "as-seen-by": f"w:{frame}",
                "of-position": f"w:position-wall-corner-{i}-wrt-{frame}",
                "x": x,
                "y": y,
                "z": z,
            }
        )
    graph.append(
        {
            "@id": "w:pose-coord-wall",
            "@type": ["PoseReference", "PoseCoordinate", "VectorXYZ", "EulerAngles"],
            "as-seen-by": "w:world-frame",
            "of-pose": f"w:pose-{frame}-wrt-world-frame",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "alpha": 0.0,
        }
    )
    graph.append({"@id": "w:wall-polyhedron", "@type": "Polyhedron", "points": ids})
    jdir = env / "json-ld"
    jdir.mkdir(parents=True)
    (jdir / "floorplan.fpm.json").write_text(json.dumps({"@context": {}, "@graph": graph}))


def _stl(path):
    """The same box as a closed binary STL."""
    c = _box_corners(LO, HI)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, cc, d in quads for t in ((c[a], c[b], c[cc]), (c[a], c[cc], c[d]))]
    path.parent.mkdir(parents=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0, 0, 1))
            for v in t:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))


@pytest.fixture
def mesh(tmp_path):
    env = tmp_path / "env"
    _jsonld(env)
    path = env / "3d-mesh" / "wall.stl"
    _stl(path)
    return str(path)


def _world_aabb(model, data, geom):
    """World-frame (lo, hi) of a geom, from its compiled local AABB."""
    centre, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
    rot = data.geom_xmat[geom].reshape(3, 3)
    corners = [
        data.geom_xpos[geom] + rot @ (centre + half * np.array(s))
        for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    ]
    return np.min(corners, axis=0), np.max(corners, axis=0)


def _bounds(**placement):
    cfg = {"floorplan": {**placement}}
    engine = Engine(load_config_from_dict({"sim": {}, "components": [cfg]}), preview=True)
    engine.setup()
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)

    def geom(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

    return (
        _world_aabb(m, d, geom("floorplan")),
        _world_aabb(m, d, geom("floorplan_col_0")),
        d.geom_xpos[geom("floor")][:2].copy(),
    )


@pytest.mark.parametrize(
    "placement",
    [{}, {"mesh_pos": [10.0, -3.0, 0.0]}, {"mesh_scale": 2.0}, {"mesh_scale": [2.0, 1.0, 1.0]}],
    ids=["unplaced", "offset", "scaled", "scaled-per-axis"],
)
def test_the_colliders_stand_where_the_mesh_is(mesh, placement):
    (vis_lo, vis_hi), (col_lo, col_hi), floor_xy = _bounds(mesh=mesh, **placement)

    np.testing.assert_allclose(col_lo, vis_lo, atol=1e-6)
    np.testing.assert_allclose(col_hi, vis_hi, atol=1e-6)
    # The ground plane is centred on the placed walls, not on where the json-ld states them.
    np.testing.assert_allclose(floor_xy, (vis_lo[:2] + vis_hi[:2]) / 2.0, atol=1e-6)


@pytest.mark.parametrize("key, value", [("mesh_scale", 2.0), ("mesh_pos", [1.0, 0.0, 0.0])])
def test_a_floorplan_from_segments_refuses_a_mesh_placement(key, value):
    lines = [{"id": 0, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 6.0, "y1_m": 0.0}]
    errors = FloorplanPlugin().validate_config({"lines": lines, key: value})

    assert any(key in e and "segments" in e for e in errors), errors


def test_a_mesh_pos_that_is_not_three_numbers_is_refused(mesh):
    errors = FloorplanPlugin().validate_config({"mesh": mesh, "mesh_pos": [1.0, 2.0]})

    assert any("'mesh_pos'" in e for e in errors), errors
