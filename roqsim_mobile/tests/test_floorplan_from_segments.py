# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``floorplan`` built from wall segments: a layout as the world, with no mesh anywhere.

The mesh source is right when the building already exists as geometry. This one is right when the
walls are the experiment's variable -- a corridor width becomes an ordinary config value a sweep
varies and the run's provenance records, instead of a file baked ahead of time that nothing
downstream can tell apart from another file.

One plugin, not two, because a floorplan is a floorplan whichever way it arrives. These tests are
about what a world author can state, and about the two pieces of arithmetic that are quiet when
they are wrong.
"""

from __future__ import annotations

import json
import math

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

ROOM = [
    {"id": 0, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 6.0, "y1_m": 0.0},
    {"id": 1, "x0_m": 6.0, "y0_m": 0.0, "x1_m": 6.0, "y1_m": 4.0},
    {"id": 2, "x0_m": 6.0, "y0_m": 4.0, "x1_m": 0.0, "y1_m": 4.0},
    {"id": 3, "x0_m": 0.0, "y0_m": 4.0, "x1_m": 0.0, "y1_m": 0.0},
]


def _world(**config):
    engine = Engine(
        load_config_from_dict({"sim": {}, "components": [{"floorplan": config}]}), preview=True
    )
    engine.setup()
    return engine


def _geoms(engine) -> dict:
    m = engine.ctx.model
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(m.ngeom)}


def _walls(engine) -> list:
    return [n for n in _geoms(engine) if n and n.startswith("floorplan_wall_")]


def test_a_layout_becomes_a_world_with_no_mesh():
    """The capability: geometry the simulator compiles, not a file baked beforehand."""
    engine = _world(lines=ROOM)

    assert engine.ctx.model.nmesh == 0
    assert len(_walls(engine)) == 4
    assert "floor" in _geoms(engine), "the same named ground plane the mesh source provides"


def test_a_wall_built_from_a_segment_is_solid():
    """Unlike the mesh source there is nothing to hide behind: a box is already convex, so what is
    drawn is what is collided with -- no invisible companion collider, and no way for the two to
    disagree."""
    engine = _world(lines=ROOM)
    m = engine.ctx.model
    g = _geoms(engine)["floorplan_wall_0"]

    assert m.geom_contype[g] != 0 and m.geom_conaffinity[g] != 0
    assert m.geom_rgba[g][3] > 0.0, "and it is visible, not a hidden collider"


def test_a_door_is_a_hole_with_a_beam_over_it():
    """Not a gap: the wall above an opening is built, so a room stays enclosed above head height."""
    engine = _world(lines=ROOM, doors=[{"line_id": 0, "t": 0.5, "width_m": 0.9}])

    assert len(_walls(engine)) == 6, "five solid pieces and one lintel"


def test_a_full_height_opening_keeps_a_true_doorway():
    engine = _world(
        lines=ROOM, height=2.0, opening_height=2.0,
        doors=[{"line_id": 0, "t": 0.5, "width_m": 0.9}],
    )

    assert len(_walls(engine)) == 5, "no beam where the opening reaches the ceiling"


def test_a_wall_is_rotated_onto_its_segment():
    """The arithmetic that is easy to get silently wrong.

    A wall along +y built unrotated is a wall along +x in the wrong place, and a plan view still
    looks like a room until something drives through it.
    """
    engine = _world(lines=[{"id": 0, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 0.0, "y1_m": 4.0}])
    rot = engine.ctx.model.geom_quat[_geoms(engine)["floorplan_wall_0"]]

    assert math.isclose(abs(rot[3]), math.sin(math.pi / 4), abs_tol=1e-6)


def test_the_ground_plane_is_fitted_to_the_walls():
    """The mesh source sizes the floor to the building's footprint, and so must this one -- an
    origin-centred default would put the room in one quadrant of an oversized plane."""
    engine = _world(lines=ROOM)
    m = engine.ctx.model
    floor = _geoms(engine)["floor"]

    assert list(m.geom_pos[floor])[:2] == pytest.approx([3.0, 2.0], abs=0.2)


def test_a_layout_can_come_from_the_file_the_sketch_tools_write(tmp_path):
    path = tmp_path / "rooms.json"
    path.write_text(json.dumps({"lines": ROOM, "doors": []}), encoding="utf-8")

    assert len(_walls(_world(floorplan=str(path)))) == 4


@pytest.mark.parametrize("config,expected", [
    ({}, "exactly one source"),
    ({"lines": ROOM, "mesh": "x.stl"}, "exactly one source"),
    ({"lines": [], "doors": []}, "'lines' is empty"),
    ({"lines": ROOM, "height": 1.0, "opening_height": 2.0}, "taller than"),
    ({"lines": ROOM, "thickness": 0.0}, "'thickness' must be > 0"),
])
def test_an_unusable_floorplan_is_refused_at_load(config, expected):
    """Before a world is built, and naming the key. Naming BOTH sources is refused too: there is
    no rule for which would win, and picking one silently is how a world stops meaning what it
    says."""
    with pytest.raises(Exception, match=expected):
        _world(**config)


def test_a_missing_floorplan_file_says_so():
    with pytest.raises(Exception, match="does not exist"):
        _world(floorplan="nowhere/rooms.json")
