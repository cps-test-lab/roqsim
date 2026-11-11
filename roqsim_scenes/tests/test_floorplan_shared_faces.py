"""No two drawn parts of a world `roqsim scenes floorplan-to-world` generates show a face in one place
(roqsim_assets.shared_faces), wherever its walls meet: such a pair flickers in every render as the
camera moves, most visibly on the wall tops seen from above."""

from __future__ import annotations

import contextlib
import io

import mujoco
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim_assets.shared_faces import shared_faces
from roqsim_scenes.cli import floorplan_to_world as fw


def _line(i, a, b):
    return {"id": i, "x0_m": a[0], "y0_m": a[1], "x1_m": b[0], "y1_m": b[1]}


ROOM = [
    _line(1, (0, 0), (4, 0)),
    _line(2, (4, 0), (4, 3)),
    _line(3, (4, 3), (0, 3)),
    _line(4, (0, 3), (0, 0)),
]

FLOORPLANS = {
    "l_corner": {"lines": ROOM[:2]},
    "room": {"lines": ROOM},
    "t_junction_on_a_wall": {"lines": [_line(1, (0, 0), (4, 0)), _line(2, (2, 0), (2, 3))]},
    "t_junction_of_three_ends": {
        "lines": [_line(1, (0, 0), (2, 0)), _line(2, (2, 0), (4, 0)), _line(3, (2, 0), (2, 3))]
    },
    "three_walls_at_a_point": {
        "lines": [_line(1, (0, 0), (3, 0)), _line(2, (0, 0), (0, 3)), _line(3, (0, 0), (-2, -2))]
    },
    "four_walls_at_a_point": {
        "lines": [
            _line(1, (0, 0), (2, 0)),
            _line(2, (0, 0), (-2, 0)),
            _line(3, (0, 0), (0, 2)),
            _line(4, (0, 0), (0, -2)),
        ]
    },
    "walls_crossing": {"lines": [_line(1, (-2, 0), (2, 0)), _line(2, (0, -2), (0, 2))]},
    "walls_in_line": {"lines": [_line(1, (0, 0), (2, 0)), _line(2, (2, 0), (4, 0))]},
    "angled_room": {
        "lines": [
            _line(1, (0, 0), (4, 0)),
            _line(2, (4, 0), (5, 2.5)),
            _line(3, (5, 2.5), (1, 3.2)),
            _line(4, (1, 3.2), (0, 0)),
        ]
    },
    "acute_corner": {"lines": [_line(1, (0, 0), (3, 0)), _line(2, (3, 0), (1, 2))]},
    "door_with_a_lintel": {
        "lines": ROOM,
        "doors": [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}],
    },
    "door_at_a_corner": {
        "lines": ROOM,
        "doors": [{"id": 1, "line_id": 1, "t": 1.0, "width_m": 0.9}],
    },
    "lintels_meeting_at_a_corner": {
        "lines": ROOM,
        "doors": [
            {"id": 1, "line_id": 1, "t": 1.0, "width_m": 0.9},
            {"id": 2, "line_id": 2, "t": 0.0, "width_m": 0.9},
        ],
    },
    "door_in_a_t_junction_stem": {
        "lines": [_line(1, (0, 0), (4, 0)), _line(2, (2, 0), (2, 3))],
        "doors": [{"id": 1, "line_id": 2, "t": 0.0, "width_m": 0.9}],
    },
}


def _shared_faces(tmp_path, floorplan, *, ceiling=False, doors_map=None):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        world = fw.generate(
            floorplan,
            tmp_path / "scene",
            "plan",
            tmp_path / "worlds" / "plan.yaml",
            {},
            ceiling_h=2.5,
            wall_thickness=0.1,
            opening_h=2.0,
            doors_map=doors_map,
            ceiling=ceiling,
        )
    engine = Engine(load_config(world))
    engine.setup()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        mujoco.mj_kinematics(model, data)  # setup compiles; the pose to compare is computed here
        return [f.describe() for f in shared_faces(model, data)]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("name", FLOORPLANS)
def test_walls_meeting_show_no_face_in_one_place(tmp_path, name):
    # Two doors' frames at one corner overlap each other whatever the walls do; this compares the
    # walls and lintels, so those leaves are left out.
    doors_map = (
        {"1": {"skip": True}, "2": {"skip": True}} if name == "lintels_meeting_at_a_corner" else {}
    )
    assert _shared_faces(tmp_path, FLOORPLANS[name], doors_map=doors_map) == []


def test_a_roofed_room_with_a_door_shows_no_face_in_one_place(tmp_path):
    assert _shared_faces(tmp_path, FLOORPLANS["door_with_a_lintel"], ceiling=True) == []
