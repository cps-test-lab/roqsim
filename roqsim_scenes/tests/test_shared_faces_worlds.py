"""No two drawn parts of a world this package ships show a face in one place, as the simulator
builds it (roqsim_assets.shared_faces): such a pair flickers in every render as the camera moves."""

from __future__ import annotations

from pathlib import Path

import pytest

import roqsim_scenes
from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim_assets.shared_faces import shared_faces

WORLDS = sorted((Path(roqsim_scenes.__file__).parent / "worlds").glob("*.yaml"))


def test_there_are_worlds_to_compare():
    assert WORLDS


@pytest.mark.parametrize("world", WORLDS, ids=lambda p: p.stem)
def test_no_two_drawn_parts_of_a_world_show_a_face_in_one_place(world):
    engine = Engine(load_config(world))
    engine.setup()
    try:
        found = shared_faces(engine.ctx.model, engine.ctx.data)
    finally:
        engine.shutdown()
    assert [f.describe() for f in found] == []
