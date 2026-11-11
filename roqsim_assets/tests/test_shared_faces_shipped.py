"""No two drawn parts of a prop or world this package ships, or of the default room, show a face in
one place.

inspect-prop reaches only a prop with an OBJ; a hand-written MJCF of primitives never passes through
it, so every prop folder is compared here, and every world, as the simulator builds it; a new one is
caught when it is added. Robots are left to their own packages: their meshes run to hundreds of
thousands of faces, too many to compare on every test run.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim.models import apply_assets, resolve_model
from roqsim.world import build_world
from roqsim_assets.models import MODELS_DIR
from roqsim_assets.shared_faces import shared_faces

PROPS = sorted(p.name for p in MODELS_DIR.iterdir() if (p / f"{p.name}.xml").is_file())
WORLDS = sorted((MODELS_DIR.parent / "worlds").glob("*.yaml"))


def test_there_are_props_and_worlds_to_compare():
    assert {"conveyor", "door_glass", "humanoid_gantry"} <= set(PROPS)
    assert WORLDS


@pytest.mark.parametrize("name", PROPS)
def test_no_two_drawn_parts_of_a_prop_show_a_face_in_one_place(name):
    asset = resolve_model(f"roqsim_assets:{name}")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    assert [f.describe() for f in shared_faces(spec.compile())] == []


def test_the_default_room_s_walls_meet_without_a_shared_face():
    spec = mujoco.MjSpec()
    build_world(spec, None)
    assert [f.describe() for f in shared_faces(spec.compile())] == []


@pytest.mark.parametrize("world", WORLDS, ids=lambda p: p.stem)
def test_no_two_drawn_parts_of_a_world_show_a_face_in_one_place(world):
    engine = Engine(load_config(world))
    engine.setup()
    try:
        found = shared_faces(engine.ctx.model, engine.ctx.data)
    finally:
        engine.shutdown()
    assert [f.describe() for f in found] == []
