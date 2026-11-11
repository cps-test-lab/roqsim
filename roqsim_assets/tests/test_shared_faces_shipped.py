"""No two drawn parts of a prop roqsim ships, or of the default room, show a face in one place.

inspect-prop reaches only a prop with an OBJ; a hand-written MJCF of primitives never passes through
it, so every prop folder is compared here, and a new one is caught when it is added. Robots are
left to their own packages: their meshes run to hundreds of thousands of faces, which makes the
comparison take from seconds to over ten minutes per robot.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.models import apply_assets, resolve_model
from roqsim.world import build_world
from roqsim_assets.models import MODELS_DIR
from roqsim_assets.shared_faces import shared_faces

PROPS = sorted(p.name for p in MODELS_DIR.iterdir() if (p / f"{p.name}.xml").is_file())


def test_there_are_props_to_compare():
    assert {"conveyor", "door_glass", "humanoid_gantry"} <= set(PROPS)


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
