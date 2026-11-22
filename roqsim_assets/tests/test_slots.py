"""The slots component states slots on a body as sites, by the convention the moves read."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError


def _world(slots, body="rack_box"):
    return {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "box": {
                    "prefix": "rack_",
                    "pose": {"position": {"x": 2.0, "y": 0.0}, "orientation": {"yaw": math.pi / 2}},
                    "size": [1.2, 0.6, 0.9],
                    "motion": "static",
                },
                "name": "rack",
                "components": [{"slots": {"body": body, "slots": slots}}],
            },
        ],
    }


def _compiled(world):
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _frame(d, name):
    s = d.site(name)
    return s.xpos.copy(), s.xmat.reshape(3, 3).copy()


def test_a_slot_is_a_hand_target_and_its_stances_in_the_bodys_frame():
    engine = _compiled(
        _world(
            {
                "bin_0": {
                    "kind": "grip",
                    "at": [0.0, -0.1, 0.55, 0.0],
                    "stand": [[0.0, -0.9, 1.5708]],
                },
                "tray": {"kind": "place", "at": [0.2, 0.0, 0.9, 0.5]},
                "go": {"kind": "press", "at": [0.0, -0.31, 0.8, -1.5708]},
                "two": {"stand": [[0.0, -0.9, 1.5708], [0.0, 0.9, -1.5708]]},
            }
        )
    )
    try:
        d = engine.ctx.data
        pos, rot = _frame(d, "rack/bin_0/grip")
        # the box is turned a quarter turn (its -y is the world's +x), its origin at its centre
        assert np.allclose(pos, [2.1, 0.0, 1.0], atol=1e-6), pos
        assert np.allclose(rot[:, 2], [0, 0, -1], atol=1e-6)  # +z along the fingers, down
        assert np.allclose(rot[:, 0], [0, 1, 0], atol=1e-6)  # +x: the slot's yaw, in the world
        pos, rot = _frame(d, "rack/bin_0/stand")
        assert np.allclose(pos[:2], [2.9, 0.0], atol=1e-6)
        assert np.allclose(rot[:, 0], [-1, 0, 0], atol=1e-4)  # facing back toward it
        assert np.allclose(_frame(d, "rack/tray")[1][:, 2], [0, 0, 1], atol=1e-6)
        assert d.site("rack/go") is not None
        names = {engine.ctx.model.site(i).name for i in range(engine.ctx.model.nsite)}
        assert {"rack/two/stand_0", "rack/two/stand_1"} <= names
        assert "rack/two/stand" not in names
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("slots", "says"),
    [
        ({"a": {"kind": "hold", "at": [0, 0, 1]}}, "'kind' is one of"),
        ({"a": {"at": [0, 0]}}, "'at' is [x, y, z]"),
        ({"a": {"stand": [[0, 0]]}}, "'stand' is a list of [x, y, yaw]"),
        ({"a": {}}, "neither a hand target"),
        ({"a": {"at": [0, 0, 1], "stands": [[0.6, 0, 3.1]]}}, "unknown key 'stands'"),
        ({"a": {"at": [0, 0, "up"]}}, "'at' is [x, y, z]"),
        ({"a": {"at": [0, 0, 1], "stand": [[0.6, 0, float("nan")]]}}, "'stand' is a list of"),
        ({"a/grip": {"kind": "place", "at": [0, 0, 1]}}, "a slot's name is one word"),
        ({"grip_l": {"kind": "place", "at": [0, 0, 1]}}, "a slot's name is one word"),
        ({"stand": {"at": [0, 0, 1]}}, "a slot's name is one word"),
        ({"a": {"kind": "grip", "stand": [[0.6, 0, 3.1]]}}, "'kind' 'grip' names a hand target"),
    ],
)
def test_a_slot_that_states_nothing_usable_is_refused_saying_why(slots, says):
    with pytest.raises(PluginError, match=says.replace("[", r"\[").replace("]", r"\]")):
        _compiled(_world(slots))


def test_slots_on_a_body_that_is_not_there_are_refused():
    with pytest.raises(PluginError, match="no body 'rack' to state slots on"):
        _compiled(_world({"a": {"at": [0, 0, 1]}}, body="rack"))


def test_slots_are_nested_under_the_entry_they_are_on():
    world = {
        "sim": {"pacing": "asap"},
        "components": [{"slots": {"body": "x", "slots": {"a": {"at": [0, 0, 1]}}}}],
    }
    with pytest.raises(PluginError, match="components"):
        _compiled(world)
