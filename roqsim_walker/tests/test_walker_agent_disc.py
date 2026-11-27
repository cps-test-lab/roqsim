"""A walker is a person's disc to the others' avoidance model, whatever navigator it has."""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_nav.avoidance import SERVICE_KEY


def test_a_walker_with_the_default_navigator_is_a_persons_disc_to_the_others(tmp_path):
    """A walker the world nests no navigator under gets the default one, which plans with the
    walker's 0.26 m disc; to the others' avoidance model it is a person's half width, as a walker
    with a navigator of its own is."""
    mover = {
        "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": -3.0, "y": 0.0}}},
        "name": "m",
        "components": [
            {
                "navigator": {
                    "output": "walker",
                    "speed": 1.0,
                    "goals": [[3.0, 0.0]],
                    "avoidance": {"steer": "give_way", "stop": False},
                }
            }
        ],
    }
    standing = {
        "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": 2.0, "y": 1.0}}},
        "name": "s",
    }
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"}, "components": [mover, standing]}, base_dir=tmp_path
        )
    )
    engine.setup()
    engine.reset()
    try:
        for _ in range(int(0.2 / engine.ctx.dt)):  # past the first solve
            engine.step()
        model = engine.ctx.blackboard.get(SERVICE_KEY).model
        radius = {a["key"]: a["radius"] for a in model._agents.values()}
        assert radius["s"] == pytest.approx(0.23)
        assert radius["m"] == pytest.approx(0.23)
    finally:
        engine.shutdown()
