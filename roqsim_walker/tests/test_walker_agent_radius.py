"""A walker is a person's disc to an avoidance model, not its model's hips."""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_walker.output import WalkerOutput


def test_a_walker_joins_the_avoidance_model_with_a_persons_disc():
    nav = {
        "output": "walker",
        "speed": 1.0,
        "goals": [[2.0, 0.0]],
        "avoidance": {"steer": "give_way", "stop": True},
    }
    world = {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": 0.0, "y": 0.0}}},
                "name": "pedestrian",
                "components": [{"navigator": nav}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        navigator = next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")
        model = navigator._avoid.model
        (agent,) = model._agents.values()
        assert agent["radius"] == pytest.approx(WalkerOutput.agent_radius)
        # the planner still inflates walls by the walker's measured footprint
        assert navigator.radius(engine.ctx) > 0.3
    finally:
        engine.shutdown()
