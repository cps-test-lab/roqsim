"""A walker meets what is in its way as its output declares, where its world does not say.

The walker output declares a person's pace (``roqsim_walker.output.WALKER_AVOIDANCE``): how far
ahead it looks, how hard it slows, how far short of something it stands and how it walks off. A
world's own ``avoidance:`` key wins over it; a mover that is no walker takes none of it.
"""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_walker.output import WALKER_AVOIDANCE


def _probe(avoidance):
    world = {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": 0.0, "y": 0.0}}},
                "name": "pedestrian",
                "components": [
                    {
                        "navigator": {
                            "output": "walker",
                            "speed": 0.5,
                            "goals": [[2.0, 0.0]],
                            "avoidance": avoidance,
                        }
                    }
                ],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.setup()
    try:
        nav = next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")
        return nav._caution, nav.avoidance_defaults_taken
    finally:
        engine.shutdown()


def test_a_walker_takes_its_outputs_pace():
    probe, taken = _probe({"steer": "none", "stop": True})
    assert taken == dict(WALKER_AVOIDANCE)
    for key in ("width", "horizon", "decel", "stop_gap", "wait_gap", "clear_time", "tau"):
        assert getattr(probe, key) == pytest.approx(WALKER_AVOIDANCE[key])


def test_a_worlds_own_key_wins_over_the_outputs():
    probe, taken = _probe({"steer": "none", "stop": True, "decel": 0.5, "clear_time": 0.0})
    assert probe.decel == pytest.approx(0.5)
    assert probe.clear_time == 0.0
    assert "decel" not in taken and "clear_time" not in taken
    assert probe.horizon == pytest.approx(WALKER_AVOIDANCE["horizon"])
