"""Walkers side by side at people's spacing do not hold each other up.

A walker's hips are 0.66 m across, so two walking abreast 0.6 m apart, as people do, touch at the
shoulder, and as they sway one's hip reaches past the other's front. A walker follows what moves
along its way and never stands for it, so neither holds for the other; nor does one a step behind
the other in a group setting off together.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _walker(name, y, x=-3.0):
    return {
        "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": x, "y": y}}},
        "name": name,
        "components": [
            {
                "navigator": {
                    "output": "walker",
                    "speed": 1.4,
                    "goals": [[3.0, y]],
                    "avoidance": {"steer": "none", "stop": True},
                }
            }
        ],
    }


def _xy(engine, name):
    model = engine.ctx.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/pelvis")
    return engine.ctx.data.mocap_pos[model.body_mocapid[bid]][:2].copy()


def test_two_walkers_abreast_at_peoples_spacing_both_walk_their_way():
    world = {"sim": {"pacing": "asap"}, "components": [_walker("a", -0.3), _walker("b", 0.3)]}
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        for _ in range(int(4.5 / engine.ctx.dt)):  # 6 m at 1.4 m/s, and the arrival
            engine.step()
        for name, y in (("a", -0.3), ("b", 0.3)):
            assert np.linalg.norm(_xy(engine, name) - (3.0, y)) < 0.35, f"{name} held for the other"
    finally:
        engine.shutdown()


def test_a_walker_a_step_behind_another_walks_off_with_it():
    """The group's start: one walker 0.3 m ahead of the other and 0.6 m beside it."""
    world = {
        "sim": {"pacing": "asap"},
        "components": [_walker("a", -0.3), _walker("b", 0.3, x=-3.3)],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        # 6.3 m at 1.4 m/s, and the arrival; b walks off once a has, its start-up delay later
        for _ in range(int(6.0 / engine.ctx.dt)):
            engine.step()
        assert np.linalg.norm(_xy(engine, "b") - (3.0, 0.3)) < 0.35, "b held for a, ahead of it"
    finally:
        engine.shutdown()
