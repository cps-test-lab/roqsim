"""A Cartesian controller drives an arm whose arm_controller names its joints.

Such a controller publishes its ArmHandle under its own name (`arm:<arm>:<controller_name>`), since
one entity may carry several. The Interbotix arms ship that way, so a Cartesian controller that only
looked at `arm:<arm>` could not be put on them at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _engine(tmp_path, **cartesian):
    world = {
        "sim": {},
        "components": [
            {
                "spawn_arm": {"model": "vx300s", "prefix": "vx_"},
                "name": "vx",
                "components": [
                    {
                        "cartesian_admittance": {
                            "controller_type": "cartesian_motion_controller",
                            "site": "pinch",
                            **cartesian,
                        }
                    }
                ],
            }
        ],
    }
    return Engine(load_config_from_dict(world, base_dir=tmp_path))


def test_it_moves_the_arm_through_the_controller_it_names(tmp_path):
    engine = _engine(tmp_path, arm_controller="arm_controller")
    engine.setup()
    engine.reset()
    handle = engine.ctx.blackboard.require("cartesian:vx")
    start, _ = handle.read_pose()
    goal = start + np.array([0.0, 0.0, 0.03])
    handle.set_goal(goal)
    for _ in range(1500):
        engine.step()
    reached, _ = handle.read_pose()
    assert np.linalg.norm(reached - goal) < 0.005


def test_without_the_name_it_says_where_the_handle_is(tmp_path):
    with pytest.raises(RuntimeError, match=r"name it here with `arm_controller:`"):
        _engine(tmp_path).setup()
