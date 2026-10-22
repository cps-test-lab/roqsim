"""``spot_locomotion``'s ``joint_names``: the names joint_states publishes, and nothing else renamed.

The spot_ros2 names are its ``kFriendlyJointNames``
(bdaiinstitute/spot_ros2 @ 949feb7b0d702d1923e5b5dd5c0a4a7728ddc88f,
spot_driver/include/spot_driver/conversions/robot_state.hpp:34-42).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from roqsim_quadruped.policy import DEFAULT_POLICY

from roqsim.config import PluginError, load_config_from_dict
from roqsim.engine import Engine

POLICY = os.environ.get("SPOT_POLICY_PATH") or str(DEFAULT_POLICY)
LEGS = {"fl": "front_left", "fr": "front_right", "hl": "rear_left", "hr": "rear_right"}
JOINTS = {"hx": "hip_x", "hy": "hip_y", "kn": "knee"}
SPOT_ROS2_NAMES = {
    f"{a}_{b}": f"{leg}_{joint}" for a, leg in LEGS.items() for b, joint in JOINTS.items()
}


def engine_order(engine: Engine) -> list[str]:
    plugin = next(p for p in engine.plugins if p.address == "robot.spot_locomotion")
    return list(plugin._joint_order)


def _engine(**locomotion) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.002},
            "components": [
                {
                    "spawn_robot": {"model": "spot"},
                    "name": "robot",
                    "components": [{"spot_locomotion": locomotion}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize(
    "names, match",
    [
        ("front_left_hip_x", "maps an MJCF joint name"),
        ({"fl_hx": ""}, "maps an MJCF joint name"),
        ({"fl_hx": "a", "fr_hx": "a"}, "two joints under one name"),
    ],
)
def test_joint_names_is_checked(names, match):
    with pytest.raises(PluginError, match=match):
        _engine(joint_names=names)


@pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)
def test_joint_states_publishes_the_mapped_names_and_the_policy_keeps_walking():
    engine = _engine(joint_names=SPOT_ROS2_NAMES)
    for _ in range(500):  # 1 s
        engine.step()
    ep = next(e for e in engine.ctx.interface.all() if e.name == "joint_states")
    state = ep.read()
    # In the policy's order, each under its spot_ros2 name.
    assert state.names == [SPOT_ROS2_NAMES[j] for j in engine_order(engine)]
    odom = next(e for e in engine.ctx.interface.all() if e.name == "odom").read()
    assert odom.position[2] > 0.4  # still standing
    engine.shutdown()


@pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)
def test_a_joint_spot_does_not_have_is_refused_by_name():
    with pytest.raises(RuntimeError, match="fl_knee"):
        _engine(joint_names={"fl_knee": "front_left_knee"})


@pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)
def test_a_name_another_joint_keeps_is_refused():
    with pytest.raises(RuntimeError, match="two joints under one name"):
        _engine(joint_names={"fl_hx": "fl_hy"})
