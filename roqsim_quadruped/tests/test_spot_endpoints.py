"""``spot_locomotion``'s endpoints: declared on its methods, typed, and wired to the policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from roqsim_quadruped.policy import DEFAULT_POLICY

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details

POLICY = os.environ.get("SPOT_POLICY_PATH") or str(DEFAULT_POLICY)


def test_the_endpoints_describe_their_units():
    rows = {row["name"]: row for row in get_plugin_details("spot_locomotion")["endpoints"]}
    assert set(rows) == {"cmd_vel", "odom", "joint_states"}
    assert [(p["name"], p["unit"]) for p in rows["cmd_vel"]["params"]] == [
        ("vx", "m/s"),
        ("vy", "m/s"),
        ("w", "rad/s"),
    ]
    odom = rows["odom"]["result"]["items"]
    assert [i["unit"] for i in odom] == ["m", "m", "rad", "m/s", "m/s", "rad/s", "m"]


@pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)
def test_cmd_vel_is_a_typed_stream_the_policy_receives():
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {"model": "spot", "prefix": "r_", "default_plugins": False},
                "name": "r",
                "components": [{"spot_locomotion": {}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    try:
        cmd_vel = next(e for e in engine.ctx.interface.all() if e.name == "cmd_vel")
        loco = next(p for p in engine.plugins if p.entity == "r")
        cmd_vel.write({"vx": 0.3, "vy": 0.1})
        engine.step()
        assert loco._cmd.tolist() == pytest.approx([0.3, 0.1, 0.0])
    finally:
        engine.shutdown()
