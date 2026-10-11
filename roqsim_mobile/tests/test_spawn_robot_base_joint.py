# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The base joint ``spawn_robot`` places a robot by must exist and be a free joint.

The spawn pose is seven qpos values, a position and a quaternion. A model without the named joint
would start wherever its MJCF puts it while the world states a pose, and a joint of another type
would take those seven values into whatever joints follow it -- either way a run that looks
configured and is not.
"""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

SPAWN_ROBOT = "roqsim_mobile.plugins.spawn_robot:SpawnRobotPlugin"

MJCF = """<mujoco model="cart">
  <worldbody>
    <body name="base_link" pos="0 0 0.1">
      {joint}
      <geom type="box" size="0.2 0.1 0.05" mass="2"/>
      <body name="arm" pos="0.2 0 0">
        <joint name="arm_joint" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _engine(tmp_path, joint, **config):
    model = tmp_path / "cart.xml"
    model.write_text(MJCF.format(joint=joint))
    cfg = {"model": str(model), "pose": {"position": {"x": 1.0, "y": 2.0}}, **config}
    engine = Engine(
        load_config_from_dict(
            {"sim": {"world": "empty_room"}, "components": [{SPAWN_ROBOT: cfg, "name": "cart"}]}
        )
    )
    engine.setup()
    return engine


def test_a_free_base_joint_takes_the_pose(tmp_path):
    engine = _engine(tmp_path, '<freejoint name="base_free"/>')
    m, d = engine.ctx.model, engine.ctx.data
    assert tuple(d.qpos[:2]) == pytest.approx((1.0, 2.0))
    assert engine.ctx.entities.get("cart").meta["base_joint"] == "base_free"
    assert m.njnt == 2


def test_a_model_without_the_base_joint_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="no joint 'base_free'"):
        _engine(tmp_path, "")


def test_a_base_joint_that_is_not_free_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="not a free joint"):
        _engine(tmp_path, '<joint name="base_free" type="slide" axis="1 0 0"/>')


def test_a_named_free_joint_other_than_the_default_is_used(tmp_path):
    engine = _engine(tmp_path, '<freejoint name="root"/>', base_joint="root")
    assert tuple(engine.ctx.data.qpos[:2]) == pytest.approx((1.0, 2.0))
