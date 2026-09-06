# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``present: false`` on a robot: compiled in, but not in the trial yet.

The same declaration ``spawn_model`` takes, on the plugin that places a robot -- for a machine that
arrives partway through a trial (a second robot, a unit under test that must not exist during a
warm-up). It is NOT how a per-run start pose is applied: that is ``SetEntityState``, which moves a
robot present the whole time.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import instantiate_plugins, load_config_from_dict
from roqsim.engine import Engine
from roqsim.presence import ABSENT_GEOM_GROUP, entity_geom_ids, set_present

pytest.importorskip("roqsim_mobile", reason="spawn_robot lives in roqsim_mobile")

SPAWN_ROBOT = "roqsim_mobile.plugins.spawn_robot:SpawnRobotPlugin"


def _engine(present=None):
    config = {"model": "turtlebot4", "pose": {"position": {"x": 0.0, "y": 0.0}}}
    if present is not None:
        config["present"] = present
    return Engine(
        load_config_from_dict(
            {"sim": {"world": "empty_room"}, "plugins": [{SPAWN_ROBOT: config, "name": "robot"}]},
            # The manifest's camera wants a GL context the moment the engine steps, and nothing here
            # is about what a camera sees. Presence is decided in the model, which the lidar and the
            # contact set read without one.
            overrides={"components": {"robot.oakd_camera": {"enabled": False}}},
        )
    )


def _entity(engine):
    return engine.ctx.entities.get("robot")


def test_a_robot_declared_absent_starts_absent():
    engine = _engine(present=False)
    engine.setup()
    entity = _entity(engine)
    assert entity.present is False
    groups = {
        int(engine.ctx.model.geom_group[g]) for g in entity_geom_ids(engine.ctx.model, entity.body)
    }
    assert groups == {ABSENT_GEOM_GROUP}


def test_the_default_is_present():
    engine = _engine()
    engine.setup()
    assert _entity(engine).present is True


def test_an_absent_robot_does_not_sink_through_the_floor():
    """It is out of the contact set, so the floor is not holding it up -- something else must."""
    engine = _engine(present=False)
    engine.setup()
    engine.reset()
    entity = _entity(engine)
    bid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
    before = float(engine.ctx.data.xpos[bid][2])
    for _ in range(2000):
        engine.step()
    assert float(engine.ctx.data.xpos[bid][2]) == pytest.approx(before, abs=1e-3)


def test_a_robot_spawned_in_one_episode_is_absent_again_in_the_next():
    engine = _engine(present=False)
    engine.setup()
    entity = _entity(engine)
    set_present(engine.ctx, entity, True)
    engine.reset()
    assert entity.present is False


def test_present_must_be_a_bool():
    cfg = load_config_from_dict(
        {
            "sim": {"world": "empty_room"},
            "plugins": [{SPAWN_ROBOT: {"model": "turtlebot4", "present": "no"}, "name": "robot"}],
        }
    )
    with pytest.raises(Exception, match="'present' must be true or false"):
        instantiate_plugins(cfg)
