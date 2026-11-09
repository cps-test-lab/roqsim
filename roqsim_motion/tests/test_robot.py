"""The robot output: a humanoid robot's joints driven by the motion core, kinematic or physical,
on a minimal humanoid written for the tests (no vendor's model)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from minimal_humanoid import TABLE, mjcf

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim_motion import quat as Q
from roqsim_motion.retarget import TableError


@pytest.fixture
def files(tmp_path):
    model = tmp_path / "minimal_humanoid.xml"
    model.write_text(mjcf())
    table = tmp_path / "minimal_humanoid.retarget.yaml"
    table.write_text(yaml.safe_dump({**TABLE, "hands": {
        "l": {"frame": "left_grasp"}, "r": {"frame": "right_grasp"}}, "rest": {"left_elbow_joint": -1.2, "right_elbow_joint": -1.2}}))
    return model, table


def _engine(model, table, drive, spawn="spawn_model"):
    world = {
        "sim": {"pacing": "asap", "timestep": 0.002},
        "components": [
            {spawn: {"model": str(model), "prefix": "r_", "motion": "static"}, "name": "bot",
             "components": [
                 {"humanoid_robot": {"table": str(table), "drive": drive}},
                 {"navigator": {"output": "humanoid", "speed": 0.5, "arrival_radius": 0.05,
                                "avoidance": {"steer": "none", "stop": False}}},
                 {"humanoid": {}},
             ]},
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize("drive", ["kinematic", "physical"])
def test_a_hand_is_put_on_a_pose_by_the_robots_own_joints(files, drive):
    engine = _engine(*files, drive)
    try:
        body = engine.ctx.blackboard.get("humanoid_body:bot")
        d = engine.ctx.data
        for _ in range(300):
            engine.step()
        p, q = body.grip(d, "l")
        goal = p + np.array([0.15, 0.05, 0.15])
        body.set_hand("l", goal, q)
        for _ in range(2500):
            engine.step()
        p2, q2 = body.grip(d, "l")
        tol = 0.005 if drive == "kinematic" else 0.01
        assert np.linalg.norm(p2 - goal) < tol
        assert Q.angle_between(q2, q) < np.radians(8)
    finally:
        engine.shutdown()


def test_an_arms_reach_is_its_chain_straightened(files):
    """Shoulder to elbow, elbow to wrist and wrist to grip site, as the minimal humanoid builds them."""
    engine = _engine(*files, "kinematic")
    try:
        body = engine.ctx.blackboard.get("humanoid_body:bot")
        for side in "lr":
            assert abs(body.arm_reach(side) - (0.24 + 0.22 + float(np.hypot(0.02, 0.09)))) < 1e-6
    finally:
        engine.shutdown()


def test_the_physical_robots_base_hangs_on_an_anchor_the_navigator_moves(files):
    engine = _engine(*files, "physical")
    try:
        nav = engine.ctx.blackboard.get("nav:bot:handle")
        seq = nav.send_goals([(1.0, 0.5)])
        while engine.ctx.sim_time < 15 and not (nav.status()[0] == seq and nav.status()[1]):
            engine.step()
        assert nav.status()[1] and not nav.status()[4]
        base = engine.ctx.data.body("r_pelvis").xpos
        assert np.hypot(base[0] - 1.0, base[1] - 0.5) < 0.1
        assert abs(base[2] - 0.8) < 0.05  # held up: balance is not simulated
    finally:
        engine.shutdown()


def test_a_table_naming_what_the_model_lacks_is_refused(files, tmp_path):
    model, table = files
    doc = yaml.safe_load(table.read_text())
    doc["joints"]["elbow_l"] = [{"y": "left_elbow_pitch_joint"}]
    doc.pop("rest")
    bad = tmp_path / "bad.retarget.yaml"
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(TableError, match="left_elbow_pitch_joint"):
        _engine(model, bad, "kinematic")


def test_humanoid_robot_belongs_under_a_spawn_model(files):
    model, table = files
    with pytest.raises(PluginError, match="nest it under the spawn_model"):
        _engine(model, table, "kinematic", spawn="box")
