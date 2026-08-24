"""The articulated G1: leg/arm ownership stays separate, and it stands rather than drifting away.

Two things are worth guarding here. The `joints:` allowlist on `arm_controller` is load-bearing on this
robot -- without it the prefix scan claims all 29 joint actuators including the 12 leg motors and writes
position targets into torque commands, fighting `g1_locomotion` in the same `pre_step`. And station
keeping is what makes the platform usable for manipulation at all.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_humanoid.plugins.g1_locomotion import LEG_JOINTS

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _world(tmp_path, loco=None):
    robot = {
        "spawn_robot": {"model": "unitree_g1_dex1", "pos": [0, 0]},
        "name": "robot",
    }
    if loco is not None:
        robot["components"] = [{"g1_locomotion": dict(loco)}]
    return load_config_from_dict(
        {"sim": {"timestep": 0.002}, "components": [robot]}, base_dir=tmp_path
    )


def _run(engine, steps):
    for _ in range(steps):
        engine.step()


def test_manifest_brings_locomotion_two_arms_and_a_lidar(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    kinds = [type(p).__name__ for p in engine.plugins]
    # Three: left arm, right arm, and the waist. The waist one exists so the three waist joints are
    # REPORTED -- without an owner they appear in no /joint_states message and robot_state_publisher
    # silently assumes zero for the joints the arms hang off.
    assert kinds.count("ArmControllerPlugin") == 3
    assert kinds.count("G1LocomotionPlugin") == 1
    # 12 leg + 3 waist + 14 arm + 2 gripper tendon actuators.
    assert engine.ctx.model.nu == 31


def test_each_arm_owns_only_its_own_joints_and_gripper(tmp_path):
    """The regression that matters: the arms must not claim the legs."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    model = engine.ctx.model
    leg_actuators = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_JOINTS}
    for side in ("left", "right"):
        handle = engine.ctx.blackboard.require(f"arm:robot:{side}_arm_controller")
        assert len(handle.joint_names) == 7
        assert all(n.startswith(side) for n in handle.joint_names)
        claimed = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in handle.joint_names
        }
        assert not (claimed & leg_actuators)
    # Distinct gripper readers: keyed per controller, else the bridge's GripperCommand handler would
    # watch one gripper's state for both arms and report the wrong one's motion.
    assert engine.ctx.blackboard.get("gripper:robot:left_gripper_controller") is not None
    assert engine.ctx.blackboard.get("gripper:robot:right_gripper_controller") is not None


def test_legs_are_torque_driven_by_the_policy(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    _run(engine, 500)
    model, data = engine.ctx.model, engine.ctx.data
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
    # A torque, not a joint-angle target: the policy's PD output, well outside the joint's range.
    assert abs(data.ctrl[knee]) > 0.5


def test_grippers_open_and_close_independently(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    _run(engine, 500)
    eps = {(e.name, e.backend["ros2"].get("name", "")): e for e in engine.ctx.interface.all()}
    for side in ("left", "right"):
        ep = eps[("gripper_cmd", f"{side}_gripper_controller/gripper_cmd")]
        reader = engine.ctx.blackboard.require(ep.backend["ros2"]["state_key"])
        for target in (0.0245, -0.02):  # measured travel limits: 94.9 mm and 5.9 mm aperture
            ep.write(target)
            _run(engine, 1200)
            assert reader()[0] == pytest.approx(target, abs=0.002)


def test_station_keeping_bounds_the_drift(tmp_path):
    """A zero cmd_vel means "walk at zero speed" to this policy, not "stay here".

    Without station keeping the robot drifts ~0.9 m in 10 s -- enough to walk away from the table it is
    reaching for. Pre-existing behaviour (unitree_g1 drifts 0.625 m over the same span), so this guards
    the fix rather than a regression.
    """
    drift = {}
    for label, loco in (("off", {}), ("on", {"station_keeping": True})):
        engine = Engine(_world(tmp_path, loco=loco))
        engine.setup()
        engine.reset()
        model, data = engine.ctx.model, engine.ctx.data
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        start = data.xpos[base][:2].copy()
        _run(engine, 5000)  # 10 s
        drift[label] = float(np.linalg.norm(data.xpos[base][:2] - start))
        assert data.xpos[base][2] > 0.6, f"fell over with station_keeping {label}"
    assert drift["off"] > 0.4
    assert drift["on"] < 0.15
