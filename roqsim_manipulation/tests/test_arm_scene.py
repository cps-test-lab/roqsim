"""Integration: the arm scene compiles, holds its home pose, and tracks a commanded target."""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _world(tmp_path, arm_extra=None, ctrl_extra=None):
    # No floorplan plugin: the arm's bare floor + light come from the default world (empty_room).
    plugins = [
        {
            "spawn_arm": {
                "model": "ur10e",
                "prefix": "ur10e_",
                **(arm_extra or {}),
            },
            "name": "ur10e",
            "components": [{"arm_controller": dict(ctrl_extra or {})}],
        },
    ]
    return load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path)


def _run(engine, steps):
    engine.setup()
    engine.reset()
    for _ in range(steps):
        engine.step()


def test_arm_compiles_and_registers(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    assert "ur10e" in engine.ctx.entities.names()
    assert engine.ctx.model.nu == 6  # six UR10e joint actuators
    handle = engine.ctx.blackboard.require("arm:ur10e")
    assert handle.joint_names[0] == "shoulder_pan_joint"


def test_arm_endpoints_inherit_spawn_namespace(tmp_path):
    # spawn_arm's `namespace` rides on the entity meta; arm_controller (here the world-declared
    # override, same path as the manifest-injected default) stamps it on its endpoints.
    engine = Engine(_world(tmp_path, arm_extra={"namespace": "ur10e"}))
    engine.setup()
    eps = {e.name: e for e in engine.ctx.interface.all()}
    assert eps["joint_states"].namespace == "ur10e"
    fjt = eps["follow_joint_trajectory"]
    assert fjt.namespace == "ur10e"
    assert fjt.direction == "in"
    assert fjt.backend["ros2"]["action"] == "control_msgs.action.FollowJointTrajectory"
    assert fjt.backend["ros2"]["name"] == "arm_controller/follow_joint_trajectory"
    # The action endpoint's write takes the neutral (names, positions) waypoint payload.
    fjt.write((["shoulder_pan_joint"], [0.5]))


def test_stream_commands_declares_joint_trajectory_topic(tmp_path):
    # Off by default: no streaming topic input, only the action.
    engine = Engine(_world(tmp_path, arm_extra={"namespace": "ur10e"}))
    engine.setup()
    assert "joint_command" not in {e.name for e in engine.ctx.interface.all()}

    # On: a high-rate JointTrajectory *topic* input at <controller>/joint_trajectory (the same
    # interface a ros2_control JointTrajectoryController / moveit_servo target expects).
    engine = Engine(
        _world(tmp_path, arm_extra={"namespace": "ur10e"}, ctrl_extra={"stream_commands": True})
    )
    engine.setup()
    ep = {e.name: e for e in engine.ctx.interface.all()}["joint_command"]
    assert ep.direction == "in" and ep.namespace == "ur10e"
    assert ep.backend["ros2"]["type"] == "trajectory_msgs.msg.JointTrajectory"
    assert ep.backend["ros2"]["topic"] == "arm_controller/joint_trajectory"
    # Same neutral (names, positions) payload as the action -> set_targets.
    ep.write((["shoulder_pan_joint"], [0.4]))
    handle = engine.ctx.blackboard.require("arm:ur10e")
    engine.reset()
    ep.write((["shoulder_pan_joint"], [0.4]))
    for _ in range(400):
        engine.step()
    _, pos, *_ = handle.read_state()
    assert abs(pos[0] - 0.4) < 0.05  # a streamed position target servos the joint


def test_arm_holds_home(tmp_path):
    home = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    engine = Engine(_world(tmp_path))
    _run(engine, 500)
    names, pos, vel, eff = engine.ctx.blackboard.require("arm:ur10e").read_state()
    assert names[:6] == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert np.allclose(pos[:6], home, atol=0.05)  # settled at home


def test_manual_control_leaves_ctrl_to_the_sliders(tmp_path):
    """--manual-control: the plugin stops stamping ctrl, so a slider drag survives and servos."""
    engine = Engine(_world(tmp_path))
    engine.ctx.manual_control = True
    engine.setup()
    engine.reset()
    engine.ctx.data.ctrl[0] = 0.4  # what dragging the shoulder_pan slider does
    for _ in range(500):
        engine.step()
    _, pos, *_ = engine.ctx.blackboard.require("arm:ur10e").read_state()
    assert engine.ctx.data.ctrl[0] == 0.4  # not stamped back to the held target
    assert abs(pos[0] - 0.4) < 0.05  # the arm went where the slider said


def test_controller_owns_ctrl_by_default(tmp_path):
    """The mirror of the above: without the flag the same drag is overwritten every tick."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    engine.ctx.data.ctrl[0] = 0.4
    for _ in range(500):
        engine.step()
    _, pos, *_ = engine.ctx.blackboard.require("arm:ur10e").read_state()
    assert abs(pos[0] - (-1.5708)) < 0.05  # back at home; the controller won


def test_arm_tracks_commanded_target(tmp_path):
    target = [0.0, -1.0, 1.2, -1.6, -1.57, 0.0]
    engine = Engine(_world(tmp_path, ctrl_extra={"test_target": target}))
    _run(engine, 1500)
    _, pos, *_ = engine.ctx.blackboard.require("arm:ur10e").read_state()
    assert np.allclose(pos[:6], target, atol=0.05)  # position servos reached the target


def test_joint_state_reports_effort(tmp_path):
    """A real driver publishes effort (ros2_control fills it; the G1's /lowstate carries tau_est)."""
    engine = Engine(_world(tmp_path))
    _run(engine, 500)
    names, _, _, eff = engine.ctx.blackboard.require("arm:ur10e").read_state()
    assert len(eff) == len(names)
    # Holding the home pose against gravity costs non-zero torque somewhere in the chain.
    assert max(abs(e) for e in eff) > 1e-3


def test_controller_state_endpoint_mirrors_a_jtc(tmp_path):
    """The third interface a ros2_control JointTrajectoryController exposes, beside action + topic."""
    engine = Engine(_world(tmp_path))
    _run(engine, 500)
    ep = {e.name: e for e in engine.ctx.interface.all()}["controller_state"]
    assert ep.direction == "out"
    assert ep.backend["ros2"]["type"] == "control_msgs.msg.JointTrajectoryControllerState"
    assert ep.backend["ros2"]["topic"] == "arm_controller/controller_state"
    names, desired, actual, velocities = ep.read()
    # Only the *commanded* joints: a JTC states its control loop, not every reported joint.
    assert len(names) == len(desired) == len(actual) == len(velocities) == 6
    assert np.allclose(actual, desired, atol=0.05)  # settled at home, so error is small


def test_joints_allowlist_scopes_ownership(tmp_path):
    """`joints:` claims only the named actuators, leaving the rest of the entity alone.

    The prefix scan is right for a standalone arm and wrong for an arm sharing its entity with other
    actuated parts (a humanoid's legs, a mobile base's wheels): it claims those too and then fights
    their owner, writing position targets into what may be torque actuators. This is the regression
    guard for that -- the unclaimed actuators must be left untouched.
    """
    owned = ["shoulder_pan_joint", "shoulder_lift_joint"]
    engine = Engine(_world(tmp_path, ctrl_extra={"joints": owned}))
    engine.setup()
    handle = engine.ctx.blackboard.require("arm:ur10e:arm_controller")
    assert handle.joint_names == owned  # exactly the named joints, in the given order

    engine.reset()
    # Stamp a sentinel on an actuator the controller does NOT own; it must survive every step.
    unowned = engine.ctx.model.actuator("ur10e_elbow").id
    engine.ctx.data.ctrl[unowned] = 0.25
    for _ in range(200):
        engine.step()
    assert engine.ctx.data.ctrl[unowned] == 0.25


def test_joints_allowlist_raises_on_unknown_joint(tmp_path):
    """Fail loudly: a silently dropped joint is an arm that reports but never moves."""
    engine = Engine(_world(tmp_path, ctrl_extra={"joints": ["shoulder_pan_joint", "nope_joint"]}))
    with pytest.raises(RuntimeError, match="nope_joint"):
        engine.setup()
