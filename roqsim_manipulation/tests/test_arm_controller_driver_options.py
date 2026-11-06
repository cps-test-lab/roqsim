"""The arm_controller options a vendor driver's interface needs, each against what ros2_control does.

``state_interfaces``, the topic rates, the declared-but-unloaded ``forward_command_controller``, the
motion gate a driver interface closes, ``command_feedforward`` and the constraint hints a trajectory
action is graded by. Each is off by default, so an arm that names none behaves as before.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import (
    FORWARD_COMMAND_TYPE,
    ArmControllerPlugin,
    motion_gate_key,
    state_interfaces_key,
)

from roqsim.config import load_config_from_dict
from roqsim.controllers import ACTIVE, INACTIVE, STRICT, registry_for
from roqsim.engine import Engine


def _engine(**arm) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [{"arm_controller": arm, "name": "arm_controller"}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _endpoints(engine) -> dict:
    return {e.name: e for e in engine.ctx.interface.all() if e.owner == "ur5e"}


def _arm(engine) -> ArmControllerPlugin:
    return next(p for p in engine.plugins if isinstance(p, ArmControllerPlugin))


def _steps(engine, n: int) -> None:
    for _ in range(n):
        engine.step()


def test_state_interfaces_leave_out_what_the_hardware_has_not():
    engine = _engine(state_interfaces=["position", "velocity"])
    state = _endpoints(engine)["joint_states"].read()
    assert len(state.positions) == len(state.velocities) == 6
    assert len(state.efforts) == 0


def test_topic_rates_are_configurable():
    eps = _endpoints(_engine(joint_state_rate=500.0, controller_state_rate=100.0))
    assert eps["joint_states"].rate_hz == 500.0
    assert eps["controller_state"].rate_hz == 100.0
    eps = _endpoints(_engine())
    assert eps["joint_states"].rate_hz == eps["controller_state"].rate_hz == 50.0


@pytest.mark.parametrize("bad", [["effort"], ["position", "torque"], [], "position"])
def test_state_interfaces_are_checked(bad):
    errors = ArmControllerPlugin({"state_interfaces": bad}).validate_config(
        {"state_interfaces": bad}
    )
    assert any("state_interfaces" in e for e in errors), errors


def test_the_forward_controller_is_unlisted_until_loaded_then_drives_the_joints():
    engine = _engine(forward_command_controller="forward_position_controller")
    registry = registry_for(engine.ctx)
    fwd = registry.get("forward_position_controller")
    assert fwd.type == FORWARD_COMMAND_TYPE and not fwd.listed and fwd.state == INACTIVE
    # Not loaded, so not switchable: what a spawner meets on the robot before load_controller.
    ok, message = registry.switch(
        activate=["forward_position_controller"], deactivate=["arm_controller"], strictness=STRICT
    )
    assert not ok and registry.get("arm_controller").state == ACTIVE, message
    fwd.listed = True  # what the bridge's load_controller does
    # It claims the trajectory controller's joints, so one replaces the other.
    ok, _ = registry.switch(activate=["forward_position_controller"], strictness=STRICT)
    assert not ok
    ok, _ = registry.switch(
        activate=["forward_position_controller"], deactivate=["arm_controller"], strictness=STRICT
    )
    assert ok and fwd.state == ACTIVE
    eps = _endpoints(engine)
    target = [0.2, -1.2, 1.3, -1.6, -1.5, 0.3]
    eps["forward_commands"].write({"data": target})
    _steps(engine, 1500)
    assert np.allclose(eps["joint_states"].read().positions, target, atol=1e-3)
    # A command of the wrong size is refused, as upstream refuses it.
    eps["forward_commands"].write({"data": [0.0, 0.0]})
    _steps(engine, 200)
    assert np.allclose(eps["joint_states"].read().positions, target, atol=1e-3)


def test_forward_commands_do_nothing_while_the_controller_is_inactive():
    engine = _engine(forward_command_controller="forward_position_controller")
    eps = _endpoints(engine)
    before = np.array(eps["joint_states"].read().positions)
    eps["forward_commands"].write({"data": [0.5] * 6})
    _steps(engine, 500)
    assert np.allclose(eps["joint_states"].read().positions, before, atol=1e-3)


def test_the_motion_gate_holds_the_arm_and_ignores_commands_until_it_opens():
    engine = _engine()
    arm = engine.ctx.blackboard.get("arm:ur5e")
    reason = {"now": None}
    engine.ctx.blackboard.set(motion_gate_key("ur5e"), lambda: reason["now"])
    start = np.array(arm.read_state()[1])
    arm.follow_trajectory(arm.joint_names, [0.0, 4.0], [start, start + 0.2])
    _steps(engine, 300)
    reason["now"] = "stopped"
    _steps(engine, 1)
    assert arm.motion_blocked() == "stopped"
    assert not arm.trajectory_active()
    plugin = _arm(engine)
    held = dict(plugin._target)  # the pose the arm was at when the gate closed
    arm.set_targets(arm.joint_names, list(start + 1.0))  # ignored while closed
    _steps(engine, 1500)
    assert plugin._target == held
    assert np.allclose(arm.read_state()[1], [held[n] for n in arm.joint_names], atol=1e-3)
    reason["now"] = None
    _steps(engine, 1)
    assert arm.motion_blocked() is None
    arm.set_targets(arm.joint_names, list(start))
    _steps(engine, 1500)
    assert np.allclose(arm.read_state()[1], start, atol=1e-3)


def _ramp_error(**arm) -> float:
    """Worst lag behind a streamed ramp of 1 rad/s on the shoulder, fed once per step."""
    engine = _engine(**arm)
    handle = engine.ctx.blackboard.get("arm:ur5e")
    m = engine.ctx.model
    q0 = list(handle.read_state()[1])
    worst = 0.0
    dt = m.opt.timestep
    for k in range(1, 400):
        q = list(q0)
        q[0] = q0[0] + 1.0 * k * dt
        handle.set_targets(handle.joint_names, q)
        engine.step()
        if k > 100:
            worst = max(worst, abs(handle.read_state()[1][0] - q[0]))
    return worst


def test_command_feedforward_takes_out_the_servos_lag_on_a_stream():
    plain = _ramp_error()
    fed = _ramp_error(command_feedforward="velocity")
    # The UR5e's shoulder servo trails a 1 rad/s ramp by d/p = 0.2 rad without it.
    assert plain > 0.1
    assert fed < plain / 10


def test_constraints_reach_the_trajectory_action_as_hints():
    eps = _endpoints(
        _engine(
            goal_tolerance=0.05,
            goal_time_tolerance=0.0,
            path_tolerance=0.25,
            allow_partial_joints_goal=False,
            allow_nonzero_velocity_at_trajectory_end=False,
        )
    )
    hints = eps["follow_joint_trajectory"].backend["ros2"]
    assert hints["goal_tolerance"] == 0.05
    assert hints["goal_time_tolerance"] == 0.0
    assert hints["path_tolerance"] == 0.25
    assert hints["allow_partial_joints_goal"] is False
    assert hints["allow_nonzero_velocity_at_trajectory_end"] is False
    assert len(hints["joint_names"]) == 6


def test_dynamic_joint_states_carries_the_joints_and_what_a_driver_adds():
    engine = _engine(dynamic_joint_states=True, state_interfaces=["position", "velocity"])
    engine.ctx.blackboard.set(
        state_interfaces_key("ur5e"), lambda: [("Status", {"motion_possible": 1.0})]
    )
    rows = dict(_endpoints(engine)["dynamic_joint_states"].read())
    assert set(rows["shoulder_pan_joint"]) == {"position", "velocity"}
    assert rows["Status"] == {"motion_possible": 1.0}
    assert not math.isnan(rows["shoulder_pan_joint"]["position"])
