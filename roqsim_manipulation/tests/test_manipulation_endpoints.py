# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The typed endpoints of ``arm_controller`` and ``cartesian_admittance``.

A trajectory waypoint, a streamed joint command and a joint velocity are FIFO commands, so every
message is applied in order even when several arrive within one step; the Cartesian setpoints are
streams, of which a step applies the latest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from roqsim_manipulation.plugins.arm_controller import (
    ArmControllerPlugin,
    ControllerState,
    JointVelocities,
)
from roqsim_manipulation.plugins.cartesian_admittance import CartesianAdmittancePlugin

from roqsim.config import load_config_from_dict
from roqsim.endpoint import ParameterError, declared
from roqsim.engine import Engine
from roqsim.types import JointPositions, JointState, Pose, Wrench

JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint")


def _engine(*components: dict, ctrl=None) -> Engine:
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "ur5e", "prefix": "ur5e_", "namespace": "ur5e"},
                "name": "ur5e",
                "components": [
                    {"arm_controller": dict(ctrl or {})},
                    {"force_torque": {"site": "fts_site", "frame": "world"}, "name": "ft"},
                    *components,
                ],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")), preview=True)
    engine.setup()
    engine.reset()
    return engine


def _endpoints(engine: Engine) -> dict:
    return {e.name: e for e in engine.ctx.interface.all() if e.owner == "ur5e"}


def _plugin(engine: Engine, cls: type):
    return next(p for p in engine.plugins if isinstance(p, cls))


def test_arm_kinds_follow_what_each_input_must_keep():
    kinds = {spec.name: spec.kind for spec in declared(ArmControllerPlugin)}
    assert kinds == {
        "joint_states": "out",
        "follow_joint_trajectory": "command",
        "controller_state": "out",
        "joint_command": "command",
        "joint_velocity": "command",
        "gripper_cmd": "command",
    }
    kinds = {spec.name: spec.kind for spec in declared(CartesianAdmittancePlugin)}
    assert kinds == {
        "target_frame": "stream",
        "target_wrench": "stream",
        "current_pose": "out",
        "tracking_error": "out",
    }


def test_the_payloads_are_the_neutral_types():
    engine = _engine(
        {"cartesian_admittance": {"site": "tool_site", "ft": "ft"}},
        ctrl={"stream_commands": True, "velocity_commands": True},
    )
    eps = _endpoints(engine)
    for name in ("follow_joint_trajectory", "joint_command"):
        assert eps[name].payload_type.cls is JointPositions
    assert eps["joint_velocity"].payload_type.cls is JointVelocities
    assert [p.name for p in eps["joint_velocity"].params] == ["names", "velocities"]
    joints = eps["joint_states"].read()
    assert isinstance(joints, JointState) and len(joints.efforts) == len(joints.names)
    state = eps["controller_state"].read()
    assert isinstance(state, ControllerState)
    assert state.error.positions == pytest.approx(
        [r - f for r, f in zip(state.reference.positions, state.feedback.positions, strict=True)]
    )
    assert eps["target_frame"].payload_type.cls is Pose
    assert eps["target_wrench"].payload_type.cls is Wrench
    assert isinstance(eps["current_pose"].read(), Pose)


@pytest.mark.parametrize("name", ["follow_joint_trajectory", "joint_command"])
def test_every_waypoint_in_one_step_is_applied_in_order(name):
    engine = _engine(ctrl={"stream_commands": True})
    ep = _endpoints(engine)[name]
    arm = _plugin(engine, ArmControllerPlugin)
    first = ep.write({"names": [JOINTS[0]], "positions": [0.3]})
    second = ep.write({"names": [JOINTS[1]], "positions": [-0.4]})
    third = ep.write({"names": [JOINTS[0]], "positions": [0.2]})
    engine.step()
    for future in (first, second, third):
        assert future.result(0) is None
    # A stream would have kept only the last message and lost the second joint's target.
    assert arm._target[JOINTS[0]] == pytest.approx(0.2)
    assert arm._target[JOINTS[1]] == pytest.approx(-0.4)


def test_joint_velocities_of_two_messages_in_one_step_merge():
    engine = _engine(ctrl={"velocity_commands": True})
    ep = _endpoints(engine)["joint_velocity"]
    arm = _plugin(engine, ArmControllerPlugin)
    ep.write({"names": [JOINTS[0]], "velocities": [0.1]})
    ep.write({"names": [JOINTS[1]], "velocities": [-0.2]})
    engine.step()
    assert arm._vel_cmd == {JOINTS[0]: 0.1, JOINTS[1]: -0.2}


def test_a_misfit_command_is_refused_into_its_future():
    engine = _engine()
    ep = _endpoints(engine)["follow_joint_trajectory"]
    future = ep.write({"names": [JOINTS[0]], "position": [0.3]})
    with pytest.raises(ParameterError, match="did you mean 'positions'"):
        future.result(0)


def test_cartesian_setpoints_apply_the_latest_per_step():
    engine = _engine({"cartesian_admittance": {"site": "tool_site", "ft": "ft"}})
    eps = _endpoints(engine)
    law = _plugin(engine, CartesianAdmittancePlugin)
    eps["target_wrench"].write({"force": [0.0, 0.0, -3.0], "torque": [0.0, 0.0, 0.0]})
    eps["target_wrench"].write({"force": [1.0, 2.0, -8.0], "torque": [0.0, 0.0, 0.5]})
    eps["target_frame"].write({"position": [0.4, 0.1, 0.3], "orientation": [1.0, 0.0, 0.0, 0.0]})
    engine.step()
    assert law.w_d == pytest.approx([1.0, 2.0, -8.0, 0.0, 0.0, 0.5])
    assert law._goal_pos == pytest.approx([0.4, 0.1, 0.3])
    with pytest.raises(ParameterError, match="shape"):
        eps["target_frame"].write({"position": [0.4, 0.1], "orientation": [1.0, 0.0, 0.0, 0.0]})


def test_cartesian_endpoints_take_the_arm_namespace_and_describe_their_units():
    engine = _engine({"cartesian_admittance": {"site": "tool_site", "ft": "ft"}})
    eps = _endpoints(engine)
    for name in ("target_frame", "target_wrench", "current_pose", "tracking_error"):
        assert eps[name].namespace == "ur5e"
    wrench = {p.name: p.type.unit for p in eps["target_wrench"].params}
    assert wrench == {"force": "N", "torque": "N*m"}
    fields = {f.name: f.type.unit for f in eps["tracking_error"].result.fields}
    assert fields["distance"] == "m" and fields["angle"] == "rad"
