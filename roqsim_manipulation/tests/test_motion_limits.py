# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""``arm_controller``'s ``max_velocity`` / ``max_acceleration``: a position command becomes a ramp.

A single position handed to a stiff servo is a step, and the joint takes it as fast as the servo's
force range allows -- a lift handed one leaps and throws what stands on it. With a limit, the held
target (what the servo is given, and what ``controller_state`` reports as its reference) travels to the
commanded position on a trapezoidal profile. These pin the profile at the setpoint, where it is
exact, rather than at the joint, where the servo's own dynamics blur it.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]
VMAX, AMAX = 0.5, 2.0
DT = 0.002


def _errors(**config) -> list[str]:
    cfg = {"arm": "ur5e", **config}
    return ArmControllerPlugin(cfg).validate_config(cfg)


@pytest.mark.parametrize("key", ["max_velocity", "max_acceleration"])
@pytest.mark.parametrize("value", [0.0, -1.0, "fast", True])
def test_a_limit_that_is_not_a_positive_number_is_refused(key, value):
    errors = _errors(**{key: value})
    assert any("positive number" in e for e in errors), errors
    errors = _errors(joints=JOINTS, **{key: {"elbow_joint": value}})
    assert any("positive number" in e for e in errors), errors


def test_a_limit_for_an_unowned_joint_is_refused():
    """It would be dropped, and the joint the author believes limited would still take steps."""
    errors = _errors(joints=JOINTS, max_velocity={"wrist_3_joint": 1.0})
    assert any("does not list" in e for e in errors), errors


def test_plain_and_per_joint_limits_are_accepted():
    assert _errors(max_velocity=1.0, max_acceleration=2.0) == []
    assert _errors(joints=JOINTS, max_velocity={"elbow_joint": 1.0}) == []


def _arm(tmp_path, **controller):
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": DT},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [{"arm_controller": controller}],
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    plugin = next(p for p in engine.plugins if isinstance(p, ArmControllerPlugin))
    return engine, plugin


def _desired(plugin) -> dict[str, float]:
    state = plugin.controller_state()
    return dict(zip(state.joint_names, state.reference.positions, strict=True))


def test_a_commanded_position_is_reached_on_a_trapezoid_and_exactly(tmp_path):
    engine, arm = _arm(
        tmp_path,
        max_velocity={"shoulder_pan_joint": VMAX},
        max_acceleration={"shoulder_pan_joint": AMAX},
    )
    start = _desired(arm)
    goal = start["shoulder_pan_joint"] + 1.0
    arm.set_targets(["shoulder_pan_joint", "elbow_joint"], [goal, start["elbow_joint"] + 0.1])

    assert _desired(arm)["elbow_joint"] == pytest.approx(start["elbow_joint"] + 0.1), (
        "a joint with no limit still takes its command at once"
    )
    path = []
    for _ in range(round(3.5 / DT)):
        engine.step()
        path.append(_desired(arm)["shoulder_pan_joint"])
    path = np.array([start["shoulder_pan_joint"], *path])
    speed = np.diff(path) / DT
    accel = np.diff(speed) / DT

    arrived_step = int(np.argmax(path == goal))
    assert speed.max() == pytest.approx(VMAX, abs=1e-9)
    assert speed.min() >= -1e-9, "it never runs back"
    # Every step keeps to the limit but the one that puts the target on the goal and stops it.
    assert np.abs(accel[: arrived_step - 2]).max() <= AMAX + 1e-6
    assert np.abs(accel).max() <= 1.5 * AMAX
    assert path.max() <= goal + 1e-12, "no overshoot of the goal"
    assert path[-1] == goal, "it lands on the goal exactly"
    # 1 rad at 0.5 rad/s with a 0.25 s ramp at each end: 2.25 s.
    arrived = arrived_step * DT
    assert arrived == pytest.approx(1.0 / VMAX + VMAX / AMAX, abs=2 * DT)


def test_a_goal_moved_back_mid_ramp_is_approached_without_a_jump(tmp_path):
    engine, arm = _arm(tmp_path, max_velocity=VMAX, max_acceleration=AMAX)
    start = _desired(arm)["shoulder_pan_joint"]
    arm.set_targets(["shoulder_pan_joint"], [start + 1.0])
    for _ in range(round(1.0 / DT)):
        engine.step()
    arm.set_targets(["shoulder_pan_joint"], [start])
    path = []
    for _ in range(round(3.0 / DT)):
        engine.step()
        path.append(_desired(arm)["shoulder_pan_joint"])
    path = np.array(path)
    arrived_step = int(np.argmax(path == start))
    speed = np.diff(path) / DT
    accel = np.abs(np.diff(speed) / DT)
    assert accel[: arrived_step - 2].max() <= AMAX + 1e-6, "the reversal is braked, not stepped"
    assert path[-1] == start


def test_a_limited_test_target_ramps_instead_of_stepping(tmp_path):
    engine, arm = _arm(tmp_path, max_velocity=VMAX, test_target=[0.5, -1.2, 1.0])
    before = _desired(arm)
    engine.step()
    after = _desired(arm)
    for name in JOINTS:
        assert abs(after[name] - before[name]) <= VMAX * DT + 1e-12


def test_a_reset_leaves_nothing_in_motion(tmp_path):
    engine, arm = _arm(tmp_path, max_velocity=VMAX, max_acceleration=AMAX)
    rest = _desired(arm)
    arm.set_targets(["shoulder_pan_joint"], [rest["shoulder_pan_joint"] + 1.0])
    for _ in range(200):
        engine.step()
    engine.reset()
    held = _desired(arm)
    for _ in range(50):
        engine.step()
    assert _desired(arm) == pytest.approx(held), "the old goal does not survive the reset"


def test_a_trajectory_waypoint_and_a_streamed_point_are_limited_alike(tmp_path):
    """The limiter is the drive's: the action's waypoints and the stream's points pass through it."""
    engine, arm = _arm(tmp_path, stream_commands=True, max_velocity=VMAX)
    start = _desired(arm)["shoulder_pan_joint"]
    for command in (arm.follow_joint_trajectory, arm.joint_command):
        before = _desired(arm)["shoulder_pan_joint"]
        command(["shoulder_pan_joint"], np.array([before + 1.0]))
        engine.step()
        assert _desired(arm)["shoulder_pan_joint"] - before == pytest.approx(VMAX * DT)
    assert _desired(arm)["shoulder_pan_joint"] > start


def test_a_velocity_command_is_clipped_to_the_limit_and_brakes_when_it_stops(tmp_path):
    engine, arm = _arm(
        tmp_path,
        velocity_commands=True,
        velocity_timeout_s=0.1,
        max_velocity=VMAX,
        max_acceleration=AMAX,
    )
    start = _desired(arm)["shoulder_pan_joint"]
    path = []
    for _ in range(round(0.5 / DT)):
        arm.set_velocities(["shoulder_pan_joint"], [4 * VMAX])
        engine.step()
        path.append(_desired(arm)["shoulder_pan_joint"])
    for _ in range(round(1.0 / DT)):
        engine.step()
        path.append(_desired(arm)["shoulder_pan_joint"])
    speed = np.diff(np.array([start, *path])) / DT
    assert speed.max() <= VMAX + 1e-9, "a command four times the limit moves no faster than it"
    assert speed.min() >= -1e-9, "it brakes onto the integrated goal rather than running back"
    assert speed[-1] == 0.0, "and then stands"
