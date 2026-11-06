"""``ur_driver_interface`` and ``arm_controller``'s ``controller_type``: a simulated UR5e answering to
the controller names, types and services ``ur_robot_driver`` gives the real one, with the speed
slider slowing the scaled trajectory controller as it slows the real one."""

from __future__ import annotations

import math

import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin
from roqsim_manipulation.plugins.ur_driver_interface import URDriverInterfacePlugin

from roqsim.config import load_config_from_dict
from roqsim.controllers import ACTIVE, BEST_EFFORT, INACTIVE, STRICT, registry_for
from roqsim.engine import Engine

SCALED = "ur_controllers/ScaledJointTrajectoryController"
JOINT = "shoulder_pan_joint"


def _engine(controller_type: str | None = SCALED, **driver) -> Engine:
    arm = {"controller_name": "scaled_joint_trajectory_controller"}
    if controller_type is not None:
        arm["controller_type"] = controller_type
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {"arm_controller": arm, "name": "arm_controller"},
                        {
                            "force_torque": {
                                "site": "fts_site",
                                "controller_name": "force_torque_sensor_broadcaster",
                            },
                            "name": "force_torque_sensor_broadcaster",
                        },
                        {
                            "ur_driver_interface": {
                                "ft": "force_torque_sensor_broadcaster",
                                **driver,
                            },
                            "name": "ur_driver",
                        },
                    ],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    for _ in range(50):
        engine.step()
    return engine


def _plugin(engine, cls):
    return next(p for p in engine.plugins if isinstance(p, cls))


def test_the_controller_manager_lists_the_drivers_names_and_types():
    engine = _engine()
    listed = {c.name: (c.type, c.state) for c in registry_for(engine.ctx).all()}
    assert listed["scaled_joint_trajectory_controller"] == (SCALED, ACTIVE)
    assert listed["joint_state_broadcaster"][1] == ACTIVE
    assert listed["force_torque_sensor_broadcaster"][1] == ACTIVE
    assert listed["speed_scaling_state_broadcaster"] == (
        "ur_controllers/SpeedScalingStateBroadcaster",
        ACTIVE,
    )
    assert listed["io_and_status_controller"] == ("ur_controllers/GPIOController", ACTIVE)
    # Listed inactive as the driver lists it, so MoveIt chooses the scaled controller.
    assert listed["joint_trajectory_controller"] == (
        "joint_trajectory_controller/JointTrajectoryController",
        INACTIVE,
    )


def test_activating_the_unscaled_controller_is_refused_naming_the_scaled_one():
    engine = _engine()
    registry = registry_for(engine.ctx)
    ok, message = registry.switch(
        activate=["joint_trajectory_controller"],
        deactivate=["scaled_joint_trajectory_controller"],
        strictness=STRICT,
    )
    assert not ok
    assert "use scaled_joint_trajectory_controller" in message
    # Refused before anything changed: the hand-over did not half-happen.
    assert registry.get("joint_trajectory_controller").state == INACTIVE
    assert registry.get("scaled_joint_trajectory_controller").state == ACTIVE
    assert registry.get("joint_trajectory_controller").transitions == []


def test_a_best_effort_refusal_still_names_the_reason_and_changes_nothing():
    engine = _engine()
    registry = registry_for(engine.ctx)
    _, message = registry.switch(activate=["joint_trajectory_controller"], strictness=BEST_EFFORT)
    assert "use scaled_joint_trajectory_controller" in message
    assert registry.get("joint_trajectory_controller").state == INACTIVE


def _steps(engine, seconds: float) -> None:
    for _ in range(round(seconds / engine.ctx.model.opt.timestep)):
        engine.step()


def _run_line(engine, seconds: float):
    """Start a 2 s straight line of the shoulder over 0.4 rad in process, run *seconds*, and return
    ``(start, arm)``."""
    arm = _plugin(engine, ArmControllerPlugin)
    q0 = arm._target[JOINT]
    engine.ctx.blackboard.get("arm:ur5e").follow_trajectory([JOINT], [0.0, 2.0], [[q0], [q0 + 0.4]])
    _steps(engine, seconds)
    return q0, arm


def _reached(arm, q0: float) -> float:
    """The trajectory time the held target has reached, read off where it is along the line."""
    return (arm._target[JOINT] - q0) / 0.4 * 2.0


def test_the_speed_scaling_starts_at_100_percent():
    engine = _engine()
    assert _plugin(engine, URDriverInterfacePlugin).speed_scaling() == 100.0
    assert engine.ctx.blackboard.get("arm:ur5e").trajectory_time_scale() == 1.0


def test_the_slider_at_half_halves_the_trajectory_progress_per_sim_second():
    full, half = _engine(), _engine()
    assert _plugin(half, URDriverInterfacePlugin).set_speed_slider(0.5) is True
    for engine, expected in ((full, 1.0), (half, 0.5)):
        q0, arm = _run_line(engine, 1.0)
        assert _reached(arm, q0) == pytest.approx(expected, abs=2e-3)


def test_a_slider_change_mid_trajectory_keeps_the_time_travelled():
    engine = _engine()
    q0, arm = _run_line(engine, 0.5)
    assert _reached(arm, q0) == pytest.approx(0.5, abs=2e-3)
    _plugin(engine, URDriverInterfacePlugin).set_speed_slider(0.25)
    _steps(engine, 1.0)
    assert _reached(arm, q0) == pytest.approx(0.75, abs=2e-3)


def test_the_unscaled_controller_type_is_not_slowed_by_the_slider():
    engine = _engine(controller_type=None)
    assert _plugin(engine, URDriverInterfacePlugin).set_speed_slider(0.5) is True
    assert engine.ctx.blackboard.get("arm:ur5e").trajectory_time_scale() == 1.0
    q0, arm = _run_line(engine, 1.0)
    assert _reached(arm, q0) == pytest.approx(1.0, abs=2e-3)


def test_speed_scaling_publishes_the_factor_in_effect():
    engine = _engine()
    driver = _plugin(engine, URDriverInterfacePlugin)
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "set_speed_slider")
    future = endpoint.write({"speed_slider_fraction": 0.25})  # as the bridge's service hands it
    engine.step()
    assert future.result(1.0) is True
    assert driver.speed_scaling() == 25.0
    assert engine.ctx.blackboard.get("arm:ur5e").trajectory_time_scale() == 0.25


@pytest.mark.parametrize("fraction", [0.0, 0.005, -0.5, 1.0001, 2.0, math.nan])
def test_a_fraction_outside_the_drivers_range_is_refused_and_changes_nothing(fraction):
    engine = _engine()
    driver = _plugin(engine, URDriverInterfacePlugin)
    assert driver.set_speed_slider(0.5) is True
    assert driver.set_speed_slider(fraction) is False
    assert driver.speed_scaling() == 50.0


@pytest.mark.parametrize("fraction", [0.01, 1.0])
def test_the_ends_of_the_drivers_range_are_accepted(fraction):
    driver = _plugin(_engine(), URDriverInterfacePlugin)
    assert driver.set_speed_slider(fraction) is True
    assert driver.speed_scaling() == pytest.approx(100.0 * fraction)


def test_a_reset_returns_the_slider_to_100_percent():
    engine = _engine()
    driver = _plugin(engine, URDriverInterfacePlugin)
    driver.set_speed_slider(0.3)
    engine.reset()
    assert driver.speed_scaling() == 100.0


def test_zero_ftsensor_tares_the_named_sensor():
    engine = _engine()
    reader = engine.ctx.blackboard.get("ft:force_torque_sensor_broadcaster")
    before, _ = reader.read()
    assert np.linalg.norm(before) > 1.0  # the flange's own load, untared
    _plugin(engine, URDriverInterfacePlugin).zero_ftsensor()
    after, _ = reader.read()
    assert np.allclose(after, 0.0, atol=1e-9)


def test_a_sensor_that_is_not_there_is_refused():
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [{"ur_driver_interface": {"ft": "nope"}, "name": "ur_driver"}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    with pytest.raises(Exception, match="no force_torque sensor labelled 'nope'"):
        engine.setup()


@pytest.mark.parametrize(
    "config, fragment",
    [({}, "'ft' is required"), ({"ft": "x", "rate_hz": 0}, "'rate_hz' must be > 0")],
)
def test_config_is_checked(config, fragment):
    errors = URDriverInterfacePlugin(config).validate_config(config)
    assert any(fragment in e for e in errors)


def test_an_unknown_trajectory_controller_type_is_refused():
    config = {"controller_type": "my_controllers/Teleport"}
    errors = ArmControllerPlugin(config).validate_config(config)
    assert any("controller_type" in e for e in errors)
