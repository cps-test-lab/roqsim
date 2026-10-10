"""``ur_driver_interface`` and ``arm_controller``'s ``controller_type``: a simulated UR5e answering to
the controller names, types and services ``ur_robot_driver`` gives the real one."""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin
from roqsim_manipulation.plugins.ur_driver_interface import URDriverInterfacePlugin

from roqsim.config import load_config_from_dict
from roqsim.controllers import ACTIVE, INACTIVE, registry_for
from roqsim.engine import Engine

SCALED = "ur_controllers/ScaledJointTrajectoryController"


def _engine(**driver) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {
                            "arm_controller": {
                                "controller_name": "scaled_joint_trajectory_controller",
                                "controller_type": SCALED,
                            },
                            "name": "arm_controller",
                        },
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
    with pytest.raises(RuntimeError, match="use scaled_joint_trajectory_controller"):
        registry_for(engine.ctx).switch(
            activate=["joint_trajectory_controller"], deactivate=[], strictness=1
        )


def test_speed_scaling_is_the_stated_percentage():
    assert _plugin(_engine(), URDriverInterfacePlugin).speed_scaling() == 100.0
    assert _plugin(_engine(speed_scaling=40.0), URDriverInterfacePlugin).speed_scaling() == 40.0


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
    [({}, "'ft' is required"), ({"ft": "x", "speed_scaling": 150}, "percentage")],
)
def test_config_is_checked(config, fragment):
    errors = URDriverInterfacePlugin(config).validate_config(config)
    assert any(fragment in e for e in errors)


def test_an_unknown_trajectory_controller_type_is_refused():
    config = {"controller_type": "my_controllers/Teleport"}
    errors = ArmControllerPlugin(config).validate_config(config)
    assert any("controller_type" in e for e in errors)
