# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A world states a gripper's finger servo stiffness and damping; without it the model's hold.

The Franka Hand is part of the panda model, so its servo is named under the arm's ``actuators:``;
a gripper welded on with ``end_effector:`` is named under ``end_effector.actuators``. The table and
the refusals are ``roqsim/tests/test_actuator_override.py``'s; this checks that the numbers reach
the compiled model and the recorded table.
"""

from __future__ import annotations

import pytest
from roqsim_manipulation.plugins.spawn_arm import SpawnArmPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

#: panda.xml's finger servo: kp 2000 N/m, kv 40 N*s/m, its 0..255 command scaled onto 0..0.04 m.
PANDA_P, PANDA_D, PANDA_SCALE = 2000.0, 40.0, 0.04 / 255
#: robotiq_2f85.xml's: kp 100, kv 10, from Menagerie, with a 0..255 command and its gain as stated.
ROBOTIQ_P, ROBOTIQ_D, ROBOTIQ_GAIN = 100.0, 10.0, 0.3137255


def _servo(tmp_path, spawn: dict, actuator: str, entity: str):
    """The compiled finger servo's (gain, stiffness, damping) and its row in the run's table."""
    cfg = load_config_from_dict(
        {"sim": {"timestep": 0.001}, "components": [{"spawn_arm": spawn, "name": entity}]},
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    try:
        engine.setup()
        model = engine.ctx.model
        i = model.actuator(actuator).id
        gains = (
            float(model.actuator_gainprm[i][0]),
            -float(model.actuator_biasprm[i][1]),
            -float(model.actuator_biasprm[i][2]),
        )
        (row,) = [r for r in engine.ctx.actuator_tables[entity] if r.name == actuator]
        return gains, row
    finally:
        engine.shutdown()


def test_the_panda_hand_keeps_its_own_servo_when_the_world_states_none(tmp_path):
    (gain, p, d), row = _servo(tmp_path, {"model": "panda"}, "actuator8", "panda")
    assert (p, d) == (PANDA_P, PANDA_D)
    assert gain == pytest.approx(PANDA_P * PANDA_SCALE)
    assert (row.p, row.d, row.source) == (PANDA_P, PANDA_D, "model")


def test_the_panda_hand_takes_the_worlds_servo(tmp_path):
    spawn = {"model": "panda", "actuators": {"each": {"actuator8": {"p": 3000.0, "d": 50.0}}}}
    (gain, p, d), row = _servo(tmp_path, spawn, "actuator8", "panda")
    assert (p, d) == (3000.0, 50.0)
    # The command scale is kept, so a commanded width is still the width the fingers close to.
    assert gain == pytest.approx(3000.0 * PANDA_SCALE)
    assert (row.p, row.d, row.source) == (3000.0, 50.0, "each")


def test_a_welded_gripper_keeps_its_own_servo_when_the_world_states_none(tmp_path):
    spawn = {"model": "ur5e", "prefix": "u_", "end_effector": {"model": "robotiq_2f85"}}
    (_, p, d), row = _servo(tmp_path, spawn, "u_fingers_actuator", "ur5e")
    assert (p, d) == (ROBOTIQ_P, ROBOTIQ_D)
    assert row.source == "model"


def test_a_welded_gripper_takes_the_worlds_servo_under_its_own_prefix(tmp_path):
    spawn = {
        "model": "ur5e",
        "prefix": "u_",
        "actuators": {"control": "impedance", "stiffness": 5.0, "damping": 1.0},
        "end_effector": {
            "model": "robotiq_2f85",
            "prefix": "g_",
            "actuators": {"each": {"fingers_actuator": {"p": 300.0, "d": 20.0}}},
        },
    }
    (gain, p, d), row = _servo(tmp_path, spawn, "u_g_fingers_actuator", "ur5e")
    assert (p, d) == (300.0, 20.0)
    assert gain == pytest.approx(ROBOTIQ_GAIN * 300.0 / ROBOTIQ_P)
    assert (row.control, row.source) == ("position", "each")


def test_an_actuator_the_gripper_does_not_have_is_refused_naming_its_actuators(tmp_path):
    spawn = {
        "model": "ur5e",
        "end_effector": {"model": "robotiq_2f85", "actuators": {"each": {"actuator8": {"p": 1.0}}}},
    }
    with pytest.raises(PluginError) as exc:
        _servo(tmp_path, spawn, "fingers_actuator", "ur5e")
    message = str(exc.value)
    assert "'end_effector.actuators.each.actuator8': robotiq_2f85 has no actuator" in message
    assert "fingers_actuator" in message


def test_the_gripper_block_is_checked_before_anything_is_compiled():
    config = {
        "model": "ur5e",
        "end_effector": {
            "model": "robotiq_2f85",
            "actuators": {"each": {"fingers_actuator": {"kp": 300.0, "d": -1.0}}},
        },
    }
    errors = SpawnArmPlugin(config).config_errors(config)
    assert any(
        e.startswith("'end_effector.actuators.each.fingers_actuator.kp' is not a key")
        for e in errors
    ), errors
    assert any(
        "'end_effector.actuators.each.fingers_actuator.d' must be >= 0.0" in e for e in errors
    )
