# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""After a reset, the sensors read the arm holding its pose, before the first step as after it.

A reset zeroes every actuator command, and the arm controller wrote its own only at its first
step. Until then the physics state was the arm at its home pose with every position servo pulling
toward zero, so a wrench read at that moment was a transient of tens of newtons -- and a sensor
tared then carried it as a bias through the whole trial, with nothing reported.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_sensors.plugins.force_torque import ForceTorquePlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _arm_with_wrench():
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {"force_torque": {"site": "fts_site", "invert": False}, "name": "ft"}
                    ],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    ft = next(p for p in engine.plugins if isinstance(p, ForceTorquePlugin))
    return engine, ft


def test_the_wrench_before_the_first_step_is_the_wrench_at_rest():
    engine, ft = _arm_with_wrench()
    first = np.array(ft.read()[0])
    for _ in range(200):
        engine.step()
    settled = np.array(ft.read()[0])
    engine.shutdown()
    assert np.linalg.norm(settled) > 1.0, "the tool's own weight is on the sensor"
    assert np.linalg.norm(first - settled) < 0.5, (
        f"read {first} right after the reset, {settled} at rest"
    )


def test_a_sensor_tared_at_the_reset_reads_nothing_at_rest():
    """The consequence that mattered: zeroing the sensor at the reset must zero the static load."""
    engine, ft = _arm_with_wrench()
    ft.tare()
    for _ in range(200):
        engine.step()
    residual = np.linalg.norm(ft.read()[0])
    engine.shutdown()
    assert residual == pytest.approx(0.0, abs=0.5)
