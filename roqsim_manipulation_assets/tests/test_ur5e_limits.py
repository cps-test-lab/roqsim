# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The UR5e's joint torque limits are Universal Robots' own, as the model's actuator forcerange.

150 N*m on the size-3 joints (base, shoulder, elbow) and 28 N*m on the size-1 wrists
(Universal_Robots_ROS2_Description, config/ur5e/joint_limits.yaml, max_effort).
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine

#: UR's max_effort per joint, N*m.
TORQUE = {
    "shoulder_pan": 150.0,
    "shoulder_lift": 150.0,
    "elbow": 150.0,
    "wrist_1": 28.0,
    "wrist_2": 28.0,
    "wrist_3": 28.0,
}


def _engine(tmp_path):
    world = tmp_path / "ur5e.yaml"
    world.write_text(
        "sim: {timestep: 0.002}\ncomponents:\n"
        '  - spawn_arm: {model: ur5e, prefix: "ur5e_"}\n'
        "    name: ur5e\n",
        encoding="utf-8",
    )
    engine = Engine(load_config(world))
    engine.setup()
    return engine


def test_torque_limits_are_urs(tmp_path):
    engine = _engine(tmp_path)
    try:
        model = engine.ctx.model
        for name, limit in TORQUE.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ur5e_" + name)
            assert list(model.actuator_forcerange[aid]) == pytest.approx([-limit, limit]), name
    finally:
        engine.shutdown()
