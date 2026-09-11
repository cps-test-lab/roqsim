# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A Cartesian command reaches the actuators in the step that computed it.

`cartesian_admittance` writes joint targets through the `ArmHandle`; `arm_controller` writes those
targets into `data.ctrl`. Both run in `pre_step`, in world-YAML order, and the Cartesian controller
can only be declared AFTER the arm -- its `configure` requires the handle to exist. So the arm wrote
the targets it was holding and the Cartesian controller computed the next ones immediately after,
every tick, and each one reached the actuators a step late.

Nothing reported it, and a phase lag comparable to the loop's own time constant is not visible as a
lag: it reads as a controller that is slightly softer than the one that was configured.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_manipulation.plugins.cartesian_admittance import CartesianAdmittancePlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _world(tmp_path):
    return load_config_from_dict(
        {
            # The control rate equals the physics rate, so the law ticks on EVERY step and every
            # step is a chance for the command to arrive late. At a slower rate the ticks are
            # sparse and a lag hides in the steps that legitimately hold.
            "sim": {"timestep": 0.001},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {"arm_controller": {}},
                        {"force_torque": {"site": "fts_site", "frame": "world"}, "name": "ft"},
                        {
                            "cartesian_admittance": {
                                "controller_type": "cartesian_force_controller",
                                "site": "tool_site",
                                "ft": "ft",
                                "rate_hz": 1000.0,
                                "target_wrench": [0.0, 0.0, -10.0, 0.0, 0.0, 0.0],
                            },
                            "name": "compliance",
                        },
                    ],
                }
            ],
        },
        base_dir=tmp_path,
    )


def _plain_world(tmp_path, ctrl=None):
    """An arm with no Cartesian controller: nothing else commands it, so what reaches `ctrl` is
    only ever what the caller set."""
    return load_config_from_dict(
        {
            "sim": {"timestep": 0.001},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [{"arm_controller": dict(ctrl or {})}],
                }
            ],
        },
        base_dir=tmp_path,
    )


def test_a_cartesian_command_reaches_ctrl_in_the_step_that_computed_it(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    law = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))

    # Nothing but the arm is actuated, and `_q_target` is in the same actuator order.
    assert engine.ctx.model.nu == 6

    for _ in range(50):
        engine.step()
        commanded = np.array(engine.ctx.data.ctrl[:6], dtype=float)
        assert law._q_target is not None, "the law must be driving for this to test anything"
        assert law._q_target == pytest.approx(commanded, abs=1e-12), (
            "the actuators are holding a target from an earlier step"
        )


def test_two_command_sources_on_one_arm_are_refused_naming_both(tmp_path):
    """One arm takes its targets from one controller. Two would each overwrite the other within a
    step, and which won would be decided by their order in the world file."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    arm = engine.ctx.blackboard.require("arm:ur5e")

    with pytest.raises(RuntimeError, match="already does"):
        arm.set_command_source(lambda ctx: None, "a-second-controller")


def test_a_source_may_re_register_itself(tmp_path):
    """Re-registering the same owner is not a conflict -- `configure` runs again on a rebuild."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    arm = engine.ctx.blackboard.require("arm:ur5e")
    law = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))

    arm.set_command_source(law.update, law.label or "cartesian_admittance")  # no raise


# -- taking and releasing the arm ----------------------------------------------------------------


def test_an_inactive_arm_controller_holds_its_target_and_takes_no_new_one(tmp_path):
    """What a deactivated ros2_control controller does: the interfaces are not claimed, so a
    command has nowhere to land -- but the joints hold rather than falling under gravity."""
    engine = Engine(_plain_world(tmp_path))
    engine.setup()
    engine.reset()
    arm = engine.ctx.blackboard.require("arm:ur5e")
    for _ in range(20):
        engine.step()

    held = np.array(engine.ctx.data.ctrl[:6], dtype=float)
    arm.set_active(False)
    arm.set_targets(arm.joint_names, [0.9] * len(arm.joint_names))
    for _ in range(20):
        engine.step()

    assert np.array(engine.ctx.data.ctrl[:6]) == pytest.approx(held), (
        "an inactive controller must neither take the command nor slacken"
    )

    arm.set_active(True)
    arm.set_targets(arm.joint_names, [0.9] * len(arm.joint_names))
    engine.step()
    assert np.array(engine.ctx.data.ctrl[:6]) == pytest.approx([0.9] * 6), (
        "an activated controller takes commands again"
    )


def test_initial_state_inactive_comes_up_not_holding_the_arm(tmp_path):
    engine = Engine(_plain_world(tmp_path, {"initial_state": "inactive"}))
    engine.setup()
    assert engine.ctx.blackboard.require("arm:ur5e").is_active() is False


def test_a_controller_with_no_initial_state_is_active(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    assert engine.ctx.blackboard.require("arm:ur5e").is_active() is True
