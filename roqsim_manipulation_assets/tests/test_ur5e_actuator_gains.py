# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""What the gains a world declares actually do to a real arm.

``roqsim/tests/test_actuator_override.py`` checks the table and the refusals against a two-joint
probe; this checks that the numbers reach the solver and behave like the law they name. The UR5e is
the case the feature exists for: it ships a servo of 2000 N*m/rad, and a published controller's joint
compliance can be three orders of magnitude softer.

Every number an assertion turns on is a named constant here rather than a literal in the test, so a
tolerance that has to move is a decision someone makes once and can see.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine

#: The arm's own servo, from ur5e.xml's `<default class="ur5e">`. Not a target -- the baseline the
#: overrides below are measured against.
MODEL_P, MODEL_D = 2000.0, 400.0

#: A compliance far below the model's, of the order a published joint PD states.
SOFT_STIFFNESS, SOFT_DAMPING = 2.0, 0.02

#: The pair the inverse-stiffness check uses. A decade apart, so the ratio is unambiguous.
STIFF_LO, STIFF_HI, STIFF_D = 300.0, 3000.0, 20.0

#: Steady-state joint error goes as 1/p for a fixed load, so a decade of gain is a decade of error.
#: The band is wide on the upper side because the arm's own gravity torque is configuration
#: dependent: at the softer gain it sags further, which raises the load it is sagging under.
RATIO_LO, RATIO_HI = 8.0, 13.0

#: Long enough for the arm to stop moving at the softest gain here.
SETTLE_STEPS = 4000

#: An uncompensated drive of this stiffness cannot hold a UR5e up. Measured at ~2.3 rad; the bar
#: only has to separate "folded" from "held".
SAG_RAD = 1.0

#: What "held" means for the same stiffness on a drive that carries its own weight -- a
#: milliradian is already far below any joint tolerance a task states.
HELD_RAD = 1e-3

#: The arm's weight put back as the load, so a gain can be measured against something. A drive
#: that carries its own weight is the default (:mod:`roqsim.actuators`), which leaves a joint at
#: its commanded angle whatever the gain -- correct, and nothing to measure a gain by.
UNCOMPENSATED = "gravity_compensation: false"

_WORLD = """
sim: {{timestep: 0.002, gravity: [0.0, 0.0, -9.81]}}
components:
  - spawn_arm: {{model: ur5e, prefix: "ur5e_", actuators: {actuators}, {extra}}}
    name: ur5e
"""


def _settled(tmp_path, actuators: str, steps: int = SETTLE_STEPS, extra: str = ""):
    """|q - q_des| per joint after the arm has been asked to hold its home pose.

    The target is the home vector `spawn_arm` applies on reset and `arm_controller` then holds, so
    this measures the plant the gains describe rather than anything the test commands by hand.
    """
    world = tmp_path / "arm.yaml"
    world.write_text(_WORLD.format(actuators=actuators, extra=extra), encoding="utf-8")
    engine = Engine(load_config(world))
    try:
        engine.setup()
        engine.reset()
        for _ in range(steps):
            engine.step()
        _, position, _, _ = engine.ctx.blackboard.require("arm:ur5e").read_state()
        target = engine.plugins[0]._home_vector()[:6]
        return np.abs(np.asarray(position[:6]) - np.asarray(target))
    finally:
        engine.shutdown()


def test_steady_state_error_scales_inversely_with_stiffness(tmp_path):
    """The gain a world declares is the gain the joint runs under -- measured, not read back.

    Measured on an uncompensated arm, because a gain is only visible against a load and this is
    the load the arm brings with it. A drive that carries its own weight -- the default -- sits at
    its commanded angle whatever its gain, which is the point of it and leaves nothing to divide.
    """
    soft = _settled(tmp_path, f"{{control: position, p: {STIFF_LO}, d: {STIFF_D}}}",
                    extra=UNCOMPENSATED).max()
    stiff = _settled(tmp_path, f"{{control: position, p: {STIFF_HI}, d: {STIFF_D}}}",
                     extra=UNCOMPENSATED).max()
    ratio = soft / stiff
    assert RATIO_LO < ratio < RATIO_HI, (
        f"p {STIFF_LO} sagged {soft:.5f} rad and p {STIFF_HI} sagged {stiff:.5f} rad, a ratio of "
        f"{ratio:.2f}. A decade of gain should be about a decade of steady-state error; it is not, "
        f"so the declared gain is not what reached the solver."
    )


def test_a_drive_that_does_not_carry_its_weight_folds(tmp_path):
    """What ``gravity_compensation: false`` buys, and the baseline the next test contrasts with.

    This is a real machine too -- a small hobby servo, a backdrivable joint, any drive with no
    gravity term of its own -- and at this stiffness it cannot hold a UR5e out. It is not what a
    UR5e is, which is why it has to be asked for.
    """
    sag = _settled(tmp_path, f"{{control: position, p: {SOFT_STIFFNESS}, d: {SOFT_DAMPING}}}",
                   extra=UNCOMPENSATED).max()
    assert sag > SAG_RAD, (
        f"an uncompensated servo of {SOFT_STIFFNESS} N*m/rad held the arm to {sag:.3f} rad. It "
        f"should not be able to: that gain cannot carry the arm's own weight unaided."
    )


@pytest.mark.parametrize("law", [
    f"{{control: position, p: {SOFT_STIFFNESS}, d: {SOFT_DAMPING}}}",
    f"{{control: impedance, stiffness: {SOFT_STIFFNESS}, damping: {SOFT_DAMPING}}}",
])
def test_a_drive_that_carries_its_weight_holds_the_pose_at_that_stiffness(tmp_path, law):
    """Same stiffness as above, opposite outcome -- and both laws, because both model a drive.

    A real joint holds its own weight inside its own loop, so its gain says how hard it resists a
    DISTURBANCE, not how much of the arm it can carry. That is true of an industrial position
    servo and of a compliance controller alike; what separates ``position`` from ``impedance`` is
    the law and the units its gains are stated in, not which of them fights gravity.
    """
    held = _settled(tmp_path, law).max()
    assert held < HELD_RAD, (
        f"{law} drifted {held:.5f} rad from the commanded pose; the body-level gravity term is not "
        f"reaching the compiled model."
    )


def test_the_stock_model_is_untouched_when_no_gains_are_declared(tmp_path):
    """A world that declares nothing must compile the arm exactly as the package ships it."""
    world = tmp_path / "stock.yaml"
    world.write_text(
        "sim: {timestep: 0.002}\ncomponents:\n"
        '  - spawn_arm: {model: ur5e, prefix: "ur5e_"}\n    name: ur5e\n',
        encoding="utf-8",
    )
    engine = Engine(load_config(world))
    try:
        engine.setup()
        model = engine.ctx.model
        assert model.actuator_gainprm[0][0] == pytest.approx(MODEL_P)
        assert model.actuator_biasprm[0][1] == pytest.approx(-MODEL_P)
        assert model.actuator_biasprm[0][2] == pytest.approx(-MODEL_D)
        # Nothing asked for gravity compensation, so no body carries it but the world's -- which
        # `roqsim.presence` marks for every world, and which cannot move anyway.
        rows = engine.ctx.actuator_tables["ur5e"]
        assert [row.source for row in rows] == ["model"] * len(rows)
    finally:
        engine.shutdown()


def test_the_effort_limit_a_world_declares_reaches_the_model(tmp_path):
    world = tmp_path / "limit.yaml"
    world.write_text(
        "sim: {timestep: 0.002}\ncomponents:\n"
        '  - spawn_arm: {model: ur5e, prefix: "ur5e_", actuators: {control: impedance, '
        "stiffness: 5.0, damping: 1.0, each: {shoulder_lift: {effort_limit: 150}}}}\n"
        "    name: ur5e\n",
        encoding="utf-8",
    )
    engine = Engine(load_config(world))
    try:
        engine.setup()
        model = engine.ctx.model
        limits = dict(zip(
            [model.actuator(i).name for i in range(model.nu)], model.actuator_forcerange
        ))
        assert list(limits["ur5e_shoulder_lift"]) == pytest.approx([-150.0, 150.0])
        # Its neighbour keeps the model's own limit: an `each:` entry changes one actuator.
        assert list(limits["ur5e_shoulder_pan"]) == pytest.approx([-120.0, 120.0])
    finally:
        engine.shutdown()


def test_a_grippers_tendon_actuator_is_out_of_scope(tmp_path):
    """The regression the insertion point exists for.

    `actuators:` is resolved before the end effector is grafted into the arm's spec, so a shared
    joint law lands on the arm's own actuators and never on the gripper's tendon -- which has no
    joint stiffness, and whose presence would otherwise refuse a block that was only ever about the
    arm's joints.
    """
    world = tmp_path / "gripper.yaml"
    world.write_text(
        "sim: {timestep: 0.002}\ncomponents:\n"
        '  - spawn_arm: {model: ur5e, prefix: "ur5e_", end_effector: {model: robotiq_2f85}, '
        f"actuators: {{control: impedance, stiffness: {SOFT_STIFFNESS}, damping: {SOFT_DAMPING}}}}}\n"
        "    name: ur5e\n",
        encoding="utf-8",
    )
    engine = Engine(load_config(world))
    try:
        engine.setup()
        model = engine.ctx.model
        names = [model.actuator(i).name for i in range(model.nu)]
        assert "ur5e_fingers_actuator" in names, "the gripper lost its actuator"
        table = {row.name: row for row in engine.ctx.actuator_tables["ur5e"]}
        assert "ur5e_fingers_actuator" not in table, "the gripper's tendon was rewritten as a joint"
        assert len(table) == 6 and {r.control for r in table.values()} == {"impedance"}
    finally:
        engine.shutdown()
