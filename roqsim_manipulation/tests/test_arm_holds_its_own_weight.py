# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""An arm under a position servo stands where it was sent, because a real one does.

A position or velocity drive holds its own weight inside its own loop: its gain says how hard the
joint resists a *disturbance*, not how much of the arm it can carry. MuJoCo's actuator of that name
carries nothing, so without ``body_gravcomp`` the servo trades position error for holding torque
and the arm settles below where it was commanded -- by 9 mm on a UR5e at its shipped gains, and by
more as the gain falls. Nothing reports it: the pose is wrong, the model compiles, the run
succeeds.

``effort`` is the exception and stays uncompensated, because supplying the gravity term is exactly
what a torque controller is for; compensating it here would answer the question such an experiment
is asking. ``gravity_compensation:`` states either case explicitly.
"""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

HOLD_S = 6.0

#: A torque-commanded arm. The ctrlrange comes with it because the model's is in radians and
#: switching the unit without restating it is refused -- see :mod:`roqsim.actuators`.
EFFORT = {"control": "effort", "ctrlrange": [-330.0, 330.0]}


def _droop_mm(model: str, prefix: str, **arm) -> float:
    """How far the flange falls over :data:`HOLD_S`, commanded to the pose it starts in."""
    cfg = load_config_from_dict({"sim": {}, "components": [
        {"spawn_arm": {"model": model, "prefix": prefix, **arm}, "name": "arm"}]})
    engine = Engine(cfg)
    engine.setup()
    engine.reset()

    model_, data = engine.ctx.model, engine.ctx.data
    # Command the configuration the arm is standing in: "hold still", stated the way a servo
    # takes it. Anything else measures the arm moving somewhere, which is a different question.
    for i in range(model_.nu):
        jid = model_.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model_.jnt_qposadr[jid]]

    site = model_.site(f"{prefix}attachment_site").id
    z0 = float(data.site_xpos[site][2])
    for _ in range(int(HOLD_S / model_.opt.timestep)):
        engine.step()
    fell = (float(data.site_xpos[site][2]) - z0) * 1e3
    engine.shutdown()
    return fell


@pytest.mark.parametrize("gain", [2000.0, 200.0])
def test_a_position_servo_holds_the_pose_it_was_given(gain):
    """Both a stiff and a soft gain, because the point is that the gain does not decide this.

    Uncompensated the two differ by an order of magnitude in how far the arm falls, which is what
    made a gain read as a load rating.
    """
    assert abs(_droop_mm("ur10e", "ur10e_",
                         actuators={"control": "position", "p": gain})) < 1.0


def test_the_shipped_arm_holds_too():
    """The default path: a world that names a model and no ``actuators:`` at all.

    This is what an experiment writes, and what a reconstruction ran against for a task whose
    tolerance was a millimetre.
    """
    assert abs(_droop_mm("ur5e", "ur5e_")) < 1.0


def test_an_uncompensated_arm_falls_and_the_key_is_what_says_so():
    """The opt-out works, and the amount is the reason the default changed.

    Stated rather than merely allowed: an experiment on a drooping arm is a real experiment, and
    it should be visible in the world document rather than implied by the actuator law.
    """
    fell = _droop_mm("ur10e", "ur10e_",
                     actuators={"control": "position", "p": 200.0},
                     gravity_compensation=False)
    assert fell < -100.0, "a soft position servo alone does not hold a UR10e up"


def test_a_torque_controlled_arm_is_left_alone():
    """``effort`` is not compensated, because the gravity term is the controller's to supply.

    Compensating it would make a controller that omits gravity look as good as one that has it,
    which is the comparison such an experiment exists to make.
    """
    cfg = load_config_from_dict({"sim": {}, "components": [
        {"spawn_arm": {"model": "ur10e", "prefix": "ur10e_",
                       "actuators": EFFORT}, "name": "arm"}]})
    engine = Engine(cfg)
    engine.setup()
    # 1 is the world-body marker that makes the field writable at run time; a compensated arm
    # counts its own bodies on top.
    assert int(engine.ctx.model.ngravcomp) == 1
    engine.shutdown()


def test_a_torque_controlled_arm_can_ask_for_it():
    """...and the same key says so, for a controller that is told its gravity term."""
    cfg = load_config_from_dict({"sim": {}, "components": [
        {"spawn_arm": {"model": "ur10e", "prefix": "ur10e_",
                       "actuators": EFFORT,
                       "gravity_compensation": True}, "name": "arm"}]})
    engine = Engine(cfg)
    engine.setup()
    assert int(engine.ctx.model.ngravcomp) > 1
    engine.shutdown()


def test_the_effort_report_still_carries_the_holding_torque():
    """A compensated arm's joint torque is real torque, and a driver must still report it.

    ``body_gravcomp`` supplies the holding term outside the actuator, so ``qfrc_actuator`` alone
    would report a motor doing nothing while the arm hangs off it -- and a monitor watching effort
    for a collision would see a quieter arm than exists.
    """
    cfg = load_config_from_dict({"sim": {}, "components": [
        {"spawn_arm": {"model": "ur5e", "prefix": "ur5e_"}, "name": "ur5e"}]})
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(500):
        engine.step()
    _, _, _, effort = engine.ctx.blackboard.require("arm:ur5e").read_state()
    engine.shutdown()

    assert max(abs(e) for e in effort) > 1.0, "holding a UR5e out costs more than a newton-metre"
