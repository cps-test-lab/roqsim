# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The Cartesian tracker against a servo that cannot keep up: a bench, and an open defect.

``_apply`` evaluates the Jacobian at the MEASURED configuration and integrates the result onto an
accumulated joint target. That target is an absolute setpoint the servo chases, so the tracker is
an integrator wrapped around a lag -- and when the lag is large enough the tool runs past the goal,
comes back, and limit-cycles instead of arriving.

**What the arm's weight has to do with it, and why it is held out.** A position servo models a
drive that holds its own weight; without :func:`roqsim.actuators.apply_gravity_compensation` the
same joint gain has to carry the arm too, and the tracker's accumulated lead becomes the only thing
holding it up. Measured on an uncompensated UR10e, roughly two thirds of the peak lead was that,
not commanded motion -- which is why the two obvious fixes each looked refuted: both discard or
bound the total lead, so both dropped the arm. Compensation is the default now, so this file
measures the tracker rather than the arm's weight, and pins the difference below.

This is a bench before it is a test. Its guards pass today and are here because a candidate fix
must not break them; the defect itself is a strict ``xfail``. They are cheap, so a candidate is
judged in seconds rather than in a campaign.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: A position gain the UR10e's servo tracks closely -- an ordinary, healthy arm.
STIFF_GAIN = 2000.0

#: A gain low enough that the servo lags its target visibly. Not a pathological number: it is the
#: regime a compliant or under-tuned arm is in, and the one a contact task is most likely to want.
SOFT_GAIN = 200.0

#: The commanded move: straight down, well inside the arm's reach from its home pose.
GOAL_DZ = -0.05

#: How long each run is, and when measurement starts. The lead-in is skipped because the transient
#: of getting under way is not what any of these properties are about.
RUN_S = 6.0
SETTLE_S = 2.0


def _arm(gain: float, gravity_compensation: bool | None = None):
    """A UR10e under a position servo of *gain*, with the tracker on its flange site."""
    arm = {
        "model": "ur10e",
        "prefix": "ur10e_",
        "actuators": {"control": "position", "p": gain},
    }
    if gravity_compensation is not None:
        arm["gravity_compensation"] = gravity_compensation
    cfg = load_config_from_dict({
        "sim": {},
        "components": [
            {
                "spawn_arm": arm,
                "name": "ur10e",
                "components": [
                    {"cartesian_admittance": {"site": "attachment_site", "law": "position"}},
                ],
            },
        ],
    })
    # No seed resolved and none needed: nothing in this world draws, and `rng_for` is the only
    # thing that asks. A bench that pinned one would be claiming a reproducibility property it
    # does not depend on.
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine, engine.ctx.blackboard.get("cartesian:ur10e")


def _track(gain: float, dz: float = GOAL_DZ, gravity_compensation: bool | None = None) -> dict:
    """Command a move of *dz* and report what the tool did after the transient.

    ``final_mm`` is how far short of the goal it ended; ``ripple_mm`` is the peak-to-peak travel
    over the measured window, which is zero for an arm that arrives and stays.
    """
    engine, handle = _arm(gain, gravity_compensation)
    start, quat = handle.read_pose()
    goal = np.array(start, dtype=float) + np.array([0.0, 0.0, dz])
    handle.set_goal(goal.tolist(), quat)

    dt = engine.ctx.model.opt.timestep
    heights = []
    for step in range(int(RUN_S / dt)):
        engine.step()
        if step * dt > SETTLE_S:
            heights.append(float(handle.read_pose()[0][2]))
    engine.shutdown()

    heights = np.array(heights)
    return {"final_mm": abs(heights[-1] - goal[2]) * 1e3,
            "ripple_mm": (heights.max() - heights.min()) * 1e3}


def test_a_healthy_arm_arrives():
    """The control case, and the one every candidate fix must leave alone.

    Without it a "fix" that simply refuses to move would pass every other property here. Arrival
    only: what a healthy arm's *ripple* is belongs to the defect below, which is where it is
    measured, because it turns out not to be zero.
    """
    assert _track(STIFF_GAIN)["final_mm"] < 1.0


def test_the_arm_holds_its_pose_with_nothing_commanded():
    """Guard: the accumulated target is an ABSOLUTE setpoint, and reseeding it discards the move.

    The tracker is given the pose the arm is already in, so it commands nothing for six seconds.
    A tracker that reseeded its target from the measured pose each cycle keeps only the current
    cycle's increment, which is what makes it a velocity command with no integral action -- it
    holds here and then fails to arrive, which is the property above.
    """
    engine, handle = _arm(STIFF_GAIN)
    start, quat = handle.read_pose()
    handle.set_goal(list(start), quat)
    for _ in range(int(RUN_S / engine.ctx.model.opt.timestep)):
        engine.step()
    drift_mm = (handle.read_pose()[0][2] - start[2]) * 1e3
    engine.shutdown()

    assert abs(drift_mm) < 1.0, "an arm asked to stay put must not sag"


def test_a_soft_servo_still_gets_near_its_goal():
    """Guard: whatever stops the ripple must not stop the arm.

    A clamp on how far the target may lead does stop the ripple, and it does it by starving the
    servo of the error it generates force from -- the arm then parks short of where it was sent.
    Today's tracker overshoots and oscillates, which is the defect below, but it arrives.
    """
    assert _track(SOFT_GAIN)["final_mm"] < 5.0


@pytest.mark.xfail(strict=True, reason="the open defect: the target is integrated open loop")
def test_a_soft_servo_arrives_and_stays():
    """The defect. Flip this to a passing test with the fix, and delete the marker.

    The tool reaches the goal and will not settle on it: the target keeps integrating while the
    arm lags, so it runs past, comes back, and limit-cycles -- against a task whose tolerance is a
    millimetre.
    """
    measured = _track(SOFT_GAIN)
    assert measured["ripple_mm"] < 1.0
    assert measured["final_mm"] < 1.0


def test_the_ripple_survives_a_weightless_arm():
    """The defect is the integrator, not the load -- which is what makes it worth fixing.

    Compensating the arm's weight removes most of the *error* (a soft servo goes from tens of
    millimetres short to under one) and roughly a quarter of the ripple, and what is left is the
    lag alone. It scales with the lag, so it is the integrator: a healthy arm's ripple is
    ~1 mm and a soft one's is tens.
    """
    healthy = _track(STIFF_GAIN)["ripple_mm"]
    soft = _track(SOFT_GAIN)["ripple_mm"]

    assert healthy > 0.1, "even a stiff servo lags enough to show it"
    assert soft > 10 * healthy, "and the ripple grows with the lag, which is what names the cause"


def test_an_uncompensated_arm_measures_its_own_weight_instead():
    """Why the two obvious fixes were refuted, as a number: without compensation, most of the
    lead is holding the arm up rather than commanding it anywhere.

    Held out rather than deleted, because it is the confound -- a bench run against an
    uncompensated arm measures the servo's load rating and reports it as a tracker defect.
    """
    compensated = _track(SOFT_GAIN)
    uncompensated = _track(SOFT_GAIN, gravity_compensation=False)

    assert uncompensated["final_mm"] > 10 * compensated["final_mm"]
    assert uncompensated["ripple_mm"] > 3 * compensated["ripple_mm"]


def test_a_healthy_arms_lead_is_far_below_any_plausible_clamp():
    """Why a windup clamp cannot be the fix, as a number rather than an argument.

    A bound has to sit above the lead a healthy arm needs, or it throttles the arm that was
    working. That leaves it well above what a healthy arm ever reaches -- so it never fires where
    there is nothing wrong, and where there IS, the lag has already filled it.
    """
    engine, handle = _arm(STIFF_GAIN)
    start, quat = handle.read_pose()
    handle.set_goal((np.array(start, dtype=float) + [0.0, 0.0, GOAL_DZ]).tolist(), quat)

    tracker = next(p for p in engine.plugins if type(p).__name__ == "CartesianAdmittancePlugin")
    worst = 0.0
    for _ in range(int(RUN_S / engine.ctx.model.opt.timestep)):
        engine.step()
        if tracker._q_target is None:
            continue
        _, positions, _, _ = tracker._arm_handle.read_state()
        by_name = dict(zip(tracker._arm_handle.joint_names, positions, strict=False))
        measured = np.array([by_name[n] for n in tracker._joint_names])
        worst = max(worst, float(np.max(np.abs(tracker._q_target - measured))))
    engine.shutdown()

    assert worst < 0.05, "a healthy arm's lead is a fraction of a radian, whatever the bound"
