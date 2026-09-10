# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The Cartesian tracker against a servo that cannot keep up: a bench, and an open defect.

``_apply`` evaluates the Jacobian at the MEASURED configuration and integrates the result onto an
accumulated joint target. On an arm whose servo tracks that target closely the two configurations
are near enough that nothing shows. On a soft one they diverge, the target runs ahead of an arm
that never catches it, and the tool limit-cycles about the goal instead of arriving at it.

This file is a bench before it is a test. Two of its four properties **pass today** and are here
as guards, because the two obvious fixes for the third each break one of them:

* *Reseed the target from the measured pose* (``target = q_now + dq*dt``). The accumulated target
  is an ABSOLUTE setpoint, and leading the measured pose is how a position servo generates the
  force that holds the arm up. Reseeding hands the arm back to gravity --
  ``test_the_arm_holds_its_pose_with_nothing_commanded`` is what says so.
* *Bound how far the target may lead* (a windup clamp). A servo generates force BY leading, so
  bounding the lead bounds the force. It is inert on a healthy arm, whose lead is a fraction of any
  plausible bound (this file measures it), and on a soft one it stops the ripple by parking the arm
  short of the goal -- ``test_a_soft_servo_still_gets_near_its_goal`` is what says so.

Whatever fixes the defect has to separate lead that represents commanded motion from lead that is
accumulated lag. Bounding or discarding the total does not, and these four properties together are
the evidence for that -- they are cheap to run, so a candidate fix can be judged in seconds rather
than in a campaign.
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


def _arm(gain: float):
    """A UR10e under a position servo of *gain*, with the tracker on its flange site."""
    cfg = load_config_from_dict({
        "sim": {},
        "components": [
            {
                "spawn_arm": {
                    "model": "ur10e",
                    "prefix": "ur10e_",
                    "actuators": {"control": "position", "p": gain},
                },
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


def _track(gain: float, dz: float = GOAL_DZ) -> dict:
    """Command a move of *dz* and report what the tool did after the transient.

    ``final_mm`` is how far short of the goal it ended; ``ripple_mm`` is the peak-to-peak travel
    over the measured window, which is zero for an arm that arrives and stays.
    """
    engine, handle = _arm(gain)
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


def test_a_healthy_arm_arrives_and_stays():
    """The control case, and the one every candidate fix must leave alone.

    Without it a "fix" that simply refuses to move would pass every other property here.
    """
    measured = _track(STIFF_GAIN)
    assert measured["final_mm"] < 1.0
    assert measured["ripple_mm"] < 1.0


def test_the_arm_holds_its_pose_with_nothing_commanded():
    """Guard: the accumulated target is an ABSOLUTE setpoint, and that is what holds the arm up.

    The tracker is given the pose the arm is already in, so it commands nothing for six seconds.
    A tracker that reseeded its target from the measured pose each cycle would follow the arm's own
    sag downward instead, and the measured drop would grow without bound rather than stay at zero.
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
    servo of the very error it generates force from -- the arm then parks, quietly, hundreds of
    millimetres short of where it was sent. Today's tracker overshoots and oscillates, which is the
    defect below, but it does at least end up in the neighbourhood.
    """
    assert _track(SOFT_GAIN)["final_mm"] < 100.0


@pytest.mark.xfail(strict=True, reason="the open defect: the target is integrated open loop")
def test_a_soft_servo_arrives_and_stays():
    """The defect. Flip this to a passing test with the fix, and delete the marker.

    The tool does not settle at the goal: the target keeps integrating while the arm lags, so it
    runs past, comes back, and limit-cycles -- against a task whose tolerance is a millimetre.
    """
    measured = _track(SOFT_GAIN)
    assert measured["ripple_mm"] < 1.0
    assert measured["final_mm"] < 1.0


def test_a_healthy_arms_lead_is_far_below_any_plausible_clamp():
    """Why a windup clamp cannot be the fix, as a number rather than an argument.

    A bound has to sit above the lead a healthy arm needs, or it throttles the arm that was working.
    That leaves it an order of magnitude above what a healthy arm ever reaches -- so it never fires
    where there is nothing wrong, and where there IS, the lag has already filled it.
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
