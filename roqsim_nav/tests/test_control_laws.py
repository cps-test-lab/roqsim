"""The three control laws: a planner's preferred velocity, shaped for a base that must realise it.

Pure functions, so these are exhaustive and instant. Each test pins a property that is easy to get
subtly wrong and expensive to notice in a trajectory plot.
"""

from __future__ import annotations

import math

import pytest

from roqsim_nav.control import ackermann, approach_angle, holonomic, unicycle, wrap_angle

EAST = (1.0, 0.0)


# -- angles ------------------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.0, 0.7, math.pi, 3 * math.pi, -3 * math.pi, -12.0, 100.0])
def test_wrap_angle_folds_into_one_turn(angle):
    """Same angle, in (-pi, pi].

    The property is the equivalence, not a particular sign at the seam: +pi and -pi are one angle,
    and which one comes back is an artefact of `atan2`, so asserting it would pin a non-fact.
    """
    wrapped = wrap_angle(angle)
    assert abs(wrapped) <= math.pi + 1e-12
    assert math.sin(wrapped) == pytest.approx(math.sin(angle), abs=1e-12)
    assert math.cos(wrapped) == pytest.approx(math.cos(angle), abs=1e-12)


def test_approach_angle_takes_the_short_way_round():
    """Just past pi the short way is negative, not a near-full turn the other way."""
    assert approach_angle(0.0, -0.1, 1.0) == pytest.approx(-0.1)
    assert approach_angle(3.0, -3.0, 0.1) == pytest.approx(3.1)  # over the +pi seam, not back down


def test_approach_angle_clamps_to_the_step():
    assert approach_angle(0.0, 1.0, 0.25) == pytest.approx(0.25)


# -- unicycle ----------------------------------------------------------------------------------
def test_unicycle_drives_straight_when_already_facing_the_goal():
    v, vy, w = unicycle(EAST, 0.0, gain=2.0, max_w=1.5, turn_in_place=0.8)
    assert v == pytest.approx(1.0)
    assert vy == 0.0  # a differential base never strafes
    assert w == pytest.approx(0.0)


def test_unicycle_slows_into_a_turn_rather_than_arcing_wide():
    """`v = speed * cos(error)`: the forward term shrinks as the heading error grows."""
    straight = unicycle(EAST, 0.0, gain=2.0, max_w=9.0, turn_in_place=0.8)[0]
    angled = unicycle(EAST, 0.5, gain=2.0, max_w=9.0, turn_in_place=0.8)[0]
    assert angled < straight
    assert angled == pytest.approx(math.cos(0.5))


def test_unicycle_pivots_in_place_past_the_threshold():
    """Beyond `turn_in_place` there is no useful forward component -- stop and rotate."""
    v, _, w = unicycle(EAST, 2.0, gain=2.0, max_w=9.0, turn_in_place=0.8)
    assert v == 0.0
    assert w != 0.0


def test_unicycle_reverses_only_when_the_command_points_nearly_backwards():
    """Forward, pivot, reverse -- three bands, and only the last one is negative.

    Reversing exists because recovery expresses "back away from the blocker" as a world velocity,
    and a law that always turned to face it could never retreat at all. It must stay confined to
    that band, or a mover would reverse its way around ordinary corners.
    """
    for yaw in [i * math.pi / 32 for i in range(-32, 33)]:
        v, _, _ = unicycle(EAST, yaw, gain=2.0, max_w=9.0, turn_in_place=0.8)
        err = abs(wrap_angle(-yaw))  # EAST is heading 0, so the error is -yaw
        if err <= 0.8:
            assert v > 0.0, "should drive forward"
        elif err >= math.pi - 0.8:
            assert v < 0.0, "should reverse rather than pivot through 180 degrees"
        else:
            assert v == 0.0, "should pivot on the spot"


def test_unicycle_clamps_yaw_to_the_configured_cap():
    _, _, w = unicycle(EAST, -1.0, gain=100.0, max_w=1.5, turn_in_place=3.2)
    assert w == pytest.approx(1.5)


def test_a_stopped_command_produces_no_motion_and_no_spin():
    """Below the threshold the direction is rounding error; steering toward it would spin a body."""
    for law, kw in [
        (unicycle, {"gain": 2.0, "max_w": 1.5, "turn_in_place": 0.8}),
        (ackermann, {"gain": 2.0, "max_w": 1.5, "min_speed": 0.15}),
    ]:
        assert law((1e-9, 1e-9), 0.7, **kw) == (0.0, 0.0, 0.0)


# -- holonomic ---------------------------------------------------------------------------------
def test_holonomic_rotates_the_command_into_the_body_frame():
    """Facing +Y and asked to go +X, the base must strafe right, not drive forward."""
    vx, vy, _ = holonomic(EAST, math.pi / 2, gain=2.0, max_w=1.5)
    assert vx == pytest.approx(0.0, abs=1e-12)
    assert vy == pytest.approx(-1.0)


def test_holonomic_face_hold_keeps_the_heading():
    """A cart that must stay square to the aisle crabs instead of turning."""
    _, _, w = holonomic(EAST, math.pi / 2, gain=2.0, max_w=1.5, face="hold")
    assert w == 0.0


def test_holonomic_face_travel_turns_toward_the_goal():
    _, _, w = holonomic(EAST, math.pi / 2, gain=2.0, max_w=9.0)
    assert w < 0.0  # turn back toward +X


# -- ackermann ---------------------------------------------------------------------------------
def test_ackermann_never_commands_zero_speed_while_it_still_has_somewhere_to_go():
    """A stationary car has no curvature: `ackermann_drive` derives the rack angle from w/v, so
    v == 0 steers the wheels nowhere however large w is. The unicycle law's pivot would freeze it."""
    for yaw in [i * math.pi / 16 for i in range(-16, 17)]:
        v, _, _ = ackermann(EAST, yaw, gain=2.0, max_w=1.5, min_speed=0.15)
        assert v >= 0.15


def test_ackermann_floors_a_crawl_but_does_not_cap_a_cruise():
    assert ackermann((0.01, 0.0), 0.0, gain=2.0, max_w=1.5, min_speed=0.15)[0] == pytest.approx(
        0.15
    )
    assert ackermann((2.0, 0.0), 0.0, gain=2.0, max_w=1.5, min_speed=0.15)[0] == pytest.approx(2.0)


def test_a_unicycle_reverses_rather_than_pivoting_through_180_degrees():
    """Recovery commands a world velocity pointing away from the blocker; a law that always turned
    to face it would spin for the whole backup window and never retreat."""
    v, vy, w = unicycle(
        (-1.0, 0.0), yaw=0.0, gain=2.0, max_w=1.5, turn_in_place=0.8
    )  # straight back
    assert v == pytest.approx(-1.0)
    assert vy == 0.0
    assert w == pytest.approx(0.0, abs=1e-9)


def test_a_unicycle_still_pivots_for_a_sideways_command():
    """Between the two thresholds it is neither: turn on the spot."""
    v, _, w = unicycle((0.0, 1.0), yaw=0.0, gain=2.0, max_w=1.5, turn_in_place=0.8)
    assert v == 0.0
    assert w > 0.0


def test_a_unicycle_still_drives_forward_for_a_forward_command():
    v, _, w = unicycle((1.0, 0.0), yaw=0.0, gain=2.0, max_w=1.5, turn_in_place=0.8)
    assert v == pytest.approx(1.0)
    assert w == pytest.approx(0.0, abs=1e-9)
