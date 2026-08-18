"""What "it moved" means. Pure arithmetic, so it is a table and needs no world, no ROS, no tree.

Two things here are decisions rather than maths, and they are what the table defends: an axis mode is
SIGNED (so one parameter expresses "risen" and "fell"), and the measure is NET displacement rather
than path length (so out-and-back is zero).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scenario_execution_roqsim.displacement import (
    AXES,
    MAGNITUDE_MODES,
    MODES,
    displacement,
    rotation_angle,
    satisfied,
)

ORIGIN = np.zeros(3)


@pytest.mark.parametrize(
    "mode,now,expected",
    [
        ("distance", [0.3, 0.4, 0.0], 0.5),
        ("distance", [0.0, 0.0, -0.5], 0.5),  # a magnitude ignores direction
        ("planar", [0.3, 0.4, 9.0], 0.5),  # ...and `planar` ignores z entirely
        ("x", [0.25, 0.0, 0.0], 0.25),
        ("y", [0.0, -0.25, 0.0], -0.25),  # an axis keeps the sign
        ("z", [0.0, 0.0, 0.05], 0.05),
    ],
)
def test_each_mode_measures_what_it_says(mode, now, expected):
    assert displacement(mode, ORIGIN, np.array(now)) == pytest.approx(expected)


def test_every_declared_mode_is_measurable():
    """A mode named in the .osc enum but not implemented here would raise at trigger time."""
    for mode in MODES:
        displacement(mode, ORIGIN, np.array([0.1, 0.1, 0.1]))


def test_an_unknown_mode_says_which_ones_exist():
    with pytest.raises(ValueError, match="distance"):
        displacement("sideways", ORIGIN, ORIGIN)


def test_out_and_back_has_moved_nothing():
    """NET displacement, the whole contrast with odometry_distance_traveled's integrated path."""
    start = np.array([1.0, 2.0, 3.0])
    assert displacement("distance", start, start) == 0.0


@pytest.mark.parametrize("mode", MAGNITUDE_MODES)
def test_a_magnitude_threshold_compares_upward(mode):
    assert satisfied(mode, 0.05, 0.05) is True
    assert satisfied(mode, 0.05, 0.049) is False


def test_an_axis_threshold_is_one_sided_in_the_direction_of_its_sign():
    """The decision this whole design rests on: `z: 0.05` is *risen* 5 cm, not |dz| >= 5 cm.

    A parcel that FALLS 5 cm must not satisfy a rise, and vice versa -- otherwise a fault meant to
    fire on a lift would fire on the object settling.
    """
    assert satisfied("z", 0.05, 0.05) is True
    assert satisfied("z", 0.05, -0.20) is False, "a fall must not satisfy a rise"
    assert satisfied("z", -0.05, -0.05) is True
    assert satisfied("z", -0.05, 0.20) is False, "a rise must not satisfy a fall"


def test_the_boundary_is_inclusive():
    """`>=`, so a threshold that is exactly reached fires: a campaign level of 0.05 must be reachable."""
    assert satisfied("distance", 0.05, 0.05) is True
    for axis in AXES:
        assert satisfied(axis, 0.05, 0.05) is True


# -- rotation ------------------------------------------------------------------------------------
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_z(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


@pytest.mark.parametrize("turn", [0.0, 0.1, math.pi / 2, math.pi - 1e-6])
def test_the_geodesic_angle_is_the_angle_turned(turn):
    assert rotation_angle(IDENTITY, _quat_z(turn)) == pytest.approx(turn, abs=1e-6)


def test_the_axis_does_not_matter():
    """It is the angle of the single rotation between two orientations, whatever axis that is about."""
    about_x = np.array([math.cos(0.4 / 2), math.sin(0.4 / 2), 0.0, 0.0])
    assert rotation_angle(IDENTITY, about_x) == pytest.approx(0.4, abs=1e-6)


def test_a_turn_past_half_a_turn_reads_as_the_shorter_way_round():
    """`q` and `-q` are the same orientation, so the measure saturates at pi rather than folding back.

    Without the absolute value in the dot product, a 359 deg turn would read as 359 deg one tick and
    1 deg the next, depending only on which sign MuJoCo happened to normalise to.
    """
    assert rotation_angle(IDENTITY, _quat_z(math.pi)) == pytest.approx(math.pi, abs=1e-6)
    assert rotation_angle(IDENTITY, _quat_z(1.9 * math.pi)) == pytest.approx(0.1 * math.pi, abs=1e-5)
    assert rotation_angle(IDENTITY, -_quat_z(0.3)) == pytest.approx(0.3, abs=1e-6)


def test_an_unnormalised_quaternion_is_still_measurable():
    """A caller may hand us a pose it built itself; a scaled quaternion is the same orientation."""
    assert rotation_angle(IDENTITY * 3.0, _quat_z(0.5) * 2.0) == pytest.approx(0.5, abs=1e-6)
