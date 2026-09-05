"""Pure-pursuit geometry: the carrot on the path, and the arc curvature to it.

These are the pieces a path-tracking follower is built from, tested as pure geometry. They are not
yet wired into the ``navigator``: doing so correctly means replacing the follower's
**goal-advancement** as well, because pure pursuit deliberately does not drive at the end of the
current leg, so a follower that advances on proximity to that endpoint reaches it late or never.
Overriding only the commanded direction leaves the two disagreeing, which measurably stalls the
mover. The geometry lands first, and is correct on its own terms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roqsim_nav.control import closest_point_on_polyline, lookahead_point, pure_pursuit

# An L: east along y = 0 to the origin, then north along x = 0.
CORNER = [(-2.0, 0.0), (0.0, 0.0), (0.0, 2.0)]
STRAIGHT = [(0.0, 0.0), (10.0, 0.0)]


# -- projection --------------------------------------------------------------------------------
def test_the_projection_lands_on_the_nearest_segment():
    _, _, point = closest_point_on_polyline(CORNER, (-1.0, 0.5))
    assert point == pytest.approx([-1.0, 0.0])


def test_the_projection_clamps_to_a_segment_end_rather_than_running_past_it():
    _, _, point = closest_point_on_polyline(STRAIGHT, (20.0, 3.0))
    assert point == pytest.approx([10.0, 0.0])


# -- the carrot --------------------------------------------------------------------------------
def test_the_carrot_sits_a_lookahead_along_the_path():
    carrot = lookahead_point(STRAIGHT, (2.0, 0.0), 1.5)
    assert carrot == pytest.approx([3.5, 0.0])


def test_the_carrot_measures_along_the_path_from_the_projection_not_from_the_mover():
    """A mover off to the side must not get a nearer carrot for being off-track."""
    on = lookahead_point(STRAIGHT, (2.0, 0.0), 1.5)
    off = lookahead_point(STRAIGHT, (2.0, 0.9), 1.5)
    assert off == pytest.approx(on)


def test_the_carrot_rounds_the_corner_onto_the_next_leg():
    """The property that distinguishes this from chasing waypoints: past the corner, the carrot is
    already on the following leg, so the mover starts turning before it arrives."""
    carrot = lookahead_point(CORNER, (-0.5, 0.0), 1.0)
    assert carrot == pytest.approx([0.0, 0.5])


def test_the_carrot_stops_at_the_end_of_the_path():
    """Otherwise a mover orbits a carrot that ran out instead of converging on the goal."""
    assert lookahead_point(STRAIGHT, (9.5, 0.0), 5.0) == pytest.approx([10.0, 0.0])


def test_a_single_point_path_is_its_own_carrot():
    assert lookahead_point([(1.0, 2.0)], (0.0, 0.0), 3.0) == pytest.approx([1.0, 2.0])


# -- curvature ---------------------------------------------------------------------------------
def test_a_carrot_dead_ahead_needs_no_curvature():
    _, curvature = pure_pursuit(STRAIGHT, (2.0, 0.0), 0.0, lookahead=1.0, speed=1.0)
    assert curvature == pytest.approx(0.0)


def test_curvature_signs_follow_the_carrot():
    """Left of the heading is positive curvature (a left turn), and vice versa."""
    _, left = pure_pursuit([(0.0, 0.0), (0.0, 5.0)], (0.0, 0.0), 0.0, lookahead=1.0, speed=1.0)
    _, right = pure_pursuit([(0.0, 0.0), (0.0, -5.0)], (0.0, 0.0), 0.0, lookahead=1.0, speed=1.0)
    assert left > 0 and right < 0
    assert left == pytest.approx(-right)


def test_curvature_is_the_arc_through_the_carrot():
    """`2 * y_body / L^2` is the exact geometric quantity, not an approximation of it.

    A carrot exactly abeam at distance L means a half-circle of radius L/2, i.e. curvature 2/L.
    """
    _, curvature = pure_pursuit([(0.0, 0.0), (0.0, 2.0)], (0.0, 0.0), 0.0, lookahead=2.0, speed=1.0)
    assert curvature == pytest.approx(2.0 / 2.0)


def test_speed_is_eased_off_as_the_path_runs_out():
    """The carrot stops receding near the goal; without easing, the mover circles it forever."""
    far, _ = pure_pursuit(STRAIGHT, (2.0, 0.0), 0.0, lookahead=1.0, speed=1.0)
    near, _ = pure_pursuit(STRAIGHT, (9.9, 0.0), 0.0, lookahead=1.0, speed=1.0)
    assert float(np.linalg.norm(far)) == pytest.approx(1.0)
    assert float(np.linalg.norm(near)) < 0.2


def test_the_command_points_at_the_carrot():
    pref, _ = pure_pursuit(CORNER, (-0.5, 0.0), 0.0, lookahead=1.0, speed=2.0)
    heading = math.atan2(pref[1], pref[0])
    assert heading == pytest.approx(math.atan2(0.5, 0.5))
    assert float(np.linalg.norm(pref)) == pytest.approx(2.0)
