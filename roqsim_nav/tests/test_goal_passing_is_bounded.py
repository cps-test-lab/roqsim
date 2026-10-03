"""Passing goals within one tick is bounded for the waypoint follower, as it is for pure pursuit.

A looping route whose every goal is already within ``arrival_radius`` -- one waypoint looped onto
itself, or coincident goals -- reads as reached again and again in the same tick. Unbounded, that
would blow the Python stack from inside ``pre_step``; the follower holds, with a warning, after
``_MAX_GOALS_PER_TICK`` goals.
"""

from __future__ import annotations

import numpy as np

from roqsim_nav.behavior import NavCore, NavParams
from roqsim_nav.state import NavState


def _no_draw(lo, hi):
    raise AssertionError(f"a dwell was drawn ({lo}, {hi}); this route has none")


def _core(waypoints):
    st = NavState(name="mover", waypoints=waypoints, speed=0.5, loop=True)
    return NavCore(st, None, NavParams(), uniform=_no_draw)


def test_a_looping_route_of_one_point_holds_instead_of_recursing_for_ever():
    core = _core([(0.0, 0.0), (0.0, 0.0)])
    core.observe(0.05, (0.0, 0.0), None)
    assert core.ensure_path()
    assert core.follow_path() is True
    assert np.allclose(core.pref_vel, 0.0)


def test_a_route_with_somewhere_to_go_still_moves():
    core = _core([(0.0, 0.0), (3.0, 0.0)])
    core.observe(0.05, (0.0, 0.0), None)
    assert core.ensure_path()
    assert core.follow_path() is True
    assert core.pref_vel[0] > 0.0
