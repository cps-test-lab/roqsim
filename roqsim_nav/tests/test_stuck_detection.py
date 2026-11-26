"""Stuck detection: the window has to span the time the check asks for.

``observe`` keeps a sliding window of recent positions and ``is_stuck`` asks whether it spans at
least ``stuck_time``. Those two have to agree about the window, and a trim that drops *every* sample
older than the cutoff leaves one that always spans slightly less -- so with samples arriving at a
fixed rate the check can never pass and recovery is unreachable however wedged the mover is.

That is a silent failure: nothing errors, a stuck mover simply sits there for ever.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_nav.behavior import NavCore, NavParams
from roqsim_nav.state import NavState

_PERIOD = 0.05  # a navigator's default 20 Hz tick


def _core(**params):
    st = NavState(name="mover", waypoints=[(0.0, 0.0), (5.0, 0.0)], speed=0.5)

    # `uniform` is required and only feeds a waypoint dwell, which this suite never configures --
    # so it is wired to raise rather than to a value that would hide an unexpected draw.
    def _no_draw(lo, hi):
        raise AssertionError(f"stuck detection drew a dwell ({lo}, {hi}); it should not")

    return NavCore(
        st,
        None,
        NavParams(**{"stuck_time": 1.5, "stuck_eps": 0.10, **params}),
        uniform=_no_draw,
    )


def _hold(core, seconds, at=(1.0, 1.0)):
    """Report the same position every tick, as a mover held against an obstacle does."""
    t = 0.0
    while t < seconds:
        t += _PERIOD
        core.observe(t, at, None)
    return t


def test_a_mover_that_does_not_move_becomes_stuck():
    core = _core()
    _hold(core, 5.0)
    assert core.is_stuck(), (
        "a stationary mover never declaring itself stuck makes recovery dead code"
    )


def test_it_is_not_stuck_before_stuck_time_has_passed():
    core = _core()
    _hold(core, 1.0)
    assert not core.is_stuck()


def test_a_moving_mover_is_never_stuck():
    core = _core()
    t = 0.0
    for i in range(200):
        t += _PERIOD
        core.observe(t, (i * 0.05, 0.0), None)
    assert not core.is_stuck()


def test_forgetting_progress_re_arms_the_window():
    """Yielding to traffic must not accumulate toward a recovery -- and must not disable it either."""
    core = _core()
    _hold(core, 5.0)
    assert core.is_stuck()
    core.forget_progress()
    assert not core.is_stuck(), "the window should start again from the rebase"
    _hold(core, 5.0)
    assert core.is_stuck(), "and fill again if the mover still does not move"


def test_a_route_sent_exact_never_recovers_however_long_it_is_held():
    """A recovery backs off and re-plans: on a route sent ``exact`` it would leave the route that was
    checked, so a mover held there waits instead."""
    core = _core()
    core.exact = True
    _hold(core, 5.0)
    assert core.is_stuck()  # it is held, as anything is
    assert not core.needs_recovery()


# -- a mover slower than `stuck_eps` a `stuck_time` --------------------------------------------
def _walking(speed, seconds, held_from=None):
    """A mover on its route at ``speed``, ticked as the navigator's tree ticks it: recovery first,
    else the route, moved each tick by the velocity it asks for; held where it is from
    ``held_from`` on. Its x at the end, and how many recoveries it started."""
    st = NavState(name="mover", waypoints=[(0.0, 0.0), (50.0, 0.0)], speed=speed)

    def _no_draw(lo, hi):
        raise AssertionError("this route has no dwell to draw")

    core = NavCore(st, None, NavParams(stuck_time=1.5, stuck_eps=0.10), uniform=_no_draw)
    pos, t, recoveries = np.zeros(2), 0.0, 0
    while t < seconds:
        t += _PERIOD
        st.pos = pos
        core.observe(t, pos, None)
        if core.needs_recovery():
            if not core.recovery_active():
                core.start_recovery()
                recoveries += 1
            core.recovery_vel()
        else:
            core.ensure_path()
            core.follow_path()
        if held_from is None or t < held_from:
            pos = pos + core.pref_vel * _PERIOD
    return float(pos[0]), recoveries


@pytest.mark.parametrize("speed", [0.02, 0.044, 0.055])
def test_a_mover_slower_than_stuck_eps_a_window_walks_its_route(speed):
    """At its own speed it covers less than ``stuck_eps`` in ``stuck_time``: that is its pace, not
    a wedge. Read as one, every recovery backs it off its route, and it walks away from its goal."""
    x, recoveries = _walking(speed, 30.0)
    assert recoveries == 0, f"at {speed} m/s a free mover recovered {recoveries} times"
    assert x == pytest.approx(speed * 30.0, rel=0.02)


@pytest.mark.parametrize("speed", [0.02, 0.044])
def test_a_slow_mover_held_still_is_still_stuck(speed):
    """The threshold follows the mover's speed down; a slow mover that stops moving is wedged."""
    _, recoveries = _walking(speed, 6.0, held_from=1.0)
    assert recoveries >= 1


def test_a_mover_of_speed_zero_is_never_stuck_and_stands():
    """It is asked to make no progress, so making none is not being stuck: it never backs up."""
    x, recoveries = _walking(0.0, 30.0)
    assert (x, recoveries) == (0.0, 0)


@pytest.mark.parametrize("travelled, stuck", [(0.08, True), (0.13, False)])
def test_at_usual_speeds_stuck_eps_is_the_threshold(travelled, stuck):
    """A mover at a walking pace (0.5 m/s) is judged as before: stuck below ``stuck_eps`` (0.10 m)
    over the window, whatever its speed would have covered."""
    core = _core()
    t = 0.0
    while t < 1.6:
        t += _PERIOD
        core.observe(t, (t / 1.6 * travelled, 0.0), None)
    assert core.is_stuck() is stuck
