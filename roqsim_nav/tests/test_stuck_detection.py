"""Stuck detection: the window has to span the time the check asks for.

``observe`` keeps a sliding window of recent positions and ``is_stuck`` asks whether it spans at
least ``stuck_time``. Those two have to agree about the window, and a trim that drops *every* sample
older than the cutoff leaves one that always spans slightly less -- so with samples arriving at a
fixed rate the check can never pass and recovery is unreachable however wedged the mover is.

That is a silent failure: nothing errors, a stuck mover simply sits there for ever.
"""

from __future__ import annotations

from roqsim_nav.behavior import NavCore, NavParams
from roqsim_nav.state import NavState

_PERIOD = 0.05  # a navigator's default 20 Hz tick


def _core(**params):
    st = NavState(name="mover", waypoints=[(0.0, 0.0), (5.0, 0.0)], speed=0.5)
    return NavCore(st, None, NavParams(**{"stuck_time": 1.5, "stuck_eps": 0.10, **params}))


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
