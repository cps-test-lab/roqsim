"""A walker's body drawn behind its nav root: smooth at a start, a turn and a stop, at rest where
the root is."""

from __future__ import annotations

import math

import numpy as np

from roqsim_walker.animation import _follow


class _State:
    def __init__(self, lag):
        self.body_lag = lag
        self.body_pos = np.zeros(2)
        self.body_vel = np.zeros(2)
        self.yaw = self.yaw_rate = 0.0


def _run(lag, dt=1 / 60):
    """The root walks 2 m east at 1 m/s from standing, turns north for 1 m, and stops."""
    st, root, out = _State(lag), np.zeros(2), []
    for k in range(int(5.0 / dt)):
        t = k * dt
        if t < 2.0:
            root = np.array([t, 0.0])
        elif t < 3.0:
            root = np.array([2.0, t - 2.0])
        else:
            root = np.array([2.0, 1.0])
        st.body_pos = _follow(st, root, dt)
        out.append(st.body_pos.copy())
    return np.array(out), root


def test_without_a_lag_the_body_is_on_the_root():
    path, root = _run(0.0)
    assert np.allclose(path[-1], root)


def test_with_a_lag_the_body_moves_smoothly_and_comes_to_rest_on_the_root():
    dt = 1 / 60
    path, root = _run(0.15, dt)
    acc = np.linalg.norm(np.diff(path, 2, axis=0), axis=1) / dt**2
    assert acc.max() < 8.0  # m/s^2: the root's own changes of velocity are instant
    assert np.linalg.norm(path[-1] - root) < 1e-3
    raw, _ = _run(0.0, dt)
    assert np.linalg.norm(np.diff(raw, 2, axis=0), axis=1).max() / dt**2 > 50.0


def test_with_a_lag_a_turn_speeds_up_and_slows_down_and_ends_on_the_heading():
    from roqsim_walker.animation import _MAX_TURN_RATE, _turn

    st, dt, rates = _State(0.15), 1 / 60, []
    st.yaw, st.yaw_rate = 0.0, 0.0
    for _ in range(120):
        before = st.yaw
        st.yaw = _turn(st, math.pi / 2, dt)
        rates.append((st.yaw - before) / dt)
    assert abs(st.yaw - math.pi / 2) < 1e-9 and st.yaw_rate == 0.0
    assert max(rates) <= _MAX_TURN_RATE + 1e-9
    assert np.abs(np.diff(rates)).max() / dt < _MAX_TURN_RATE / (2 * 0.15) * 1.6
