# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A goal streamed as a moving setpoint is tracked, not trailed.

A law that closes only on the pose error follows a goal moving at ``v`` a steady ``v / kp`` behind,
and nothing reports it: every pose is a valid pose, just late. The feedforward commands the goal's
own velocity alongside the correction, which is what a trajectory-following controller on a real
arm does. It must not turn a goal that JUMPS into a burst of speed, so a step to a stationary goal
has to behave exactly as it does without it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from roqsim_manipulation.plugins.cartesian_admittance import (
    CartesianAdmittancePlugin,
    _GoalStream,
)

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

TIMESTEP = 0.001

#: The streamed motion: a goal moving at this speed, re-sent at this rate -- a ROS client's pace.
SPEED = 0.05
STREAM_HZ = 50.0

#: A gain above the default, so the proportional law settles on its lag well inside a short run.
KP = 4.0

#: How long the stream runs, and when measuring starts: after five time constants of the P law.
RUN_S = 2.0
SETTLE_S = 1.25

#: What is left with the feedforward on is the goal's own staircase -- a stream holds each pose for a
#: period, so the goal the controller sees is up to ``SPEED / STREAM_HZ`` (1 mm) behind the moving
#: one -- and the servo. Measured against the moving goal it stays under this.
MAX_FF_LAG_M = 1.0e-3


def _engine(feedforward: str | None = None, *, kp: float = KP):
    cart = {
        "site": "tool_site",
        "controller_type": "cartesian_motion_controller",
        "rate_hz": 500.0,
        "kp": [kp, kp, kp, 2.0, 2.0, 2.0],
    }
    if feedforward is not None:
        cart["feedforward"] = feedforward
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": TIMESTEP, "gravity": [0.0, 0.0, 0.0]},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [{"cartesian_admittance": cart}],
                },
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine, engine.ctx.blackboard.get("cartesian:ur5e")


def _stream(feedforward: str, *, supplied: bool = False, axis: int = 0) -> dict:
    """Stream a goal moving at SPEED along *axis*; report the lag behind the MOVING goal."""
    engine, handle = _engine(feedforward)
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "tracking_error")
    try:
        start, mat = handle.read_pose()
        direction = np.eye(3)[axis]
        twist = np.concatenate([SPEED * direction, np.zeros(3)]) if supplied else None
        every = round(1.0 / STREAM_HZ / TIMESTEP)
        lag, reported = [], []
        for k in range(round(RUN_S / TIMESTEP)):
            if k % every == 0:
                goal = start + SPEED * engine.ctx.sim_time * direction
                handle.set_goal(goal.tolist(), mat.reshape(9), twist)
            engine.step()
            if engine.ctx.sim_time >= SETTLE_S:
                pos, _ = handle.read_pose()
                moving_goal = start + SPEED * engine.ctx.sim_time * direction
                lag.append(float(np.dot(moving_goal - pos, direction)))
                reported.append(endpoint.read().distance)
    finally:
        engine.shutdown()
    return {
        "lag": float(np.mean(lag)),
        "worst": float(np.max(np.abs(lag))),
        "reported": float(np.mean(reported)),
    }


# -- a moving goal ------------------------------------------------------------------------------


def test_without_feedforward_a_streamed_goal_is_trailed_by_v_over_kp():
    """The defect the feedforward exists for, pinned so the number in the docs stays true."""
    out = _stream("off")
    assert out["lag"] == pytest.approx(SPEED / KP, rel=0.1)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_an_estimated_feedforward_removes_the_lag(axis):
    out = _stream("auto", axis=axis)
    assert out["worst"] < MAX_FF_LAG_M, (
        f"a goal streamed at {SPEED * 1e3:.0f} mm/s is still trailed by up to "
        f"{out['worst'] * 1e3:.2f} mm with the feedforward on"
    )


def test_a_supplied_goal_twist_removes_the_lag():
    out = _stream("supplied", supplied=True)
    assert out["worst"] < MAX_FF_LAG_M


def test_supplied_mode_does_not_estimate():
    """`supplied` feeds forward only what the caller states, so a bare pose stream is trailed."""
    out = _stream("supplied")
    assert out["lag"] == pytest.approx(SPEED / KP, rel=0.1)


def test_the_tracking_error_endpoint_reads_the_lag():
    """The endpoint measures against the goal the controller HAS, which is up to one stream period
    behind the moving one -- half a period on average."""
    held = SPEED / STREAM_HZ / 2
    trailed = _stream("off")
    assert trailed["reported"] == pytest.approx(SPEED / KP - held, rel=0.1)
    tracked = _stream("auto")
    assert tracked["reported"] < MAX_FF_LAG_M


# -- a goal that jumps ---------------------------------------------------------------------------


def _step_response(feedforward: str, *, republish_hz: float = 0.0) -> np.ndarray:
    """The site's path after a step to a new stationary goal, the goal re-sent at *republish_hz*.

    Re-sending the same pose is what a client that streams a setpoint does while it stands still,
    so the step lands inside a stream: the case where differencing goals would read a jump as speed.
    """
    engine, handle = _engine(feedforward)
    try:
        start, mat = handle.read_pose()
        goal = start
        every = round(1.0 / republish_hz / TIMESTEP) if republish_hz else 0
        path = []
        for k in range(round(1.5 / TIMESTEP)):
            if k == round(0.5 / TIMESTEP):
                goal = start + np.array([0.0, 0.0, -0.01])
                handle.set_goal(goal.tolist(), mat.reshape(9))
            elif every and k % every == 0:
                handle.set_goal(goal.tolist(), mat.reshape(9))
            engine.step()
            path.append(handle.read_pose()[0])
    finally:
        engine.shutdown()
    return np.array(path)


@pytest.mark.parametrize("republish_hz", [0.0, 50.0])
def test_a_step_to_a_stationary_goal_is_unchanged_by_the_feedforward(republish_hz):
    """Same path, to the bit: a lone goal and a re-sent stationary one both feed nothing forward,
    and a jump inside a stream of them is one outlier, which the estimate discards."""
    with_ff = _step_response("auto", republish_hz=republish_hz)
    without = _step_response("off", republish_hz=republish_hz)
    assert np.array_equal(with_ff, without)


def test_a_jump_inside_a_moving_stream_adds_no_velocity():
    """A goal displaced once mid-stream must not be read as a burst of speed along the jump."""
    stream = _GoalStream("auto", 0.2)
    v = np.array([0.05, 0.0, 0.0])
    for k in range(5):
        stream.observe(0.02 * k, v * 0.02 * k, None)
    assert stream.velocity(0.08)[:3] == pytest.approx(v)

    stream.observe(0.10, v * 0.10 + [0.0, 0.0, 0.02], None)  # 20 mm down, in one goal
    assert stream.velocity(0.10)[:3] == pytest.approx(v), "the jump must not be fed forward"


def test_goals_further_apart_than_the_window_are_not_a_stream():
    """A sequence of waypoints sent seconds apart is a set of stationary goals, each to converge on."""
    stream = _GoalStream("auto", 0.2)
    for k in range(4):
        stream.observe(1.0 * k, np.array([0.01 * k, 0.0, 0.0]), None)
    assert stream.velocity(3.0) is None


def test_a_stream_that_stops_stops_feeding_forward():
    """Once the next goal is overdue the arm settles on the last one rather than coasting past it."""
    stream = _GoalStream("auto", 0.2)
    for k in range(4):
        stream.observe(0.02 * k, np.array([0.001 * k, 0.0, 0.0]), None)
    assert stream.velocity(0.06 + 0.02) is not None, "still live when the next goal is only due"
    assert stream.velocity(0.06 + 0.031) is None


def test_a_turning_goal_feeds_its_angular_velocity_forward():
    stream = _GoalStream("auto", 0.2)
    rate = 0.5  # rad/s about z
    for k in range(3):
        a = rate * 0.02 * k
        rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1.0]])
        stream.observe(0.02 * k, np.zeros(3), rot)
    assert stream.velocity(0.04)[3:] == pytest.approx([0.0, 0.0, rate])


def test_off_feeds_nothing_forward_even_when_a_twist_is_supplied():
    stream = _GoalStream("off", 0.2)
    stream.observe(0.0, np.zeros(3), None, np.ones(6))
    assert stream.velocity(0.0) is None


# -- the compliance law ---------------------------------------------------------------------------


def _compliance(stiffness):
    """The compliance law alone, in free space, with the tool on its goal and moving with it."""
    plugin = CartesianAdmittancePlugin.__new__(CartesianAdmittancePlugin)
    plugin.M = np.ones(6)
    plugin.D = np.full(6, 80.0)
    plugin.C = np.array(stiffness, float)
    plugin.w_d = np.zeros(6)
    plugin.axes = np.ones(6)
    plugin._uses_stiffness = True
    plugin.v_lin, plugin.v_ang = 1.0, 1.0
    plugin._ctx = SimpleNamespace(sim_time=0.0)
    plugin._stream = _GoalStream("auto", 0.2)
    plugin._goal_pos = plugin._goal_mat = None
    plugin._rest_pos, plugin._rest_mat = np.zeros(3), np.eye(3)
    plugin._ft = SimpleNamespace(
        read=lambda: (np.zeros(3), np.zeros(3)), measures="environment_on_tool"
    )
    plugin.read_pose = lambda: (np.zeros(3), np.eye(3))
    return plugin


def test_the_compliance_law_damps_relative_to_the_goal_velocity():
    """Damping on absolute velocity brakes a tool that is exactly keeping pace with its goal; the
    steady-state lag it leaves on a stiff axis is ``D v / C``. Relative to the goal's velocity it
    holds that pace, and a zero-stiffness axis -- under force control -- is given nothing to follow."""
    law = _compliance([500.0, 500.0, 0.0, 0.0, 0.0, 0.0])
    law._twist = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    law.set_goal(np.zeros(3), None, np.array([0.05, 0.0, 0.03, 0.0, 0.0, 0.0]))

    twist = law._wrench_twist(0.002)
    assert twist[0] == pytest.approx(0.05), "a stiff axis moving with its goal must hold its pace"
    assert twist[2] == pytest.approx(0.0), "a force-controlled axis must not follow the goal"

    law._stream = _GoalStream("off", 0.2)
    assert law._wrench_twist(0.002)[0] < 0.05, "without it, damping brakes the tool off its goal"


def test_the_feedforward_key_is_validated():
    plugin = CartesianAdmittancePlugin({}, entity="arm")
    assert plugin.validate_config({"feedforward": "sometimes"})
    assert plugin.validate_config({"feedforward_window_s": 0.0})
    assert not plugin.validate_config({"feedforward": "off", "feedforward_window_s": 0.1})
