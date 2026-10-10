# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""``ArmHandle.follow_trajectory``: a joint trajectory executed on simulated time, with feed-forward.

The sampler reproduces what it is given (a quintic between knots carrying accelerations, a cubic with
velocities only, a line otherwise). The controller commands the sample every step, holds the last knot,
and gives way to any other command. ``trajectory_feedforward: dynamics`` makes a position servo track a
sweep at the joint's speed limit, which the bare position target cannot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin
from roqsim_manipulation.trajectory import JointTrajectory

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

DT = 0.0005


def _poly(t):
    c = [0.3, -1.2, 0.5, 2.0, -0.7, 0.4]
    q = sum(c[k] * t**k for k in range(6))
    qd = sum(k * c[k] * t ** (k - 1) for k in range(1, 6))
    qdd = sum(k * (k - 1) * c[k] * t ** (k - 2) for k in range(2, 6))
    return q, qd, qdd


def test_quintic_knots_reproduce_a_quintic():
    t = np.array([0.0, 0.7])
    q, qd, qdd = (np.array(v)[:, None] for v in zip(*[_poly(x) for x in t], strict=True))
    tr = JointTrajectory.build(t, q, qd, qdd)
    for x in np.linspace(0.0, 0.7, 15):
        a, b, c = tr.sample(x)
        e = _poly(x)
        assert a[0] == pytest.approx(e[0], abs=1e-12)
        assert b[0] == pytest.approx(e[1], abs=1e-10)
        assert c[0] == pytest.approx(e[2], abs=1e-9)


def test_cubic_and_linear_knots():
    t = [0.0, 1.0]
    cubic = JointTrajectory.build(t, [[0.0], [1.0]], [[0.0], [0.0]])
    assert cubic.sample(0.5)[0][0] == pytest.approx(0.5)
    assert cubic.sample(0.5)[1][0] == pytest.approx(1.5)  # smoothstep's peak slope
    line = JointTrajectory.build(t, [[0.0], [2.0]])
    assert line.sample(0.25)[0][0] == pytest.approx(0.5)
    assert line.sample(0.25)[1][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"times": [0.0, 0.0], "positions": [[0.0], [1.0]]}, "increase"),
        ({"times": [0.0, 1.0], "positions": [[0.0]]}, "one row per time"),
        (
            {"times": [0.0, 1.0], "positions": [[0.0], [1.0]], "accelerations": [[0.0], [0.0]]},
            "need velocities",
        ),
        ({"times": [0.0, 1.0], "positions": [[0.0], [math.nan]]}, "finite"),
    ],
)
def test_malformed_knots_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JointTrajectory.build(**kwargs)


def test_an_unknown_feedforward_is_refused():
    cfg = {"arm": "ur5e", "trajectory_feedforward": "torque"}
    errors = ArmControllerPlugin(cfg).validate_config(cfg)
    assert any("trajectory_feedforward" in e for e in errors), errors


def _arm(tmp_path, **controller):
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": DT},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "ur5e_",
                        "home": [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
                        "actuators": {"control": "position", "p": 5000, "d": 100},
                    },
                    "name": "ur5e",
                    "components": [{"arm_controller": controller}],
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    plugin = next(p for p in engine.plugins if isinstance(p, ArmControllerPlugin))
    return engine, plugin


def _reference(plugin):
    st = plugin.controller_state()
    return dict(zip(st.joint_names, st.reference.positions, strict=True))


def _measured(plugin):
    st = plugin.controller_state()
    return dict(zip(st.joint_names, st.feedback.positions, strict=True))


def _sweep(amplitude=1.0, peak=0.95 * math.pi, start=0.0):
    """One joint swept by ``amplitude`` with a smooth (raised-cosine) speed peaking at ``peak``; knots
    every 2 ms with exact velocities and accelerations."""
    T = 2.0 * amplitude / peak
    ts = np.append(np.arange(0.0, T, 0.002), T)
    w = 2.0 * math.pi / T
    q = start + amplitude * (ts / T - np.sin(w * ts) / (w * T))
    qd = amplitude / T * (1.0 - np.cos(w * ts))
    qdd = amplitude / T * w * np.sin(w * ts)
    return ts, q[:, None], qd[:, None], qdd[:, None]


def test_the_trajectory_is_commanded_on_sim_time_and_its_end_is_held(tmp_path):
    engine, arm = _arm(tmp_path)
    handle = engine.ctx.blackboard.get("arm:ur5e")
    start = _reference(arm)["elbow_joint"]
    ts, q, qd, qdd = _sweep(amplitude=0.5, peak=1.0, start=start)
    handle.follow_trajectory(["elbow_joint"], ts, q, qd, qdd)
    assert handle.trajectory_active()
    t0 = engine.ctx.sim_time
    for _ in range(int(round(0.25 / DT))):
        engine.step()
    # the reference is the sample of the step just commanded (at sim time before that step)
    expect = JointTrajectory.build(ts, q, qd, qdd).sample(engine.ctx.sim_time - DT - t0)[0][0]
    assert _reference(arm)["elbow_joint"] == pytest.approx(expect, abs=1e-9)
    for _ in range(int(round((ts[-1] + 0.3) / DT))):
        engine.step()
    assert not handle.trajectory_active()
    assert _reference(arm)["elbow_joint"] == pytest.approx(start + 0.5, abs=1e-12)


def test_another_command_ends_the_trajectory(tmp_path):
    engine, arm = _arm(tmp_path)
    handle = engine.ctx.blackboard.get("arm:ur5e")
    start = _reference(arm)["elbow_joint"]
    ts, q, qd, qdd = _sweep(amplitude=0.5, peak=1.0, start=start)
    handle.follow_trajectory(["elbow_joint"], ts, q, qd, qdd)
    engine.step()
    handle.set_targets(["elbow_joint"], [start])
    assert not handle.trajectory_active()
    for _ in range(200):
        engine.step()
    assert _reference(arm)["elbow_joint"] == pytest.approx(start, abs=1e-12)


def test_an_unknown_joint_and_an_inactive_controller_are_refused(tmp_path):
    engine, arm = _arm(tmp_path)
    handle = engine.ctx.blackboard.get("arm:ur5e")
    with pytest.raises(ValueError, match="unknown joints"):
        handle.follow_trajectory(["knee"], [0.0, 1.0], [[0.0], [1.0]])
    handle.set_active(False)
    with pytest.raises(RuntimeError, match="inactive"):
        handle.follow_trajectory(["elbow_joint"], [0.0, 1.0], [[0.0], [1.0]])


def _tracking_error(tmp_path, feedforward):
    engine, arm = _arm(tmp_path, trajectory_feedforward=feedforward)
    handle = engine.ctx.blackboard.get("arm:ur5e")
    start = _reference(arm)["elbow_joint"]
    ts, q, qd, qdd = _sweep(amplitude=1.2, start=start)
    tr = JointTrajectory.build(ts, q, qd, qdd)
    handle.follow_trajectory(["elbow_joint"], ts, q, qd, qdd)
    t0 = engine.ctx.sim_time
    worst = 0.0
    for _ in range(int(round(ts[-1] / DT))):
        engine.step()
        planned = tr.sample(engine.ctx.sim_time - t0)[0][0]
        worst = max(worst, abs(_measured(arm)["elbow_joint"] - planned))
    return math.degrees(worst)


def test_dynamics_feedforward_tracks_a_sweep_at_the_speed_limit(tmp_path):
    """Swept at 171 deg/s (95 % of a UR's 180): the bare target trails by degrees, feed-forward does not."""
    bare = _tracking_error(tmp_path, "none")
    fed = _tracking_error(tmp_path, "dynamics")
    assert bare > 1.0
    assert fed < 0.05


def test_feedforward_needs_an_affine_position_servo(tmp_path):
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": DT},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "ur5e_",
                        "actuators": {"control": "effort", "ctrlrange": [-150, 150]},
                    },
                    "name": "ur5e",
                    "components": [{"arm_controller": {"trajectory_feedforward": "velocity"}}],
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    with pytest.raises(Exception, match="affine position servo"):
        engine.setup()
