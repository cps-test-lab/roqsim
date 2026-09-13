"""``diff_drive``'s ``odom_noise``: the odometry is wrong, the motion is not, and the error reproduces.

Driven on a real base (the TurtleBot 3 Waffle: a true differential drive, no camera), because the
properties that matter are about what a stack would see against what the body did -- a noise model
that leaked into the wheel commands would still produce a plausible-looking odometry trace.
"""

from __future__ import annotations

# `roqsim` selects MuJoCo's GL backend on import, so it comes first (see test_wheels_roll.py).
import roqsim  # noqa: F401, I001
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim_mobile.plugins.diff_drive import DiffDrivePlugin  # noqa: E402

MODEL = "turtlebot3_waffle"
SPEED = 0.2
STEPS = 1500  # 3 s at 2 ms
NOISE = {"linear_stddev": 0.05, "angular_stddev": 0.05}


def _engine(noise=None, seed=7):
    components = [{"diff_drive": {"odom_noise": noise}}] if noise is not None else []
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {"spawn_robot": {"model": MODEL, "prefix": "z_"}, "name": "z", "components": components}
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    # A test that steps an Engine is its driver, and the seed is the driver's to set.
    engine.ctx.seed = seed
    engine.setup()
    engine.reset()
    return engine


def _drive(engine):
    """Drive straight for STEPS; return (odometry pose, the base body's true position)."""
    drive = next(p for p in engine.plugins if isinstance(p, DiffDrivePlugin))
    handle = engine.ctx.blackboard.get("robot:z")
    handle.drive(SPEED, 0.0, 0.0)
    for _ in range(STEPS):
        engine.step()
    truth = engine.ctx.data.body(f"z_{drive.base_body}").xpos.copy()
    return np.array(handle.read_odom()[:3]), truth


def _run(noise=None, seed=7):
    engine = _engine(noise, seed)
    try:
        return _drive(engine)
    finally:
        engine.shutdown()


def test_zero_noise_leaves_the_odometry_exact():
    exact, truth = _run()
    zero, truth_zero = _run({"linear_stddev": 0.0, "angular_stddev": 0.0})
    assert np.allclose(exact, zero)
    assert np.allclose(truth, truth_zero)


def test_noise_changes_the_odometry_and_not_the_motion():
    exact, truth = _run()
    noisy, truth_noisy = _run({**NOISE, "linear_scale": 1.1})
    assert np.allclose(truth, truth_noisy), (
        "the body moved differently: noise leaked into the motion"
    )
    assert not np.allclose(exact, noisy), "the odometry did not change"


def test_the_same_seed_reproduces_and_another_does_not():
    a, _ = _run(NOISE, seed=7)
    b, _ = _run(NOISE, seed=7)
    c, _ = _run(NOISE, seed=8)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_a_second_episode_draws_new_noise():
    """reset() restarts sim time; the episode in the key keeps trial 2 from replaying trial 1."""
    engine = _engine(NOISE)
    try:
        first, _ = _drive(engine)
        engine.reset()
        second, _ = _drive(engine)
    finally:
        engine.shutdown()
    assert not np.allclose(first, second)


def test_a_scale_bias_overstates_the_distance_by_that_factor():
    exact, _ = _run()
    biased, _ = _run({"linear_scale": 1.1})
    assert np.hypot(*biased[:2]) / np.hypot(*exact[:2]) == pytest.approx(1.1, rel=0.01)


@pytest.mark.parametrize(
    "noise",
    [{"linear_stddev": -0.1}, {"angular_scale": 0.0}, {"bias": 0.1}, 0.1],
    ids=["negative-stddev", "zero-scale", "unknown-key", "not-a-mapping"],
)
def test_a_bad_block_is_refused(noise):
    assert DiffDrivePlugin({}).validate_config({"odom_noise": noise})


def test_a_good_block_is_accepted():
    good = {
        "linear_stddev": 0.01,
        "angular_stddev": 0.02,
        "linear_scale": 1.02,
        "angular_scale": 0.98,
    }
    assert not DiffDrivePlugin({}).validate_config({"odom_noise": good})
