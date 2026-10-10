"""``force_torque``'s transducer: a per-episode bias, a linear drift and a measuring range.

The scene is ``test_force_torque``'s hanging link, static after settling, so any change in the
reading over time is the transducer's and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_sensors.plugins.force_torque import ForceTorquePlugin
from test_force_torque import EXPECTED_FZ, _plugin, _settled


def _read_at(engine, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    for _ in range(steps):
        engine.step()
    force, torque = _plugin(engine).read()
    return force, torque, engine.ctx.sim_time


def test_defaults_leave_the_reading_alone():
    force, _ = _plugin(_settled(frame="sensor", invert=False)).read()
    assert force[2] == pytest.approx(EXPECTED_FZ, abs=1e-6)


def test_a_bias_is_constant_in_an_episode_and_within_its_bound():
    engine = _settled(frame="sensor", invert=False, bias_force=0.5, bias_torque=0.1)
    f1, t1, _ = _read_at(engine, 0)
    f2, t2, _ = _read_at(engine, 500)
    offset = f1 - np.array([0.0, 0.0, EXPECTED_FZ])
    assert np.all(np.abs(offset) <= 0.5) and np.abs(offset).max() > 1e-3
    assert np.all(np.abs(t1) <= 0.1 + 1e-6)
    assert np.allclose(f1, f2, atol=1e-6) and np.allclose(t1, t2, atol=1e-6)


def test_a_bias_follows_the_seed_and_is_redrawn_per_episode():
    def offset(engine):
        return _plugin(engine).read()[0] - np.array([0.0, 0.0, EXPECTED_FZ])

    a = _settled(frame="sensor", invert=False, bias_force=0.5)
    b = _settled(frame="sensor", invert=False, bias_force=0.5)
    assert np.allclose(offset(a), offset(b), atol=1e-9)  # same seed, same draw
    first = offset(a)
    a.reset()
    for _ in range(200):
        a.step()
    assert not np.allclose(offset(a), first, atol=1e-6)  # a new episode draws its own


def test_drift_grows_linearly_from_the_episode_start():
    engine = _settled(frame="sensor", invert=False, drift_force=0.2)
    f1, _, t1 = _read_at(engine, 0)
    f2, _, t2 = _read_at(engine, 1000)
    f3, _, t3 = _read_at(engine, 1000)
    rate = (f2 - f1) / (t2 - t1)
    assert np.all(np.abs(rate) <= 0.2) and np.abs(rate).max() > 1e-3
    assert np.allclose((f3 - f2) / (t3 - t2), rate, atol=1e-6)
    assert np.allclose(f1 - np.array([0.0, 0.0, EXPECTED_FZ]), rate * t1, atol=1e-6)


def test_a_tare_removes_the_bias_and_the_drift_so_far():
    engine = _settled(frame="sensor", invert=False, bias_force=0.5, drift_force=0.2)
    plugin = _plugin(engine)
    plugin.tare()
    f0, _ = plugin.read()
    assert np.allclose(f0, 0.0, atol=1e-9)
    f1, _, _ = _read_at(engine, 1000)
    # What comes back after a tare is the drift since the tare, no more.
    assert np.all(np.abs(f1) <= 0.2 * 1000 * engine.ctx.dt + 1e-9) and np.abs(f1).max() > 0


def test_the_range_saturates_each_channel():
    force, _ = _plugin(_settled(frame="sensor", invert=False, range_force=10.0)).read()
    assert EXPECTED_FZ > 10.0
    assert force[2] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "key, value",
    [("bias_force", -1.0), ("drift_torque", -0.1), ("range_force", 0.0), ("range_torque", -5.0)],
)
def test_negative_bounds_and_an_empty_range_are_refused(key, value):
    errors = ForceTorquePlugin({"site": "s"}).validate_config({"site": "s", key: value})
    assert any(key in e for e in errors)
