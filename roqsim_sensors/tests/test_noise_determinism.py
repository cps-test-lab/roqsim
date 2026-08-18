"""Sensor noise must be reproducible, and reproducible *from a restored state*.

Before this existed, the sensors read a ``ctx.rng`` that nothing ever set, so they silently fell back to
a module-level generator seeded from OS entropy at import -- and a run with noisy sensors could not be
repeated at all. That matters beyond capture: for a repo whose object of study is reproducing published
experiments, an irreproducible run is a defect on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="6 6 .1"/>
    <body pos="2 0 .5"><geom type="box" size=".3 .3 .5"/></body>
    <body name="base" pos="0 0 .2">
      <geom type="cylinder" size=".2 .2"/>
      <site name="scan" pos="0 0 .3"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def world(tmp_path):
    scene = tmp_path / "s.xml"
    scene.write_text(_SCENE)
    return {
        "sim": {"world": str(scene)},
        "plugins": [
            {"lidar": {"name": "front", "site": "scan", "num_rays": 64, "range_stddev": 0.05}}
        ],
    }


def _scan_after(world, seed, steps=240):
    """Step a fresh world with ``seed`` and read the lidar's published scan."""
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = seed
    engine.setup()
    engine.reset()
    endpoint = next(
        e for e in engine.ctx.interface.all() if e.direction == "out" and "scan" in e.name
    )
    try:
        for _ in range(steps):
            engine.step()
        return np.asarray(endpoint.read().ranges, dtype=float).copy()
    finally:
        engine.shutdown()


def test_the_same_seed_reproduces_the_noisy_scan(world):
    assert np.array_equal(_scan_after(world, 7), _scan_after(world, 7))


def test_a_different_seed_changes_it(world):
    assert not np.array_equal(_scan_after(world, 7), _scan_after(world, 99))


def test_the_noise_is_actually_applied(world):
    """Guards against the tests above passing because noise was silently off."""
    scan = _scan_after(world, 7)
    finite = scan[np.isfinite(scan)]
    assert len(finite) > 5
    assert finite.std() > 0.005, "no measurable spread -- is range_stddev reaching the sensor?"


def test_noise_is_a_function_of_sim_time_not_of_call_count(world):
    """The property that makes a *replay* exact: the same sim time gives the same draw.

    A stateful generator would give a different answer depending on how many draws came before, so a
    sensor re-run from a recording could never match what the live run published.
    """
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 11
    engine.setup()
    engine.reset()
    try:
        engine.ctx.data.time = 12.5
        first = engine.ctx.rng_for("front").normal(size=8)
        # Burn a lot of draws in between: a stateful generator would have moved on.
        for _ in range(50):
            engine.ctx.rng_for("front").normal(size=8)
        engine.ctx.data.time = 12.5
        again = engine.ctx.rng_for("front").normal(size=8)
        assert np.array_equal(first, again)
    finally:
        engine.shutdown()


def test_two_sensors_get_independent_streams(world):
    """Keyed on the plugin's own name, so two lidars on one robot do not share noise."""
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 5
    engine.setup()
    engine.reset()
    try:
        engine.ctx.data.time = 3.0
        assert not np.array_equal(
            engine.ctx.rng_for("front").normal(size=8),
            engine.ctx.rng_for("rear").normal(size=8),
        )
    finally:
        engine.shutdown()


def test_the_stream_key_is_stable_across_processes():
    """Python's hash() is salted per process, which would make a run irreproducible across runs."""
    import subprocess
    import sys

    code = (
        "from roqsim.context import SimContext;"
        "c=SimContext({});c.seed=3;"
        "import numpy as np;print(list(np.round(c.rng_for('front').normal(size=3), 8)))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        for _ in range(2)
    }
    assert len(outs) == 1 and outs != {""}


def test_the_seed_is_recorded_in_a_recordings_provenance(world, tmp_path):
    """A recomputed sensor needs the seed the live run drew, so it has to travel with the recording."""
    import json

    from roqsim.capture import StateRecorder, snap_fps

    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 4242
    engine.setup()
    engine.reset()
    rec = StateRecorder(engine.ctx, tmp_path / "r.npz", snap_fps(25, 0.002), world="w")
    try:
        for _ in range(120):
            engine.step()
            rec.sample(engine.ctx)
        rec.close()
    finally:
        engine.shutdown()
    meta = json.loads(str(np.load(tmp_path / "r.npz")["meta"]))
    assert meta["seed"] == 4242
