"""A sensor whose period is a whole number of physics steps fires on that step, every period.

``sim_time`` is accumulated float steps, and fifty steps of 0.002 land a few ULPs short of 0.1. A
gate that compares the elapsed time against the period exactly therefore fires one step late every
other period: 99 scans in ten seconds from a 10 Hz scanner, spaced 50 and 51 steps alternately. A
stack that derives the scan rate from the stamps reads that alias as the sensor's behaviour.
"""

from __future__ import annotations

import roqsim  # noqa: F401, I001  (selects MuJoCo's GL backend before anything imports mujoco)

import pytest  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim_sensors.plugins.lidar import LidarPlugin  # noqa: E402

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

SECONDS = 10.0


@pytest.mark.parametrize("rate_hz", [10.0, 25.0, 100.0])
def test_a_scanner_fires_exactly_rate_times_a_second(tmp_path, rate_hz):
    scene = tmp_path / "s.xml"
    scene.write_text(_SCENE)
    world = {
        "sim": {"world": str(scene)},
        "components": [{"lidar": {"site": "scan", "num_rays": 16, "rate_hz": rate_hz}}],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 7
    engine.setup()
    engine.reset()
    try:
        lidar = next(p for p in engine.plugins if isinstance(p, LidarPlugin))
        casts = 0
        last = lidar._last_cast
        for _ in range(round(SECONDS / engine.ctx.dt)):
            engine.step()
            if lidar._last_cast != last:
                casts += 1
                last = lidar._last_cast
        assert casts == round(rate_hz * SECONDS), f"{casts} casts in {SECONDS} s at {rate_hz} Hz"
    finally:
        engine.shutdown()
