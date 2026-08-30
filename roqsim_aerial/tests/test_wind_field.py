"""wind_field: the signal itself, measured without a vehicle in the way.

A body's response mixes the wind with the controller driving it, so these read ``model.opt.wind``
directly on a bare free body. What a gust does to a real airframe is ``test_flight_envelope``.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim_aerial.plugins.wind_field import WindFieldPlugin

# A single free body in air (density and viscosity set: wind acts through MuJoCo's drag terms, so a
# vacuum would make every assertion here read zero). Gravity is off -- nothing is flying, the wind
# signal is read straight off the model.
SCENE = """
<mujoco model="wind_test">
  <option timestep="0.002" gravity="0 0 0" density="1.225" viscosity="1.8e-5"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <freejoint name="cart_free"/>
      <geom name="cart" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ctx(scene=SCENE, *, wind=None, seed=7):
    model = mujoco.MjModel.from_xml_string(scene)
    if wind is not None:
        model.opt.wind = wind
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.seed = seed
    # An entity so the plugin is configured the way a world configures it; wind owns no entity.
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="cart", meta={"base_joint": "cart_free"})
    )
    return ctx


def _plugin(config, *, name="wind"):
    """A configured plugin. Config errors are raised rather than returned."""
    plugin = WindFieldPlugin(config, name=name, entity="robot")
    errors = plugin.validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return plugin


def _wind_track(config, seconds, *, seed=7, timestep=None):
    """The wind signal itself, sampled per tick -- not a body's response to it."""
    scene = SCENE if timestep is None else SCENE.replace('timestep="0.002"', f'timestep="{timestep}"')
    ctx = _ctx(scene, seed=seed)
    plugin = _plugin(config)
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    track = []
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        plugin.pre_step(ctx)
        mujoco.mj_step(ctx.model, ctx.data)
        track.append(np.array(ctx.model.opt.wind))
    return np.array(track)


def test_steady_wind_is_the_mean_flow():
    track = _wind_track({"steady": [3.0, 0.0, 0.0]}, seconds=1.0)
    assert np.allclose(track, [3.0, 0.0, 0.0])


def test_the_gust_rises_and_falls_inside_its_window():
    track = _wind_track(
        {"steady": [1.0, 0.0, 0.0], "gust": {"magnitude": 4.0, "onset": 1.0, "duration": 1.0}},
        seconds=3.0,
    )
    dt = 0.002
    before, peak, after = track[int(0.5 / dt), 0], track[int(1.5 / dt), 0], track[int(2.5 / dt), 0]
    assert before == pytest.approx(1.0)
    assert peak == pytest.approx(5.0, rel=1e-3)   # steady + full magnitude at the 1-cosine peak
    assert after == pytest.approx(1.0)
    # Zero slope at both ends: the 1-cosine shape does not hand a controller a step to ring on.
    assert track[int(1.0 / dt), 0] == pytest.approx(1.0, abs=1e-3)


def test_turbulence_is_reproducible():
    """Same seed, same weather -- turbulence is a function of the world, not of draw order."""
    config = {"steady": [2.0, 0.0, 0.0], "turbulence": {"intensity": 0.8, "length_scale": 4.0}}
    assert np.array_equal(_wind_track(config, 2.0), _wind_track(config, 2.0))


def test_turbulence_follows_the_run_seed():
    config = {"steady": [2.0, 0.0, 0.0], "turbulence": {"intensity": 0.8, "length_scale": 4.0}}
    assert not np.allclose(
        _wind_track(config, 2.0, seed=7), _wind_track(config, 2.0, seed=99), atol=1e-6
    )


def test_turbulence_variance_does_not_track_the_timestep():
    """The sqrt(2a) term is what makes this a wind model rather than a timestep artefact.

    Without it, refining the timestep would quietly make the world calmer, and a convergence study
    would then read as a physics result. The length scale is short and the window long so that the
    sample holds ~80 correlation times: at the default 5 m scale the measured spread is dominated by
    estimator noise, which a correct filter would fail on.
    """
    sigma, length, seconds = 1.0, 0.5, 120.0
    config = {"steady": [2.0, 0.0, 0.0],
              "turbulence": {"intensity": sigma, "length_scale": length}}
    spreads = {}
    for timestep in (0.002, 0.001):
        track = _wind_track(config, seconds, timestep=timestep)[:, :2]
        # Drop the transient (the filter starts at zero) and pool both horizontal axes, which carry
        # the same sigma: the estimate is of a standard deviation, and it needs the samples.
        spreads[timestep] = float(np.std(track[len(track) // 2:], axis=0).mean())

    assert spreads[0.001] == pytest.approx(spreads[0.002], rel=0.2), spreads
    # And both must sit at the sigma that was asked for -- equal-but-wrong would pass the line above.
    for timestep, spread in spreads.items():
        assert spread == pytest.approx(sigma, rel=0.2), (timestep, spread)


def test_refuses_a_second_wind_owner():
    # sim.wind and wind_field are two owners of one global: the compiled model would say one thing
    # and the first tick another, and the run's provenance would record the overwritten value.
    ctx = _ctx(wind=[1.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="One owner per knob"):
        _plugin({"steady": [1.0, 0.0, 0.0]}).configure(ctx)


def test_refuses_a_seed_of_its_own():
    with pytest.raises(ValueError, match="seed"):
        _plugin({"steady": [1.0, 0.0, 0.0], "seed": 3})


def test_warns_when_there_is_no_medium(caplog):
    vacuum = SCENE.replace('density="1.225" viscosity="1.8e-5"', 'density="0" viscosity="0"')
    with caplog.at_level(logging.WARNING):
        _plugin({"steady": [3.0, 0.0, 0.0]}).configure(_ctx(vacuum))
    assert any("wind has no effect" in r.getMessage() for r in caplog.records)
