"""payload and wind_field: carried mass on a body, and the weather the world is in.

Both are stated in the world and both change the physics without changing any robot's code, so the
things worth pinning are the ones a "does it load" check cannot see: that a mass write actually
reaches the dynamics, that turbulence is a function of the world rather than of draw order, and that
the two refusals (an offset payload, a second owner of ``opt.wind``) are loud.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.payload import PayloadPlugin
from roqsim.plugins.wind_field import WindFieldPlugin

# A single free body in air. Gravity is off so that qacc IS force/mass: the payload assertion is then
# arithmetic rather than the outcome of a constraint solve.
SCENE = """
<mujoco model="payload_test">
  <option timestep="0.002" gravity="0 0 0" density="1.225" viscosity="1.8e-5"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <freejoint name="cart_free"/>
      <geom name="cart" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ctx(scene=SCENE, *, wind=None, seed=7, body="cart"):
    model = mujoco.MjModel.from_xml_string(scene)
    if wind is not None:
        model.opt.wind = wind
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.seed = seed
    # The entity a spawn plugin would register: name, root body, and the base joint that is the
    # general fallback when a model does not call its root `base_link`.
    ctx.entities.add(
        Entity(name="robot", kind="robot", body=body, meta={"base_joint": "cart_free"})
    )
    return ctx


def _plugin(cls, config, *, name="p"):
    """A configured plugin. Config errors are raised rather than returned."""
    plugin = cls(config, name=name, entity="robot")
    errors = plugin.validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return plugin


# -- payload ---------------------------------------------------------------------------------

def test_payload_adds_mass_to_the_body():
    ctx = _ctx()
    _plugin(PayloadPlugin, {"mass": 0.5}).configure(ctx)
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    assert float(ctx.model.body_mass[bid]) == pytest.approx(1.5, abs=1e-9)


def test_the_added_mass_reaches_the_dynamics():
    """mj_setConst is what makes a body_mass write take effect rather than sit in an array."""
    accelerations = {}
    for mass in (0.0, 1.0):
        ctx = _ctx()
        _plugin(PayloadPlugin, {"mass": mass}).configure(ctx)
        ctx.data.xfrc_applied[
            mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "cart"), 0
        ] = 1.0
        mujoco.mj_forward(ctx.model, ctx.data)
        accelerations[mass] = float(ctx.data.qacc[0])
    # 1 N on 1 kg, then on 2 kg.
    assert accelerations[0.0] == pytest.approx(1.0, rel=1e-6)
    assert accelerations[1.0] == pytest.approx(0.5, rel=1e-6)


def test_zero_payload_leaves_the_model_untouched():
    # The unloaded cell of a sweep must be identical to a world that never declared a payload,
    # otherwise the sweep's baseline is its own separate configuration.
    loaded, bare = _ctx(), _ctx()
    _plugin(PayloadPlugin, {"mass": 0.0}).configure(loaded)
    assert np.array_equal(loaded.model.body_mass, bare.model.body_mass)


def test_payload_resolves_the_body_owning_the_base_joint():
    """The entity's registered body may be absent from the compiled model; the base joint is not."""
    ctx = _ctx(body="base_link")
    _plugin(PayloadPlugin, {"mass": 0.5}).configure(ctx)
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    assert float(ctx.model.body_mass[bid]) == pytest.approx(1.5, abs=1e-9)


def test_refuses_an_offset_payload():
    # An offset payload shifts the centre of mass and adds a parallel-axis term. Approximating it as
    # a centred point mass would report a result for a body nobody configured.
    with pytest.raises(ValueError, match="offset"):
        _plugin(PayloadPlugin, {"mass": 0.5, "offset": [0.1, 0.0, 0.0]})


def test_requires_a_mass():
    with pytest.raises(ValueError, match="mass"):
        _plugin(PayloadPlugin, {})


def test_names_the_entity_it_could_not_find():
    with pytest.raises(RuntimeError, match="no entity named"):
        _plugin(PayloadPlugin, {"mass": 0.5, "robot": "absent"}).configure(_ctx())


# -- wind ------------------------------------------------------------------------------------

def _wind_track(config, seconds, *, seed=7, timestep=None):
    """The wind signal itself, sampled per tick -- not a body's response to it."""
    scene = SCENE if timestep is None else SCENE.replace('timestep="0.002"', f'timestep="{timestep}"')
    ctx = _ctx(scene, seed=seed)
    plugin = _plugin(WindFieldPlugin, config)
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
        _plugin(WindFieldPlugin, {"steady": [1.0, 0.0, 0.0]}).configure(ctx)


def test_refuses_a_seed_of_its_own():
    with pytest.raises(ValueError, match="seed"):
        _plugin(WindFieldPlugin, {"steady": [1.0, 0.0, 0.0], "seed": 3})


def test_warns_when_there_is_no_medium(caplog):
    vacuum = SCENE.replace('density="1.225" viscosity="1.8e-5"', 'density="0" viscosity="0"')
    with caplog.at_level(logging.WARNING):
        _plugin(WindFieldPlugin, {"steady": [3.0, 0.0, 0.0]}).configure(_ctx(vacuum))
    assert any("wind has no effect" in r.getMessage() for r in caplog.records)
