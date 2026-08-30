"""Payload and wind: the two things an aerial *experiment* varies, and the traps in each.

These pin behaviour that a "does it load" check cannot see:

* ``test_flight_envelope_collapses_with_payload`` is the substantive one. The Crazyflie's
  thrust-to-weight ratio is 1.32, so payload is not a detail -- it is the flight envelope, and there
  is a real boundary near T/W = 1 that a campaign is meant to find. If this test ever goes flat
  (every payload hovering, or none), the model's thrust authority has drifted and every aerial
  campaign built on it is measuring nothing.
* ``test_turbulence_is_reproducible`` pins the seeding contract. Turbulence drawn from a stateful
  generator would still *look* like turbulence while making a campaign's cells differ by noise
  nobody chose -- indistinguishable from the effect under test, and invisible until someone tried to
  reproduce a result.
* ``test_refuses_a_second_wind_owner`` and ``test_refuses_an_offset_payload`` pin two loud failures
  chosen over quiet approximations.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: Airframe mass and peak collective thrust, from the model. T/W = 1 at 35.7 g total.
AIRFRAME_KG = 0.027
MAX_THRUST_N = 0.35
GRAVITY = 9.81


def _world(*, payload=None, wind=None, sim_extra=None):
    """A drone holding 1 m, with whatever payload/wind the test is about."""
    components = [{"quadrotor_controller": {"target": [0.0, 0.0, 1.0]}}]
    if payload is not None:
        components.append({"payload": payload})
    world = {
        "sim": {
            "pacing": "asap",
            "integrator": "rk4",
            # Not optional: with no medium the drone is undamped and wind does nothing at all.
            "density": 1.225,
            "viscosity": 1.8e-5,
            "seed": 7,
            **(sim_extra or {}),
        },
        "components": [
            {
                "spawn_robot": {"model": "crazyflie_2", "prefix": "cf2_", "pos": [0.0, 0.0]},
                "name": "drone",
                "components": components,
            }
        ],
    }
    if wind is not None:
        world["components"].append({"wind_field": wind})
    return world


def _fly(world, seconds, seed=7):
    """Run the world and return the drone's position track as an (N, 3) array.

    ``ctx.seed`` is **driver-owned** (see ``roqsim.context``): the runner resolves it from
    ``sim.seed`` / ``--seed``, and an Engine driven directly has none. These tests are the driver,
    so they assign it here -- otherwise every seed would silently collapse to 0 and the two
    reproducibility tests below would pass for the wrong reason.
    """
    engine = Engine(load_config_from_dict(world, base_dir=None))
    engine.setup()
    engine.ctx.seed = seed
    engine.reset()
    controller = next(
        p for p in engine.plugins if type(p).__name__ == "QuadrotorControllerPlugin"
    )
    track = []
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()
        track.append(np.array(controller.read_state()[:3]))
    return np.array(track)


# -- payload -------------------------------------------------------------------------------------

def test_payload_adds_mass_to_the_airframe():
    engine = Engine(load_config_from_dict(_world(payload={"mass": 0.005}), base_dir=None))
    engine.setup()
    total = float(engine.ctx.model.body_mass.sum())
    assert total == pytest.approx(AIRFRAME_KG + 0.005, abs=1e-9)


def test_zero_payload_leaves_the_model_untouched():
    # The unloaded cell of a sweep must be identical to a world that never declared a payload,
    # otherwise the sweep's baseline is its own separate configuration.
    loaded = Engine(load_config_from_dict(_world(payload={"mass": 0.0}), base_dir=None))
    loaded.setup()
    bare = Engine(load_config_from_dict(_world(), base_dir=None))
    bare.setup()
    assert float(loaded.ctx.model.body_mass.sum()) == pytest.approx(
        float(bare.ctx.model.body_mass.sum()), abs=1e-12
    )


@pytest.mark.parametrize(
    "payload_kg, expect_hover",
    [
        (0.000, True),    # T/W 1.32
        (0.003, True),    # T/W 1.19
        (0.009, False),   # T/W 0.99 -- just under, and it cannot leave the floor
        (0.012, False),   # T/W 0.91
    ],
)
def test_flight_envelope_collapses_with_payload(payload_kg, expect_hover):
    """There is a real boundary near T/W = 1, and it is what a campaign searches for."""
    track = _fly(_world(payload={"mass": payload_kg}), seconds=8.0)
    final_z = float(track[-1][2])
    thrust_to_weight = MAX_THRUST_N / ((AIRFRAME_KG + payload_kg) * GRAVITY)
    if expect_hover:
        assert thrust_to_weight > 1.0
        assert final_z > 0.85, f"T/W={thrust_to_weight:.2f} should still fly, got z={final_z:.3f}"
    else:
        assert thrust_to_weight < 1.05
        assert final_z < 0.2, f"T/W={thrust_to_weight:.2f} should not fly, got z={final_z:.3f}"


def test_refuses_an_offset_payload():
    # An offset payload shifts the centre of mass and adds a parallel-axis term. Approximating it as
    # a centred point mass would report an attitude result for an airframe nobody configured.
    engine = Engine
    with pytest.raises(Exception) as err:
        cfg = load_config_from_dict(
            _world(payload={"mass": 0.005, "offset": [0.1, 0.0, 0.0]}), base_dir=None
        )
        engine(cfg).setup()
    assert "offset" in str(err.value)


# -- wind ----------------------------------------------------------------------------------------

def test_steady_wind_pushes_the_drone_downwind():
    calm = _fly(_world(), seconds=6.0)
    windy = _fly(_world(wind={"steady": [3.0, 0.0, 0.0]}), seconds=6.0)
    # A PD position loop trims into the wind and holds a steady offset rather than returning to zero.
    assert float(windy[-1][0]) > float(calm[-1][0]) + 0.2


def test_gust_displaces_further_than_the_steady_wind_alone():
    steady = {"steady": [3.0, 0.0, 0.0]}
    gusty = dict(steady, gust={"magnitude": 6.0, "onset": 2.0, "duration": 1.5})
    without = _fly(_world(wind=steady), seconds=6.0)
    with_gust = _fly(_world(wind=gusty), seconds=6.0)
    assert float(np.max(with_gust[:, 0])) > float(np.max(without[:, 0])) + 0.5


def test_turbulence_is_reproducible():
    """Same seed, same weather -- turbulence is a function of the world, not of draw order."""
    world = _world(wind={"steady": [2.0, 0.0, 0.0],
                         "turbulence": {"intensity": 0.8, "length_scale": 4.0}})
    assert np.allclose(_fly(world, seconds=4.0), _fly(world, seconds=4.0), atol=0.0)


def test_turbulence_follows_the_run_seed():
    wind = {"steady": [2.0, 0.0, 0.0], "turbulence": {"intensity": 0.8, "length_scale": 4.0}}
    a = _fly(_world(wind=wind), seconds=4.0, seed=7)
    b = _fly(_world(wind=wind), seconds=4.0, seed=99)
    assert not np.allclose(a, b, atol=1e-6)


def test_turbulence_variance_does_not_track_the_timestep():
    """The sqrt(2a) term is what makes this a wind model rather than a timestep artefact.

    Without it, refining the timestep would quietly make the world calmer -- and a convergence study
    would then read as a physics result.

    Two things about the shape of this test, both learned the hard way:

    * It measures the **wind signal**, not the drone. The drone's response is the wrong probe: the
      controller runs once per step, so halving the timestep also doubles the control rate, and the
      two effects are not separable in a position track.
    * The turbulence length scale is short (0.5 m) and the window long, so the sample actually
      contains ~80 correlation times. At the plugin's default 5 m scale, 20 s of run is about five
      independent samples and the measured spread is dominated by estimator noise -- the first
      version of this test failed against a filter that was perfectly correct.
    """
    sigma, length, seconds = 1.0, 0.5, 40.0
    spreads = {}
    for timestep in (0.002, 0.001):
        world = _world(
            wind={"steady": [2.0, 0.0, 0.0],
                  "turbulence": {"intensity": sigma, "length_scale": length}},
            sim_extra={"timestep": timestep},
        )
        engine = Engine(load_config_from_dict(world, base_dir=None))
        engine.setup()
        engine.ctx.seed = 7
        engine.reset()
        samples = []
        for _ in range(int(seconds / timestep)):
            engine.step()
            samples.append(float(engine.ctx.model.opt.wind[0]))
        # Drop the transient: the filter starts at zero and needs a few correlation times to reach
        # its stationary distribution.
        spreads[timestep] = float(np.std(samples[len(samples) // 2:]))

    assert spreads[0.001] == pytest.approx(spreads[0.002], rel=0.2), spreads
    # And both must sit at the sigma that was asked for -- equal-but-wrong would pass the line above.
    for timestep, spread in spreads.items():
        assert spread == pytest.approx(sigma, rel=0.2), (timestep, spread)


def test_refuses_a_second_wind_owner():
    # sim.wind and wind_field are two owners of one global: the compiled model would say one thing
    # and the first tick another, and the run's provenance would record the overwritten value.
    with pytest.raises(RuntimeError, match="One owner per knob"):
        cfg = load_config_from_dict(
            _world(wind={"steady": [1.0, 0.0, 0.0]}, sim_extra={"wind": [1.0, 0.0, 0.0]}),
            base_dir=None,
        )
        Engine(cfg).setup()


def test_warns_when_there_is_no_medium(caplog):
    world = _world(wind={"steady": [3.0, 0.0, 0.0]})
    world["sim"]["density"] = 0.0
    world["sim"]["viscosity"] = 0.0
    with caplog.at_level(logging.WARNING):
        Engine(load_config_from_dict(world, base_dir=None)).setup()
    assert any("wind has no effect" in r.getMessage() for r in caplog.records
               if r.name.endswith("wind_field"))
