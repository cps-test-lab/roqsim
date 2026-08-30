"""The flight envelope: what the core ``payload`` and ``wind_field`` plugins do to a real airframe.

The plugins themselves are pinned in ``roqsim/tests/test_payload_and_wind.py``. What lives here is
the aerial consequence, which only exists once a model with bounded thrust is flying:

* ``test_flight_envelope_collapses_with_payload`` is the substantive one. The Crazyflie's
  thrust-to-weight ratio is 1.32, so payload is the flight envelope, and there is a real boundary
  near T/W = 1 that a campaign is meant to find. If this test goes flat (every payload hovering, or
  none), the model's thrust authority has drifted and every aerial campaign built on it is measuring
  nothing.
* the wind tests check that a gust reaches the vehicle through MuJoCo's drag terms at all, which is
  what makes wind a disturbance rather than a number in the model.
"""


from __future__ import annotations

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
    ``sim.seed`` / ``--seed``, and an Engine driven directly has none, so this test assigns it.
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


