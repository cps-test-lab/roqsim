"""The rough-terrain demo world: the two things about it that are invisible when wrong.

A demo world usually needs no test -- it loads or it does not. This one has a coupling no assertion
about its YAML would catch, because both halves look fine on their own.

``spawn_robot`` places a robot at the rest height its model states, which is measured against flat
ground at z=0. A height field's ground at the spawn is whatever the terrain says, and only its
lowest sample is at z=0, so a robot placed anywhere else starts *inside* the hill and is thrown
clear on the first step. The world answers that by choosing a seed whose minimum is the grid centre
-- a fact about generated noise, not about the world file, so editing either half silently breaks
it.

The second is the point of the world at all: a terrain that compiles but is not driven over is
indistinguishable from a plane in every structural check. So the robot is driven, and must climb and
tilt.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from mobile_scene_utils import named

from roqsim.config import load_config
from roqsim.engine import Engine

WORLD = (
    Path(__file__).resolve().parents[1]
    / "src/roqsim_mobile/worlds/warthog_terrain_demo.yaml"
)


def _engine():
    engine = Engine(load_config(WORLD))
    engine.setup()
    engine.reset()
    return engine


def _terrain(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "HeightfieldPlugin")


def test_spawn_sits_on_the_lowest_ground():
    """The world's seed must keep its minimum where the robot is put down.

    Not "near" the minimum: the spawn uses a rest height stated for z=0 ground, so any elevation
    under it is penetration. Asserting the *grid* rather than the settled pose is deliberate -- a
    robot dropped into a hill also ends up resting on top of it, a second later and several metres
    from where the world asked for it.
    """
    engine = _engine()
    terrain = _terrain(engine)
    assert terrain.height_at(0.0, 0.0) == 0.0
    assert float(terrain.elevation.min()) == 0.0


def test_the_ground_is_not_a_plane():
    """The relief the world asks for is the relief it gets, over the ground it covers."""
    terrain = _terrain(_engine())
    relief = float(terrain.elevation.max() - terrain.elevation.min()) * terrain.height
    assert math.isclose(relief, 2.5, rel_tol=1e-6)
    # A sample of corners and mid-edges: terrain, not a bowl with one bump in it.
    heights = [terrain.height_at(x, y)
               for x in (-10.0, 0.0, 10.0) for y in (-10.0, 0.0, 10.0)]
    assert np.std(heights) > 0.3


def test_the_warthog_climbs_out_of_its_valley():
    """Drive the world's own ``test_cmd`` and require what only terrain produces.

    The thresholds are the run's behaviour with room under them, not a re-measurement: a plane gives
    exactly zero on all three, so what is being separated is terrain from no terrain.
    """
    engine = _engine()
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    z0 = float(data.xpos[bid][2])

    climb, tilt = 0.0, 0.0
    for _ in range(int(20.0 / engine.ctx.dt)):
        engine.step()
        climb = max(climb, float(data.xpos[bid][2]) - z0)
        rotation = data.xmat[bid].reshape(3, 3)
        # Angle between the base's own up-axis and the world's: pitch and roll in one number, with
        # no Euler convention to get wrong.
        tilt = max(tilt, float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))

    assert climb > 0.5, f"the robot gained only {climb:.2f} m -- is it driving on a plane?"
    assert math.degrees(tilt) > 10.0, f"peak tilt {math.degrees(tilt):.1f} deg is not a hillside"
    # And it is still upright: a demo that ends on its roof is not a demo.
    final = data.xmat[bid].reshape(3, 3)
    assert final[2, 2] > 0.7
