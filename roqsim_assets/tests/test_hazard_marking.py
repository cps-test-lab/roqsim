"""The floor marking in warning stripes.

What matters about it is that it is paint: it covers exactly the area it was given, in two colours
that share it about evenly, and nothing collides with it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.raycast import VISIBLE_GROUPS
from roqsim_assets.plugins.hazard_marking import (
    HazardMarkingPlugin,
    _area,
    marked_rectangles,
    stripe_polygons,
)


def _built(tmp_path, **marking):
    cfg = {
        "sim": {},
        "components": [{"hazard_marking": {"prefix": "m_", **marking}, "name": "spot"}],
    }
    engine = Engine(load_config_from_dict(cfg, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    return engine


def _painted(length, width, band, stripe=0.1, angle=math.radians(45.0)):
    first, second = stripe_polygons(marked_rectangles(length, width, band), stripe, angle)
    return sum(map(_area, first)), sum(map(_area, second))


def test_a_filled_marking_covers_its_rectangle_half_in_each_colour():
    first, second = _painted(2.0, 1.0, 0.0)
    assert first + second == pytest.approx(2.0)
    assert first == pytest.approx(second, rel=0.1)


def test_an_outline_covers_the_band_and_leaves_the_inside_bare():
    first, second = _painted(1.5, 1.1, 0.1)
    assert first + second == pytest.approx(1.5 * 1.1 - 1.3 * 0.9)


def test_a_band_that_leaves_no_inside_is_the_filled_rectangle():
    assert marked_rectangles(1.0, 0.4, 0.2) == [(-0.5, -0.2, 0.5, 0.2)]


def test_no_stripe_is_wider_than_asked():
    normal = (-math.sin(math.radians(30.0)), math.cos(math.radians(30.0)))
    for polygons in stripe_polygons(marked_rectangles(2.0, 1.0, 0.0), 0.07, math.radians(30.0)):
        for polygon in polygons:
            reach = [normal[0] * x + normal[1] * y for x, y in polygon]
            assert max(reach) - min(reach) <= 0.07 + 1e-9


def test_the_marking_lies_on_the_floor_where_its_pose_says_and_collides_with_nothing(tmp_path):
    engine = _built(
        tmp_path,
        pose={"position": {"x": 3.0, "y": -2.0, "z": 0.0}, "orientation": {"yaw": math.pi / 2}},
        length=1.5,
        width=1.1,
    )
    model = engine.ctx.model
    geoms = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("m_stripes_")
    ]
    assert len(geoms) == 2
    for g in geoms:
        assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
        # the paint's own vertices in the world: MuJoCo turns a mesh into its principal axes, so
        # the geom's frame says nothing about where the paint lies
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (
            model.mesh_vert[start : start + count] @ engine.ctx.data.geom_xmat[g].reshape(3, 3).T
            + engine.ctx.data.geom_xpos[g]
        )
        # turned a quarter: 'length' runs along the world's y
        assert np.abs(world[:, 0] - 3.0).max() == pytest.approx(0.55, abs=1e-6)
        assert np.abs(world[:, 1] + 2.0).max() == pytest.approx(0.75, abs=1e-6)
        assert world[:, 2].min() == pytest.approx(0.0, abs=1e-6)
        assert world[:, 2].max() == pytest.approx(0.002, abs=1e-6)
    assert engine.ctx.entities.get("spot").body == "m_hazard_marking"


def test_a_ray_aimed_at_the_floor_returns_from_the_paint(tmp_path):
    """Paint has no contact, but it is in the groups a scanner's rays see: a downward ray returns
    from its top, the paint's thickness above the floor."""
    engine = _built(tmp_path, band=0.0)
    hit = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        engine.ctx.model,
        engine.ctx.data,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        VISIBLE_GROUPS,
        1,
        -1,
        hit,
    )
    name = mujoco.mj_id2name(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0]))
    assert name.startswith("m_stripes_")
    assert distance == pytest.approx(0.998, abs=1e-6)


@pytest.mark.parametrize(
    "config, word",
    [
        ({"length": 0}, "'length'"),
        ({"band": -0.1}, "'band'"),
        ({"angle": 0}, "'angle'"),
        ({"colors": [[1, 1, 0]]}, "'colors'"),
        ({"yaw": 1.0}, "'yaw'"),
        ({"pose": {"orientation": {"roll": 0.3}}}, "tilts"),
    ],
)
def test_a_wrong_value_is_named(config, word):
    assert any(word in e for e in HazardMarkingPlugin(config).validate_config(config))
