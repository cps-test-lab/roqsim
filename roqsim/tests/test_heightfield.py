# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``heightfield``: ground that is not flat, and that something can actually stand on.

The check that matters is physical rather than structural: a ball dropped over a hill must come to
rest ABOVE the ground plane's height, and one dropped over a valley below it. A height field that
compiles but is not collided against looks identical in every other assertion -- the geom is there,
the sizes are right, and everything falls to z=0.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.plugin import Plugin
from roqsim.plugins.heightfield import HeightfieldPlugin

SIZE = 20.0


def _engine(**config):
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [{"heightfield": {"size": [SIZE, SIZE], **config}}],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine


def _plugin(engine) -> HeightfieldPlugin:
    return next(p for p in engine.plugins if isinstance(p, HeightfieldPlugin))


# -- the ground it builds ---------------------------------------------------------------------


def test_it_builds_a_height_field_and_the_world_underneath_it_is_skipped():
    """`provides_world`: a terrain that let a floor be built would have a plane through its hills."""
    engine = _engine(height=1.5, resolution=48)
    model = engine.ctx.model
    assert model.nhfield == 1
    assert HeightfieldPlugin.provides_world is True
    # No `floor` from the built-in world definition -- this plugin is the ground.
    names = [
        __import__("mujoco").mj_id2name(model, __import__("mujoco").mjtObj.mjOBJ_GEOM, g)
        for g in range(model.ngeom)
    ]
    assert "floor" not in names
    assert any(n and n.endswith("_ground") for n in names)


def test_the_extent_is_metres_and_the_relief_is_metres():
    engine = _engine(height=2.5, resolution=32)
    size = engine.ctx.model.hfield_size[0]
    # MuJoCo stores HALF extents; a world that asked for 20 m must not get 40.
    assert size[0] == pytest.approx(SIZE / 2) and size[1] == pytest.approx(SIZE / 2)
    assert size[2] == pytest.approx(2.5)


def test_generated_terrain_is_reproducible_and_seed_dependent():
    """Reproducible from the run's own seed is the difference between terrain and noise: two cells
    of a campaign that meant to share a world must not be comparing different hills."""
    first = _plugin(_engine(seed=7, resolution=24)).elevation
    again = _plugin(_engine(seed=7, resolution=24)).elevation
    other = _plugin(_engine(seed=8, resolution=24)).elevation
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_roughness_and_octaves_shape_it_without_changing_its_range():
    smooth = _plugin(_engine(seed=1, octaves=1, resolution=32)).elevation
    rough = _plugin(_engine(seed=1, octaves=5, roughness=0.9, resolution=32)).elevation
    # Both are normalised, so "rougher" is about local variation, not about being taller.
    assert smooth.min() == pytest.approx(0.0) and smooth.max() == pytest.approx(1.0)
    assert rough.min() == pytest.approx(0.0) and rough.max() == pytest.approx(1.0)
    assert np.abs(np.diff(rough, axis=0)).mean() > np.abs(np.diff(smooth, axis=0)).mean()


# -- something stands on it -------------------------------------------------------------------


def _drop_ball(engine, x: float, y: float) -> float:
    """Settle a ball dropped at (x, y) and return the height its centre came to rest at."""
    import mujoco

    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    adr = model.jnt_qposadr[model.body_jntadr[bid]]
    data.qpos[adr : adr + 3] = [x, y, 6.0]
    data.qvel[:] = 0.0
    for _ in range(3000):
        engine.step()
    return float(data.xpos[bid][2])


class _BallScene(Plugin):
    """One free ball, so the terrain has something to hold up."""

    RADIUS = 0.25

    def build(self, spec, ctx) -> None:
        import mujoco

        ball = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 6.0])
        ball.add_freejoint()
        ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.RADIUS, 0, 0], mass=1.0)


def test_a_ball_rests_on_the_terrain_not_at_zero():
    """The physical check: a compiled-but-uncollided height field passes every other assertion."""
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {"heightfield": {"size": [SIZE, SIZE], "height": 2.0, "resolution": 64, "seed": 3}},
                {f"{__name__}:_BallScene": {}},
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    plugin = _plugin(engine)

    grid = plugin.elevation
    rows, cols = grid.shape
    # The highest and lowest samples of this terrain, as world coordinates.
    high_r, high_c = np.unravel_index(np.argmax(grid), grid.shape)
    low_r, low_c = np.unravel_index(np.argmin(grid), grid.shape)

    def to_world(row, col):
        # The inverse of height_at's mapping, so the two cannot disagree about the flip.
        return (col / (cols - 1) - 0.5) * SIZE, (row / (rows - 1) - 0.5) * SIZE

    hx, hy = to_world(high_r, high_c)
    lx, ly = to_world(low_r, low_c)
    on_hill = _drop_ball(engine, hx * 0.9, hy * 0.9)
    in_valley = _drop_ball(engine, lx * 0.9, ly * 0.9)
    assert on_hill > in_valley + 0.3, "the hill must hold the ball higher than the valley does"
    assert in_valley > 0.0, "and the ball rests on the ground, not through it"


def test_height_at_agrees_with_where_the_ball_lands():
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {"heightfield": {"size": [SIZE, SIZE], "height": 2.0, "resolution": 64, "seed": 3}},
                {f"{__name__}:_BallScene": {}},
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    plugin = _plugin(engine)
    x, y = 3.0, -2.0
    rest = _drop_ball(engine, x, y)
    # The ball's centre sits a radius above the surface; the sample is nearest-neighbour and the
    # ground is faceted, so this is a metre-scale agreement, not a millimetre one.
    assert rest == pytest.approx(plugin.height_at(x, y) + _BallScene.RADIUS, abs=0.4)


# -- sources ------------------------------------------------------------------------------------


def test_an_array_on_disk_is_read_as_it_is_written(tmp_path):
    ramp = np.tile(np.linspace(0.0, 1.0, 16), (16, 1))
    path = tmp_path / "dem.npy"
    np.save(path, ramp)
    plugin = _plugin(_engine(source=str(path), height=4.0))
    assert plugin.elevation.shape == (16, 16)
    # A ramp rising along +x must still rise along +x in the world.
    assert plugin.height_at(-SIZE / 2 + 0.1, 0.0) < plugin.height_at(SIZE / 2 - 0.1, 0.0)


def test_a_greyscale_image_is_read_at_its_own_bit_depth(tmp_path):
    from PIL import Image

    # Every sample distinct, and finer than 8 bits can express: a DEM read as 8-bit would collapse
    # these 4096 levels to at most 256, which on a 200 m hill is 80 cm steps a legged controller
    # feels as a staircase.
    dem = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16
    path = tmp_path / "dem.png"
    Image.fromarray(dem).save(path)
    plugin = _plugin(_engine(source=str(path), height=10.0))
    assert len(np.unique(plugin.elevation)) > 256


def test_flat_data_is_refused_rather_than_silently_building_a_plane(tmp_path):
    path = tmp_path / "flat.npy"
    np.save(path, np.ones((8, 8)))
    with pytest.raises(RuntimeError, match="perfectly flat"):
        _engine(source=str(path))


def test_a_missing_source_names_the_path(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        _engine(source=str(tmp_path / "nope.npy"))


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"resolution": 1}, "'resolution' must be >= 2"),
        ({"octaves": 0}, "'octaves' must be >= 1"),
        ({"roughness": 2.0}, "'roughness' must be in [0, 1]"),
        ({"size": [1.0]}, "'size' must be [x, y]"),
        ({"height": -1}, "'height' must be >= 0"),
        ({"friction": [1.0]}, "'friction' must be MuJoCo's three"),
        ({"source": "dem.tiff.gz"}, "must be 'generated' or a path ending in"),
    ],
)
def test_config_errors_are_reported_by_name(config, expected):
    errors = HeightfieldPlugin(config, label="terrain").validate_config(config)
    assert any(expected in e for e in errors), errors


def test_the_geotiff_advice_names_the_tool_that_owns_reprojection():
    errors = HeightfieldPlugin({"source": "dem.hgt"}, label="terrain").validate_config(
        {"source": "dem.hgt"}
    )
    assert any("gdal_translate" in e for e in errors)
