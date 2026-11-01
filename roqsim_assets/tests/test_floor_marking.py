"""The floor marking: one colour, solid or dashed, or warning stripes.

What matters about it is that it is paint: it covers exactly the area it was given -- in one colour,
in dashes with gaps between, or in two colours that share it about evenly -- and nothing collides
with it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.raycast import VISIBLE_GROUPS
from roqsim_assets.plugins.floor_marking import (
    FloorMarkingPlugin,
    _area,
    dash_rectangles,
    marked_rectangles,
    stripe_polygons,
)


def _built(tmp_path, **marking):
    cfg = {
        "sim": {},
        "components": [{"floor_marking": {"prefix": "m_", **marking}, "name": "spot"}],
    }
    engine = Engine(load_config_from_dict(cfg, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    return engine


def _painted(length, width, band, stripe=0.1, angle=math.radians(45.0)):
    first, second = stripe_polygons(marked_rectangles(length, width, band), stripe, angle)
    return sum(map(_area, first)), sum(map(_area, second))


def test_a_filled_hazard_marking_covers_its_rectangle_half_in_each_colour():
    first, second = _painted(2.0, 1.0, 0.0)
    assert first + second == pytest.approx(2.0)
    assert first == pytest.approx(second, rel=0.1)


def test_a_hazard_outline_covers_the_band_and_leaves_the_inside_bare():
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
        pattern="hazard",
        pose={"position": {"x": 3.0, "y": -2.0, "z": 0.0}, "orientation": {"yaw": math.pi / 2}},
        length=1.5,
        width=1.1,
    )
    model = engine.ctx.model
    geoms = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("m_paint_")
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
        assert world[:, 2].max() == pytest.approx(0.0005, abs=1e-6)
    assert engine.ctx.entities.get("spot").body == "m_floor_marking"


def test_a_ray_aimed_at_the_floor_returns_from_the_paint(tmp_path):
    """Paint has no contact, but it is in the groups a scanner's rays see: a downward ray returns
    from its top, the paint's thickness above the floor."""
    engine = _built(tmp_path, pattern="hazard", band=0.0)
    hit = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        engine.ctx.model,
        engine.ctx.data,
        np.array([0.02, 0.0, 1.0]),  # inside a stripe: the centre lies on the edge between two
        np.array([0.0, 0.0, -1.0]),
        VISIBLE_GROUPS,
        1,
        -1,
        hit,
    )
    name = mujoco.mj_id2name(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0]))
    assert name.startswith("m_paint_")
    assert distance == pytest.approx(0.9995, abs=1e-6)


@pytest.mark.parametrize(
    "config, word",
    [
        ({"length": 0}, "'length'"),
        ({"band": -0.1}, "'band'"),
        ({"pattern": "hazard", "angle": 0}, "'angle'"),
        ({"pattern": "hazard", "colors": [[1, 1, 0]]}, "'colors'"),
        ({"pattern": "dashed", "dash": 0}, "'dash'"),
        ({"color": [1, 1]}, "'color'"),
        ({"pattern": "stripes"}, "'pattern'"),
        ({"lenght": 1.0}, "unknown key"),
        ({"yaw": 1.0}, "'yaw'"),
        ({"pose": {"orientation": {"roll": 0.3}}}, "tilts"),
    ],
)
def test_a_wrong_value_is_named(config, word):
    assert any(word in e for e in FloorMarkingPlugin(config).validate_config(config))


def test_an_outline_is_five_centimetres_wide_unless_told():
    assert FloorMarkingPlugin({}).band == 0.05


def test_hazard_stripes_are_in_proportion_to_the_band_unless_told():
    assert FloorMarkingPlugin({"pattern": "hazard"}).stripe == pytest.approx(0.035)
    assert FloorMarkingPlugin({"pattern": "hazard", "band": 0.1}).stripe == pytest.approx(0.07)
    assert FloorMarkingPlugin({"pattern": "hazard", "band": 0}).stripe == pytest.approx(0.1)
    assert FloorMarkingPlugin(
        {"pattern": "hazard", "band": 0.1, "stripe": 0.2}
    ).stripe == pytest.approx(0.2)


def test_the_coat_covers_unless_a_colour_states_an_alpha(tmp_path):
    assert [c[3] for c in FloorMarkingPlugin({"pattern": "hazard"}).colors] == [1.0, 1.0]
    worn = FloorMarkingPlugin({"pattern": "hazard", "colors": [[1, 1, 0, 0.6], [0, 0, 0]]})
    assert [c[3] for c in worn.colors] == [0.6, 1.0]


# -- one colour: solid and dashed -----------------------------------------------------------------


def _paint_vertices(engine):
    """Every painted geom's name, colour and vertices in the world."""
    model, data = engine.ctx.model, engine.ctx.data
    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.startswith("m_paint_"):
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (
            model.mesh_vert[start : start + count] @ data.geom_xmat[g].reshape(3, 3).T
            + data.geom_xpos[g]
        )
        out.append((g, model.mat_rgba[model.geom_matid[g]], world))
    return out


def _covered(plugin):
    return sum(_area(p) for polygons in plugin.paint_polygons() for p in polygons)


def test_a_marking_is_a_solid_yellow_outline_unless_told():
    plugin = FloorMarkingPlugin({})
    assert plugin.pattern == "solid"
    assert plugin.colors == [[0.90, 0.66, 0.06, 1.0]]
    assert _covered(plugin) == pytest.approx(1.4 * 1.0 - 1.3 * 0.9)


def test_a_solid_line_is_one_strip_of_paint_in_its_colour_that_nothing_touches(tmp_path):
    engine = _built(
        tmp_path,
        pose={"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
        length=10.0,
        width=0.1,
        band=0.0,
        color=[0.1, 0.4, 0.8],
    )
    (painted,) = _paint_vertices(engine)
    g, rgba, world = painted
    assert engine.ctx.model.geom_contype[g] == 0 and engine.ctx.model.geom_conaffinity[g] == 0
    assert list(rgba) == pytest.approx([0.1, 0.4, 0.8, 1.0])
    assert world[:, 0].min() == pytest.approx(-4.0, abs=1e-6)
    assert world[:, 0].max() == pytest.approx(6.0, abs=1e-6)
    assert world[:, 1].min() == pytest.approx(1.95, abs=1e-6)
    assert world[:, 1].max() == pytest.approx(2.05, abs=1e-6)
    assert world[:, 2].max() == pytest.approx(0.0005, abs=1e-6)
    assert engine.ctx.entities.get("spot").body == "m_floor_marking"


def test_a_solid_outline_paints_the_band_and_leaves_the_inside_bare():
    plugin = FloorMarkingPlugin({"length": 4.0, "width": 2.0, "band": 0.1})
    assert len(plugin.paint_polygons()[0]) == 4
    assert _covered(plugin) == pytest.approx(4.0 * 2.0 - 3.8 * 1.8)
    filled = FloorMarkingPlugin({"length": 4.0, "width": 2.0, "band": 0.0})
    assert len(filled.paint_polygons()[0]) == 1
    assert _covered(filled) == pytest.approx(8.0)


def test_dashes_cut_a_side_into_pieces_with_gaps():
    pieces = dash_rectangles([(0.0, 0.0, 3.0, 0.1)], 0.5)
    assert [p[0] for p in pieces] == pytest.approx([0.0, 1.0, 2.0])
    assert all(p[2] - p[0] == pytest.approx(0.5) for p in pieces)


def test_the_last_dash_is_cut_short_where_the_side_ends():
    pieces = dash_rectangles([(0.0, -1.3, 0.1, 1.3)], 0.5)
    assert [p[3] - p[1] for p in pieces] == pytest.approx([0.5, 0.5, 0.5])
    assert pieces[-1][3] == pytest.approx(1.2)
    pieces = dash_rectangles([(0.0, 0.0, 2.3, 0.1)], 0.5)
    assert [p[2] - p[0] for p in pieces] == pytest.approx([0.5, 0.5, 0.3])


def test_a_dashed_outline_is_dashes_on_all_four_sides(tmp_path):
    marking = {"length": 4.0, "width": 2.0, "band": 0.1, "pattern": "dashed", "dash": 0.5}
    # long sides: 4 dashes each over 4 m; short sides, 1.8 m between them: 2 dashes each
    assert len(FloorMarkingPlugin(marking).paint_polygons()[0]) == 4 + 4 + 2 + 2
    assert _covered(FloorMarkingPlugin(marking)) == pytest.approx(
        2 * 4 * 0.5 * 0.1 + 2 * 2 * 0.5 * 0.1
    )
    (painted,) = _paint_vertices(_built(tmp_path, **marking))
    assert painted[2][:, 0].min() == pytest.approx(-2.0, abs=1e-6)


def test_a_dashed_line_covers_half_its_length():
    plugin = FloorMarkingPlugin(
        {"length": 6.0, "width": 0.1, "band": 0.0, "pattern": "dashed", "dash": 0.5}
    )
    assert len(plugin.paint_polygons()[0]) == 6
    assert _covered(plugin) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "config, key",
    [
        ({"dash": 0.3}, "dash"),
        ({"stripe": 0.1}, "stripe"),
        ({"pattern": "dashed", "colors": [[1, 1, 0], [0, 0, 0]]}, "colors"),
        ({"pattern": "hazard", "color": [1, 1, 0]}, "color"),
        ({"pattern": "hazard", "dash": 0.5}, "dash"),
        ({"pattern": "solid", "angle": 30.0}, "angle"),
    ],
)
def test_a_key_of_another_pattern_is_refused(config, key):
    errors = FloorMarkingPlugin(config).validate_config(config)
    assert any(f"'{key}' is read by pattern" in e for e in errors), errors


def test_a_key_of_another_pattern_fails_the_world(tmp_path):
    with pytest.raises(Exception, match="'dash' is read by pattern dashed only"):
        _built(tmp_path, pattern="solid", dash=0.5)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"pattern": "solid", "color": [0.2, 0.6, 0.2, 0.8]},
        {"pattern": "dashed", "color": [1, 1, 1], "dash": 0.25},
        {"pattern": "hazard", "stripe": 0.05, "angle": 60, "colors": [[1, 1, 0], [0, 0, 0]]},
    ],
)
def test_a_config_of_one_pattern_is_accepted(config):
    assert FloorMarkingPlugin(config).validate_config(config) == []


@pytest.mark.parametrize("pattern", ["Hazard", None, ["hazard"]])
def test_a_pattern_that_is_none_of_the_three_is_refused_naming_them(pattern):
    errors = FloorMarkingPlugin({"pattern": pattern}).validate_config({"pattern": pattern})
    assert any("'pattern' is one of solid, dashed, hazard" in e for e in errors), errors
