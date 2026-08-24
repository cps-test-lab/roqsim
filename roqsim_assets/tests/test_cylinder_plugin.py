"""`cylinder`: the parametric cylindrical obstacle.

The contract worth pinning is the same pair `box` pins -- the size convention (a true radius and a
FULL height, against MuJoCo's `[radius, half-height]`) and the floor-standing default -- plus the one
property that is the whole reason this plugin exists rather than a squared-off box: two tangent
cylinders on a grid pitch leave a diagonal gap where two boxes would not.
"""

from __future__ import annotations

import math

import mujoco
import pytest

from roqsim.context import SimContext
from roqsim_assets.plugins.cylinder import CylinderPlugin


def _build(*, name=None, **cfg):
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 0.05]
    ctx = SimContext(config={})
    plugin = CylinderPlugin(cfg, label=name)
    plugin.build(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx.model, ctx.data = model, data
    plugin.configure(ctx)
    return model, data, plugin, ctx


def _cyl_gid(model, suffix="cylinder"):
    for gid in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").endswith(suffix):
            return gid
    raise AssertionError("no cylinder geom in the compiled model")


def test_height_is_full_not_half_and_radius_is_a_radius():
    model, _, _, _ = _build(pos=[1.0, 2.0], radius=0.075, height=0.5)
    gid = _cyl_gid(model)
    assert float(model.geom_size[gid][0]) == pytest.approx(0.075)
    assert float(model.geom_size[gid][1]) == pytest.approx(0.25)


def test_two_element_pos_stands_the_cylinder_on_the_floor():
    model, data, _, _ = _build(pos=[1.0, 2.0], radius=0.075, height=0.5)
    gid = _cyl_gid(model)
    z = float(data.geom_xpos[gid][2])
    assert z == pytest.approx(0.25)  # centre at half height => base at z = 0
    assert z - float(model.geom_size[gid][1]) == pytest.approx(0.0, abs=1e-9)


def test_three_element_pos_places_the_centre():
    model, data, _, _ = _build(pos=[0.0, 0.0, 1.5], radius=0.1, height=0.4)
    assert float(data.geom_xpos[_cyl_gid(model)][2]) == pytest.approx(1.5)


def _geom_rgba(model, gid):
    """Colour lives on the geom's MATERIAL, not on `geom_rgba` (which stays at its 0.5 default
    whenever a material is assigned) -- the trap that made the first version of these tests fail."""
    matid = int(model.geom_matid[gid])
    assert matid >= 0, "geom has no material"
    return list(model.mat_rgba[matid])


def test_defaults_are_grey_and_colliding():
    model, _, plugin, _ = _build(pos=[0.0, 0.0], radius=0.1, height=1.0)
    gid = _cyl_gid(model)
    assert plugin.collide is True
    assert int(model.geom_contype[gid]) != 0
    assert _geom_rgba(model, gid) == pytest.approx([0.86, 0.86, 0.83, 1.0])


def test_registers_an_entity_with_its_geometry():
    _, _, _, ctx = _build(pos=[1.0, 2.0], radius=0.075, height=0.5, name="obstacle_3")
    entity = ctx.entities.get("obstacle_3")
    assert entity is not None and entity.kind == "prop"
    assert entity.meta["radius"] == pytest.approx(0.075)
    assert entity.meta["height"] == pytest.approx(0.5)
    assert entity.meta["pos"] == pytest.approx([1.0, 2.0, 0.25])


def test_collide_false_keeps_the_geom_but_disables_contact():
    model, _, _, _ = _build(pos=[0.0, 0.0], radius=0.1, height=1.0, collide=False)
    gid = _cyl_gid(model)
    assert int(model.geom_contype[gid]) == 0
    assert int(model.geom_conaffinity[gid]) == 0


def test_color_and_friction_reach_the_geom():
    model, _, _, _ = _build(
        pos=[0.0, 0.0], radius=0.1, height=1.0, color=[0.648, 0.192, 0.192], friction=0.4
    )
    gid = _cyl_gid(model)
    assert _geom_rgba(model, gid) == pytest.approx([0.648, 0.192, 0.192, 1.0])
    assert float(model.geom_friction[gid][0]) == pytest.approx(0.4)


def test_validate_config_requires_pos_radius_height_and_rejects_nonpositive():
    plugin = CylinderPlugin({})
    errors = plugin.validate_config({})
    assert any("'pos' is required" in e for e in errors)
    assert any("'radius' is required" in e for e in errors)
    assert any("'height' is required" in e for e in errors)

    errors = plugin.validate_config({"pos": [0, 0], "radius": 0.0, "height": -1.0})
    assert any("'radius' must be positive" in e for e in errors)
    assert any("'height' must be positive" in e for e in errors)


def test_several_cylinders_coexist_under_distinct_prefixes():
    spec = mujoco.MjSpec()
    ctx = SimContext(config={})
    for i in range(3):
        CylinderPlugin(
            {"prefix": f"o{i}_", "pos": [float(i), 0.0], "radius": 0.075, "height": 0.5}
        ).build(spec, ctx)
    model = spec.compile()
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) for gid in range(model.ngeom)}
    assert {"o0_cylinder", "o1_cylinder", "o2_cylinder"} <= names


def test_diagonal_neighbours_leave_a_gap_a_box_would_not():
    """The reason this plugin is not `box` with rounded corners.

    Two cells of pitch p, occupied diagonally. Boxes of side p touch corner-to-corner and seal the
    diagonal; cylinders of radius p/2 sit at distance p*sqrt(2) apart and leave p*(sqrt(2) - 1) of
    clear space. For a benchmark whose difficulty IS clearance, that gap is the experiment.
    """
    pitch = 0.15
    radius = pitch / 2
    spec = mujoco.MjSpec()
    ctx = SimContext(config={})
    for i, (x, y) in enumerate([(0.0, 0.0), (pitch, pitch)]):
        CylinderPlugin({"prefix": f"c{i}_", "pos": [x, y], "radius": radius, "height": 0.5}).build(
            spec, ctx
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    a = data.geom_xpos[_cyl_gid(model, "c0_cylinder")][:2]
    b = data.geom_xpos[_cyl_gid(model, "c1_cylinder")][:2]
    centre_dist = math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
    surface_gap = centre_dist - 2 * radius

    assert centre_dist == pytest.approx(pitch * math.sqrt(2))
    assert surface_gap == pytest.approx(pitch * (math.sqrt(2) - 1))
    assert surface_gap > 0.0
