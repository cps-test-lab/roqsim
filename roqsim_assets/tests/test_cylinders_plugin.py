"""`cylinders`: a population of parametric round props declared as one list.

The geometry is `cylinder`'s and this plugin delegates to it, so what is pinned here is what the
list exists for: **how many** and **which radii** are the contents of one config value, so a
campaign can override the whole population. Appending a plugin entry is something `apply_overrides`
cannot do by design, and a heterogeneous set of diameters is something uniform scaling of a modelled
asset cannot express at all.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim_assets.plugins.cylinders import CylindersPlugin

_TWO = [
    {"pos": [0.2, 0.1], "radius": 0.035, "height": 0.15},
    {"pos": [0.4, -0.2], "radius": 0.045, "height": 0.15},
]


def _build(*, name="cylinders", **cfg):
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20, 20, 0.05]
    ctx = SimContext(config={})
    plugin = CylindersPlugin(cfg, label=name)
    plugin.build(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx.model, ctx.data = model, data
    plugin.configure(ctx)
    return model, plugin, ctx


def _body_names(model, needle):
    return [model.body(i).name for i in range(model.nbody) if needle in model.body(i).name]


def _geom(model, suffix="cylinder"):
    return next(
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(suffix)
    )


def test_every_instance_becomes_a_body():
    model, _, _ = _build(name="clutter", instances=_TWO)
    assert len(_body_names(model, "clutter")) == 2


def test_instances_get_distinct_names():
    """Not cosmetic: MuJoCo names must be unique, so a collision fails model compilation."""
    model, _, _ = _build(name="clutter", instances=_TWO)
    names = _body_names(model, "clutter")
    assert len(set(names)) == len(names)


def test_an_instance_may_name_itself():
    model, _, _ = _build(
        name="clutter",
        instances=[{"pos": [0.1, 0.1], "radius": 0.03, "height": 0.1, "name": "target"}],
    )
    assert _body_names(model, "target")


def test_an_empty_population_is_legal():
    """A sweep that includes "no clutter" as a level needs this to build, not to fail."""
    model, _, _ = _build(name="clutter", instances=[])
    assert _body_names(model, "clutter") == []


def test_geometry_is_the_cylinder_plugins():
    """Delegation, not reimplementation: true radius, FULL height, and pos [x,y] stands it up."""
    model, _, _ = _build(
        name="clutter", instances=[{"pos": [0.0, 0.0], "radius": 0.04, "height": 0.15}]
    )
    gid = _geom(model)
    assert list(model.geom_size[gid][:2]) == pytest.approx([0.04, 0.075])
    assert model.body(model.geom_bodyid[gid]).pos[2] == pytest.approx(0.075)


def test_radii_may_differ_within_one_population():
    """The property that forces a list of parametric entries rather than one scaled asset."""
    model, _, _ = _build(name="clutter", instances=_TWO)
    radii = sorted(
        model.geom_size[g][0]
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("cylinder")
    )
    assert radii == pytest.approx([0.035, 0.045])


# -- free + mass: what turns a post into a workpiece ---------------------------------------------


def test_free_instances_get_a_base_joint_registered():
    """`SetEntityState` refuses an entity without a free `base_joint`, so this is the contract."""
    _, _, ctx = _build(
        name="clutter",
        instances=[{"pos": [0.1, 0.0], "radius": 0.03, "height": 0.15, "free": True}],
    )
    entity = ctx.entities.get("clutter_0")
    assert entity.kind == "object"
    assert entity.meta["base_joint"].endswith("free")


def test_a_welded_instance_has_no_free_joint():
    model, _, ctx = _build(name="clutter", instances=[_TWO[0]])
    assert model.njnt == 0
    assert ctx.entities.get("clutter_0").kind == "prop"


def test_mass_overrides_the_default_density():
    """Unset, a 0.15 m x 0.035 m cylinder weighs ~0.58 kg at MuJoCo's 1000 kg/m^3 -- far too much
    for anything hollow, and grasp stability is a function of it."""
    heavy, _, _ = _build(
        name="clutter", instances=[{"pos": [0.0, 0.0], "radius": 0.035, "height": 0.15}]
    )
    light, _, _ = _build(
        name="clutter",
        instances=[{"pos": [0.0, 0.0], "radius": 0.035, "height": 0.15, "mass": 0.3}],
    )
    assert light.body_mass[light.geom_bodyid[_geom(light)]] == pytest.approx(0.3)
    assert heavy.body_mass[heavy.geom_bodyid[_geom(heavy)]] > 0.5


def test_a_free_instance_is_reseated_on_reset():
    """A per-trial layout must be a reset, not a reload: a trial cannot inherit the last one."""
    model, plugin, ctx = _build(
        name="clutter",
        instances=[{"pos": [0.2, 0.1, 0.5], "radius": 0.03, "height": 0.15, "free": True}],
    )
    ctx.data.qpos[0:3] = [9.0, 9.0, 9.0]
    ctx.data.qvel[0:6] = 1.0
    plugin.on_reset(ctx)
    assert list(ctx.data.qpos[0:3]) == pytest.approx([0.2, 0.1, 0.5])
    assert list(ctx.data.qvel[0:6]) == pytest.approx([0.0] * 6)


# -- the reason this plugin exists ---------------------------------------------------------------


def test_an_override_changes_the_population_count():
    """The whole point: a campaign varies the layout without editing the world's structure."""
    world = {"sim": {}, "components": [{"cylinders": {"instances": _TWO}, "name": "clutter"}]}
    twenty = [
        {"pos": [0.02 * i, 0.0], "radius": 0.03, "height": 0.15, "free": True} for i in range(20)
    ]

    cfg = load_config_from_dict(world, overrides={"components": {"clutter": {"instances": twenty}}})

    cylinders = next(s for s in cfg.plugins if s.ref == "cylinders")
    assert len(cylinders.config["instances"]) == 20


def test_validation_names_the_offending_instance():
    """ "'radius' is required" is not actionable when the world declares twenty cylinders."""
    plugin = CylindersPlugin({"instances": _TWO}, label="clutter")
    errors = plugin.validate_config({"instances": [_TWO[0], {"pos": [0.1, 0.1]}]})
    assert errors and all(e.startswith("instances[1]:") for e in errors)


def test_a_negative_mass_is_refused():
    plugin = CylindersPlugin({"instances": []}, label="clutter")
    errors = plugin.validate_config({"instances": [{**_TWO[0], "mass": -1.0}]})
    assert any("'mass' must be positive" in e for e in errors)


def test_instances_is_required():
    plugin = CylindersPlugin({}, label="cylinders")
    assert plugin.validate_config({}) == ["'instances' is required (a list of cylinder configs)"]


def test_instances_must_be_a_list():
    plugin = CylindersPlugin({}, label="cylinders")
    assert "must be a list" in plugin.validate_config({"instances": {"pos": [0, 0]}})[0]
