"""`boxes`: a population of parametric obstacles declared as one list.

What is worth pinning is not the geometry -- that is `box`'s, and this plugin delegates to it -- but
the property the list exists for: **how many** is the length of a config value, so a campaign can
override it. Appending a plugin entry is something `apply_overrides` cannot do by design.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import apply_overrides
from roqsim.context import SimContext
from roqsim_assets.plugins.boxes import BoxesPlugin

_TWO = [
    {"pos": [2.0, 1.0], "size": [0.5, 0.5, 1.0]},
    {"pos": [4.0, -1.0], "size": [0.8, 0.8, 1.0], "yaw": 0.4},
]


def _build(*, name="boxes", **cfg):
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20, 20, 0.05]
    ctx = SimContext(config={})
    plugin = BoxesPlugin(cfg, label=name)
    plugin.build(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx.model, ctx.data = model, data
    plugin.configure(ctx)
    return model, plugin


def _body_names(model, needle):
    return [model.body(i).name for i in range(model.nbody) if needle in model.body(i).name]


def test_every_instance_becomes_a_body():
    model, _ = _build(name="obstacles", instances=_TWO)
    assert len(_body_names(model, "obstacles")) == 2


def test_instances_get_distinct_names():
    """Not cosmetic: MuJoCo names must be unique, so a collision fails model compilation."""
    model, _ = _build(name="obstacles", instances=_TWO)
    names = _body_names(model, "obstacles")
    assert len(set(names)) == len(names)


def test_an_instance_may_name_itself():
    model, _ = _build(name="obstacles", instances=[{"pos": [1.0, 1.0], "name": "pillar"}])
    assert _body_names(model, "pillar")


def test_an_empty_population_is_legal():
    """A sweep that includes "no obstacles" as a level needs this to build, not to fail."""
    model, _ = _build(name="obstacles", instances=[])
    assert _body_names(model, "obstacles") == []


def test_geometry_is_the_box_plugins():
    """Delegation, not reimplementation: full extents, and two-element pos sits on the floor."""
    model, _ = _build(name="obstacles", instances=[{"pos": [0.0, 0.0], "size": [0.4, 0.6, 1.0]}])
    gid = next(
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("box")
    )
    assert list(model.geom_size[gid][:3]) == pytest.approx([0.2, 0.3, 0.5])
    assert model.body(model.geom_bodyid[gid]).pos[2] == pytest.approx(0.5)


# -- the reason this plugin exists ---------------------------------------------------------------


def test_an_override_changes_the_population_count():
    """The whole point: a campaign varies "how many" without editing the world's structure.

    `apply_overrides` deep-merges into a plugin resolved by name and refuses one matching no
    plugin, so it can replace a list value but could never append a second `box:` entry. With one
    entry per obstacle, obstacle count was a structural edit and therefore not a factor at all.
    """
    world = {"plugins": [{"boxes": {"name": "obstacles", "instances": _TWO}}]}
    five = [{"pos": [float(i), 0.0], "size": [0.3, 0.3, 0.3]} for i in range(5)]

    merged = apply_overrides(world, {"plugins": {"boxes": {"instances": five}}})

    assert len(merged["components"][0]["boxes"]["instances"]) == 5


def test_validation_names_the_offending_instance():
    """ "size must have three elements" is not actionable when the world declares twelve boxes."""
    plugin = BoxesPlugin({"instances": _TWO}, label="obstacles")
    errors = plugin.validate_config(
        {"name": "obstacles", "instances": [_TWO[0], {"pos": [1.0, 1.0], "size": "big"}]}
    )
    assert errors and all(e.startswith("instances[1]:") for e in errors)


def test_instances_is_required():
    plugin = BoxesPlugin({}, label="boxes")
    assert plugin.validate_config({}) == ["'instances' is required (a list of box configs)"]


def test_instances_must_be_a_list():
    plugin = BoxesPlugin({}, label="boxes")
    assert "must be a list" in plugin.validate_config({"instances": {"pos": [0, 0]}})[0]
