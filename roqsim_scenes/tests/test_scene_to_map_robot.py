"""A world-derived map is the ENVIRONMENT: the robot must not rasterise into it.

``scene-to-map --world`` compiles the world through the plugin pipeline, which is the only way to see
obstacles a world adds on top of an imported scene. But it also puts the robot at its spawn pose, and a
robot baked into the map is permanent occupancy exactly where every trial begins: AMCL localises against
a phantom obstacle at the start and a planner refuses to leave it.

The failure is quiet, which is why it is worth a test. ``--free-from`` is documented as "normally the
robot's start", so the flood fill gets seeded *on* the robot blob and the symptom is a slightly smaller
free area rather than an error. It was found only because an independent clearance check on the start
pose failed.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Entity, SimContext
from roqsim_scenes.cli.scene_to_map import _robot_geoms


def _model_with(bodies: dict[str, str | None]) -> mujoco.MjModel:
    """Compile a model where each key is a body carrying one geom; the value is its parent body."""
    spec = mujoco.MjSpec()
    made: dict[str, object] = {}
    for name, parent in bodies.items():
        attach_to = made[parent] if parent else spec.worldbody
        body = attach_to.add_body()
        body.name = name
        g = body.add_geom()
        g.name = f"{name}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.1, 0.1, 0.1]
        made[name] = body
    return spec.compile()


def _ctx(model, entities: list[Entity]) -> SimContext:
    ctx = SimContext(config={})
    ctx.model = model
    for e in entities:
        ctx.entities.add(e)
    return ctx


def _geom(model, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def test_robot_subtree_is_excluded_wheels_and_all():
    """The SUBTREE, not just the base: a map with four wheel-sized dots at the start is no better."""
    model = _model_with(
        {"base_link": None, "wheel_l": "base_link", "wheel_r": "base_link", "shelf": None}
    )
    ctx = _ctx(model, [Entity(name="robot", kind="robot", body="base_link")])
    skip = _robot_geoms(model, ctx)
    assert _geom(model, "base_link_geom") in skip
    assert _geom(model, "wheel_l_geom") in skip
    assert _geom(model, "wheel_r_geom") in skip
    assert _geom(model, "shelf_geom") not in skip, "scenery was dropped from the map"


def test_scenery_entities_are_kept():
    """Only kind='robot' is excluded. A prop or a movable object IS environment and must be mapped."""
    model = _model_with({"base_link": None, "crate": None, "pallet": None})
    ctx = _ctx(
        model,
        [
            Entity(name="robot", kind="robot", body="base_link"),
            Entity(name="crate", kind="object", body="crate"),
            Entity(name="pallet", kind="prop", body="pallet"),
        ],
    )
    skip = _robot_geoms(model, ctx)
    assert skip == {_geom(model, "base_link_geom")}


def test_a_world_with_no_robot_is_untouched():
    """An environment-only world must map exactly as before this change."""
    model = _model_with({"shelf": None, "crate": None})
    assert _robot_geoms(model, _ctx(model, [])) == set()


def test_two_robots_are_both_excluded():
    """A multi-robot world would otherwise seed one robot's map with the other's body."""
    model = _model_with({"r1_base": None, "r2_base": None, "wall": None})
    ctx = _ctx(
        model,
        [
            Entity(name="r1", kind="robot", body="r1_base"),
            Entity(name="r2", kind="robot", body="r2_base"),
        ],
    )
    skip = _robot_geoms(model, ctx)
    assert skip == {_geom(model, "r1_base_geom"), _geom(model, "r2_base_geom")}


def test_a_robot_entity_naming_a_missing_body_is_ignored_not_fatal():
    """A stale entity must not abort a map build; it simply excludes nothing."""
    model = _model_with({"shelf": None})
    ctx = _ctx(model, [Entity(name="robot", kind="robot", body="does_not_exist")])
    assert _robot_geoms(model, ctx) == set()


def test_geom_ids_are_plain_ints_so_the_skip_set_actually_matches():
    """The caller tests `g in skip` over range(ngeom); numpy ints would compare equal but this pins it."""
    model = _model_with({"base_link": None, "shelf": None})
    ctx = _ctx(model, [Entity(name="robot", kind="robot", body="base_link")])
    skip = _robot_geoms(model, ctx)
    assert all(isinstance(g, (int, np.integer)) for g in skip)
    assert any(g == _geom(model, "base_link_geom") for g in range(model.ngeom) if g in skip)
