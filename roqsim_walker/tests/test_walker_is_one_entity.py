# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A walker is one entity to every reader of its geometry, limbs and all.

Its limbs are mocap bodies of the world, not children of its pelvis, so the walker lists them in
``Entity.bodies``. A reader that resolved the entity to its ``body`` and that body's subtree would
see the pelvis alone: an absent walker with visible arms, a contact meter blind to a kick, an
``ignore`` that a robot still stops for. Each test here names a limb the reader must take.
"""

import mujoco
import pytest
from test_clearance_by_entity import _engine, _world

from roqsim.presence import ABSENT_GEOM_GROUP, set_present

#: Limbs of each part of the skeleton: an arm, a leg, a foot.
LIMBS = ("pedestrian/elbow_r", "pedestrian/knee_l", "pedestrian/ankle_r")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    engine = _engine(_world(tmp_path_factory.mktemp("walker"), []))
    yield engine
    engine.shutdown()


def _bid(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid > 0, name
    return bid


def _limb_geoms(model, limb: str) -> list[int]:
    bid = _bid(model, limb)
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid]
    assert geoms, f"{limb} carries no geom, so it proves nothing"
    return geoms


def _collidable_limb_geoms(model, limb: str) -> list[int]:
    geoms = [
        g
        for g in _limb_geoms(model, limb)
        if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])
    ]
    assert geoms, f"{limb} carries no collidable geom, so it proves nothing"
    return geoms


def test_an_absent_walker_hides_and_frees_every_limb(engine):
    ctx = engine.ctx
    model = ctx.model
    walker = ctx.entities.get("pedestrian")
    before = {
        g: (int(model.geom_group[g]), int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
        for limb in LIMBS
        for g in _limb_geoms(model, limb)
    }
    try:
        assert set_present(ctx, walker, False)
        for limb in LIMBS:
            for g in _limb_geoms(model, limb):
                assert int(model.geom_group[g]) == ABSENT_GEOM_GROUP, limb
                assert not int(model.geom_contype[g]) and not int(model.geom_conaffinity[g])
                assert float(model.geom_rgba[g][3]) == 0.0
    finally:
        set_present(ctx, walker, True)
    after = {
        g: (int(model.geom_group[g]), int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
        for g in before
    }
    assert after == before


def test_a_clearance_monitor_under_a_walker_watches_its_limbs(tmp_path):
    world = _world(tmp_path, [])
    world["components"][1]["components"] = [{"clearance_monitor": {}, "name": "near"}]
    engine = _engine(world)
    try:
        model = engine.ctx.model
        plugin = next(p for p in engine.plugins if p.address == "pedestrian.near")
        watched = set(plugin._watched)
        for limb in ("pedestrian/knee_l", "pedestrian/ankle_r"):
            assert set(_collidable_limb_geoms(model, limb)) <= watched, limb
    finally:
        engine.shutdown()


def test_a_contact_scope_on_a_walker_watches_its_limbs(engine):
    from roqsim.contact_scope import resolve_contact_scope

    model = engine.ctx.model
    scope = resolve_contact_scope(engine.ctx, "pedestrian", plugin="test")
    for limb in LIMBS:
        assert all(scope.watched[g] for g in _limb_geoms(model, limb)), limb


def test_a_model_override_on_a_walker_selects_its_limbs(engine):
    from roqsim.plugins.model_override import ModelOverridePlugin

    config = {"overrides": [{"field": "geom_friction", "entity": "pedestrian", "to": 0.0}]}
    plugin = ModelOverridePlugin(config, name="slip")
    assert not plugin.validate_config(config)
    plugin.configure(engine.ctx)
    ids = set(plugin._targets[0].ids)
    for limb in LIMBS:
        assert set(_limb_geoms(engine.ctx.model, limb)) <= ids, limb


def test_a_limb_s_penetration_is_the_walker_s(engine):
    from roqsim.interpenetration import _entity_by_body, _owner_of

    model = engine.ctx.model
    owner = _entity_by_body(model, engine.ctx.entities)
    for limb in LIMBS:
        assert _owner_of(model, _bid(model, limb), owner) == "pedestrian", limb


def _limbs(model) -> set[int]:
    return {g for limb in LIMBS for g in _limb_geoms(model, limb)}


def test_a_walker_s_caution_probe_owns_its_limbs(engine):
    """A probe that saw them would find the walker's own arm in its way."""
    from roqsim_nav.caution import CautionProbe

    ctx = engine.ctx
    probe = CautionProbe({})
    probe.attach(ctx, ctx.entities.get("pedestrian"))
    assert _limbs(ctx.model) <= probe._own


def test_ignoring_a_walker_ignores_its_limbs(engine):
    """`avoidance: {stop: true, ignore: [walker]}`: a robot must not stop for the walker's legs."""
    from roqsim_nav.caution import CautionProbe

    ctx = engine.ctx
    probe = CautionProbe({"ignore": ["pedestrian"]})
    probe.attach(ctx, ctx.entities.get("robot"))
    probe.resolve_ignored(ctx)
    assert _limbs(ctx.model) <= probe._ignored
    assert not _limbs(ctx.model) & probe._own
