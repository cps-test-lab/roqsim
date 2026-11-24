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
