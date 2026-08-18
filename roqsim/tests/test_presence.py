"""Presence: making a compiled entity absent, without moving it.

The contract worth pinning is that absence is *perceptual and physical* rather than positional.
Parking an entity out of sight is the obvious alternative and is wrong in a specific way: a free
body keeps accelerating under gravity while it is away, so it returns with whatever velocity it
accumulated.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim.presence import ABSENT_GEOM_GROUP, entity_geom_ids, set_present, visible_geomgroup_mask


def _world():
    """A prop with a child body, plus a wall, so subtree handling is exercised."""
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 0.05]

    prop = spec.worldbody.add_body()
    prop.name = "prop"
    prop.pos = [2.0, 0.0, 0.5]
    body_geom = prop.add_geom()
    body_geom.name, body_geom.type = "prop_geom", mujoco.mjtGeom.mjGEOM_BOX
    body_geom.size = [0.25, 0.25, 0.5]

    child = prop.add_body()
    child.name = "prop_lid"
    child.pos = [0.0, 0.0, 0.5]
    lid = child.add_geom()
    lid.name, lid.type = "lid_geom", mujoco.mjtGeom.mjGEOM_BOX
    lid.size = [0.25, 0.25, 0.05]

    model = spec.compile()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    return ctx


@pytest.fixture
def ctx():
    return _world()


@pytest.fixture
def prop():
    return Entity(name="prop", kind="object", body="prop")


def _groups(ctx, entity):
    return [int(ctx.model.geom_group[g]) for g in entity_geom_ids(ctx.model, entity.body)]


def test_an_entity_covers_its_whole_subtree(ctx, prop):
    """A prop is rarely one body; a visible child would make absence mean "mostly absent"."""
    assert len(entity_geom_ids(ctx.model, "prop")) == 2


def test_absent_geoms_move_to_the_reserved_group(ctx, prop):
    set_present(ctx, prop, False)
    assert _groups(ctx, prop) == [ABSENT_GEOM_GROUP, ABSENT_GEOM_GROUP]


def test_absent_geoms_collide_with_nothing(ctx, prop):
    set_present(ctx, prop, False)
    for gid in entity_geom_ids(ctx.model, "prop"):
        assert int(ctx.model.geom_contype[gid]) == 0
        assert int(ctx.model.geom_conaffinity[gid]) == 0


def test_the_pose_does_not_move(ctx, prop):
    """The point of doing it this way: nothing falls, drifts, or returns with velocity."""
    before = ctx.data.xpos[mujoco.mj_name2id(
        ctx.model, mujoco.mjtObj.mjOBJ_BODY, "prop")].copy()
    set_present(ctx, prop, False)
    mujoco.mj_forward(ctx.model, ctx.data)
    after = ctx.data.xpos[mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "prop")]
    assert np.allclose(before, after)


def test_returning_restores_exactly_what_the_world_declared(ctx, prop):
    """Restoring by remembering, not by guessing a default the world never had."""
    gid = entity_geom_ids(ctx.model, "prop")[0]
    ctx.model.geom_group[gid] = 1
    ctx.model.geom_contype[gid] = 5
    ctx.model.geom_conaffinity[gid] = 3
    ctx.model.geom_rgba[gid][3] = 0.4

    set_present(ctx, prop, False)
    set_present(ctx, prop, True)

    assert int(ctx.model.geom_group[gid]) == 1
    assert int(ctx.model.geom_contype[gid]) == 5
    assert int(ctx.model.geom_conaffinity[gid]) == 3
    assert ctx.model.geom_rgba[gid][3] == pytest.approx(0.4)


def test_setting_the_state_it_already_has_changes_nothing(ctx, prop):
    assert set_present(ctx, prop, True) is False
    assert set_present(ctx, prop, False) is True
    assert set_present(ctx, prop, False) is False


def test_an_entity_with_no_body_still_tracks_presence(ctx):
    """Nothing to hide, but the control plane's listing must still follow it."""
    entity = Entity(name="abstract", kind="object", body=None)
    assert set_present(ctx, entity, False) is True
    assert entity.present is False


# -- what makes absence perceptual ------------------------------------------------------------

def test_the_raycast_mask_excludes_the_absent_group():
    """Load-bearing: mj_multiRay ignores contype/conaffinity and tests the real triangles.

    Disabling contact alone would leave an "absent" obstacle a perfectly good lidar return --
    which is exactly what a navigation stack reacts to.
    """
    mask = visible_geomgroup_mask()
    assert mask[ABSENT_GEOM_GROUP] == 0
    assert all(mask[g] == 1 for g in range(mujoco.mjNGROUP) if g != ABSENT_GEOM_GROUP)


def test_the_absent_group_does_not_collide_with_the_ones_in_use():
    """0-1 visual, 2 sensor FOV, 3 collision-only -- 4 is the first free one."""
    assert ABSENT_GEOM_GROUP not in (0, 1, 2, 3)
    assert ABSENT_GEOM_GROUP < mujoco.mjNGROUP


def test_the_renderer_excludes_the_absent_group():
    from roqsim.rendering import FrameRenderer

    ctx = _world()
    renderer = FrameRenderer(ctx.model, 64, 48)
    assert renderer._vopt.geomgroup[ABSENT_GEOM_GROUP] == 0


def test_a_transition_says_so_in_the_log(ctx, prop, caplog):
    """The only trace a flip leaves, until presence rides in the capture.

    It writes model fields while a recording stores ``mjData`` state, and the pose deliberately
    does not move -- so without this line nothing in a run's recorded data can answer "did the
    obstacle ever appear?", on a campaign whose service call returned OK.
    """
    with caplog.at_level("INFO", logger="roqsim.presence"):
        assert set_present(ctx, prop, False)
        assert set_present(ctx, prop, True)
        # A no-op transition must stay silent, or the log stops meaning "something changed".
        assert not set_present(ctx, prop, True)

    lines = [r.getMessage() for r in caplog.records if r.name == "roqsim.presence"]
    assert len(lines) == 2, lines
    assert "absent" in lines[0] and prop.name in lines[0]
    assert "present" in lines[1] and prop.name in lines[1]
