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
from roqsim.presence import (
    ABSENT_GEOM_GROUP,
    arm_gravity_compensation,
    entity_geom_ids,
    set_present,
    visible_geomgroup_mask,
)


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
    before = ctx.data.xpos[mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "prop")].copy()
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


# -- what makes absence physical --------------------------------------------------------------


def _falling_world():
    """A free-jointed prop above a floor, in a world armed the way the engine arms every world."""
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 0.05]

    prop = spec.worldbody.add_body()
    prop.name = "crate"
    prop.pos = [0.0, 0.0, 1.0]
    prop.add_freejoint()
    geom = prop.add_geom()
    geom.name, geom.type = "crate_geom", mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.25, 0.25, 0.25]

    arm_gravity_compensation(spec)
    model = spec.compile()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    return ctx


def _z(ctx):
    return float(ctx.data.qpos[2])


def test_an_absent_free_body_does_not_fall():
    """The defect absence would otherwise introduce: no contacts means nothing holds it up.

    Ten seconds of free fall is ~490 m, so a tolerance in millimetres cannot pass by accident.
    """
    ctx = _falling_world()
    crate = Entity(name="crate", kind="object", body="crate")
    set_present(ctx, crate, False)
    for _ in range(5000):
        mujoco.mj_step(ctx.model, ctx.data)
    assert ctx.data.time == pytest.approx(10.0, abs=0.05)
    assert _z(ctx) == pytest.approx(1.0, abs=1e-3)


def test_a_present_free_body_still_falls():
    """The guard on the one above: it must pin absence, not a world where gravity stopped."""
    ctx = _falling_world()
    for _ in range(200):
        mujoco.mj_step(ctx.model, ctx.data)
    assert _z(ctx) < 0.9


def test_an_entity_deleted_in_flight_stops_where_it_was():
    """Compensation removes the force, not the motion the entity already had."""
    ctx = _falling_world()
    crate = Entity(name="crate", kind="object", body="crate")
    for _ in range(100):
        mujoco.mj_step(ctx.model, ctx.data)
    caught = _z(ctx)
    assert ctx.data.qvel[2] < -1.0  # genuinely moving when it goes

    set_present(ctx, crate, False)
    for _ in range(2000):
        mujoco.mj_step(ctx.model, ctx.data)
    assert _z(ctx) == pytest.approx(caught, abs=1e-3)


def test_returning_gives_back_the_gravity_the_world_compiled():
    """An entity that came back weightless would be a stranger physics stopped acting on."""
    ctx = _falling_world()
    crate = Entity(name="crate", kind="object", body="crate")
    set_present(ctx, crate, False)
    set_present(ctx, crate, True)
    for _ in range(200):
        mujoco.mj_step(ctx.model, ctx.data)
    assert _z(ctx) < 0.9
    assert (
        float(
            ctx.model.body_gravcomp[mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")]
        )
        == 0.0
    )


def test_freezing_is_inert_without_the_compile_time_arming():
    """Why arming exists at all, stated as a test rather than trusted to a comment.

    MuJoCo derives ``ngravcomp`` when the model compiles and it is read-only afterwards, so in a
    world nobody armed the gravcomp write is silently ignored and an absent body falls. A future
    MuJoCo that made the field live unconditionally would turn this test red, which is the point:
    the arming would then be removable rather than quietly redundant.
    """
    spec = mujoco.MjSpec()
    prop = spec.worldbody.add_body()
    prop.name = "crate"
    prop.pos = [0.0, 0.0, 1.0]
    prop.add_freejoint()
    geom = prop.add_geom()
    geom.name, geom.type = "crate_geom", mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.25, 0.25, 0.25]
    model = spec.compile()  # deliberately NOT armed
    assert model.ngravcomp == 0

    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    set_present(ctx, Entity(name="crate", kind="object", body="crate"), False)
    for _ in range(200):
        mujoco.mj_step(ctx.model, ctx.data)
    assert float(ctx.data.qpos[2]) < 0.9


def test_arming_compensates_nothing_that_can_move():
    """The world body carries the mark because it is the one body the mark cannot affect."""
    ctx = _falling_world()
    assert ctx.model.ngravcomp == 1
    assert float(ctx.model.body_gravcomp[0]) == 1.0
    assert [float(g) for g in ctx.model.body_gravcomp[1:]] == [0.0]
