"""Resolving a caller's NAME to a body: the two-step every consumer used to write out by hand.

The failure this guards is not an exception, it is a *plausible number*. An entity's name is not its
body's name (``spawn_model`` prefixes the MJCF's root body), so a consumer that resolves the entity
name as a body name either crashes far from the cause or -- worse -- silently watches a body whose
pose is a compile-time constant and waits for it to move until the trial's timeout.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim.lookup import LookupError_, resolve_body_id
from roqsim.presence import ABSENT_GEOM_GROUP


def _world():
    """Three bodies, one of each kind that matters: welded scenery, a free prop, a mocap body."""
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 0.05]

    # Welded to the world: no joint, so its pose can never change.
    table = spec.worldbody.add_body()
    table.name = "work_table"
    top = table.add_geom()
    top.name, top.type = "table_top", mujoco.mjtGeom.mjGEOM_BOX
    top.size = [0.5, 0.3, 0.02]

    # A `spawn_model`-style prop: the ENTITY is named `parcel`, the BODY is not.
    prop = spec.worldbody.add_body()
    prop.name = "graspable_carton"
    prop.pos = [0.0, 0.0, 0.75]
    prop.add_freejoint()
    box = prop.add_geom()
    box.name, box.type = "carton_geom", mujoco.mjtGeom.mjGEOM_BOX
    box.size = [0.02, 0.012, 0.045]

    # A walker: zero DOFs, driven through `mocap_pos`, so it moves without a joint.
    walker = spec.worldbody.add_body()
    walker.name = "pedestrian"
    walker.mocap = True
    cap = walker.add_geom()
    cap.name, cap.type = "walker_geom", mujoco.mjtGeom.mjGEOM_SPHERE
    cap.size = [0.2, 0, 0]

    model = spec.compile()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    ctx.entities.add(Entity(name="parcel", kind="object", body="graspable_carton"))
    ctx.entities.add(Entity(name="table", kind="prop", body="work_table"))
    ctx.entities.add(Entity(name="walker", kind="pedestrian", body="pedestrian"))
    ctx.entities.add(Entity(name="marker", kind="marker", body=None))
    return ctx


@pytest.fixture
def ctx():
    return _world()


def test_an_entity_resolves_to_its_body_not_to_its_own_name(ctx):
    """The whole reason this function exists: `parcel` is not a body in the compiled model."""
    assert mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "parcel") < 0
    assert resolve_body_id(ctx, "parcel") == mujoco.mj_name2id(
        ctx.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_carton"
    )


def test_a_raw_body_name_still_resolves(ctx):
    """So a caller can watch something the world never registered -- a link, inherited scenery."""
    assert resolve_body_id(ctx, "graspable_carton") >= 0
    assert resolve_body_id(ctx, "work_table", movable=False) >= 0


def test_a_geom_name_is_not_a_body_name(ctx):
    """`table_top` is a geom. Bodies are the only thing with a pose, so this must not resolve."""
    with pytest.raises(LookupError_, match="not in this world"):
        resolve_body_id(ctx, "table_top", movable=False)


def test_a_body_welded_to_the_world_is_refused_when_movement_is_expected(ctx):
    """`xpos` is a compile-time constant there, so waiting for it to move never ends.

    Refused by NAME with the remedy, rather than returning an id that produces a timeout twenty
    seconds later in a caller that cannot explain it.
    """
    with pytest.raises(LookupError_, match="welded to the world"):
        resolve_body_id(ctx, "table")
    # ...and allowed when the caller only wants a frame to read.
    assert resolve_body_id(ctx, "table", movable=False) >= 0


def test_a_mocap_body_is_accepted_although_it_has_no_joint(ctx):
    """The reason the check is not `dofnum == 0` alone: a walker has zero DOFs and moves anyway."""
    bid = resolve_body_id(ctx, "walker")
    # Not `body_weldid`: it reads 0 for a mocap body up to MuJoCo 3.11 and the body's own id from
    # 3.12 on, so asserting either number pins the test to one MuJoCo. `dofnum`/`mocapid` do not move.
    assert int(ctx.model.body_dofnum[bid]) == 0, "precondition: no DOFs of its own"
    assert int(ctx.model.body_mocapid[bid]) >= 0
    assert resolve_body_id(ctx, "pedestrian") == bid


def test_an_entity_without_a_body_says_so(ctx):
    """`Entity.body` is optional; `mj_name2id(None)` would crash several frames from the cause."""
    with pytest.raises(LookupError_, match="has no body"):
        resolve_body_id(ctx, "marker")


def test_an_absent_entity_is_refused(ctx):
    """presence leaves it in the model at its true pose, so the pose reads and means nothing."""
    ctx.entities.get("parcel").present = False
    with pytest.raises(LookupError_, match="ABSENT"):
        resolve_body_id(ctx, "parcel")
    assert ABSENT_GEOM_GROUP  # the mechanism the message points at


def test_an_unknown_name_names_the_near_misses(ctx):
    """Over BOTH namespaces, because the caller does not know which one they meant."""
    with pytest.raises(LookupError_) as err:
        resolve_body_id(ctx, "parcell", what="entity_moved entities")
    msg = str(err.value)
    assert "entity_moved entities" in msg, "the caller's own label, so the wrong key is named"
    assert "parcel" in msg


def test_the_error_says_which_namespace_was_tried(ctx):
    """A name that is neither an entity nor a body must not read as 'no such entity'."""
    with pytest.raises(LookupError_, match="no entity of that name"):
        resolve_body_id(ctx, "totally_unrelated")
