"""`box`: the parametric rectangular obstacle.

The contract worth pinning is the size convention (full extents, not MuJoCo half-extents) and the
floor-sitting default, because both are silent factor-of-two traps for a scene author.
"""

from __future__ import annotations

import math

import mujoco
import pytest

from roqsim.context import SimContext
from roqsim_assets.plugins.box import BoxPlugin


def _build(*, name="box", **cfg):
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 0.05]
    ctx = SimContext(config={})
    plugin = BoxPlugin(cfg, label=name)
    plugin.build(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx.model, ctx.data = model, data
    plugin.configure(ctx)
    return model, data, plugin, ctx


def _box_gid(model):
    for gid in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").endswith("box"):
            return gid
    raise AssertionError("no box geom in the compiled model")


def test_size_is_full_extents_not_half():
    model, _, _, _ = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.6, 0.8])
    gid = _box_gid(model)
    assert list(model.geom_size[gid][:3]) == pytest.approx([0.2, 0.3, 0.4])


def test_two_element_pos_sits_the_box_on_the_floor():
    model, data, _, _ = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.4, 0.8])
    gid = _box_gid(model)
    z = float(data.geom_xpos[gid][2])
    assert z == pytest.approx(0.4)  # centre at half height => bottom face at z = 0
    assert z - float(model.geom_size[gid][2]) == pytest.approx(0.0, abs=1e-9)


def test_three_element_pos_places_the_centre():
    model, data, _, _ = _build(pose={"position": {"x": 0.0, "y": 0.0, "z": 1.5}}, size=[0.4, 0.4, 0.8])
    assert float(data.geom_xpos[_box_gid(model)][2]) == pytest.approx(1.5)


def test_the_pose_orientation_rotates_the_box():
    model, data, _, _ = _build(
        pose={"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": math.pi / 2}},
        size=[1.0, 0.2, 0.5])
    gid = _box_gid(model)
    # After a 90 deg yaw the long axis points along world y.
    xmat = data.geom_xmat[gid].reshape(3, 3)
    assert abs(xmat[1, 0]) == pytest.approx(1.0, abs=1e-6)


def test_collide_false_makes_it_visual_only():
    model, _, _, _ = _build(pose={"position": {"x": 0.0, "y": 0.0}}, size=[0.4, 0.4, 0.4], collide=False)
    gid = _box_gid(model)
    assert model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0


def test_collides_by_default():
    model, _, _, _ = _build(pose={"position": {"x": 0.0, "y": 0.0}}, size=[0.4, 0.4, 0.4])
    gid = _box_gid(model)
    assert model.geom_contype[gid] != 0 and model.geom_conaffinity[gid] != 0


def test_registers_an_entity_with_its_geometry():
    _, _, plugin, ctx = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.4, 0.8], name="obstacle_3")
    entity = ctx.entities.get("obstacle_3")
    assert entity is not None and entity.kind == "prop"
    assert entity.meta["size"] == [0.4, 0.4, 0.8]
    assert entity.meta["pos"] == [1.0, 2.0, 0.4]


def test_validation_rejects_bad_geometry():
    assert "'pose' is required" in " ".join(BoxPlugin({}).validate_config({"size": [1, 1, 1]}))
    assert "'size' is required" in " ".join(BoxPlugin({}).validate_config({"pose": {"position": {"x": 0, "y": 0}}}))
    errs = BoxPlugin({}).validate_config({"pose": {"position": {"x": 0, "y": 0}}, "size": [0.4, 0.0, 0.8]})
    assert any("positive" in e for e in errs)
    errs = BoxPlugin({}).validate_config({"pose": {"position": {"x": 0, "y": 0}}, "size": [0.4, 0.4]})
    assert any("three numbers" in e for e in errs)


def test_several_boxes_coexist_under_distinct_prefixes():
    spec = mujoco.MjSpec()
    ctx = SimContext(config={})
    for i, x in enumerate((0.0, 1.4, 2.8)):
        BoxPlugin({"pose": {"position": {"x": x, "y": 0.0}}, "size": [0.4, 0.4, 0.8], "prefix": f"o{i}_"}).build(spec, ctx)
    model = spec.compile()
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)}
    assert {"o0_box", "o1_box", "o2_box"} <= names


def test_physics_by_default_so_a_trial_can_teleport_it():
    """A box is movable unless the world says otherwise.

    The other way round failed silently and expensively: SetEntityState refuses an entity with no
    free `base_joint`, so a world that parked an obstacle out of the way and teleported it in on
    cue failed on its first call, every run, while the world compiled, the entity existed under
    the name the caller used, and GetEntities listed it.
    """
    model, _, _, ctx = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.4, 0.8])
    assert model.njnt == 1
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert ctx.entities.get("box").meta["base_joint"] == "free"


def test_static_is_the_opt_out_for_scenery():
    """`motion: static` welds it: for anything the trial never moves, which spares the solver six
    DOFs per box and keeps a LIGHT box from being shoved out of a placement the experiment chose."""
    model, _, _, ctx = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.4, 0.8], motion="static")
    assert model.njnt == 0
    assert "base_joint" not in ctx.entities.get("box").meta


def test_physics_advertises_the_joint_as_base_joint():
    """The joint alone is not enough: simulation_interfaces' SetEntityState rejects any entity
    whose meta carries no free `base_joint`, so the plugin has to advertise it."""
    model, _, _, ctx = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.4, 0.8], motion="physics")
    assert model.njnt == 1
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    entity = ctx.entities.get("box")
    # `kind` names the role. What a consumer must ask about movability is `base_joint` -- the
    # thing SetEntityState and the planner grid actually need.
    assert entity.kind == "prop"
    assert entity.meta["base_joint"] == "free"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free")
    assert jid >= 0


def test_free_box_keeps_its_prefix_in_the_base_joint_name():
    """Two teleportable boxes must not name the same joint, or one would move the other."""
    spec = mujoco.MjSpec()
    ctx = SimContext(config={})
    plugins = []
    for i, x in enumerate((0.0, 1.4)):
        p = BoxPlugin(
            {
                "pose": {"position": {"x": x, "y": 0.0}},
                "size": [0.4, 0.4, 0.8],
                "prefix": f"o{i}_",
                "motion": "physics",
            },
            label=f"obstacle_{i}",
        )
        p.build(spec, ctx)
        plugins.append(p)
    model = spec.compile()
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    for p in plugins:
        p.configure(ctx)
    assert ctx.entities.get("obstacle_0").meta["base_joint"] == "o0_free"
    assert ctx.entities.get("obstacle_1").meta["base_joint"] == "o1_free"


def test_on_reset_returns_a_teleported_box_to_its_declared_pose():
    """A trial must not inherit where the previous trial left the obstacle."""
    model, data, plugin, ctx = _build(pose={"position": {"x": 1.0, "y": 2.0}}, size=[0.4, 0.4, 0.8], motion="physics")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free")
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr : adr + 3] = [5.0, 6.0, 7.0]  # as SetEntityState would teleport it
    data.qvel[int(model.jnt_dofadr[jid])] = 3.0
    plugin.on_reset(ctx)
    assert list(data.qpos[adr : adr + 2]) == pytest.approx([1.0, 2.0])
    assert data.qvel[int(model.jnt_dofadr[jid])] == pytest.approx(0.0)


def test_validation_rejects_a_motion_it_does_not_know():
    errs = BoxPlugin({}).validate_config({"pose": {"position": {"x": 0, "y": 0}}, "size": [0.4, 0.4, 0.8], "motion": "yes"})
    assert any("must be one of physics, static, driven" in e for e in errs)


# -- mocap: the third body state -------------------------------------------------------------------
def test_a_driven_box_has_no_dofs_and_is_a_mocap_body():
    """Immovable to the solver.

    Being a mocap body is also what keeps it out of a navigator's planner grid -- that filter is
    ``roqsim_nav``'s and is tested there; this package does not depend on it.
    """
    model, _, _, ctx = _build(pose={"position": {"x": 1.0, "y": 0.0}}, size=[0.4, 0.4, 0.5], motion="driven")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ctx.entities.get("box").body)
    assert int(model.body_mocapid[bid]) >= 0
    assert int(model.body_dofnum[bid]) == 0
    assert ctx.entities.get("box").kind == "prop"
    assert ctx.entities.get("box").meta["mocap"] is True


@pytest.mark.parametrize("gone, says", [("free", "'free' is gone"), ("mocap", "'mocap' is gone")])
def test_the_keys_motion_replaced_are_refused_not_ignored(gone, says):
    """`free` and `mocap` asked one question -- who owns this pose -- as two booleans, whose
    fourth combination was meaningless and had to be refused wherever they were offered.

    Refused rather than translated: this plugin declares no schema, so a key it merely stopped
    reading would be silently ignored, and the world would load, read as it always did, and
    behave differently."""
    cfg = {"pose": {"position": {"x": 0.0, "y": 0.0}}, "size": [1.0, 1.0, 1.0], gone: True}
    assert any(says in e for e in BoxPlugin(cfg).validate_config(cfg))
