"""A model with a ``<flexcomp>`` works as a prop: spawned, scaled, weighed, hidden, and refused by name.

Two halves, as in ``test_flex_rules.py``. The first pins the MuJoCo 3.14 behaviours
:mod:`roqsim.flex` records for a flex inside an attached model (rules 4 and 5), so a MuJoCo that
changes one fails here. The second is ``spawn_model`` and :mod:`roqsim.presence` on such a model,
measured on the running simulation -- where a vertex rests, what the floor carries, what a contact
uses, what a frame shows -- rather than read back from the fields the code wrote.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.flex import entity_flex_ids, lift_top_level_flexes
from roqsim.models import ModelError
from roqsim.presence import ABSENT_GEOM_GROUP, set_present

# A 3x3x3 lattice, 2 cm apart: small enough to settle in a fraction of a second.
_GRID = 'type="grid" count="3 3 3" spacing=".02 .02 .02" dim="3" radius=".002"'
_MATERIAL = (
    '<edge equality="false"/><elasticity young="5e4" poisson="0.3" damping="0.002"/>'
    '<contact selfcollide="none" solref="0.01 1"/>'
)

#: A free soft block: declared in a body of its own, nothing pinned.
FREE_BLOCK = f"""<mujoco><worldbody><body name="soft_block">
  <flexcomp name="soft" {_GRID} mass="0.1" rgba="1 0 0 1">{_MATERIAL}</flexcomp>
</body></worldbody></mujoco>"""

#: The same block written straight under <worldbody>.
TOP_LEVEL = f"""<mujoco><worldbody>
  <flexcomp name="soft" {_GRID} mass="0.1">{_MATERIAL}</flexcomp>
</worldbody></mujoco>"""

#: A top-level block with its bottom layer pinned -- to the world body, since it has no other.
TOP_LEVEL_PINNED = f"""<mujoco><worldbody>
  <flexcomp name="soft" {_GRID} mass="0.1" pos="0 0 0.03"><pin gridrange="0 0 0 2 2 0"/>{_MATERIAL}</flexcomp>
</worldbody></mujoco>"""

#: A block pinned to a rigid base plate that carries declared mass: one prop, two materials.
ON_A_BASE = f"""<mujoco><worldbody><body name="based">
  <geom name="plate" type="box" size=".03 .03 .005" pos="0 0 .005" mass="0.1"/>
  <flexcomp name="soft" {_GRID} mass="0.1" pos="0 0 .03"><pin gridrange="0 0 0 2 2 0"/>{_MATERIAL}</flexcomp>
</body></worldbody></mujoco>"""

#: A static slab over the world's own floor, with little friction of its own, so a contact's friction
#: is the flex's (the world floor's 2.0 would win the equal-priority maximum). Its top is at 1 cm.
SLAB = """<mujoco><worldbody><body name="slab">
  <geom name="slab" type="box" size=".5 .5 .005" pos="0 0 .005" friction="0.1 0.005 0.0001"/>
</body></worldbody></mujoco>"""
TOP = 0.01

G = 9.81


def _write(tmp_path, name, xml):
    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _engine(tmp_path, xml, *, name="soft_block", z=0.05, **spawn):
    """A slab and the model under test, spawned as the entity ``prop`` with prefix ``p_``."""
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {"spawn_model": {"model": str(_write(tmp_path, "slab", SLAB)), "motion": "static"}},
                {
                    "spawn_model": {
                        "model": str(_write(tmp_path, name, xml)),
                        "prefix": "p_",
                        "pose": {"position": {"x": 0.0, "y": 0.0, "z": z}},
                        **spawn,
                    },
                    "name": "prop",
                },
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _settle(engine, seconds=1.0):
    for _ in range(int(seconds / engine.ctx.model.opt.timestep)):
        engine.step()


def _flex_contacts(model, data):
    return [data.contact[i] for i in range(data.ncon) if (data.contact[i].flex >= 0).any()]


def _floor_load(model, data) -> float:
    """What the slab carries: the normal force summed over every contact with it."""
    floor = model.geom("slab").id
    force, total = np.zeros(6), 0.0
    for i in range(data.ncon):
        if floor in data.contact[i].geom:
            mujoco.mj_contactForce(model, data, i, force)
            total += force[0]
    return total


# -- what MuJoCo 3.14 does with a flex in an attached model ---------------------------------------


def _attach(child):
    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    spec.attach(child, prefix="p_", frame=spec.worldbody.add_frame())
    return spec.compile()


def test_attach_drops_a_flex_pinned_to_the_models_world_body():
    """Rule 4: silently -- the model compiles, with no flex in it."""
    assert _attach(mujoco.MjSpec.from_string(TOP_LEVEL_PINNED)).nflex == 0


def test_lifted_the_pinned_flex_survives_the_attach():
    model = _attach(lift_top_level_flexes(mujoco.MjSpec.from_string(TOP_LEVEL_PINNED), "blk"))
    assert model.nflex == 1
    pins = {model.body(int(b)).name for b in model.flex_vertbodyid}
    assert "p_blk" in pins


def test_a_mesh_flexcomp_reads_its_file_while_the_model_is_parsed(tmp_path):
    """Rule 5: the file resolves against the model's own folder and meshdir, before apply_assets."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "tet.obj").write_text(
        "v 0 0 0\nv .05 0 0\nv 0 .05 0\nv 0 0 .05\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    skin = (
        '<flexcomp name="skin" type="mesh" file="tet.obj" dim="2" radius=".002" mass=".05">'
        '<edge equality="true"/></flexcomp>'
    )
    beside = _write(
        tmp_path,
        "beside",
        f'<mujoco><compiler meshdir="meshes"/><worldbody><body name="beside">{skin}</body>'
        "</worldbody></mujoco>",
    )
    child = mujoco.MjSpec.from_file(str(beside))
    assert len(child.meshes) == 0  # nothing for apply_assets to rewrite
    elsewhere = _write(
        tmp_path,
        "elsewhere",
        f'<mujoco><worldbody><body name="b">{skin}</body></worldbody></mujoco>',
    )
    with pytest.raises(ValueError, match="flexcomp"):
        mujoco.MjSpec.from_file(str(elsewhere))

    # Through spawn_model: beside the model it spawns; elsewhere the refusal says where it must be.
    cfg = load_config_from_dict(
        {"sim": {}, "components": [{"spawn_model": {"model": str(beside), "prefix": "p_"}}]},
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    assert mujoco.mj_id2name(engine.ctx.model, mujoco.mjtObj.mjOBJ_FLEX, 0) == "p_skin"
    cfg = load_config_from_dict(
        {"sim": {}, "components": [{"spawn_model": {"model": str(elsewhere)}}]}, base_dir=tmp_path
    )
    with pytest.raises(ModelError, match="beside the model"):
        Engine(cfg).setup()


# -- spawn_model: a flex prop ---------------------------------------------------------------------


def test_a_free_flex_is_a_prop_of_free_vertices_that_rests_on_the_floor(tmp_path):
    engine = _engine(tmp_path, FREE_BLOCK)
    model, data = engine.ctx.model, engine.ctx.data
    entity = engine.ctx.entities.get("prop")
    # No free joint: the vertices are the prop's degrees of freedom, and a free joint on the empty
    # root would be a massless body.
    assert "base_joint" not in entity.meta
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "p_free") < 0
    assert entity.meta["flexes"] == ["p_soft"]
    assert engine.integrator.resolved == "discrete"
    _settle(engine)
    assert np.abs(data.qvel).max() < 0.05
    bottom = data.flexvert_xpos[:, 2].min()
    assert TOP < bottom < TOP + 0.01, "the block rests on the slab rather than falling through it"


def test_a_top_level_flexcomp_gets_a_root_body_named_after_the_file(tmp_path):
    engine = _engine(tmp_path, TOP_LEVEL, name="blob")
    model = engine.ctx.model
    entity = engine.ctx.entities.get("prop")
    assert entity.body == "p_blob"
    root = model.body("p_blob").id
    # Every vertex body sits under it, renamed by the same prefix as everything else in the prop.
    for b in model.flex_vertbodyid:
        assert int(model.body_parentid[b]) == root
        assert model.body(int(b)).name.startswith("p_soft")
    assert entity.meta["flexes"] == ["p_soft"]


def test_a_world_pin_holds_the_flex_where_the_prop_is_spawned(tmp_path):
    engine = _engine(tmp_path, TOP_LEVEL_PINNED, name="anchored", z=0.2, motion="static")
    model, data = engine.ctx.model, engine.ctx.data
    assert model.nflex == 1
    _settle(engine, 0.5)
    root = model.body("p_anchored").id
    pinned = [v for v, b in enumerate(model.flex_vertbodyid) if b == root]
    assert len(pinned) == 9
    # The bottom layer, 2 cm below the lattice centre at 0.03, spawned 0.2 up: it stays at 0.21.
    assert data.flexvert_xpos[pinned, 2] == pytest.approx(np.full(9, 0.21), abs=1e-9)


def test_a_world_pin_under_motion_physics_is_refused_by_name(tmp_path):
    with pytest.raises(ModelError, match="no mass of its own"):
        _engine(tmp_path, TOP_LEVEL_PINNED, name="anchored")


@pytest.mark.parametrize("motion", ["static", "driven"])
def test_welding_or_driving_a_free_flex_is_refused(tmp_path, motion):
    with pytest.raises(ModelError, match="Pin the vertices"):
        _engine(tmp_path, FREE_BLOCK, motion=motion)


def test_a_pinned_flex_can_be_welded(tmp_path):
    engine = _engine(tmp_path, ON_A_BASE, name="based", motion="static", z=0.3)
    assert engine.ctx.model.nflex == 1


def test_a_flex_that_collides_is_enough_to_rest_on(tmp_path):
    ghost = FREE_BLOCK.replace(
        'selfcollide="none"', 'selfcollide="none" contype="0" conaffinity="0"'
    )
    with pytest.raises(ModelError, match="no colliding geometry"):
        _engine(tmp_path, ghost)


def _extent(engine) -> np.ndarray:
    data = engine.ctx.data
    mujoco.mj_forward(engine.ctx.model, data)
    return data.flexvert_xpos.max(axis=0) - data.flexvert_xpos.min(axis=0)


#: The free block with its vertices interpolated from 2x2x2 nodes: no vertex has a body of its own.
TRILINEAR = FREE_BLOCK.replace('count="3 3 3"', 'count="4 4 4" dof="trilinear"')


@pytest.mark.parametrize(
    ("xml", "name"),
    [(FREE_BLOCK, "soft_block"), (ON_A_BASE, "based"), (TRILINEAR, "soft_block")],
    ids=["free", "pinned", "trilinear"],
)
def test_scale_resizes_the_flex_with_the_prop(tmp_path, xml, name):
    """Vertex bodies, and the vertices a flex keeps in its own fields, scale as one."""
    one = _extent(_engine(tmp_path, xml, name=name, z=0.5))
    two = _extent(_engine(tmp_path, xml, name=name, z=0.5, scale=2.0))
    assert two == pytest.approx(2.0 * one, rel=1e-9)
    assert one.min() >= 0.04


@pytest.mark.parametrize(
    ("xml", "name"), [(FREE_BLOCK, "soft_block"), (ON_A_BASE, "based")], ids=["free", "pinned"]
)
def test_mass_sets_the_weight_the_slab_carries(tmp_path, xml, name):
    """The override covers the flex's vertices -- measured as the load on the slab.

    Averaged over a second, because a lightly damped soft block never quite stops jiggling on its
    contacts, and over time the slab carries exactly what rests on it.
    """
    engine = _engine(tmp_path, xml, name=name, mass=0.5)
    _settle(engine, 0.5)
    loads = []
    for _ in range(int(1.0 / engine.ctx.model.opt.timestep)):
        engine.step()
        loads.append(_floor_load(engine.ctx.model, engine.ctx.data))
    assert np.mean(loads) == pytest.approx(0.5 * G, rel=0.01)


def test_friction_is_what_the_flex_contacts_use(tmp_path):
    """The slab's 0.1 is lower, so at equal priority the flex's value is the contact's."""
    stated = _engine(tmp_path, FREE_BLOCK, friction=0.6)
    default = _engine(tmp_path, FREE_BLOCK)
    for engine, expected in ((stated, 0.6), (default, 1.0)):
        _settle(engine, 0.3)
        contacts = _flex_contacts(engine.ctx.model, engine.ctx.data)
        assert contacts
        assert {round(float(c.friction[0]), 6) for c in contacts} == {expected}


# -- presence: a flex prop made absent and back ---------------------------------------------------


def _red_pixels(model, data) -> int:
    from roqsim.rendering import FrameRenderer

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.02]
    cam.distance, cam.elevation, cam.azimuth = 0.3, -60.0, 30.0
    frame = FrameRenderer(model, 96, 72, camera=cam)
    try:
        image = frame.render(data).astype(int)
    finally:
        frame.close()
    return int(((image[..., 0] - image[..., 2]) > 60).sum())


def test_an_absent_flex_is_untouchable_unseen_and_still(tmp_path):
    engine = _engine(tmp_path, FREE_BLOCK, z=0.035)
    model, data, ctx = engine.ctx.model, engine.ctx.data, engine.ctx
    entity = ctx.entities.get("prop")
    _settle(engine, 0.5)
    assert _flex_contacts(model, data)
    assert _red_pixels(model, data) > 0
    fid = entity_flex_ids(model, entity.body)[0]
    declared = (
        int(model.flex_group[fid]),
        int(model.flex_contype[fid]),
        int(model.flex_conaffinity[fid]),
        float(model.flex_rgba[fid][3]),
    )

    set_present(ctx, entity, False)
    before = data.flexvert_xpos.copy()
    _settle(engine, 0.5)
    assert not _flex_contacts(model, data), "an absent flex touches nothing"
    assert _red_pixels(model, data) == 0, "an absent flex is not drawn"
    # No contact holds it up any more, and it does not fall: frozen like any absent entity. What
    # moves is the block letting go of the squeeze the slab had put in it -- internal forces, which
    # leave its centroid where it was (to within microns; half a second of falling is 1.2 m).
    assert np.abs(data.flexvert_xpos.mean(axis=0) - before.mean(axis=0)).max() < 1e-5
    assert np.abs(data.flexvert_xpos - before).max() < 1e-3
    assert int(model.flex_group[fid]) == ABSENT_GEOM_GROUP

    set_present(ctx, entity, True)
    restored = (
        int(model.flex_group[fid]),
        int(model.flex_contype[fid]),
        int(model.flex_conaffinity[fid]),
        float(model.flex_rgba[fid][3]),
    )
    assert restored == declared
    _settle(engine, 0.5)
    assert _flex_contacts(model, data)
    assert _red_pixels(model, data) > 0


def test_a_prop_declared_absent_hides_its_flex_from_the_start(tmp_path):
    engine = _engine(tmp_path, FREE_BLOCK, z=0.035, present=False)
    _settle(engine, 0.3)
    assert not _flex_contacts(engine.ctx.model, engine.ctx.data)


def test_a_flex_that_straddles_two_entities_belongs_to_neither(caplog):
    """A cloth pinned between two bodies is logged against each, not hidden with either."""
    xml = """<mujoco><worldbody>
      <body name="left" pos="-.1 0 .5"><geom type="box" size=".01 .01 .01"/></body>
      <body name="right" pos=".1 0 .5"><geom type="box" size=".01 .01 .01"/></body>
      </worldbody>
      <deformable><flex name="cloth" dim="1" body="left right" vertex="0 0 0 0 0 0" element="0 1"/></deformable>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    with caplog.at_level(logging.WARNING, logger="roqsim.flex"):
        assert entity_flex_ids(model, "left") == []
    assert "cloth" in caplog.text and "partly" in caplog.text
    assert entity_flex_ids(model, "world") == [0]
