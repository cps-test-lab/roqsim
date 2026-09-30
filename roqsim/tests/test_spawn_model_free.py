"""spawn_model `free: true`: a prop physics moves, rather than welded scenery.

Every prop in the library is static by default, so this is the path that makes an object a robot can
pick up expressible at all. The guards matter as much as the feature: a free body with no inertial
properties simulates erratically rather than obviously, which is expensive to diagnose downstream.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _world(tmp_path, **box):
    return load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_model": {
                        "model": "industrial_table",
                        "prefix": "t_",
                        "pose": {"position": {"x": 0, "y": 0, "z": 0}},
                    },
                    "name": "table",
                },
                {
                    "spawn_model": {
                        "model": "graspable_box",
                        "prefix": "b_",
                        "pose": {"position": {"x": 0, "y": 0, "z": 0.9}},
                        **box,
                    },
                    "name": "box",
                },
            ],
        },
        base_dir=tmp_path,
    )


def test_physics_by_default_gets_a_free_joint(tmp_path):
    """A prop is movable unless the world welds it: SetEntityState rejects any entity with no
    free `base_joint`, and defaulting the other way made that failure silent."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    entity = engine.ctx.entities.get("box")
    assert entity.meta["base_joint"] == "b_free"
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "b_free") >= 0


def test_static_is_the_opt_out(tmp_path):
    engine = Engine(_world(tmp_path, motion="static"))
    engine.setup()
    entity = engine.ctx.entities.get("box")
    assert entity.kind == "prop"
    assert "base_joint" not in entity.meta
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "b_free") < 0


def test_free_prop_falls_and_settles_on_the_table(tmp_path):
    engine = Engine(_world(tmp_path, motion="physics", publish_tf="dynamic"))
    engine.setup()
    engine.reset()
    entity = engine.ctx.entities.get("box")
    # kind flips to "object", and base_joint is what lets SetEntityState re-seat it (the service
    # rejects any entity without one, which would leave the prop un-teleportable).
    assert entity.kind == "prop"
    assert entity.meta["base_joint"] == "b_free"

    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b_graspable_box")
    for _ in range(1500):
        engine.step()
    # industrial_table's top is at z=0.76; the box is 0.05 m tall, so it rests at 0.785.
    assert data.xpos[bid][2] == pytest.approx(0.785, abs=0.01)


def test_reset_reseats_a_free_prop(tmp_path):
    """Without this, repetitions of a trial are not repetitions: the prop stays where it was left."""
    engine = Engine(_world(tmp_path, motion="physics"))
    engine.setup()
    engine.reset()
    model, data = engine.ctx.model, engine.ctx.data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "b_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = [2.0, 2.0, 3.0]  # knocked across the room
    data.qvel[model.jnt_dofadr[jid]] = 5.0
    engine.reset()
    assert list(data.qpos[adr : adr + 3]) == pytest.approx([0.0, 0.0, 0.9])
    assert data.qvel[model.jnt_dofadr[jid]] == pytest.approx(0.0)


def test_free_requires_mass(tmp_path):
    """A massless free body simulates erratically rather than obviously.

    MuJoCo derives mass from geom volume x density (default 1000), so a prop almost always ends up
    with sensible inertia -- the failure case is a `density="0"` geom, which is the convention for
    visual-only decoration and is used throughout the robot models here.
    """
    prop = tmp_path / "ghost.xml"
    prop.write_text(
        '<mujoco model="ghost"><worldbody><body name="ghost">'
        '<geom type="box" size="0.1 0.1 0.1" density="0"/>'
        "</body></worldbody></mujoco>"
    )
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [{"spawn_model": {"model": str(prop), "motion": "physics"}, "name": "g"}],
        },
        base_dir=tmp_path,
    )
    # MuJoCo itself refuses a massless moving body at compile time, so no plugin-side guard is
    # needed -- but assert it, because "free: true on a decoration geom" is an easy mistake and a
    # silently-simulated massless body would be far worse than a compile error.
    with pytest.raises(ValueError):
        Engine(cfg).setup()


def test_mass_and_friction_overrides_are_campaign_factors(tmp_path):
    """Both are world-YAML keys so a sweep needs no new variation plugin."""
    engine = Engine(_world(tmp_path, motion="physics", mass=1.5, friction=[0.4, 0.005, 0.0001]))
    engine.setup()
    model = engine.ctx.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b_graspable_box")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "b_graspable_box")
    assert model.body_mass[bid] == pytest.approx(1.5)
    assert model.geom_friction[gid][0] == pytest.approx(0.4)


def _prop_mass(tmp_path, geoms: str, mass: float, name: str = "prop"):
    """Spawn an inline prop with a mass override; return the compiled model and its root body id."""
    prop = tmp_path / f"{name}.xml"
    prop.write_text(
        f'<mujoco model="{name}"><worldbody><body name="{name}">{geoms}</body></worldbody></mujoco>'
    )
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {"spawn_model": {"model": str(prop), "prefix": "p_", "mass": mass}, "name": "p"}
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.setup()
    model = engine.ctx.model
    return model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"p_{name}")


# Two density-only boxes (0.2 m and 0.1 m cubes at 1000 and 500 kg/m^3: 8 kg and 0.5 kg) and a large
# visual-only box that declares mass="0", the way a prop's render mesh does.
_DENSITY_ONLY = (
    '<geom name="big" type="box" size="0.1 0.1 0.1" density="1000"/>'
    '<geom name="small" type="box" size="0.05 0.05 0.05" pos="0.3 0 0" density="500"/>'
    '<geom name="visual" type="box" size="0.5 0.5 0.5" mass="0" contype="0" conaffinity="0"/>'
)


def test_mass_override_of_a_density_only_prop_gives_the_requested_mass(tmp_path):
    """A geom that states only a density has no mass in the spec: MjSpec reports NaN.

    NaN passes both ``mass or 0.0`` and ``total <= 0.0``, so a total summed from the spec slips past
    any guard and a rescale by it writes NaN into every geom -- a visual box's ``mass="0"`` among
    them, which the compiler then weighs at the default 1000 kg/m^3.
    """
    model, bid = _prop_mass(tmp_path, _DENSITY_ONLY, mass=40.0)
    assert model.body_mass[bid] == pytest.approx(40.0)


def test_mass_override_keeps_the_split_and_leaves_a_visual_geom_massless(tmp_path):
    """The 8 : 0.5 split survives the rescale, and the mass="0" box is handed none of it."""
    model, bid = _prop_mass(tmp_path, _DENSITY_ONLY, mass=17.0)
    # The body's centre of mass sits where the two boxes' masses put it: 16 kg at x=0 and 1 kg at
    # x=0.3. A visual box carrying any mass would pull it back towards x=0.
    assert model.body_mass[bid] == pytest.approx(17.0)
    assert model.body_ipos[bid][0] == pytest.approx(0.3 * 1.0 / 17.0)


def test_mass_override_of_a_mixed_prop(tmp_path):
    """One geom declares mass, one only a density: both scale by the one factor."""
    geoms = (
        '<geom type="box" size="0.1 0.1 0.1" mass="2"/>'  # 2 kg at x=0
        '<geom type="box" size="0.1 0.1 0.1" pos="0.4 0 0" density="250"/>'  # 2 kg at x=0.4
    )
    model, bid = _prop_mass(tmp_path, geoms, mass=10.0)
    assert model.body_mass[bid] == pytest.approx(10.0)
    assert model.body_ipos[bid][0] == pytest.approx(0.2)


def test_mass_override_scales_an_explicit_inertial(tmp_path):
    """An <inertial> replaces what the geoms weigh, so it is what the override must scale."""
    geoms = (
        '<inertial pos="0 0 0" mass="3" diaginertia="0.1 0.2 0.3"/>'
        '<geom type="box" size="0.1 0.1 0.1" mass="1"/>'
    )
    model, bid = _prop_mass(tmp_path, geoms, mass=6.0)
    assert model.body_mass[bid] == pytest.approx(6.0)
    assert list(model.body_inertia[bid]) == pytest.approx([0.2, 0.4, 0.6])


def test_mass_override_of_a_massless_prop_is_refused(tmp_path):
    """Only visual geoms: nothing to rescale, so the override is refused rather than ignored."""
    from roqsim.models import ModelError

    geoms = (
        '<geom type="box" size="0.1 0.1 0.1" mass="0"/><geom type="sphere" size="0.1" density="0"/>'
    )
    with pytest.raises(ModelError, match="mass override needs the prop to have mass"):
        _prop_mass(tmp_path, geoms, mass=5.0)


def test_friction_on_a_root_with_no_geoms_is_refused(tmp_path):
    """A friction override lands on the root body's geoms; with none there it would change nothing.

    The prop's geom sits on a child body, which is a valid model: the override is refused rather than
    applied to nothing, so a friction sweep cannot run with every level identical.
    """
    from roqsim.models import ModelError

    prop = tmp_path / "nested.xml"
    prop.write_text(
        '<mujoco model="nested"><worldbody><body name="nested">'
        '<body name="part"><geom type="box" size="0.05 0.05 0.05"/></body>'
        "</body></worldbody></mujoco>"
    )
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_model": {"model": str(prop), "motion": "static", "friction": [0.4]},
                    "name": "n",
                }
            ],
        },
        base_dir=tmp_path,
    )
    with pytest.raises(ModelError, match="friction override needs geoms"):
        Engine(cfg).setup()


def test_static_publish_tf_is_refused_for_a_free_prop(tmp_path):
    """A latched one-shot pose for a body that moves is a frame frozen at the spawn pose."""
    from roqsim.config import instantiate_plugins
    from roqsim.plugin import PluginError

    with pytest.raises(PluginError, match="publish_tf: static"):
        instantiate_plugins(_world(tmp_path, motion="physics", publish_tf="static"))
