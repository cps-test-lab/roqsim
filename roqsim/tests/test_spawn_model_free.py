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
            "plugins": [
                {
                    "spawn_model": {
                        "model": "industrial_table",
                        "prefix": "t_",
                        "pos": [0, 0, 0],
                    },
                    "name": "table",
                },
                {
                    "spawn_model": {
                        "model": "graspable_box",
                        "prefix": "b_",
                        "pos": [0, 0, 0.9],
                        **box,
                    },
                    "name": "box",
                },
            ],
        },
        base_dir=tmp_path,
    )


def test_static_by_default_has_no_free_joint(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    entity = engine.ctx.entities.get("box")
    assert entity.kind == "prop"
    assert "base_joint" not in entity.meta
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "b_free") < 0


def test_free_prop_falls_and_settles_on_the_table(tmp_path):
    engine = Engine(_world(tmp_path, free=True, publish_tf="dynamic"))
    engine.setup()
    engine.reset()
    entity = engine.ctx.entities.get("box")
    # kind flips to "object", and base_joint is what lets SetEntityState re-seat it (the service
    # rejects any entity without one, which is why every prop used to be un-teleportable).
    assert entity.kind == "object"
    assert entity.meta["base_joint"] == "b_free"

    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b_graspable_box")
    for _ in range(1500):
        engine.step()
    # industrial_table's top is at z=0.76; the box is 0.05 m tall, so it rests at 0.785.
    assert data.xpos[bid][2] == pytest.approx(0.785, abs=0.01)


def test_reset_reseats_a_free_prop(tmp_path):
    """Without this, repetitions of a trial are not repetitions: the prop stays where it was left."""
    engine = Engine(_world(tmp_path, free=True))
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
            "components": [{"spawn_model": {"model": str(prop), "free": True}, "name": "g"}],
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
    engine = Engine(_world(tmp_path, free=True, mass=1.5, friction=[0.4, 0.005, 0.0001]))
    engine.setup()
    model = engine.ctx.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b_graspable_box")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "b_graspable_box")
    assert model.body_mass[bid] == pytest.approx(1.5)
    assert model.geom_friction[gid][0] == pytest.approx(0.4)


def test_static_publish_tf_is_refused_for_a_free_prop(tmp_path):
    """A latched one-shot pose for a body that moves is a frame frozen at the spawn pose."""
    from roqsim.config import instantiate_plugins
    from roqsim.plugin import PluginError

    with pytest.raises(PluginError, match="publish_tf: static"):
        instantiate_plugins(_world(tmp_path, free=True, publish_tf="static"))
