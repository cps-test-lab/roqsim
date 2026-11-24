# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""One separation per kind of obstacle: ``clearance_monitor``'s ``entities`` and ``hit_geometry``.

A walker is the case that needs both. Its limbs are mocap bodies of the world, not children of
its pelvis, and their geoms carry no names, so neither a subtree nor an ``ignore`` list can single
it out; and its arms do not collide, so a monitor that counts only what could touch the robot
never measures them.
"""

import mujoco
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.engine import Engine
from roqsim.presence import entity_geoms
from roqsim.scene_content import COLLISION_GROUP

_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <geom name="wall" type="box" pos="0 -1.0 .5" size="3 .05 .5"/>
  </worldbody>
</mujoco>
"""

_ROBOT = """
<mujoco model="rover">
  <worldbody>
    <body name="base" pos="0 0 .2">
      <geom name="base_geom" type="cylinder" size=".2 .2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _world(tmp_path, monitors):
    (tmp_path / "s.xml").write_text(_SCENE)
    (tmp_path / "rover.xml").write_text(_ROBOT)
    return {
        "sim": {"world": str(tmp_path / "s.xml")},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "rover.xml"),
                    "pose": {"position": {"x": 0.0, "y": 0.0}},
                    "motion": "physics",
                },
                "name": "robot",
                "components": monitors,
            },
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "skin": False,
                    "pose": {"position": {"x": 2.0, "y": 0.0}},
                },
                "name": "pedestrian",
            },
        ],
    }


def _engine(world):
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    for _ in range(10):
        engine.step()
    return engine


def _report(engine, name):
    return engine.ctx.blackboard.get(f"clearance:robot.{name}")()


def _body_of(model, geom: str) -> str:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if gid < 0:
        gid = int(geom.removeprefix("geom"))
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""


def test_a_walker_entity_owns_every_limb(tmp_path):
    engine = _engine(_world(tmp_path, [{"clearance_monitor": {}}]))
    try:
        model = engine.ctx.model
        geoms = entity_geoms(model, engine.ctx.entities.get("pedestrian"))
        bodies = {model.body(int(model.geom_bodyid[g])).name for g in geoms}
        assert "pedestrian/pelvis" in bodies
        assert {"pedestrian/shoulder_l", "pedestrian/elbow_r"} <= bodies
    finally:
        engine.shutdown()


def test_two_monitors_give_one_separation_per_kind(tmp_path):
    """The wall is nearer (0.75 m) than the walker (about 1.8 m). A monitor with no `entities`
    reports the wall; one naming the walker reports the walker, from the same robot."""
    world = _world(
        tmp_path,
        [
            {"clearance_monitor": {}, "name": "everything"},
            {"clearance_monitor": {"entities": ["pedestrian"]}, "name": "humans"},
        ],
    )
    engine = _engine(world)
    try:
        model = engine.ctx.model
        everything = _report(engine, "everything")
        humans = _report(engine, "humans")
        assert everything.geom == "wall"
        assert everything.current == pytest.approx(0.75, abs=0.02)
        assert _body_of(model, humans.geom).startswith("pedestrian/")
        assert 1.0 < humans.current < 2.0
    finally:
        engine.shutdown()


def test_hit_geometry_adds_the_walker_s_arms(tmp_path):
    world = _world(
        tmp_path,
        [
            {"clearance_monitor": {"entities": ["pedestrian"]}, "name": "solid"},
            {
                "clearance_monitor": {"entities": ["pedestrian"], "hit_geometry": True},
                "name": "all",
            },
        ],
    )
    engine = _engine(world)
    try:
        model = engine.ctx.model
        plugins = {p.address: p for p in engine.plugins}

        def counted(plugin):
            return {c for w, c in plugin._pairs if plugin._counts(model, w, c)}

        solid = counted(plugins["robot.solid"])
        every = counted(plugins["robot.all"])
        added = every - solid
        assert added, "no geom added"
        for g in added:
            assert int(model.geom_group[g]) == COLLISION_GROUP
            assert not int(model.geom_contype[g]) and not int(model.geom_conaffinity[g])
        assert any("shoulder" in _body_of(model, f"geom{g}") for g in added)
        # The arms can only bring the minimum nearer.
        assert _report(engine, "all").minimum <= _report(engine, "solid").minimum + 1e-9
    finally:
        engine.shutdown()


def test_an_entity_the_world_does_not_have_fails_the_first_measurement(tmp_path):
    world = _world(tmp_path, [{"clearance_monitor": {"entities": ["nobody"]}}])
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RuntimeError, match="no entity 'nobody'"):
            for _ in range(10):
                engine.step()
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "config, match",
    [
        ({"hit_geometry": True}, "needs 'entities'"),
        ({"entities": []}, "non-empty list"),
        ({"entities": "pedestrian"}, "non-empty list"),
    ],
)
def test_a_selection_that_cannot_be_honoured_is_refused(tmp_path, config, match):
    with pytest.raises(PluginError, match=match):
        Engine(load_config_from_dict(_world(tmp_path, [{"clearance_monitor": config}]))).setup()
