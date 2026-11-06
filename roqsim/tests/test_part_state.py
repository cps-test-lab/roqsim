# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""part_state: an entity's states start hidden, are shown and hidden whole, change nothing they
should not, say each change as an event, and a recording replays them.

The part here is a test fixture with the two kinds of state a line needs: a visual one (a label) and
one that changes the shape (a flat blank against a folded box, each with its own collision).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import yaml

from roqsim.config import load_config_from_dict
from roqsim.context import Entity
from roqsim.endpoint import _ATTR
from roqsim.engine import Engine
from roqsim.frames import frame_pose, resolve_frame
from roqsim.plugins.part_state import PartStateError, PartStatePlugin
from roqsim.presence import ABSENT_GEOM_GROUP, set_present
from roqsim.recording import open_recording
from roqsim.runner import run

PART = """<mujoco model="carton">
  <worldbody>
    <body name="carton">
      <inertial pos="0 0 0.05" mass="0.6" diaginertia="0.01 0.01 0.01"/>
      <body name="look_blank">
        <geom name="blank" type="box" size="0.3 0.3 0.002" pos="0 0 0.002" rgba="0.6 0.5 0.3 1" mass="0"/>
      </body>
      <body name="look_folded">
        <geom name="folded" type="box" size="0.1 0.1 0.05" pos="0 0 0.05" rgba="0.6 0.5 0.3 1" mass="0"/>
      </body>
      <body name="look_label">
        <geom name="label" type="box" size="0.04 0.03 0.0005" pos="0 0 0.1005" rgba="1 1 1 1"
              contype="0" conaffinity="0" mass="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture
def part_file(tmp_path):
    path = tmp_path / "carton.xml"
    path.write_text(PART)
    return path


def _world(part_file, present=None, extra=()):
    look = {"part_state": {"shown": present} if present else {}}
    return {
        "sim": {},
        "components": [
            {
                "spawn_model": {
                    "model": str(part_file),
                    "prefix": "c1_",
                    "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                },
                "name": "c1",
            },
            look,
            *extra,
        ],
    }


def _engine(world):
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _hidden(model, geom: str) -> bool:
    g = model.geom(geom).id
    return model.geom_group[g] == ABSENT_GEOM_GROUP and model.geom_rgba[g][3] == 0


def test_looks_start_hidden_except_the_ones_named_present(part_file):
    engine = _engine(_world(part_file, present={"c1": ["blank"]}))
    try:
        m = engine.ctx.model
        looks = engine.ctx.blackboard.get("part_state")
        assert looks.of("c1") == ["blank", "folded", "label"]
        assert looks.shown("c1") == ["blank"]
        assert not _hidden(m, "c1_blank")
        assert _hidden(m, "c1_folded") and _hidden(m, "c1_label")
        assert engine.ctx.entities.get("c1.look_label").kind == "look"
    finally:
        engine.shutdown()


def test_a_look_that_changes_the_shape_changes_what_collides(part_file):
    """Folded: the box collides and the blank does not; a ray from above meets the box's lid."""
    engine = _engine(_world(part_file, present={"c1": ["blank"]}))
    try:
        m, d = engine.ctx.model, engine.ctx.data
        looks = engine.ctx.blackboard.get("part_state")
        assert m.geom_contype[m.geom("c1_folded").id] == 0
        looks.hide("c1", "blank")
        looks.show("c1", "folded")
        mujoco.mj_forward(m, d)
        assert m.geom_contype[m.geom("c1_folded").id] != 0
        assert m.geom_contype[m.geom("c1_blank").id] == 0
        hit = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(
            m, d, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, hit
        )
        assert m.geom(hit[0]).name == "c1_folded" and dist == pytest.approx(0.9, abs=1e-6)
    finally:
        engine.shutdown()


def test_a_look_changes_no_mass(part_file):
    engine = _engine(_world(part_file))
    try:
        m = engine.ctx.model
        before = m.body_subtreemass[m.body("c1_carton").id]
        engine.ctx.blackboard.get("part_state").show("c1", "label")
        assert m.body_subtreemass[m.body("c1_carton").id] == before == pytest.approx(0.6)
    finally:
        engine.shutdown()


def test_a_reset_puts_the_configured_looks_back(part_file):
    engine = _engine(_world(part_file, present={"c1": ["blank"]}))
    try:
        looks = engine.ctx.blackboard.get("part_state")
        looks.show("c1", "label")
        looks.hide("c1", "blank")
        engine.reset()
        assert looks.shown("c1") == ["blank"]
    finally:
        engine.shutdown()


def test_a_missing_part_or_look_is_an_error(part_file):
    engine = _engine(_world(part_file))
    try:
        looks = engine.ctx.blackboard.get("part_state")
        with pytest.raises(PartStateError, match="no state 'film'"):
            looks.show("c1", "film")
        with pytest.raises(PartStateError, match="has no states"):
            looks.show("c2", "label")
    finally:
        engine.shutdown()
    with pytest.raises(PartStateError, match="no state 'lid'"):
        _engine(_world(part_file, present={"c1": ["lid"]}))


_LABELLER = """
from roqsim.plugin import Plugin

class Labeller(Plugin):
    def pre_step(self, ctx):
        looks = ctx.blackboard.get("part_state")
        if ctx.data.time >= 0.2 and not looks.is_shown("c1", "label"):
            looks.show("c1", "label")
"""


#: Deletes the part at 0.1 s, hides its shown state at 0.2 s, and brings it back at 0.4 s.
_DELETER = """
from roqsim.plugin import Plugin
from roqsim.presence import set_present

class Deleter(Plugin):
    def pre_step(self, ctx):
        t, owner = ctx.data.time, ctx.entities.get("c1")
        states = ctx.blackboard.get("part_state")
        if 0.1 <= t < 0.4 and owner.present:
            set_present(ctx, owner, False)
        if t >= 0.2 and states.is_shown("c1", "blank"):
            states.hide("c1", "blank")
        if t >= 0.4 and not owner.present:
            set_present(ctx, owner, True)
"""


def test_a_recording_replays_the_look_from_the_step_it_changed(part_file, tmp_path):
    (tmp_path / "labeller.py").write_text(_LABELLER)
    world = _world(
        part_file, extra=[{f"{tmp_path / 'labeller.py'}:Labeller": {}, "name": "labeller"}]
    )
    (tmp_path / "w.yaml").write_text(yaml.safe_dump(world))
    out = tmp_path / "run.mcap"
    run(
        str(tmp_path / "w.yaml"),
        headless=True,
        pacing="asap",
        seconds=0.5,
        record=str(out),
        capture_fps=25,
        seed=1,
    )
    with open_recording(out) as rec:
        for when, shown in ((0.1, False), (0.4, True), (0.05, False)):
            rec.at(when)
            model, ctx = rec.build()
            assert ctx.entities.get("c1.look_label").present is shown, when
            assert _hidden(model, "c1_label") is not shown, when


#: A strip whose `bent` look moves its centre of mass and its principal inertia.
STRIP = """<mujoco model="strip">
  <custom>
    <numeric name="look_bent_inertia" data="0.03 0 0.02  1 0 0 0  0.002 0.003 0.004"/>
  </custom>
  <worldbody>
    <body name="strip">
      <inertial pos="0 0 0" mass="0.19" diaginertia="0.0001 0.0014 0.0015"/>
      <body name="look_flat"><geom type="box" size="0.15 0.02 0.001" mass="0"/></body>
      <body name="look_bent"><geom type="box" size="0.05 0.02 0.001" mass="0"/></body>
    </body>
  </worldbody>
</mujoco>"""


def test_a_look_that_names_an_inertia_moves_the_parts_mass_while_shown(tmp_path):
    (tmp_path / "strip.xml").write_text(STRIP)
    world = {
        "sim": {"gravity": [0, 0, 0]},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "strip.xml"),
                    "prefix": "s1_",
                    "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
                },
                "name": "s1",
            },
            {"part_state": {"shown": {"s1": ["flat"]}}},
        ],
    }
    engine = _engine(world)
    try:
        m = engine.ctx.model
        root = m.body("s1_strip").id
        looks = engine.ctx.blackboard.get("part_state")
        mass = float(m.body_mass[root])
        assert m.body_ipos[root] == pytest.approx([0, 0, 0])
        looks.hide("s1", "flat")
        looks.show("s1", "bent")
        assert m.body_ipos[root] == pytest.approx([0.03, 0, 0.02])
        assert m.body_inertia[root] == pytest.approx([0.002, 0.003, 0.004])
        assert float(m.body_mass[root]) == mass, "a look moves the mass, it does not add any"
        looks.hide("s1", "bent")
        assert m.body_ipos[root] == pytest.approx([0, 0, 0])
        assert m.body_inertia[root] == pytest.approx([0.0001, 0.0014, 0.0015])
        looks.show("s1", "bent")
        engine.reset()
        assert looks.shown("s1") == ["flat"]
        assert m.body_ipos[root] == pytest.approx([0, 0, 0]), "a reset puts the part's own back"
    finally:
        engine.shutdown()


def test_a_pattern_shows_on_every_entity_it_matches_and_one_matching_nothing_is_refused(part_file):
    world = _world(part_file, present={"c*": ["blank"]})
    world["components"].insert(
        1,
        {
            "spawn_model": {
                "model": str(part_file),
                "prefix": "c2_",
                "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}},
            },
            "name": "c2",
        },
    )
    engine = _engine(world)
    try:
        looks = engine.ctx.blackboard.get("part_state")
        assert looks.shown("c1") == looks.shown("c2") == ["blank"]
    finally:
        engine.shutdown()
    with pytest.raises(PartStateError, match="pattern 'disc_\\*' matches no entity with states"):
        _engine(_world(part_file, present={"disc_*": ["blank"]}))


def test_set_shows_one_state_of_an_exclusive_group_and_hides_the_others(part_file):
    world = _world(part_file, present={"c1": ["blank", "label"]})
    world["components"][1]["part_state"]["exclusive"] = {"c1": [["blank", "folded"]]}
    engine = _engine(world)
    try:
        looks = engine.ctx.blackboard.get("part_state")
        assert looks.set("c1", "folded")
        assert looks.shown("c1") == ["folded", "label"], "a state outside the group is left alone"
        assert not looks.set("c1", "folded")
    finally:
        engine.shutdown()
    bad = _world(part_file)
    bad["components"][1]["part_state"]["exclusive"] = {"c1": [["blank", "lid"]]}
    with pytest.raises(PartStateError, match="no state 'lid'"):
        _engine(bad)


def test_each_change_is_an_event_and_a_change_that_changes_nothing_is_none(part_file):
    engine = _engine(_world(part_file, present={"c1": ["blank"]}))
    try:
        looks = engine.ctx.blackboard.get("part_state")
        looks.show("c1", "label")
        looks.show("c1", "label")
        looks.hide("c1", "blank")
        events = [
            (e["event"], e["entity"], e["look"])
            for e in engine.ctx.events.all()
            if e["source"] == "part_state"
        ]
        assert events == [("shown", "c1", "label"), ("hidden", "c1", "blank")]
    finally:
        engine.shutdown()


def test_the_commands_reach_the_handle_by_name(part_file):
    engine = _engine(_world(part_file))
    try:
        show = engine.ctx.interface.find("part_state", "show")
        future = show.write({"entity": "c1", "look": "label"})
        engine.step()
        assert future.result(1.0) is True
        assert engine.ctx.blackboard.get("part_state").shown("c1") == ["label"]
    finally:
        engine.shutdown()


def test_a_bad_config_is_refused_when_the_world_loads(part_file):
    world = _world(part_file)
    world["components"][1]["part_state"] = {
        "shown": {"c1": "label"},
        "exclusive": {"c1": [["one"]]},
    }
    with pytest.raises(Exception, match="'shown' maps an entity name pattern") as exc:
        _engine(world)
    assert "two or more state names" in str(exc.value)


#: A part whose label state carries mass, as a model that forgot mass="0" on it has.
HEAVY_LABEL = """<mujoco model="carton">
  <worldbody>
    <body name="carton">
      <geom type="box" size="0.1 0.1 0.05" mass="0.2"/>
      <body name="look_label">
        <geom type="box" size="0.04 0.03 0.0005" pos="0 0 0.0505" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def test_a_state_body_with_mass_is_refused_naming_it(tmp_path):
    path = tmp_path / "heavy.xml"
    path.write_text(HEAVY_LABEL)
    with pytest.raises(PartStateError, match=r"state body 'c1_look_label' of 'c1' has mass .* kg"):
        _engine(_world(path))


def test_showing_a_second_state_that_names_an_inertia_is_refused_before_anything_changes(tmp_path):
    two = STRIP.replace(
        '<numeric name="look_bent_inertia"',
        '<numeric name="look_flat_inertia" data="0 0 0  1 0 0 0  0.0001 0.0014 0.0015"/>\n'
        '    <numeric name="look_bent_inertia"',
    )
    (tmp_path / "strip.xml").write_text(two)
    world = {
        "sim": {"gravity": [0, 0, 0]},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "strip.xml"),
                    "prefix": "s1_",
                    "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
                },
                "name": "s1",
            },
            {"part_state": {"shown": {"s1": ["flat"]}}},
        ],
    }
    engine = _engine(world)
    try:
        looks = engine.ctx.blackboard.get("part_state")
        events = len(engine.ctx.events.all())
        with pytest.raises(PartStateError, match="'s1' shows \\['flat'\\], which names an inertia"):
            looks.show("s1", "bent")
        assert looks.shown("s1") == ["flat"], "nothing changed"
        assert len(engine.ctx.events.all()) == events
    finally:
        engine.shutdown()


# -- a state's inertia, as the physics sees it -----------------------------------------------------
#: A 1 kg arm on a hinge about y at its origin, its mass at the hinge -- MuJoCo compiles that as
#: sameframe -- whose `bent` state puts the mass 0.1 m out along x, where gravity turns it.
HINGED = """<mujoco model="arm">
  <custom>
    <numeric name="look_bent_inertia" data="0.1 0 0  1 0 0 0  0.001 0.001 0.001"/>
  </custom>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
      <body name="look_flat"><geom type="box" size="0.1 0.02 0.001" contype="0" conaffinity="0" mass="0"/></body>
      <body name="look_bent"><geom type="box" size="0.05 0.02 0.001" contype="0" conaffinity="0" mass="0"/></body>
    </body>
  </worldbody>
</mujoco>"""


def _hinged(tmp_path, gravity=True):
    (tmp_path / "arm.xml").write_text(HINGED)
    world = {
        "sim": {} if gravity else {"gravity": [0, 0, 0]},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "arm.xml"),
                    "prefix": "a_",
                    "motion": "static",
                    "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                },
                "name": "arm",
            },
            {"part_state": {"shown": {"arm": ["flat"]}}},
        ],
    }
    return _engine(world)


def _hinge(engine):
    m = engine.ctx.model
    return float(engine.ctx.data.qpos[m.jnt_qposadr[m.joint("a_hinge").id]])


def test_a_state_that_names_an_inertia_moves_the_centre_of_mass_the_physics_uses(tmp_path):
    engine = _hinged(tmp_path)
    try:
        m, d = engine.ctx.model, engine.ctx.data
        root = m.body("a_arm").id
        assert m.body_sameframe[root] != 0, "compiled with its mass at its frame"
        for _ in range(200):
            engine.step()
        assert _hinge(engine) == pytest.approx(0.0, abs=1e-6), (
            "no torque with the mass at the hinge"
        )
        looks = engine.ctx.blackboard.get("part_state")
        looks.hide("arm", "flat")
        looks.show("arm", "bent")
        mujoco.mj_forward(m, d)
        assert d.xipos[root] - d.xpos[root] == pytest.approx([0.1, 0.0, 0.0], abs=1e-9)
        for _ in range(200):
            engine.step()
        assert abs(_hinge(engine)) > 0.1, "gravity turns the arm about the hinge"
        looks.hide("arm", "bent")
        mujoco.mj_forward(m, d)
        assert d.xipos[root] - d.xpos[root] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    finally:
        engine.shutdown()


def test_a_states_inertia_turns_the_principal_axes_too(tmp_path):
    # Its mass off the frame but its axes the body's: MuJoCo compiles that as BODYROT, and would
    # ignore a state's iquat.
    rotated = HINGED.replace('<inertial pos="0 0 0"', '<inertial pos="0.01 0 0"').replace(
        'data="0.1 0 0  1 0 0 0  0.001 0.001 0.001"',
        'data="0.1 0 0  0.70710678 0 0 0.70710678  0.001 0.002 0.003"',
    )
    (tmp_path / "arm.xml").write_text(rotated)
    engine = _engine(
        {
            "sim": {"gravity": [0, 0, 0]},
            "components": [
                {
                    "spawn_model": {
                        "model": str(tmp_path / "arm.xml"),
                        "prefix": "a_",
                        "motion": "static",
                        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                    },
                    "name": "arm",
                },
                {"part_state": {}},
            ],
        }
    )
    try:
        m, d = engine.ctx.model, engine.ctx.data
        root = m.body("a_arm").id
        engine.ctx.blackboard.get("part_state").show("arm", "bent")
        mujoco.mj_forward(m, d)
        # 90 degrees about z: the inertial x axis lies along the world's y.
        assert d.ximat[root].reshape(3, 3)[:, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    finally:
        engine.shutdown()


# -- what part_state refuses -----------------------------------------------------------------------
def test_an_entity_with_states_that_starts_absent_comes_back_showing_only_what_is_shown(part_file):
    world = _world(part_file, present={"c1": ["blank"]})
    world["components"][0]["spawn_model"]["present"] = False
    engine = _engine(world)
    try:
        assert _hidden(engine.ctx.model, "c1_blank"), "the owner is absent"
        assert set_present(engine.ctx, engine.ctx.entities.get("c1"), True)
        assert not _hidden(engine.ctx.model, "c1_blank")
        assert _hidden(engine.ctx.model, "c1_label")
    finally:
        engine.shutdown()


def test_set_is_refused_before_anything_changes(tmp_path):
    two = STRIP.replace(
        '<numeric name="look_bent_inertia"',
        '<numeric name="look_flat_inertia" data="0 0 0  1 0 0 0  0.0001 0.0014 0.0015"/>\n'
        '    <numeric name="look_bent_inertia"',
    ).replace(
        '<body name="look_bent">',
        '<body name="look_plain"><geom type="box" size="0.15 0.02 0.001" mass="0"/></body>\n'
        '      <body name="look_bent">',
    )
    (tmp_path / "strip.xml").write_text(two)
    world = {
        "sim": {"gravity": [0, 0, 0]},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp_path / "strip.xml"),
                    "prefix": "s1_",
                    "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
                },
                "name": "s1",
            },
            {
                "part_state": {
                    "shown": {"s1": ["flat", "plain"]},
                    "exclusive": {"s1": [["plain", "bent"]]},
                }
            },
        ],
    }
    engine = _engine(world)
    try:
        looks = engine.ctx.blackboard.get("part_state")
        with pytest.raises(PartStateError, match="names an inertia"):
            looks.set("s1", "bent")
        assert looks.shown("s1") == ["flat", "plain"], "nothing changed"
    finally:
        engine.shutdown()


def test_an_exclusive_group_has_at_most_one_state_shown(part_file):
    world = _world(part_file, present={"c1": ["blank"]})
    world["components"][1]["part_state"]["exclusive"] = {"c1": [["blank", "folded"]]}
    engine = _engine(world)
    try:
        looks = engine.ctx.blackboard.get("part_state")
        with pytest.raises(PartStateError, match=r"shows \['blank'\], of the exclusive group"):
            looks.show("c1", "folded")
        assert looks.shown("c1") == ["blank"]
        assert looks.show("c1", "blank") is False, "showing the one shown is no conflict"
    finally:
        engine.shutdown()
    world = _world(part_file, present={"c1": ["blank", "folded"]})
    world["components"][1]["part_state"]["exclusive"] = {"c1": [["blank", "folded"]]}
    with pytest.raises(PartStateError, match=r"'shown' shows \['blank', 'folded'\]"):
        _engine(world)


def test_a_state_nested_inside_another_is_refused(tmp_path):
    nested = PART.replace(
        '<body name="look_label">',
        '<body name="look_label">\n        <body name="look_stamp"><geom type="box" size="0.01 0.01 0.001" mass="0"/></body>',
    )
    path = tmp_path / "nested.xml"
    path.write_text(nested)
    with pytest.raises(
        PartStateError, match="'c1_look_stamp' of 'c1' is inside the state body 'c1_look_label'"
    ):
        _engine(_world(path))


# -- a state is not an entity of the control plane -------------------------------------------------
def test_the_control_plane_neither_lists_nor_changes_a_state(part_file):
    from roqsim.entity_control import EntityRefused, set_presence, set_state

    engine = _engine(_world(part_file))
    try:
        entities = engine.ctx.entities
        assert "c1.look_label" in entities.names(), "registered, so a recording carries it"
        assert entities.names(listed_only=True) == ["c1"]
        assert engine.ctx.interface.find("sim", "entities/c1.look_label/pose") is None
        assert engine.ctx.interface.find("sim", "entities/c1/pose") is not None
        with pytest.raises(EntityRefused, match="changed only by the plugin that keeps it"):
            set_presence(engine.ctx, "c1.look_label", True)
        with pytest.raises(EntityRefused, match="changed only by the plugin"):
            set_state(engine.ctx, "c1.look_label", [0, 0, 1], [1, 0, 0, 0])
        assert engine.ctx.blackboard.get("part_state").shown("c1") == []
    finally:
        engine.shutdown()


def test_a_state_is_no_frame_of_its_own_and_its_bodies_stay_frames_of_the_entity(part_file):
    engine = _engine(_world(part_file, present={"c1": ["blank"]}))
    try:
        assert resolve_frame(engine.ctx, "c1/look_blank").path == "c1/look_blank"
        assert frame_pose(engine.ctx, "c1/look_blank") is not None, "readable while shown"
        with pytest.raises(Exception, match="c1.look_blank"):
            resolve_frame(engine.ctx, "c1.look_blank")
    finally:
        engine.shutdown()


def test_the_commands_are_not_on_ros_where_a_bool_could_not_carry_them():
    for name in ("show", "hide", "set"):
        spec = getattr(getattr(PartStatePlugin, name), _ATTR)
        assert "ros2" in repr(spec) and "'ros2': None" in repr(spec)


def test_two_bodies_naming_one_state_are_refused(tmp_path):
    bodies = ("arm_look_x", "leg_look_x")
    geoms = "".join(f'<body name="{b}"><geom size="0.01" mass="0"/></body>' for b in bodies)
    path = tmp_path / "part.xml"
    path.write_text(
        f"""<mujoco><worldbody><body name="part"><inertial pos="0 0 0" mass="1"
        diaginertia="1 1 1"/>{geoms}</body></worldbody></mujoco>"""
    )
    with pytest.raises(PartStateError, match="both name the state 'x'"):
        _engine(_world(path))


def test_a_state_body_two_entities_hold_is_refused(part_file):
    engine = _engine(_world(part_file))
    try:
        engine.ctx.entities.add(Entity("c1_again", "object", body="c1_carton"))
        with pytest.raises(PartStateError, match="under both 'c1' and 'c1_again'"):
            engine.reset()
    finally:
        engine.shutdown()


def test_a_state_in_two_exclusive_groups_is_refused(part_file):
    world = _world(part_file)
    world["components"][1]["part_state"] = {
        "exclusive": {"c1": [["blank", "label"], ["blank", "folded"]]}
    }
    with pytest.raises(PartStateError, match="in two exclusive groups"):
        _engine(world)


@pytest.mark.parametrize(
    "data",
    [
        "0 0 0  0 0 0 0  1 1 1",
        "0 0 0  1 0 0 0  0 0 0",
        "0 0 0  1 0 0 0  -1 1 1",
        "0 0 0  1 0 0 0  1 1 5",
    ],
)
def test_a_state_inertia_that_is_none_is_refused_naming_it(tmp_path, data):
    path = tmp_path / "strip.xml"
    path.write_text(STRIP.replace("0.03 0 0.02  1 0 0 0  0.002 0.003 0.004", data))
    with pytest.raises(PartStateError, match="look_bent_inertia is no inertia"):
        _engine(_world(path))


def test_a_state_changed_while_its_owner_is_absent_is_as_left_when_it_returns(part_file):
    engine = _engine(_world(part_file, present={"c1": ["blank"]}))
    try:
        ctx, owner = engine.ctx, engine.ctx.entities.get("c1")
        states = ctx.blackboard.require("part_state")
        set_present(ctx, owner, False)
        states.hide("c1", "blank")
        states.show("c1", "label")
        assert _hidden(ctx.model, "c1_label"), "no ghost on an absent part"
        set_present(ctx, owner, True)
        assert _hidden(ctx.model, "c1_blank") and not _hidden(ctx.model, "c1_label")
        assert states.shown("c1") == ["label"]
        # What a state's geom was is not lost, and a reset puts all of it back.
        set_present(ctx, owner, False)
        engine.reset()
        assert not _hidden(ctx.model, "c1_blank") and _hidden(ctx.model, "c1_label")
        assert ctx.model.geom_contype[ctx.model.geom("c1_blank").id] == 1
    finally:
        engine.shutdown()


def test_a_reset_is_the_same_with_the_plugin_listed_before_the_model(part_file):
    world = _world(part_file, present={"c1": ["blank"]})
    world["components"] = [world["components"][1], world["components"][0]]
    engine = _engine(world)
    try:
        ctx = engine.ctx
        set_present(ctx, ctx.entities.get("c1"), False)
        engine.reset()
        assert not _hidden(ctx.model, "c1_blank") and _hidden(ctx.model, "c1_label")
        assert ctx.blackboard.require("part_state").shown("c1") == ["blank"]
    finally:
        engine.shutdown()


def test_a_replay_of_a_part_deleted_and_changed_is_right_in_any_order(part_file, tmp_path):
    (tmp_path / "deleter.py").write_text(_DELETER)
    world = _world(
        part_file,
        present={"c1": ["blank"]},
        extra=[{f"{tmp_path / 'deleter.py'}:Deleter": {}, "name": "deleter"}],
    )
    (tmp_path / "w.yaml").write_text(yaml.safe_dump(world))
    out = tmp_path / "run.mcap"
    run(
        str(tmp_path / "w.yaml"),
        headless=True,
        pacing="asap",
        seconds=0.5,
        record=str(out),
        capture_fps=25,
        seed=1,
    )
    with open_recording(out) as rec:
        for when, blank in ((0.3, True), (0.05, False), (0.15, True), (0.45, True), (0.05, False)):
            rec.at(when)
            model, _ctx = rec.build()
            assert bool(_hidden(model, "c1_blank")) is blank, when


def test_a_second_entry_is_refused():
    world = {
        "sim": {},
        "components": [{"part_state": {}, "name": "a"}, {"part_state": {}, "name": "b"}],
    }
    with pytest.raises(PartStateError, match="second entry"):
        _engine(world)


def test_an_inertia_for_a_state_the_model_lacks_is_refused(tmp_path):
    path = tmp_path / "strip.xml"
    path.write_text(STRIP.replace('name="look_bent_inertia"', 'name="look_bnet_inertia"'))
    with pytest.raises(PartStateError, match="look_bnet"):
        _engine(_world(path))
