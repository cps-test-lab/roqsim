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
from roqsim.engine import Engine
from roqsim.plugins.part_state import PartStateError
from roqsim.presence import ABSENT_GEOM_GROUP
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
