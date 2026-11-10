# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""zone_trigger: presence in a box, reported as events and on a handle, and nothing decided."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from roqsim.config import load_config_from_dict
from roqsim.endpoint import topic_of
from roqsim.engine import Engine
from roqsim.plugins.zone_trigger import ZoneHandle, ZoneTriggerPlugin
from roqsim.presence import set_present
from roqsim.scenario import ScenarioError


def _engine(*components):
    engine = Engine(
        load_config_from_dict({"sim": {"gravity": [0, 0, 0]}, "components": list(components)})
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _mover(name, start, velocity, **extra):
    return {"machine_fixtures:Mover": {"start": start, "velocity": velocity, **extra}, "name": name}


def _zone(name="gate", **config):
    base = {"pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}}, "size": [0.4, 0.6, 0.4]}
    return {"zone_trigger": {**base, **config}, "name": name}


def _run(engine, seconds):
    end = engine.ctx.sim_time + seconds
    while engine.ctx.sim_time < end - 1e-9:
        engine.step()


def _events(engine, source="gate"):
    return [e for e in engine.ctx.events.all() if e["source"] == source]


def test_an_entity_passing_through_enters_dwells_and_leaves_with_its_times():
    # 1 m/s from x = -1: inside while its centre is within x in [-0.2, 0.2], 0.8 s to 1.2 s.
    engine = _engine(_mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]), _zone(dwell=0.25))
    try:
        zone = engine.ctx.blackboard.require("zone:gate")
        occupied = engine.ctx.interface.find("gate", "occupied")
        _run(engine, 1.0)
        assert zone.inside() == ["box_1"] and zone.occupied and occupied.read()
        assert zone.entered_at("box_1") == pytest.approx(0.8, abs=0.01)
        _run(engine, 0.5)
        assert zone.inside() == [] and not occupied.read()
        events = _events(engine)
        assert [e["event"] for e in events] == ["entered", "dwelled", "left"]
        assert all(e["entity"] == "box_1" for e in events)
        assert events[1]["t"] - events[0]["t"] == pytest.approx(0.25, abs=0.01)
        assert events[2]["inside_s"] == pytest.approx(0.4, abs=0.01)
        assert zone.counts.entered == zone.counts.left == 1
    finally:
        engine.shutdown()


def test_an_entity_passing_beside_the_zone_is_never_inside():
    engine = _engine(_mover("box_1", [-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]), _zone())
    try:
        _run(engine, 2.0)
        assert _events(engine) == []
    finally:
        engine.shutdown()


def test_only_the_entities_its_patterns_name_count():
    engine = _engine(
        _mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]),
        _mover("walker_1", [-1.0, 0.1, 0.5], [1.0, 0.0, 0.0]),
        _zone(entities=["walker_*"]),
    )
    try:
        _run(engine, 1.0)
        assert engine.ctx.blackboard.require("zone:gate").inside() == ["walker_1"]
    finally:
        engine.shutdown()


def test_point_com_reads_the_centre_of_mass_and_origin_the_frame():
    # Its mass sits 0.4 m ahead of its origin: at t = 0.5 the mass is at x = -0.1, the origin at -0.5.
    mover = _mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0], com_offset=0.4)
    engine = _engine(mover, _zone("by_mass", point="com"), _zone("by_frame", point="origin"))
    try:
        _run(engine, 0.5)
        assert engine.ctx.blackboard.require("zone:by_mass").inside() == ["box_1"]
        assert engine.ctx.blackboard.require("zone:by_frame").inside() == []
    finally:
        engine.shutdown()


def test_a_zone_on_a_body_moves_with_it():
    # The zone rides 1 m ahead of a carrier moving at 1 m/s toward a block standing at x = 1.5.
    engine = _engine(
        _mover("carrier", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]),
        _mover("block", [1.5, 0.0, 0.5], [0.0, 0.0, 0.0]),
        _zone(
            body="carrier", pose={"position": {"x": 1.0, "y": 0.0, "z": 0.0}}, entities=["block"]
        ),
    )
    try:
        zone = engine.ctx.blackboard.require("zone:gate")
        _run(engine, 1.2)
        assert zone.inside() == []
        _run(engine, 0.2)
        assert zone.inside() == ["block"]
    finally:
        engine.shutdown()


def test_a_body_the_model_lacks_is_refused_and_a_pattern_matching_nothing_is_warned(caplog):
    with pytest.raises(RuntimeError, match="zone_trigger gate: no body 'trolley' to ride on"):
        _engine(_zone(body="trolley"))
    with caplog.at_level(logging.WARNING, logger="roqsim.plugins.zone_trigger"):
        engine = _engine(_zone(entities=["forklift_*"]))
        engine.shutdown()
    assert "'entities' pattern 'forklift_*' matches no entity" in caplog.text


def test_a_bad_config_is_refused_when_the_world_loads():
    with pytest.raises(Exception, match="'size' is three positive full extents") as exc:
        _engine(_zone(size=[0.1, 0.1], point="middle", dwell=-1))
    assert "'point' is com or origin" in str(exc.value) and "'dwell' is seconds" in str(exc.value)


@pytest.mark.parametrize("dwell", [float("inf"), float("nan")])
def test_a_dwell_that_is_no_number_of_seconds_is_refused(dwell):
    with pytest.raises(Exception, match="'dwell' is seconds"):
        _engine(_zone(dwell=dwell))


def test_a_reset_empties_the_zone_and_its_counts():
    engine = _engine(_mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]), _zone())
    try:
        zone = engine.ctx.blackboard.require("zone:gate")
        _run(engine, 1.0)
        engine.reset()
        assert zone.inside() == [] and zone.counts.entered == 0
    finally:
        engine.shutdown()


def test_a_scenario_waits_on_a_zone_for_an_entry_and_for_it_to_empty():
    engine = _engine(
        _mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]),
        _zone(dwell=0.1),
        {"scenario_fixtures:Watch": {}, "name": "scene"},
    )
    try:
        _run(engine, 2.0)
        scene = {e["event"]: e for e in _events(engine, "scene")}
        assert scene["saw"]["who"] == "box_1" and scene["saw"]["t"] == pytest.approx(0.8, abs=0.01)
        assert scene["dwelt"]["t"] == pytest.approx(0.9, abs=0.01)
        assert scene["empty"]["t"] == pytest.approx(1.2, abs=0.01)
    finally:
        engine.shutdown()


# -- waits that would hang ---------------------------------------------------------------------
def test_waiting_for_a_dwell_on_a_zone_that_reports_none_is_refused():
    engine = _engine(
        _mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]),
        _zone(),
        {"scenario_fixtures:WaitDwell": {}, "name": "scene"},
    )
    try:
        with pytest.raises(ScenarioError, match="zone gate has dwell 0, which reports no dwelling"):
            for _ in range(10):
                engine.step()
    finally:
        engine.shutdown()


def test_waiting_for_an_entry_times_out():
    engine = _engine(_zone(), {"scenario_fixtures:WaitEntry": {"timeout": 0.3}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="in zone gate not true after 0.3 s"):
            for _ in range(1000):
                engine.step()
        assert engine.ctx.sim_time == pytest.approx(0.3, abs=0.01)
    finally:
        engine.shutdown()


# -- the centre of mass a state moves --------------------------------------------------------------
#: A plate whose `shifted` state puts its centre of mass 0.3 m along x.
PLATE = """<mujoco model="plate">
  <custom>
    <numeric name="look_shifted_inertia" data="0.3 0 0  1 0 0 0  0.01 0.01 0.01"/>
  </custom>
  <worldbody>
    <body name="plate">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.4 0.1 0.01" contype="0" conaffinity="0" mass="0"/>
      <body name="look_shifted"><geom type="box" size="0.05 0.05 0.02" pos="0.3 0 0" contype="0" conaffinity="0" mass="0"/></body>
    </body>
  </worldbody>
</mujoco>"""


def test_point_com_follows_a_centre_of_mass_a_state_moves(tmp_path):
    (tmp_path / "plate.xml").write_text(PLATE)
    plate = {
        "spawn_model": {
            "model": str(tmp_path / "plate.xml"),
            "prefix": "p_",
            "motion": "static",
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
        },
        "name": "plate",
    }
    zone = _zone(pose={"position": {"x": 0.3, "y": 0.0, "z": 0.5}}, size=[0.1, 0.2, 0.2])
    engine = _engine(plate, {"part_state": {}}, zone)
    try:
        gate = engine.ctx.blackboard.require("zone:gate")
        _run(engine, 0.1)
        assert gate.inside() == []
        engine.ctx.blackboard.require("part_state").show("plate", "shifted")
        _run(engine, 0.1)
        assert gate.inside() == ["plate"]
    finally:
        engine.shutdown()


# -- the body a zone rides -----------------------------------------------------------------------------
def test_a_zone_counts_the_entity_whose_body_it_rides_unless_nested_under_it():
    carrier = _mover("carrier", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0])
    engine = _engine(
        carrier, _zone(body="carrier", pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    )
    try:
        _run(engine, 0.1)
        assert engine.ctx.blackboard.require("zone:gate").inside() == ["carrier"]
    finally:
        engine.shutdown()
    nested = {
        **_mover("carrier", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
        "components": [_zone(body="carrier", pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})],
    }
    engine = _engine(nested)
    try:
        _run(engine, 0.1)
        assert engine.ctx.blackboard.require("zone:carrier.gate").inside() == []
    finally:
        engine.shutdown()


def test_a_scene_busy_while_an_entity_passed_finds_it_with_since():
    engine = _engine(
        _mover("box_1", [-1.0, 0.0, 0.5], [1.0, 0.0, 0.0]),
        _zone(dwell=0.25),
        {"scenario_fixtures:LookedLate": {}, "name": "scene"},
    )
    try:
        _run(engine, 2.5)
        seen = {e["event"]: e["who"] for e in _events(engine, "scene") if "who" in e}
        assert seen == {"saw": "box_1", "dwelt": "box_1"}
        zone = engine.ctx.blackboard.require("zone:gate")
        assert zone.inside() == [] and zone.entered_since(1.0) == []
        engine.reset()
        assert zone.entered_since(0.0) == []
    finally:
        engine.shutdown()


def test_a_zone_under_an_absent_entity_reports_no_one():
    nested = {
        **_mover("carrier", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
        "components": [
            _zone(
                body="carrier", pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}, entities=["box"]
            )
        ],
    }
    engine = _engine(nested, _mover("box", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]))
    try:
        ctx = engine.ctx
        zone = ctx.blackboard.require("zone:carrier.gate")
        _run(engine, 0.05)
        assert zone.inside() == ["box"]
        set_present(ctx, ctx.entities.get("carrier"), False)
        _run(engine, 0.05)
        assert zone.inside() == [] and zone.counts.left == 1
        set_present(ctx, ctx.entities.get("carrier"), True)
        _run(engine, 0.05)
        assert zone.inside() == ["box"]
    finally:
        engine.shutdown()


def test_who_is_inside_is_on_the_occupancy_endpoint():
    engine = _engine(_mover("box_1", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]), _zone())
    try:
        _run(engine, 0.05)
        occupancy = engine.ctx.interface.find("gate", "occupancy").read()
        assert occupancy.inside == ["box_1"] and occupancy.count == 1
    finally:
        engine.shutdown()


def test_a_body_that_is_no_name_is_refused():
    with pytest.raises(Exception, match="'body' is a body's name"):
        _engine(_zone(body=5))


def test_since_adds_to_who_is_inside_now():
    engine = _engine(
        _mover("box", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
        _zone(dwell=0.05),
        {"scenario_fixtures:LookedNow": {}, "name": "scene"},
    )
    try:
        _run(engine, 0.3)
        seen = {e["event"]: e["who"] for e in _events(engine, "scene") if "who" in e}
        assert seen == {"saw": "box", "dwelt": "box"}
    finally:
        engine.shutdown()


def test_since_that_is_no_time_is_refused():
    engine = _engine(
        _zone(dwell=0.05), {"scenario_fixtures:LookedNow": {"since": float("nan")}, "name": "scene"}
    )
    try:
        with pytest.raises(ScenarioError, match="since is a sim time"):
            _run(engine, 0.3)
    finally:
        engine.shutdown()


def test_an_entity_that_comes_back_is_named_once():
    zone = ZoneHandle("gate")
    zone._entries = [(0.1, "box"), (0.2, "box"), (0.3, "other")]
    assert zone.entered_since(0.0) == ["box", "other"]


def test_two_zones_have_topics_of_their_own():
    engine = _engine(
        _mover("box", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
        _zone("one"),
        _zone("two", pose={"position": {"x": 9.0, "y": 0.0, "z": 0.5}}),
    )
    try:
        _run(engine, 0.05)
        interface = engine.ctx.interface
        one, two = interface.find("one", "occupied"), interface.find("two", "occupied")
        assert one.read() is True and two.read() is False
        assert topic_of(one, "ros2") == "one/occupied"
        assert topic_of(two, "ros2") == "two/occupied"
    finally:
        engine.shutdown()


def test_two_zones_under_one_entity_are_each_found():
    nested = {
        **_mover("carrier", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
        "components": [_zone("one"), _zone("two")],
    }
    engine = _engine(nested)
    try:
        one = engine.ctx.interface.find("carrier.one", "occupied")
        assert topic_of(one, "ros2") == "carrier/one/occupied", "a ROS name takes no dot"
        assert engine.ctx.interface.find("carrier.two", "occupied")
    finally:
        engine.shutdown()


def test_a_zone_is_silent_while_any_entity_around_it_is_absent():
    arm = {
        **_mover("arm", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
        "components": [_zone(entities=["box"])],
    }
    robot = {**_mover("robot", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]), "components": [arm]}
    engine = _engine(robot, _mover("box", [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]))
    try:
        ctx = engine.ctx
        zone = ctx.blackboard.require("zone:robot.arm.gate")
        _run(engine, 0.05)
        assert zone.inside() == ["box"]
        set_present(ctx, ctx.entities.get("robot.arm"), False)
        _run(engine, 0.05)
        assert zone.inside() == []
    finally:
        engine.shutdown()


def test_a_rotated_zone_is_a_rotated_box():
    quat = {"x": 0.0, "y": 0.0, "z": 0.7071068, "w": 0.7071068}
    engine = _engine(
        _mover("along", [0.0, 0.25, 0.5], [0.0, 0.0, 0.0]),
        _mover("across", [0.25, 0.0, 0.5], [0.0, 0.0, 0.0]),
        _zone(
            pose={"position": {"x": 0.0, "y": 0.0, "z": 0.5}, "orientation": quat},
            size=[0.6, 0.2, 0.4],
        ),
    )
    try:
        _run(engine, 0.05)
        assert engine.ctx.blackboard.require("zone:gate").inside() == ["along"]
    finally:
        engine.shutdown()


def test_an_empty_entities_list_is_refused():
    with pytest.raises(Exception, match="'entities' is a list"):
        _engine(_zone(entities=[]))


def test_a_nested_zone_is_named_once_under_its_entitys_namespace():
    zone = ZoneTriggerPlugin({"pose": {"position": {"x": 0, "y": 0, "z": 0}}, "size": [1, 1, 1]})
    zone.label, zone.entity = "gate", "robot"
    zone._ctx = SimpleNamespace(
        entities=SimpleNamespace(get=lambda _: SimpleNamespace(meta={"namespace": "robot"}))
    )
    assert zone.endpoint_namespace(zone._ctx) == "robot"
    assert zone._occupied_topic() == "gate/occupied", "not /robot/robot/gate/occupied"
