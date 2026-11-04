# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""zone_trigger: presence in a box, reported as events and on a handle, and nothing decided."""

from __future__ import annotations

import logging

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


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
    base = {"pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}}, "size": [0.2, 0.3, 0.2]}
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
    with pytest.raises(Exception, match="'size' is three positive half-extents") as exc:
        _engine(_zone(size=[0.1, 0.1], point="middle", dwell=-1))
    assert "'point' is com or origin" in str(exc.value) and "'dwell' is seconds" in str(exc.value)


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
