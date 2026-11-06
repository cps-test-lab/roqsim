"""The example sliding door: a machine with open and close programs that gives up on an obstruction;
and the scenario that opens it for whoever comes near and closes it after a delay."""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

DOOR = {
    "sliding_door": {"prefix": "d1_", "pose": {"position": {"x": 0.0, "y": 1.0}}, "speed": 0.6},
    "name": "door_1",
}
APPROACH = {
    "zone_trigger": {
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
        "size": [1.5, 1.0, 1.0],
        "entities": ["walker_*", "robot_*"],
    },
    "name": "door_approach",
}
SCENE = {"roqsim_assets.scenarios.door_for_who_comes:DoorForWhoComes": {"close_delay": 2.0}}


def _engine(*components):
    engine = Engine(load_config_from_dict({"sim": {}, "components": list(components)}))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _mover(name, y=0.0, speed=1.0):
    """A block driving along x from -4 to 4, past the door's approach."""
    return {
        "moving_box": {
            "prefix": f"{name}_",
            "size": [0.3, 0.3, 0.3],
            "pose": {"position": {"x": -4.0, "y": y}},
            "speed": speed,
            "waypoints": [[4.0, y]],
            "loop": False,
            "collide": False,
        },
        "name": name,
    }


def _finish(engine, handle, seq, limit=15.0):
    t0 = engine.ctx.sim_time
    while not (handle.status().seq == seq and handle.status().finished):
        assert engine.ctx.sim_time - t0 < limit, handle.status()
        engine.step()
    return handle.status()


def _leaf(engine):
    m = engine.ctx.model
    return float(engine.ctx.data.qpos[m.jnt_qposadr[m.joint("d1_leaf_slide").id]])


def _run(engine, seconds):
    end = engine.ctx.sim_time + seconds
    while engine.ctx.sim_time < end - 1e-9:
        engine.step()


def test_open_and_close_run_at_the_doors_speed_and_end_there():
    engine = _engine(DOOR)
    try:
        door = engine.ctx.blackboard.require("machine:door_1")
        assert engine.ctx.entities.get("door_1").kind == "door"
        t0 = engine.ctx.sim_time
        assert _finish(engine, door, door.start("open")).error == ""
        assert engine.ctx.sim_time - t0 == pytest.approx(1.2 / 0.6, abs=0.15)
        assert _leaf(engine) == pytest.approx(1.2, abs=0.003)
        assert _finish(engine, door, door.start("close")).error == ""
        assert _leaf(engine) == pytest.approx(0.0, abs=0.003)
    finally:
        engine.shutdown()


def test_a_box_in_the_way_makes_close_fail_and_the_leaf_holds():
    obstacle = {
        "box": {
            "prefix": "obs_",
            "pose": {"position": {"x": -0.2, "y": 1.08}},
            "size": [0.2, 0.2, 0.6],
            "motion": "static",
        },
        "name": "obstacle",
    }
    # The door starts open (its home); the leaf closes back toward -x and the box stands in its path.
    engine = _engine(
        {**DOOR, "sliding_door": {**DOOR["sliding_door"], "home": {"leaf": 1.2}}}, obstacle
    )
    try:
        door = engine.ctx.blackboard.require("machine:door_1")
        status = _finish(engine, door, door.start("close"))
        assert status.error.startswith("stalled at")
        held = _leaf(engine)
        assert 0.2 < held < 1.2
        _run(engine, 1.0)
        assert _leaf(engine) == pytest.approx(held, abs=0.01), "it stops pressing"
        stalled = [e for e in engine.ctx.events.all() if e["event"] == "stalled"]
        assert len(stalled) == 1
    finally:
        engine.shutdown()


def test_the_scene_opens_the_door_before_a_robot_arrives_and_closes_it_after_it_left():
    engine = _engine(DOOR, APPROACH, _mover("robot_1"), {**SCENE, "name": "doorman"})
    try:
        _run(engine, 10.0)
        zone = [e for e in engine.ctx.events.all() if e["source"] == "door_approach"]
        door = [e for e in engine.ctx.events.all() if e["source"] == "door_1"]
        entered = next(e["t"] for e in zone if e["event"] == "entered")
        left = next(e["t"] for e in zone if e["event"] == "left")
        starts = [(e["t"], e["program"]) for e in door if e["event"] == "program_start"]
        assert [p for _, p in starts] == ["open", "close"]
        # Opens as the robot enters the approach (one step to apply the start), which is 1.5 m
        # before it reaches the door's line at 1 m/s.
        assert starts[0][0] == pytest.approx(entered, abs=0.01)
        assert starts[1][0] == pytest.approx(left + 2.0, abs=0.01)
        assert _leaf(engine) == pytest.approx(0.0, abs=0.003)
    finally:
        engine.shutdown()


def test_an_entity_the_zone_does_not_count_opens_nothing():
    engine = _engine(DOOR, APPROACH, _mover("cart_1"), {**SCENE, "name": "doorman"})
    try:
        _run(engine, 10.0)
        assert not [e for e in engine.ctx.events.all() if e["source"] == "door_1"]
    finally:
        engine.shutdown()
