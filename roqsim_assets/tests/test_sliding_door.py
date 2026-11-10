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
        # before it comes level with the door at 1 m/s.
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


def test_a_blocked_leaf_never_pushes_harder_than_the_force_cap():
    obstacle = {
        "box": {
            "prefix": "obs_",
            "pose": {"position": {"x": -0.2, "y": 1.08}},
            "size": [0.2, 0.2, 0.6],
            "motion": "static",
        },
        "name": "obstacle",
    }
    door_cfg = {**DOOR["sliding_door"], "home": {"leaf": 1.2}, "max_force": 80.0}
    engine = _engine({**DOOR, "sliding_door": door_cfg}, obstacle)
    try:
        door = engine.ctx.blackboard.require("machine:door_1")
        aid = engine.ctx.model.actuator("d1_leaf").id
        seq = door.start("close")
        peak = 0.0
        while not (door.status().seq == seq and door.status().finished):
            engine.step()
            peak = max(peak, abs(float(engine.ctx.data.actuator_force[aid])))
        assert 60.0 < peak <= 80.0 + 1e-6, "pressing on the box, at most at the cap"
    finally:
        engine.shutdown()


def test_the_door_stays_open_for_someone_who_came_back_before_it_closed():
    # robot_1 is in the approach from 2.5 s to 5.5 s; robot_2, 4 m behind it, from 6.5 s to 9.5 s:
    # one second after the zone emptied, inside the two-second close delay.
    engine = _engine(
        DOOR,
        APPROACH,
        _mover("robot_1"),
        {
            **_mover("robot_2"),
            "moving_box": {
                **_mover("robot_2")["moving_box"],
                "pose": {"position": {"x": -8.0, "y": 0.0}},
                "waypoints": [[4.0, 0.0]],
            },
        },
        {**SCENE, "name": "doorman"},
    )
    try:
        _run(engine, 14.0)
        zone = [e for e in engine.ctx.events.all() if e["source"] == "door_approach"]
        lefts = [e["t"] for e in zone if e["event"] == "left"]
        door = [e for e in engine.ctx.events.all() if e["source"] == "door_1"]
        starts = [(e["t"], e["program"]) for e in door if e["event"] == "program_start"]
        assert [p for _, p in starts] == ["open", "close"], "it did not close between the two"
        assert starts[1][0] == pytest.approx(lefts[-1] + 2.0, abs=0.01), "closed after the last"
        came_back = [e for e in engine.ctx.events.all() if e["event"] == "came_back"]
        assert [e["who"] for e in came_back] == ["robot_2"]
    finally:
        engine.shutdown()


def _door_starts(engine):
    return [
        (e["t"], e["program"])
        for e in engine.ctx.events.all()
        if e["source"] == "door_1" and e["event"] == "program_start"
    ]


def test_the_door_reopens_for_someone_who_comes_back_while_it_closes():
    # robot_1 is in the approach from 2.5 s to 5.5 s, so the door closes from 7.5 s for two
    # seconds; robot_2, 5.5 m behind it, enters at 8.0 s, while the leaf is still closing.
    late = _mover("robot_2")
    late["moving_box"] = {**late["moving_box"], "pose": {"position": {"x": -9.5, "y": 0.0}}}
    engine = _engine(DOOR, APPROACH, _mover("robot_1"), late, {**SCENE, "name": "doorman"})
    try:
        _run(engine, 8.5)
        assert [p for _, p in _door_starts(engine)] == ["open", "close", "open"], "it reopened"
        assert 0.3 < _leaf(engine) < 1.2, "from where the close had brought it"
        _run(engine, 8.0)
        events = engine.ctx.events.all()
        zone = [e for e in events if e["source"] == "door_approach"]
        entered = [e["t"] for e in zone if e["event"] == "entered"]
        lefts = [e["t"] for e in zone if e["event"] == "left"]
        starts = _door_starts(engine)
        assert [p for _, p in starts] == ["open", "close", "open", "close"]
        assert starts[2][0] == pytest.approx(entered[1], abs=0.01), "as robot_2 entered"
        assert starts[3][0] == pytest.approx(lefts[-1] + 2.0, abs=0.01), "closed after it left"
        reopened = [e for e in events if e["event"] == "reopened"]
        assert [e.get("who") for e in reopened] == ["robot_2"]
        closes = [e for e in events if e["source"] == "door_1" and e["event"] == "program_end"]
        assert closes[-1]["program"] == "close" and closes[-1]["error"] == ""
        assert _leaf(engine) == pytest.approx(0.0, abs=0.003)
        assert any(e["event"] == "scenario_end" for e in events), "one cycle, then the end"
    finally:
        engine.shutdown()


def test_a_close_that_stalls_reopens_the_door_and_closes_once_the_way_is_clear():
    from roqsim.entity_control import set_state

    # A box the zone does not count, put in the doorway once the door is open: the close stalls on
    # it, and the door reopens.
    block = {
        "box": {
            "prefix": "obs_",
            "pose": {"position": {"x": -0.2, "y": 3.0, "z": 0.3}},
            "size": [0.2, 0.2, 0.6],
            "motion": "driven",
        },
        "name": "obstacle",
    }
    engine = _engine(DOOR, APPROACH, _mover("robot_1"), block, {**SCENE, "name": "doorman"})
    try:

        def reopened():
            return [e for e in engine.ctx.events.all() if e["event"] == "reopened"]

        _run(engine, 5.0)
        assert _leaf(engine) == pytest.approx(1.2, abs=0.01)
        set_state(engine.ctx, "obstacle", [-0.2, 1.08, 0.3], [1.0, 0.0, 0.0, 0.0])
        while not reopened():
            assert engine.ctx.sim_time < 15.0, _door_starts(engine)
            engine.step()
        assert reopened()[0]["error"].startswith("stalled at")
        assert "who" not in reopened()[0]
        _run(engine, 2.5)
        assert _leaf(engine) == pytest.approx(1.2, abs=0.01), "open again"
        set_state(engine.ctx, "obstacle", [-0.2, 3.0, 0.3], [1.0, 0.0, 0.0, 0.0])
        end = engine.ctx.sim_time + 20.0
        while not any(e["event"] == "scenario_end" for e in engine.ctx.events.all()):
            assert engine.ctx.sim_time < end, _door_starts(engine)
            engine.step()
        assert _leaf(engine) == pytest.approx(0.0, abs=0.003)
        programs = [p for _, p in _door_starts(engine)]
        assert programs[:3] == ["open", "close", "open"] and programs[-1] == "close"
    finally:
        engine.shutdown()


def test_a_door_that_keeps_failing_to_close_with_nobody_there_says_blocked_once_and_keeps_trying():
    from roqsim.entity_control import set_state

    # A box the zone does not count, left in the doorway: every close stalls on it.
    block = {
        "box": {
            "prefix": "obs_",
            "pose": {"position": {"x": -0.2, "y": 3.0, "z": 0.3}},
            "size": [0.2, 0.2, 0.6],
            "motion": "driven",
        },
        "name": "obstacle",
    }
    scene = {**SCENE, "name": "doorman"}
    engine = _engine(DOOR, APPROACH, _mover("robot_1"), block, scene)
    try:

        def events(kind):
            return [e for e in engine.ctx.events.all() if e["event"] == kind]

        _run(engine, 5.0)
        set_state(engine.ctx, "obstacle", [-0.2, 1.08, 0.3], [1.0, 0.0, 0.0, 0.0])
        while len(events("reopened")) < 4:
            assert engine.ctx.sim_time < 90.0, _door_starts(engine)
            engine.step()
        blocked = events("blocked")
        assert len(blocked) == 1 and blocked[0]["closes"] == 3
        assert blocked[0]["t"] > events("reopened")[2]["t"] - 1e-9, "said at the third failed close"
        assert not any(e["event"] == "scenario_end" for e in engine.ctx.events.all()), (
            "it keeps trying"
        )
    finally:
        engine.shutdown()


def test_someone_entering_during_a_close_starts_the_count_to_blocked_again():
    from roqsim.entity_control import set_state

    def box(name, y):
        return {
            "box": {
                "prefix": f"{name}_",
                "pose": {"position": {"x": -0.2, "y": y, "z": 0.3}},
                "size": [0.2, 0.2, 0.6],
                "motion": "driven",
            },
            "name": name,
        }

    scene = {**SCENE, "name": "doorman"}
    scene[next(iter(SCENE))] = {"close_delay": 2.0, "blocked_after": 2}
    # `obstacle` is not counted by the zone and stalls the leaf; `robot_2` is counted.
    engine = _engine(
        DOOR, APPROACH, _mover("robot_1"), box("obstacle", 3.0), box("robot_2", 4.0), scene
    )
    try:

        def events(kind):
            return [e for e in engine.ctx.events.all() if e["event"] == kind]

        def closes():
            return [p for _, p in _door_starts(engine) if p == "close"]

        def until(cond, what):
            while not cond():
                assert engine.ctx.sim_time < 120.0, what
                engine.step()

        _run(engine, 5.0)
        set_state(engine.ctx, "obstacle", [-0.2, 1.08, 0.3], [1.0, 0.0, 0.0, 0.0])
        until(lambda: len(events("reopened")) == 1, "a first failed close")
        until(lambda: len(closes()) == 2, "a second close")
        set_state(engine.ctx, "robot_2", [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])  # into the zone
        until(lambda: len(events("reopened")) == 2, "someone entered")
        assert events("reopened")[1].get("who") == "robot_2"
        set_state(engine.ctx, "robot_2", [0.0, 4.0, 0.3], [1.0, 0.0, 0.0, 0.0])  # gone again
        until(lambda: len(events("reopened")) == 3, "another failed close")
        assert not events("blocked"), "failed, someone, failed: not two failures in a row"
        until(lambda: len(events("reopened")) == 4, "a second failed close in a row")
        assert len(events("blocked")) == 1
    finally:
        engine.shutdown()


def test_blocked_after_is_a_whole_number_of_closes():
    with pytest.raises(Exception, match="blocked_after"):
        _engine(DOOR, APPROACH, {next(iter(SCENE)): {"blocked_after": 0}, "name": "doorman"})


def test_the_demo_world_opens_the_door_for_its_robot_closes_it_and_ends():
    from pathlib import Path

    import roqsim_assets
    from roqsim.config import load_config

    world = Path(roqsim_assets.__file__).parent / "worlds" / "sliding_door_demo.yaml"
    assert "walker" not in world.read_text().split("sim:")[1], "stand-alone: no walker"
    engine = Engine(load_config(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        _run(engine, 6.0)
        door = [e for e in engine.ctx.events.all() if e["source"] == "door_1"]
        assert [e["program"] for e in door if e["event"] == "program_end"] == ["open", "close"]
        assert all(e["error"] == "" for e in door if e["event"] == "program_end")
        assert any(e["event"] == "scenario_end" for e in engine.ctx.events.all())
        assert engine.ctx.stop_requested, "and the run with it: roqsim sim exits"
        m = engine.ctx.model
        assert engine.ctx.data.qpos[
            m.jnt_qposadr[m.joint("door_1_leaf_slide").id]
        ] == pytest.approx(0.0, abs=0.01)
    finally:
        engine.shutdown()


def test_slide_minus_one_opens_the_leaf_the_other_way_along_the_wall():
    def leaf_x(slide):
        door = {**DOOR, "sliding_door": {**DOOR["sliding_door"], "slide": slide}}
        engine = _engine(door)
        try:
            handle = engine.ctx.blackboard.require("machine:door_1")
            body = engine.ctx.model.body("d1_leaf").id
            closed = float(engine.ctx.data.xpos[body][0])
            assert _finish(engine, handle, handle.start("open")).error == ""
            return float(engine.ctx.data.xpos[body][0]) - closed
        finally:
            engine.shutdown()

    assert leaf_x(1) == pytest.approx(1.2, abs=0.01)
    assert leaf_x(-1) == pytest.approx(-1.2, abs=0.01)
