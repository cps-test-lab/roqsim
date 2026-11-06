# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Python scenarios: a generator on simulated time that drives entities through their handles."""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.scenario import ScenarioError

PRESS = {"machine_fixtures:Press": {"prefix": "p1_"}, "name": "press"}


def _engine(*components):
    engine = Engine(
        load_config_from_dict({"sim": {"gravity": [0, 0, 0]}, "components": list(components)})
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _events(engine, source):
    return [e for e in engine.ctx.events.all() if e["source"] == source]


def _until_done(engine, scenario="scene", limit=30.0):
    done = engine.ctx.interface.find(scenario, "done")
    while not done.read():
        assert engine.ctx.sim_time < limit, "the scenario never ended"
        engine.step()


def test_a_scenario_runs_a_machine_twice_and_says_when_it_is_done():
    engine = _engine(PRESS, {"scenario_fixtures:TwoStrokes": {"pause": 0.2}, "name": "scene"})
    try:
        _until_done(engine)
        press = [e["event"] for e in _events(engine, "press")]
        assert press.count("program_start") == 2 and press.count("program_end") == 2
        scene = _events(engine, "scene")
        assert [e["event"] for e in scene] == ["first", "second", "scenario_end"]
        assert scene[0]["seq"] == 1
        ends = [e["t"] for e in _events(engine, "press") if e["event"] == "program_end"]
        starts = [e["t"] for e in _events(engine, "press") if e["event"] == "program_start"]
        assert starts[1] - ends[0] == pytest.approx(0.2, abs=2 * engine.ctx.dt)
    finally:
        engine.shutdown()


def test_a_machine_error_raises_in_the_program_and_stops_the_run_naming_both():
    engine = _engine(PRESS, {"scenario_fixtures:Jammed": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match=r"scenario scene: press.jam \(seq 1\): jammed"):
            for _ in range(1000):
                engine.step()
    finally:
        engine.shutdown()


def test_a_program_that_asks_for_the_status_decides_itself():
    engine = _engine(PRESS, {"scenario_fixtures:Jammed": {"check": False}, "name": "scene"})
    try:
        _until_done(engine)
        assert _events(engine, "scene")[0]["error"] == "jammed"
    finally:
        engine.shutdown()


def test_tasks_run_beside_each_other_join_and_race():
    engine = _engine(
        PRESS,
        {"machine_fixtures:Spinner": {"ramp": 0.5}, "name": "spin"},
        {"scenario_fixtures:Concurrent": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        scene = {e["event"]: e for e in _events(engine, "scene")}
        # The press dwells 0.3 s, the spinner ramps 0.3 m/s at 0.5 m/s^2 (0.6 s): both started in
        # the same step, so the join is the longer of the two.
        starts = [e["t"] for e in engine.ctx.events.all() if e["event"] == "program_start"]
        assert starts[0] == starts[1]
        assert scene["joined"]["t"] - starts[0] == pytest.approx(0.6, abs=3 * engine.ctx.dt)
        assert scene["won"]["by"] == "short" and scene["won"]["long_cancelled"]
        assert "short_done" in scene and "long_done" not in scene
    finally:
        engine.shutdown()


def test_a_route_is_waited_on_by_its_own_number():
    engine = _engine(
        {"scenario_fixtures:FakeWalker": {"route_s": 0.5, "error": "blocked"}, "name": "worker"},
        {"scenario_fixtures:Walk": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        arrived = _events(engine, "scene")[0]
        assert arrived["event"] == "arrived" and arrived["error"] == "blocked"
        assert arrived["t"] == pytest.approx(0.5, abs=3 * engine.ctx.dt)
    finally:
        engine.shutdown()


def test_a_handle_nothing_published_is_refused_at_reset_naming_the_key():
    with pytest.raises(ScenarioError, match="scenario scene: needs 'machine:press'"):
        _engine({"scenario_fixtures:TwoStrokes": {}, "name": "scene"})


def test_needs_may_rename_a_handle_and_refuses_one_the_scenario_does_not_use():
    renamed = {"machine_fixtures:Press": {"prefix": "p2_"}, "name": "press_2"}
    engine = _engine(
        renamed,
        {"scenario_fixtures:TwoStrokes": {"needs": {"press": "machine:press_2"}}, "name": "scene"},
    )
    try:
        _until_done(engine)
        assert [e["event"] for e in _events(engine, "press_2")].count("program_end") == 2
    finally:
        engine.shutdown()
    with pytest.raises(Exception, match="'needs' names \\['gripper'\\], which it does not use"):
        _engine(
            PRESS,
            {"scenario_fixtures:TwoStrokes": {"needs": {"gripper": "machine:x"}}, "name": "scene"},
        )


def test_stop_at_end_ends_a_standalone_run_and_a_reset_starts_the_scene_again():
    engine = _engine(
        PRESS, {"scenario_fixtures:TwoStrokes": {"stop_at_end": True}, "name": "scene"}
    )
    try:
        _until_done(engine)
        assert engine.ctx.stop_requested
        engine.reset()
        assert not engine.ctx.interface.find("scene", "done").read()
        _until_done(engine)
        assert [e["event"] for e in _events(engine, "scene")].count("scenario_end") == 2
    finally:
        engine.shutdown()


def test_two_runs_of_a_scene_give_the_same_timeline():
    def run():
        engine = _engine(PRESS, {"scenario_fixtures:TwoStrokes": {}, "name": "scene"})
        try:
            _until_done(engine)
            return [(e["t"], e["source"], e["event"]) for e in engine.ctx.events.all()]
        finally:
            engine.shutdown()

    assert run() == run()


# -- set_entity_state(), with a pose as a world file writes one ---------------------------------------------
BLOCK = """<mujoco model="block">
  <worldbody>
    <body name="block"><geom type="box" size="0.05 0.05 0.05" mass="1"/></body>
  </worldbody>
</mujoco>"""


def _placing(tmp_path, pose):
    (tmp_path / "block.xml").write_text(BLOCK)
    spawn = {
        "spawn_model": {
            "model": str(tmp_path / "block.xml"),
            "pose": {"position": {"x": 2.0, "y": 2.0, "z": 0.5}},
        },
        "name": "block",
    }
    return _engine(spawn, {"scenario_fixtures:Place": {"pose": pose}, "name": "scene"})


def _block(engine):
    d, bid = engine.ctx.data, engine.ctx.model.body("block").id
    return [float(v) for v in d.xpos[bid]], [float(v) for v in d.xquat[bid]]


@pytest.mark.parametrize(
    ("pose", "where"),
    [
        ({"position": {"x": 0.5, "y": -0.5, "z": 0.3}}, [0.5, -0.5, 0.3]),
        ({"position": {"x": 0.5, "y": -0.5}}, [0.5, -0.5, 0.0]),
        (
            {"header": {"frame_id": "world"}, "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.2}}},
            [1.0, 0.0, 0.2],
        ),
    ],
)
def test_place_puts_the_entity_at_a_world_pose_a_missing_z_being_the_floor(tmp_path, pose, where):
    engine = _placing(tmp_path, pose)
    try:
        engine.step()
        pos, _quat = _block(engine)
        assert pos == pytest.approx(where, abs=1e-6)
        engine.step()
        assert [e["event"] for e in _events(engine, "scene")] == ["placed", "scenario_end"]
    finally:
        engine.shutdown()


def test_place_turns_an_orientation_as_a_world_file_states_it(tmp_path):
    engine = _placing(
        tmp_path, {"position": {"x": 0.0, "y": 0.0, "z": 0.3}, "orientation": {"yaw": 1.0}}
    )
    try:
        engine.step()
        _pos, quat = _block(engine)
        assert quat == pytest.approx([0.8775826, 0.0, 0.0, 0.4794255], abs=1e-6)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("pose", "said"),
    [
        ({"position": {"x": 0.5}}, "'pose.position.y'"),
        ("0.5, 0.5", "must be a mapping"),
        ({"position": {"x": 0.5, "y": 0.5}, "rotation": {}}, "no key"),
        ({"header": {"frame_id": "map"}, "pose": {"position": {"x": 0.0, "y": 0.0}}}, "'map'"),
    ],
)
def test_place_refuses_what_is_no_world_pose_naming_what_is_wrong(tmp_path, pose, said):
    from roqsim.pose import PoseError

    engine = _placing(tmp_path, pose)
    try:
        with pytest.raises(PoseError, match=said) as exc:
            engine.step()
        assert "raised in scenario scene's program" in "".join(getattr(exc.value, "__notes__", []))
        pos, _quat = _block(engine)
        assert pos == pytest.approx([2.0, 2.0, 0.5], abs=1e-6), "not moved"
    finally:
        engine.shutdown()


# -- what a scenario's failures say ----------------------------------------------------------------
def test_an_until_that_times_out_names_the_scenario_once():
    engine = _engine({"scenario_fixtures:Impatient": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError) as exc:
            for _ in range(1000):
                engine.step()
        assert str(exc.value) == "scenario scene: the feeder not true after 0.2 s"
    finally:
        engine.shutdown()


def test_a_program_that_raises_stops_the_run_with_its_own_exception():
    engine = _engine({"scenario_fixtures:Broken": {}, "name": "scene"})
    try:
        with pytest.raises(KeyError, match="no such part"):
            for _ in range(5):
                engine.step()
    finally:
        engine.shutdown()


def test_a_handle_named_like_a_member_of_the_scenario_is_refused():
    with pytest.raises(
        Exception, match=r"NEEDS names \['config'\], which the scenario already has"
    ):
        _engine(PRESS, {"scenario_fixtures:Clashing": {}, "name": "scene"})


def test_place_refuses_a_position_that_is_not_finite(tmp_path):
    from roqsim.pose import PoseError

    engine = _placing(tmp_path, {"position": {"x": float("nan"), "y": 0.0}})
    try:
        with pytest.raises(PoseError, match="must be a finite number"):
            engine.step()
        pos, _quat = _block(engine)
        assert pos == pytest.approx([2.0, 2.0, 0.5], abs=1e-6), "not moved"
    finally:
        engine.shutdown()


# -- a start's error reaches whoever joins it ------------------------------------------------------
def test_a_start_tasks_error_is_raised_in_the_join_where_a_try_catches_it():
    engine = _engine(
        PRESS,
        {"machine_fixtures:Spinner": {"ramp": 0.5}, "name": "spin"},
        {"scenario_fixtures:CatchAtJoin": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        scene = {e["event"]: e for e in _events(engine, "scene")}
        assert scene["caught"]["error"] == "press.jam (seq 1): jammed"
        assert "went_on" in scene and "scenario_end" in scene
    finally:
        engine.shutdown()


def test_a_start_tasks_error_nobody_joins_stops_the_run_when_the_program_returns():
    engine = _engine(PRESS, {"scenario_fixtures:Unjoined": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="scenario scene: press.jam .seq 1.: jammed"):
            for _ in range(1000):
                engine.step()
        assert engine.ctx.sim_time == pytest.approx(0.5, abs=0.01), "when the program returned"
    finally:
        engine.shutdown()


def test_a_route_another_overtook_returns_its_status_when_not_checked():
    engine = _engine(
        {"scenario_fixtures:FakeWalker": {"route_s": 0.5}, "name": "worker"},
        {"scenario_fixtures:Overtaken": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        assert _events(engine, "scene")[0]["seq"] == 2
    finally:
        engine.shutdown()


def test_after_a_program_raised_the_scene_never_reports_done():
    engine = _engine({"scenario_fixtures:Broken": {}, "name": "scene"})
    try:
        with pytest.raises(KeyError):
            for _ in range(5):
                engine.step()
        for _ in range(5):
            engine.step()
        assert not engine.ctx.interface.find("scene", "done").read()
        assert "scenario_end" not in [e["event"] for e in _events(engine, "scene")]
    finally:
        engine.shutdown()


def test_a_start_still_running_at_return_keeps_the_scene_open_and_its_error_stops_the_run():
    engine = _engine(PRESS, {"scenario_fixtures:ReturnsWhileAStartRuns": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="press.jam .seq 1.: jammed"):
            for _ in range(1000):
                engine.step()
        assert engine.ctx.sim_time > 0.05, "not before the machine failed"
        assert "scenario_end" not in {e["event"] for e in _events(engine, "scene")}
    finally:
        engine.shutdown()


def test_the_scene_ends_only_once_a_start_it_did_not_join_has():
    engine = _engine(PRESS, {"scenario_fixtures:ReturnsBesideAGoodStart": {}, "name": "scene"})
    try:
        _until_done(engine)
        machine = {e["event"]: e for e in _events(engine, "press")}
        scene = {e["event"]: e for e in _events(engine, "scene")}
        assert scene["scenario_end"]["t"] >= machine["program_end"]["t"]
    finally:
        engine.shutdown()


def test_a_run_ending_before_the_program_says_what_no_one_joined(caplog):
    engine = _engine(PRESS, {"scenario_fixtures:Outlasts": {}, "name": "scene"})
    for _ in range(1000):
        engine.step()
    with caplog.at_level("ERROR"):
        events = engine.ctx.events
        engine.shutdown()
    assert "a start's error reached no one: press.jam" in caplog.text
    failed = [e for e in events.all() if e["event"] == "start_failed"]
    assert len(failed) == 1 and failed[0]["start"] == "press.jam"


def test_a_timer_that_wins_over_a_start_leaves_the_machine_free():
    engine = _engine(PRESS, {"scenario_fixtures:TimesOut": {}, "name": "scene"})
    try:
        _until_done(engine)
        ends = [e["error"] for e in _events(engine, "press") if e["event"] == "program_end"]
        assert ends == ["cancelled", ""]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("scenario", "said"),
    [("FirstOfNothing", "no task that can finish"), ("WaitNaN", "number of seconds")],
)
def test_a_wait_that_could_not_end_is_refused(scenario, said):
    engine = _engine({f"scenario_fixtures:{scenario}": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match=said):
            for _ in range(5):
                engine.step()
    finally:
        engine.shutdown()


def test_a_program_with_no_yield_is_refused_naming_it():
    with pytest.raises(ScenarioError, match="scenario scene: program.. is a generator"):
        _engine({"scenario_fixtures:NoYield": {}, "name": "scene"})


def test_stop_at_end_is_true_or_false():
    with pytest.raises(Exception, match="'stop_at_end' is true or false"):
        _engine({"scenario_fixtures:NoYield": {"stop_at_end": "false"}, "name": "scene"})


def test_cancelling_a_finished_start_leaves_the_next_one_running():
    engine = _engine(PRESS, {"scenario_fixtures:CancelsAFinishedStart": {}, "name": "scene"})
    try:
        _until_done(engine)
        ends = [e["error"] for e in _events(engine, "press") if e["event"] == "program_end"]
        assert ends == ["", ""]
    finally:
        engine.shutdown()


def test_cancelling_a_task_leaves_a_start_another_client_made_since_alone():
    engine = _engine(PRESS, {"scenario_fixtures:CancelledAfterAnotherStart": {}, "name": "scene"})
    try:
        _until_done(engine)
        ends = [e["error"] for e in _events(engine, "press") if e["event"] == "program_end"]
        assert ends == ["cancelled", ""]
    finally:
        engine.shutdown()


def test_a_reset_says_what_no_one_joined(caplog):
    engine = _engine(PRESS, {"scenario_fixtures:Outlasts": {}, "name": "scene"})
    try:
        for _ in range(300):
            engine.step()
        with caplog.at_level("ERROR"):
            engine.reset()
        assert "a start's error reached no one: press.jam" in caplog.text
    finally:
        engine.shutdown()


def test_a_joiner_created_before_its_start_still_catches_its_error():
    engine = _engine(PRESS, {"scenario_fixtures:WatcherBeforeItsStart": {}, "name": "scene"})
    try:
        _until_done(engine)
        scene = {e["event"] for e in _events(engine, "scene")}
        assert "caught" in scene and "scenario_end" in scene
    finally:
        engine.shutdown()


def test_a_timeout_that_is_no_number_is_refused():
    engine = _engine({"scenario_fixtures:UntilNaN": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="until.. takes a timeout"):
            engine.step()
    finally:
        engine.shutdown()


def test_a_bare_yield_of_a_helper_is_refused_rather_than_doing_nothing():
    engine = _engine(PRESS, {"scenario_fixtures:YieldsAHelper": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="yield from"):
            engine.step()
    finally:
        engine.shutdown()


def test_a_join_raises_as_soon_as_one_task_fails():
    other = {"machine_fixtures:Press": {"prefix": "p2_"}, "name": "other"}
    engine = _engine(
        PRESS, other, {"scenario_fixtures:JoinsAFailureAndALongStart": {}, "name": "scene"}
    )
    try:
        _until_done(engine)
        caught = next(e for e in _events(engine, "scene") if e["event"] == "caught")
        assert "jammed" in caught["error"] and caught["long_done"] is False
        assert engine.ctx.sim_time < 1.0
    finally:
        engine.shutdown()


def test_a_spawned_run_keeps_the_scene_open_and_its_error_stops_the_run():
    engine = _engine(PRESS, {"scenario_fixtures:ReturnsBesideASpawnedRun": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="press.jam .seq 1.: jammed"):
            for _ in range(1000):
                engine.step()
        assert "scenario_end" not in {e["event"] for e in _events(engine, "scene")}
    finally:
        engine.shutdown()


def test_a_background_task_does_not_keep_the_scene_open():
    engine = _engine({"scenario_fixtures:ReturnsBesideABackgroundTask": {}, "name": "scene"})
    try:
        _until_done(engine)
        assert engine.ctx.sim_time < 0.1
    finally:
        engine.shutdown()


def test_a_timer_that_wins_over_a_spawned_run_leaves_the_machine_free():
    engine = _engine(PRESS, {"scenario_fixtures:SpawnedRunLosesToATimer": {}, "name": "scene"})
    try:
        _until_done(engine)
        ends = [e["error"] for e in _events(engine, "press") if e["event"] == "program_end"]
        assert ends == ["cancelled", ""]
    finally:
        engine.shutdown()


def test_spawn_of_what_is_no_generator_is_refused():
    engine = _engine({"scenario_fixtures:SpawnsNoGenerator": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match="spawn.. takes a generator"):
            engine.step()
    finally:
        engine.shutdown()


def test_cancelling_a_start_before_its_first_advance_cancels_its_program():
    engine = _engine(PRESS, {"scenario_fixtures:SharedDeadline": {}, "name": "scene"})
    try:
        _until_done(engine)
        ends = [e["error"] for e in _events(engine, "press") if e["event"] == "program_end"]
        assert ends[-1] == "", "the next start ran to its end"
    finally:
        engine.shutdown()


def test_a_timer_that_wins_over_a_route_stops_the_mover():
    engine = _engine(
        {"scenario_fixtures:FakeWalker": {"route_s": 5.0}, "name": "worker"},
        {"scenario_fixtures:RouteLosesToATimer": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        raced = next(e for e in _events(engine, "scene") if e["event"] == "raced")
        assert raced["cancels"] == 1
    finally:
        engine.shutdown()


def test_a_reset_does_not_stop_a_mover_for_the_next_episode():
    engine = _engine(
        {"scenario_fixtures:FakeWalker": {"route_s": 5.0}, "name": "worker"},
        {"scenario_fixtures:RouteLosesToATimer": {}, "name": "scene"},
    )
    try:
        engine.step()
        engine.step()
        engine.reset()
        assert engine.ctx.blackboard.require("walker:worker").cancels == 0
    finally:
        engine.shutdown()


def test_a_world_compiled_to_look_at_it_runs_no_scene():
    world = {
        "sim": {"gravity": [0, 0, 0]},
        "components": [PRESS, {"scenario_fixtures:TwoStrokes": {}, "name": "scene"}],
    }
    with Engine(load_config_from_dict(world), preview=True) as engine:
        engine.reset()
        for _ in range(50):
            engine.step()
        assert _events(engine, "press") == []
        assert engine.ctx.interface.find("scene", "done").read() is False
