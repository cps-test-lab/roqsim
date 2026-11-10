"""An OpenSCENARIO file named by a world entry, run inside the engine's own loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import tomllib
import yaml

pytest.importorskip("scenario_execution", reason="scenario-execution: what the plugin runs")

from test_osc_library import _entry_points  # noqa: E402

from roqsim import exit_status, runner  # noqa: E402
from roqsim.config import load_config_from_dict, world_sources  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim.recording import open_recording  # noqa: E402
from scenario_execution_roqsim.scenario_file import ScenarioFileError  # noqa: E402

PLUGIN = "scenario_execution_roqsim.scenario_file:ScenarioFilePlugin"
WORLD = [
    {"scene_fixtures:Press": {}, "name": "press"},
    {"scene_fixtures:Box": {}, "name": "box"},
    {
        "zone_trigger": {
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
            "size": [0.2, 0.3, 0.2],
        },
        "name": "gate",
    },
]
GOOD = """
import osc.roqsim
scenario good:
    do serial:
        zone_entered(zone: 'gate')
        machine_run(machine: 'press', program: 'stroke', args: '{\\"seconds\\": 0.1}')
"""
BAD = """
import osc.roqsim
scenario bad:
    do serial:
        machine_run(machine: 'press', program: 'jam')
"""
LONG = """
import osc.roqsim
scenario long:
    do serial:
        wait elapsed(100s)
"""


@pytest.fixture(autouse=True)
def entry_points():
    with (
        patch("scenario_execution.model.model_builder.entry_points", _entry_points),
        patch("scenario_execution.model.model_to_py_tree.entry_points", _entry_points),
    ):
        yield


def _world(tmp_path, text, **config):
    (tmp_path / "scene.osc").write_text(text)
    entry = {PLUGIN: {"file": "scene.osc", **config}, "name": "scene"}
    world = {"sim": {"gravity": [0, 0, 0]}, "components": [*WORLD, entry]}
    (tmp_path / "w.yaml").write_text(yaml.safe_dump(world))
    return tmp_path / "w.yaml"


def _engine(tmp_path, text, **config):
    path = _world(tmp_path, text, **config)
    cfg = load_config_from_dict(yaml.safe_load(path.read_text()), base_dir=tmp_path)
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _until_stop(engine, seconds=3.0):
    while not engine.ctx.stop_requested and engine.ctx.sim_time < seconds:
        engine.step()


def _timeline(engine):
    return [
        (e["source"], e["event"], round(e["t"], 3))
        for e in engine.ctx.events.all()
        if e["source"] in ("press", "gate", "scene")
    ]


def test_the_scenario_runs_in_the_world_and_ends_the_run(tmp_path):
    engine = _engine(tmp_path, GOOD)
    try:
        _until_stop(engine)
        assert engine.ctx.stop_requested
        assert engine.ctx.interface.find("scene", "done").read() is True
        events = [event for _source, event, _t in _timeline(engine)]
        assert events == ["entered", "program_start", "stroked", "program_end", "scenario_end"]
    finally:
        engine.shutdown()


def test_stop_at_end_false_leaves_the_run_going(tmp_path):
    engine = _engine(tmp_path, GOOD, stop_at_end=False)
    try:
        for _ in range(800):
            engine.step()
        assert engine.ctx.interface.find("scene", "done").read() is True
        assert not engine.ctx.stop_requested
    finally:
        engine.shutdown()


def test_a_scenario_that_fails_stops_the_run_naming_why(tmp_path):
    engine = _engine(tmp_path, BAD)
    try:
        with pytest.raises(ScenarioFileError, match="failed"):
            _until_stop(engine)
        assert engine.ctx.interface.find("scene", "done").read() is False
    finally:
        engine.shutdown()


def test_each_episode_runs_the_scenario_again(tmp_path):
    engine = _engine(tmp_path, GOOD)
    try:
        _until_stop(engine)
        first = _timeline(engine)
        engine.reset()
        assert engine.ctx.interface.find("scene", "done").read() is False
        _until_stop(engine)
        assert _timeline(engine)[len(first) :] == first
    finally:
        engine.shutdown()


def test_a_reset_mid_scenario_starts_it_over(tmp_path, caplog):
    engine = _engine(tmp_path, LONG, stop_at_end=False)
    try:
        for _ in range(5):
            engine.step()
        with caplog.at_level("ERROR"):
            engine.reset()
        assert "the run ended before" not in caplog.text, "a reset is no end of the run"
        assert engine.ctx.unfinished == []
        for _ in range(5):
            engine.step()
    finally:
        engine.shutdown()


def test_a_run_that_ends_first_says_so_and_writes_it(tmp_path, caplog):
    engine = _engine(tmp_path, LONG, output_dir=str(tmp_path))
    for _ in range(5):
        engine.step()
    with caplog.at_level("ERROR"):
        engine.shutdown()
    assert "the run ended before" in caplog.text
    assert engine.ctx.unfinished == ["scenario scene (scene.osc)"]
    assert "the run ended before the scenario did" in (tmp_path / "test.xml").read_text()


def test_results_go_where_output_dir_says(tmp_path):
    engine = _engine(tmp_path, GOOD, output_dir=str(tmp_path))
    try:
        _until_stop(engine)
        assert 'failures="0"' in (tmp_path / "test.xml").read_text()
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("config", "said"),
    [
        ({}, "has 2 scenarios"),
        ({"scenario": "nope"}, "no scenario 'nope'"),
    ],
)
def test_which_scenario_of_a_file_is_said_or_refused(tmp_path, config, said):
    engine = _engine(tmp_path, GOOD + BAD.replace("import osc.roqsim\n", ""), **config)
    try:
        with pytest.raises(ScenarioFileError, match=said):
            engine.step()
    finally:
        engine.shutdown()


def test_a_named_scenario_is_the_one_run(tmp_path):
    engine = _engine(tmp_path, GOOD + BAD.replace("import osc.roqsim\n", ""), scenario="good")
    try:
        _until_stop(engine)
        assert engine.ctx.interface.find("scene", "done").read() is True
    finally:
        engine.shutdown()


def test_a_file_that_does_not_parse_says_why(tmp_path):
    engine = _engine(tmp_path, "this is { not a scenario\n")
    try:
        with pytest.raises(ScenarioFileError, match="did not parse"):
            engine.step()
    finally:
        engine.shutdown()


def test_a_file_that_is_not_there_says_so(tmp_path):
    (tmp_path / "scene.osc").write_text(GOOD)
    engine = _engine(tmp_path, GOOD)
    (tmp_path / "scene.osc").unlink()
    try:
        with pytest.raises(ScenarioFileError, match="File does not exist"):
            engine.step()
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("config", "said"),
    [
        ({"file": ""}, "'file' names"),
        ({"file": "scene.osc", "stop_at_end": "yes"}, "'stop_at_end' is true or false"),
        ({"file": "scene.osc", "nope": 1}, "unknown key"),
    ],
)
def test_a_bad_entry_is_refused_when_the_world_loads(tmp_path, config, said):
    (tmp_path / "scene.osc").write_text(GOOD)
    world = {"sim": {}, "components": [{PLUGIN: config, "name": "scene"}]}
    with pytest.raises(Exception, match=said):
        Engine(load_config_from_dict(world, base_dir=tmp_path)).setup()


def test_roqsim_sim_runs_it_and_a_failure_is_not_a_success(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    argv = ["--headless", "--seconds", "5", "--pacing", "asap", "--control", "none"]
    assert runner.main([str(_world(tmp_path, GOOD)), *argv]) == 0
    with pytest.raises(ScenarioFileError, match="failed"):
        runner.main([str(_world(tmp_path, BAD)), *argv])


def test_roqsim_sim_ending_before_the_scenario_is_no_pass(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    argv = ["--headless", "--seconds", "1", "--pacing", "asap", "--control", "none"]
    with caplog.at_level("ERROR"):
        assert runner.main([str(_world(tmp_path, LONG)), *argv]) == exit_status.UNFINISHED
    assert caplog.text.count("the run ended before scenario scene") == 1


def test_a_recording_replays_without_running_the_scenario(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    path = _world(tmp_path, GOOD)
    out = tmp_path / "run.mcap"
    argv = [str(path), "--headless", "--seconds", "5", "--pacing", "asap", "--control", "none"]
    assert runner.main([*argv, "--record", str(out)]) == 0
    with (
        open_recording(out) as rec,
        patch(
            "scenario_execution_roqsim.scenario_file.ScenarioFilePlugin._start",
            side_effect=AssertionError("a replay parsed the scenario"),
        ),
    ):
        rec.at(0.5)
        assert rec.build()
        assert "scenario_end" in {e["event"] for e in rec.events}


def test_a_replay_with_sensors_does_not_run_the_scenario(tmp_path, monkeypatch):
    """A replay resets its world and drives post_step on restored samples, as a run does."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    path = _world(tmp_path, GOOD, output_dir="out")
    out = tmp_path / "run.mcap"
    argv = [str(path), "--headless", "--seconds", "5", "--pacing", "asap", "--control", "none"]
    assert runner.main([*argv, "--record", str(out)]) == 0
    verdict = (tmp_path / "out" / "test.xml").read_text()
    assert 'failures="0"' in verdict
    with open_recording(out) as rec:
        rec.enable_sensors()
        rec.at(0.5)
        assert not rec.replay_failures
        assert "scenario_end" in {e["event"] for e in rec.events}
    assert (tmp_path / "out" / "test.xml").read_text() == verdict


def test_a_settled_export_does_not_run_the_scenario(tmp_path):
    path = _world(tmp_path, GOOD, output_dir="out")
    cfg = load_config_from_dict(yaml.safe_load(path.read_text()), base_dir=tmp_path)
    with Engine(cfg, preview=True) as engine:
        engine.reset()
        for _ in range(5):
            engine.step()
    assert not (tmp_path / "out").exists()
    assert engine.ctx.unfinished == []


def test_a_second_entry_is_refused(tmp_path):
    (tmp_path / "scene.osc").write_text(GOOD)
    entries = [{PLUGIN: {"file": "scene.osc"}, "name": name} for name in ("a", "b")]
    cfg = load_config_from_dict({"sim": {}, "components": entries}, base_dir=tmp_path)
    with pytest.raises(ScenarioFileError, match="a second entry"):
        Engine(cfg).setup()


def test_an_error_in_a_tick_is_what_test_xml_says(tmp_path):
    text = """
import osc.roqsim
scenario typo:
    do serial:
        entity_call(entity: 'press', command: 'no_such_command')
"""
    engine = _engine(tmp_path, text, output_dir="out")
    try:
        with pytest.raises(Exception, match="no_such_command"):
            _until_stop(engine)
        written = (tmp_path / "out" / "test.xml").read_text()
        assert "no_such_command" in written and "the run ended before" not in written
    finally:
        engine.shutdown()


def test_a_file_that_does_not_parse_is_in_test_xml(tmp_path):
    engine = _engine(tmp_path, "this is { not a scenario\n", output_dir="out")
    try:
        with pytest.raises(ScenarioFileError, match="did not parse"):
            engine.step()
        assert "parsing failed" in (tmp_path / "out" / "test.xml").read_text()
    finally:
        engine.shutdown()


def test_output_dir_is_relative_to_the_world_and_made_whole(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    engine = _engine(tmp_path, GOOD, output_dir="a/b/c")
    try:
        _until_stop(engine)
        assert (tmp_path / "a/b/c/test.xml").exists()
    finally:
        engine.shutdown()


def test_the_file_is_among_a_worlds_sources_from_any_directory(tmp_path, monkeypatch):
    path = _world(tmp_path, GOOD)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert str(tmp_path / "scene.osc") in {str(p) for p in world_sources(str(path))}


def test_the_short_name_is_the_plugin():
    project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    target = project["project"]["entry-points"]["roqsim.plugins"]["scenario"]
    assert target == PLUGIN
