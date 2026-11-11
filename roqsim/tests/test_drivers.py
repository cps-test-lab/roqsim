"""Phase 2: pacing, the standalone runner (headless), and the scenario-execution adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim.clock import SHORTFALL_REPORT_SHARE, Pacer
from roqsim.config import load_config_from_dict
from roqsim.runner import run
from roqsim.scenario_adapter import MujocoSim

DUMMY_WORLD = """
sim:
  timestep: 0.005
components:
  - dummy: {}
    name: d0
"""


def _write_world(tmp_path: Path) -> str:
    p = tmp_path / "dummy_world.yaml"
    p.write_text(DUMMY_WORLD)
    return str(p)


def test_pacer_asap_never_sleeps():
    pacer = Pacer.from_config("asap", dt=0.01)
    pacer.reset()
    pacer.wait()  # returns immediately; just assert no exception and correct mode
    assert pacer.realtime is False


def test_pacer_factor_parsing():
    pacer = Pacer.from_config({"factor": 4.0}, dt=0.01)
    assert pacer.factor == 4.0 and pacer.realtime is True


# -- pacing shortfall ---------------------------------------------------------------------
# Falling behind is absorbed on purpose: catching up would deliver sim time faster than the
# requested rate, which against a live stack is worse than being slow. Counting it is what
# stops that absorption from being silent -- uncounted, a run holding 0.34x realtime for five
# minutes reports nothing until its job deadline kills it, and the shortfall has to be
# reconstructed from the recording's clock channel afterwards.


def test_pacer_reports_no_shortfall_when_it_keeps_up():
    """The promise is SILENCE, not a zero count.

    Asserted through ``report_line`` rather than as ``late_steps == 0``, because the count is a
    wall-clock measurement of the machine the test runs on: a stray late step under load is
    normal, and is exactly what ``SHORTFALL_REPORT_SHARE`` exists to absorb. Pinning the count
    made this test assert that CI was idle, and it duly failed on a loaded runner while the
    feature worked. The 50 ms budget is roomy for the same reason -- unlike the 1 ms one the
    counting test needs, overrunning it means the machine is in real trouble.
    """
    pacer = Pacer.from_config("realtime", dt=0.05)
    pacer.wait()
    for _ in range(6):
        pacer.wait()
    report = pacer.shortfall()
    assert report["requested_factor"] == 1.0
    assert report["late_share"] < SHORTFALL_REPORT_SHARE
    assert pacer.report_line() is None


def test_pacer_counts_steps_it_could_not_pace():
    """A step whose own work overruns the budget leaves no time to sleep."""
    import time as _time

    pacer = Pacer.from_config("realtime", dt=0.001)
    pacer.wait()
    for _ in range(20):
        _time.sleep(0.003)  # three budgets' worth of work in one step
        pacer.wait()
    report = pacer.shortfall()
    assert report["late_steps"] == 20
    assert report["late_share"] == 1.0
    assert report["behind_seconds"] > 0
    # Roughly dt / (dt + owed): well under the requested rate, which is the whole point.
    assert 0.2 < report["achieved_factor"] < 0.6


def test_asap_has_no_rate_to_fall_short_of():
    """No target, so no shortfall -- reporting one would invent a contract nobody asked for."""
    assert Pacer.from_config("asap", dt=0.01).shortfall() == {}


def test_a_pacer_that_never_ran_reports_nothing():
    """Distinct from "kept up perfectly": zero steps is not evidence about pacing."""
    assert Pacer.from_config("realtime", dt=0.01).shortfall() == {}


def test_reset_keeps_the_counters():
    """``reset`` drops the SCHEDULE for a new series; the counters describe the whole run, so
    a run that was reset mid-flight must not report a clean bill for the part before it."""
    import time as _time

    pacer = Pacer.from_config("realtime", dt=0.001)
    pacer.wait()
    _time.sleep(0.003)
    pacer.wait()
    assert pacer.shortfall()["late_steps"] == 1
    pacer.reset()
    assert pacer.shortfall()["late_steps"] == 1


def test_runner_headless_runs_fixed_steps(tmp_path: Path):
    world = _write_world(tmp_path)
    engine = run(world, headless=True, pacing="asap", max_steps=20)
    counts = engine.ctx.blackboard.get("dummy_counts::d0")
    assert counts["pre_step"] == 20
    assert counts["post_step"] == 20


def test_runner_seconds_to_steps(tmp_path: Path):
    world = _write_world(tmp_path)
    # 0.05s / 0.005 dt = 10 steps
    engine = run(world, headless=True, pacing="asap", seconds=0.05)
    assert engine.ctx.blackboard.get("dummy_counts::d0")["pre_step"] == 10


def test_config_view_accessor():
    cfg = load_config_from_dict({"sim": {"view": {"azimuth": 42}}, "components": []})
    assert cfg.view == {"azimuth": 42}
    assert load_config_from_dict({"components": []}).view == {}


def test_scenario_adapter_lifecycle(tmp_path: Path):
    world = _write_world(tmp_path)
    sim = MujocoSim(world=world)
    sim.setup()
    assert abs(sim.dt - 0.005) < 1e-9
    sim.reset()
    for _ in range(7):
        sim.step()
    counts = sim._engine.ctx.blackboard.get("dummy_counts::d0")
    assert counts["pre_step"] == 7
    sim.shutdown()
    assert sim._engine is None


def test_scenario_adapter_shuts_the_engine_down_when_the_recording_cannot_close(
    tmp_path: Path, monkeypatch
):
    """A recording whose close raises (a full disk) still leaves the engine shut down."""
    from roqsim.capture import RecordingError

    class _Rec:
        frames = 0

        def __init__(self, ctx, path, rate, **kw):
            pass

        def sample(self, *a, **k):
            return False

        def close(self):
            raise RecordingError("disk full")

    monkeypatch.setattr("roqsim.capture.StateRecorder", _Rec)
    monkeypatch.setenv("ROQSIM_RECORD", str(tmp_path / "run.mcap"))
    sim = MujocoSim(world=_write_world(tmp_path))
    sim.setup()
    sim.reset()
    sim.step()
    ctx = sim.context
    with pytest.raises(RecordingError, match="disk full"):
        sim.shutdown()
    assert ctx.blackboard.get("dummy_counts::d0")["shutdown"] == 1
    assert sim._engine is None


def test_scenario_adapter_shuts_the_engine_down_when_the_viewer_cannot_close(tmp_path: Path):
    """A viewer whose close raises still leaves the engine shut down."""

    class _Viewer:
        def close(self):
            raise RuntimeError("GL context lost")

    sim = MujocoSim(world=_write_world(tmp_path))
    sim.setup()
    sim.reset()
    sim._viewer = _Viewer()
    ctx = sim.context
    with pytest.raises(RuntimeError, match="GL context lost"):
        sim.shutdown()
    assert ctx.blackboard.get("dummy_counts::d0")["shutdown"] == 1
    assert sim._engine is None and sim._viewer is None


def test_scenario_adapter_holds_no_engine_whose_setup_failed(tmp_path: Path, monkeypatch):
    """After a failed setup the next call builds a fresh engine rather than stepping the broken one."""
    from roqsim.engine import Engine

    real_setup = Engine.setup

    def failing_setup(self):
        raise RuntimeError("configure failed")

    monkeypatch.setattr(Engine, "setup", failing_setup)
    sim = MujocoSim(world=_write_world(tmp_path))
    sim.setup()
    with pytest.raises(RuntimeError, match="configure failed"):
        sim.reset()
    assert sim.context is None

    monkeypatch.setattr(Engine, "setup", real_setup)
    sim.step()
    assert sim.context.blackboard.get("dummy_counts::d0")["pre_step"] == 1
    sim.shutdown()


def test_scenario_adapter_records_each_scenario_into_its_own_directory(tmp_path: Path, monkeypatch):
    """A reset() told the scenario's output directory finishes the recording before it and
    starts one there; a reset() in the same directory is a reset within that recording."""
    from roqsim.mcap_format import is_finished

    world = _write_world(tmp_path)
    monkeypatch.setenv("ROQSIM_RECORD", "run.mcap")
    monkeypatch.delenv("RUN_OUTPUT_DIR", raising=False)
    sim = MujocoSim(world=world)
    sim.setup(output_dir=str(tmp_path / "out"))
    sim.reset(output_dir=str(tmp_path / "out" / "cfg" / "0"))
    for _ in range(3):
        sim.step()
    sim.reset(output_dir=str(tmp_path / "out" / "cfg" / "1"))
    first = tmp_path / "out" / "cfg" / "0" / "run.mcap"
    assert is_finished(first), "the first scenario's recording is finished"
    sim.reset(output_dir=str(tmp_path / "out" / "cfg" / "1"))
    for _ in range(3):
        sim.step()
    sim.shutdown()
    second = tmp_path / "out" / "cfg" / "1" / "run.mcap"
    assert is_finished(second)
    assert not (tmp_path / "out" / "run.mcap").exists()


def test_scenario_adapter_exports_the_scene_into_each_scenarios_directory(tmp_path: Path, monkeypatch):
    """A reset() into a new scenario directory exports the scene there too, though the world is
    not rebuilt; a reset() in the same directory does not export it again."""
    world = _write_world(tmp_path)
    monkeypatch.setenv("ROQSIM_SCENE_EXPORT_DIR", "scene")
    monkeypatch.delenv("RUN_OUTPUT_DIR", raising=False)
    sim = MujocoSim(world=world)
    sim.setup(output_dir=str(tmp_path / "out"))
    sim.reset(output_dir=str(tmp_path / "out" / "0"))
    sim.reset(output_dir=str(tmp_path / "out" / "1"))
    second = tmp_path / "out" / "1" / "scene" / "scene.glb"
    assert second.read_bytes()[:4] == b"glTF"
    second.unlink()
    sim.reset(output_dir=str(tmp_path / "out" / "1"))
    sim.shutdown()
    assert (tmp_path / "out" / "0" / "scene" / "scene.glb").is_file()
    assert not second.exists()


OTHER_WORLD = """
sim:
  timestep: 0.004
components:
  - dummy: {}
    name: d1
"""


def _write_other_world(tmp_path: Path) -> str:
    p = tmp_path / "other_world.yaml"
    p.write_text(OTHER_WORLD)
    return str(p)


def _recording_of(run_dir: Path) -> dict:
    import json

    from roqsim.mcap_format import META_KEY, META_RECORDING, is_finished, read_metadata

    path = run_dir / "run.mcap"
    assert is_finished(path), f"{path} is not a finished recording"
    return json.loads(read_metadata(path)[META_RECORDING][META_KEY])


def _count_builds(monkeypatch) -> list:
    from roqsim.engine import Engine

    built = []
    original = Engine.setup

    def counting_setup(self):
        built.append(self)
        return original(self)

    monkeypatch.setattr(Engine, "setup", counting_setup)
    return built


def test_scenario_adapter_runs_two_worlds_in_one_process_each_run_recorded_apart(
    tmp_path: Path, monkeypatch
):
    """A parameter file of two worlds, three runs each, in one process: the world changes once,
    every run has its own finished recording naming its world, the one seed and an episode no
    other run has -- the counter continues across the rebuild."""
    monkeypatch.setenv("ROQSIM_RECORD", "run.mcap")
    monkeypatch.delenv("RUN_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("ROQSIM_SEED", raising=False)
    world_a, world_b = _write_world(tmp_path), _write_other_world(tmp_path)
    built = _count_builds(monkeypatch)
    sim = MujocoSim(world=world_a)
    sim.setup(output_dir=str(tmp_path / "out"))
    runs = [("a", world_a, n) for n in range(3)] + [("b", world_b, n) for n in range(3)]
    for cell, world, n in runs:
        sim.reset(
            world=world, world_overrides={}, output_dir=str(tmp_path / "out" / cell / str(n))
        )
        for _ in range(3):
            sim.step()
    sim.shutdown()
    assert len(built) == 2, "one build per world, none per run"
    records = [_recording_of(tmp_path / "out" / cell / str(n)) for cell, _, n in runs]
    assert [r["world"] for r in records] == [world_a] * 3 + [world_b] * 3
    assert len({r["seed"] for r in records}) == 1, "the drawn seed is kept across the rebuild"
    assert [r["episode"] for r in records] == [1, 2, 3, 4, 5, 6]


def test_scenario_adapter_rebuilds_in_order_recording_finished_engine_down_built_reset_recording_started(
    tmp_path: Path, monkeypatch
):
    from roqsim.capture import StateRecorder
    from roqsim.engine import Engine

    monkeypatch.setenv("ROQSIM_RECORD", "run.mcap")
    monkeypatch.delenv("RUN_OUTPUT_DIR", raising=False)
    events: list[str] = []

    def record(cls, name, label):
        original = getattr(cls, name)

        def wrapped(self, *args, **kwargs):
            events.append(label)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, wrapped)

    record(StateRecorder, "close", "recording finished")
    record(StateRecorder, "__init__", "recording started")
    record(Engine, "shutdown", "engine shut down")
    record(Engine, "setup", "engine built")
    record(Engine, "reset", "engine reset")
    world_a, world_b = _write_world(tmp_path), _write_other_world(tmp_path)
    sim = MujocoSim(world=world_a)
    sim.reset(world=world_a, output_dir=str(tmp_path / "a"))
    sim.step()
    events.clear()
    sim.reset(world=world_b, output_dir=str(tmp_path / "b"))
    assert events == [
        "recording finished",
        "engine shut down",
        "engine built",
        "engine reset",
        "recording started",
    ]
    sim.shutdown()


def test_scenario_adapter_keeps_the_engine_when_a_documents_overrides_match_the_defaults(
    tmp_path: Path, monkeypatch
):
    """The runner asks ``dt`` before the first ``reset()``, which builds the process default; a
    first document naming that same world with the same overrides rebuilds nothing, and an
    empty mapping is "no overrides" rather than "inherit the default" or "rebuild"."""
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text("sim:\n  timestep: 0.004\n")
    monkeypatch.setenv("ROQSIM_WORLD_OVERRIDES", str(overrides))
    world = _write_world(tmp_path)
    built = _count_builds(monkeypatch)
    sim = MujocoSim(world=world)
    assert sim.dt == pytest.approx(0.004)
    sim.reset(
        world=world, world_overrides={"sim": {"timestep": 0.004}}, output_dir=str(tmp_path / "0")
    )
    assert len(built) == 1
    sim.reset(world=world, world_overrides={}, output_dir=str(tmp_path / "1"))
    assert len(built) == 2 and sim.dt == pytest.approx(0.005), "an empty mapping is no overrides"
    sim.reset(world=world, world_overrides=None, output_dir=str(tmp_path / "2"))
    assert len(built) == 3 and sim.dt == pytest.approx(0.004), "no mapping is the default"
    sim.shutdown()


def test_scenario_adapter_scene_export_lands_in_the_run_directory(tmp_path: Path, monkeypatch):
    """A relative ``ROQSIM_SCENE_EXPORT_DIR`` resolves against ``RUN_OUTPUT_DIR``, then ``output_dir``."""
    world = _write_world(tmp_path)
    run_dir, root = tmp_path / "run", tmp_path / "shared"
    monkeypatch.setenv("ROQSIM_SCENE_EXPORT_DIR", "scene")
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(run_dir))
    sim = MujocoSim(world=world)
    sim.setup(output_dir=str(root))
    sim.reset()
    sim.shutdown()
    assert (run_dir / "scene" / "scene.glb").read_bytes()[:4] == b"glTF"
    assert not (root / "scene").exists()

    # Without a per-run directory the scenario's own output_dir is the fallback.
    monkeypatch.delenv("RUN_OUTPUT_DIR")
    sim = MujocoSim(world=world)
    sim.setup(output_dir=str(root))
    sim.reset()
    sim.shutdown()
    assert (root / "scene" / "scene.glb").is_file()
