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
    sim.reset()
    sim.step()
    assert sim.context.blackboard.get("dummy_counts::d0")["pre_step"] == 1
    sim.shutdown()


def test_scenario_adapter_step_before_reset_raises_and_builds_nothing(tmp_path: Path):
    sim = MujocoSim(world=_write_world(tmp_path))
    sim.setup()
    with pytest.raises(RuntimeError, match=r"call reset\(\) first"):
        sim.step()
    assert sim.context is None
    sim.reset()
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


def test_scenario_adapter_exports_the_scene_into_each_scenarios_directory(
    tmp_path: Path, monkeypatch
):
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


# -- a run's name, as the runner gives it ------------------------------------------------------

PROBE = '''
from roqsim.plugin import Plugin


class Probe(Plugin):
    """Emits the draws a plugin of this world makes: one held for the run, two keyed on time."""

    def on_reset(self, ctx):
        values = ctx.rng_for("probe", per_run=True).standard_normal(3).tolist()
        ctx.events.emit("probe", "drawn", values=values)

    def post_step(self, ctx):
        if round(ctx.sim_time / ctx.dt) in (1, 5):
            ctx.events.emit("probe", "drawn", values=ctx.rng_for("probe").standard_normal(3).tolist())
'''


def _probe_worlds(tmp_path: Path) -> dict[str, str]:
    """Two worlds that differ only in their probe's name, which its draw key does not carry:
    nothing but the run tells their draws apart."""
    (tmp_path / "probe.py").write_text(PROBE)
    worlds = {}
    for name in ("a", "b"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            f"sim:\n  timestep: 0.005\ncomponents:\n  - probe.py:Probe: {{}}\n    name: {name}\n"
        )
        worlds[name] = str(path)
    return worlds


def _drawn(ctx, run: str) -> list:
    return [e["values"] for e in ctx.events.all() if e["event"] == "drawn" and e["run"] == run]


def _run_documents(sim, worlds, documents) -> dict[str, list]:
    """Run each ``(world, run)`` as scenario-execution runs a parameter file: a reset, then steps."""
    drawn = {}
    for world, name in documents:
        sim.reset(world=worlds[world], run=name)
        for _ in range(6):
            sim.step()
        drawn[sim.context.run] = _drawn(sim.context, sim.context.run)
    return drawn


def test_runs_of_two_worlds_in_one_process_never_draw_the_same_values(tmp_path, monkeypatch):
    """The worst case: one seed for the whole process, and a world rebuilt between configurations.
    Only the run's name tells the runs apart, so every draw of every run must differ."""
    monkeypatch.setenv("ROQSIM_SEED", "7")
    monkeypatch.delenv("ROQSIM_RUN", raising=False)
    worlds = _probe_worlds(tmp_path)
    documents = [("a", "a/0"), ("a", "a/1"), ("b", "b/0"), ("b", "b/1"), ("a", "a/2")]
    sim = MujocoSim(world=worlds["a"])
    sim.setup()
    try:
        drawn = _run_documents(sim, worlds, documents)
    finally:
        sim.shutdown()
    assert list(drawn) == [run for _, run in documents]
    values = [tuple(v) for draws in drawn.values() for v in draws]
    assert len(values) == 3 * len(documents)
    assert len(set(values)) == len(values), "two draws of the process were the same"


def test_unnamed_runs_keep_counting_across_a_rebuild(tmp_path, monkeypatch):
    monkeypatch.setenv("ROQSIM_SEED", "7")
    monkeypatch.delenv("ROQSIM_RUN", raising=False)
    worlds = _probe_worlds(tmp_path)
    sim = MujocoSim(world=worlds["a"])
    sim.setup()
    try:
        drawn = _run_documents(sim, worlds, [(w, None) for w in ("a", "a", "b", "b", "a")])
    finally:
        sim.shutdown()
    assert list(drawn) == ["#0", "#1", "#2", "#3", "#4"]
    values = [tuple(v) for draws in drawn.values() for v in draws]
    assert len(set(values)) == len(values)


def test_the_same_seed_and_run_repeat_a_run_in_another_process(tmp_path, monkeypatch):
    """What a packed process drew for run b/1, a fresh adapter and `roqsim sim --run b/1` draw
    again, whatever ran before it."""
    monkeypatch.setenv("ROQSIM_SEED", "7")
    monkeypatch.delenv("ROQSIM_RUN", raising=False)
    worlds = _probe_worlds(tmp_path)
    packed = MujocoSim(world=worlds["a"])
    packed.setup()
    try:
        drawn = _run_documents(packed, worlds, [("a", "a/0"), ("b", "b/0"), ("b", "b/1")])
    finally:
        packed.shutdown()

    alone = MujocoSim(world=worlds["b"])
    alone.setup()
    try:
        again = _run_documents(alone, worlds, [("b", "b/1")])
    finally:
        alone.shutdown()
    assert again["b/1"] == drawn["b/1"]

    engine = run(worlds["b"], headless=True, pacing="asap", max_steps=6, seed=7, run_name="b/1")
    assert _drawn(engine.ctx, "b/1") == drawn["b/1"]


def test_a_reset_s_own_name_beats_the_deployment_s(tmp_path, monkeypatch):
    monkeypatch.setenv("ROQSIM_RUN", "job/0")
    sim = MujocoSim(world=_write_world(tmp_path))
    sim.setup()
    try:
        sim.reset()
        assert sim.context.run == "job/0", "ROQSIM_RUN names the first run"
        sim.reset()
        assert sim.context.run == "job/0#1"
        sim.reset(run="cfg/3")
        assert sim.context.run == "cfg/3", "a document's run beats the deployment's"
    finally:
        sim.shutdown()


def test_a_run_name_that_cannot_be_one_is_refused_before_anything_is_built(tmp_path):
    sim = MujocoSim(world=_write_world(tmp_path))
    sim.setup()
    with pytest.raises(ValueError, match="empty"):
        sim.reset(run="")
    assert sim._engine is None
    sim.shutdown()


def test_roqsim_sim_names_its_run_from_the_flag_else_the_deployment(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    monkeypatch.setenv("ROQSIM_RUN", "job/7")
    assert run(world, headless=True, pacing="asap", max_steps=2, seed=1).ctx.run == "job/7"
    named = run(world, headless=True, pacing="asap", max_steps=2, seed=1, run_name="x/1")
    assert named.ctx.run == "x/1"
    monkeypatch.delenv("ROQSIM_RUN")
    assert run(world, headless=True, pacing="asap", max_steps=2, seed=1).ctx.run == "#0"


def test_roqsim_sim_refuses_a_run_name_that_cannot_be_one(tmp_path, capsys):
    from roqsim.runner import main

    with pytest.raises(SystemExit) as exc:
        main([_write_world(tmp_path), "--headless", "--steps", "1", "--run", ""])
    assert exc.value.code == 2
    assert "empty" in capsys.readouterr().err
