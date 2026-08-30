"""Phase 2: pacing, the standalone runner (headless), and the scenario-execution adapter."""

from __future__ import annotations

from pathlib import Path

from roqsim.clock import SHORTFALL_REPORT_SHARE, Pacer
from roqsim.config import load_config_from_dict
from roqsim.runner import run
from roqsim.scenario_adapter import MujocoSim

DUMMY_WORLD = """
sim:
  timestep: 0.005
plugins:
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
# stops that absorption from being silent -- a run that held 0.34x realtime for five minutes
# reported nothing until its job deadline killed it, and the shortfall had to be
# reconstructed from run.clock_map.csv afterwards.


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
    cfg = load_config_from_dict({"sim": {"view": {"azimuth": 42}}, "plugins": []})
    assert cfg.view == {"azimuth": 42}
    assert load_config_from_dict({"plugins": []}).view == {}


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
