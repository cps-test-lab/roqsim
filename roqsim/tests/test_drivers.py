"""Phase 2: pacing, the standalone runner (headless), and the scenario-execution adapter."""

from __future__ import annotations

from pathlib import Path

from roqsim.clock import Pacer
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
