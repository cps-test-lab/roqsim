# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``step_timers`` and ``--profile``: where a physics step's time goes.

The scene is a box dropped onto the floor, so the step has contacts to collide and a constraint to
solve, and is still a handful of bodies. The plugin owns a process-global MuJoCo callback, so every
test shuts its engine down in a ``finally`` and the fixture below checks nothing is left installed;
a leaked callback would make every later test that configures the plugin refuse.
"""

from __future__ import annotations

import csv

import mujoco
import numpy as np
import pytest

from roqsim import runner
from roqsim.config import PluginError, load_config_from_dict
from roqsim.engine import PHYSICS_GROUP, Engine

STEPS = 100
DT = 0.002


@pytest.fixture(autouse=True)
def _no_time_callback():
    mujoco.set_mjcb_time(None)
    yield
    leaked = mujoco.get_mjcb_time()
    mujoco.set_mjcb_time(None)
    assert leaked is None, "a test left a MuJoCo time callback installed"


def _engine(*extra: dict, profile: bool = False) -> Engine:
    components = [{"dummy": {"size": 0.1}, "name": "box"}, *extra]
    cfg = load_config_from_dict({"sim": {"timestep": DT}, "components": components})
    engine = Engine(cfg, profile=profile)
    engine.ctx.seed = 0
    return engine


def _timers(path) -> dict:
    return {"step_timers": {"every_s": 0.05, "path": str(path)}}


def _run(engine: Engine, steps: int = STEPS) -> np.ndarray:
    """Step a set-up-and-reset engine, returning the qpos trajectory; shuts it down either way."""
    try:
        engine.reset()
        qpos = []
        for _ in range(steps):
            engine.step()
            qpos.append(engine.ctx.data.qpos.copy())
        return np.array(qpos)
    finally:
        engine.shutdown()


def _rows(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_rows_carry_nonzero_step_and_forward(tmp_path):
    out = tmp_path / "timers.csv"
    engine = _engine(_timers(out))
    engine.setup()
    _run(engine)
    rows = _rows(out)
    # One row per 0.05 s of a 0.2 s run (t = 0.002 first), plus the final one from shutdown.
    assert len(rows) >= 5
    assert float(rows[-1]["t"]) == pytest.approx(STEPS * DT)
    last = rows[-1]
    assert float(last["step_us"]) > 0.0 and int(last["step_n"]) == STEPS
    assert float(last["forward_us"]) > 0.0 and int(last["forward_n"]) >= STEPS
    assert float(last["wall_s"]) > 0.0
    # Cumulative within an episode: a later row never reports less.
    steps = [float(r["step_us"]) for r in rows]
    assert steps == sorted(steps)
    columns = {"episode", "position_us", "constraint_us", "pos_collision_us", "col_narrow_n"}
    assert columns <= set(last)


def test_relative_path_lands_in_the_run_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    engine = _engine({"step_timers": {}})
    engine.setup()
    _run(engine, steps=5)
    assert (tmp_path / "run" / "step_timers.csv").is_file()


def test_refuses_when_a_time_callback_is_already_installed(tmp_path):
    def other_clock() -> float:
        return 0.0

    mujoco.set_mjcb_time(other_clock)
    try:
        engine = _engine(_timers(tmp_path / "timers.csv"))
        with pytest.raises(RuntimeError, match="already installed"):
            engine.setup()
        # Refused, not replaced: the other owner's callback is still the one installed.
        assert mujoco.get_mjcb_time() is other_clock
    finally:
        mujoco.set_mjcb_time(None)


def test_uninstalls_the_callback_at_shutdown(tmp_path):
    engine = _engine(_timers(tmp_path / "timers.csv"))
    engine.setup()
    try:
        assert mujoco.get_mjcb_time() is not None
    finally:
        _run(engine, steps=3)
    assert mujoco.get_mjcb_time() is None


def test_the_callback_does_not_change_physics(tmp_path):
    plain = _engine()
    plain.setup()
    without = _run(plain)
    timed = _engine(_timers(tmp_path / "timers.csv"))
    timed.setup()
    with_timers = _run(timed)
    # The box fell and landed, so the comparison covers contact and the solver, not a free fall.
    assert without[0, 2] != without[-1, 2]
    np.testing.assert_array_equal(without, with_timers)


def test_every_s_must_be_positive(tmp_path):
    with pytest.raises(PluginError, match="every_s"):
        _engine({"step_timers": {"every_s": 0.0, "path": str(tmp_path / "t.csv")}})


def test_profile_times_the_physics_step():
    engine = _engine(profile=True)
    engine.setup()
    _run(engine, steps=5)
    report = engine.timing_report()
    assert set(report[PHYSICS_GROUP]) == {"mj_step"}
    assert report[PHYSICS_GROUP]["mj_step"] > 0.0
    assert "mj_step" in engine.format_timing()


def test_physics_step_is_not_timed_without_profile():
    engine = _engine()
    engine.setup()
    _run(engine, steps=5)
    assert engine.timing_report() == {}


def test_sim_profile_reports_the_physics_step(tmp_path, capsys):
    world = tmp_path / "world.yaml"
    world.write_text(f"sim:\n  timestep: {DT}\ncomponents:\n  - dummy: {{}}\n    name: d0\n")
    rc = runner.main(
        [str(world), "--headless", "--steps", "10", "--pacing", "asap", "--seed", "0", "--profile"]
    )
    assert rc in (0, None)
    err = capsys.readouterr().err
    line = next(ln for ln in err.splitlines() if "mj_step" in ln)
    assert PHYSICS_GROUP in line and line.rstrip().endswith("µs")
