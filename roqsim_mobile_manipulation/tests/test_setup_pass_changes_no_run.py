"""``Engine.setup()`` ends with ``mj_forward``; a shipped world's run and recording do not change by it.

The worlds are a wheeled base whose lidar draws noise and a base with an arm, each run as a driver
runs it -- setup, reset, step -- once as the engine builds it and once with the setup's forward pass
taken out, and compared byte for byte: the state, the sensor data and what a recording holds.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import roqsim.engine as engine_module
from roqsim.capture import StateRecorder, snap_fps
from roqsim.config import drop_transport, load_config
from roqsim.engine import Engine
from roqsim.recording import open_recording
from roqsim.world import resolve_world_yaml_ref

WORLDS = ["roqsim_mobile:turtlebot3_demo", "roqsim_mobile_manipulation:frankie_demo"]
FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "sensordata", "xpos")


class _WithoutForward:
    """``mujoco`` as the engine module sees it, with ``mj_forward`` doing nothing."""

    def __getattr__(self, name):
        return getattr(mujoco, name)

    @staticmethod
    def mj_forward(model, data) -> None:
        return None


def _run(world: str, out, monkeypatch, *, without_the_setup_pass: bool, steps: int = 400):
    cfg = load_config(resolve_world_yaml_ref(world))
    drop_transport(cfg)
    engine = Engine(cfg)
    engine.ctx.seed = 3
    if without_the_setup_pass:
        monkeypatch.setattr(engine_module, "mujoco", _WithoutForward())
    try:
        engine.setup()
    finally:
        monkeypatch.undo()
    try:
        engine.reset()
        recorder = StateRecorder(engine.ctx, out / "run.mcap", snap_fps(50, engine.dt))
        for _ in range(steps):
            engine.step()
            recorder.sample(engine.ctx)
        path = recorder.close()
        state = {f: np.array(getattr(engine.ctx.data, f)).tobytes() for f in FIELDS}
    finally:
        engine.shutdown()
    with open_recording(path) as rec:
        # Without "w", the wall clock, which no two runs share.
        recorded = (
            rec.samples["s"].tobytes(),
            rec.times.tobytes(),
            [{k: v for k, v in rec.poses(i).items() if k != "w"} for i in range(len(rec))],
            [{k: v for k, v in rec.joints(i).items() if k != "w"} for i in range(len(rec))],
        )
    return state, recorded


@pytest.mark.parametrize("world", WORLDS)
def test_a_shipped_world_runs_and_records_the_same(tmp_path, monkeypatch, world):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    before = _run(world, tmp_path / "a", monkeypatch, without_the_setup_pass=True)
    after = _run(world, tmp_path / "b", monkeypatch, without_the_setup_pass=False)
    for field in FIELDS:
        assert after[0][field] == before[0][field], field
    assert after[1] == before[1]
