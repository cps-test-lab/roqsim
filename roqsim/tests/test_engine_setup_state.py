"""After ``setup()`` the data describes the world's initial state, and that changes no run.

A fresh ``MjData`` holds zero orientations and every body, geom and site at the origin; code that
inspects a world straight after ``setup()`` -- a geometry check, an export, a test -- would compare
nothing, so ``setup()`` ends with ``mj_forward``. It advances no time and ``reset()`` starts from
``mj_resetData``, so a run is the same, byte for byte, as one whose setup skips the pass: the two
runs' data and recordings are compared field by field.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import roqsim.engine as engine_module
from roqsim.capture import StateRecorder, snap_fps
from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.recording import open_recording

#: Away from the origin and turned, so neither position nor orientation is a default value.
AWAY = (2.0, -1.0, 0.5)
AWAY_YAW = 0.7


class _World(Plugin):
    """A turned body away from the origin, a falling ball, a driven hinge, a mocap body, sensors."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, [0.0, 0.0, 1.0], AWAY_YAW)
        away = spec.worldbody.add_body(name="away", pos=AWAY, quat=q)
        away.add_geom(name="away_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.1, 0.05])
        away.add_site(name="away_site", pos=[0.2, 0.0, 0.0])
        away.add_camera(name="away_cam", pos=[0.0, 0.0, 0.3])
        arm = away.add_body(name="arm", pos=[0.0, 0.0, 0.1])
        arm.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 1.0, 0.0])
        arm.add_geom(
            name="arm_rod",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0.3, 0, 0],
            size=[0.02],
        )
        arm.add_site(name="arm_tip", pos=[0.3, 0.0, 0.0])
        act = spec.add_actuator(name="hinge_pos", target="hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=20.0, kv=1.0)
        ball = spec.worldbody.add_body(name="ball", pos=[0.5, 0.5, 1.0])
        ball.add_freejoint()
        ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1])
        spec.worldbody.add_body(name="carried", mocap=True, pos=[-1.0, 2.0, 0.3]).add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], contype=0, conaffinity=0
        )
        spec.add_sensor(
            name="tip_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="arm_tip",
        )
        spec.add_sensor(
            name="tip_acc",
            type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            objname="arm_tip",
            objtype=mujoco.mjtObj.mjOBJ_SITE,
        )
        spec.add_sensor(
            name="hinge_q",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objname="hinge",
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
        )

    def on_reset(self, ctx: SimContext) -> None:
        ctx.data.ctrl[ctx.model.actuator("hinge_pos").id] = 0.6

    def pre_step(self, ctx: SimContext) -> None:
        m, d = ctx.model, ctx.data
        d.mocap_pos[m.body("carried").mocapid[0]][0] += 0.001
        # Reads a derived quantity before the step: what a controller closing a loop on a pose does.
        d.ctrl[m.actuator("hinge_pos").id] = 0.6 - 0.1 * d.site_xpos[m.site("arm_tip").id][2]


def _engine() -> Engine:
    engine = Engine(
        load_config_from_dict(
            {"sim": {"timestep": 0.002}, "components": [{f"{__name__}:_World": {}}]}
        )
    )
    engine.ctx.seed = 1
    return engine


class _WithoutForward:
    """``mujoco`` as the engine module sees it, with ``mj_forward`` doing nothing."""

    def __getattr__(self, name):
        return getattr(mujoco, name)

    @staticmethod
    def mj_forward(model, data) -> None:
        return None


def _setup(engine: Engine, monkeypatch, *, without_the_setup_pass: bool) -> None:
    if without_the_setup_pass:
        monkeypatch.setattr(engine_module, "mujoco", _WithoutForward())
    try:
        engine.setup()
    finally:
        monkeypatch.undo()


def test_after_setup_a_body_away_from_the_origin_is_posed(monkeypatch):
    engine = _engine()
    engine.setup()
    try:
        m, d = engine.ctx.model, engine.ctx.data
        assert d.time == 0.0
        np.testing.assert_allclose(d.geom_xpos[m.geom("away_box").id], AWAY, atol=1e-12)
        expected = np.zeros(9)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, [0.0, 0.0, 1.0], AWAY_YAW)
        mujoco.mju_quat2Mat(expected, q)
        np.testing.assert_allclose(d.geom_xmat[m.geom("away_box").id], expected, atol=1e-12)
        np.testing.assert_allclose(
            d.site_xpos[m.site("away_site").id],
            np.asarray(AWAY) + expected.reshape(3, 3) @ [0.2, 0.0, 0.0],
            atol=1e-12,
        )
        assert d.cam_xpos[m.camera("away_cam").id][2] == pytest.approx(AWAY[2] + 0.3)
        np.testing.assert_allclose(d.xpos[m.body("carried").id], [-1.0, 2.0, 0.3], atol=1e-12)
        # The sensors read the same state: the hinge tip's position, not zeros.
        np.testing.assert_allclose(d.sensor("tip_pos").data, d.site_xpos[m.site("arm_tip").id])
        posed = {f: np.array(getattr(d, f)) for f in ("xpos", "xmat", "geom_xpos", "site_xpos")}
        # The pose reset() gives this world, which writes only ctrl in on_reset: setup's is the same.
        engine.reset()
        for field, before in posed.items():
            np.testing.assert_array_equal(getattr(d, field), before, err_msg=field)
    finally:
        engine.shutdown()


def test_without_the_setup_pass_the_data_is_at_the_origin(monkeypatch):
    # What the first test guards against, shown: the same world without the pass has every geom at
    # the origin.
    engine = _engine()
    _setup(engine, monkeypatch, without_the_setup_pass=True)
    try:
        assert not engine.ctx.data.geom_xpos.any()
    finally:
        engine.shutdown()


_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc",
    "qacc_warmstart",
    "mocap_pos",
    "mocap_quat",
    "sensordata",
    "xpos",
    "xquat",
    "geom_xpos",
    "site_xpos",
    "cam_xpos",
)


def _run(tmp_path, monkeypatch, *, without_the_setup_pass: bool, steps: int = 300):
    engine = _engine()
    _setup(engine, monkeypatch, without_the_setup_pass=without_the_setup_pass)
    try:
        engine.reset()
        recorder = StateRecorder(engine.ctx, tmp_path / "run.mcap", snap_fps(100, engine.dt))
        for _ in range(steps):
            engine.step()
            recorder.sample(engine.ctx)
        path = recorder.close()
        data = {f: np.array(getattr(engine.ctx.data, f)).tobytes() for f in _FIELDS}
    finally:
        engine.shutdown()
    with open_recording(path) as rec:
        recorded = (
            rec.samples["s"].tobytes(),
            rec.times.tobytes(),
            # Without "w", the wall clock, which no two runs share.
            [{k: v for k, v in rec.poses(i).items() if k != "w"} for i in range(len(rec))],
            [{k: v for k, v in rec.joints(i).items() if k != "w"} for i in range(len(rec))],
        )
    return data, recorded


def test_the_setup_pass_changes_no_run_and_no_recording(tmp_path, monkeypatch):
    # setup, reset, step: as every driver runs a world. The world's pre_step reads a pose, and still
    # not a byte of state or recording moves.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    before = _run(tmp_path / "a", monkeypatch, without_the_setup_pass=True)
    after = _run(tmp_path / "b", monkeypatch, without_the_setup_pass=False)
    for field in _FIELDS:
        assert after[0][field] == before[0][field], field
    assert after[1] == before[1]
