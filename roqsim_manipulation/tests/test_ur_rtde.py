"""``ur_rtde``: a program written against ``ur_rtde``'s control and receive interfaces drives the
simulated UR5e in lock-step with the physics, in the controller's base frame and conventions."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_manipulation.plugins.ur_rtde import RTDEError, _rotmat

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

HOME = [0.0, -1.3798, 1.8479, -1.6902, -1.571, -1.571]
TIMESTEP = 0.001  # two physics steps per 2 ms control period

# Universal Robots' DH parameters for the UR5e: the controller's own kinematics.
UR5E_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
UR5E_ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]


class _Block(Plugin):
    """A static block for the tool to press on; placed by the test."""

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        body = spec.worldbody.add_body(name="press_block", pos=[0, 0, -1.0])
        g = body.add_geom(name="press_block_geom")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.05, 0.05, 0.02]


TOOL = """<mujoco><worldbody><body name="probe">
  <geom type="box" size=".02 .02 .02" pos="0 0 .02" mass="0.2"/>
</body></worldbody></mujoco>"""


def _engine(block: bool = False, ft_frame: str = "world", tmp_path=None, **rtde) -> Engine:
    arm = {
        "model": "ur5e",
        "prefix": "ur5e_",
        "home": HOME,
        "actuators": {"control": "position", "p": 50000, "d": 450},
    }
    if tmp_path is not None:
        (tmp_path / "probe.xml").write_text(TOOL)
        arm["end_effector"] = {"model": "probe.xml"}
    components = [
        {
            "spawn_arm": arm,
            "name": "ur5e",
            "components": [
                {"force_torque": {"site": "fts_site", "frame": ft_frame}, "name": "ft"},
                {"ur_rtde": {"ft": "ft", **rtde}, "name": "rtde"},
            ],
        }
    ]
    if block:
        components.append({f"{__name__}:_Block": {}})
    engine = Engine(
        load_config_from_dict(
            {"sim": {"timestep": TIMESTEP}, "components": components}, base_dir=tmp_path
        )
    )
    engine.ctx.seed = 0
    engine.setup()
    for _ in range(200):
        engine.step()
    return engine


def _run(engine: Engine, program, *args, max_s: float = 10.0):
    """Run *program* to its end and return what it returned."""
    handle = engine.ctx.blackboard.get("ur_rtde:ur5e")
    handle.run(program, *args)
    for _ in range(int(max_s / TIMESTEP)):
        if not handle.running():
            break
        engine.step()
    engine.step()  # the step that sees the program has ended raises what it raised
    assert not handle.running(), "the program did not end"
    return handle.result()


def _dh(q) -> np.ndarray:
    T = np.eye(4)
    for th, d, a, al in zip(q, UR5E_D, UR5E_A, UR5E_ALPHA, strict=True):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        T = T @ np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ]
        )
    return T


def _hold(c, r, seconds):
    for _ in range(round(seconds / 0.002)):
        c.waitPeriod(c.initPeriod())


def test_the_tcp_pose_is_the_controllers_base_frame_and_kinematics():
    """With no TCP offset the pose is the flange's, as UR's DH kinematics put it in the base frame:
    ur_description's base_link turned half a turn, not the MJCF body it is taken from."""
    engine = _engine()
    try:
        q, pose = _run(engine, lambda c, r: (r.getActualQ(), r.getActualTCPPose()))
    finally:
        engine.shutdown()
    T = _dh(q)
    # The MJCF's flange sits 100 mm out of wrist 3 where UR's DH has 99.6 mm.
    assert np.linalg.norm(np.array(pose[:3]) - T[:3, 3]) < 1e-3
    R = _rotmat(pose[3:])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R.T @ T[:3, :3]) - 1) / 2)))
    assert angle < 1e-3


def test_a_control_period_is_one_period_of_sim_time():
    engine = _engine()
    try:
        t0 = engine.ctx.sim_time

        def loop(c, r):
            stamps = []
            for _ in range(100):
                start = c.initPeriod()
                stamps.append(start)
                c.waitPeriod(start)
            return stamps

        stamps = _run(engine, loop)
    finally:
        engine.shutdown()
    assert stamps[0] == pytest.approx(t0)
    assert np.allclose(np.diff(stamps), 0.002, atol=1e-9)


def _descend(c, r, seconds):
    poses, speeds = [], []
    for _ in range(round(seconds / 0.002)):
        t0 = c.initPeriod()
        c.speedL([0.0, 0.0, -0.02, 0.0, 0.0, 0.0], 0.5)
        poses.append(r.getActualTCPPose())
        speeds.append(r.getTargetTCPSpeed())
        c.waitPeriod(t0)
    c.speedStop(1.0)
    _hold(c, r, 0.2)
    poses.append(r.getActualTCPPose())
    return np.array(poses), np.array(speeds), r.getActualTCPSpeed()


def test_speedl_ramps_at_its_acceleration_and_keeps_a_straight_line():
    engine = _engine()
    try:
        poses, speeds, rest = _run(engine, _descend, 1.0)
    finally:
        engine.shutdown()
    # The commanded speed reaches 20 mm/s after 0.02 / 0.5 = 40 ms (plus one period of delay), and
    # not before.
    assert speeds[10, 2] > -0.02 + 1e-6
    assert speeds[30, 2] == pytest.approx(-0.02, abs=1e-9)
    travel = poses[-1, :3] - poses[0, :3]
    # 1 s at 20 mm/s less the ramps and the delay, about 19 mm down.
    assert -0.0205 < travel[2] < -0.0180
    assert np.linalg.norm(travel[:2]) < 5e-4
    assert np.allclose(poses[-1, 3:], poses[0, 3:], atol=2e-3)
    assert np.linalg.norm(rest[:3]) < 1e-3


def test_movel_blocks_until_the_tcp_has_arrived():
    engine = _engine()
    try:

        def move(c, r):
            start = np.array(r.getActualTCPPose())
            goal = start + [0.05, 0.0, -0.03, 0.0, 0.0, 0.0]
            t0 = r.getTimestamp()
            c.moveL(goal.tolist(), 0.1, 0.5)
            t1 = r.getTimestamp()
            _hold(c, r, 0.3)
            return start, goal, t1 - t0, np.array(r.getActualTCPPose())

        start, goal, took, end = _run(engine, move)
    finally:
        engine.shutdown()
    dist = math.hypot(0.05, 0.03)
    profile = 2 * 0.1 / 0.5 + (dist - 0.1 * 0.1 / 0.5) / 0.1
    assert took == pytest.approx(profile + 0.002, abs=0.003)
    assert np.linalg.norm(end[:3] - goal[:3]) < 1e-3
    assert np.allclose(end[3:], start[3:], atol=2e-3)


def test_the_contact_force_is_the_environments_on_the_tool_in_the_base_frame(tmp_path):
    """A 40 mm probe pressing down on a block: the TCP force reads up (+z of the base), as the
    force_torque reader does in the world, and a tare in free air zeroes it."""
    engine = _engine(block=True, tmp_path=tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    tip = np.array(d.site_xpos[m.site("ur5e_tool_site").id])
    assert (
        np.array(d.site_xmat[m.site("ur5e_tool_site").id]).reshape(3, 3)[2, 2] < -0.9
    )  # tool down, 20 deg off
    m.body_pos[m.body("press_block").id] = tip - [0.0, 0.0, 0.04 + 0.02 + 0.01]
    mujoco.mj_forward(m, d)
    reader = engine.ctx.blackboard.get("ft:ft")
    try:

        def press(c, r):
            c.zeroFtSensor()
            _hold(c, r, 0.01)
            tared = r.getActualTCPForce()
            for _ in range(2000):
                t0 = c.initPeriod()
                if r.getActualTCPForce()[2] > 10.0:
                    break
                c.speedL([0.0, 0.0, -0.01, 0.0, 0.0, 0.0], 0.5)
                c.waitPeriod(t0)
            c.speedStop(2.0)
            _hold(c, r, 0.05)
            return tared, r.getActualTCPForce()

        tared, pressed = _run(engine, press)
        world, _ = reader.read()
    finally:
        engine.shutdown()
    assert np.linalg.norm(tared[:3]) < 0.5
    assert pressed[2] > 10.0
    assert pressed[2] == pytest.approx(world[2], rel=0.05)


def test_a_program_that_never_waits_stops_the_simulation():
    engine = _engine(client_timeout_s=0.3)
    handle = engine.ctx.blackboard.get("ur_rtde:ur5e")

    def spin(c, r):
        while True:
            r.getActualTCPPose()

    try:
        handle.run(spin)
        with pytest.raises(RuntimeError, match="waitPeriod"):
            engine.step()
    finally:
        engine.shutdown()


def test_what_the_program_raises_is_raised_by_the_simulation():
    engine = _engine()

    def broken(c, r):
        _hold(c, r, 0.01)
        raise ValueError("a bug in the law")

    try:
        with pytest.raises(RuntimeError, match="a bug in the law"):
            _run(engine, broken)
    finally:
        engine.shutdown()


def test_a_reset_ends_the_program():
    engine = _engine()
    handle = engine.ctx.blackboard.get("ur_rtde:ur5e")
    calls = []

    def forever(c, r):
        try:
            while True:
                t0 = c.initPeriod()
                c.speedL([0.0, 0.0, -0.01, 0.0, 0.0, 0.0], 0.5)
                c.waitPeriod(t0)
        except RTDEError:
            calls.append("closed")
            raise

    try:
        handle.run(forever)
        for _ in range(20):
            engine.step()
        engine.reset()
        assert not handle.running() and calls == ["closed"]
        for _ in range(20):
            engine.step()  # nothing waits on, or raises from, the ended program
        assert handle.result() is None
    finally:
        engine.shutdown()


def test_a_run_is_reproducible():
    def final_q():
        engine = _engine()
        try:
            _run(engine, _descend, 0.3)
            return np.array(engine.ctx.data.qpos)
        finally:
            engine.shutdown()

    assert np.array_equal(final_q(), final_q())


def test_a_wrench_not_in_the_world_frame_is_refused():
    with pytest.raises(RuntimeError, match="frame: world"):
        _engine(ft_frame="sensor")


def test_an_unkicked_watchdog_shuts_the_control_down():
    engine = _engine()
    handle = engine.ctx.blackboard.get("ur_rtde:ur5e")

    def kicks_then_forgets(c, r):
        c.setWatchdog(10.0)
        for i in range(200):
            t0 = c.initPeriod()
            c.speedL([0.0, 0.0, -0.01, 0.0, 0.0, 0.0], 0.5)
            if i < 100:  # 0.2 s kicked, then not
                c.kickWatchdog()
            c.waitPeriod(t0)
        return "never reached"

    try:
        handle.run(kicks_then_forgets)
        for _ in range(int(0.6 / TIMESTEP)):
            engine.step()
        assert not handle.running()
        with pytest.raises(RTDEError, match="watchdog"):
            handle.result()
        # The control is down at 0.2 + 0.1 s; the arm decelerates and stays at rest.
        _, _, qd, _ = engine.ctx.blackboard.get("arm:ur5e").read_state()
        assert np.max(np.abs(qd)) < 1e-3
    finally:
        engine.shutdown()


def test_speedl_with_a_time_is_refused():
    engine = _engine()
    try:
        with pytest.raises(RuntimeError, match="not emulated"):
            _run(engine, lambda c, r: c.speedL([0, 0, 0.01, 0, 0, 0], 0.5, 0.1))
    finally:
        engine.shutdown()
