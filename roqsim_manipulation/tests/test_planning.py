# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""roqsim_manipulation.planning: IK, collision-free paths, time-optimal timing, margins, holds.

A UR5e with a Robotiq Hand-E on the floor, a thin wall in front of it. The planner must reach poses
exactly or refuse them, go over the wall rather than through it, time a path within its limits from
rest to rest as fast as they allow, retime it when a torque limit says so, and measure a pose's
margins and a gripper's hold by what the physics says.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_manipulation.planning import (
    CollisionChecker,
    HoldMonitor,
    IKError,
    PlanningError,
    cartesian_path,
    pads_touch,
    plan_path,
    pose_margins,
    time_parameterize,
)
from roqsim_manipulation.planning.margins import ArmDynamics, joint_torque_limits

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
#: The tool pointing straight down, its fingers closing along world x.
DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
VMAX = [math.pi] * 6
AMAX = [8.0] * 6


def _world(extra=(), gravity=None):
    sim = {"timestep": 0.002, "cone": "elliptic", "impratio": 10, "noslip_iterations": 10}
    if gravity is not None:
        sim["gravity"] = gravity
    return {"sim": sim, "components": [
        {"spawn_arm": {"model": "ur5e", "prefix": "ur5e_", "end_effector": {"model": "robotiq_hande"}},
         "components": [{"arm_controller": {"trajectory_feedforward": "dynamics"}}], "name": "ur5e"},
        *extra]}


WALL = {"box": {"prefix": "wall_", "pose": {"position": {"x": 0.45, "y": 0.0, "z": 0.3}},
                "size": [0.4, 0.05, 0.4], "motion": "static"}, "name": "wall"}


@pytest.fixture
def cell():
    engine = Engine(load_config_from_dict(_world([WALL])))
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    joints = [m.joint("ur5e_" + n).id for n in JOINTS]
    checker = CollisionChecker(m, d, joints, m.site("ur5e_pinch").id, m.body("ur5e_base").id,
                               clearance=0.01)
    yield engine, checker
    engine.shutdown()


def test_ik_reaches_a_pose_exactly_and_refuses_one_out_of_reach(cell):
    _, checker = cell
    kin = checker.kin
    q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    q = kin.ik([0.4, 0.3, 0.3], DOWN, q0, rng=np.random.default_rng(1))
    pos, rot = kin.fk(q)
    assert np.linalg.norm(pos - [0.4, 0.3, 0.3]) < 1e-4
    angle = np.linalg.norm(
        np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / 2
    )
    assert np.degrees(np.arcsin(min(1.0, angle))) < 0.1 and np.trace(rot.T @ DOWN) > 2.99
    assert kin.within_limits(q)
    with pytest.raises(IKError, match="no configuration"):
        kin.ik([1.2, 0.0, 0.3], DOWN, q0, rng=np.random.default_rng(1), restarts=5)


def test_ik_stays_on_the_branch_nearest_its_seed(cell):
    _, checker = cell
    kin = checker.kin
    seed = np.array([0.5, -1.2, 1.4, -1.77, -1.57, 0.5])
    pos, rot = kin.fk(seed)
    q = kin.ik(pos + [0.02, 0.0, 0.0], rot, seed)
    assert np.max(np.abs(q - seed)) < 0.15


def test_a_path_goes_over_the_wall_and_is_free_everywhere(cell):
    _, checker = cell
    kin, rng = checker.kin, np.random.default_rng(3)
    q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    qa = kin.ik([0.45, 0.3, 0.25], DOWN, q0, rng=rng, accept=checker.free)
    qb = kin.ik([0.45, -0.3, 0.25], DOWN, qa, rng=rng, accept=checker.free)
    assert not checker.segment_free(qa, qb), "the wall must be in the way of the straight segment"
    path = plan_path(checker, qa, qb, rng)
    assert np.allclose(path[0], qa) and np.allclose(path[-1], qb)
    for a, b in zip(path[:-1], path[1:], strict=True):
        assert checker.segment_free(a, b, step=0.01)
    timed = time_parameterize(path, VMAX, AMAX, checker=checker)
    for q in timed.positions:
        assert checker.free(q)


def test_an_end_in_collision_is_refused_by_name(cell):
    _, checker = cell
    inside = checker.kin.ik([0.45, 0.0, 0.3], DOWN, np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]))
    with pytest.raises(PlanningError, match="goal configuration is in collision: .*wall_box"):
        plan_path(checker, np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]), inside,
                  np.random.default_rng(0))


def test_one_joint_moves_on_the_time_optimal_trapezoid():
    """1 rad at v 1 rad/s and a 2 rad/s^2: 0.5 s up, 0.5 s cruise, 0.5 s down -- 1.5 s."""
    start, goal = np.zeros(6), np.array([1.0, 0, 0, 0, 0, 0])
    timed = time_parameterize([start, goal], [1.0] * 6, [2.0] * 6, grid=0.002)
    assert timed.duration == pytest.approx(1.5, rel=0.01)
    assert np.max(np.abs(timed.velocities[:, 0])) == pytest.approx(1.0, rel=0.01)
    assert np.max(np.abs(timed.accelerations[:, 0])) <= 2.0 * 1.01
    assert np.allclose(timed.velocities[[0, -1]], 0.0)
    assert np.allclose(timed.positions[-1], goal)


def test_a_timed_path_keeps_every_joint_within_its_limits(cell):
    _, checker = cell
    path = [np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]), np.array([1.2, -1.0, 1.2, -1.8, -1.0, 2.0]),
            np.array([2.0, -1.4, 1.8, -1.4, -1.57, 0.5])]
    vmax, amax = np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0]), np.array([5.0, 4.0, 6.0, 8.0, 8.0, 8.0])
    timed = time_parameterize(path, vmax, amax)
    assert np.all(np.max(np.abs(timed.velocities), axis=0) <= vmax * 1.01)
    assert np.all(np.max(np.abs(timed.accelerations), axis=0) <= amax * 1.02)
    assert not timed.stops_at_waypoints
    # It blends through the middle waypoint rather than stopping there.
    mid = np.argmin(np.linalg.norm(timed.positions - path[1], axis=1))
    assert np.linalg.norm(timed.velocities[mid]) > 0.2


def test_a_trajectory_over_the_torque_limit_is_timed_slower(cell):
    engine, _ = cell
    m = engine.ctx.model
    joints = [m.joint("ur5e_" + n).id for n in JOINTS]
    dyn = ArmDynamics(m, joints, payload=(m.body("ur5e_wrist_3_link").id, 4.5, [0, 0, 0.2]),
                      data=engine.ctx.data)
    assert list(dyn.limits) == [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
    path = [np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0]), np.array([0.0, -0.2, 0.0, -1.57, 0.0, 0.0])]
    fast = time_parameterize(path, VMAX, [40.0] * 6)
    timed = time_parameterize(path, VMAX, [40.0] * 6, torque=dyn.torque, torque_limit=dyn.limits)
    assert timed.accel_scale < 1.0
    assert timed.duration > fast.duration
    worst = max(np.max(np.abs(dyn.torque(q, v, a)) / dyn.limits)
                for q, v, a in zip(timed.positions, timed.velocities, timed.accelerations, strict=True))
    assert worst <= 0.95 + 1e-6


def test_a_timed_trajectory_is_followed_by_the_arm(cell):
    engine, checker = cell
    m, d = engine.ctx.model, engine.ctx.data
    arm = engine.ctx.blackboard.get("arm:ur5e")
    qadr = [m.jnt_qposadr[m.joint("ur5e_" + n).id] for n in JOINTS]
    start = d.qpos[qadr].copy()
    goal = start + np.array([1.0, 0.3, -0.4, 0.5, 0.8, -1.0])
    timed = time_parameterize([start, goal], VMAX, AMAX, checker=checker)
    arm.follow_trajectory(arm.joint_names, timed.times, timed.positions, timed.velocities,
                          timed.accelerations)
    worst = 0.0
    for k in range(int(timed.duration / m.opt.timestep)):
        engine.step()
        t = (k + 1) * m.opt.timestep
        ref = np.array([np.interp(t, timed.times, timed.positions[:, j]) for j in range(6)])
        worst = max(worst, float(np.max(np.abs(d.qpos[qadr] - ref))))
    for _ in range(100):
        engine.step()
    assert np.max(np.abs(d.qpos[qadr] - goal)) < math.radians(0.1)
    assert worst < math.radians(1.0)


def test_a_cartesian_path_is_a_straight_line_of_the_site(cell):
    _, checker = cell
    kin = checker.kin
    start = kin.ik([0.3, 0.3, 0.4], DOWN, np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]))
    path = cartesian_path(checker, kin, start, [0.3, 0.3, 0.2], DOWN, step=0.005)
    for q in path:
        p, _ = kin.fk(q)
        assert np.linalg.norm(p[:2] - [0.3, 0.3]) < 2e-4
    assert kin.fk(path[-1])[0][2] == pytest.approx(0.2, abs=1e-4)
    with pytest.raises(PlanningError, match="collision"):
        cartesian_path(checker, kin, kin.ik([0.45, 0.3, 0.3], DOWN, start), [0.45, -0.3, 0.3], DOWN)


def test_margins_name_a_wrist_singularity_and_a_joint_at_its_limit(cell):
    engine, checker = cell
    m = engine.ctx.model
    kin = checker.kin
    joints = [m.joint("ur5e_" + n).id for n in JOINTS]
    dyn = ArmDynamics(m, joints, data=engine.ctx.data)
    straight = pose_margins(kin, [0.0, -1.2, 1.4, -1.77, 0.0, 0.0], dyn)
    assert straight.wrist == pytest.approx(0.0, abs=1e-6)
    bent = pose_margins(kin, [0.0, -1.2, 1.4, -1.77, -1.57, 0.0], dyn)
    assert bent.wrist == pytest.approx(1.0, abs=1e-3)
    at_stop = pose_margins(kin, [0.0, -1.2, 3.1, -1.77, -1.57, 0.0], dyn)
    assert at_stop.limit_rad[2] == pytest.approx(3.1415 - 3.1, abs=1e-6)
    assert at_stop.limit_fraction[2] < 0.02
    assert bent.torque_ratio[1] > 0.05, "the shoulder carries the arm"
    assert list(joint_torque_limits(m, joints)) == [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]


def _gripper_world(with_box):
    extra = []
    if with_box:
        extra.append({"box": {"prefix": "part_", "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
                              "size": [0.02, 0.06, 0.03], "color": [0.8, 0.2, 0.2, 1]}, "name": "part"})
    return _world(extra, gravity=[0, 0, 0])


@pytest.mark.parametrize("with_box", [True, False])
def test_the_hold_is_the_pads_contacts_not_the_fingers_closing(with_box):
    engine = Engine(load_config_from_dict(_gripper_world(with_box)))
    engine.setup()
    engine.reset()
    try:
        m, d = engine.ctx.model, engine.ctx.data
        arm = engine.ctx.blackboard.get("arm:ur5e")
        kin_site = m.site("ur5e_pinch").id
        # Put the pinch where the part is and close.
        joints = [m.joint("ur5e_" + n).id for n in JOINTS]
        checker = CollisionChecker(m, d, joints, kin_site, m.body("ur5e_base").id)
        q = checker.kin.ik([0.0, 0.45, 0.5], DOWN, d.qpos[[m.jnt_qposadr[j] for j in joints]])
        qadr = [m.jnt_qposadr[j] for j in joints]
        d.qpos[qadr] = q
        arm.set_targets(arm.joint_names, list(q))
        mujoco.mj_forward(m, d)
        if with_box:
            part = m.body("part_box").id
            adr = m.jnt_qposadr[m.body_jntadr[part]]
            d.qpos[adr : adr + 3] = d.site_xpos[kin_site]
            d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        pads = [m.geom("ur5e_hande_left_pad").id, m.geom("ur5e_hande_right_pad").id]
        arm.set_gripper(0.025)  # Robotiq's convention: 0 open .. 0.025 m closed
        for _ in range(400):
            engine.step()
        if with_box:
            assert pads_touch(m, d, pads, part)
            monitor = HoldMonitor(m, kin_site, pads, part)
            monitor.grip(d)
            for _ in range(200):
                engine.step()
                monitor.update(d)
            out = monitor.release()
            assert out["steps_without_both_pads"] == 0 and out["max_drift_m"] < 1e-3
        else:
            # Closed on nothing: the fingers are at the end of their travel, and the pads touch no part.
            assert not pads_touch(m, d, pads, m.body("ur5e_base").id)
    finally:
        engine.shutdown()


def test_roqsim_arm_margins_reads_a_recording(tmp_path, capsys):
    import json

    import yaml

    from roqsim.runner import run
    from roqsim_manipulation.cli.margins import main

    world = _world()
    world["components"][0]["components"][0]["arm_controller"]["test_target"] = [0.5, -1.2, 1.4, -1.77,
                                                                                 -1.57, 0.0]
    (tmp_path / "w.yaml").write_text(yaml.safe_dump(world))
    out = tmp_path / "run.mcap"
    run(str(tmp_path / "w.yaml"), headless=True, pacing="asap", seconds=1.0, record=str(out),
        capture_fps=50, seed=1)
    code = main([str(out), "--arm", "ur5e", "--site", "ur5e_pinch", "--at", "0.9",
                 "--payload", "ur5e_hande_left_finger", "0.19", "0", "0", "0.05", "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    pose = report["poses"][0]
    assert pose["q"][0] == pytest.approx(0.5, abs=0.01)
    assert len(pose["limit_rad"]) == 6 and 0.0 < pose["wrist"] <= 1.0
    assert all(0.0 < r < 1.0 for r in report["peak_torque_ratio"][1:3])
    assert report["torque_limit"] == [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
    assert main([str(out), "--arm", "nope", "--site", "ur5e_pinch"]) == 2


def test_a_hold_counts_its_longest_stretch_without_both_pads(monkeypatch):
    import roqsim_manipulation.planning.hold as hold

    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><site name="s"/><body name="part"><freejoint/><geom size="0.01"/></body>'
        "</worldbody></mujoco>")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    touching = iter([True, False, False, True, False, False, False, True, False])
    monkeypatch.setattr(hold, "pads_touch", lambda *a: next(touching))
    monitor = hold.HoldMonitor(m, m.site("s").id, [0], m.body("part").id)
    monitor.grip(d)
    for _ in range(9):
        monitor.update(d)
    out = monitor.release()
    assert out["steps_without_both_pads"] == 6
    assert out["longest_without_both_pads"] == 3
