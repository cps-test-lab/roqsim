"""The motion core on its own: forward kinematics, the solver, hand poses, targets."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim_motion import hands as H
from roqsim_motion import quat as Q
from roqsim_motion.description import JOINTS, Description
from roqsim_motion.ik import HandGoal, Posture, solve
from roqsim_motion.targets import TargetError, make_target

DESC = Description()
REST = {j: Q.IDENTITY.copy() for j in JOINTS}
LAYOUTS = {s: H.layout(s) for s in "lr"}
ROOT = np.array([0.0, 0.0, DESC.root_height])


def _random_reachable(rng, side):
    """A grip pose the arm can take: a random arm pose's grip frame, the trunk upright."""
    sh = Q.from_euler("yxz", [rng.uniform(-1.8, 0.3), rng.uniform(-0.3, 0.6) * (1 if side == "l" else -1),
                              rng.uniform(-0.5, 0.5)])
    el = Q.axis_angle("y", -rng.uniform(0.3, 1.8))
    wr = Q.from_euler("zyx", [rng.uniform(-1.0, 1.0), rng.uniform(-0.8, 0.8), rng.uniform(-0.4, 0.4)])
    poses = DESC.fk(ROOT, Q.IDENTITY, {**REST, f"shoulder_{side}": sh, f"elbow_{side}": el, f"wrist_{side}": wr})
    wp, wq = poses[f"wrist_{side}"]
    lay = LAYOUTS[side]
    return wp + Q.rotate(wq, lay.grip_pos), Q.mul(wq, lay.grip_quat)


@pytest.mark.parametrize("side", ["l", "r"])
def test_a_reachable_grip_pose_is_reached_within_5_mm_and_5_degrees(side):
    rng = np.random.default_rng(7)
    misses = 0
    for _ in range(200):
        pos, rot = _random_reachable(rng, side)
        sol = solve(DESC, ROOT, Q.IDENTITY, REST, {side: HandGoal(pos, rot)}, LAYOUTS,
                    Posture(turn=0.0, lean=0.0, crouch=0.0))
        merged = {**REST, **sol.rotations}
        poses = DESC.fk(ROOT - [0, 0, sol.root_drop], Q.IDENTITY, merged)
        wp, wq = poses[f"wrist_{side}"]
        grip = wp + Q.rotate(wq, LAYOUTS[side].grip_pos)
        err_p = np.linalg.norm(grip - pos)
        err_r = Q.angle_between(Q.mul(wq, LAYOUTS[side].grip_quat), rot)
        assert DESC.within(f"wrist_{side}", sol.rotations[f"wrist_{side}"])
        if err_p > 0.005 or err_r > math.radians(5):
            misses += 1
    # Two-bone arms reach any point within 97 % of the arm; a pose generated at full stretch
    # is the only one left short, and those are few.
    assert misses <= 10


def test_the_solver_is_deterministic():
    rng = np.random.default_rng(3)
    pos, rot = _random_reachable(rng, "r")
    goal = {"r": HandGoal(pos + [0.3, 0.0, -0.4], rot)}
    a = solve(DESC, ROOT, Q.IDENTITY, REST, goal, LAYOUTS)
    b = solve(DESC, ROOT, Q.IDENTITY, REST, goal, LAYOUTS)
    for k in a.rotations:
        assert np.array_equal(a.rotations[k], b.rotations[k])


def test_an_unreachable_target_is_reported_by_its_residual_not_stretched_to():
    sol = solve(DESC, ROOT, Q.IDENTITY, REST, {"r": HandGoal(np.array([2.0, 0.0, 1.0]))}, LAYOUTS)
    assert sol.residual["r"] > 0.5
    lengths = [np.linalg.norm(DESC.offset(j)) for j in ("elbow_r", "wrist_r")]
    poses = DESC.fk(ROOT, Q.IDENTITY, {**REST, **sol.rotations})
    assert np.linalg.norm(poses["wrist_r"][0] - poses["shoulder_r"][0]) <= sum(lengths) + 1e-9


def test_a_low_target_crouches_with_the_feet_where_they_stood():
    goal = {"r": HandGoal(np.array([0.45, -0.15, 0.25]))}
    sol = solve(DESC, ROOT, Q.IDENTITY, REST, goal, LAYOUTS)
    assert sol.root_drop > 0.1
    stand = DESC.fk(ROOT, Q.IDENTITY, REST)
    crouch = DESC.fk(ROOT - [0, 0, sol.root_drop], Q.IDENTITY, {**REST, **sol.rotations})
    for s in "lr":
        assert np.linalg.norm(crouch[f"ankle_{s}"][0] - stand[f"ankle_{s}"][0]) < 1e-6
        assert Q.angle_between(crouch[f"ankle_{s}"][1], stand[f"ankle_{s}"][1]) < 1e-6
    assert sol.residual["r"] < 0.005


def test_a_target_beside_the_body_turns_it_and_one_far_to_the_side_leans_it():
    side = solve(DESC, ROOT, Q.IDENTITY, REST, {"l": HandGoal(np.array([0.3, 0.5, 1.0]))}, LAYOUTS)
    assert side.posture.turn > 0.3
    far = solve(DESC, ROOT, Q.IDENTITY, REST, {"l": HandGoal(np.array([0.0, 0.72, 1.15]))}, LAYOUTS,
                Posture(turn=0.0))
    assert far.posture.lean > 0.0
    assert far.residual["l"] < 0.01


def test_gaze_turns_the_head_within_its_limits():
    sol = solve(DESC, ROOT, Q.IDENTITY, REST, {}, LAYOUTS, gaze=np.array([0.0, -5.0, 1.6]))
    yaw = Q.euler(sol.rotations["head"], "zyx")[0]
    assert math.isclose(yaw, math.radians(-70), abs_tol=1e-6)


def test_hand_poses_blend_and_a_grip_closes_finger_by_finger():
    st = H.HandState()
    st.set_pose("power")
    st.close_on(lambda i: i == 2)  # the middle finger meets the handle at once
    for _ in range(100):
        done = st.advance(0.01)
    assert done
    assert np.allclose(st.curls[1], H.POSES["power"][1])
    assert np.allclose(st.curls[2], H.POSES["open"][2])
    with pytest.raises(ValueError, match="no hand pose 'fist'"):
        st.set_pose("fist")


def test_a_pose_change_eases_each_finger_whole_and_a_grip_closes_finger_by_finger():
    st = H.HandState()
    dt = 0.002
    start = st.curls.copy()
    st.set_pose("power")
    goal = np.array(H.POSES["power"])
    arrived = np.full((5, 3), np.nan)
    moved = np.full(5, np.nan)
    last = st.curls.copy()
    t = 0.0
    while t < 1.0:
        st.advance(dt)
        t += dt
        step = np.abs(st.curls - last)
        assert step.max() <= H.CURL_RATE * dt * 1.001  # never faster than the rate: no jump
        last = st.curls.copy()
        for f in range(5):
            if np.isnan(moved[f]) and np.any(np.abs(st.curls[f] - start[f]) > 1e-9):
                moved[f] = t
            for k in range(3):
                if np.isnan(arrived[f, k]) and abs(st.curls[f, k] - goal[f, k]) < 1e-9:
                    arrived[f, k] = t
    assert not np.isnan(arrived).any()
    # each finger's bones arrive together: a finger folds as one, not joint by joint
    assert np.all(np.ptp(arrived, axis=1) < 1e-9)
    # index first, little last, the thumb over them after
    assert moved[1] < moved[2] < moved[3] < moved[4] < moved[0]
    # changed part-way, a pose starts from where the fingers are
    st.set_pose("open")
    st.advance(dt)
    mid = st.curls.copy()
    st.set_pose("point")
    st.advance(dt)
    assert np.abs(st.curls - mid).max() <= H.CURL_RATE * dt * 1.001


def test_fingers_curl_toward_the_palm_of_each_hand():
    for side, palm in (("l", -1.0), ("r", 1.0)):
        lay = H.layout(side)
        flat = H.fingertips(lay, [0, 0, 0], Q.IDENTITY, np.array(H.POSES["flat"]))
        fist = H.fingertips(lay, [0, 0, 0], Q.IDENTITY, np.array(H.POSES["power"]))
        assert np.all(palm * (fist[1:, 1] - flat[1:, 1]) > 0.02)
        assert np.all(fist[1:, 2] - flat[1:, 2] > 0.04)  # folded back toward the wrist
        pos, rot = H.finger_fk(lay, [0, 0, 0], Q.IDENTITY, np.array(H.POSES["open"]))
        assert pos.shape == (15, 3) and rot.shape == (15, 4)
        assert len(H.bone_names(side)) == 15


MJCF = """<mujoco><worldbody>
  <body name="box" pos="1 0 0.5"><freejoint/><geom type="box" size=".1 .1 .1"/>
    <site name="box/grip" pos="0 0.1 0.1" euler="0 0 90"/></body>
</worldbody></mujoco>"""


class _Entities:
    def get(self, name):
        from roqsim.context import Entity

        return Entity(name="tote", kind="prop", body="box") if name == "tote" else None


def test_targets_resolve_sites_bodies_entities_and_world_poses():
    m = mujoco.MjModel.from_xml_string(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p, r = make_target(m, _Entities(), "box/grip").resolve(m, d)
    assert np.allclose(p, [1, 0.1, 0.6])
    p, _ = make_target(m, _Entities(), "tote", offset=(0, 0, 0.2)).resolve(m, d)
    assert np.allclose(p, [1, 0, 0.7])
    p, r = make_target(m, None, {"position": {"x": 1, "y": 2, "z": 3}, "orientation": {"yaw": 1.0}}).resolve(m, d)
    assert np.allclose(p, [1, 2, 3]) and math.isclose(Q.yaw_of(r), 1.0)
    with pytest.raises(TargetError, match="unknown target 'nowhere'"):
        make_target(m, _Entities(), "nowhere")


def _twist(rot, axis) -> float:
    """``rot``'s twist about ``axis`` (deg)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return math.degrees(2.0 * math.atan2(float(np.asarray(rot[1:]) @ a), float(rot[0])))


@pytest.mark.parametrize("side", ["l", "r"])
@pytest.mark.parametrize("turn", [-60.0, 60.0])
def test_a_hand_turned_about_the_forearm_turns_the_forearm_too(side, turn, monkeypatch):
    """Pronation is the forearm's: a hand turned about the forearm's line turns the forearm too, and
    the wrist keeps about half the twist it would keep alone -- and the hand still arrives."""
    from roqsim_motion import ik

    arm = {**REST, f"elbow_{side}": Q.axis_angle("y", -1.2)}  # the forearm forward
    poses = DESC.fk(ROOT, Q.IDENTITY, arm)
    wp, wq = poses[f"wrist_{side}"]
    forearm = Q.rotate(wq, DESC.offset(f"wrist_{side}"))
    goal_q = Q.mul(Q.axis_angle(forearm, math.radians(turn)), Q.mul(wq, LAYOUTS[side].grip_quat))
    goal_p = wp + Q.rotate(goal_q, Q.rotate(Q.inv(LAYOUTS[side].grip_quat), LAYOUTS[side].grip_pos))
    goals = {side: HandGoal(goal_p, goal_q)}
    sol = solve(DESC, ROOT, Q.IDENTITY, arm, goals, LAYOUTS)
    assert sol.residual[side] < 0.005
    shared = _twist(sol.rotations[f"wrist_{side}"], DESC.offset(f"wrist_{side}"))
    monkeypatch.setattr(ik, "FOREARM_TWIST", 0.0, raising=False)
    alone = _twist(solve(DESC, ROOT, Q.IDENTITY, arm, goals, LAYOUTS).rotations[f"wrist_{side}"],
                   DESC.offset(f"wrist_{side}"))
    assert abs(shared) < 0.6 * abs(alone), f"the wrist keeps {shared:.0f} deg; alone it would {alone:.0f}"
