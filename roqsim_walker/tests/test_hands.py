"""A walker's hands: off adds nothing to the walker; on adds wrists, fingers and the motion core."""

from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_motion import hands as H
from roqsim_motion import quat as Q
from roqsim_walker.skin import hand_weights

#: The two bundled walkers with hands off: the counts and a digest over their bodies, geoms and
#: skins, as the walker builds them without the hands code. Hands off keeps them, byte for byte.
HANDS_OFF = {
    "nbody": 35,
    "ngeom": 47,
    "nmocap": 34,
    "nsite": 0,
    "nskin": 2,
    "nskinbone": 34,
    "sha256": "3e048fda946c33276898288ea9e6021ca4a3af3b40428f28e4524f266fec5337",
}
FIELDS = (
    "body_pos",
    "body_quat",
    "body_mocapid",
    "geom_type",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_contype",
    "geom_conaffinity",
    "geom_group",
    "skin_vertnum",
    "skin_bonenum",
    "skin_bonebindpos",
    "skin_bonebindquat",
    "skin_bonebodyid",
    "skin_bonevertid",
    "skin_bonevertweight",
)


def _engine(components, preview=False):
    engine = Engine(
        load_config_from_dict({"sim": {"pacing": "asap"}, "components": components}),
        preview=preview,
    )
    if not preview:
        engine.ctx.seed = 1
    engine.setup()
    return engine


def _walkers(hands=None):
    extra = {} if hands is None else {"hands": hands}
    return [
        {
            "walker": {"walker": w, **extra, "pose": {"position": {"x": 0.0, "y": float(i)}}},
            "name": f"w{i}",
        }
        for i, w in enumerate(["MaleVisitorWalk", "FemaleVisitorWalk"])
    ]


@pytest.mark.parametrize("hands", [None, False])
def test_hands_off_adds_nothing_to_the_walkers_model(hands):
    engine = _engine(_walkers(hands), preview=True)
    try:
        m = engine.ctx.model
        digest = hashlib.sha256()
        for field in FIELDS:
            digest.update(np.ascontiguousarray(np.asarray(getattr(m, field))).tobytes())
        got = {
            k: int(getattr(m, k))
            for k in ("nbody", "ngeom", "nmocap", "nsite", "nskin", "nskinbone")
        }
        assert {**got, "sha256": digest.hexdigest()} == HANDS_OFF
        assert engine.ctx.blackboard.get("humanoid_body:w0") is None
    finally:
        engine.shutdown()


@pytest.fixture(scope="module")
def worker():
    engine = _engine(
        [
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "hands": True,
                    "pose": {"position": {"x": 0.0, "y": 0.0}},
                },
                "name": "worker",
            }
        ]
    )
    yield engine
    engine.shutdown()


@pytest.fixture
def sim(worker):
    worker.reset()
    for _ in range(50):
        worker.step()
    return worker


def test_hands_on_adds_thirty_geomless_finger_bones_two_grip_sites_and_two_proxies(sim):
    m = sim.ctx.model
    assert m.nmocap == 17 + 30 + 2
    for side in "lr":
        for bone in H.bone_names(side):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"worker/{bone}")
            assert bid >= 0 and m.body_geomnum[bid] == 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"worker/grip_{side}") >= 0
    bound = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in m.skin_bonebodyid}
    # the source rig weights the thumbs and the finger chain: those bones carry vertices
    assert any("thumb" in b for b in bound) and any("index" in b for b in bound)


def test_the_fingers_sit_on_the_wrist_and_follow_its_pose(sim):
    d = sim.ctx.data
    body = sim.ctx.blackboard.get("humanoid_body:worker")
    for side in "lr":
        wrist = d.mocap_pos[body._wrist[side]]
        tips = d.mocap_pos[body._fingers[side]]
        assert np.all(np.linalg.norm(tips - wrist, axis=1) < 0.2)


@pytest.mark.parametrize("pose", sorted(H.POSES))
def test_each_named_hand_pose_is_reached(sim, pose):
    body = sim.ctx.blackboard.get("humanoid_body:worker")
    body.set_pose("r", pose)
    for _ in range(400):
        sim.step()
    assert np.allclose(body.hand["r"].curls, H.POSES[pose])
    assert body.hand_done("r")


def test_reach_with_hands_on_puts_the_wrist_on_its_point(sim):
    handle = sim.ctx.blackboard.get("walker:worker")
    left, right = handle.hands()
    point = right + np.array([0.35, 0.1, 0.15])
    handle.reach(right=point)
    for _ in range(1000):
        sim.step()
    _, right = handle.hands()
    assert np.linalg.norm(right - point) < 0.005
    handle.reach()
    for _ in range(1000):
        sim.step()
    body = sim.ctx.blackboard.get("humanoid_body:worker")
    assert body.weight["r"] == 0.0


def test_a_grip_goal_with_a_rotation_is_reached(sim):
    d = sim.ctx.data
    body = sim.ctx.blackboard.get("humanoid_body:worker")
    pos, rot = body.grip(d, "r")
    goal_pos = pos + np.array([0.4, 0.05, 0.2])
    # palm toward the body's left, fingers forward-down: a hand closing on a handle in front
    goal_rot = Q.mul(rot, Q.axis_angle("y", -0.6))
    body.set_hand("r", goal_pos, goal_rot)
    for _ in range(1200):
        sim.step()
    pos, rot = body.grip(d, "r")
    assert np.linalg.norm(pos - goal_pos) < 0.005
    assert Q.angle_between(rot, goal_rot) < math.radians(5)


def test_a_low_goal_crouches_and_the_feet_stay_on_the_floor(sim):
    d = sim.ctx.data
    body = sim.ctx.blackboard.get("humanoid_body:worker")
    x, y, _ = body.root()
    pelvis0 = d.mocap_pos[body.st.mocap["pelvis"]][2]
    body.set_hand("r", [x + 0.45, y - 0.15, 0.3])
    for _ in range(1500):
        sim.step()
    assert d.mocap_pos[body.st.mocap["pelvis"]][2] < pelvis0 - 0.1
    assert body.residual["r"] < 0.01
    for side in "lr":
        assert d.mocap_pos[body.st.mocap[f"ankle_{side}"]][2] < 0.15


def test_a_hand_without_finger_weights_stays_rigid_on_its_wrist():
    w17 = np.zeros((4, 17))
    w17[:, 13] = 1.0  # wrist_l
    w = hand_weights(w17, np.zeros((4, 3)), None, {})
    assert w.shape == (4, 47)
    assert np.allclose(w[:, 13], 1.0) and np.allclose(w[:, 17:], 0.0)


def test_finger_shares_move_from_the_wrist_to_their_bones():
    """A vertex's share on a thumb bone goes to that thumb bone, and one on the four-finger chain to
    that bone of the finger the vertex is nearest; what stays on the wrist is the rest."""
    from roqsim_walker.humanoid import JOINT_NAMES
    from roqsim_walker.skin import _bind_bones, _bind_fingers, _finger_bones

    poses = _bind_bones()
    poses = {**poses, **_bind_fingers(poses, {s: H.layout(s) for s in "lr"})}
    bones = _finger_bones("r")
    index_mid, little_mid = poses[bones[3 * 1 + 1]][0], poses[bones[3 * 4 + 1]][0]
    verts = np.array([index_mid, little_mid, poses[bones[0]][0]])
    wrist = JOINT_NAMES.index("wrist_r")
    w17 = np.zeros((3, 17))
    w17[:, wrist] = 1.0
    shares = np.zeros((3, 12))
    shares[0, 6 + 3 + 1] = 0.6  # right, four-finger chain, bone 2: at the index
    shares[1, 6 + 3 + 1] = 0.6  # the same bone, at the little finger
    shares[2, 6 + 0] = 0.4  # right thumb, bone 1
    w = hand_weights(w17, verts, shares, poses)
    col = {b: 17 + 15 + k for k, b in enumerate(bones)}  # the right hand's bones follow the left's
    assert np.isclose(w[0, col[bones[3 * 1 + 1]]], 0.6) and np.isclose(w[0, wrist], 0.4)
    assert np.isclose(w[1, col[bones[3 * 4 + 1]]], 0.6) and np.isclose(
        w[1, col[bones[3 * 1 + 1]]], 0.0
    )
    assert np.isclose(w[2, col[bones[0]]], 0.4) and np.isclose(w[2, wrist], 0.6)
    assert np.allclose(w[:, 17:32], 0.0)  # nothing on the left hand
    assert np.allclose(w.sum(axis=1), 1.0)


def test_finger_weights_of_another_mesh_are_refused(tmp_path):
    from roqsim_walker.skin import _load_fingers

    path = tmp_path / "other.fingers.npz"
    np.savez(path, positions=np.zeros((2, 3)), weights=np.zeros((2, 12)))
    lookup = _load_fingers(str(path))
    assert lookup(np.zeros((1, 3))).shape == (1, 12)
    with pytest.raises(ValueError, match="not of this mesh"):
        lookup(np.array([[0.0, 0.0, 0.001]]))


def test_the_blueprints_hand_weights_decide_the_fingers_skin(monkeypatch):
    """With the blueprint's ``hand_weights`` unset, the hand is bound rigidly though the file is
    there: the key is what the skin reads."""
    import roqsim_walker.plugins.walker as walker_plugin

    def counts(engine):
        m = engine.ctx.model
        return int(m.nskinbone)

    on = _engine(
        [{"walker": {"walker": "MaleVisitorWalk", "hands": True}, "name": "w"}], preview=True
    )
    try:
        with_weights = counts(on)
    finally:
        on.shutdown()
    real = walker_plugin.resolve_walker
    monkeypatch.setattr(
        walker_plugin, "resolve_walker", lambda *a, **k: {**real(*a, **k), "hand_weights": None}
    )
    rigid = _engine(
        [{"walker": {"walker": "MaleVisitorWalk", "hands": True}, "name": "w"}], preview=True
    )
    try:
        assert counts(rigid) < with_weights
    finally:
        rigid.shutdown()


def test_hands_is_true_or_false():
    with pytest.raises(Exception, match="'hands' is true or false"):
        _engine(
            [{"walker": {"walker": "MaleVisitorWalk", "hands": "yes"}, "name": "w"}], preview=True
        )


def test_bench_hands_runs_and_hands_off_costs_nothing_new():
    from roqsim_walker.cli.bench_hands import measure

    off = measure(1, "off", robot="none", steps=60, render=False)
    on = measure(1, "on", robot="none", steps=60, render=False)
    assert off["nmocap"] == 17 and on["nmocap"] == 49
    # the fingers carry no geom: the only new ones are the two parked press proxies
    assert on["ngeom"] == off["ngeom"] + 2
    for row in (off, on):
        assert row["mj_step_us"] > 0 and row["posing_us_per_step"] > 0 and row["lidar_us"] > 0


def test_bench_hands_refuses_a_robot_with_no_lidar_rather_than_scan_from_a_fixed_point():
    from roqsim_walker.cli.bench_hands import measure

    with pytest.raises(SystemExit, match="robot 'piracer' has no lidar site"):
        measure(1, "off", robot="piracer", steps=10, render=False)


def test_a_press_proxy_is_parked_until_a_move_presses_then_rides_the_index_fingertip(sim):
    d = sim.ctx.data
    body = sim.ctx.blackboard.get("humanoid_body:worker")
    pid = body._proxy["r"]
    assert d.mocap_pos[pid][2] < -40
    body.press_proxy(d, "r", True)
    for _ in range(30):
        sim.step()
    wrist = d.mocap_pos[body._wrist["r"]]
    assert 0.08 < np.linalg.norm(d.mocap_pos[pid] - wrist) < 0.25
    body.press_proxy(d, "r", False)
    assert d.mocap_pos[pid][2] < -40
