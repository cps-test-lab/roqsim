"""A walker's hand as it is drawn: the size of an adult hand for its body, fingers that fold into the
pose they are named for, and the fingertip the moves aim at where the drawn one is.

Everything here is measured on the rendered skin (``mjv_updateScene``'s skinned vertices), in the
right wrist's frame, on ``MaleVisitorWalk`` (the walker of the handling scenes, finger weights from
its source rig).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_motion import hands as H
from roqsim_walker.blueprint import resolve_walker

#: An adult hand at 1.75 m: wrist crease to middle fingertip, and across the palm at the knuckles
#: (m); scaled to a walker's height. A drawn hand is within :data:`TOLERANCE` of it, the spread of
#: adult hands about that mean and the reach of a low-polygon character's.
REF_HEIGHT, REF_LENGTH, REF_BREADTH = 1.75, 0.19, 0.085
TOLERANCE = 0.25


@pytest.fixture(scope="module")
def worker():
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"},
             "components": [{"walker": {"walker": "MaleVisitorWalk", "hands": True,
                                        "pose": {"position": {"x": 0.0, "y": 0.0}}}, "name": "worker"}]}
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


class Drawn:
    """The right hand's skinned vertices in the wrist's frame, by the bone each mostly follows."""

    def __init__(self, engine):
        m, d = engine.ctx.model, engine.ctx.data
        scn = mujoco.MjvScene(m, maxgeom=20000)
        mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera(),
                               mujoco.mjtCatBit.mjCAT_ALL, scn)
        names = ["wrist_r"] + H.bone_names("r")
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"worker/{n}") for n in names]
        wrist = d.body("worker/wrist_r")
        rot = wrist.xmat.reshape(3, 3)
        verts, owner = [], []
        for s in range(m.nskin):
            adr, n = m.skin_vertadr[s], m.skin_vertnum[s]
            v = scn.skinvert[3 * adr : 3 * (adr + n)].reshape(-1, 3)
            w = np.zeros((n, len(ids)))
            for b in range(m.skin_boneadr[s], m.skin_boneadr[s] + m.skin_bonenum[s]):
                body = int(m.skin_bonebodyid[b])
                if body in ids:
                    a, k = m.skin_bonevertadr[b], m.skin_bonevertnum[b]
                    np.add.at(w[:, ids.index(body)], m.skin_bonevertid[a : a + k], m.skin_bonevertweight[a : a + k])
            hand = w.sum(1) > 0.5
            verts.append((v[hand] - wrist.xpos) @ rot)
            owner.append(np.argmax(w[hand], axis=1))  # 0 the wrist, 1 + 3 * finger + bone
        self.origin = (np.array([d.xpos[i] for i in ids]) - wrist.xpos) @ rot  # each bone's joint
        self.v = np.vstack(verts)
        self.owner = np.concatenate(owner)
        self.palm = np.array([0.0, 1.0, 0.0])  # the right palm faces +y of its wrist

    def bone(self, finger: int, k: int) -> np.ndarray:
        return self.v[self.owner == 1 + 3 * finger + k]

    def tip(self, finger: int) -> np.ndarray:
        """The vertex of a finger's last drawn bone farthest from its joint (a low-polygon little
        finger may have two bones)."""
        k = next(k for k in (2, 1, 0) if len(self.bone(finger, k)))
        last = self.bone(finger, k)
        return last[np.argmax(np.linalg.norm(last - self.origin[1 + 3 * finger + k], axis=1))]


def _posed(engine, pose: str) -> Drawn:
    body = engine.ctx.blackboard.get("humanoid_body:worker")
    body.set_pose("r", pose)
    for _ in range(400):
        engine.step()
    return Drawn(engine)


def _height() -> float:
    return float(resolve_walker("MaleVisitorWalk")["skeleton"]["height"])


def test_the_drawn_hand_has_an_adult_hands_size_for_its_body_in_every_pose(worker):
    scale = _height() / REF_HEIGHT
    length, breadth = REF_LENGTH * scale, REF_BREADTH * scale
    flat = _posed(worker, "flat")
    fingers = flat.v[(flat.owner >= 4)]  # the four fingers' bones
    got_length = -float(flat.v[:, 2].min())  # from the wrist's crease (its joint) down the fingers
    got_breadth = float(np.ptp(fingers[:, 0]))
    assert abs(got_length / length - 1.0) < TOLERANCE, f"length {got_length:.3f} m of {length:.3f}"
    assert abs(got_breadth / breadth - 1.0) < TOLERANCE, f"breadth {got_breadth:.3f} m of {breadth:.3f}"
    for pose in H.POSES:
        drawn = _posed(worker, pose)
        thick = float(np.ptp(drawn.v @ drawn.palm))
        reach = float(np.linalg.norm(drawn.v, axis=1).max())
        # a closed hand is no deeper than a palm is broad; no pose makes the hand longer
        assert thick < breadth, f"{pose}: {thick:.3f} m deep, a palm is {breadth:.3f} m across"
        assert reach < 1.1 * length, f"{pose}: reaches {reach:.3f} m from the wrist"


@pytest.mark.parametrize("pose", ["open", "point", "power"])
def test_the_index_fingertip_the_moves_aim_is_the_drawn_one(worker, pose):
    drawn = _posed(worker, pose)
    body = worker.ctx.blackboard.get("humanoid_body:worker")
    d = worker.ctx.data
    wrist = d.body("worker/wrist_r")
    tip = (body.tip(d, "r") - wrist.xpos) @ wrist.xmat.reshape(3, 3)
    last = drawn.bone(1, 2)
    origin = (d.body("worker/f_r_index_3").xpos - wrist.xpos) @ wrist.xmat.reshape(3, 3)
    along = tip - origin
    along /= np.linalg.norm(along)
    # the drawn tip: the end of the cap of the finger's last bone, on its line
    t = (last - origin) @ along
    cap = last[t > t.max() - 0.003]
    drawn_tip = cap.mean(axis=0) + along * (t.max() - float(((cap - origin) @ along).mean()))
    assert np.linalg.norm(drawn_tip - tip) < 0.005


def _angle(a, b) -> float:
    return math.degrees(math.acos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def _reach(drawn: Drawn, finger: int) -> float:
    """How far a finger's drawn tip is from its knuckle (m): its length straight, less folded."""
    return float(np.linalg.norm(drawn.tip(finger) - drawn.origin[1 + 3 * finger]))


def test_each_pose_reads_as_its_name(worker):
    """``flat`` straight; ``open`` relaxed, the thumb alongside; ``point`` the index straight and
    the others folded, the thumb over them; ``power`` every finger folded, the thumb over them;
    ``pinch`` the thumb's tip on the index's."""
    flat = _posed(worker, "flat")
    lay = worker.ctx.blackboard.get("humanoid_body:worker").layouts["r"]
    full = [_reach(flat, f) for f in range(5)]
    # straight: each finger reaches its whole length
    assert all(abs(full[f] / float(lay.lengths[f].sum()) - 1.0) < 0.1 for f in range(1, 5))

    def folded(drawn, f):
        return _reach(drawn, f) / full[f]

    open_ = _posed(worker, "open")
    index_dir = open_.tip(1) - open_.origin[4]
    thumb_dir = open_.tip(0) - open_.origin[1]
    assert all(folded(open_, f) > 0.9 for f in range(1, 5))
    assert _angle(thumb_dir, index_dir) < 30, "open: the thumb is spread off the fingers"
    assert np.dot(open_.tip(0) - open_.origin[4], open_.palm) > -0.005, "open: the thumb is raised"

    point = _posed(worker, "point")
    assert folded(point, 1) > 0.97
    assert all(folded(point, f) < 0.75 for f in (2, 3, 4))
    assert np.linalg.norm(point.tip(0) - point.bone(2, 1).mean(axis=0)) < 0.035, "point: the thumb is not over the fingers"

    power = _posed(worker, "power")
    assert all(folded(power, f) < 0.75 for f in range(1, 5))
    assert np.dot(power.tip(0) - power.origin[4], power.palm) > 0.0, "power: the thumb is not opposed"
    assert np.linalg.norm(power.tip(0) - power.bone(1, 1).mean(axis=0)) < 0.035, "power: the thumb is not over the fingers"

    pinch = _posed(worker, "pinch")
    assert np.linalg.norm(pinch.tip(0) - pinch.tip(1)) < 0.02, "pinch: thumb and index do not meet"
