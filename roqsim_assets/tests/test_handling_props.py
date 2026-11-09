"""The props a hand works: a button pressed by contact, drawers that slide, a tote with handles."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _engine(components):
    engine = Engine(load_config_from_dict({"sim": {"pacing": "asap"}, "components": components}))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.fixture
def props():
    engine = _engine(
        [
            {"push_button": {"pose": {"position": {"x": 1.0, "y": 0.0, "z": 1.0}}}, "name": "start"},
            {"drawer_cabinet": {"pose": {"position": {"x": 0.0, "y": 2.0}}, "drawers": 2}, "name": "cab"},
            {"tote": {"pose": {"position": {"x": -1.0, "y": 0.0}}, "contents": True}, "name": "tote_1"},
            # a kinematic pusher, as a hand's press proxy is
            {"box": {"pose": {"position": {"x": 1.2, "y": 0.0, "z": 1.0}}, "size": [0.02, 0.02, 0.02], "motion": "driven"},
             "name": "finger"},
        ]
    )
    yield engine
    engine.shutdown()


def test_a_button_is_pressed_by_what_pushes_it_and_springs_back(props):
    m, d = props.ctx.model, props.ctx.data
    handle = props.ctx.blackboard.get("button:start")
    finger = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, props.ctx.entities.get("finger").body)])
    site = d.site("start/button").xpos.copy()
    for k in range(600):
        d.mocap_pos[finger] = site + [0.03 - min(k, 400) * 0.0001, 0.0, 0.0]
        props.step()
    assert handle.pressed and handle.presses == 1
    d.mocap_pos[finger] = site + [0.2, 0.0, 0.0]
    for _ in range(500):
        props.step()
    assert not handle.pressed and handle.travel < 0.002
    events = [e["event"] for e in props.ctx.events.all() if e.get("source") == "start"]
    assert events == ["pressed", "released"]


def test_a_button_ringing_between_its_release_and_its_threshold_is_pressed_once(props):
    """A snap action: in past the threshold, then ringing about it above the release point, is one
    press; back under the release point, it is released."""
    m, d = props.ctx.model, props.ctx.data
    handle = props.ctx.blackboard.get("button:start")
    q = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "start/stroke")])
    stroke = 0.012  # threshold 0.6, release 0.4 of it
    for travel in [0.75, 0.55, 0.7, 0.5, 0.65, 0.45, 0.62] + [0.3]:
        for _ in range(5):
            d.qpos[q] = travel * stroke
            d.qvel[:] = 0.0
            props.step()
        assert handle.pressed == (travel > 0.4), travel
    assert handle.presses == 1


def test_the_button_is_not_held_by_its_housing():
    engine = _engine([{"push_button": {"pose": {"position": {"x": 0.0, "y": 0.0, "z": 1.0}}}, "name": "b"}])
    try:
        m = engine.ctx.model
        h = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b/housing")
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b/button")
        pairs = {(int(m.exclude_signature[i] >> 16), int(m.exclude_signature[i] & 0xFFFF)) for i in range(m.nexclude)}
        assert (h, b) in pairs or (b, h) in pairs
    finally:
        engine.shutdown()


def test_a_drawer_slides_out_when_pulled_and_stops_at_its_travel(props):
    m, d = props.ctx.model, props.ctx.data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cab/drawer_0/slide")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cab/drawer_0")
    assert props.ctx.entities.get("cab/drawer_0") is not None
    for _ in range(1500):
        d.xfrc_applied[bid] = [60.0, 0, 0, 0, 0, 0]
        props.step()
    q = d.qpos[m.jnt_qposadr[jid]]
    assert 0.43 < q <= 0.46
    grip = d.site("cab/drawer_0/grip")
    # the grip frame: +x toward the handle (the cabinet's -x), +z down
    assert np.allclose(grip.xmat.reshape(3, 3)[:, 0], [-1, 0, 0], atol=1e-6)
    assert np.allclose(grip.xmat.reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-6)


def test_a_tote_stands_on_the_floor_with_two_handle_sites_facing_it(props):
    d = props.ctx.data
    for _ in range(300):
        props.step()
    body = d.body("tote_1")
    assert abs(body.xpos[2]) < 0.01
    left, right = d.site("tote_1/grip_l"), d.site("tote_1/grip_r")
    centre = body.xpos + [0, 0, 0.1]
    for site in (left, right):
        x = site.xmat.reshape(3, 3)[:, 0]
        assert np.dot(x, centre - site.xpos) > 0  # the palm faces the tote
        assert np.allclose(site.xmat.reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-6)


def test_a_release_point_at_or_above_the_threshold_is_refused():
    with pytest.raises(Exception, match=r"'release' \(0.7\).*below 'threshold' \(0.6\)"):
        _engine([{"push_button": {"pose": {"position": {"x": 1.0, "y": 0.0, "z": 1.0}}, "release": 0.7},
                  "name": "start"}])
