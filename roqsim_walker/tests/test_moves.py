"""A humanoid's moves on a walker with hands: each does what it says, and each failure is a reason."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.machine import MachineError
from roqsim.plugin import PluginError

NAV = {
    "output": "walker",
    "speed": 1.0,
    "arrival_radius": 0.05,
    "avoidance": {"steer": "none", "stop": False},
}


def _world(humanoid=None, hands=True, navigator=NAV):
    worker = {
        "walker": {
            "walker": "MaleVisitorWalk",
            "hands": hands,
            "skin": False,
            "pose": {"position": {"x": 0.0, "y": 0.0}},
        },
        "name": "worker",
        "components": [
            {"navigator": navigator},
            {"humanoid": humanoid or {"graspable": ["cab", "tote_1"]}},
        ],
    }
    return {
        "sim": {"pacing": "asap"},
        "components": [
            worker,
            {
                "push_button": {
                    "pose": {
                        "position": {"x": 2.0, "y": 1.0, "z": 1.1},
                        "orientation": {"yaw": math.pi},
                    }
                },
                "name": "start",
            },
            {
                "drawer_cabinet": {
                    "pose": {"position": {"x": 2.0, "y": -1.0}, "orientation": {"yaw": math.pi}}
                },
                "name": "cab",
            },
            {"tote": {"pose": {"position": {"x": -1.5, "y": 0.0}}}, "name": "tote_1"},
            {
                "box": {
                    "pose": {"position": {"x": 0.0, "y": 2.5}},
                    "size": [0.8, 0.6, 0.75],
                    "motion": "static",
                },
                "name": "bench",
            },
        ],
    }


def _engine(world):
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    return engine


@pytest.fixture(scope="module")
def _shared():
    engine = _engine(_world())
    yield engine
    engine.shutdown()


class Sim:
    def __init__(self, engine):
        self.e = engine
        self.h = engine.ctx.blackboard.get("humanoid:worker")
        self.body = engine.ctx.blackboard.get("humanoid_body:worker")
        self.m, self.d = engine.ctx.model, engine.ctx.data

    def run(self, move, limit=60.0, **args):
        seq = self.h.start(move, **args)
        t0 = self.e.ctx.sim_time
        while True:
            self.e.step()
            st = self.h.status()
            if st.seq == seq and st.finished:
                return st
            assert self.e.ctx.sim_time - t0 < limit, f"{move} did not end in {limit} s"

    def ok(self, move, **args):
        st = self.run(move, **args)
        assert st.error == "", f"{move}: {st.error}"
        return st

    def events(self, since=0.0):
        return [
            ev["event"]
            for ev in self.e.ctx.events.all()
            if ev["source"].startswith("worker") and ev["t"] >= since
        ]

    def qpos(self, joint):
        return float(
            self.d.qpos[
                self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)]
            ]
        )


@pytest.fixture
def sim(_shared):
    _shared.reset()
    return Sim(_shared)


def test_approach_walks_to_a_standoff_and_faces_the_target(sim):
    sim.ok("approach", target="start/button", standoff=0.5)
    x, y, yaw = sim.body.root()
    assert math.hypot(x - 1.5, y - 1.0) < 0.3
    assert abs(math.atan2(math.sin(yaw), math.cos(yaw))) < math.radians(4)
    assert {"walking", "arrived"} <= set(sim.events())


def test_approach_to_nothing_fails_naming_it(sim):
    st = sim.run("approach", target="nowhere")
    assert "unknown target 'nowhere'" in st.error


def test_reach_puts_the_hand_on_a_target_and_far_is_out_of_reach(sim):
    sim.ok("approach", target="start/button", standoff=0.5)
    sim.ok("reach", hand="right", target="start/button", offset=[0.1, 0.0, 0.0], orient=False)
    p, _ = sim.body.grip(sim.d, "r")
    assert np.linalg.norm(p - (sim.d.site("start/button").xpos + [-0.1, 0, 0])) < 0.012
    st = sim.run("reach", hand="left", target="tote_1")
    assert st.error.startswith("out of reach")


def test_press_pushes_a_button_by_contact_and_lets_go(sim):
    button = sim.e.ctx.blackboard.get("button:start")
    sim.ok("approach", target="start/button", standoff=0.5)
    t = sim.e.ctx.sim_time
    sim.ok("press", hand="right", target="start/button")
    assert button.presses == 1 and not button.pressed
    steps = [ev for ev in sim.events(t) if ev in ("contact", "pressed", "released")]
    assert steps == ["contact", "pressed", "released"]
    assert sim.d.mocap_pos[sim.body._proxy["r"]][2] < -40  # parked again


def test_press_on_something_that_cannot_move_fails(sim):
    sim.ok("approach", target="cab/drawer_1/grip", standoff=0.75)
    st = sim.run("press", hand="right", target="cab/drawer_1/grip")  # a closed drawer, pushed in
    assert st.error.startswith("button did not move")


def test_grasp_pull_push_and_release_a_drawer(sim):
    sim.ok("approach", target="cab/drawer_1/grip", standoff=0.75)
    sim.ok("grasp", hand="right", target="cab/drawer_1/grip")
    sim.ok("pull", hand="right", distance=0.3)
    assert abs(sim.qpos("cab/drawer_1/slide") - 0.3) < 0.02  # pulled by the hold, not set
    sim.ok("push", hand="right", distance=0.25)
    assert sim.qpos("cab/drawer_1/slide") < 0.1
    sim.ok("release")
    assert {"closing", "held", "moving", "pulled", "pushed", "opened"} <= set(sim.events())


def test_a_drawer_pulled_to_its_stop_is_out_and_one_held_shut_is_stuck(sim):
    sim.ok("approach", target="cab/drawer_1/grip", standoff=0.75)
    sim.ok("grasp", hand="right", target="cab/drawer_1/grip")
    sim.ok("pull", hand="right", distance=0.6)  # more than its travel: it ends at the stop
    assert abs(sim.qpos("cab/drawer_1/slide") - 0.45) < 0.01
    pulled = [e for e in sim.e.ctx.events.all() if e["event"] == "pulled"][-1]
    assert pulled["distance"] < 0.5


def test_pulling_what_does_not_move_is_stuck():
    world = _world({"graspable": ["cab", "tote_1"]})
    world["components"][2]["drawer_cabinet"]["friction"] = 2000.0  # a drawer that will not budge
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="cab/drawer_1/grip", standoff=0.75)
        sim.ok("grasp", hand="right", target="cab/drawer_1/grip")
        st = sim.run("pull", hand="right", distance=0.3)
        assert st.error.startswith("stuck") or st.error.startswith("lost hold")
    finally:
        engine.shutdown()


def test_grasp_what_is_not_graspable_or_has_no_grip_fails(sim):
    sim.ok("approach", target="start/button", standoff=0.5)
    assert sim.run("grasp", hand="right", target="start/button").error.startswith(
        "nothing to grasp"
    )
    assert "has no grip site" in sim.run("grasp", hand="right", target="bench").error


def test_release_pull_carry_and_place_without_a_hold_fail(sim):
    assert sim.run("release").error == "not holding anything"
    assert sim.run("pull", hand="right", distance=0.1).error == "not holding with the right hand"
    assert sim.run("carry", to="bench").error == "not holding anything"
    assert sim.run("place", at="bench").error == "not holding anything"


def test_both_hands_carry_a_tote_and_place_it_on_a_bench(sim):
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    for s in "lr":
        assert (
            np.linalg.norm(sim.body.grip(sim.d, s)[0] - sim.d.site(f"tote_1/grip_{s}").xpos) < 0.03
        )
    sim.ok("carry", to="bench", standoff=0.55)
    assert sim.d.body("tote_1").xpos[2] > 0.5  # carried at the hands' height, by the hold
    sim.ok("approach", target="bench", standoff=0.55, facing=math.pi / 2)
    sim.ok("place", at="bench")
    for _ in range(300):
        sim.e.step()
    tote = sim.d.body("tote_1").xpos
    assert abs(tote[2] - 0.75) < 0.02 and abs(tote[1] - 2.5) < 0.3  # resting on the bench top
    assert sim.h.status().error == ""


def test_place_where_the_target_is_beyond_the_hands_fails_without_dropping(sim):
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=1.2)
    st = sim.run("place", at="bench")
    assert st.error.startswith("out of reach") and "not bench" in st.error
    assert sim.d.body("tote_1").xpos[2] > 0.5  # still held


def test_a_start_while_busy_is_refused_and_a_cancel_keeps_a_hold(sim):
    sim.ok("approach", target="cab/drawer_1/grip", standoff=0.75)
    sim.ok("grasp", hand="right", target="cab/drawer_1/grip")
    seq = sim.h.start("pull", hand="right", distance=0.3)
    with pytest.raises(MachineError, match="busy"):
        sim.h.start("release")
    for _ in range(200):
        sim.e.step()
    sim.h.cancel(seq)
    for _ in range(5):
        sim.e.step()
    st = sim.h.status()
    assert st.finished and st.error == "cancelled"
    plugin = next(p for p in sim.e.plugins if type(p).__name__ == "HumanoidPlugin")
    assert plugin.holding["r"] is not None and plugin.holding["r"].active(sim.d)
    sim.e.reset()  # a reset drops the move and the hold
    assert plugin.holding["r"] is None and not sim.d.eq_active.any()


def test_a_cancelled_approach_stops_where_it_is(sim):
    seq = sim.h.start("approach", target="start/button", standoff=0.5)
    for _ in range(300):
        sim.e.step()
    x0, y0, _ = sim.body.root()
    assert math.hypot(x0, y0) > 0.1, "the approach did not start walking"
    sim.h.cancel(seq)
    for _ in range(5):
        sim.e.step()
    assert sim.h.status().error == "cancelled"
    x1, y1, _ = sim.body.root()
    for _ in range(500):
        sim.e.step()
    x2, y2, _ = sim.body.root()
    assert math.hypot(x2 - x1, y2 - y1) < 0.05, (
        "the walker walked on after its approach was cancelled"
    )


def test_a_defect_in_a_move_stops_the_run_unless_it_reports(sim, monkeypatch):
    plugin = next(p for p in sim.e.plugins if type(p).__name__ == "HumanoidPlugin")
    assert plugin.on_error == "raise"

    def broken(*_a, **_k):
        raise ZeroDivisionError("a defect")

    monkeypatch.setattr(plugin, "_stand_for", broken)
    sim.h.start("approach", target="start/button")
    with pytest.raises(RuntimeError, match="program 'approach'.*a defect"):
        for _ in range(5):
            sim.e.step()
    plugin.on_error = "report"
    try:
        sim.e.reset()
        st = sim.run("approach", target="start/button")
        assert "ZeroDivisionError: a defect" in st.error
    finally:
        plugin.on_error = "raise"


def test_a_humanoid_registers_no_entity_so_takes_no_present_and_owns_no_components():
    with pytest.raises(PluginError, match="HumanoidPlugin registers none"):
        Engine(load_config_from_dict(_world({"graspable": ["cab"], "present": False})))
    world = _world()
    world["components"][0]["components"][1]["components"] = [{"navigator": NAV}]
    with pytest.raises(Exception, match="humanoid"):
        load_config_from_dict(world)


def test_a_hold_pulled_past_its_force_lets_go_and_fails():
    engine = _engine(_world({"graspable": ["cab"], "hold": {"max_force": 5.0}}))
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="cab/drawer_1/grip", standoff=0.75)
        sim.ok("grasp", hand="right", target="cab/drawer_1/grip")
        st = sim.run("pull", hand="right", distance=0.3)
        assert st.error.startswith("lost hold")
        assert not sim.d.eq_active.any()
    finally:
        engine.shutdown()


def test_moves_need_hands_and_no_patrol_and_a_graspable_that_exists():
    with pytest.raises(PluginError, match="set 'hands: true' on walker worker"):
        Engine(load_config_from_dict(_world(hands=False)))
    with pytest.raises(PluginError, match="arrival_radius is 0.25 m"):
        Engine(
            load_config_from_dict(
                _world(navigator={k: v for k, v in NAV.items() if k != "arrival_radius"})
            )
        )
    with pytest.raises(PluginError, match="patrol"):
        Engine(
            load_config_from_dict(_world(navigator={**NAV, "goals": [[1.0, 0.0]], "loop": True}))
        )
    engine = Engine(load_config_from_dict(_world({"graspable": ["crate_*"]})))
    engine.ctx.seed = 1
    with pytest.raises(Exception, match="graspable 'crate_\\*' matches no grip site"):
        engine.setup()
