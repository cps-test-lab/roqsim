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
from roqsim.presence import PARKED_BELOW_Z

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
            {
                "box": {
                    "prefix": "wall_",
                    "pose": {"position": {"x": 0.0, "y": -3.0}},
                    "size": [2.0, 0.02, 0.9],
                    "motion": "static",
                },
                "name": "wall",
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
        # the shared engine's events of the tests before this one are not this test's
        self._first = len(engine.ctx.events.all())

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
            for ev in self.e.ctx.events.all()[self._first :]
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


def _nearest_on_the_way(sim, seq, point, limit=30.0):
    """Step until the move ``seq`` ends; how near the body's root came to ``point`` on the way."""
    nearest, t0 = math.inf, sim.e.ctx.sim_time
    while True:
        sim.e.step()
        x, y, _ = sim.body.root()
        nearest = min(nearest, math.hypot(x - point[0], y - point[1]))
        st = sim.h.status()
        if st.seq == seq and st.finished:
            return st, nearest
        assert sim.e.ctx.sim_time - t0 < limit


def test_approach_and_carry_walk_through_their_via_points_first(sim):
    # The straight way to the button passes nowhere near (-0.5, 1.5); the via makes it.
    seq = sim.h.approach("start/button", standoff=0.5, via=[[-0.5, 1.5]])
    st, nearest = _nearest_on_the_way(sim, seq, (-0.5, 1.5))
    assert st.error == "" and nearest < 0.1
    x, y, _ = sim.body.root()
    assert math.hypot(x - 1.5, y - 1.0) < 0.3
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    seq = sim.h.carry("bench", standoff=0.55, via=[[-1.4, 1.6]])
    st, nearest = _nearest_on_the_way(sim, seq, (-1.4, 1.6))
    assert st.error == "" and nearest < 0.1
    assert sim.d.body("tote_1").xpos[2] > 0.5  # still held


def test_a_carry_stands_facing_the_way_it_is_given(sim):
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
    x, y, yaw = sim.body.root()
    assert abs(math.atan2(math.sin(yaw - math.pi / 2), math.cos(yaw - math.pi / 2))) < math.radians(
        4
    )
    assert math.hypot(x - 0.0, y - (2.5 - 0.55)) < 0.1


def test_with_a_body_lag_a_carried_tote_stays_in_front_of_the_drawn_body():
    """A walker drawn behind its nav root carries a tote in front of the body drawn, not of the nav
    root: the hands start and stop with the body, so the tote neither runs ahead of it nor is
    jerked by a root that starts at once."""
    world = _world()
    world["components"][0]["walker"]["body_lag"] = 0.15
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)  # walks some 3 m
        pelvis = sim.m.body("worker/pelvis").id
        tote = sim.m.body("tote_1").id
        ahead = []
        seq = sim.h.approach("start/button", standoff=0.6)
        while True:
            sim.e.step()
            d = sim.d.xpos[tote][:2] - sim.d.xpos[pelvis][:2]
            ahead.append(float(np.linalg.norm(d)))
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error == ""
        assert max(ahead) - min(ahead) < 0.12, (min(ahead), max(ahead))
        # and a walk ends once the drawn body has come to rest, not when its root arrives
        st = sim.body.st
        assert np.hypot(*(np.asarray(st.body_pos[:2]) - np.asarray(st.pos[:2]))) < 0.006
    finally:
        engine.shutdown()


def test_a_carry_whose_first_via_point_is_where_it_stands_sets_off_toward_the_next(sim):
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    x, y, _ = sim.body.root()
    seq = sim.h.carry("bench", standoff=0.55, via=[[x + 0.01, y], [-1.4, 1.6]])
    st, nearest = _nearest_on_the_way(sim, seq, (-1.4, 1.6))
    assert st.error == "" and nearest < 0.1


def test_a_carry_along_a_way_turns_at_its_corners_so_the_load_stays_over_the_way(sim):
    """Carrying, the body turns at a carrying pace; along a way it stops and turns at each point,
    so the tote in front of it stays over the way's legs instead of swinging out at a corner."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    x, y, _ = sim.body.root()
    corner = (x, y + 1.6)  # straight north, then a right angle east toward the bench's side
    seq = sim.h.carry(
        "bench", standoff=0.55, facing=math.pi / 2, via=[list(corner), [-0.2, corner[1]]]
    )
    tote = sim.m.body("tote_1").id
    off = 0.0
    while True:
        sim.e.step()
        p = sim.d.xpos[tote]
        if y + 0.6 < p[1] < corner[1] - 0.1 and p[0] < x + 0.3:  # along the first leg: over it
            off = max(off, abs(p[0] - x))
        elif p[1] > corner[1] - 0.1:  # past the corner: over the second leg
            off = max(off, abs(p[1] - corner[1]) if x + 0.6 < p[0] < -0.5 else 0.0)
        st = sim.h.status()
        if st.seq == seq and st.finished:
            break
    assert st.error == "" and off < 0.12, off


def test_carrying_it_turns_the_way_its_load_sweeps_clear():
    """A post stands where the short way round would swing the tote: the body turns the long way,
    and the tote never meets the post."""
    world = _world()
    world["components"][0]["walker"]["pose"] = {
        "position": {"x": 0.0, "y": 0.0},
        "orientation": {"yaw": 0.0},
    }
    world["components"] = [c for c in world["components"] if c.get("name") != "tote_1"] + [
        {
            "tote": {"pose": {"position": {"x": 0.6, "y": 0.0}, "orientation": {"yaw": 1.5708}}},
            "name": "tote_1",
        },
        {
            "box": {
                "prefix": "post_",
                "pose": {"position": {"x": 0.1, "y": 0.6}},
                "size": [0.1, 0.1, 1.8],
                "motion": "static",
            },
            "name": "post",
        },
    ]
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        x, y, _ = sim.body.root()
        to = {"position": {"x": x - 1.6, "y": y + 0.9, "z": 0.0}}
        seq = sim.h.carry(to, standoff=0.6)
        post = sim.m.body("post_box").id
        tote = sim.m.body("tote_1").id
        nearest = math.inf
        while True:
            sim.e.step()
            nearest = min(
                nearest, float(np.linalg.norm(sim.d.xpos[tote][:2] - sim.d.xpos[post][:2]))
            )
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error == "", st.error
        assert nearest > 0.3, nearest
    finally:
        engine.shutdown()


def test_a_carry_lifts_straight_up_before_it_draws_the_load_in(sim):
    """Off what it stood on, the load goes straight up first: drawn in at once, a box would drag
    what it stands in (a drawer's stops) along with it."""
    sim.ok("approach", target="tote_1", standoff=0.6)
    sim.ok("grasp", hand="both", target="tote_1")
    tote = sim.m.body("tote_1").id
    start = sim.d.xpos[tote].copy()
    seq = sim.h.carry("bench", standoff=0.55)
    drift = 0.0
    while sim.d.xpos[tote][2] < start[2] + 0.04:
        sim.e.step()
        drift = max(drift, float(np.linalg.norm(sim.d.xpos[tote][:2] - start[:2])))
        assert not (sim.h.status().seq == seq and sim.h.status().finished)
    assert drift < 0.005, drift


def test_a_place_lowers_only_once_the_load_is_over_the_place(sim):
    """The hold is a spring: the load comes over the place a moment after the hands. It is lowered
    once it is there, so it does not come down on a rim beside the place; and it is set down only
    once it rests at the place's height."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
    seq = sim.h.place({"position": {"x": 0.1, "y": 2.45, "z": 0.75}}, release=False)
    tote = sim.m.body("tote_1").id
    seen = 0
    at_lowering = at_set_down = None
    while True:
        sim.e.step()
        events = [e for e in sim.e.ctx.events.all()[sim._first :] if e["source"] == "worker"]
        for e in events[seen:]:
            if e["event"] == "lowering":
                at_lowering = sim.d.xpos[tote].copy()
            if e["event"] == "set_down":
                at_set_down = sim.d.xpos[tote].copy()
        seen = len(events)
        st = sim.h.status()
        if st.seq == seq and st.finished:
            break
    assert st.error == ""
    assert np.linalg.norm(at_lowering[:2] - [0.1, 2.45]) < 0.015, at_lowering
    assert abs(at_set_down[2] - 0.75) < 0.02, at_set_down


def test_a_place_beyond_reach_sets_the_load_down_as_far_forward_as_the_hands_reach(sim):
    """A load held by its ends has its hands to the sides: a place past its reach sets it down with
    the hands as far in front of the body as they reach, not short of that by their spread."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
    x, y, yaw = sim.body.root()
    ahead = np.array([math.cos(yaw), math.sin(yaw)])
    seq = sim.h.place({"position": {"x": x, "y": 2.75, "z": 0.75}}, release=False)
    out = None
    while True:
        sim.e.step()
        if "lowering" in sim.events() and out is None:
            out = max(float((sim.body.grip(sim.d, s)[0][:2] - [x, y]) @ ahead) for s in "lr")
        st = sim.h.status()
        if st.seq == seq and st.finished:
            break
    assert st.error == "", st.error
    assert out > sim.body.place_reach() - 0.015, out


def test_a_place_the_hands_cannot_bring_the_load_over_fails_holding_it(sim, monkeypatch):
    """Hands that stop short of the place -- an arm at its full stretch -- leave the load short of
    it: the place fails, saying how near it came, and keeps holding it rather than lowering it
    beside the place."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
    tote = sim.m.body("tote_1").id
    # each hand stops 2 cm before it would bring the load over the place
    limit = {
        s: 2.45 - 0.02 + float(sim.body.grip(sim.d, s)[0][1] - sim.d.xpos[tote][1]) for s in "lr"
    }
    set_hand = sim.body.set_hand

    def short(side, pos, quat=None, frame="grip"):
        pos = np.array(pos, float)
        pos[1] = min(pos[1], limit[side])
        set_hand(side, pos, quat, frame)

    monkeypatch.setattr(sim.body, "set_hand", short)
    st = sim.run("place", at={"position": {"x": 0.0, "y": 2.45, "z": 0.75}})
    assert "and no nearer" in st.error, st.error
    assert "lowering" not in sim.events()
    assert not {"set_down", "released"} & set(sim.events())  # still held


def test_a_place_whose_load_comes_down_on_something_above_the_place_fails_caught_on_it():
    """A load that comes to rest on something above the place's surface fails, caught on it, and
    is held: let go there, it would fall."""
    world = _world()
    world["components"].append(
        {
            "box": {
                "prefix": "stop_",
                "pose": {"position": {"x": 5.0, "y": 5.0, "z": 0.05}},
                "size": [0.1, 0.1, 0.1],
                "motion": "driven",
            },
            "name": "stop",
        }
    )
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
        tote = sim.m.body("tote_1").id
        stop = int(sim.m.body_mocapid[sim.m.body("stop_box").id])
        seq = sim.h.place({"position": {"x": 0.0, "y": 2.45, "z": 0.75}})
        slipped = False
        while True:
            sim.e.step()
            if not slipped and "lowering" in sim.events():
                # a block slips in under it, its top 5 cm above the place
                slipped = True
                p = sim.d.xpos[tote]
                sim.d.mocap_pos[stop] = [p[0], p[1], p[2] - 0.055]
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error.startswith("caught on stop_box"), st.error
        assert not {"set_down", "released"} & set(sim.events())  # still held
    finally:
        engine.shutdown()


def test_a_place_with_a_heading_sets_the_load_down_square_to_it(sim):
    """A place with a frame -- a site's, or a world pose with a heading -- takes the load square to
    it, to the nearest quarter turn: the hold keeps a load as it was taken, a few degrees off."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
    at = {"position": {"x": 0.0, "y": 2.45, "z": 0.75}, "orientation": {"yaw": 0.25}}
    sim.ok("place", at=at)
    for _ in range(300):
        sim.e.step()
    rot = sim.d.body("tote_1").xmat.reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    off = (yaw - 0.25 + math.pi / 4) % (math.pi / 2) - math.pi / 4
    assert abs(math.degrees(off)) < 2.0, math.degrees(yaw)


def test_a_via_that_is_no_list_of_floor_points_fails_naming_it(sim):
    st = sim.run("approach", target="start/button", via=[[1.0, "north"]])
    assert st.error.startswith("via: a point is [x, y]")
    st = sim.run("approach", target="start/button", via=[[1.0, 2.0, 3.0]])
    assert st.error.startswith("via: a point is [x, y]")


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
    assert sim.d.mocap_pos[sim.body._proxy["r"]][2] < PARKED_BELOW_Z  # parked again


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
    pulled = [e for e in sim.e.ctx.events.all()[sim._first :] if e["event"] == "pulled"][-1]
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


def test_a_tote_is_carried_to_and_placed_at_a_world_pose_looking_at_a_world_point(sim):
    top = {"position": {"x": 0.0, "y": 2.5, "z": 0.75}}
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to=top, standoff=0.55, facing=math.pi / 2)
    sim.ok("look_at", target=[0.0, 2.5, 0.75])
    sim.ok("place", at=top)
    for _ in range(300):
        sim.e.step()
    tote = sim.d.body("tote_1").xpos
    assert abs(tote[2] - 0.75) < 0.02 and abs(tote[1] - 2.5) < 0.3


def test_a_place_goes_over_what_it_passes_on_the_way_not_through_it(sim):
    """What is set down is moved over the highest thing between where it is and where it goes,
    not only over the target's own surface: a way across the bench (0.75 m) passes over it, one
    beside it does not."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    plugin = next(p for p in sim.e.plugins if type(p).__name__ == "HumanoidPlugin")
    tote = sim.m.body("tote_1").id
    across = plugin._path_top([-1.5, 2.5], [1.5, 2.5], tote, 0.0)
    beside = plugin._path_top([-1.5, 1.5], [1.5, 1.5], tote, 0.0)
    assert abs(across - 0.75) < 0.01 and beside < 0.01
    # a wall 2 cm thick, thinner than the tote, is found under it on the way across
    assert abs(plugin._top([0.0, -3.07, 0.0], tote) - 0.9) < 0.01
    assert abs(plugin._path_top([0.0, -2.6], [0.0, -3.4], tote, 0.0) - 0.9) < 0.01


def test_what_rides_in_a_held_tote_is_no_surface_to_set_it_down_on():
    world = _world()
    world["components"].append(
        {
            "box": {
                "pose": {"position": {"x": -1.5, "y": 0.0, "z": 0.06}},
                "size": [0.1, 0.1, 0.1],
                "motion": "physics",
                "prefix": "part_",
            },
            "name": "part",
        }
    )
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        for _ in range(200):  # the part settles in the tote
            sim.e.step()
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        sim.ok("carry", to="bench", standoff=1.3)  # over the floor, short of the bench
        plugin = next(p for p in sim.e.plugins if type(p).__name__ == "HumanoidPlugin")
        tote = sim.m.body("tote_1").id
        part = sim.d.xpos[sim.m.body("part_box").id].copy()
        assert part[2] > 0.5  # carried in the tote
        top, under = plugin._surface(np.array([part[0], part[1], 0.0]), tote)
        assert top[2] < 0.01 and under == 0  # the floor, under the tote and the part in it
        # jolted off the tote's floor by the walk, touching nothing, it still rides in it
        free = sim.m.jnt_qposadr[sim.m.body("part_box").jntadr[0]]
        sim.d.qpos[free + 2] += 0.03
        mujoco.mj_forward(sim.m, sim.d)
        top, under = plugin._surface(np.array([part[0], part[1], 0.0]), tote)
        assert top[2] < 0.01 and under == 0
        here = sim.d.xpos[tote][:2].copy()
        assert plugin._path_top(here, here, tote, 0.0) < 0.01
        # set down, it comes down onto the bench, not let go where the part touches it
        sim.ok("approach", target="bench", standoff=0.55, facing=math.pi / 2)
        sim.ok("place", at="bench", release=False)
        assert abs(sim.d.xpos[tote][2] - 0.75) < 0.02  # on the bench, still in the hands
    finally:
        engine.shutdown()


def test_a_place_that_would_leave_it_half_off_what_is_under_it_fails(sim):
    """What is set down stands on one thing, level: set down where the hands reach, a tote half
    over the bench's edge is refused, and one wholly on it is set down."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.75, facing=math.pi / 2)  # the tote over the edge
    st = sim.run("place", at="bench")
    assert st.error.startswith("out of reach") and "closer" in st.error, st.error
    assert sim.d.body("tote_1").xpos[2] > 0.7  # still held
    sim.ok("approach", target="bench", standoff=0.55, facing=math.pi / 2)
    sim.ok("place", at="bench")
    for _ in range(300):
        sim.e.step()
    assert abs(sim.d.body("tote_1").xpos[2] - 0.75) < 0.02


def test_a_place_half_over_a_floor_that_collides_with_nothing_fails_too(sim):
    """A floor drawn but colliding with nothing (a hall's) is nothing to stand on: a corner over it
    is a corner over nothing."""
    floor = [g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] == 0]
    saved = sim.m.geom_contype[floor].copy(), sim.m.geom_conaffinity[floor].copy()
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.75, facing=math.pi / 2)
    try:
        sim.m.geom_contype[floor] = 0
        sim.m.geom_conaffinity[floor] = 0
        st = sim.run("place", at={"position": {"x": 0.0, "y": 2.5, "z": 0.75}})
        assert st.error.startswith("out of reach") and "closer" in st.error, st.error
    finally:
        sim.m.geom_contype[floor], sim.m.geom_conaffinity[floor] = saved


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


def _raise_clip(path, frames=31):
    """One second of the right arm raised sideways to the shoulder."""
    from roqsim_walker.humanoid import JOINT_NAMES
    from roqsim_walker.motion import Clip

    rot = np.tile([1.0, 0.0, 0.0, 0.0], (frames, len(JOINT_NAMES), 1))
    a = -math.pi / 2
    rot[:, JOINT_NAMES.index("shoulder_r")] = [math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]
    Clip(rot, np.zeros(frames), fps=30.0, stride_len=1.0, loop=False).save(path)
    return str(path)


def test_play_ends_with_its_clip_and_a_looping_one_once_it_has_faded_in(sim, tmp_path):
    clip = _raise_clip(tmp_path / "raise.npz")
    t = sim.e.ctx.sim_time
    sim.ok("play", clip=clip, joints=["arm_r"], fade=0.1)
    assert sim.e.ctx.sim_time - t > 0.9  # the whole clip
    assert sim.e.ctx.blackboard.get("walker:worker").playing() == []
    sim.ok("play", clip=clip, joints=["arm_r"], loop=True, fade=0.2)
    assert len(sim.e.ctx.blackboard.get("walker:worker").playing()) == 1
    sim.ok("stop_clip", clip=clip)
    assert sim.e.ctx.blackboard.get("walker:worker").playing() == []
    assert {"playing", "played", "stopped"} <= set(sim.events(t))


def test_a_cancelled_play_fades_its_clip_out_and_a_hand_comes_over_a_clip(sim, tmp_path):
    clip = _raise_clip(tmp_path / "raise.npz", frames=300)
    sim.h.start("play", clip=clip, joints=["arm_r"], fade=0.1)
    for _ in range(50):
        sim.e.step()
    sim.h.cancel()
    for _ in range(200):
        sim.e.step()
    assert sim.e.ctx.blackboard.get("walker:worker").playing() == []
    sim.ok("play", clip=clip, joints=["arm_r"], loop=True, fade=0.0)
    sim.ok("approach", target="start/button", standoff=0.5)
    sim.ok("reach", hand="right", target="start/button", offset=[0.1, 0.0, 0.0], orient=False)
    p, _ = sim.body.grip(sim.d, "r")
    assert np.linalg.norm(p - (sim.d.site("start/button").xpos + [-0.1, 0, 0])) < 0.012


def test_play_of_what_cannot_be_played_fails_with_the_reason(sim):
    assert "no clip 'wave'" in sim.run("play", clip="wave").error
    assert "no joint or joint group" in sim.run("play", clip="walk", joints=["wing"]).error


def test_look_at_turns_the_head_onto_a_target_and_back(sim):
    t = sim.e.ctx.sim_time
    sim.ok("look_at", target="start/button")
    assert "looking" in sim.events(t)
    head = sim.body.st.mocap["head"]
    q, pos = sim.d.mocap_quat[head], sim.d.mocap_pos[head]
    to = sim.d.site("start/button").xpos - pos
    from roqsim_walker.humanoid import quat_rotate

    f = quat_rotate(q, [1.0, 0.0, 0.0])
    assert abs(math.atan2(f[1], f[0]) - math.atan2(to[1], to[0])) < math.radians(3)
    sim.ok("look_at")
    assert "unknown target" in sim.run("look_at", target="nobody").error


def test_reach_through_the_handle_is_refused_while_a_move_runs(sim):
    handle = sim.e.ctx.blackboard.get("walker:worker")
    sim.h.start("approach", target="start/button", standoff=0.5)
    sim.e.step()
    with pytest.raises(RuntimeError, match="reach refused while its move 'approach' runs"):
        handle.reach(right=[1.0, 0.0, 1.0])
    sim.h.cancel()


def test_a_body_that_plays_no_clips_or_turns_no_head_says_so():
    from roqsim_motion.body import HumanoidBody

    body = HumanoidBody("robot")
    with pytest.raises(NotImplementedError, match="robot plays no clips"):
        body.play("wave")
    with pytest.raises(NotImplementedError, match="robot turns no head"):
        body.set_gaze([1.0, 0.0, 1.0])


def test_a_look_at_a_new_target_ends_only_once_the_head_is_on_it(sim):
    sim.ok("look_at", target="start/button")  # ahead, to its left
    t = sim.e.ctx.sim_time
    sim.ok("look_at", target="cab/drawer_1/grip")  # ahead, to its right: the head turns a while
    assert sim.e.ctx.sim_time - t > 0.15
    head = sim.body.st.mocap["head"]
    to = sim.d.site("cab/drawer_1/grip").xpos - sim.d.mocap_pos[head]
    from roqsim_walker.humanoid import quat_rotate

    f = quat_rotate(sim.d.mocap_quat[head], [1.0, 0.0, 0.0])
    assert abs(math.atan2(f[1], f[0]) - math.atan2(to[1], to[0])) < math.radians(3)
