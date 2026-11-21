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


def test_a_carry_along_a_way_passes_its_points_and_its_load_meets_nothing(sim):
    """Carrying, the body walks a way planned for it and what it holds through the floor points of
    ``via``, in order; the tote touches nothing but the hands on the way."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    x, y, _ = sim.body.root()
    corner = (x, y + 1.3)  # north, then east toward the bench's side
    points = [corner, (-0.6, corner[1])]
    t0 = sim.e.ctx.sim_time
    seq = sim.h.carry("bench", standoff=0.55, facing=math.pi / 2, via=[list(p) for p in points])
    tote = sim.m.body("tote_1").id
    nearest = [math.inf, math.inf]
    touched = []
    while True:
        sim.e.step()
        bx, by, _ = sim.body.root()
        for i, p in enumerate(points):
            nearest[i] = min(nearest[i], math.hypot(bx - p[0], by - p[1]))
        if [
            ev
            for ev in sim.e.ctx.events.all()[sim._first :]
            if ev["event"] == "lifted" and ev["t"] >= t0
        ]:
            for c in sim.d.contact[: sim.d.ncon]:
                bodies = {int(sim.m.geom_bodyid[c.geom1]), int(sim.m.geom_bodyid[c.geom2])}
                if tote in bodies and c.dist < -1e-4:
                    other = (bodies - {tote}).pop() if len(bodies) > 1 else tote
                    if not sim.body.owns(sim.m, other):
                        touched.append(sim.m.body(other).name)
        st = sim.h.status()
        if st.seq == seq and st.finished:
            break
    assert st.error == "", st.error
    assert max(nearest) < 0.06, nearest
    assert touched == [], touched[:5]
    assert "planned" in sim.events(t0)


def test_carrying_it_turns_the_way_its_load_sweeps_clear():
    """A post stands where the short way round would swing the tote: the body turns the way its
    load sweeps clear, and the tote never meets the post."""
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


def test_lifted_says_how_high_the_hands_are(sim):
    """``lifted`` says the height the hands came to, not the one they were sent to: a lift ends
    within a few centimetres of where it aims."""
    sim.ok("approach", target="tote_1", standoff=0.6)
    sim.ok("grasp", hand="both", target="tote_1")
    seq = sim.h.carry("bench", standoff=0.55)
    while True:
        sim.e.step()
        lifted = [e for e in sim.e.ctx.events.all()[sim._first :] if e["event"] == "lifted"]
        if lifted:
            break
        assert not (sim.h.status().seq == seq and sim.h.status().finished)
    hands = [sim.d.site(f"worker/grip_{s}").xpos[2] for s in "lr"]
    assert lifted[0]["height"] == pytest.approx(float(np.mean(hands)), abs=0.002), (lifted, hands)


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
    the hands as far in front of the body as they reach at the height they set it down at, not
    short of that by their spread."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.9, facing=math.pi / 2)
    x, y, yaw = sim.body.root()
    ahead = np.array([math.cos(yaw), math.sin(yaw)])
    # how far the hand that holds it reaches with the tote set down on the bench (0.75 m); the
    # other rides on its grip
    plugin = next(p for p in sim.e.plugins if type(p).__name__ == "HumanoidPlugin")
    low = float(plugin._bottom(sim.m.body("tote_1").id)[2])
    (side,) = [s for s in "lr" if not plugin._riding[s]]
    reach = sim.body.place_reach(sim.d, side, 0.75 + float(sim.body.grip(sim.d, side)[0][2]) - low)
    seq = sim.h.place({"position": {"x": x, "y": 2.75, "z": 0.75}}, release=False)
    out = None
    while True:
        sim.e.step()
        if "lowering" in sim.events() and out is None:
            out = float((sim.body.grip(sim.d, side)[0][:2] - [x, y]) @ ahead)
        st = sim.h.status()
        if st.seq == seq and st.finished:
            break
    assert st.error == "", st.error
    assert reach > 0.7, reach  # the arm reaches farther than it stands from the bench's edge
    assert out > reach - 0.03, (out, reach)


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


def test_a_walk_whose_body_does_not_come_to_rest_fails(sim, monkeypatch):
    """A body that does not come to rest where it was walked to fails the move rather than
    starting the next one from where it is still moving."""
    monkeypatch.setattr(sim.body, "at_rest", lambda: False)
    st = sim.run("approach", target="tote_1", standoff=0.6)
    assert st.error.startswith("cannot stand at tote_1: the body did not come to rest"), st.error


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


def test_a_back_max_that_is_no_distance_fails_naming_it(sim):
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    st = sim.run("carry", to="bench", back_max=-0.1)
    assert st.error.startswith("back_max: a distance"), st.error


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


_SLIDING = """<mujoco><worldbody>
<geom name="floor" type="plane" size="6 6 .05"/>
<body name="slider" pos="1.0 0 1.1">
  <joint name="slider_slide" type="slide" axis="0 1 0" range="0 0.8" damping="20"/>
  <geom name="slider_leaf" type="box" size="0.02 0.4 0.3" mass="8"/>
  <site name="slider/grip" pos="-0.05 -0.32 0" quat="0 1 0 0"/>
</body></worldbody></mujoco>"""


def _box(name, x, y, z, size):
    return {
        "box": {
            "prefix": f"{name}_",
            "pose": {"position": {"x": x, "y": y, "z": z}},
            "size": size,
            "motion": "static",
        },
        "name": name,
    }


def test_a_place_through_an_opening_goes_under_its_lintel_not_over_it():
    """What is above the load on its way is not a surface it is set down over: a tote set on a
    fixture inside a housing, through an opening above a sill, passes under the lintel and the
    roof over the fixture, and comes down on the fixture."""
    world = _world()
    world["components"] += [
        _box("bed", -3.4, 0.0, 0.4, [0.8, 1.2, 0.8]),
        _box("sill", -2.93, 0.0, 0.425, [0.06, 1.2, 0.85]),
        _box("lintel", -2.93, 0.0, 1.85, [0.06, 1.2, 0.5]),
        _box("roof", -3.4, 0.0, 2.15, [1.0, 1.2, 0.1]),
        _box("fixture", -3.2, 0.0, 0.85, [0.44, 0.8, 0.1]),
    ]
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        sim.ok("carry", to="fixture", standoff=0.55, facing=math.pi)
        tote = sim.m.body("tote_1").id
        highest = 0.0
        seq = sim.h.place("fixture", release=False)
        while True:
            sim.e.step()
            highest = max(highest, float(sim.d.xpos[tote][2]))
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error == "", st.error
        assert highest < 1.6 - 0.15  # under the lintel's underside, the tote's height below it
        assert abs(sim.d.xpos[tote][2] - 0.9) < 0.02  # on the fixture, still in the hands
    finally:
        engine.shutdown()


def test_a_shallow_load_is_lifted_clear_of_the_table_it_was_taken_from_before_its_way():
    """A carry lifts what it holds clear of what it was taken from -- the load by CLEAR, and the
    arms holding it, which the way checks down to the hands -- before it plans the way: a shallow
    tote taken from a table, its grips near the table's top, is carried off it."""
    world = _world()
    for c in world["components"]:
        if c.get("name") == "tote_1":
            c["tote"]["size"] = [0.5, 0.35, 0.04]
            c["tote"]["pose"]["position"]["z"] = 0.75
    world["components"].append(_box("table", -1.5, -0.4, 0.375, [0.6, 1.0, 0.75]))
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        sim.ok("carry", to="bench", standoff=0.55, facing=math.pi / 2)
    finally:
        engine.shutdown()


_LONG_BAR = """<mujoco><worldbody>
<geom name="floor" type="plane" size="6 6 .05"/>
<geom name="arm_a" type="box" pos="-1.6 -0.45 0.95" size="0.3 0.04 0.03"/>
<geom name="arm_b" type="box" pos="-1.6 0.45 0.95" size="0.3 0.04 0.03"/>
<body name="bar" pos="-1.55 0 1.0035"><freejoint name="bar_free"/>
  <geom name="bar_box" type="box" size="0.0225 0.8 0.0225" mass="2.2"/>
  <site name="bar/grip_l" pos="0.0525 -0.3 0" quat="{left}"/>
  <site name="bar/grip_r" pos="0.0525 0.3 0" quat="{right}"/>
</body></worldbody></mujoco>"""


def _grip_quat(yaw):
    """A grip frame pushing along ``yaw``: +x that way, +z down (``roqsim_motion.hands``)."""
    c, s = math.cos(yaw), math.sin(yaw)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.array([c, s, 0.0, s, -c, 0.0, 0.0, 0.0, -1.0]))
    return " ".join(f"{v:.6f}" for v in q)


def test_a_long_bar_in_both_hands_is_held_level_and_carried_off_the_arms_it_lay_on(tmp_path):
    """The hand that holds a load the other rides on holds it level: a 1.6 m bar lifted off two arms
    by grips 0.6 m apart has its far grip within LEVEL_GRIPS of the near one before its way is
    planned, and is carried away. Hanging from the holding hand as the arm left it, turned a few
    degrees, the bar's far end stayed down on the arms and the way was refused where it began."""
    xml = tmp_path / "long_bar.xml"
    xml.write_text(
        _LONG_BAR.format(left=_grip_quat(math.pi - 0.5), right=_grip_quat(math.pi + 0.5))
    )
    world = _world({"graspable": ["bar"]})
    world["sim"]["world"] = str(xml)
    world["components"] = [c for c in world["components"] if c.get("name") == "worker"]
    world["components"][0]["walker"]["pose"]["position"] = {"x": -0.8, "y": 0.0}
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        here = {"position": {"x": -1.0, "y": 0.0, "z": 0.0}}
        sim.ok("approach", target=here, standoff=0.0, facing=math.pi)
        sim.ok("grasp", hand="both", target="bar")
        near, far = sim.m.site("bar/grip_l").id, sim.m.site("bar/grip_r").id
        seq = sim.h.carry({"position": {"x": 1.5, "y": 0.0, "z": 0.0}}, standoff=0.0, facing=0.0)
        level = None
        while True:
            sim.e.step()
            if level is None and "lifted" in sim.events():
                level = abs(float(sim.d.site_xpos[far][2] - sim.d.site_xpos[near][2]))
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error == "", st.error
        assert level is not None and level < 0.012, level
    finally:
        engine.shutdown()


_BAR_IN_A_RACK = """<mujoco><worldbody>
<geom name="floor" type="plane" size="6 6 .05"/>
<geom name="post_a" type="box" pos="-1.975 -0.45 1.1" size="0.075 0.075 1.1"/>
<geom name="post_b" type="box" pos="-1.975 0.45 1.1" size="0.075 0.075 1.1"/>
<geom name="arm_a" type="box" pos="-1.6 -0.45 0.95" size="0.3 0.04 0.03"/>
<geom name="arm_b" type="box" pos="-1.6 0.45 0.95" size="0.3 0.04 0.03"/>
<geom name="arm_a_up" type="box" pos="-1.6 -0.45 1.35" size="0.3 0.04 0.03"/>
<geom name="arm_b_up" type="box" pos="-1.6 0.45 1.35" size="0.3 0.04 0.03"/>
<body name="bar" pos="-1.5 0 1.0035"><freejoint name="bar_free"/>
  <geom name="bar_box" type="box" size="0.0225 0.8 0.0225" mass="2.2"/>
  <site name="bar/grip_l" pos="0.0525 -0.3 0" quat="{left}"/>
  <site name="bar/grip_r" pos="0.0525 0.3 0" quat="{right}"/>
</body></worldbody></mujoco>"""


def test_a_bar_lifted_between_a_racks_arms_at_the_side_is_drawn_out_and_stepped_away_from(
    tmp_path,
):
    """A bar lying across a rack's arms, taken by a body standing beside it facing along it, the
    bar at its right side, is lifted between the arms and the ones above: no way starts from
    there. The carry draws it out toward the body -- the hands as far as the trunk leaves room, the
    body stepping aside, its heading kept, for the rest -- until it is clear of the arms' tips, and
    then walks its way. The bar never comes into the trunk and leaves the arms level."""
    xml = tmp_path / "bar_in_a_rack.xml"
    xml.write_text(
        _BAR_IN_A_RACK.format(left=_grip_quat(math.pi - 0.5), right=_grip_quat(math.pi + 0.5))
    )
    world = _world({"graspable": ["bar"]})
    world["sim"]["world"] = str(xml)
    world["components"] = [c for c in world["components"] if c.get("name") == "worker"]
    world["components"][0]["walker"]["pose"]["position"] = {"x": -0.6, "y": 0.0}
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        here = {"position": {"x": -1.15, "y": 0.0, "z": 0.0}}
        sim.ok("approach", target=here, standoff=0.0, facing=-math.pi / 2)
        sim.ok("grasp", hand="both", target="bar")
        bar = sim.m.body("bar").id
        trunk = {
            sim.m.body(f"worker/{p}").id
            for p in ("pelvis", "spine", "chest", "neck", "head")
            if mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, f"worker/{p}") >= 0
        }
        assert trunk
        arms = {sim.m.geom(n).id for n in ("arm_a", "arm_b", "arm_a_up", "arm_b_up")}
        near, far = sim.m.site("bar/grip_l").id, sim.m.site("bar/grip_r").id
        seq = sim.h.carry({"position": {"x": 1.5, "y": -1.5, "z": 0.0}}, standoff=0.0, facing=0.0)
        bar_geom = sim.m.geom("bar_box").id
        trunk_geoms = [g for g in range(sim.m.ngeom) if int(sim.m.geom_bodyid[g]) in trunk]
        fromto = np.zeros(6)
        nearest, touched, tilt, out_at = 1.0, set(), 0.0, None
        while True:
            sim.e.step()
            # how near the bar comes to the trunk from the draw on, by the geometry (a held
            # load's contacts with the body that holds it are not all generated)
            for g in trunk_geoms if "drawing_out" in sim.events() else ():
                nearest = min(
                    nearest, mujoco.mj_geomDistance(sim.m, sim.d, bar_geom, g, 0.2, fromto)
                )
            for c in sim.d.contact[: sim.d.ncon]:
                b1, b2 = int(sim.m.geom_bodyid[c.geom1]), int(sim.m.geom_bodyid[c.geom2])
                if bar in (b1, b2):
                    if {int(c.geom1), int(c.geom2)} & arms and "lifted" in sim.events():
                        touched.add(sim.m.geom(int(c.geom1) if c.geom1 in arms else c.geom2).name)
            if "drawing_out" in sim.events() and out_at is None and "planned" in sim.events():
                out_at = float(sim.d.xpos[bar][0])
                tilt = abs(float(sim.d.site_xpos[far][2] - sim.d.site_xpos[near][2]))
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error == "", st.error
        events = sim.events()
        assert "drawing_out" in events and "stepping_aside" in events, events
        assert nearest > 0.0, nearest  # drawn out and carried, the bar never came into the trunk
        assert not touched, touched  # nor touched an arm once lifted
        assert out_at is not None and out_at > -1.3 + 0.0225, out_at  # clear of the arms' tips
        assert tilt < 0.012, tilt  # level as it left
    finally:
        engine.shutdown()


_BAR_TO_A_TABLE = _BAR_IN_A_RACK.replace(
    "</worldbody>",
    """<body name="table" pos="2.5 -0.7 0">
  <geom name="table_top" type="box" pos="0 0 0.425" size="1.0 0.3 0.425"/>
  <site name="table/top" pos="0.05 0.1 0.85"/>
</body></worldbody>""",
)


def test_a_bar_beside_the_body_is_set_down_to_the_side_as_far_as_the_arms_reach_there(tmp_path):
    """A bar carried at the body's side, along its heading, set down onto a table beside the body
    whose place is 0.2 m further out than the bar is held and a few centimetres ahead, with the
    front hand at its reach ahead already: the set-down goes out sideways by the arms' reach that
    way, the bar comes down wholly on the table. Clamped by the hands' reach ahead, a shift that
    was nearly all sideways was cut to nothing, and the bar came down half off the table's edge."""
    xml = tmp_path / "bar_to_a_table.xml"
    xml.write_text(
        _BAR_TO_A_TABLE.format(left=_grip_quat(math.pi - 0.5), right=_grip_quat(math.pi + 0.5))
    )
    world = _world({"graspable": ["bar"]})
    world["sim"]["world"] = str(xml)
    world["components"] = [c for c in world["components"] if c.get("name") == "worker"]
    world["components"][0]["walker"]["pose"]["position"] = {"x": -0.6, "y": 0.0}
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        here = {"position": {"x": -1.15, "y": 0.0, "z": 0.0}}
        sim.ok("approach", target=here, standoff=0.0, facing=-math.pi / 2)
        sim.ok("grasp", hand="both", target="bar")
        # beside the table, along it, the bar at the right side over the table's near edge
        sim.ok("carry", to={"position": {"x": 2.5, "y": -0.05, "z": 0.0}}, standoff=0.0, facing=0.0)
        bar = sim.m.geom("bar_box").id
        assert sim.d.geom_xpos[bar][1] > -0.45  # held over the edge, short of the place
        sim.ok("place", at="table/top")
        for _ in range(300):
            sim.e.step()
        rot = sim.d.geom_xmat[bar].reshape(3, 3)
        half = sim.m.geom_size[bar]
        corners = [
            sim.d.geom_xpos[bar] + rot @ (half * [sx, sy, -1.0]) for sx in (-1, 1) for sy in (-1, 1)
        ]
        for c in corners:  # every corner of its underside over the table, resting on it
            assert 1.5 < c[0] < 3.5 and -1.0 < c[1] < -0.4, c
            assert abs(c[2] - 0.85) < 0.01, c
    finally:
        engine.shutdown()


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


def test_a_drawn_mark_flush_on_a_bench_does_not_hide_the_bench_from_a_place():
    """A mark drawn on the bench's top -- a thin box that collides with nothing, lying flush on
    it -- is no surface, and does not hide the bench under it: the tote is set down on the bench.
    Skipped by restarting the ray just under it, the ray started inside the bench, missed it, and
    the place read the floor."""
    world = _world()
    world["components"].append(
        {
            "box": {
                "prefix": "mark_",
                "pose": {"position": {"x": 0.0, "y": 2.45, "z": 0.751}},
                "size": [0.5, 0.4, 0.002],
                "motion": "static",
                "collide": False,
            },
            "name": "mark",
        }
    )
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="tote_1", standoff=0.45)
        sim.ok("grasp", hand="both", target="tote_1")
        sim.ok("carry", to="bench", standoff=1.1, facing=math.pi / 2)
        sim.ok("approach", target="bench", standoff=0.55, facing=math.pi / 2)
        sim.ok("place", at="bench")
        for _ in range(300):
            sim.e.step()
        assert abs(sim.d.body("tote_1").xpos[2] - 0.75) < 0.02, sim.d.body("tote_1").xpos
    finally:
        engine.shutdown()


def test_a_place_that_would_leave_it_half_off_what_is_under_it_fails(sim):
    """What is set down stands on one thing, level: set down where the hands reach, a tote half
    over the bench's edge is refused, and one wholly on it is set down."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=1.1, facing=math.pi / 2)  # the tote over the edge
    st = sim.run("place", at="bench")
    assert st.error.startswith("out of reach") and "closer" in st.error, st.error
    assert sim.d.body("tote_1").xpos[2] > 0.7  # still held
    sim.ok("approach", target="bench", standoff=0.55, facing=math.pi / 2)
    sim.ok("place", at="bench")
    for _ in range(300):
        sim.e.step()
    assert abs(sim.d.body("tote_1").xpos[2] - 0.75) < 0.02


def test_a_place_reaches_as_far_as_the_arm_solve_puts_the_hand(sim):
    """A place reaches as far ahead as the walker's arm solve puts the hand that holds the load, at
    the height it sets it down at, bending over as the solve does: standing 0.75 m from the bench's
    middle, the tote is set down wholly on it. Held to a fixed 0.55 m, it came down half over the
    bench's near edge, and the place was refused."""
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=0.75, facing=math.pi / 2)
    sim.ok("place", at="bench")
    for _ in range(300):
        sim.e.step()
    assert abs(sim.d.body("tote_1").xpos[2] - 0.75) < 0.02  # on the bench


def test_a_place_half_over_a_floor_that_collides_with_nothing_fails_too(sim):
    """A floor drawn but colliding with nothing (a hall's) is nothing to stand on: a corner over it
    is a corner over nothing."""
    floor = [g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] == 0]
    saved = sim.m.geom_contype[floor].copy(), sim.m.geom_conaffinity[floor].copy()
    sim.ok("approach", target="tote_1", standoff=0.45)
    sim.ok("grasp", hand="both", target="tote_1")
    sim.ok("carry", to="bench", standoff=1.2, facing=math.pi / 2)
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
    sim.ok("carry", to="bench", standoff=1.6)
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


@pytest.mark.parametrize("side_of_handle", [0.25, -0.25])
def test_a_slide_along_the_face_of_its_handle_opens_pulled_and_shuts_pushed(
    tmp_path, side_of_handle
):
    """A slide along the face its handle is on -- a sliding door's -- is pulled open and pushed shut
    along its joint's axis, whichever side of the handle the body stands: toward or away from the
    body says nothing along that face. Taken toward and away from the body, the door slid open or
    to its shut stop by where the body happened to stand."""
    xml = tmp_path / "sliding.xml"
    xml.write_text(_SLIDING)
    world = _world({"graspable": ["slider"]})
    world["sim"]["world"] = str(xml)
    world["components"] = [c for c in world["components"] if c.get("name") == "worker"]
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        stand = {"position": {"x": 0.5, "y": -0.32 + side_of_handle, "z": 0.0}}
        sim.ok("approach", target=stand, standoff=0.0, facing=0.0)
        sim.ok("grasp", hand="right", target="slider/grip")
        sim.ok("pull", hand="right", distance=0.25)
        opened = sim.qpos("slider_slide")
        assert opened > 0.2, opened  # opened along the joint's axis
        sim.ok("push", hand="right", distance=0.2)
        assert sim.qpos("slider_slide") < opened - 0.15  # shut along it again
    finally:
        engine.shutdown()


def _part_on_a_table(z_board):
    """A small part with a grip on a table, and a board at ``z_board`` (m) to set it on."""
    world = _world({"graspable": ["part"]})
    grip = {"kind": "grip", "at": [0.0, 0.0, 0.0, -math.pi / 2], "stand": [[0.0, 0.5, -math.pi / 2]]}
    world["components"] += [
        {
            "box": {
                "prefix": "table_",
                "pose": {"position": {"x": -1.5, "y": -0.3, "z": 0.375}},
                "size": [0.6, 0.8, 0.75],
                "motion": "static",
            },
            "name": "table",
        },
        {
            "box": {
                "prefix": "part_",
                "pose": {"position": {"x": -1.5, "y": 0.0, "z": 0.78}},
                "size": [0.1, 0.08, 0.06],
                "motion": "physics",
            },
            "name": "part",
            "components": [{"slots": {"body": "part_box", "slots": {"grip": grip}}}],
        },
        {
            "box": {
                "prefix": "board_",
                "pose": {"position": {"x": 2.0, "y": 2.5, "z": z_board - 0.02}},
                "size": [0.6, 0.4, 0.04],
                "motion": "static",
            },
            "name": "board",
        },
    ]
    return world


def test_a_part_in_one_hand_is_carried_at_the_side_where_the_hand_hangs_to_a_high_board():
    """A part taken in one hand goes to a board above the shoulder carried at the body's side
    where the hand hangs: held palm forward beside the shoulder, lifted to clear the board, the
    hand could not make it."""
    engine = _engine(_part_on_a_table(1.45))
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="part/grip")
        sim.ok("grasp", hand="right", target="part/grip")
        sim.ok("carry", to="board", standoff=0.5, facing=math.pi / 2)
        part = sim.d.xpos[sim.m.body("part_box").id]
        assert part[2] < 1.2, part  # at the side, below the board
    finally:
        engine.shutdown()


@pytest.mark.parametrize("height", [1.3, 1.45])
def test_a_part_in_one_hand_is_set_on_a_board_above_the_shoulder(height):
    """Set down above where the hand hangs, a part held in one hand is first brought before the
    shoulder, the palm turned in and the fingers ahead, and raised from there onto the board: held
    palm forward and the fingers down, as it was carried, the hand could not raise it."""
    engine = _engine(_part_on_a_table(height))
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="part/grip")
        sim.ok("grasp", hand="right", target="part/grip")
        sim.ok("carry", to="board", standoff=0.5, facing=math.pi / 2)
        sim.ok("place", at="board", release=False)
        sim.ok("release", hand="right")
        for _ in range(500):
            sim.e.step()
        part = sim.d.xpos[sim.m.body("part_box").id]
        assert height + 0.02 < part[2] < height + 0.06, part  # resting on the board
        assert abs(part[0] - 2.0) < 0.3 and abs(part[1] - 2.5) < 0.2, part
    finally:
        engine.shutdown()


def test_a_part_in_one_hand_is_raised_over_the_bench_under_the_board_not_through_it():
    """Brought before the shoulder to be raised onto a board over a bench, a part in one hand goes
    up clear of the bench first: coming forward where the hand hangs, it went into the bench."""
    world = _part_on_a_table(1.45)
    world["components"].append(
        {
            "box": {
                "prefix": "bench2_",
                "pose": {"position": {"x": 2.0, "y": 2.45, "z": 0.475}},
                "size": [0.8, 0.5, 0.95],
                "motion": "static",
            },
            "name": "bench2",
        }
    )
    engine = _engine(world)
    try:
        engine.reset()
        sim = Sim(engine)
        sim.ok("approach", target="part/grip")
        sim.ok("grasp", hand="right", target="part/grip")
        sim.ok("carry", to="board", standoff=0.6, facing=math.pi / 2)
        part, bench = sim.m.body("part_box").id, sim.m.body("bench2_box").id
        deepest = 0.0
        seq = sim.h.place("board", release=False)
        while True:
            sim.e.step()
            for c in sim.d.contact[: sim.d.ncon]:
                if {int(sim.m.geom_bodyid[c.geom1]), int(sim.m.geom_bodyid[c.geom2])} == {
                    part,
                    bench,
                }:
                    deepest = max(deepest, -float(c.dist))
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
        assert st.error == "", st.error
        assert deepest < 0.001, deepest
    finally:
        engine.shutdown()
