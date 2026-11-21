"""A door on a hinge, opened by the hand: ``pull`` and ``push`` by an ``angle`` go round the hinge.

The hand follows the arc the hinge sets, and where the arc leaves its reach the body steps round
the hinge with the leaf, so a body outside the leaf's swing stays outside it. The leaf turns because
the hand pulls it (the hold is a weld); its closer, the hinge's spring, shuts it once let go. A hand
on something on a joint keeps it where it is while the body walks, the way kept within its reach.
The door is ``data/hinge_door.xml``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_motion import quat as Q
from roqsim_walker.humanoid import COLLISION_RADIUS

DATA = Path(__file__).parent / "data"
NAV = {
    "output": "walker",
    "speed": 1.0,
    "arrival_radius": 0.05,
    "avoidance": {"steer": "none", "stop": False},
}


def _engine(start, yaw, extra=(), graspable=("gate",), world="hinge_door.xml"):
    world = {
        "sim": {"pacing": "asap", "world": str(DATA / world)},
        "components": [
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "hands": True,
                    "skin": False,
                    "pose": {
                        "position": {"x": start[0], "y": start[1]},
                        "orientation": {"yaw": yaw},
                    },
                },
                "name": "worker",
                "components": [
                    {"navigator": NAV},
                    {"humanoid": {"graspable": list(graspable)}},
                ],
            },
            *extra,
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


class Door:
    def __init__(self, engine):
        self.e = engine
        self.h = engine.ctx.blackboard.get("humanoid:worker")
        self.body = engine.ctx.blackboard.get("humanoid_body:worker")
        self.m, self.d = engine.ctx.model, engine.ctx.data
        self.qadr = int(self.m.jnt_qposadr[self.m.joint("gate_hinge").id])
        self.leaf = self.m.body("gate").id
        self.deepest = 0.0  # how far a body geom went into the leaf, at most (m)

    def angle(self) -> float:
        return math.degrees(float(self.d.qpos[self.qadr]))

    def step(self):
        self.e.step()
        for c in self.d.contact[: self.d.ncon]:
            bodies = {int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])}
            if self.leaf in bodies:
                other = (bodies - {self.leaf}).pop() if len(bodies) > 1 else self.leaf
                if self.body.owns(self.m, other):
                    self.deepest = max(self.deepest, -float(c.dist))

    def leaf_to_root(self) -> float:
        """How far the body's root is from the leaf's middle line, on the floor (m)."""
        g = self.m.geom("gate_leaf").id
        c, rot, half = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3), self.m.geom_size[g]
        a, b = c - half[1] * rot[:, 1], c + half[1] * rot[:, 1]
        x, y, _ = self.body.root()
        e = (b - a)[:2]
        t = min(max(float(np.dot(np.array([x, y]) - a[:2], e) / np.dot(e, e)), 0.0), 1.0)
        return float(np.hypot(*(np.array([x, y]) - a[:2] - t * e)))

    def run(self, move, limit=60.0, **args):
        seq = self.h.start(move, **args)
        t0 = self.e.ctx.sim_time
        while True:
            self.step()
            st = self.h.status()
            if st.seq == seq and st.finished:
                return st
            assert self.e.ctx.sim_time - t0 < limit, f"{move} did not end"

    def events(self, name):
        return [
            ev
            for ev in self.e.ctx.events.all()
            if ev["source"].startswith("worker") and ev["event"] == name
        ]


def test_a_door_pulled_round_its_hinge_opens_and_the_body_steps_back_round_with_it():
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        st = door.run("pull", hand="right", angle=60.0)
        assert st.error == "", st.error
        assert 50.0 < door.angle() < 70.0, door.angle()
        pulled = door.events("pulled")[-1]
        assert 48.0 < pulled["angle"] < 70.0, pulled
        assert door.events("stepping"), "the handle's arc left the hand's reach"
        x, y, _ = door.body.root()
        assert math.hypot(x - 1.0, y - 0.0) > 0.9  # outside the leaf's swing
        assert door.deepest < 0.01, door.deepest  # the body did not walk into the leaf
    finally:
        engine.shutdown()


def test_a_door_pushed_open_is_walked_through_kept_open_and_shuts_behind():
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        st = door.run("push", hand="right", angle=80.0)
        assert st.error == "", st.error
        assert door.angle() > 65.0, door.angle()
        opened = door.angle()
        st = door.run("approach", target="gate/in")
        assert st.error == "", st.error
        x, _y, _ = door.body.root()
        assert x < 0.6, x  # through the opening
        assert abs(door.angle() - opened) < 10.0, (door.angle(), opened)  # kept where it was
        assert door.run("release", hand="right").error == ""
        for _ in range(int(4.0 / engine.ctx.dt)):
            door.step()
        assert door.angle() < 10.0, door.angle()  # the closer shut it
        assert door.deepest < 0.01, door.deepest
    finally:
        engine.shutdown()


def test_a_hand_holding_a_door_turns_about_its_grip_without_pulling_the_leaf():
    """A hold on a door holds the hand's grip on the handle, its position only: a hand turning
    30 deg about its grip, as an arm turns it when the body turns on the spot, leaves the leaf
    where it is and pulls on it hardly at all. Held at a point of the hand 8 cm from its grip,
    the turning hand moved that point, turned the leaf and pulled thousands of newtons along its
    arc."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        assert door.run("push", hand="right", angle=30.0).error == ""
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        hold = hp.holding["r"]
        p0, q0 = door.body.grip(door.d, "r")
        p0, q0 = p0.copy(), np.asarray(q0, float).copy()
        before, most = door.angle(), 0.0
        for k in range(400):
            turn = math.radians(30.0) * min(1.0, k / 200)
            door.body.set_hand("r", p0, Q.mul(Q.axis_angle("z", turn), q0))
            door.step()
            if k > 20:
                most = max(most, hold.force(door.d, hold.slide_axis(door.m, door.d)))
        assert np.linalg.norm(door.body.grip(door.d, "r")[0] - p0) < 0.002  # the grip stayed
        assert abs(door.angle() - before) < 0.2, (door.angle(), before)
        assert most < 500.0, most
    finally:
        engine.shutdown()


def test_a_door_kept_by_its_hold_keeps_the_shoulder_of_its_hand_within_reach_as_the_body_turns():
    """The way of a walk keeping a door keeps the shoulder of the hand on the handle within the
    arm's reach of it, at each heading: turned round on the spot, the shoulder would stand on the
    far side, out of reach, so the turn is refused before the body moves, naming the door, and
    the hold holds. Bounding the root alone let the body turn, and the hand slipped off. The door
    has no closer, so the hand keeps it by its hold alone (with one, it would push on the leaf)."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        engine.ctx.model.jnt_stiffness[engine.ctx.model.joint("gate_hinge").id] = 0.0
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        assert door.run("push", hand="right", angle=80.0).error == ""
        x, y, yaw = door.body.root()
        st = door.run(
            "approach",
            target={"position": {"x": x, "y": y, "z": 0.0}},
            standoff=0.0,
            facing=yaw + 3.0,
        )
        assert st.error.startswith("no way holding gate to "), st.error
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        assert hp.holding["r"] is not None  # still on the handle
        assert math.hypot(door.body.root()[0] - x, door.body.root()[1] - y) < 0.02
    finally:
        engine.shutdown()


def test_a_door_kept_open_is_not_walked_away_from_out_of_reach():
    """Keeping the door, the body's way stays within the hand's reach of it: a stance far beyond
    has no way, and the failure says so before the body sets off."""
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        assert door.run("pull", hand="right", angle=40.0).error == ""
        st = door.run(
            "approach", target={"position": {"x": -2.0, "y": 0.8, "z": 0.0}}, facing=math.pi
        )
        # naming what it holds: the door it keeps
        assert st.error.startswith("no way holding gate to "), st.error
    finally:
        engine.shutdown()


@pytest.mark.parametrize("move", ["pull", "push"])
def test_an_angle_for_what_turns_on_no_hinge_is_refused(move):
    tote = {"tote": {"pose": {"position": {"x": -0.6, "y": 0.8}}}, "name": "tote_1"}
    engine = _engine((0.0, 0.8), math.pi, extra=[tote], graspable=("gate", "tote_1"))
    try:
        door = Door(engine)
        assert door.run("approach", target="tote_1", standoff=0.45).error == ""
        assert door.run("grasp", hand="both", target="tote_1").error == ""
        st = door.run(move, hand="both", angle=30.0)
        assert "turns on no hinge" in st.error, st.error
        st = door.run(move, hand="both")
        assert "a distance (m) along a slide or an angle (deg) round a hinge" in st.error
    finally:
        engine.shutdown()


def test_a_door_that_will_not_turn_is_stuck():
    """A post in the leaf's swing: the pull fails rather than reporting the door open -- before it
    sets off where the body would step round with the leaf into the post, else stuck against it,
    or its hold let go."""
    post = {
        "box": {
            "prefix": "post_",
            "pose": {"position": {"x": 0.70, "y": 0.52}},
            "size": [0.06, 0.06, 1.8],
            "motion": "static",
        },
        "name": "post",
    }
    engine = _engine((0.0, 0.8), 0.0, extra=[post])
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        st = door.run("pull", hand="right", angle=55.0)
        assert st.error.startswith(
            ("cannot follow gate round: ", "stuck: gate turned", "lost hold")
        ), st.error
        if st.error.startswith("cannot follow"):
            assert "post_box" in st.error and door.angle() < 1.0, (st.error, door.angle())
        assert door.angle() < 35.0, door.angle()
    finally:
        engine.shutdown()


def test_a_door_turned_toward_its_shut_stop_fails_rather_than_reporting_it_turned():
    """Pulled by the handle on the face it is pushed from, a shut door would turn further shut: the
    pull fails saying so, before anything moves, rather than reporting a pull of about 0 deg."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        st = door.run("pull", hand="right", angle=30.0)
        assert "is shut" in st.error, st.error
        assert not door.events("moving")
        assert abs(door.angle()) < 1.0
    finally:
        engine.shutdown()


def test_a_body_that_leaves_its_way_round_a_hinge_fails_there():
    """Stepping round the hinge with the leaf, the body is watched against the path its step was
    checked on: a navigator that walks it 10 cm off every point it is sent (outward from the hinge)
    leaves that path, and the pull fails saying so rather than walking a path nobody checked."""
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        nav = engine.ctx.blackboard.get("nav:worker:handle")
        send = nav.send_goals

        def off(goals, exact=False):
            out = []
            for x, y in goals:
                r = math.hypot(x - 1.0, y) or 1.0
                out.append((x + 0.1 * (x - 1.0) / r, y + 0.1 * y / r))
            return send(out, exact=exact)

        nav.send_goals = off
        st = door.run("pull", hand="right", angle=85.0)
        assert st.error.startswith("left its way to round gate"), st.error
    finally:
        engine.shutdown()


def test_a_turn_on_a_hinge_that_is_not_upright_is_refused():
    """A flap on a level hinge turns up, not round: a pull by angle is refused before it moves,
    saying so, rather than stepping the body round an axis it cannot walk round."""
    engine = _engine((1.0, -0.6), math.pi / 2, graspable=("flap",), world="hinge_flap.xml")
    try:
        door = Door.__new__(Door)
        door.e, door.h = engine, engine.ctx.blackboard.get("humanoid:worker")
        door.body = engine.ctx.blackboard.get("humanoid_body:worker")
        door.m, door.d, door.deepest = engine.ctx.model, engine.ctx.data, 0.0
        door.leaf = door.m.body("flap").id
        assert door.run("grasp", hand="right", target="flap/grip").error == ""
        st = door.run("pull", hand="right", angle=30.0)
        assert "not upright" in st.error, st.error
        assert not door.events("moving")
    finally:
        engine.shutdown()


def test_a_turn_that_needs_the_body_to_step_round_is_refused_without_a_navigator():
    """Pulled far enough that the arc leaves the hand's reach, the body steps round with the leaf
    on its navigator: with none, the pull fails before the door moves, not part way round."""
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        # as a body without one: its navigator's handle gone
        engine.ctx.blackboard.set("nav:worker:handle", None)
        st = door.run("pull", hand="right", angle=85.0)
        assert "no navigator" in st.error, st.error
        assert not door.events("moving")
        assert abs(door.angle()) < 2.0
    finally:
        engine.shutdown()


def test_a_door_kept_open_holds_the_way_within_a_reach_that_takes_in_where_the_body_stands():
    """A pull steps the body round with the leaf to where its hand reaches the handle, which may be
    farther from the root than 0.9 of the arm's reach: the way of a walk keeping the door keeps the
    shoulder within the arm's reach, and at least within where it stands -- the walk is not refused
    where it starts."""
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        assert door.run("pull", hand="right", angle=40.0).error == ""
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        x, y, yaw = door.body.root()
        (tether,), ((ox, oy),) = hp._tethers()
        ax, ay, bx, by, r = tether[:5]
        # the point the tether holds: the shoulder of the hand, turned with the heading
        px = x + ox * math.cos(yaw) - oy * math.sin(yaw)
        py = y + ox * math.sin(yaw) + oy * math.cos(yaw)
        ex, ey = bx - ax, by - ay
        t = min(max(((px - ax) * ex + (py - ay) * ey) / max(ex * ex + ey * ey, 1e-12), 0.0), 1.0)
        assert math.hypot(px - ax - t * ex, py - ay - t * ey) < r, (px, py, tether)
        st = door.run(
            "approach", target={"position": {"x": x - 0.15, "y": y, "z": 0.0}}, facing=yaw
        )
        assert "already meets" not in st.error, st.error
    finally:
        engine.shutdown()


def test_a_door_turned_shut_by_more_than_it_is_open_shuts_at_its_stop():
    """Pushed open 60 deg and pulled back by 80, a door turns until it is shut, at its stop: that is
    as far as it goes, not a door stuck at 60 of 80 deg."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        assert door.run("push", hand="right", angle=60.0).error == ""
        st = door.run("pull", hand="right", angle=80.0)
        assert st.error == "", st.error
        assert abs(door.angle()) < 3.0, door.angle()
    finally:
        engine.shutdown()


def test_a_door_pulled_open_is_kept_open_by_the_hand_pushing_on_it_and_walked_through():
    """Pulled open toward the body, the handle ends at the leaf's far edge, out of reach of the
    opening. Walking through, the hand keeps the door by contact: its hold stays on while the body
    is behind the leaf; once the body has come round onto the face the closer pushes the leaf
    toward, the hold goes off and the fingertip goes round the leaf's edge onto that face, the
    closer pressing the leaf into it as the hand slides along. The door stays open while the body
    walks through; once the body is past the line the leaf shuts onto, the hand lets go behind it,
    and the closer shuts the door without the leaf sweeping into the body."""
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        assert door.run("pull", hand="right", angle=85.0).error == ""
        opened = door.angle()
        assert opened > 70.0, opened
        least, nearest = opened, math.inf
        seq = door.h.start("approach", target="gate/through")
        t0 = engine.ctx.sim_time
        while not (door.h.status().seq == seq and door.h.status().finished):
            door.step()
            if not door.events("let go"):
                least = min(least, door.angle())
            else:  # the leaf shutting behind the body keeps clear of it
                nearest = min(nearest, door.leaf_to_root())
            assert engine.ctx.sim_time - t0 < 60.0
        st = door.h.status()
        assert st.error == "", st.error
        keeping = [ev for ev in door.events("keeping") if ev["t"] >= t0]
        assert keeping and keeping[0]["how"] == "by contact", keeping
        assert [ev for ev in door.events("pushing") if ev["t"] >= t0]
        kept = [ev for ev in door.events("kept") if ev["t"] >= t0]
        assert kept and kept[0]["how"] == "contact" and kept[0]["force"] > 1.0, kept
        x, _y, _ = door.body.root()
        assert x > 1.3, x  # through the opening, on the far side
        assert opened - least < 15.0, (opened, least)  # kept open all the way through
        assert [ev for ev in door.events("let go") if ev["t"] >= t0], "the hand never let go"
        for _ in range(int(4.0 / engine.ctx.dt)):
            door.step()
            nearest = min(nearest, door.leaf_to_root())
        assert nearest > COLLISION_RADIUS, nearest
        assert door.angle() < 10.0, door.angle()  # the closer shut it behind
        assert door.deepest < 0.01, door.deepest
    finally:
        engine.shutdown()


def test_a_door_pushed_open_and_walked_past_its_free_end_is_let_go_and_shuts_clear_of_the_body():
    """Pushed open and walked through to a stance past the leaf's free end, out of its sweep: the
    hand keeps the door while the body passes, lets go once the body is out of the sweep, and the
    closer shuts the door behind it without the leaf touching the body."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        assert door.run("push", hand="right", angle=80.0).error == ""
        t0 = engine.ctx.sim_time
        st = door.run("approach", target="gate/beyond")
        assert st.error == "", st.error
        x, y, _ = door.body.root()
        assert math.hypot(x - 1.0, y) > 0.9 + COLLISION_RADIUS, (x, y)  # out of the sweep
        assert [ev for ev in door.events("let go") if ev["t"] >= t0], "the hand never let go"
        # from here on, the shutting leaf meets no part of the body (its fingertip pushed on the
        # leaf while it kept it)
        door.deepest, nearest = 0.0, math.inf
        for _ in range(int(4.0 / engine.ctx.dt)):
            door.step()
            nearest = min(nearest, door.leaf_to_root())
        assert door.angle() < 10.0, door.angle()  # the closer shut it behind
        assert nearest > COLLISION_RADIUS, nearest
        assert door.deepest == 0.0, door.deepest
    finally:
        engine.shutdown()


def test_a_door_kept_by_contact_is_let_go_only_once_the_way_left_keeps_out_of_its_sweep(
    monkeypatch,
):
    """Pulled open toward the body, the door is kept by contact on the way back through it: the
    hand is done with it only once neither the body nor any pose of the way still to walk, with
    what it holds, goes where the leaf sweeps as it shuts or through the opening. Standing at the
    pull stance, out of the sweep, with the way through still to walk, it keeps the door -- a way
    back that starts with a step back out of the sweep let it go there, still on the wrong side,
    and the door shut before the body had gone through; with no way left, it lets go."""
    from roqsim_motion import moves

    real, ways = moves.plan_way_steps, []

    def keep(*args, **kw):
        way = yield from real(*args, **kw)
        ways.append(way)
        return way

    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        assert door.run("pull", hand="right", angle=85.0).error == ""
        monkeypatch.setattr(moves, "plan_way_steps", keep)
        assert door.run("plan", to="gate/through").error == ""
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        hp._keep_start()
        hp._keep["r"] = "contact"
        hp._way_pieces = list(hp._footprint_now()[0].pieces)
        hp._way_rest = [tuple(p) for sg in ways[-1].segments for p in sg.poses]
        assert not hp._through("r")  # the way through is still to walk
        hp._way_rest = []
        assert hp._through("r")  # out of the sweep, nothing left to walk through it
    finally:
        engine.shutdown()


def _where(engine):
    """Where the walker stands and its heading."""
    hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
    return hp.body.root()


def _walk_through_pulled(door, engine):
    """Pull the door open and walk through it keeping it by contact; the walk's status."""
    assert door.run("approach", target="gate/pull/grip").error == ""
    assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
    assert door.run("pull", hand="right", angle=85.0).error == ""
    return door.run("approach", target="gate/through")


def test_a_reset_in_the_middle_of_keeping_a_door_leaves_nothing_of_it():
    """Reset while the hand keeps a door by contact, nothing of that keeping is left: the next run
    pulls the door open and walks through it, keeping it from the start as the first did."""
    engine = _engine((0.0, 0.8), 0.0)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/pull/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/pull/grip").error == ""
        assert door.run("pull", hand="right", angle=85.0).error == ""
        door.h.start("approach", target="gate/through")
        t0 = engine.ctx.sim_time
        while not door.events("pushing"):
            door.step()
            assert engine.ctx.sim_time - t0 < 30.0, "it never pushed on the door"
        engine.reset()
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        for state in (
            "_keep",
            "_kept_at",
            "_push",
            "_tip_goal",
            "_keep_z",
            "_keep_at_face",
        ):
            assert getattr(hp, state) == {}, (state, getattr(hp, state))
        assert hp._way_rest == [] and hp._way_pieces == []
        door = Door(engine)
        st = _walk_through_pulled(door, engine)
        assert st.error == "", st.error
    finally:
        engine.shutdown()


def test_a_kept_door_turned_again_is_kept_at_its_new_angle():
    """A door a hand keeps, turned again by that hand, is kept at the angle it is turned to: the
    next walk counts how far it closes from there, not from where the first walk found it."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        assert door.run("push", hand="right", angle=60.0).error == ""
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        # as a walk keeping it by its hold leaves it
        hp._keep["r"], hp._kept_at["r"] = "weld", math.radians(door.angle())
        assert door.run("push", hand="right", angle=15.0).error == ""
        assert math.degrees(hp._kept_at["r"]) == pytest.approx(door.angle(), abs=0.5)
    finally:
        engine.shutdown()


def test_a_door_kept_by_pushing_on_its_face_is_not_turned_by_its_handle():
    """A hand that keeps a door by pushing on its face holds no handle: a pull or a push by it fails
    before anything moves, saying to grasp the handle again."""
    engine = _engine((2.2, 0.8), math.pi)
    try:
        door = Door(engine)
        assert door.run("approach", target="gate/push/grip").error == ""
        assert door.run("grasp", hand="right", target="gate/push/grip").error == ""
        assert door.run("push", hand="right", angle=60.0).error == ""
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        hp._keep["r"] = "contact"  # as a walk keeping it by contact leaves it
        st = door.run("push", hand="right", angle=15.0)
        assert "by pushing on its face" in st.error, st.error
    finally:
        engine.shutdown()
