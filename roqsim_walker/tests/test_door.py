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

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

DATA = Path(__file__).parent / "data"
NAV = {
    "output": "walker",
    "speed": 1.0,
    "arrival_radius": 0.05,
    "avoidance": {"steer": "none", "stop": False},
}


def _engine(start, yaw, extra=(), graspable=("gate",)):
    world = {
        "sim": {"pacing": "asap", "world": str(DATA / "hinge_door.xml")},
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
