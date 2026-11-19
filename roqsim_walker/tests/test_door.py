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
from roqsim_walker.humanoid import COLLISION_RADIUS

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
        assert st.error.startswith("no way "), st.error
        assert door.d.body("gate") is not None
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
