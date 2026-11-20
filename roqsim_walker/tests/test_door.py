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
        for state in ("_keep", "_kept_at", "_push", "_tip_goal", "_keep_z", "_keep_at_face"):
            assert getattr(hp, state) == {}, (state, getattr(hp, state))
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


def _door_in_a_wall(hinge: str, toward: bool = False, extra=(), graspable=("door",)):
    """The `door` prop, moved by hand with a closer, in a 0.06 m wall along y = 0 whose ends meet the
    opening's (x -0.45 .. 0.45), opening away from the walker before it at y < 0 (toward +y), or
    ``toward`` it."""
    swing = 1 if hinge == "left" else -1
    wall = {"size": [2.0, 0.06, 2.4], "motion": "static"}
    world = {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "box": {**wall, "prefix": "wall_w_", "pose": {"position": {"x": -1.45, "y": 0.0}}},
                "name": "wall_w",
            },
            {
                "box": {**wall, "prefix": "wall_e_", "pose": {"position": {"x": 1.45, "y": 0.0}}},
                "name": "wall_e",
            },
            {
                "door": {
                    "prefix": "door_",
                    "pose": {"position": {"x": 0.0, "y": 0.0}},
                    "width": 0.9,
                    "hinge_side": hinge,
                    "swing": -swing if toward else swing,
                    "actuation": "hand",
                    "closer": {"stiffness": 15.0, "damping": 6.0},
                },
                "name": "door",
            },
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "hands": True,
                    "skin": False,
                    "pose": {
                        "position": {"x": 0.0, "y": -2.0},
                        "orientation": {"yaw": math.pi / 2},
                    },
                },
                "name": "worker",
                "components": [{"navigator": NAV}, {"humanoid": {"graspable": list(graspable)}}],
            },
            *extra,
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize("hinge", ["left", "right"])
def test_a_door_in_a_wall_is_pushed_open_from_its_own_stance_and_the_body_passes(hinge):
    """Pushed from the stance the door states, by the hand on the handle's side, the body steps round
    with the leaf through the opening and meets neither wall: the stance is near enough the hinge
    that the body's arc about it passes between the opening's ends."""
    engine = _door_in_a_wall(hinge)
    try:
        h = engine.ctx.blackboard.get("humanoid:worker")
        m, d = engine.ctx.model, engine.ctx.data
        qadr = int(m.jnt_qposadr[m.joint("door_hinge").id])

        def run(move, **args):
            seq = h.start(move, **args)
            t0 = engine.ctx.sim_time
            while not (h.status().seq == seq and h.status().finished):
                engine.step()
                assert engine.ctx.sim_time - t0 < 60.0, f"{move} did not end"
            return h.status().error

        # the handle is on the body's left with the hinge on its left: the latch then is to the
        # right of the hinge as seen from the side it is pushed from
        hand = "right" if hinge == "left" else "left"
        assert run("approach", target="door/push/grip") == ""
        assert run("grasp", hand=hand, target="door/push/grip") == ""
        assert run("push", hand=hand, angle=90.0) == ""
        assert abs(math.degrees(float(d.qpos[qadr]))) > 80.0
        root = _root(engine)
        assert root[1] > 0.0, root  # through the opening
    finally:
        engine.shutdown()


def _root(engine):
    hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
    return hp.body.root()


def test_a_load_in_one_hand_is_carried_at_the_side_to_a_door_the_other_hand_opens():
    """A tote taken by one end in one hand is carried at the body's side, the load ahead of the hand,
    so at the stance before a door's handle it is clear of the wall and the other hand takes the
    handle. Held in front, as two hands hold a load, it would meet the wall there."""
    tote = {"tote": {"pose": {"position": {"x": 1.0, "y": -1.2}}}, "name": "tote_1"}
    engine = _door_in_a_wall("left", toward=True, extra=[tote], graspable=("door", "tote_1"))
    try:
        h = engine.ctx.blackboard.get("humanoid:worker")
        d = engine.ctx.data
        body = engine.ctx.model.body("tote_1").id

        def run(move, **args):
            seq = h.start(move, **args)
            t0 = engine.ctx.sim_time
            tilt = 0.0
            while not (h.status().seq == seq and h.status().finished):
                engine.step()
                up = float(d.xmat[body].reshape(3, 3)[2, 2])
                tilt = max(tilt, math.degrees(math.acos(min(1.0, up))))
                assert engine.ctx.sim_time - t0 < 60.0, f"{move} did not end"
            return h.status().error, tilt

        assert run("approach", target="tote_1/grip_r")[0] == ""
        assert run("grasp", hand="left", target="tote_1/grip_r")[0] == ""
        error, tilt = run("carry", to="door/pull/grip", height=0.85)
        assert error == "", error
        assert tilt < 5.0, tilt  # level all the way
        x, y, yaw = _root(engine)
        grip = d.site("worker/grip_l").xpos
        beside = -(grip[0] - x) * math.sin(yaw) + (grip[1] - y) * math.cos(yaw)
        assert beside > 0.1, beside  # at its left side
        # the handle is on the body's right, the other hand free for it
        assert run("grasp", hand="right", target="door/pull/grip")[0] == ""
    finally:
        engine.shutdown()
