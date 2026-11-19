"""A carry walks the way planned for the body and what it holds: what was checked is what is
walked, and what it carries touches nothing on the way.

The review cases, each a test world in ``data/`` (``roqsim_nav``'s planner tests have their
twins): a post beside a straight stretch; a dead end the body steps back out of, with a post where
a turn after a short step back would sweep the tote; three close bends of 30 degrees; and a tote
too wide for its passage that goes over a bench beside it, and not under a shelf at its own height.
A walker with hands takes a tote, carries it through, and every contact of the tote with anything
but the walker is counted from the moment it is lifted.
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


def _engine(world_xml, start, tote_at, yaw=0.0, size=None):
    tote = {
        "pose": {
            "position": {"x": tote_at[0], "y": tote_at[1]},
            "orientation": {"yaw": yaw + math.pi / 2},
        }
    }
    if size:
        tote["size"] = size
    world = {
        "sim": {"pacing": "asap", "world": str(DATA / f"{world_xml}.xml")},
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
                "components": [{"navigator": NAV}, {"humanoid": {"graspable": ["tote_1"]}}],
            },
            {"tote": tote, "name": "tote_1"},
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


class Carry:
    """Take the tote and carry it to ``to``, recording its every contact once it is lifted."""

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

    def carry(self, to, facing, limit=120.0, **kw):
        st = self.run("approach", target="tote_1", standoff=0.45)
        assert st.error == "", st.error
        st = self.run("grasp", hand="both", target="tote_1")
        assert st.error == "", st.error
        m, d = self.m, self.d
        root = int(m.body_weldid[m.body("tote_1").id])
        t0 = self.e.ctx.sim_time
        seq = self.h.carry(
            {"position": {"x": to[0], "y": to[1], "z": 0.0}}, standoff=0.6, facing=facing, **kw
        )
        touches = []
        while True:
            self.e.step()
            if self.events("lifted", t0):
                for c in d.contact[: d.ncon]:
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    r1, r2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
                    if root in (r1, r2) and c.dist < -1e-4:
                        other = b2 if r1 == root else b1
                        if not self.body.owns(m, other):
                            touches.append((round(self.e.ctx.sim_time, 2), m.body(other).name))
            st = self.h.status()
            if st.seq == seq and st.finished:
                return st, touches, t0
            assert self.e.ctx.sim_time - t0 < limit, "the carry did not end"

    def events(self, name, since):
        return [
            ev
            for ev in self.e.ctx.events.all()
            if ev["source"].startswith("worker") and ev["event"] == name and ev["t"] >= since
        ]


CASES = {
    "carry_post_straight": dict(start=(-0.5, 0.0), tote_at=(0.05, 0.0), to=(4.6, 0.0), facing=0.0),
    "carry_post_behind": dict(
        start=(0.0, 0.0), tote_at=(0.55, 0.0), to=(-2.0, 1.2), facing=math.pi
    ),
    "carry_three_bends": dict(start=(-1.2, 0.0), tote_at=(-0.65, 0.0), to=(6.1, 0.92), facing=0.0),
    "carry_shelf_over_bench": dict(
        start=(-0.6, -0.09),
        tote_at=(-0.05, -0.09),
        to=(4.5, -0.09),
        facing=0.0,
        size=[0.6, 0.4, 0.2],
        height=1.0,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_a_carry_through_a_review_case_touches_nothing(name):
    case = dict(CASES[name])
    size, height = case.pop("size", None), case.pop("height", None)
    engine = _engine(name, case["start"], case["tote_at"], size=size)
    try:
        sim = Carry(engine)
        kw = {"height": height} if height else {}
        st, touches, t0 = sim.carry(case["to"], case["facing"], **kw)
        assert st.error == "", st.error
        assert touches == [], touches[:5]
        planned = sim.events("planned", t0)
        assert len(planned) == 1 and planned[0]["seconds"] > 0.0, planned
        x, y, yaw = sim.body.root()
        stand = (
            case["to"][0] - 0.6 * math.cos(case["facing"]),
            case["to"][1] - 0.6 * math.sin(case["facing"]),
        )
        assert math.hypot(x - stand[0], y - stand[1]) < 0.06, (x, y, stand)
        if name == "carry_post_behind":
            backs = sim.events("stepping_back", t0)
            assert backs and all(0.0 < b["distance"] <= 0.5 + 1e-9 for b in backs), backs
    finally:
        engine.shutdown()


def test_a_carry_with_no_way_fails_naming_what_is_in_it_and_holds_on():
    """A shelf board at the tote's height over the bench: no way, a failure naming the shelf, and
    the tote still in the hands where the walker stands."""
    engine = _engine(
        "carry_shelf_over_bench_shelf", (-0.6, -0.09), (-0.05, -0.09), size=[0.6, 0.4, 0.2]
    )
    try:
        sim = Carry(engine)
        st, _touches, t0 = sim.carry((4.5, -0.09), 0.0, height=1.0)
        assert st.error.startswith("no way holding tote_1 to "), st.error
        assert "shelf" in st.error and "back_max 0.50 m, long_way false" in st.error, st.error
        assert sim.d.body("tote_1").xpos[2] > 0.5  # still held
        assert not sim.events("walking", t0)  # it did not set off on a way it has not got
    finally:
        engine.shutdown()


def test_a_body_that_leaves_its_way_fails_saying_by_how_much(monkeypatch):
    """What is walked is what was checked: a body pushed off its way (here, the tolerance shrunk to
    nothing, as a body that strays would meet it) stops with a failure saying how far it strayed."""
    import roqsim_motion.moves as moves

    monkeypatch.setattr(moves, "FOLLOW_TOL", -1.0)
    case = CASES["carry_three_bends"]
    engine = _engine("carry_three_bends", case["start"], case["tote_at"])
    try:
        sim = Carry(engine)
        st, _touches, _t0 = sim.carry(case["to"], case["facing"])
        assert st.error.startswith("left its way to where it is sent by "), st.error
    finally:
        engine.shutdown()


def test_a_dead_end_without_a_step_back_has_no_way():
    case = CASES["carry_post_behind"]
    engine = _engine("carry_post_behind", case["start"], case["tote_at"])
    try:
        sim = Carry(engine)
        st, _touches, _t0 = sim.carry(case["to"], case["facing"], back_max=0.0)
        assert st.error.startswith("no way holding tote_1 to "), st.error
        assert "back_max 0.00 m" in st.error, st.error
        assert any(n in st.error for n in ("side_n", "side_s", "end_wall")), st.error
    finally:
        engine.shutdown()


def test_without_long_way_no_turn_on_the_spot_passes_half_a_turn():
    """Boxed in, with a post on its left, to face three quarters of a turn to the left: without
    ``long_way`` its way has no turn past half a turn (it shuffles round in a few turns where there
    is room, or fails); ``roqsim_nav``'s planner tests hold the case with no room at all."""
    engine = _engine("carry_long_way", (-0.05, 0.0), (0.5, 0.0))
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        x, y, _ = sim.body.root()
        facing = 0.75 * math.pi
        to = {
            "position": {"x": x + 0.6 * math.cos(facing), "y": y + 0.6 * math.sin(facing), "z": 0}
        }
        t0 = sim.e.ctx.sim_time
        st = sim.run("carry", to=to, standoff=0.6, facing=facing)
        if st.error:
            assert "long_way false" in st.error, st.error
        else:
            planned = sim.events("planned", t0)
            assert planned and planned[0]["long_turns"] == 0, planned
            assert not sim.events("turning_long_way", t0)
    finally:
        engine.shutdown()


def test_an_approach_holding_something_walks_a_way_planned_as_a_carry_does():
    case = CASES["carry_three_bends"]
    engine = _engine("carry_three_bends", case["start"], case["tote_at"])
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        assert (
            sim.run(
                "carry", to={"position": {"x": 0.6, "y": 0.0, "z": 0.0}}, standoff=0.6, facing=0.0
            ).error
            == ""
        )
        t0 = sim.e.ctx.sim_time
        to = {"position": {"x": 6.1, "y": 0.92, "z": 0.0}}
        st = sim.run("approach", target=to, standoff=0.6, facing=0.0, limit=120.0)
        assert st.error == "", st.error
        assert len(sim.events("planned", t0)) == 1
        assert sim.d.body("tote_1").xpos[2] > 0.5  # still held
    finally:
        engine.shutdown()
