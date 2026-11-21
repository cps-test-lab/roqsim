"""A carry walks the way planned for the body and what it holds: what was checked is what is
walked, and what it carries touches nothing on the way.

The cases a disc's way gets wrong, each a test world in ``data/`` (``roqsim_nav``'s planner tests
have their twins): a post beside a straight stretch; a dead end the body steps back out of, with a
post where a turn after a short step back would sweep the tote; three close bends of 30 degrees; and
a tote too wide for its passage that goes over a bench beside it, and not under a shelf at its own
height. A walker with hands takes a tote, carries it through, and every contact of the tote with
anything but the walker is counted from the moment it is lifted.
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


def _engine(world_xml, start, tote_at, yaw=0.0, size=None, body_lag=0.0, nav=NAV, extra=()):
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
                    "body_lag": body_lag,
                    "pose": {
                        "position": {"x": start[0], "y": start[1]},
                        "orientation": {"yaw": yaw},
                    },
                },
                "name": "worker",
                "components": [{"navigator": nav}, {"humanoid": {"graspable": ["tote_1"]}}],
            },
            {"tote": tote, "name": "tote_1"},
            *extra,
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
def test_a_carry_through_a_case_touches_nothing(name):
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


def _case(name, **kw):
    case = dict(CASES[name])
    size, height = case.pop("size", None), case.pop("height", None)
    engine = _engine(name, case["start"], case["tote_at"], size=size, **kw)
    return engine, case, ({"height": height} if height else {})


@pytest.mark.parametrize("name", list(CASES))
def test_a_body_drawn_behind_its_root_plans_with_the_margin_it_strays_by_and_keeps_to_it(name):
    """A walker with a ``body_lag`` of 0.15 s (a person's pace) is drawn on a point that trails its
    nav root and cuts inside an arc: it plans with that margin -- ``(speed * lag)**2 / R`` at the
    tightest arc a way has -- says so in ``planned``, and walks every one of these cases without
    leaving its way or the tote touching anything."""
    engine, case, kw = _case(name, body_lag=0.15)
    try:
        sim = Carry(engine)
        st, touches, t0 = sim.carry(case["to"], case["facing"], **kw)
        assert st.error == "", st.error
        assert touches == [], touches[:5]
        margin = sim.events("planned", t0)[0]["margin"]
        assert margin == pytest.approx(sim.body.follow_error(NAV["speed"]), abs=1e-3)
        assert 0.015 < margin < 0.025, margin
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("name", "way"),
    [
        ("carry_post_straight", True),
        ("carry_post_behind", True),
        ("carry_three_bends", False),
        ("carry_shelf_over_bench", False),
    ],
)
def test_a_body_drawn_further_behind_keeps_further_off_and_says_where_that_leaves_no_way(name, way):
    """With a ``body_lag`` of 0.3 s the margin is four times as wide (7.9 cm at 1 m/s): the
    straight stretch and the dead end are walked within it, touching nothing; the close bends and
    the passage beside the bench leave no way that keeps it, and the carry fails before it sets
    off, saying how far clear it planned."""
    engine, case, kw = _case(name, body_lag=0.3)
    try:
        sim = Carry(engine)
        st, touches, t0 = sim.carry(case["to"], case["facing"], **kw)
        if way:
            assert st.error == "", st.error
            assert touches == [], touches[:5]
            assert sim.events("planned", t0)[0]["margin"] == pytest.approx(0.0785, abs=2e-3)
        else:
            assert st.error.startswith("no way holding tote_1 "), st.error
            assert "planned 0.079 m clear of everything" in st.error, st.error
            assert not sim.events("walking", t0)
    finally:
        engine.shutdown()


def test_a_step_back_is_walked_on_the_line_it_was_checked_on():
    """In the dead end the way out steps straight back: the body keeps to that line, keeping its
    heading, and the tote touches nothing."""
    engine, case, kw = _case("carry_post_behind")
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        t0 = sim.e.ctx.sim_time
        seq = sim.h.carry(
            {"position": {"x": case["to"][0], "y": case["to"][1], "z": 0.0}},
            standoff=0.6,
            facing=case["facing"],
        )
        start, worst, turned = None, 0.0, 0.0
        while True:
            sim.e.step()
            if sim.body.st.keep is not None:  # stepping back, its heading kept
                if start is None:
                    start = sim.body.root()
                x, y, h = sim.body.root()
                ux, uy = math.cos(start[2]), math.sin(start[2])
                worst = max(worst, abs((x - start[0]) * uy - (y - start[1]) * ux))
                turned = max(
                    turned, abs(math.atan2(math.sin(h - start[2]), math.cos(h - start[2])))
                )
            st = sim.h.status()
            if st.seq == seq and st.finished:
                break
            assert sim.e.ctx.sim_time - t0 < 120.0
        assert st.error == "", st.error
        assert start is not None, "the way out of the dead end steps back"
        assert worst < 0.01 and turned < math.radians(2), (worst, math.degrees(turned))
    finally:
        engine.shutdown()


def test_a_body_that_strays_off_a_step_back_fails_saying_by_how_much():
    """A step back is a leg of the way as a walk is: where the navigator takes the body off the
    straight step (here every goal sent while the body keeps its heading comes out 6 cm to the
    side, as a navigator that steers or plans round something would), the carry fails there, on the
    step, saying how far it strayed, rather than going on from where nobody checked."""
    engine, case, kw = _case("carry_post_behind")
    try:
        sim = Carry(engine)
        nav = engine.ctx.blackboard.get("nav:worker:handle")
        send = nav.send_goals

        def aside(goals, **kw):
            if sim.body.st.keep is not None:  # stepping back, its heading kept
                x, y, h = sim.body.root()
                ox, oy = -math.sin(h) * 0.06, math.cos(h) * 0.06
                goals = [(g[0] + ox, g[1] + oy) for g in goals]
            return send(goals, **kw)

        nav.send_goals = aside
        st, _touches, t0 = sim.carry(case["to"], case["facing"])
        back = sim.events("stepping_back", t0)
        assert back
        assert st.error.startswith("left its way to where it is sent by "), st.error
        # on the step back, not later on a leg that set off from where the stray step left it
        assert not [ev for ev in sim.events("walking", t0) if ev["t"] > back[0]["t"]]
    finally:
        engine.shutdown()


def test_a_carry_plans_its_margin_for_the_speed_it_walks_at_under_avoidance():
    """The margin a carry plans with is the body's stray at the speed it walks its way at. Under
    avoidance the model may command a body up to ``max_speed``; a carry's legs are sent ``exact``,
    the avoidance model's velocity is not taken, and the body walks at the navigator's ``speed``:
    measured, it never goes faster, and the margin is the one for that speed."""
    nav = {**NAV, "speed": 0.6, "max_speed": 2.0}
    engine, case, kw = _case("carry_post_straight", body_lag=0.15, nav=nav)
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        t0 = engine.ctx.sim_time
        to = case["to"]
        seq = sim.h.carry(
            {"position": {"x": to[0], "y": to[1], "z": 0.0}}, standoff=0.6, facing=case["facing"]
        )
        # over 0.9 s at a time: the body is drawn at its own rate (every 18 ms), not every step
        window = int(round(0.9 / engine.ctx.dt))
        fastest, last, k = 0.0, sim.body.root(), 0
        while not (sim.h.status().seq == seq and sim.h.status().finished):
            engine.step()
            k += 1
            if k % window == 0:
                now = sim.body.root()
                fastest = max(fastest, math.hypot(now[0] - last[0], now[1] - last[1]) / 0.9)
                last = now
            assert engine.ctx.sim_time - t0 < 120.0
        assert sim.h.status().error == "", sim.h.status().error
        assert fastest <= 0.6 * 1.02, fastest
        margin = sim.events("planned", t0)[0]["margin"]
        assert margin == pytest.approx(sim.body.follow_error(0.6), abs=1e-3)
    finally:
        engine.shutdown()


def test_a_leg_that_ends_short_fails_rather_than_walking_on(monkeypatch):
    """A navigator that stops 20 cm short of where it is sent (as one with a wide arrival radius
    does) leaves the body short of a leg's end after every aim on: the carry fails saying so,
    rather than walking on from where its way was not checked."""
    from roqsim_nav import behavior

    monkeypatch.setattr(behavior, "EXACT_ARRIVE", 0.2)
    engine, case, kw = _case("carry_post_straight")
    try:
        st, _touches, _t0 = Carry(engine).carry(case["to"], case["facing"], **kw)
        assert "short of a leg's end" in st.error, st.error
    finally:
        engine.shutdown()


def test_a_step_back_that_ends_short_fails_rather_than_walking_on(monkeypatch):
    """A navigator that stops a metre short of where it is sent hardly moves a body stepping back
    0.4 m, however far on the step is aimed: after its tries the step back fails saying how far it
    got, rather than letting the way go on from where it was not checked."""
    from roqsim.machine import ProgramFailed
    from roqsim_nav import behavior

    monkeypatch.setattr(behavior, "EXACT_ARRIVE", 1.0)
    engine, _case_, _kw = _case("carry_post_straight")
    try:
        hp = next(p for p in engine.plugins if type(p).__name__ == "HumanoidPlugin")
        with pytest.raises(ProgramFailed, match="stepped back 0.* m of 0.400"):
            for _ in hp._step_back(0.4, lambda: None, "the test's goal", 0.05):
                engine.step()
    finally:
        engine.shutdown()


def test_a_turn_ends_at_the_heading_planned_so_a_step_back_after_it_keeps_to_the_way(monkeypatch):
    """A turn on the spot ends at the heading the way planned, not at its angle from where the body
    faces: a body that stands off the planned heading when it turns (here 0.25 rad, as a short leg
    ends facing where it walked) turns that away too, and the step back after the turn keeps to the
    way, so the leg after it sets off on its line. Turned by the angle alone, the error stayed, the
    step back went off along it, and the next leg failed leaving its way."""
    from roqsim_motion import moves
    from roqsim_nav.lattice import Segment, Way

    engine, _case_, _kw = _case("carry_post_straight")
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        x0, y0, h0 = sim.body.root()
        h = h0 - 0.25  # the heading the way thinks the body has
        hq = h - math.pi / 2  # turned right, a quarter
        bx, by = x0 - 0.3 * math.cos(hq), y0 - 0.3 * math.sin(hq)
        way = Way(
            segments=[
                Segment("turn", [(x0, y0, h), (x0, y0, hq)], angle=-math.pi / 2),
                Segment("back", [(x0, y0, hq), (bx, by, hq)], distance=0.3),
                Segment("walk", [(bx, by, hq), (x0, y0, hq)]),
            ],
            seconds=0.0,
            expanded=0,
            margin=0.0,
            slack=0.0,
        )

        def planned(*_args, **_kw):
            return way
            yield

        monkeypatch.setattr(moves, "plan_way_steps", planned)
        t0 = engine.ctx.sim_time
        seq = sim.h.carry(
            {"position": {"x": x0 + 0.6 * math.cos(hq), "y": y0 + 0.6 * math.sin(hq), "z": 0.0}},
            standoff=0.6,
            facing=hq,
        )
        while not (sim.h.status().seq == seq and sim.h.status().finished):
            engine.step()
            assert engine.ctx.sim_time - t0 < 60.0
        assert sim.h.status().error == "", sim.h.status().error
        back = sim.events("stepping_back", t0)
        assert back
        x, y, _ = sim.body.root()
        assert math.hypot(x - x0, y - y0) < 0.03, (x - x0, y - y0)
    finally:
        engine.shutdown()


def test_a_carry_plans_its_way_while_the_simulation_steps_on():
    """A carry's way is planned in steps, the simulation stepping on between them: no tick waits for
    the whole search."""
    import time

    engine, case, kw = _case("carry_post_behind")
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        t0, to = engine.ctx.sim_time, case["to"]
        seq = sim.h.carry(
            {"position": {"x": to[0], "y": to[1], "z": 0.0}}, standoff=0.6, facing=case["facing"]
        )
        longest = 0.0
        while not (sim.h.status().seq == seq and sim.h.status().finished):
            a = time.perf_counter()
            engine.step()
            longest = max(longest, time.perf_counter() - a)
            assert engine.ctx.sim_time - t0 < 120.0
        assert sim.h.status().error == "", sim.h.status().error
        planned = sim.events("planned", t0)[0]
        assert longest < 0.5 * planned["seconds"], (longest, planned["seconds"])
    finally:
        engine.shutdown()


def test_a_carry_cancelled_while_it_plans_leaves_the_heading_the_bodys_own():
    """While a carry plans its way the body keeps its heading; a cancel then leaves it unkept, so
    a walk after it faces where it goes."""
    engine, case, _kw = _case("carry_post_behind")
    try:
        sim = Carry(engine)
        assert sim.run("approach", target="tote_1", standoff=0.45).error == ""
        assert sim.run("grasp", hand="both", target="tote_1").error == ""
        t0, to = engine.ctx.sim_time, case["to"]
        seq = sim.h.carry(
            {"position": {"x": to[0], "y": to[1], "z": 0.0}}, standoff=0.6, facing=case["facing"]
        )
        while not sim.events("lifted", t0):
            engine.step()
            assert engine.ctx.sim_time - t0 < 30.0
        engine.step()
        assert not sim.events("planned", t0) and sim.body.st.keep is not None
        sim.h.cancel(seq)
        while not (sim.h.status().seq == seq and sim.h.status().finished):
            engine.step()
        assert sim.h.status().error == "cancelled", sim.h.status().error
        assert sim.body.st.keep is None
    finally:
        engine.shutdown()
