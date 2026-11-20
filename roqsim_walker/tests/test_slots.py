"""Slots: where a target says to stand, and where the hand goes, read by the moves.

A slot is sites (``docs/plugins.rst``, "Slots"): a hand target named for the slot and its stances,
``<slot>/stand`` or ``<slot>/stand_<k>``, on the floor, +x the way to face. ``approach`` and
``carry`` to a target that states stances walk to the nearest, unless they are given a
``standoff`` or a ``facing``, and ``arrived`` says which they used. The rack in
``data/slot_rack.xml`` states one slot with two stances.
"""

from __future__ import annotations

import math
from pathlib import Path

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

DATA = Path(__file__).parent / "data"
NAV = {
    "output": "walker",
    "speed": 1.0,
    "arrival_radius": 0.05,
    "avoidance": {"steer": "none", "stop": False},
}


def _engine(start=(0.0, 0.0)):
    world = {
        "sim": {"pacing": "asap", "world": str(DATA / "slot_rack.xml")},
        "components": [
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "hands": True,
                    "skin": False,
                    "pose": {"position": {"x": start[0], "y": start[1]}},
                },
                "name": "worker",
                "components": [{"navigator": NAV}, {"humanoid": {"graspable": ["tote_1"]}}],
            },
            {"tote": {"pose": {"position": {"x": -0.6, "y": 0.0}}}, "name": "tote_1"},
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _run(engine, move, limit=60.0, **args):
    h = engine.ctx.blackboard.get("humanoid:worker")
    seq = h.start(move, **args)
    t0 = engine.ctx.sim_time
    while True:
        engine.step()
        st = h.status()
        if st.seq == seq and st.finished:
            return st
        assert engine.ctx.sim_time - t0 < limit, f"{move} did not end"


def _arrived(engine):
    return [
        ev
        for ev in engine.ctx.events.all()
        if ev["source"].startswith("worker") and ev["event"] == "arrived"
    ][-1]


def _site(engine, name):
    d = engine.ctx.data
    s = d.site(name)
    x = s.xmat.reshape(3, 3)[:, 0]
    return s.xpos[:2].copy(), math.atan2(x[1], x[0])


def test_an_approach_to_a_slot_stands_at_its_nearest_stance_facing_as_it_says():
    engine = _engine(start=(0.0, 0.0))
    try:
        st = _run(engine, "approach", target="rack/bin_0")
        assert st.error == "", st.error
        body = engine.ctx.blackboard.get("humanoid_body:worker")
        x, y, yaw = body.root()
        (sx, sy), syaw = _site(engine, "rack/bin_0/stand_0")
        assert math.hypot(x - sx, y - sy) < 0.06, (x, y, sx, sy)
        assert abs(math.atan2(math.sin(yaw - syaw), math.cos(yaw - syaw))) < math.radians(4)
        assert _arrived(engine)["stance"] == "rack/bin_0/stand_0"
    finally:
        engine.shutdown()


def test_of_several_stances_the_nearest_is_taken():
    engine = _engine(start=(2.0, 2.2))
    try:
        assert _run(engine, "approach", target="rack/bin_0").error == ""
        assert _arrived(engine)["stance"] == "rack/bin_0/stand_1"
        body = engine.ctx.blackboard.get("humanoid_body:worker")
        (sx, sy), _ = _site(engine, "rack/bin_0/stand_1")
        x, y, _ = body.root()
        assert math.hypot(x - sx, y - sy) < 0.06
    finally:
        engine.shutdown()


def test_a_standoff_or_a_facing_given_overrides_the_slot():
    engine = _engine(start=(0.0, 0.0))
    try:
        assert _run(engine, "approach", target="rack/bin_0", standoff=0.9).error == ""
        assert _arrived(engine)["stance"] == "0.90 m from its front"
        assert _run(engine, "approach", target="rack/bin_0", facing=0.0).error == ""
        assert _arrived(engine)["stance"] == "0.60 m from its front"
    finally:
        engine.shutdown()


def test_a_carry_to_a_slot_stands_at_its_stance_and_places_on_its_place():
    engine = _engine(start=(-1.3, 0.0))
    try:
        assert _run(engine, "approach", target="tote_1", standoff=0.45).error == ""
        assert _run(engine, "grasp", hand="both", target="tote_1").error == ""
        st = _run(engine, "carry", to="rack/bin_0", limit=90.0)
        assert st.error == "", st.error
        assert _arrived(engine)["stance"] == "rack/bin_0/stand_0"
        st = _run(engine, "place", at="rack/bin_0")
        assert st.error == "", st.error
        for _ in range(500):
            engine.step()
        (px, py), _ = _site(engine, "rack/bin_0")
        tote = engine.ctx.data.body("tote_1").xpos
        assert math.hypot(tote[0] - px, tote[1] - py) < 0.05, (tote, px, py)
    finally:
        engine.shutdown()


def test_without_a_slot_an_approach_stands_off_as_it_did():
    engine = _engine(start=(0.0, 0.0))
    try:
        assert _run(engine, "approach", target="tote_1").error == ""
        assert _arrived(engine)["stance"] == "0.60 m from its front"
    finally:
        engine.shutdown()


def test_an_approach_to_a_grip_a_slot_does_not_have_is_refused_before_it_walks():
    """``rack/bin_0`` is a place slot, with stances and no grip: an approach to ``rack/bin_0/grip``
    is refused at once, naming the slot, rather than walking to its stance for a grasp to fail."""
    engine = _engine()
    try:
        t0 = engine.ctx.sim_time
        st = _run(engine, "approach", target="rack/bin_0/grip")
        assert "states no grip" in st.error, st.error
        moved = [
            ev
            for ev in engine.ctx.events.all()
            if ev["source"].startswith("worker") and ev["event"] == "walking" and ev["t"] >= t0
        ]
        assert moved == [], moved
    finally:
        engine.shutdown()


def test_a_carry_to_a_slot_that_is_where_to_stand_alone_walks_to_its_stance():
    """``rack/side`` states a stance and no hand target: a carry to it (at a height it is given) walks
    there, the tote held, rather than failing for want of a target to stand against."""
    engine = _engine()
    try:
        assert _run(engine, "approach", target="tote_1", standoff=0.45).error == ""
        assert _run(engine, "grasp", hand="both", target="tote_1").error == ""
        st = _run(engine, "carry", to="rack/side", height=0.85, limit=90.0)
        assert st.error == "", st.error
        assert _arrived(engine)["stance"] == "rack/side/stand"
    finally:
        engine.shutdown()
