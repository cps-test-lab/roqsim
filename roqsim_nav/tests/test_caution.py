"""Caution: stop for what the planner cannot see, and never re-route around it.

Two halves. The first is the classification -- what counts as a blocker -- because getting it wrong
is silent in both directions: treat walls as blockers and a mover freezes wherever its path hugs
one; treat a robot as a wall and it drives straight through the thing it was supposed to yield to.

The second is what happens *while* blocked, which is where the subtle failure lives. Recovery is on
by default, and a mover waiting for traffic looks exactly like a mover that is stuck. If the
stuck clock keeps running, "wait for the robot to pass" turns itself into "back up and plan around
it" after ``stuck_time`` -- silently converting a stop into the re-route it was meant to avoid. So
these run with recovery ENABLED; asserting them with it off would test nothing.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim_nav.caution import CautionProbe, is_dynamic_body, subtree_geoms
from roqsim_nav.plugins.navigator import AVOIDANCE_SCHEMA

DATA = Path(__file__).parent / "data"
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""

# A corridor with walls well clear of the run, so the only thing that can block is what we put there.
CORRIDOR = """<mujoco model="corridor">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .05"/>
    <geom name="wall_n" type="box" pos="0  3 1" size="6 .1 1"/>
    <geom name="wall_s" type="box" pos="0 -3 1" size="6 .1 1"/>
  </worldbody>
</mujoco>"""

START, GOAL = (-4.0, 0.0), (4.0, 0.0)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _world(tmp_path, *, blocker=None, nav=None, blocker_driven=True, blocker_half_height=None):
    """The mover, plus optionally a second prop sitting in its way at the origin."""
    crate = _write(tmp_path, "crate.xml", CRATE)
    navigator = {
        "speed": 0.8,
        "goals": [list(GOAL)],
        "obstacle_height": [0.05, 0.6],
        # ON: these tests are about the interaction, not about avoiding it.
        "recovery": {"enabled": True, "stuck_time": 0.5},
        "avoidance": {"lookahead": 1.0, "clear_time": 0.0},
    }
    navigator.update(nav or {})
    components = [
        {
            "spawn_model": {
                "model": crate,
                "motion": "driven",
                "pose": {"position": {"x": START[0], "y": START[1], "z": 0.25}},
            },
            "name": "cart",
            "components": [{"navigator": navigator}],
        }
    ]
    if blocker is not None:
        half = 0.25 if blocker_half_height is None else blocker_half_height
        low = _write(
            tmp_path,
            "low.xml",
            CRATE.replace('size=".25 .25 .25"', f'size=".25 .25 {half}"'),
        )
        components.append(
            {
                "spawn_model": {
                    "model": crate if blocker_half_height is None else low,
                    "prefix": "b_",
                    "pose": {"position": {"x": blocker[0], "y": blocker[1], "z": half}},
                    **({"motion": "driven"} if blocker_driven else {"motion": "physics"}),
                },
                "name": "blocker",
            }
        )
    return load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": _write(tmp_path, "corridor.xml", CORRIDOR)},
            "components": components,
        },
        base_dir=tmp_path,
    )


def _mocapid(engine, name="cart"):
    model = engine.ctx.model
    body = engine.ctx.entities.get(name).body
    return int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])


def _xy(engine, name="cart"):
    return engine.ctx.data.mocap_pos[_mocapid(engine, name)][:2].copy()


def _run(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def _navigator(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")


# -- what counts as a blocker ------------------------------------------------------------------
def test_a_wall_ahead_is_not_a_blocker(tmp_path):
    """Walls are the planner's business. A mover that stopped for them could not follow a path."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        assert np.linalg.norm(_xy(engine) - np.asarray(GOAL)) < 0.3
    finally:
        engine.shutdown()


def test_a_mocap_body_in_the_way_is_a_blocker(tmp_path):
    """It sets off, then holds clear of the blocker rather than driving through it.

    The clearance is asserted against the mover's own measured footprint, not a magic number: a
    mocap body passes straight through anything, so "did not overlap" is the only evidence that
    something stopped it rather than the geometry doing so.
    """
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0)))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        x = _xy(engine)[0]
        probe = _navigator(engine)._caution
        nose = x + probe.footprint_radius(engine.ctx, _xy(engine))
        assert x > START[0] + 1.0, "never set off at all"
        assert nose < -0.25, f"its front overlapped the blocker (nose at {nose})"
        assert probe.blocked, "it stopped for some reason other than caution"
    finally:
        engine.shutdown()


def test_a_free_jointed_body_in_the_way_is_a_blocker(tmp_path):
    """The other way to move without being a wall: real degrees of freedom.

    A mocap mover is infinitely massive to the solver, so if caution failed here it would not stop
    at the blocker -- it would shovel it down the corridor. That the blocker has not moved is the
    assertion, and it is a stronger one than the mover's own position.
    """
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), blocker_driven=False))
    engine.setup()
    engine.reset()
    try:
        model = engine.ctx.model
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("blocker").body
        )
        before = engine.ctx.data.xpos[bid].copy()
        _run(engine, 12.0)
        moved = float(np.linalg.norm(engine.ctx.data.xpos[bid][:2] - before[:2]))
        assert moved < 0.05, f"the blocker was pushed {moved:.2f} m -- it drove into it"
        assert _xy(engine)[0] > START[0] + 1.0, "never set off at all"
    finally:
        engine.shutdown()


def test_its_own_geoms_are_never_blockers(tmp_path):
    """A robot is a base plus wheels; excluding only the root body leaves it blocked by itself."""
    engine = Engine(_world(tmp_path))
    engine.setup()
    try:
        model = engine.ctx.model
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("cart").body
        )
        own = subtree_geoms(model, bid)
        assert own, "the mover has no geoms, so the exclusion set proves nothing"
        assert all(int(model.geom_bodyid[g]) != 0 for g in own)
    finally:
        engine.shutdown()


def test_an_ignored_entity_is_not_a_blocker(tmp_path):
    engine = Engine(
        _world(tmp_path, blocker=(0.0, 0.0), nav={"avoidance": {"ignore": ["blocker"]}})
    )
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        assert np.linalg.norm(_xy(engine) - np.asarray(GOAL)) < 0.4
    finally:
        engine.shutdown()


def test_the_classification_is_the_complement_of_the_planner_grid(tmp_path):
    """Stated as an invariant, not just exercised: every body is one side or the other."""
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0)))
    engine.setup()
    try:
        model = engine.ctx.model
        walls = [b for b in range(model.nbody) if not is_dynamic_body(model, b)]
        movers = [b for b in range(model.nbody) if is_dynamic_body(model, b)]
        assert len(walls) + len(movers) == model.nbody
        # The two crates move; the worldbody (floor and walls) does not.
        assert 0 in walls
        assert len(movers) == 2
    finally:
        engine.shutdown()


# -- what happens while blocked ------------------------------------------------------------------
def test_blocked_it_holds_its_path_instead_of_recovering(tmp_path):
    """The load-bearing one. With recovery ON, a mover held for many `stuck_time`s must not back up.

    If the stuck clock were allowed to run, this is exactly where a stop would silently become a
    re-route -- and the mover would arrive at the goal by a path the experiment never chose.
    """
    engine = Engine(
        _world(tmp_path, blocker=(0.0, 0.0), nav={"recovery": {"enabled": True, "stuck_time": 0.5}})
    )
    engine.setup()
    engine.reset()
    try:
        _run(engine, 8.0)  # ~16 x stuck_time of being blocked
        held = _xy(engine)
        path_before = list(_navigator(engine)._core.st.path or [])

        _run(engine, 6.0)
        assert _xy(engine) == pytest.approx(held, abs=0.05), "it wandered while blocked"
        assert list(_navigator(engine)._core.st.path or []) == path_before, "it re-planned"
        assert _xy(engine)[0] > START[0] + 1.0, "it backed up"
    finally:
        engine.shutdown()


def test_it_resumes_on_the_same_path_when_the_way_clears(tmp_path):
    """Stopping preserves the trajectory; the mover picks up the leg it was already on."""
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0)))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 8.0)
        blocked_at = _xy(engine)
        path_while_blocked = list(_navigator(engine)._core.st.path or [])

        # Take the blocker away, as a crossing robot would by moving on.
        engine.ctx.data.mocap_pos[_mocapid(engine, "blocker")] = [0.0, 8.0, 0.25]
        _run(engine, 12.0)

        assert list(_navigator(engine)._core.st.path or []) == path_while_blocked, "re-planned"
        assert _xy(engine)[0] > blocked_at[0] + 1.0, "never resumed"
        assert np.linalg.norm(_xy(engine) - np.asarray(GOAL)) < 0.4
    finally:
        engine.shutdown()


def test_caution_can_be_turned_off(tmp_path):
    """Off, it drives into the blocker -- which is the evidence that on, it was caution stopping it."""
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"avoidance": {"stop": False}}))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        assert _xy(engine)[0] > -0.5, "something other than caution was holding it back"
    finally:
        engine.shutdown()


# -- configuration -------------------------------------------------------------------------------
def test_replan_reports_a_blocker_the_same_way_stop_does(tmp_path):
    """Both modes see the same thing; they differ only in what they do about it.

    The routing half is ``test_navigator_replan.py``. What matters here is that turning it on does
    not change what counts as a blocker -- so a world can switch modes without re-tuning the probe.
    """
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"avoidance": {"reroute": True}}))
    engine.setup()
    engine.reset()
    try:
        seen = []
        for _ in range(int(12.0 / engine.ctx.dt)):
            engine.step()
            probe = _navigator(engine)._caution
            if probe.blocked:
                seen.extend(probe.blocker_points)
        assert seen, "the blocker went unseen"
        # It reports points across the corridor as far as the blocker spans it, not one point for
        # the whole of it.
        assert any(not np.allclose(p, seen[0]) for p in seen)
    finally:
        engine.shutdown()


def test_an_unknown_avoidance_key_is_refused(tmp_path):
    """A typo must not read as a default. The block names what it accepts."""
    with pytest.raises(PluginError, match="'avoidance.on_blocked' is not a key of 'avoidance'"):
        Engine(_world(tmp_path, nav={"avoidance": {"on_blocked": "replan"}}))


def test_ignoring_an_entity_that_does_not_exist_is_refused(tmp_path):
    """Raised at the first reset, not at setup: `ignore` may name an entity declared later."""
    engine = Engine(_world(tmp_path, nav={"avoidance": {"ignore": ["ghost"]}}))
    engine.setup()
    with pytest.raises(RuntimeError, match="not an entity"):
        engine.reset()


def test_a_stopped_mover_looks_at_nothing(tmp_path):
    """Cost: nothing to run into if we are not going anywhere, and no direction to cast along."""
    probe = CautionProbe({})
    assert probe.check(None, (0.0, 0.0), 0.25, np.zeros(2)) is False


# -- presence -------------------------------------------------------------------------------------
def test_an_absent_obstacle_is_not_planned_around(tmp_path):
    """A thing nothing can see or touch must not bend a trajectory.

    ``roqsim.presence`` exists because the model cannot gain a body at runtime: an obstacle that must
    *appear* mid-trial is compiled in and made absent until it is wanted. Every raycaster honours
    that. The planner's grid is the same kind of consumer, and getting it wrong is silent in the
    worst way -- the mover drives a smooth, plausible detour around empty space, and nothing in the
    run says why.

    It is why the grid is rasterized on the first tick rather than in ``configure``: ``configure``
    runs before any presence has been applied, so a grid built there contains every obstacle the
    world compiled, present or not.
    """
    from roqsim import presence

    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"avoidance": {"stop": False}}))
    engine.setup()
    engine.reset()
    try:
        assert presence.set_present(engine.ctx, engine.ctx.entities.get("blocker"), False)
        _run(engine, 14.0)
        track_end = _xy(engine)
        assert np.linalg.norm(track_end - np.asarray(GOAL)) < 0.4, "it did not reach the goal"
    finally:
        engine.shutdown()


def test_a_present_obstacle_is_still_planned_around(tmp_path):
    """The other half: deferring the grid must not stop it seeing what IS there."""
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"avoidance": {"stop": False}}))
    engine.setup()
    engine.reset()
    try:
        navigator = _navigator(engine)
        engine.step()
        assert navigator._core.planner is not None, "the grid was never built"
        _run(engine, 14.0)
        # The blocker is a 0.5 m mocap box on the straight line, so any real path leaves y = 0.
        # (Mocap, so it is NOT in the grid -- see the module docstring; this asserts the walls are.)
        assert navigator._core.planner.grid is not None
    finally:
        engine.shutdown()


# -- the traffic policy ---------------------------------------------------------------------------
def test_traffic_ignore_drives_on_and_respect_stops(tmp_path):
    """Both are first-class choices, and the difference is the whole of the policy.

    ``ignore`` is for an opponent that must be in the same place at the same time in every
    repetition: stopping for the robot under test would make its trajectory a function of that
    robot's behaviour, which is the coupling a controlled trial removes. ``respect`` is for one
    sharing a corridor.
    """
    outcomes = {}
    for policy, stop in (("ignore", False), ("respect", True)):
        engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"avoidance": {"stop": stop}}))
        engine.setup()
        engine.reset()
        try:
            _run(engine, 12.0)
            outcomes[policy] = float(_xy(engine)[0])
        finally:
            engine.shutdown()
    assert outcomes["ignore"] > -0.5, "it stopped although it was told not to care"
    assert outcomes["respect"] < -0.5, "it drove on although it was told to stop"


def test_avoidance_as_a_bare_value_is_refused(tmp_path):
    """It is a block rather than a model name, and the message has to say so."""
    with pytest.raises(PluginError, match="is a block, not a value"):
        Engine(_world(tmp_path, nav={"avoidance": "give_way"}))


def test_steer_needs_a_model_name_not_a_yes_or_no(tmp_path):
    with pytest.raises(PluginError, match="names a model, not a yes/no"):
        Engine(_world(tmp_path, nav={"avoidance": {"steer": True}}))


def test_reroute_without_stopping_is_refused(tmp_path):
    """A blockage is only ever discovered by looking ahead and stopping for it."""
    with pytest.raises(PluginError, match="reroute"):
        Engine(_world(tmp_path, nav={"avoidance": {"reroute": True, "stop": False}}))


def test_the_probe_counts_what_reaches_into_the_movers_own_obstacle_band():
    """What is in the way is what reaches up into the band the mover's planner rasterizes walls in."""
    assert CautionProbe({"band": (0.1, 1.8)}).band == (0.1, 1.8)
    assert CautionProbe({}).band == (0.1, 1.8)


@pytest.mark.parametrize("key", ["rays", "height"])
def test_a_key_of_the_rays_is_refused_naming_why(tmp_path, key):
    with pytest.raises(PluginError, match=rf"'avoidance\.{key}' is not read any more: the probe"):
        _probe(tmp_path, **{key: 5})


# -- blockage_radius: derived when absent, used as written when set -------------------------------
DERIVED = {"blockage_radius": 0.3}  # width 0.6 / 2


def _probe(tmp_path, **avoidance):
    engine = Engine(_world(tmp_path, nav={"avoidance": {"lookahead": 1.0, **avoidance}}))
    engine.setup()
    try:
        return _navigator(engine)._caution
    finally:
        engine.shutdown()


@pytest.mark.parametrize("key", sorted(DERIVED))
def test_an_absent_key_is_derived(tmp_path, key):
    assert getattr(_probe(tmp_path), key) == pytest.approx(DERIVED[key])


@pytest.mark.parametrize("key", sorted(DERIVED))
def test_a_written_value_is_used_as_written(tmp_path, key):
    assert getattr(_probe(tmp_path, **{key: 0.42}), key) == pytest.approx(0.42)


@pytest.mark.parametrize("key", sorted(DERIVED))
def test_a_written_zero_is_refused_by_name(tmp_path, key):
    with pytest.raises(PluginError, match=rf"'avoidance\.{key}' must be > 0"):
        _probe(tmp_path, **{key: 0})


@pytest.mark.parametrize("key", sorted(DERIVED))
def test_the_probe_derives_only_an_absent_key(key):
    """A written value is never swapped for the derived one, whatever its value."""
    assert getattr(CautionProbe({"band": (0.05, 0.6), key: 0.0}), key) == 0.0


@pytest.mark.parametrize("key", sorted(DERIVED))
def test_the_schema_publishes_what_is_derived_not_a_number(key):
    described = AVOIDANCE_SCHEMA[key].describe(key)
    assert described["default"] is None
    assert described["doc"].split("; ")[1].startswith("default: ")


def test_a_short_robot_is_seen_by_another_short_robot(tmp_path):
    """The case itself: two movers no taller than a 0.30 m fixed scan height."""
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), blocker_half_height=0.12))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        assert _xy(engine)[0] < -0.5, "it drove over something it should have stopped for"
    finally:
        engine.shutdown()


# -- yielding to traffic vs being held by something that will not move ------------------------------
def test_a_block_stops_reading_as_traffic_after_yield_time():
    """Waiting for a robot to pass must not look like being stuck; a wall is not going to pass.

    The navigator rebases its progress clock for as long as this says the block is traffic, so this
    is what decides whether recovery can ever engage. Without the second half a mover in a pocket
    can never get out, because leaving means driving toward the obstacle beside it and every plan it
    makes is refused by this same probe. That a mover without `reroute` ignores it -- being blocked
    must not change the path in that mode -- is asserted end to end by
    `test_blocked_it_holds_its_path_instead_of_recovering`.
    """
    probe = CautionProbe({"yield_time": 2.0})
    probe._blocked_since = 10.0
    assert probe.blocked_for(11.0) == pytest.approx(1.0)
    assert probe.yielding(11.0), "a fresh blockage reads as traffic"
    assert not probe.yielding(12.5), "one that outlasts yield_time does not"
    probe.reset()
    assert probe.blocked_for(12.5) == 0.0 and probe.yielding(12.5)


def test_a_momentary_gap_does_not_re_arm_the_yield_window(tmp_path):
    """A clear tick during the settle is not the same as being through.

    Treating it as such re-arms the window on every flicker, which is exactly the state a mover
    stuttering against an obstacle is in -- so it would yield for ever and never recover.
    """
    probe = CautionProbe({"yield_time": 1.0, "clear_time": 5.0})
    probe._blocked_since = 0.0
    probe._clear_since = 2.0  # the corridor came up clear, but the settle has not elapsed
    assert probe.blocked_for(3.0) == pytest.approx(3.0), "the counter was reset by a flicker"


def test_a_mover_does_not_wait_before_it_has_ever_been_blocked(tmp_path):
    """`clear_time` settles a way that has REOPENED, so it must not run from a cold start.

    Otherwise every mover holds for `clear_time` at the beginning of every run, having never
    met anything -- the timer starts from an unset value and reads as "the way only just cleared".
    """
    # nothing in the way at all, and a settle long enough to see
    engine = Engine(_world(tmp_path, nav={"avoidance": {"lookahead": 1.0, "clear_time": 0.5}}))
    engine.setup()
    engine.reset()
    try:
        probe = _navigator(engine)._caution
        start = _xy(engine)
        for _ in range(int(1.0 / engine.ctx.dt)):
            engine.step()
            assert not probe.blocked, "it held despite never having been blocked"
        # 0.8 m/s for a second, less the first tick
        assert _xy(engine)[0] - start[0] > 0.7, "it stood before it walked"
    finally:
        engine.shutdown()


# -- setting off again ---------------------------------------------------------------------------
# What a held mover may set off into is judged against its corridor, filled in: each test holds the
# cart for a blocker, then puts one where the cart's body would meet it -- a base lower than the
# cart's obstacle band, or one beside the cart's front.

#: A robot whose base is lower than the floor of the cart's obstacle band (0.05 m) and whose post
#: is 6 cm wide: a TurtleBot's shape.
LOW_ROBOT = """<mujoco model="low_robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".2 .03" pos="0 0 .03"/>
    <geom name="post" type="cylinder" size=".03 .2" pos="0 0 .26"/>
  </body></worldbody>
</mujoco>"""

#: A slim body as tall as the cart.
POST = """<mujoco model="post">
  <worldbody><body name="base"><geom name="g" type="cylinder" size=".06 .25" pos="0 0 .25"/></body>
  </worldbody>
</mujoco>"""


def _held_world(
    tmp_path, blocker_xml, *, clear_time=0.0, blocker_spawn=None, goals=None, avoidance=None
):
    """The cart of :func:`_world`, and a driven blocker in its way at the origin.

    ``blocker_spawn`` overrides keys of the blocker's ``spawn_model``; ``goals`` the cart's route;
    ``avoidance`` adds keys to its ``avoidance:``.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    navigator = {
        "speed": 0.8,
        "goals": goals or [list(GOAL)],
        "obstacle_height": [0.05, 0.6],
        "recovery": {"enabled": True, "stuck_time": 0.5},
        "avoidance": {"lookahead": 1.0, "clear_time": clear_time, **(avoidance or {})},
    }
    cart = {
        "spawn_model": {
            "model": _write(tmp_path, "crate.xml", CRATE),
            "motion": "driven",
            "pose": {"position": {"x": START[0], "y": START[1], "z": 0.25}},
        },
        "name": "cart",
        "components": [{"navigator": navigator}],
    }
    blocker = {
        "spawn_model": {
            "model": _write(tmp_path, "blocker.xml", blocker_xml),
            "prefix": "b_",
            "motion": "driven",
            **(blocker_spawn or {}),
        },
        "name": "blocker",
    }
    return load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": _write(tmp_path, "corridor.xml", CORRIDOR)},
            "components": [cart, blocker],
        },
        base_dir=tmp_path,
    )


def _place(engine, xy, name="blocker"):
    engine.ctx.data.mocap_pos[_mocapid(engine, name)][:2] = xy


def _held_engine(tmp_path, blocker_xml, **kw):
    engine = Engine(_held_world(tmp_path, blocker_xml, **kw))
    engine.setup()
    engine.reset()
    _run(engine, 6.0)
    assert _navigator(engine)._caution.blocked, "the cart never held for the blocker"
    return engine


def test_a_held_mover_does_not_set_off_into_a_low_base_beside_its_line(tmp_path):
    """The blocker shifts 0.075 m beside the cart's line; its base is still in the way."""
    engine = _held_engine(tmp_path, LOW_ROBOT)
    try:
        held = _xy(engine)
        _place(engine, (0.0, 0.075))  # 0.075 m beside the line, its base in the way
        _run(engine, 3.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), "it set off into the robot's base"
        assert _navigator(engine)._caution.blocked

        _place(engine, (0.0, 8.0))  # and once it has gone, the way is open
        _run(engine, 3.0)
        assert _xy(engine)[0] > held[0] + 1.0, "it never set off after the robot left"
    finally:
        engine.shutdown()


def test_a_held_mover_does_not_set_off_into_a_body_beside_its_front(tmp_path):
    """A body just ahead of the cart's front, nearer than its circumscribed radius."""
    engine = _held_engine(tmp_path, POST)
    try:
        held = _xy(engine)
        probe = _navigator(engine)._caution
        # The cart's front is 0.25 m ahead of its centre and its circumscribed radius 0.43 m; the
        # post spans 0.28 to 0.40 m ahead, 0.12 to 0.24 m to the side.
        assert probe.footprint_radius(engine.ctx, held) > 0.41
        _place(engine, held + np.array([0.34, 0.18]))
        _run(engine, 3.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), "it set off into the post"
    finally:
        engine.shutdown()


def test_a_held_mover_sets_off_past_a_body_clear_of_its_corridor(tmp_path):
    """Beside the cart and clear of the corridor it stops for, the robot is not in the way."""
    engine = _held_engine(tmp_path, LOW_ROBOT)
    try:
        held = _xy(engine)
        _place(engine, held + np.array([0.3, 0.52]))  # base edge 0.32 m to the side, corridor's 0.3
        _run(engine, 3.0)
        assert _xy(engine)[0] > held[0] + 1.0, "it held for a robot clear of its corridor"
    finally:
        engine.shutdown()


def _release_time(tmp_path) -> float:
    """When the cart sets off after the low robot has driven across its way and on."""
    engine = _held_engine(tmp_path, LOW_ROBOT, clear_time=0.5)
    try:
        start, held = engine.ctx.sim_time, _xy(engine)
        while engine.ctx.sim_time < start + 8.0:
            # Across the way at 0.15 m/s, sideways out of it.
            _place(engine, (0.0, CROSSING * (engine.ctx.sim_time - start)))
            engine.step()
            if np.linalg.norm(_xy(engine) - held) > 1e-3:
                return engine.ctx.sim_time - start
        raise AssertionError("it never set off")
    finally:
        engine.shutdown()


#: How fast the low robot crosses.
CROSSING = 0.15


def test_a_robot_that_drives_across_and_on_is_held_for_until_its_base_has_left(tmp_path):
    """Held for as long as its base is in the corridor and ``clear_time`` after, not its post.

    The base (0.2 m) leaves the 0.6 m corridor when the robot's centre is 0.5 m to the side; its
    post 0.17 m before that.
    """
    base_out = (0.3 + 0.2) / CROSSING
    assert _release_time(tmp_path) == pytest.approx(base_out + 0.5, abs=0.06)  # within a tick


def test_a_robot_back_in_the_corridor_before_the_settle_is_over_holds_it_again(tmp_path):
    """The corridor must stay clear for the whole settle, not only at its end."""
    engine = _held_engine(tmp_path, LOW_ROBOT, clear_time=0.5)
    try:
        held = _xy(engine)
        _place(engine, (0.0, 0.075))  # 0.075 m beside the line, the base in the way
        _run(engine, 2.0)
        _place(engine, (0.0, 8.0))  # out of the corridor for less than `clear_time` ...
        _run(engine, 0.2)
        _place(engine, (0.0, 0.075))  # ... and back
        _run(engine, 3.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), (
            "it set off into the base that came back"
        )
    finally:
        engine.shutdown()


@pytest.mark.parametrize("clear_time", [0.0, 0.5])
def test_a_tick_without_speed_during_a_hold_does_not_end_it(tmp_path, clear_time):
    """A tick with no speed (an avoidance output can be zero) ends no hold: the corridor is looked at."""
    engine = _held_engine(tmp_path, LOW_ROBOT, clear_time=clear_time)
    try:
        held = _xy(engine)
        _place(engine, (0.0, 0.075))  # 0.075 m beside the line, the base in the way
        _navigator(engine)._caution.check(engine.ctx, held, 0.0, (0.0, 0.0))
        _run(engine, 3.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), "it set off into the robot's base"
    finally:
        engine.shutdown()


#: A loose mat, lower than the floor of the cart's obstacle band: walked, or driven, over.
MAT = """<mujoco model="mat">
  <worldbody><body name="base"><geom name="g" type="box" size=".15 .15 .02" pos="0 0 .02"/></body>
  </worldbody>
</mujoco>"""


def test_a_mat_below_the_movers_band_does_not_hold_it(tmp_path):
    """A loose mat beside its way, lower than its obstacle band, does not stop it."""
    beside_its_way = {"position": {"x": START[0] + 0.9, "y": 0.2, "z": 0.0}}
    engine = Engine(
        _held_world(
            tmp_path,
            MAT,
            clear_time=0.5,
            blocker_spawn={"motion": "physics", "pose": beside_its_way},
        )
    )
    engine.setup()
    engine.reset()
    try:
        start = _xy(engine)
        _run(engine, 3.0)
        assert _xy(engine)[0] > start[0] + 1.0, "it held for a mat below its band"
    finally:
        engine.shutdown()


# -- what the corridor sees: the whole of a thing, in the body's way along its path -------------------
def test_a_robot_with_a_low_base_and_a_slim_post_holds_the_mover_before_it(tmp_path):
    """A TurtleBot's shape on the way: a base below the band's floor, a post 6 cm wide, 0.075 m to
    the side of the mover's line."""
    stands = {"position": {"x": 0.0, "y": 0.075, "z": 0.0}}
    engine = Engine(_held_world(tmp_path, LOW_ROBOT, blocker_spawn={"pose": stands}))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 8.0)
        # the cart's front (0.25 m) stays clear of the base's edge (0.2 m before its centre)
        assert _xy(engine)[0] < -0.45, "it drove into the robot's base"
        assert _navigator(engine)._caution.blocked
    finally:
        engine.shutdown()


def test_the_corridor_follows_the_path_round_a_turn(tmp_path):
    """A post 0.3 m down the leg after a right-angle turn is seen before the corner (a guard: a
    probe looking along its heading meets it too, once it has turned)."""
    stands = {"position": {"x": 0.0, "y": -0.3, "z": 0.0}}
    world = _held_world(
        tmp_path, POST, blocker_spawn={"pose": stands}, goals=[[0.0, 0.0], [0.0, -2.5]]
    )
    engine = Engine(world)
    engine.setup()
    engine.reset()
    try:
        post = np.array([0.0, -0.3])
        least = np.inf
        for _ in range(int(10.0 / engine.ctx.dt)):
            engine.step()
            least = min(least, float(np.linalg.norm(_xy(engine) - post)))
        # the cart's box (0.25 m half side, 0.35 m to a corner) never reaches the post (0.06 m)
        assert least > 0.25 + 0.06, f"it came {least:.2f} m from the post's centre"
    finally:
        engine.shutdown()


def test_a_robot_that_drives_into_the_movers_body_holds_it(tmp_path):
    """Something that already reaches into the mover's body is in its way, as from its front."""
    engine = Engine(
        _held_world(
            tmp_path,
            LOW_ROBOT,
            blocker_spawn={"pose": {"position": {"x": 0.0, "y": 8.0, "z": 0.0}}},
        )
    )
    engine.setup()
    engine.reset()
    try:
        _run(engine, 2.0)
        cart = _xy(engine)
        _place(engine, cart + np.array([0.15, 0.35]))  # its base 0.1 m into the cart's flank
        _run(engine, 0.2)
        held = _xy(engine)
        _run(engine, 1.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), "it walked on through the robot"
    finally:
        engine.shutdown()


def test_a_robot_straight_ahead_past_a_turn_does_not_hold_the_mover(tmp_path):
    """A robot standing 0.8 m beyond the corner, on the straight line the mover leaves there: a
    corridor laid along the heading holds for it, for good; laid along the path, the mover turns
    before it and walks on."""
    stands = {"position": {"x": 0.8, "y": 0.0, "z": 0.0}}
    world = _held_world(
        tmp_path, POST, blocker_spawn={"pose": stands}, goals=[[0.0, 0.0], [0.0, -2.5]]
    )
    engine = Engine(world)
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        assert _xy(engine)[1] < -2.0, f"it stood at {_xy(engine)} for a robot past the corner"
    finally:
        engine.shutdown()


def test_a_mover_held_once_walks_off_after_a_dwell_without_the_settle(tmp_path):
    """Held on its first leg, then standing at a goal for a dwell with the way clear: it walks off
    when the dwell is over, not `clear_time` later."""
    blocker = _write(tmp_path, "blocker.xml", CRATE)
    cart = _write(tmp_path, "crate.xml", CRATE)
    nav = {
        "speed": 0.8,
        "goals": [[-1.0, 0.0], [4.0, 0.0]],
        "dwell": [0.0, 2.0, 0.0],
        "obstacle_height": [0.05, 0.6],
        "recovery": {"enabled": False},
        "avoidance": {"lookahead": 1.0, "clear_time": 0.5},
    }
    world = load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": _write(tmp_path, "corridor.xml", CORRIDOR)},
            "components": [
                {
                    "spawn_model": {
                        "model": cart,
                        "motion": "driven",
                        "pose": {"position": {"x": -4.0, "y": 0.0, "z": 0.25}},
                    },
                    "name": "cart",
                    "components": [{"navigator": nav}],
                },
                {
                    "spawn_model": {
                        "model": blocker,
                        "prefix": "b_",
                        "motion": "driven",
                        "pose": {"position": {"x": -2.4, "y": 0.0, "z": 0.25}},
                    },
                    "name": "blocker",
                },
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(world)
    engine.ctx.seed = 1  # a dwell draws its length, even a fixed one
    engine.setup()
    engine.reset()
    try:
        _run(engine, 2.0)
        assert _navigator(engine)._caution.blocked, "the cart never held for the blocker"
        _place(engine, (-2.4, 8.0))
        # On to the goal at x = -1, its 2 s dwell, and off again: it stands at the goal for the
        # dwell and nothing more. Standing is measured over 0.1 s, longer than a navigator tick.
        stood, last = 0.0, _xy(engine)[0]
        for _ in range(100):
            _run(engine, 0.1)
            x = _xy(engine)[0]
            if -1.4 < x < -0.6 and abs(x - last) < 1e-6:
                stood += 0.1
            last = x
        assert x > 0.0, "it never went on from the goal"
        assert stood == pytest.approx(2.0, abs=0.15), f"it stood {stood:.2f} s at the goal"
    finally:
        engine.shutdown()


#: A long cart, 1.2 m by 0.4 m in plan.
LONG = """<mujoco model="long">
  <worldbody><body name="long"><geom name="g" type="box" size=".6 .2 .25"/></body></worldbody>
</mujoco>"""


def test_a_long_mover_that_turns_to_its_way_has_the_front_of_the_side_it_goes(tmp_path):
    """Spawned across its way, its first look meets its short side; once it has turned to face its
    way, a post beside its flank, 0.4 m ahead of its centre and behind its 0.6 m front, is not in its
    way."""
    long = _write(tmp_path, "long.xml", LONG)
    post = _write(tmp_path, "post.xml", POST)
    nav = {
        "speed": 0.5,
        "goals": [[4.0, 0.0]],
        "obstacle_height": [0.05, 0.6],
        "recovery": {"enabled": False},
        "avoidance": {"lookahead": 1.0, "clear_time": 0.0},
    }
    world = load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": _write(tmp_path, "corridor.xml", CORRIDOR)},
            "components": [
                {
                    "spawn_model": {
                        "model": long,
                        "motion": "driven",
                        "pose": {
                            "position": {"x": -4.0, "y": 0.0, "z": 0.25},
                            "orientation": {"yaw": 1.5708},
                        },
                    },
                    "name": "cart",
                    "components": [{"navigator": nav}],
                },
                {
                    "spawn_model": {
                        "model": post,
                        "prefix": "b_",
                        "motion": "driven",
                        "pose": {"position": {"x": 0.0, "y": 8.0, "z": 0.0}},
                    },
                    "name": "blocker",
                },
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(world)
    engine.setup()
    engine.reset()
    try:
        _run(engine, 3.0)  # turned to its way and walking
        start = _xy(engine)
        for _ in range(int(1.0 / engine.ctx.dt)):
            _place(engine, _xy(engine) + (0.4, 0.3))  # abreast of its flank, as it goes
            engine.step()
        assert _xy(engine)[0] > start[0] + 0.3, "it held for a post beside its flank"
    finally:
        engine.shutdown()


# -- the speed follows the gap ahead (decel, stop_gap, wait_gap, tau, horizon) ----------------------
PACE = {"horizon": 3.0, "decel": 1.0, "stop_gap": 0.3, "wait_gap": 0.85}


def _speeds(engine, seconds, each=None):
    """The cart's x and speed at every navigator tick for ``seconds``; ``each(t)`` runs first."""
    xs, ts = [], []
    start = engine.ctx.sim_time
    for _ in range(int(seconds / engine.ctx.dt)):
        if each is not None:
            each(engine.ctx.sim_time - start)
        engine.step()
        x = _xy(engine)[0]
        if not xs or x != xs[-1]:
            xs.append(x)
            ts.append(engine.ctx.sim_time)
    x, t = np.array(xs), np.array(ts)
    return x, t, np.r_[0.0, np.diff(x) / np.maximum(np.diff(t), 1e-9)]


def test_with_decel_it_slows_to_a_stand_short_of_what_stands_in_its_way(tmp_path):
    """Slowing on the stop that ``decel`` allows, not all at once, and standing ``stop_gap`` short."""
    engine = Engine(_held_world(tmp_path, LOW_ROBOT, avoidance=PACE))
    engine.setup()
    engine.reset()
    try:
        x, _, v = _speeds(engine, 8.0)
        slowing = v[(v > 0.1) & (v < 0.6)]  # set speed 0.8
        assert slowing.size >= 5, "it went from walking to standing in a tick or two"
        # the cart's front (0.25 m) stands 0.3 m short of the base's edge (0.2 m before its centre)
        assert x[-1] == pytest.approx(-0.2 - 0.3 - 0.25, abs=0.05)
    finally:
        engine.shutdown()


def test_with_decel_it_stands_further_back_for_a_mover_crossing_ahead(tmp_path):
    """A robot crossing slowly ahead is stood short of by ``wait_gap``, not ``stop_gap``."""
    away = {"pose": {"position": {"x": 0.0, "y": 8.0, "z": 0.0}}}
    engine = Engine(_held_world(tmp_path, LOW_ROBOT, blocker_spawn=away, avoidance=PACE))
    engine.setup()
    engine.reset()
    try:
        # across the cart's way at 0.15 m/s, from 0.4 m to its left: in the corridor until 6 s,
        # and still in it at 4.5 s, when the cart has stood a while
        x, t, _ = _speeds(engine, 4.5, each=lambda t: _place(engine, (0.0, 0.4 - 0.15 * t)))
        assert x[-1] == pytest.approx(-0.2 - 0.85 - 0.25, abs=0.08)
    finally:
        engine.shutdown()


def test_a_mover_crossing_that_will_have_left_is_not_slowed_for(tmp_path):
    """A robot crossing fast, well ahead: gone from the corridor before the cart gets there."""
    away = {"pose": {"position": {"x": 0.0, "y": 8.0, "z": 0.0}}}
    engine = Engine(_held_world(tmp_path, LOW_ROBOT, blocker_spawn=away, avoidance=PACE))
    engine.setup()
    engine.reset()
    try:
        # 1.2 m/s across, through the cart's line when the cart is 2.8 m short of it
        x, _, v = _speeds(engine, 4.0, each=lambda t: _place(engine, (0.0, 1.5 - 1.2 * t)))
        assert v[5:].min() > 0.9 * 0.8, "it slowed for a robot that crossed ahead of it"
    finally:
        engine.shutdown()


def test_after_a_hold_it_walks_off_with_tau(tmp_path):
    """Set off by the robot leaving, its speed rises to the set speed with ``tau``."""
    engine = Engine(
        _held_world(tmp_path, LOW_ROBOT, avoidance={**PACE, "tau": 0.5}, clear_time=0.0)
    )
    engine.setup()
    engine.reset()
    try:
        _run(engine, 8.0)
        assert _navigator(engine)._caution.blocked
        _place(engine, (0.0, 8.0))
        x, t, v = _speeds(engine, 3.0)
        t = t - t[0]
        fast = t[np.argmax(v >= 0.9 * 0.8)]
        assert fast == pytest.approx(0.5 * np.log(10.0), abs=0.15)
    finally:
        engine.shutdown()


def test_a_run_after_one_that_ended_held_walks_off_as_a_fresh_one(tmp_path):
    """A run is a function of its world: the pace one run ended on does not carry into the next."""
    pace = {**PACE, "tau": 0.5}

    def first_half_second(engine):
        start = _xy(engine)[0]
        _run(engine, 0.5)
        return _xy(engine)[0] - start

    fresh = Engine(_held_world(tmp_path / "a", LOW_ROBOT, avoidance=pace))
    fresh.setup()
    fresh.reset()
    try:
        walked_fresh = first_half_second(fresh)
    finally:
        fresh.shutdown()
    engine = Engine(_held_world(tmp_path / "b", LOW_ROBOT, avoidance=pace))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 8.0)
        assert _navigator(engine)._caution.blocked, "the first run did not end held"
        engine.reset()
        assert first_half_second(engine) == pytest.approx(walked_fresh, abs=1e-3)
    finally:
        engine.shutdown()


# -- while a model steers the mover, the probe is the last resort ------------------------------------
def _probe_with(tmp_path, xy):
    """The cart's probe, with the low robot placed at ``xy`` relative to the cart, after one step."""
    away = {"pose": {"position": {"x": 0.0, "y": 8.0, "z": 0.0}}}
    engine = Engine(_held_world(tmp_path, LOW_ROBOT, blocker_spawn=away, avoidance=PACE))
    engine.setup()
    engine.reset()
    engine.step()
    _place(engine, _xy(engine) + np.asarray(xy))
    engine.step()
    return engine, _navigator(engine)._caution


def test_steered_it_is_not_held_for_a_thing_that_reaches_into_its_body(tmp_path):
    engine, probe = _probe_with(tmp_path, (0.15, 0.35))  # the base 0.1 m into the cart's flank
    try:
        cart = _xy(engine)
        assert probe.check(engine.ctx, cart, 0.0, (0.8, 0.0), steering=False)
        probe.reset()
        assert not probe.check(engine.ctx, cart, 0.0, (0.8, 0.0), steering=True)
    finally:
        engine.shutdown()


def test_steered_it_looks_only_lookahead_ahead(tmp_path):
    """1.5 m ahead of the front: within the horizon at 0.8 m/s (2.4 m), past lookahead (1.0 m)."""
    engine, probe = _probe_with(tmp_path, (0.25 + 0.2 + 1.5, 0.0))
    try:
        cart = _xy(engine)
        probe.check(engine.ctx, cart, 0.0, (0.8, 0.0), steering=False)
        assert probe.gap == pytest.approx(1.5, abs=0.02), "not steered, it saw nothing ahead"
        probe.reset()
        probe.check(engine.ctx, cart, 0.0, (0.8, 0.0), steering=True)
        assert probe.gap is None and probe.limit is None, "steered, it looked past lookahead"
    finally:
        engine.shutdown()


def test_steered_it_stands_stop_gap_short_of_a_mover_crossing_not_wait_gap(tmp_path):
    engine, probe = _probe_with(tmp_path, (0.25 + 0.2 + 0.6, 0.0))  # 0.6 m ahead of the front
    try:
        # 0.6 m ahead, crossing from its left at 0.3 m/s: it reaches the cart's line first
        cart = _xy(engine)
        things = [
            {
                "dist": 0.6,
                "v_lat": -0.3,
                "v_along": 0.0,
                "lat_lo": -0.2,
                "lat_hi": 0.2,
                "points": [],
                "xy": cart + (1.05, 0.3),
                "v": np.array([0.0, -0.3]),
            }
        ]
        _, _, held = probe._grade(things, 0.8, origin=cart, heading=0.0)
        assert held, "it does not wait for a mover crossing 0.6 m ahead"
        _, _, held = probe._grade(things, 0.8, steering=True, origin=cart, heading=0.0)
        assert not held, "steered, it waits wait_gap short of a mover crossing"
    finally:
        engine.shutdown()


#: A box carried at hip height: off the floor, so no agent the model steers round, but within the
#: cart's obstacle band, so its probe sees it.
CARRIED = """<mujoco model="carried">
  <worldbody><body name="base"><geom name="g" type="box" size=".15 .15 .05" pos="0 0 .4"/></body>
  </worldbody>
</mujoco>"""


def test_a_navigator_whose_model_steers_it_gives_its_probe_only_lookahead(tmp_path):
    """The navigator tells its probe when its model steers: past ``lookahead``, what is ahead is the
    model's, not the probe's. A carried box some 1.5 m ahead of the cart's front, within the horizon
    at 0.8 m/s, is no agent, so the model leaves the cart's way straight at it, and only the probe's
    reach decides whether the cart slows for it yet."""
    ahead = {"pose": {"position": {"x": START[0] + 0.25 + 0.2 + 1.5, "y": START[1], "z": 0.0}}}
    world = _held_world(
        tmp_path, CARRIED, blocker_spawn=ahead, avoidance={**PACE, "steer": "give_way"}
    )
    engine = Engine(world)
    engine.setup()
    engine.reset()
    try:
        _run(engine, 0.1)  # two navigator ticks
        probe = _navigator(engine)._caution
        assert probe.gap is None and probe.limit is None, (
            f"steered, its probe saw {probe.gap} m ahead and set {probe.limit} m/s"
        )
    finally:
        engine.shutdown()


def _crosser(cart, ahead, left, speed=0.3):
    """A thing ``ahead`` of the cart's centre and ``left`` of its line, crossing it toward the right."""
    half = 0.2
    return {
        "dist": ahead - 0.45,  # from the cart's front
        "v_lat": -speed,
        "v_along": 0.0,
        "lat_lo": left - half,
        "lat_hi": left + half,
        "points": [],
        "xy": cart + (ahead, left),
        "v": np.array([0.0, -speed]),
    }


def test_of_two_crossing_the_one_that_reaches_the_crossing_first_goes_first(tmp_path):
    """At 0.8 m/s, within wait_gap of a thing crossing at 0.3 m/s: the cart walks on where it reaches
    the crossing point well before the thing does, and waits where the thing gets there first;
    within 0.1 s of each other, the one coming from the other's right goes first."""
    engine, probe = _probe_with(tmp_path, (0.0, 8.0))
    try:
        cart = _xy(engine)
        # it there at 0.94 s, the thing at 1.17 s
        _, _, held = probe._grade([_crosser(cart, 0.75, 0.35)], 0.8, origin=cart, heading=0.0)
        assert not held, "it waited for a thing it reaches the crossing well before"
        # the thing there at 0.33 s
        _, _, held = probe._grade([_crosser(cart, 0.75, 0.1)], 0.8, origin=cart, heading=0.0)
        assert held, "it walked on in front of a thing that reaches the crossing first"
        # a tie: it there at 1.2 s, the thing at 1.17 s
        _, _, held = probe._grade([_crosser(cart, 0.96, 0.35)], 0.8, origin=cart, heading=0.0)
        assert not held, "a tie with a thing on its left: it comes from the thing's right, and goes"
        mirrored = _crosser(cart, 0.96, -0.35, speed=-0.3)
        mirrored["lat_lo"], mirrored["lat_hi"] = -0.55, -0.15
        _, _, held = probe._grade([mirrored], 0.8, origin=cart, heading=0.0)
        assert held, "a tie with a thing on its right: the thing comes from its right, and goes"
    finally:
        engine.shutdown()
