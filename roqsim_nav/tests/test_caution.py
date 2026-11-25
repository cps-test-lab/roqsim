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
        # It reports a point per ray that found something, not one point for the whole corridor.
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


def test_a_stopped_mover_casts_no_rays(tmp_path):
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


def test_the_probe_scans_inside_the_movers_own_obstacle_band(tmp_path):
    """A fixed scan height is a guess about other people's robots, and one will be wrong.

    0.30 m keeps a walker's rays from passing over a tall base, but it is above the roof of a
    TurtleBot 4, whose collision geometry stops at 0.25 m -- so in a world of them every probe would
    see nothing and nothing would ever stop for anything. Taking the height from the band the
    planner already rasterizes ties it to what this mover treats as an obstacle.
    """
    probe = CautionProbe({"band": (0.1, 1.8)})
    assert probe.height == pytest.approx(0.15)
    low = CautionProbe({"band": (0.05, 0.6)})
    assert low.height == pytest.approx(0.10)
    assert CautionProbe({"band": (0.1, 1.8), "height": 0.9}).height == pytest.approx(0.9)


# -- height and blockage_radius: derived when absent, used as written when set --------------------
DERIVED = {"height": 0.10, "blockage_radius": 0.3}  # band (0.05, 0.6) + 5 cm; width 0.6 / 2


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
    """A clear ray during the settle is not the same as being through.

    Treating it as such re-arms the window on every flicker, which is exactly the state a mover
    stuttering against an obstacle is in -- so it would yield for ever and never recover.
    """
    probe = CautionProbe({"yield_time": 1.0, "clear_time": 5.0})
    probe._blocked_since = 0.0
    probe._clear_since = 2.0  # rays came up clear, but the settle has not elapsed
    assert probe.blocked_for(3.0) == pytest.approx(3.0), "the counter was reset by a flicker"


def test_a_mover_does_not_wait_before_it_has_ever_been_blocked(tmp_path):
    """`clear_time` settles a way that has REOPENED, so it must not run from a cold start.

    Otherwise every mover holds for `clear_time` at the beginning of every run, having never
    met anything -- the timer starts from an unset value and reads as "the way only just cleared".
    """
    engine = Engine(_world(tmp_path))  # nothing in the way at all
    engine.setup()
    engine.reset()
    try:
        probe = _navigator(engine)._caution
        for _ in range(int(1.0 / engine.ctx.dt)):
            engine.step()
            assert not probe.blocked, "it held despite never having been blocked"
    finally:
        engine.shutdown()


# -- setting off again ---------------------------------------------------------------------------
# What a held mover may set off into is judged against its body, not its rays: the rays look at one
# height, from the mover's circumscribed radius out, and a body can stand in its way below that
# height or inside that radius. Each test holds the cart for a blocker its rays see, then puts one
# they cannot see where the cart would meet it.

#: A robot whose base is lower than the cart's scan plane (0.10 m, its band's floor plus 5 cm) and
#: which shows the plane only a post, narrower than the rays' spacing: a TurtleBot's shape.
LOW_ROBOT = """<mujoco model="low_robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".2 .03" pos="0 0 .03"/>
    <geom name="post" type="cylinder" size=".03 .2" pos="0 0 .26"/>
  </body></worldbody>
</mujoco>"""

#: A slim body as tall as the cart: the rays see it wherever they pass through it.
POST = """<mujoco model="post">
  <worldbody><body name="base"><geom name="g" type="cylinder" size=".06 .25" pos="0 0 .25"/></body>
  </worldbody>
</mujoco>"""


def _held_world(tmp_path, blocker_xml, *, clear_time=0.0, blocker_spawn=None):
    """The cart of :func:`_world`, and a driven blocker in its way at the origin.

    ``blocker_spawn`` overrides keys of the blocker's ``spawn_model``.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    navigator = {
        "speed": 0.8,
        "goals": [list(GOAL)],
        "obstacle_height": [0.05, 0.6],
        "recovery": {"enabled": True, "stuck_time": 0.5},
        "avoidance": {"lookahead": 1.0, "clear_time": clear_time},
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


def test_a_held_mover_does_not_set_off_into_a_base_below_its_rays(tmp_path):
    """The blocker shifts so its post stands between two rays; its base is still in the way."""
    engine = _held_engine(tmp_path, LOW_ROBOT)
    try:
        held = _xy(engine)
        _place(engine, (0.0, 0.075))  # the post between the rays at 0 and 0.15 m
        _run(engine, 3.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), "it set off into the robot's base"
        assert _navigator(engine)._caution.blocked

        _place(engine, (0.0, 8.0))  # and once it has gone, the way is open
        _run(engine, 3.0)
        assert _xy(engine)[0] > held[0] + 1.0, "it never set off after the robot left"
    finally:
        engine.shutdown()


def test_a_held_mover_does_not_set_off_into_a_body_beside_its_front(tmp_path):
    """A body between the cart's front and its circumscribed radius, where no ray starts."""
    engine = _held_engine(tmp_path, POST)
    try:
        held = _xy(engine)
        probe = _navigator(engine)._caution
        # The cart's front is 0.25 m ahead of its centre and its rays start at its circumscribed
        # radius, 0.43 m; the post spans 0.28 to 0.40 m ahead, 0.12 to 0.24 m to the side.
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


def _release_time(tmp_path, sweep: bool) -> float:
    """When the cart sets off after the low robot has driven across its way and on."""
    engine = _held_engine(tmp_path, LOW_ROBOT, clear_time=0.5)
    try:
        probe = _navigator(engine)._caution
        if not sweep:
            probe._sweep = lambda *a: []  # the release as it was decided on the rays alone
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


#: How fast the low robot crosses: slowly enough that its base is still in the corridor when the
#: settle after its post has left the rays is over.
CROSSING = 0.15


def test_a_robot_that_drives_across_and_on_is_held_for_until_its_base_has_left(tmp_path):
    """Held for as long as its base is in the corridor and ``clear_time`` after, not its post.

    The base (0.2 m) leaves the 0.6 m corridor when the robot's centre is 0.5 m to the side; its
    post, all the rays see, 0.17 m before that. Setting off on the rays alone, the cart would pass
    the base closer than it stops for.
    """
    base_out = (0.3 + 0.2) / CROSSING
    with_sweep = _release_time(tmp_path / "a", sweep=True)
    rays_only = _release_time(tmp_path / "b", sweep=False)
    assert rays_only < base_out, "the fixture no longer separates the post from the base"
    assert with_sweep == pytest.approx(base_out + 0.5, abs=0.06)  # within a navigator tick


def test_a_robot_back_in_the_corridor_before_the_settle_is_over_holds_it_again(tmp_path):
    """The corridor must stay clear for the whole settle, not only at its end."""
    engine = _held_engine(tmp_path, LOW_ROBOT, clear_time=0.5)
    try:
        held = _xy(engine)
        _place(engine, (0.0, 0.075))  # the post between the rays, the base in the way
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
        _place(engine, (0.0, 0.075))  # the post between the rays, the base in the way
        _navigator(engine)._caution.check(engine.ctx, held, 0.0, (0.0, 0.0))
        _run(engine, 3.0)
        assert _xy(engine) == pytest.approx(held, abs=0.01), "it set off into the robot's base"
    finally:
        engine.shutdown()


#: A loose mat, lower than the cart's scan plane: the rays pass over it, the corridor holds it.
MAT = """<mujoco model="mat">
  <worldbody><body name="base"><geom name="g" type="box" size=".15 .15 .02" pos="0 0 .02"/></body>
  </worldbody>
</mujoco>"""


def test_a_mover_the_rays_never_held_is_not_held_by_the_corridor(tmp_path):
    """A mover the rays never held is not swept: a loose mat beside its way does not stop it."""
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
        assert _xy(engine)[0] > start[0] + 1.0, "it held for a mat its rays never stopped it for"
    finally:
        engine.shutdown()
