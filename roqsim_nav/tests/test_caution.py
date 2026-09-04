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


def _world(tmp_path, *, blocker=None, nav=None, blocker_mocap=True):
    """The mover, plus optionally a second prop sitting in its way at the origin."""
    crate = _write(tmp_path, "crate.xml", CRATE)
    navigator = {
        "speed": 0.8,
        "goals": [list(GOAL)],
        "obstacle_height": [0.05, 0.6],
        # ON: these tests are about the interaction, not about avoiding it.
        "recovery": {"enabled": True, "stuck_time": 0.5},
        "caution": {"lookahead": 1.0, "clear_time": 0.0},
    }
    navigator.update(nav or {})
    components = [
        {
            "spawn_model": {"model": crate, "pos": [START[0], START[1], 0.25], "mocap": True},
            "name": "cart",
            "components": [{"navigator": navigator}],
        }
    ]
    if blocker is not None:
        components.append(
            {
                "spawn_model": {
                    "model": crate,
                    "prefix": "b_",
                    "pos": [blocker[0], blocker[1], 0.25],
                    **({"mocap": True} if blocker_mocap else {"free": True}),
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
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), blocker_mocap=False))
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
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"caution": {"ignore": ["blocker"]}}))
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
    engine = Engine(_world(tmp_path, blocker=(0.0, 0.0), nav={"caution": {"enabled": False}}))
    engine.setup()
    engine.reset()
    try:
        _run(engine, 12.0)
        assert _xy(engine)[0] > -0.5, "something other than caution was holding it back"
    finally:
        engine.shutdown()


# -- configuration -------------------------------------------------------------------------------
def test_replan_is_refused_by_name_rather_than_ignored(tmp_path):
    with pytest.raises(PluginError, match="not implemented"):
        Engine(_world(tmp_path, nav={"caution": {"on_blocked": "replan"}}))


def test_an_unknown_on_blocked_is_refused(tmp_path):
    with pytest.raises(PluginError, match="on_blocked"):
        Engine(_world(tmp_path, nav={"caution": {"on_blocked": "swerve"}}))


def test_ignoring_an_entity_that_does_not_exist_is_refused(tmp_path):
    """Raised at the first reset, not at setup: `ignore` may name an entity declared later."""
    engine = Engine(_world(tmp_path, nav={"caution": {"ignore": ["ghost"]}}))
    engine.setup()
    with pytest.raises(RuntimeError, match="not an entity"):
        engine.reset()


def test_a_stopped_mover_casts_no_rays(tmp_path):
    """Cost: nothing to run into if we are not going anywhere, and no direction to cast along."""
    probe = CautionProbe({})
    assert probe.check(None, (0.0, 0.0), 0.25, np.zeros(2)) is False
