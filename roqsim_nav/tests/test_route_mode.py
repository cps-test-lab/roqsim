"""`route_mode` and `autostart`: a route given exactly, or planned, and when it begins.

Two independent knobs on the same route, both there so that a *scripted* mover is the same plugin as
a goal-driven one rather than a second mechanism beside it.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

DATA = Path(__file__).parent / "data"
ROOM = DATA / "divided_room.xml"  # a divider spanning y in [-2.5, 2.5] at x = 0
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""

START, GOAL = (-2.5, 0.0), (2.5, 0.0)


def _engine(tmp_path, **nav):
    crate = tmp_path / "crate.xml"
    crate.write_text(CRATE)
    navigator = {"speed": 0.8, "goals": [list(GOAL)], "obstacle_height": [0.05, 0.6]}
    navigator.update(nav)
    return Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap", "world": str(ROOM)},
                "components": [
                    {
                        "spawn_model": {
                            "model": str(crate),
                            "pos": [START[0], START[1], 0.25],
                            "mocap": True,
                        },
                        "name": "cart",
                        "components": [{"navigator": navigator}],
                    }
                ],
            },
            base_dir=tmp_path,
        )
    )


def _xy(engine):
    model = engine.ctx.model
    body = engine.ctx.entities.get("cart").body
    mid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
    return engine.ctx.data.mocap_pos[mid][:2].copy()


def _track(engine, seconds=14.0, every=50):
    out = []
    for i in range(int(seconds / engine.ctx.dt)):
        engine.step()
        if i % every == 0:
            out.append(_xy(engine))
    return np.array(out)


def _navigator(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")


# -- route_mode ----------------------------------------------------------------------------------
def test_plan_routes_around_the_divider(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    try:
        track = _track(engine)
        assert np.abs(track[:, 1]).max() > 2.5, "did not route around the wall"
    finally:
        engine.shutdown()


def test_exact_follows_the_given_polyline_through_the_wall(tmp_path):
    """The given points ARE the path: no search, so it does not route around anything.

    Driving straight through a wall is the *point* of the assertion, not an accident -- it is the
    sharpest available evidence that no planner ran. A mocap mover passes through geometry, so this
    is observable rather than a collision.
    """
    engine = _engine(tmp_path, route_mode="exact")
    engine.setup()
    engine.reset()
    try:
        track = _track(engine)
        assert np.abs(track[:, 1]).max() < 0.05, "it deviated -- something planned"
        assert np.linalg.norm(track[-1] - np.asarray(GOAL)) < 0.3
    finally:
        engine.shutdown()


def _max_offset_from_polyline(track):
    """How far the track ever strays from the two legs x = -2.5 and y = 2.0."""
    return float(np.minimum(np.abs(track[:, 0] + 2.5), np.abs(track[:, 1] - 2.0)).max())


@pytest.mark.parametrize(("arrival_radius", "tolerance"), [(0.25, 0.25), (0.05, 0.06)])
def test_exact_walks_the_polyline_leg_by_leg(tmp_path, arrival_radius, tolerance):
    """It stays on the given segments, to within the radius at which it switches to the next one.

    `exact` means **no planner** -- nothing routes around anything -- but it does not mean zero
    corner-cutting. The follower chases the active goal and advances once inside `arrival_radius`,
    so a corner is rounded by about that much. That is a property of chasing waypoints at a finite
    speed, and `arrival_radius` is the lever (measured: 0.25 -> 0.23 m of corner, 0.05 -> 0.04 m).

    `planner.waypoint_radius` is NOT the lever here, which is worth writing down because it reads
    like it should be: in `exact` mode the path is a single point, so it is always the *last* one and
    the follower uses `arrival_radius` for it. A tracker that bounded the error independently of
    either would have to follow the path rather than its endpoints -- see `test_pure_pursuit.py`.
    """
    engine = _engine(
        tmp_path, route_mode="exact", goals=[[-2.5, 2.0], [2.5, 2.0]], arrival_radius=arrival_radius
    )
    engine.setup()
    engine.reset()
    try:
        offset = _max_offset_from_polyline(_track(engine, seconds=18.0))
        assert offset <= tolerance, f"strayed {offset:.3f} m from the polyline"
    finally:
        engine.shutdown()


def test_a_tighter_arrival_radius_tracks_the_corner_more_closely(tmp_path):
    """The lever above, as a comparison so it cannot pass by both cases simply being loose."""
    offsets = {}
    for radius in (0.25, 0.05):
        engine = _engine(
            tmp_path, route_mode="exact", goals=[[-2.5, 2.0], [2.5, 2.0]], arrival_radius=radius
        )
        engine.setup()
        engine.reset()
        try:
            offsets[radius] = _max_offset_from_polyline(_track(engine, seconds=18.0))
        finally:
            engine.shutdown()
    assert offsets[0.05] < offsets[0.25] / 2.0


def test_exact_refuses_recovery(tmp_path):
    """Backing up and re-planning is, by definition, leaving the path that was given."""
    with pytest.raises(PluginError, match="route_mode: exact"):
        _engine(tmp_path, route_mode="exact", recovery={"enabled": True})


def test_an_unknown_route_mode_is_refused(tmp_path):
    with pytest.raises(PluginError, match="route_mode"):
        _engine(tmp_path, route_mode="spline")


# -- autostart -----------------------------------------------------------------------------------
def test_autostart_false_holds_at_the_start_until_started(tmp_path):
    """The route is armed and planned; only the trigger is missing.

    This is what lets a world own the trajectory -- so it is identical in every repetition and shows
    up in a campaign's config diff -- while the scenario owns only its timing.
    """
    engine = _engine(tmp_path, autostart=False)
    engine.setup()
    engine.reset()
    try:
        nav = _navigator(engine)
        assert not nav.started
        _track(engine, seconds=5.0)
        assert _xy(engine) == pytest.approx(START, abs=1e-9), "it moved without being started"

        # `start()` returns immediately with a sequence number and marshals the change onto the
        # physics thread, so it takes effect on the next step -- the single-writer rule, not a
        # delay to work around.
        seq = nav.start()
        assert seq > 0
        _track(engine, seconds=0.1)
        assert nav.started
        assert np.linalg.norm(_track(engine)[-1] - np.asarray(GOAL)) < 0.3
    finally:
        engine.shutdown()


def test_the_route_is_planned_before_anything_triggers_it(tmp_path):
    """An armed route is planned on the first tick, not when it is started.

    So a route the planner cannot solve surfaces at the start of the episode rather than at the
    moment a scenario finally triggers it, minutes in. Planning is deliberately NOT done in
    `configure`: that runs before any presence has been applied, so the grid would contain the
    obstacles a world compiled in precisely so they could appear mid-trial, and the mover would route
    around things nothing can see or touch.
    """
    engine = _engine(tmp_path, autostart=False)
    engine.setup()
    engine.reset()
    try:
        nav = _navigator(engine)
        assert nav._core.planner is None, "planned before presence could be applied"
        engine.step()
        assert nav._core.planner is not None, "not planned until something started it"
        assert not nav.started, "planning should not have started the route"
    finally:
        engine.shutdown()


def test_reset_re_arms_an_unstarted_route(tmp_path):
    """Episode two must not inherit episode one's trigger."""
    engine = _engine(tmp_path, autostart=False)
    engine.setup()
    engine.reset()
    try:
        nav = _navigator(engine)
        nav.start()
        _track(engine, seconds=3.0)
        assert np.linalg.norm(_xy(engine) - np.asarray(START)) > 0.5

        engine.reset()
        assert not nav.started, "it stayed started into the next episode"
        _track(engine, seconds=3.0)
        assert _xy(engine) == pytest.approx(START, abs=1e-9)
    finally:
        engine.shutdown()


def test_starting_an_already_running_route_does_nothing(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    try:
        nav = _navigator(engine)
        assert nav.started
        _track(engine, seconds=3.0)
        moved = _xy(engine)
        nav.start()  # already running: a no-op returning the live sequence
        _track(engine, seconds=0.5)
        assert np.linalg.norm(_xy(engine) - moved) > 0.05, "start() restarted or stalled it"
    finally:
        engine.shutdown()


# -- tracker -------------------------------------------------------------------------------------
# These assert that pure pursuit *works* -- that goals advance, routes finish and laps close under a
# tracker that deliberately does not drive at its goals. They do NOT assert that it tracks better
# than the waypoint follower, because on this embodiment it does not: a mocap body's pose is written
# rather than steered, so there is no steering constraint for pure pursuit to respect and the simple
# follower is already near-optimal. Measured, round a right-angle corner: pure pursuit cuts by about
# its lookahead (0.27 m at 0.15, 0.85 m at 1.0), the chaser by its arrival radius (0.04 m at 0.05).
# The tracker earns its place on a base with wheels; see the navigator's TRACKERS docstring.


def _run_route(tmp_path, seconds=20.0, **nav):
    engine = _engine(tmp_path, route_mode="exact", goals=[[-2.5, 2.0], [2.5, 2.0]], **nav)
    engine.setup()
    engine.reset()
    try:
        track = _track(engine, seconds=seconds)
        core = _navigator(engine)._core
        return track, core.st.done, core.st.goal_idx
    finally:
        engine.shutdown()


def test_pure_pursuit_finishes_the_route_and_reports_done(tmp_path):
    """The end-to-end regression for two silent bugs, both of which let the mover keep moving.

    A follower that advances goals on *proximity* stalls at the corner, because pure pursuit does
    not drive at the corner. Fixing that with a "have I crossed the plane through the goal" test
    fails differently and more quietly: a tracker that rounds a corner passes *inside* it, so it
    never crosses that plane -- ``goal_idx`` freezes while the carrot drags the mover along the whole
    route, which looks correct from outside and never reports arrival. Only arc-length progress is
    monotone under cutting.
    """
    track, done, goal_idx = _run_route(tmp_path, tracker="pure_pursuit", lookahead=0.6)
    assert done, "the route never completed -- goal advancement is stuck"
    assert goal_idx == 2, f"goals did not advance one at a time (ended on {goal_idx})"
    assert np.linalg.norm(track[-1] - np.asarray([2.5, 2.0])) < 0.3


def test_pure_pursuit_advances_each_goal_once_and_in_order(tmp_path):
    """Guards the other direction: arc-length progress measured from the route's own start makes
    every goal read as reached on the first tick, and the mover skips straight to the last."""
    engine = _engine(
        tmp_path,
        route_mode="exact",
        goals=[[-2.5, 2.0], [2.5, 2.0]],
        tracker="pure_pursuit",
        lookahead=0.6,
    )
    engine.setup()
    engine.reset()
    try:
        core = _navigator(engine)._core
        seen, t_first_advance = [core.st.goal_idx], None
        for _ in range(int(20.0 / engine.ctx.dt)):
            engine.step()
            if core.st.goal_idx != seen[-1]:
                seen.append(core.st.goal_idx)
                t_first_advance = t_first_advance or engine.ctx.sim_time
        assert seen == [1, 2], f"goal sequence was {seen}"
        assert t_first_advance > 1.0, "the first goal was reached instantly -- progress reads zero"
    finally:
        engine.shutdown()


def test_pure_pursuit_tolerates_a_tiny_arrival_radius(tmp_path):
    """A proximity-based follower needs the mover to come within `arrival_radius` of every goal. A
    2 cm radius would therefore stall it at the corner; arc progress does not care."""
    _, done, _ = _run_route(tmp_path, tracker="pure_pursuit", lookahead=0.6, arrival_radius=0.02)
    assert done


def test_pure_pursuit_corner_error_scales_with_the_lookahead(tmp_path):
    """Its one real knob behaves as the geometry says: a longer carrot cuts a wider corner."""
    corners = {
        la: _max_offset_from_polyline(_run_route(tmp_path, tracker="pure_pursuit", lookahead=la)[0])
        for la in (0.15, 0.6)
    }
    assert corners[0.15] < corners[0.6]


def test_a_looping_route_wraps_the_carrot_onto_its_own_start(tmp_path):
    """Otherwise the mover stops dead at the end of a lap and turns on the spot."""
    engine = _engine(
        tmp_path,
        route_mode="exact",
        goals=[[-2.5, 2.0], [2.5, 2.0], [2.5, 0.0]],
        loop=True,
        tracker="pure_pursuit",
        lookahead=0.4,
    )
    engine.setup()
    engine.reset()
    try:
        track = _track(engine, seconds=40.0)
        left = float(np.linalg.norm(track - np.asarray(START), axis=1).max()) > 2.0
        returned = float(np.linalg.norm(track[-60:] - np.asarray(START), axis=1).min()) < 1.2
        assert left and returned, "the lap did not close"
    finally:
        engine.shutdown()


def test_lookahead_without_pure_pursuit_is_refused(tmp_path):
    """Silently ignoring a knob that does nothing is what a validator exists to prevent."""
    with pytest.raises(PluginError, match="no meaning with tracker: waypoint"):
        _engine(tmp_path, lookahead=0.5)


def test_an_unknown_tracker_is_refused(tmp_path):
    with pytest.raises(PluginError, match="tracker"):
        _engine(tmp_path, tracker="stanley")
