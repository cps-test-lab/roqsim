"""End-to-end plugin tests: patrol, the goal-route interface, and reset -- driven through the engine.

These are the in-process equivalent of what the ROS 2 ``NavigateThroughPoses`` handler does: send a
route via the blackboard :class:`~roqsim_walker.plugins.walker.WalkerHandle`, poll ``status()``
until it reports finished, and check the walker actually got there.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

WAYPOINTS = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0]]


def _world(*, name="pedestrian", **walker_overrides):
    walker = {
        "walker": "MaleVisitorWalk",
        "speed": 1.2,
        "loop": True,
        "arrival_radius": 0.25,
        "avoidance": False,
        "skin": False,  # capsules: keeps the test fast (no 5 MB OBJ load / skin rig)
        "waypoints": WAYPOINTS,
    }
    walker.update(walker_overrides)
    return load_config_from_dict(
        {
            "sim": {"headless": True, "pacing": "asap"},
            "components": [{"walker": walker, "name": name}],
        }
    )


@pytest.fixture(scope="module")
def _engine():
    engine = Engine(_world())
    engine.setup()
    yield engine
    engine.shutdown()


@pytest.fixture
def sim(_engine):
    """A fresh episode on one shared engine.

    Building the engine is cheap (~0.1 s); its FIRST ``reset()`` is not (~0.9 s -- the character
    meshes and the CARLA locomotion clips load lazily there). A second reset on the same engine is
    about a millisecond, so ten tests each building their own engine paid that boot ten times.
    ``reset()`` is the engine's episode boundary and the same call this fixture always made, so a
    test still starts from the route's start with every plugin's ``on_reset`` having run.
    """
    _engine.reset()
    return _engine


def _xy(engine):
    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pedestrian/pelvis")
    return data.mocap_pos[model.body_mocapid[bid]][:2].copy()


def _run(engine, seconds):
    for _ in range(max(1, int(seconds / engine.ctx.dt))):
        engine.step()


def _run_until(engine, predicate, timeout=40.0, tick=0.02):
    """Step until ``predicate()`` or ``timeout`` sim-seconds elapse. Returns whether it fired."""
    deadline = engine.ctx.sim_time + timeout
    while engine.ctx.sim_time < deadline:
        _run(engine, tick)
        if predicate():
            return True
    return False


# -- registration ------------------------------------------------------------------------------
def test_plugin_registers_entity_handle_and_goal_endpoint(sim):
    ctx = sim.ctx
    entity = ctx.entities.get("pedestrian")
    assert entity is not None and entity.kind == "pedestrian"
    assert entity.body == "pedestrian/pelvis"

    assert ctx.blackboard.get("walker:pedestrian") is not None

    # The walker itself registers the body_poses TF stream; its navigator registers the goal
    # interface, as it does for a robot or a prop -- which is why there are now TWO goal endpoints
    # rather than one. `navigate_through_poses` is unchanged in name and type, so an existing client
    # and an existing world are unaffected; `navigate_to_pose` is new surface a walker never had.
    endpoints = {e.name: e for e in ctx.interface.all() if e.owner == "pedestrian"}
    assert set(endpoints) == {"body_poses", "navigate_through_poses", "navigate_to_pose"}
    assert endpoints["body_poses"].direction == "out"

    through = endpoints["navigate_through_poses"]
    assert through.direction == "in"
    assert through.backend["ros2"]["action"] == "nav2_msgs.action.NavigateThroughPoses"
    assert through.backend["ros2"]["name"] == "navigate_through_poses"

    single = endpoints["navigate_to_pose"]
    assert single.direction == "in"
    assert single.backend["ros2"]["action"] == "nav2_msgs.action.NavigateToPose"


def test_walker_spawns_at_the_first_waypoint(sim):
    np.testing.assert_allclose(_xy(sim), WAYPOINTS[0], atol=1e-6)


# -- patrol ------------------------------------------------------------------------------------
def test_walker_patrols_toward_its_next_waypoint(sim):
    _run(sim, 2.0)
    pos = _xy(sim)
    assert pos[0] > -1.0, "should have walked east along the first leg"
    assert pos[1] == pytest.approx(-2.0, abs=0.2)


def test_reset_returns_the_walker_to_the_route_start(sim):
    _run(sim, 2.0)
    assert _xy(sim)[0] > -1.5
    sim.reset()
    np.testing.assert_allclose(_xy(sim), WAYPOINTS[0], atol=1e-6)


# -- goal route (what the NavigateThroughPoses handler drives) ---------------------------------
def test_send_route_overrides_patrol_and_reports_arrival(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    goals = [(0.0, 2.0), (-2.0, 0.0)]

    seq = handle.send_route(goals)
    assert seq >= 1
    _run(sim, 0.05)  # let the posted command reach the physics thread
    applied_seq, finished, goals_left, _ = handle.status()
    assert (applied_seq, finished, goals_left) == (seq, False, 2)

    arrived = _run_until(sim, lambda: handle.status()[1])
    assert arrived, "walker never reported finishing its route"

    applied_seq, finished, goals_left, dist = handle.status()
    assert (applied_seq, finished, goals_left, dist) == (seq, True, 0, 0.0)
    # It really is at the final pose (within its arrival radius).
    assert np.linalg.norm(_xy(sim) - np.array(goals[-1])) < 0.26


def test_route_feedback_counts_goals_down(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    handle.send_route([(0.0, 2.0), (-2.0, 0.0)])
    _run(sim, 0.05)

    seen = []
    _run_until(sim, lambda: seen.append(handle.status()[2]) or handle.status()[1])
    assert seen[0] == 2
    assert 1 in seen, "the first goal should be retired before the second"
    assert seen[-1] == 0


def test_patrol_resumes_after_the_route_completes(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    handle.send_route([(0.0, 0.0)])
    _run(sim, 0.05)
    assert _run_until(sim, lambda: handle.status()[1]), "route never finished"

    at_goal = _xy(sim)
    _run(sim, 2.0)
    assert np.linalg.norm(_xy(sim) - at_goal) > 0.5, "walker should resume patrolling, not stand"


def test_cancel_route_stops_the_walker(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    handle.send_route([(2.0, 2.0)])
    _run(sim, 1.0)

    seq = handle.cancel_route()
    _run(sim, 0.3)
    applied_seq, finished, _, _ = handle.status()
    assert (applied_seq, finished) == (seq, True)

    stopped_at = _xy(sim)
    _run(sim, 1.5)
    assert np.linalg.norm(_xy(sim) - stopped_at) < 0.05, "walker kept moving after cancel"


def test_a_newer_route_supersedes_an_older_one(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    first = handle.send_route([(2.0, 2.0)])
    _run(sim, 0.5)
    second = handle.send_route([(-2.0, -2.0)])
    _run(sim, 0.05)

    applied_seq, finished, _, _ = handle.status()
    assert applied_seq == second > first
    assert not finished
    # The handler for `first` sees a larger seq and aborts its goal.


# -- goal-driven only (no patrol) --------------------------------------------------------------
def test_walker_without_waypoints_stands_at_pos_until_commanded():
    engine = Engine(_world(waypoints=[], pos=[1.0, 1.0]))
    engine.setup()
    engine.reset()
    try:
        np.testing.assert_allclose(_xy(engine), [1.0, 1.0], atol=1e-6)
        _run(engine, 2.0)
        np.testing.assert_allclose(_xy(engine), [1.0, 1.0], atol=0.02)  # stands

        handle = engine.ctx.blackboard.get("walker:pedestrian")
        handle.send_route([(-1.0, 1.0)])
        _run(engine, 0.05)
        assert _run_until(engine, lambda: handle.status()[1]), "route never finished"
        assert np.linalg.norm(_xy(engine) - np.array([-1.0, 1.0])) < 0.26
    finally:
        engine.shutdown()


# -- clearance to an articulated obstacle ------------------------------------


def test_clearance_measures_the_nearest_limb_not_the_walker_origin(tmp_path):
    """A pedestrian is not a point with a radius around it.

    Its nearest part is whichever limb happens to be extended, and a metric that reduced
    it to an origin plus a circle would report the wrong distance in both directions: too
    far when an arm is reaching toward the robot, too near when the walker is turned away.
    `clearance_monitor` measures geom to geom, so the limb is what it finds -- and the
    walker's six render-only geoms are excluded, or it would report clearance to
    decoration the robot passes straight through.
    """
    import mujoco

    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    scene = tmp_path / "s.xml"
    scene.write_text(
        '<mujoco><worldbody><geom name="floor" type="plane" size="10 10 .1"/></worldbody></mujoco>'
    )
    rover = tmp_path / "rover.xml"
    rover.write_text(
        '<mujoco model="rover"><worldbody><body name="base" pos="0 0 .2">'
        '<geom name="base_geom" type="cylinder" size=".2 .2"/>'
        "</body></worldbody></mujoco>"
    )

    world = {
        "sim": {"world": str(scene)},
        "components": [
            {
                "spawn_model": {"model": str(rover), "pos": [0.0, 0.0], "free": True},
                "name": "robot",
                "components": [{"clearance_monitor": {"ignore": ["floor"], "distmax": 8.0}}],
            },
            {
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "speed": 0.0,
                    "waypoints": [[1.0, 0.0]],
                    "avoidance": False,
                },
                "name": "pedestrian",
            },
        ],
    }
    engine = Engine(load_config_from_dict(world))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    try:
        engine.step()
        report = engine.ctx.blackboard.get("clearance:robot.clearance_monitor")()

        model = engine.ctx.model
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, report.geom)
        if gid < 0:  # unnamed limb geoms report as geom<id>
            gid = int(report.geom.removeprefix("geom"))
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""

        # It found a part of the pedestrian, and that part is collidable rather than skin.
        assert "pedestrian" in body, f"measured to {body!r}, not the walker"
        assert int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]), (
            "measured to a render-only geom"
        )
        # A body-part distance, not the ~1.0 m to the walker's origin.
        assert 0.0 < report.current < 1.0
    finally:
        engine.shutdown()


def test_a_world_may_write_the_walkers_navigator_itself(tmp_path):
    """The escape hatch from the compatibility expansion, for the navigator's newer options.

    It builds the humanoid exactly once. `expand` contributes entries *beside* the walker, and the
    caller keeps the walker -- so returning it from the branch that steps aside built the skeleton
    twice and MuJoCo refused the duplicate body names.
    """
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap"},
                "components": [
                    {
                        "walker": {"walker": "MaleVisitorWalk", "skin": False, "pos": [0.0, 0.0]},
                        "name": "pedestrian",
                        "components": [
                            {
                                "navigator": {
                                    "output": "walker",
                                    "speed": 1.0,
                                    "goals": [[2.0, 0.0]],
                                    # A walker's own default is to look ahead at nothing.
                                    "avoidance": {"stop": True},
                                }
                            }
                        ],
                    }
                ],
            }
        )
    )
    engine.setup()
    engine.reset()
    try:
        navigator = next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")
        assert navigator._caution.enabled, "the world's own policy did not take effect"
        for _ in range(int(6.0 / engine.ctx.dt)):
            engine.step()
        assert np.linalg.norm(_xy(engine) - np.array([2.0, 0.0])) < 0.4
    finally:
        engine.shutdown()


def test_writing_navigation_in_both_places_is_refused(tmp_path):
    """One place or the other, so a reader does not have to guess which one won."""
    with pytest.raises(PluginError, match="both in its own block and in a nested"):
        Engine(
            load_config_from_dict(
                {
                    "sim": {},
                    "components": [
                        {
                            "walker": {"walker": "MaleVisitorWalk", "speed": 1.0},
                            "name": "pedestrian",
                            "components": [{"navigator": {"output": "walker", "speed": 2.0}}],
                        }
                    ],
                }
            )
        )


# -- the per-waypoint dwell reaches the navigator --------------------------------------------------


def _walker_navigator(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")


def _engine_with_waypoints(waypoints, **walker):
    cfg = {"walker": "MaleVisitorWalk", "skin": False, "speed": 1.0, "loop": True}
    cfg["waypoints"] = waypoints
    cfg.update(walker)
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"}, "components": [{"walker": cfg, "name": "pedestrian"}]}
        )
    )
    engine.ctx.seed = 5  # before setup: the navigator draws its generator in `configure`
    return engine


def test_a_per_waypoint_dwell_reaches_the_navigator():
    """`[x, y, dwell]` is what the walker's config block documents and its validator accepts.

    It used to be truncated to `(x, y)` on the way to the navigator, so a world asking for a pause
    got none and the crowd simply never stopped walking -- with no warning, because nothing had
    rejected the value. It cannot ride along as a goal's third element (that position is the goal's
    yaw), so it travels as the navigator's own `dwell`.
    """
    engine = _engine_with_waypoints([[0.0, 0.0, [0.0, 3.0]], [2.0, 0.0, [1.0, 2.0]]])
    engine.setup()
    engine.reset()
    # One entry per route point, in order, starting with where the walker stands.
    assert _walker_navigator(engine)._core.st.dwell == [(0.0, 3.0), (1.0, 2.0)]


def test_a_dwell_on_only_some_waypoints_reaches_the_navigator():
    """The shape the shipped `walker_patrol` world writes: a pause at two of four waypoints. The
    per-point list is then a MIX of bare numbers and pairs, which is what broke that world."""
    engine = _engine_with_waypoints(
        [[-2.0, -2.0], [2.0, -2.0, [2.0, 4.0]], [2.0, 2.0], [-2.0, 2.0, [1.0, 3.0]]]
    )
    engine.setup()
    engine.reset()
    assert _walker_navigator(engine)._core.st.dwell == [
        (0.0, 0.0),
        (2.0, 4.0),
        (0.0, 0.0),
        (1.0, 3.0),
    ]


def test_two_bare_per_waypoint_dwells_are_not_read_as_one_random_pause():
    """The navigator reads two bare numbers as `[lo, hi]`; the walker knows its values are
    per-waypoint, so it normalises them to pairs and never relies on that tie-break."""
    engine = _engine_with_waypoints([[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]])
    engine.setup()
    engine.reset()
    assert _walker_navigator(engine)._core.st.dwell == [(1.0, 1.0), (3.0, 3.0)]


def test_the_walkers_default_dwell_applies_to_every_waypoint():
    """`dwell:` on the walker block is the other half of the same feature."""
    engine = _engine_with_waypoints([[0.0, 0.0], [2.0, 0.0]], dwell=1.5)
    engine.setup()
    engine.reset()
    assert _walker_navigator(engine)._core.st.dwell == [(1.5, 1.5), (1.5, 1.5)]


def test_a_walker_with_no_dwell_says_nothing_about_it():
    """The common case must not start carrying a dwell of zeros into every navigator's config."""
    engine = _engine_with_waypoints([[0.0, 0.0], [2.0, 0.0]])
    engine.setup()
    engine.reset()
    assert _walker_navigator(engine)._core.st.dwell == [(0.0, 0.0), (0.0, 0.0)]
