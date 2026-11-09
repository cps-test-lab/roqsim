"""End-to-end plugin tests: patrol, the goal-route interface, and reset -- driven through the engine.

These are the in-process equivalent of what the ROS 2 ``NavigateThroughPoses`` handler does: send a
route through the walker's navigator handle (``nav:<name>:handle``), poll ``status()`` until it
reports finished, and check the walker actually got there.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

WAYPOINTS = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0]]

#: The patrol: the walker stands at the first waypoint facing the second, and its navigator cycles
#: through the rest.
START = {"position": {"x": -2.0, "y": -2.0}, "orientation": {"yaw": 0.0}}
PATROL = {
    "output": "walker",
    "speed": 1.2,
    "loop": True,
    "arrival_radius": 0.25,
    "avoidance": {"steer": "none", "stop": False},
    "goals": WAYPOINTS[1:],
}


def _world(*, pose=START, navigator=PATROL):
    """A walker at ``pose``; with ``navigator=None`` it nests none and gets the default one."""
    entry = {
        # capsules: keeps the test fast (no 5 MB OBJ load / skin rig)
        "walker": {"walker": "MaleVisitorWalk", "skin": False, "pose": pose},
        "name": "pedestrian",
    }
    if navigator is not None:
        entry["components"] = [{"navigator": navigator}]
    return load_config_from_dict({"sim": {"pacing": "asap"}, "components": [entry]})


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
    about a millisecond, so ten tests each building their own engine would pay that boot ten times.
    ``reset()`` is the engine's episode boundary, so a test still starts from the route's start with
    every plugin's ``on_reset`` having run.
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
    # interface, as it does for a robot or a prop -- which is why there are TWO goal endpoints.
    # `navigate_through_poses` keeps the name and type a walker client expects; `navigate_to_pose`
    # is the navigator's single-goal surface.
    # `start_route` is a ROS action because this walker has a patrol: it runs the configured route.
    # `route_status` and `cancel_route` follow and stop a route by its sequence number.
    endpoints = {e.name: e for e in ctx.interface.all() if e.owner == "pedestrian"}
    assert set(endpoints) == {
        "body_poses",
        "navigate_through_poses",
        "navigate_to_pose",
        "start_route",
        "route_status",
        "cancel_route",
        "play",
        "stop_clip",
        "look_at",
        "clips",
    }
    assert endpoints["body_poses"].direction == "out"
    assert (
        endpoints["start_route"].backend["ros2"]["action"]
        == "roqsim_nav_interfaces.action.StartRoute"
    )

    through = endpoints["navigate_through_poses"]
    assert through.direction == "in"
    assert through.backend["ros2"]["action"] == "nav2_msgs.action.NavigateThroughPoses"
    assert through.backend["ros2"]["name"] == "navigate_through_poses"

    single = endpoints["navigate_to_pose"]
    assert single.direction == "in"
    assert single.backend["ros2"]["action"] == "nav2_msgs.action.NavigateToPose"


def test_walker_spawns_at_its_pose(sim):
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
def test_a_route_sent_to_its_navigator_overrides_patrol_and_reports_arrival(sim):
    handle = sim.ctx.blackboard.get("nav:pedestrian:handle")
    goals = [(0.0, 2.0), (-2.0, 0.0)]

    seq = handle.send_goals(goals)
    assert seq >= 1
    _run(sim, 0.05)  # let the posted command reach the physics thread
    applied_seq, finished, goals_left, _, _ = handle.status()
    assert (applied_seq, finished, goals_left) == (seq, False, 2)

    arrived = _run_until(sim, lambda: handle.status()[1])
    assert arrived, "walker never reported finishing its route"

    applied_seq, finished, goals_left, dist, error = handle.status()
    assert (applied_seq, finished, goals_left, dist, error) == (seq, True, 0, 0.0, "")
    # It really is at the final pose (within its arrival radius).
    assert np.linalg.norm(_xy(sim) - np.array(goals[-1])) < 0.26


def test_route_feedback_counts_goals_down(sim):
    handle = sim.ctx.blackboard.get("nav:pedestrian:handle")
    handle.send_goals([(0.0, 2.0), (-2.0, 0.0)])
    _run(sim, 0.05)

    seen = []
    _run_until(sim, lambda: seen.append(handle.status()[2]) or handle.status()[1])
    assert seen[0] == 2
    assert 1 in seen, "the first goal should be retired before the second"
    assert seen[-1] == 0


def test_patrol_resumes_after_the_route_completes(sim):
    handle = sim.ctx.blackboard.get("nav:pedestrian:handle")
    handle.send_goals([(0.0, 0.0)])
    _run(sim, 0.05)
    assert _run_until(sim, lambda: handle.status()[1]), "route never finished"

    at_goal = _xy(sim)
    _run(sim, 2.0)
    assert np.linalg.norm(_xy(sim) - at_goal) > 0.5, "walker should resume patrolling, not stand"


def test_cancelling_its_route_stops_the_walker(sim):
    handle = sim.ctx.blackboard.get("nav:pedestrian:handle")
    handle.send_goals([(2.0, 2.0)])
    _run(sim, 1.0)

    seq = handle.cancel()
    _run(sim, 0.3)
    applied_seq, finished, _, _, _ = handle.status()
    assert (applied_seq, finished) == (seq, True)

    stopped_at = _xy(sim)
    _run(sim, 1.5)
    assert np.linalg.norm(_xy(sim) - stopped_at) < 0.05, "walker kept moving after cancel"


def test_a_newer_route_supersedes_an_older_one(sim):
    handle = sim.ctx.blackboard.get("nav:pedestrian:handle")
    first = handle.send_goals([(2.0, 2.0)])
    _run(sim, 0.5)
    second = handle.send_goals([(-2.0, -2.0)])
    _run(sim, 0.05)

    applied_seq, finished, _, _, _ = handle.status()
    assert applied_seq == second > first
    assert not finished
    # The handler for `first` sees a larger seq and aborts its goal.


# -- goal-driven only (no patrol) --------------------------------------------------------------
def test_a_walker_with_the_default_navigator_stands_at_its_pose_until_commanded():
    engine = Engine(_world(pose={"position": {"x": 1.0, "y": 1.0}}, navigator=None))
    engine.setup()
    engine.reset()
    try:
        np.testing.assert_allclose(_xy(engine), [1.0, 1.0], atol=1e-6)
        _run(engine, 2.0)
        np.testing.assert_allclose(_xy(engine), [1.0, 1.0], atol=0.02)  # stands

        handle = engine.ctx.blackboard.get("nav:pedestrian:handle")
        handle.send_goals([(-1.0, 1.0)])
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
                "spawn_model": {
                    "model": str(rover),
                    "motion": "physics",
                    "pose": {"position": {"x": 0.0, "y": 0.0}},
                },
                "name": "robot",
                "components": [{"clearance_monitor": {"ignore": ["floor"], "distmax": 8.0}}],
            },
            {
                "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": 1.0, "y": 0.0}}},
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


def test_a_nested_navigator_replaces_the_default_one(tmp_path):
    """It builds the humanoid exactly once. `expand` contributes entries *beside* the walker, and
    the caller keeps the walker -- so returning it from the branch that steps aside would build the
    skeleton twice, and MuJoCo refuses the duplicate body names."""
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap"},
                "components": [
                    {
                        "walker": {
                            "walker": "MaleVisitorWalk",
                            "skin": False,
                            "pose": {"position": {"x": 0.0, "y": 0.0}},
                        },
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


def _walker_navigator(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")


def test_the_default_navigator_is_goal_driven_and_never_stops_the_walker():
    """What a walker without a nested navigator gets: no route, its own disc, and an avoidance
    that never stops it, since a walker does not look ahead."""
    from roqsim_walker.plugins.walker import WalkerPlugin

    engine = Engine(_world(navigator=None))
    engine.setup()
    engine.reset()
    try:
        config = _walker_navigator(engine).config
        assert {key: config.get(key) for key in WalkerPlugin.DEFAULT_NAVIGATOR} == (
            WalkerPlugin.DEFAULT_NAVIGATOR
        )
        assert "goals" not in config
        assert not _walker_navigator(engine)._caution.enabled
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("key", "value", "there"),
    [
        ("speed", 1.2, r"navigator's 'speed'"),
        ("waypoints", [[0.0, 0.0], [1.0, 0.0]], r"navigator's 'goals', with the walker starting"),
        ("dwell", 1.5, r"navigator's 'dwell'"),
        ("avoidance", True, r"\{steer: give_way\} for true"),
        ("orca", {"radius": 0.3}, r"navigator's 'radius' and 'max_speed'"),
        ("action_name", "go", r"action_names"),
    ],
)
def test_a_navigation_key_on_the_walker_is_refused_naming_the_navigators(key, value, there):
    """Where a walker goes is stated once, on its navigator; the walker's block names where."""
    from roqsim_walker.plugins.walker import WalkerPlugin

    config = {"walker": "MaleVisitorWalk", key: value}
    errors = WalkerPlugin(config).validate_config(config)
    assert any(re.search(rf"'{key}' is not read.*{there}", e) for e in errors), errors


# -- the start is a pose ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expect"),
    [
        (
            {"pos": [1.0, 2.0]},
            r"'pos' is not read -- a walker's start is stated as 'pose'.*"
            r"pose: \{position: \{x: 1\.0, y: 2\.0\}\}",
        ),
        (
            {"pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.5}}},
            r"'pose\.position\.z' is not read",
        ),
        (
            {"pose": {"position": {"x": 0, "y": 0}, "orientation": {"roll": 0.2}}},
            r"tilts the walker",
        ),
    ],
)
def test_a_start_that_is_not_a_walker_pose_is_refused(overrides, expect):
    from roqsim_walker.plugins.walker import WalkerPlugin

    config = {"walker": "MaleVisitorWalk", "skin": False, **overrides}
    errors = WalkerPlugin(config).validate_config(config)
    assert any(re.search(expect, e) for e in errors), errors


def test_a_goal_driven_walker_starts_at_its_pose_heading_every_episode():
    from roqsim_walker.output import STATE_KEY

    pose = {"position": {"x": 1.0, "y": -1.0}, "orientation": {"yaw": 1.2}}
    engine = Engine(_world(pose=pose, navigator=None))
    engine.setup()
    try:
        for _ in range(2):
            engine.reset()
            state = engine.ctx.blackboard.get(STATE_KEY)["pedestrian"]
            assert state.yaw == pytest.approx(1.2)
            np.testing.assert_allclose(_xy(engine), [1.0, -1.0], atol=1e-6)
            handle = engine.ctx.blackboard.get("nav:pedestrian:handle")
            handle.send_goals([(-1.0, -1.0)])
            _run(engine, 1.0)
    finally:
        engine.shutdown()


@pytest.fixture(scope="module")
def _standing_engine():
    engine = Engine(_world(pose={"position": {"x": 1.0, "y": 1.0}}, navigator=None))
    engine.setup()
    yield engine
    engine.shutdown()


@pytest.fixture
def standing(_standing_engine):
    """A walker standing at (1, 1) facing +x with the default navigator, in a fresh episode."""
    _standing_engine.reset()
    return _standing_engine


def _pair(z, x=1.5):
    """A left and a right point ahead of the standing walker, at height ``z``."""
    return np.array([x, 1.3, z]), np.array([x, 0.7, z])


def _anim(engine):
    from roqsim_walker.output import STATE_KEY

    return engine.ctx.blackboard.get(STATE_KEY)["pedestrian"]


def _head(engine):
    model = engine.ctx.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pedestrian/head")
    return engine.ctx.data.mocap_pos[model.body_mocapid[bid]].copy()


@pytest.mark.parametrize("height", [1.0, 0.7])
def test_a_reaching_walker_puts_its_wrists_on_the_points(standing, height):
    handle = standing.ctx.blackboard.get("walker:pedestrian")
    points = _pair(height)
    handle.reach(left=points[0], right=points[1])
    _run(standing, 3.0)
    # On them, not as far under them as the body is shifted down onto its feet.
    for wrist, point in zip(handle.hands(), points, strict=True):
        assert np.linalg.norm(wrist - point) < 0.005, (wrist, point)


def test_a_walker_straightens_as_the_points_in_its_hands_come_up(standing):
    handle = standing.ctx.blackboard.get("walker:pedestrian")
    handle.reach(*_pair(0.7))
    _run(standing, 3.0)
    # A box lifted 30 cm in a quarter of a second: the body comes up with it, not after it.
    steps = int(0.25 / standing.ctx.dt)
    for k in range(1, steps + 1):
        handle.reach(*_pair(0.7 + 0.3 * k / steps))
        standing.step()
    lifted = _head(standing)[2]
    _run(standing, 3.0)
    assert lifted > _head(standing)[2] - 0.02, (lifted, _head(standing)[2])


def test_a_quick_lift_lowered_at_once_never_bends_the_body_back(standing):
    from roqsim_walker.reach import MAX_BEND

    handle = standing.ctx.blackboard.get("walker:pedestrian")
    handle.reach(*_pair(0.7))
    _run(standing, 3.0)
    steps = int(0.1 / standing.ctx.dt)
    bends = []
    for heights in (
        [0.7 + 0.3 * k / steps for k in range(1, steps + 1)],
        [1.0 - 0.3 * k / steps for k in range(1, steps + 1)],
    ):
        for z in heights:
            handle.reach(*_pair(z))
            standing.step()
            bends.append(_anim(standing).reach_bend)
    assert 0.0 <= min(bends) and max(bends) <= MAX_BEND, (min(bends), max(bends))


def test_one_hand_reaches_and_the_other_keeps_the_gait(standing):
    from roqsim_walker.animation import _reach
    from roqsim_walker.humanoid import JOINT_NAMES

    handle = standing.ctx.blackboard.get("walker:pedestrian")
    point = np.array([1.5, 0.7, 0.75])
    handle.reach(right=point)
    _run(standing, 3.0)
    assert np.linalg.norm(handle.hands()[1] - point) < 0.005
    st = _anim(standing)
    assert st.reach_bend > 0.0 and st.reach["l"].w == 0.0 and st.reach["l"].pose is None
    # Blending the reach over a gait pose leaves the left arm's joints as the gait has them and
    # moves the right arm's and the trunk's.
    gait, root_z = st.idle.sample_array(0.3)
    q = gait.copy()
    _reach(st, q, st.body_pos, root_z, standing.ctx.dt)
    for joint in ("shoulder_l", "elbow_l", "wrist_l"):
        j = JOINT_NAMES.index(joint)
        assert np.array_equal(q[j], gait[j]), joint
    for joint in ("shoulder_r", "elbow_r", "pelvis"):
        j = JOINT_NAMES.index(joint)
        assert not np.allclose(q[j], gait[j]), joint


def test_letting_go_with_one_hand_keeps_the_other_on_its_point(standing):
    handle = standing.ctx.blackboard.get("walker:pedestrian")
    left, right = _pair(0.8)
    handle.reach(left, right)
    _run(standing, 3.0)
    handle.reach(right=right)
    # The left arm eases back to the walk, never jumping off its point, and the right hand stays on
    # its own all the while.
    wrist_l, wrist_r = handle.hands()
    for _ in range(int(2.0 / standing.ctx.dt)):
        standing.step()
        now_l, wrist_r = handle.hands()
        assert np.linalg.norm(now_l - wrist_l) < 0.05, (now_l, wrist_l)
        assert np.linalg.norm(wrist_r - right) < 0.005, wrist_r
        wrist_l = now_l
    assert np.linalg.norm(wrist_l - left) > 0.1, wrist_l
    st = _anim(standing)
    assert st.reach["l"].w == 0.0 and st.reach["r"].w == 1.0


def test_letting_go_straightens_the_body_and_the_next_reach_starts_upright(standing):
    handle = standing.ctx.blackboard.get("walker:pedestrian")
    handle.reach(*_pair(0.7))
    _run(standing, 3.0)
    st = _anim(standing)
    assert st.reach_bend > 0.5
    handle.reach()
    _run(standing, 2.0)
    assert st.reach_bend == 0.0 and st.reach_bend_rate == 0.0
    assert all(hand.w == 0.0 and hand.pose is None for hand in st.reach.values())


@pytest.mark.parametrize(
    "point",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, float("nan"), 0.5],
        (1.0, 0.0, float("inf")),
        "abc",
        [True, 0.0, 0.0],
        ["1", "2", "3"],
        5.0,
    ],
)
@pytest.mark.parametrize("hand", ["left", "right"])
def test_a_point_that_is_not_three_finite_numbers_is_refused_at_the_call(standing, hand, point):
    handle = standing.ctx.blackboard.get("walker:pedestrian")
    left, right = _pair(1.0)
    handle.reach(left, right)
    with pytest.raises(ValueError, match=f"the {hand} hand's point"):
        handle.reach(**{"left": left, "right": right, hand: point})
    # Neither hand changed.
    st = _anim(standing)
    assert np.array_equal(st.reach["l"].target, left)
    assert np.array_equal(st.reach["r"].target, right)


@pytest.mark.parametrize("lag", [-0.1, float("nan"), float("inf"), "0.1", True])
def test_a_body_lag_that_is_no_finite_time_constant_is_refused(lag):
    from roqsim_walker.plugins.walker import WalkerPlugin

    config = {"body_lag": lag}
    errors = WalkerPlugin(config).validate_config(config)
    assert any("'body_lag' is a time constant" in e for e in errors), errors
