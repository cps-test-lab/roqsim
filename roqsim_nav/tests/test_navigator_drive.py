"""The navigator driving a real robot: a second TurtleBot 4 whose wheels actually turn.

The point of this output is that *nothing about the robot changes*. The navigator feeds the same
``drive(vx, vy, w)`` entry point the ROS bridge writes ``/cmd_vel`` into, so ``diff_drive`` still
does its own inverse kinematics, its Create-3 acceleration ramp and its wheel-encoder odometry. Only
the source of the twist differs. Two assertions carry that claim: the wheel joints must actually
spin, and the navigator must never write ``data.ctrl`` itself.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _world(*, model="turtlebot4", nav=None, pos=(-3.0, 0.0), extra=None):
    navigator = {"speed": 0.25, "goals": [[3.0, 0.0]], "caution": {"enabled": False}}
    navigator.update(nav or {})
    components = [
        {
            "spawn_robot": {"model": model, "pos": list(pos)},
            "name": "cart",
            "components": [{"navigator": navigator}],
        }
    ]
    components.extend(extra or [])
    return load_config_from_dict({"sim": {"pacing": "asap"}, "components": components})


@pytest.fixture(scope="module")
def _engine():
    engine = Engine(_world())
    engine.setup()
    yield engine
    engine.shutdown()


@pytest.fixture
def sim(_engine):
    _engine.reset()
    return _engine


def _base_xy(engine):
    model = engine.ctx.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("cart").body)
    return engine.ctx.data.xpos[bid][:2].copy()


def _wheel_dofs(engine):
    model = engine.ctx.model
    dofs = []
    for name in ("left_wheel_joint", "right_wheel_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"{name} not found -- the TurtleBot 4 model's joint naming changed"
        dofs.append(int(model.jnt_dofadr[jid]))
    return dofs


def _run(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def test_the_wheels_actually_turn(sim):
    """The load-bearing assertion: the twist went *through* diff_drive's inverse kinematics.

    A navigator that moved the base some other way -- writing its pose, or its free joint -- would
    still reach the goal, with the wheels sitting perfectly still.
    """
    dofs = _wheel_dofs(sim)
    peak = 0.0
    for _ in range(int(3.0 / sim.ctx.dt)):
        sim.step()
        peak = max(peak, max(abs(float(sim.ctx.data.qvel[d])) for d in dofs))
    assert peak > 1.0, f"the wheels barely moved ({peak:.3f} rad/s)"


def test_it_reaches_the_goal_under_physics(sim):
    _run(sim, 30.0)
    assert np.linalg.norm(_base_xy(sim) - np.asarray([3.0, 0.0])) < 0.4


def test_the_navigator_never_writes_ctrl_itself(sim):
    """It is a controller, not a second driver. Writing ``ctrl`` would bypass the acceleration ramp
    and the wheel-velocity servos, and quietly fight whatever else commands the robot."""
    navigator = next(p for p in sim.plugins if type(p).__name__ == "NavigatorPlugin")
    ctx = sim.ctx
    seen = []

    original = navigator._output.emit

    def spy(ctx_, pref_vel, yaw, dt):
        before = ctx.data.ctrl.copy()
        original(ctx_, pref_vel, yaw, dt)
        seen.append(np.array_equal(before, ctx.data.ctrl))

    navigator._output.emit = spy
    try:
        _run(sim, 2.0)
    finally:
        navigator._output.emit = original
    assert seen, "the navigator never emitted, so this proves nothing"
    assert all(seen), "the navigator wrote data.ctrl -- diff_drive owns the actuators"


def test_it_yields_the_actuators_to_manual_control(sim):
    """The viewer's sliders own the robot when a human takes it; two writers would fight."""
    navigator = next(p for p in sim.plugins if type(p).__name__ == "NavigatorPlugin")
    _run(sim, 1.0)
    sim.ctx.manual_control = True
    try:
        commanded = []
        original = navigator._output.emit
        navigator._output.emit = lambda *a, **k: commanded.append(1)
        _run(sim, 1.0)
        navigator._output.emit = original
        assert not commanded, "it kept commanding the base while a human held the controls"
    finally:
        sim.ctx.manual_control = False


def test_the_base_geometry_is_taken_from_the_drive_not_guessed(sim):
    """`diff_drive` declares `unicycle`; the navigator asks rather than keying on a robot name."""
    handle = sim.ctx.blackboard.get("robot:cart")
    assert handle is not None
    assert handle.kinematics == "unicycle"
    navigator = next(p for p in sim.plugins if type(p).__name__ == "NavigatorPlugin")
    assert navigator._output.kinematics == "unicycle"


def test_an_explicit_kinematics_overrides_the_declaration(sim):
    """For an out-of-tree drive that has not declared one yet."""
    engine = Engine(_world(nav={"kinematics": "holonomic"}))
    engine.setup()
    try:
        navigator = next(p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin")
        assert navigator._output.kinematics == "holonomic"
    finally:
        engine.shutdown()


def test_a_robot_without_a_drive_names_the_fix():
    """A bare model with no controller cannot be commanded, and the error has to say why."""
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {},
                "components": [
                    {
                        "spawn_robot": {"model": "turtlebot4", "pos": [0.0, 0.0]},
                        "name": "cart",
                        "components": [
                            {"diff_drive": {}, "enabled": False},
                            {"navigator": {"speed": 0.2, "goals": [[1.0, 0.0]], "output": "drive"}},
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(Exception, match="RobotHandle|diff_drive"):
        engine.setup()


def test_auto_picks_drive_for_a_robot_that_publishes_a_handle(sim):
    navigator = next(p for p in sim.plugins if type(p).__name__ == "NavigatorPlugin")
    assert type(navigator._output).__name__ == "DriveOutput"


def test_reset_stops_the_base_and_returns_it_to_the_start(sim):
    _run(sim, 5.0)
    assert np.linalg.norm(_base_xy(sim) - np.asarray([-3.0, 0.0])) > 0.5
    sim.reset()
    assert _base_xy(sim) == pytest.approx([-3.0, 0.0], abs=0.05)
    dofs = _wheel_dofs(sim)
    assert all(abs(float(sim.ctx.data.qvel[d])) < 1e-6 for d in dofs), "the wheels kept spinning"


# -- the three base geometries -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "kinematics", "goal"),
    [
        ("turtlebot4", "unicycle", (3.0, 2.0)),  # differential: drives and turns
        ("mpo_700", "holonomic", (3.0, 2.0)),  # swerve: any planar velocity
        ("piracer", "ackermann", (4.0, 2.0)),  # a car: cannot turn in place at all
    ],
)
def test_each_base_geometry_navigates_with_its_own_law(model, kinematics, goal):
    """One navigator, three bases, no branch on a robot's name anywhere.

    ``piracer`` is the one that would expose a wrong law rather than merely a suboptimal one: the
    unicycle law commands ``v = 0`` to pivot when the heading error is large, and a car derives its
    steering angle from ``w / v``, so it would sit still with its wheels straight and never arrive.
    """
    engine = Engine(_world(model=model, nav={"speed": 0.3, "goals": [list(goal)]}))
    engine.setup()
    engine.reset()
    try:
        handle = engine.ctx.blackboard.get("robot:cart")
        assert handle.kinematics == kinematics, "the drive plugin declares a different geometry"
        _run(engine, 60.0)
        assert np.linalg.norm(_base_xy(engine) - np.asarray(goal)) < 0.4
    finally:
        engine.shutdown()


# -- several at once ------------------------------------------------------------------------------
def test_three_robots_navigate_at_once_and_stop_for_each_other():
    """Nothing about the navigator is per-world singular, and this is what says so.

    Three TurtleBot 4s on paths that all cross the origin. Each gets its own goal handle and its own
    caution probe; they share one rasterized grid, because the walls are the same walls. They must
    each arrive, and they must do it by *stopping* for one another rather than by happening to miss
    each other in time -- so the test asserts both that they came close and that caution fired.
    """
    routes = [
        ("r0", (-4.0, 0.0), (4.0, 0.0)),
        ("r1", (0.0, -4.0), (0.0, 4.0)),
        ("r2", (-3.0, -3.0), (3.0, 3.0)),
    ]
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap"},
                "components": [
                    {
                        "spawn_robot": {
                            "model": "turtlebot4",
                            "prefix": f"{name}_",
                            "namespace": name,
                            "pos": list(start),
                        },
                        "name": name,
                        "components": [
                            {
                                "navigator": {
                                    "speed": 0.4,
                                    "goals": [list(goal)],
                                    "caution": {"lookahead": 0.8},
                                }
                            }
                        ],
                    }
                    for name, start, goal in routes
                ],
            }
        )
    )
    engine.setup()
    engine.reset()
    try:
        navigators = {p.entity: p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin"}
        assert sorted(navigators) == ["r0", "r1", "r2"]
        for name in navigators:
            assert engine.ctx.blackboard.get(f"nav:{name}:handle") is not None

        def xy(name):
            model = engine.ctx.model
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get(name).body
            )
            return engine.ctx.data.xpos[bid][:2].copy()

        engine.step()
        grids = {id(n._core.planner.grid) for n in navigators.values()}
        assert len(grids) == 1, "each robot rasterized its own copy of the same walls"

        pairs = [("r0", "r1"), ("r0", "r2"), ("r1", "r2")]
        closest, blocked = float("inf"), 0
        for i in range(int(60.0 / engine.ctx.dt)):
            engine.step()
            if i % 25 == 0:
                closest = min(closest, min(np.linalg.norm(xy(a) - xy(b)) for a, b in pairs))
                blocked += sum(1 for n in navigators.values() if n._caution.blocked)

        for name, _start, goal in routes:
            assert np.linalg.norm(xy(name) - np.asarray(goal)) < 0.4, f"{name} never arrived"
        assert closest < 1.0, "they never came near each other, so nothing was tested"
        assert blocked > 0, "they passed without ever yielding -- caution never engaged"
    finally:
        engine.shutdown()
