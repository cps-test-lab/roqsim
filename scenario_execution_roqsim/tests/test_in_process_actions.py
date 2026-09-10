"""The four actions against a real simulation, on the in-process transport.

Deliberately NOT against the tiago pick world: this package must not depend on an experiment. A crate
on a ramp with a friction override is enough to exercise everything that can go wrong -- a baseline
taken at the wrong moment, a dwell measured on the wrong clock, a queued write observed too early, and
the two verdicts (`landed` / `no_effect`) that decide whether a trial is a result or a lie.

The scene is the one `roqsim/tests/test_model_override.py` uses, and for its reason: the crate carries
``priority="1"``, so overriding the CRATE governs the contact while overriding the RAMP cannot. That is
what lets one scene test both verdicts.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

pytest.importorskip(
    "scenario_execution",
    reason="the actions import scenario_execution; the displacement maths is tested without it",
)

import py_trees  # noqa: E402
from scenario_execution.actions.base_action import ActionError  # noqa: E402

from roqsim.context import Entity, SimContext  # noqa: E402
from roqsim.plugins.model_override import ModelOverridePlugin  # noqa: E402
from scenario_execution_roqsim.actions.entity_moved import EntityMoved
from scenario_execution_roqsim.actions.entity_navigate import (  # noqa: E402
    EntityNavigate,
    EntityNavigateStart,
)
from scenario_execution_roqsim.actions.entity_rotated import EntityRotated  # noqa: E402
from scenario_execution_roqsim.actions.set_entity_state import SetEntityState  # noqa: E402
from scenario_execution_roqsim.actions.set_model_override import SetModelOverride  # noqa: E402
from scenario_execution_roqsim.actions.set_sensor_override import SetSensorOverride  # noqa: E402
from scenario_execution_roqsim.actions.spawn_entity import SpawnEntity  # noqa: E402

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE

SCENE = """
<mujoco model="access_test">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ramp" type="box" size="1 1 0.02" euler="0 20 0" friction="1.0 0.005 0.0001"/>
    <body name="crate" pos="0 0 0.4">
      <freejoint/>
      <geom name="crate" type="box" size="0.05 0.05 0.05" mass="1"
            priority="1" friction="0.7 0.02 0.001" euler="0 20 0"/>
    </body>
    <!-- A second movable body, so the `require: any|all` quantifier can be tested on two entities
         that CAN both move. (A welded one would be refused before the quantifier is reached, which
         is a different test.) -->
    <body name="crate_b" pos="0.5 0 0.4">
      <freejoint/>
      <geom name="crate_b" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class FakeClock:
    """The runner's clock, under the test's control. Sim seconds, like SimulationClock's."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t


class FakeSim:
    """A stand-in for `MujocoSim`: the ONLY thing an in-process action needs is `context`.

    Which is the point of the narrow seam -- a test does not have to build an Engine, and an adapter
    of someone else's making satisfies the same contract with one property.
    """

    def __init__(self, ctx):
        self.context = ctx


@pytest.fixture
def world():
    """ctx, clock, sim, plus an inert `grip_fault` on the geom that GOVERNS the contact."""
    model = mujoco.MjModel.from_xml_string(SCENE)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    # The entity's name is NOT its body's name -- the case the whole resolver exists for.
    ctx.entities.add(Entity(name="parcel", kind="object", body="crate"))
    return ctx, FakeClock(), FakeSim(ctx)


def _override(ctx, select=("crate",), name="grip_fault"):
    plugin = ModelOverridePlugin(
        {"overrides": [{"field": "geom_friction", "select": list(select), "to": 0.0}]}, name=name
    )
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return plugin


def _handle(ctx, name="grip_fault"):
    """The plugin's blackboard handle -- what an in-process consumer reads, `is_active` included."""
    return ctx.blackboard.get(f"model_override:{name}")


def _step(ctx, clock, plugin=None, seconds=0.002):
    """One or more engine steps, as the engine does them: drain, step, post_step, advance the clock."""
    for _ in range(max(1, int(seconds / ctx.model.opt.timestep))):
        ctx.drain_commands()
        mujoco.mj_step(ctx.model, ctx.data)
        if plugin is not None:
            plugin.post_step(ctx)
        clock.t = float(ctx.data.time)


def _start(action, sim, clock, **args):
    action.setup(simulation=sim, clock=clock)
    action.execute(**args)
    return action


# -- entity_moved ---------------------------------------------------------------------------------
def test_the_baseline_is_where_the_entity_was_when_the_action_started(world):
    """Not its absolute pose, and not a world-side plugin's reference: the crate starts at z = 0.4.

    A trigger measuring absolute z would fire instantly on any sensible threshold. That is exactly the
    bug the predecessor action needed an `_armed` hysteresis flag to survive, and it disappears when the
    action owns its own baseline.
    """
    ctx, clock, sim = world
    action = _start(
        EntityMoved(),
        sim,
        clock,
        entities=["parcel"],
        threshold=0.05,
        mode="z",
        dwell=0.0,
        require="all",
    )
    assert action.update() is RUNNING, "0.4 m above the floor is not 0.05 m of MOVEMENT"
    assert "parcel" in action.feedback_message


def test_it_succeeds_once_the_entity_has_actually_moved(world):
    """The crate slides down the ramp; the action fires when the displacement passes the threshold."""
    ctx, clock, sim = world
    action = _start(
        EntityMoved(),
        sim,
        clock,
        entities=["parcel"],
        threshold=0.05,
        mode="distance",
        dwell=0.0,
        require="all",
    )
    assert action.update() is RUNNING
    for _ in range(2000):
        _step(ctx, clock)
        if action.update() is SUCCESS:
            break
    assert action.update() is SUCCESS
    moved = float(
        np.linalg.norm(
            ctx.data.xpos[mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")]
            - np.array([0.0, 0.0, 0.4])
        )
    )
    assert moved >= 0.05


def test_the_dwell_is_measured_on_the_runners_clock_and_restarts_on_a_dip(world):
    """A crossing flatters the result, so the condition must hold CONTINUOUSLY for `dwell`.

    Driven synthetically rather than by physics: what is under test is the dwell bookkeeping, and a
    real dip would take a scene built to bounce.
    """
    ctx, clock, sim = world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    action = _start(
        EntityMoved(),
        sim,
        clock,
        entities=["parcel"],
        threshold=0.05,
        mode="z",
        dwell=1.0,
        require="all",
    )
    action.update()  # capture the baseline at z = 0.4

    ctx.data.xpos[bid][2] = 0.46  # +60 mm: satisfied, dwell starts
    clock.t = 10.0
    assert action.update() is RUNNING
    clock.t = 10.5
    assert action.update() is RUNNING, "half the dwell is not the dwell"

    ctx.data.xpos[bid][2] = 0.41  # dipped back under: the dwell must restart, not accumulate
    assert action.update() is RUNNING
    ctx.data.xpos[bid][2] = 0.46
    clock.t = 11.0
    assert action.update() is RUNNING, "the dwell restarted, so 11.0 is only the crossing again"
    clock.t = 12.01
    assert action.update() is SUCCESS


def test_a_list_is_quantified_by_require(world):
    """`all` is the default because that is how the rest of the vocabulary reads a list."""
    ctx, clock, sim = world
    ctx.entities.add(Entity(name="parcel_b", kind="object", body="crate_b"))
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")

    pair = dict(entities=["parcel", "parcel_b"], threshold=0.05, mode="distance", dwell=0.0)
    every = _start(EntityMoved(), sim, clock, require="all", **pair)
    either = _start(EntityMoved(), sim, clock, require="any", **pair)
    every.update()
    either.update()
    ctx.data.xpos[bid][2] = 0.5  # only the first crate moves

    assert either.update() is SUCCESS, "`any` is satisfied by one of them"
    assert every.update() is RUNNING, "`all` is not"


@pytest.mark.parametrize(
    "args,message",
    [
        (dict(entities=[], threshold=0.05, mode="z"), "empty"),
        # None is what the PARSER hands over for an omitted required argument (measured
        # in test_osc_library), so it must be refused the same way an empty list is.
        (dict(entities=None, threshold=0.05, mode="z"), "empty"),
        (dict(entities=["parcel"], threshold=0.0, mode="z"), "SIGN"),
        (dict(entities=["parcel"], threshold=0.0, mode="distance"), "must be > 0"),
        (dict(entities=["parcel"], threshold=0.05, mode="sideways"), "unknown mode"),
        (dict(entities=["parcel"], threshold=0.05, mode="z", require="most"), "unknown `require`"),
        (dict(entities=["parcel"], threshold=0.05, mode="z", dwell=-1.0), "must be >= 0"),
    ],
)
def test_an_unusable_configuration_raises_at_execute(world, args, message):
    """These are AUTHORING errors -- no run could recover -- so they raise rather than fail a trial."""
    _ctx, clock, sim = world
    action = EntityMoved()
    action.setup(simulation=sim, clock=clock)
    full = {"dwell": 0.0, "require": "all", **args}
    with pytest.raises(ActionError, match=message):
        action.execute(**full)


def test_an_unknown_entity_raises_and_names_the_near_miss(world):
    """A typo'd entity would otherwise wait out the scenario timeout with nothing to explain it."""
    _ctx, clock, sim = world
    action = _start(
        EntityMoved(),
        sim,
        clock,
        entities=["parcell"],
        threshold=0.05,
        mode="z",
        dwell=0.0,
        require="all",
    )
    with pytest.raises(ActionError, match="parcel"):
        action.update()


def test_a_welded_entity_raises_rather_than_waiting_forever(world):
    """Its pose is a compile-time constant, so "wait until it moves" can never be satisfied."""
    ctx, clock, sim = world
    ctx.entities.add(Entity(name="ramp_entity", kind="prop", body="world"))
    action = _start(
        EntityMoved(),
        sim,
        clock,
        entities=["ramp_entity"],
        threshold=0.05,
        mode="distance",
        dwell=0.0,
        require="all",
    )
    with pytest.raises(ActionError, match="welded to the world"):
        action.update()


def test_it_waits_rather_than_building_a_world(world):
    """The tree is set up before the first reset, so `context` is None and the action must wait."""
    _ctx, clock, _sim = world

    class NotBuilt:
        context = None

    action = _start(
        EntityMoved(),
        NotBuilt(),
        clock,
        entities=["parcel"],
        threshold=0.05,
        mode="z",
        dwell=0.0,
        require="all",
    )
    assert action.update() is RUNNING
    assert "simulation" in action.feedback_message


# -- entity_rotated -------------------------------------------------------------------------------
def test_rotation_fires_on_the_geodesic_angle(world):
    ctx, clock, sim = world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    action = _start(
        EntityRotated(),
        sim,
        clock,
        entities=["parcel"],
        angle=0.5,
        dwell=0.0,
        require="all",
    )
    assert action.update() is RUNNING
    turn = 0.6
    ctx.data.xquat[bid] = [math.cos(turn / 2), 0.0, 0.0, math.sin(turn / 2)]
    assert action.update() is SUCCESS
    assert "deg" in action.feedback_message


def test_an_angle_beyond_pi_is_refused(world):
    """The geodesic angle saturates at pi, so a larger threshold could never be met."""
    _ctx, clock, sim = world
    action = EntityRotated()
    action.setup(simulation=sim, clock=clock)
    with pytest.raises(ActionError, match="<= pi"):
        action.execute(entities=["parcel"], angle=7.0, dwell=0.0, require="all")


# -- set_model_override ---------------------------------------------------------------------------
def test_the_fault_is_applied_and_the_verdict_read_back(world):
    """SUCCESS only after the queued write has landed AND the plugin has verified it."""
    ctx, clock, sim = world
    plugin = _override(ctx)
    _step(
        ctx, clock, plugin, seconds=2.0
    )  # let the crate settle onto the ramp, so a contact exists

    action = _start(
        SetModelOverride(), sim, clock, instance="grip_fault", active=True, require_landed=True
    )
    assert action.update() is RUNNING, "the write is queued, not yet applied"
    assert _handle(ctx).is_active() is False

    _step(ctx, clock, plugin)  # drains the command, applies it, and verifies in the same step
    assert _handle(ctx).is_active() is True
    assert action.update() is SUCCESS
    assert "landed" in action.feedback_message


def test_a_fault_that_changed_nothing_fails_the_trial_instead_of_raising(world):
    """The distinction the whole base class exists for.

    Overriding the RAMP cannot lower the contact: MuJoCo takes friction from the higher-`priority`
    geom, and the crate carries priority 1. A raise here would kill the run with no test.xml and no
    result row -- a campaign cell that reads as "never scheduled" rather than as a failed trial.
    """
    ctx, clock, sim = world
    plugin = _override(ctx, select=("ramp",))
    _step(ctx, clock, plugin, seconds=2.0)

    action = _start(
        SetModelOverride(), sim, clock, instance="grip_fault", active=True, require_landed=True
    )
    action.update()
    _step(ctx, clock, plugin)
    assert plugin.read_state().verified == "no_effect", "precondition: the write did nothing"
    assert action.update() is FAILURE
    assert "no_effect" in action.feedback_message


def test_no_effect_is_tolerated_when_the_scenario_says_so(world):
    ctx, clock, sim = world
    plugin = _override(ctx, select=("ramp",))
    _step(ctx, clock, plugin, seconds=2.0)
    action = _start(
        SetModelOverride(), sim, clock, instance="grip_fault", active=True, require_landed=False
    )
    action.update()
    _step(ctx, clock, plugin)
    assert action.update() is SUCCESS


def test_asking_for_the_state_it_is_already_in_succeeds_immediately(world):
    """`set_active` returns early when the state matches, so `changes` never moves.

    An action waiting for a transition would hang here forever -- which is why completion is keyed on
    `changes` and this case is answered without posting anything at all.
    """
    ctx, clock, sim = world
    _override(ctx)  # armed but inert, which is the state the action is about to ask for
    action = _start(
        SetModelOverride(), sim, clock, instance="grip_fault", active=False, require_landed=True
    )
    assert action.update() is SUCCESS, "already nominal"
    assert "already" in action.feedback_message


def test_a_restore_completes_although_there_is_nothing_to_verify(world):
    """`active: false` writes saved values back; the plugin reports `untested`, which is not a failure."""
    ctx, clock, sim = world
    plugin = _override(ctx)
    _step(ctx, clock, plugin, seconds=2.0)
    plugin.set_active(True)
    _step(ctx, clock, plugin)

    action = _start(
        SetModelOverride(), sim, clock, instance="grip_fault", active=False, require_landed=True
    )
    assert action.update() is RUNNING
    _step(ctx, clock, plugin)
    assert _handle(ctx).is_active() is False
    assert action.update() is SUCCESS


def test_an_unknown_instance_raises_and_says_where_it_comes_from(world):
    _ctx, clock, sim = world
    action = _start(
        SetModelOverride(), sim, clock, instance="typo_fault", active=True, require_landed=True
    )
    with pytest.raises(ActionError, match="model_override:typo_fault"):
        action.update()


# -- set_entity_state --------------------------------------------------------------------------------
#
# A separate tiny scene: the shared `world` fixture's freejoint is unnamed (fine for entity_moved,
# which resolves the BODY), and set_entity_state needs a named joint to write qpos through.
TELEPORT_SCENE = """
<mujoco model="teleport_test">
  <worldbody>
    <body name="robot" pos="1 2 0.1">
      <freejoint name="robot_free"/>
      <geom name="robot" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="fixed_prop" pos="3 3 0.1">
      <geom name="prop" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="driven_prop" pos="4 4 0.1" mocap="true">
      <geom name="driven" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def teleport_world():
    model = mujoco.MjModel.from_xml_string(TELEPORT_SCENE)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="robot", meta={"base_joint": "robot_free"})
    )
    # No `base_joint` in meta: exercises the "cannot be teleported" outcome (a static prop).
    ctx.entities.add(Entity(name="prop", kind="object", body="fixed_prop", meta={}))
    # `motion: driven`: no base_joint either, and placeable all the same -- the body is what
    # carries the answer, which is why a mocap prop registers exactly the meta a welded one does.
    ctx.entities.add(Entity(name="driven", kind="object", body="driven_prop", meta={"mocap": True}))
    return ctx, FakeClock(), FakeSim(ctx)


def test_set_entity_state_places_the_entity_and_zeroes_its_velocity(teleport_world):
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    dof = ctx.model.jnt_dofadr[jid]
    ctx.data.qvel[dof : dof + 6] = 1.0  # nonzero, so the teleport's zeroing is actually exercised

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 5.0, "y": -1.0, "z": 0.0}, "orientation": {"yaw": math.pi / 2}},
    )
    assert action.update() is RUNNING, "the write is posted, not yet drained"
    _step(ctx, clock)
    assert action.update() is SUCCESS

    assert np.allclose(ctx.data.xpos[bid][:2], [5.0, -1.0], atol=1e-6)
    quat = ctx.data.xquat[bid]
    assert np.allclose(quat, [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], atol=1e-6)
    # atol, not exactly 0: `_step` drains the write and then steps physics once, so gravity has
    # already pulled qvel[z] away from the zero the write itself set by one timestep's worth
    # (9.81 * 0.002 s here) -- the write is verified where it happens, not frozen against physics.
    assert np.allclose(ctx.data.qvel[dof : dof + 6], 0.0, atol=0.03)


def test_set_entity_state_fails_the_trial_rather_than_raise_when_the_entity_has_no_free_joint(
    teleport_world,
):
    """A static prop is a fact about the world the campaign chose, not a malformed call."""
    ctx, clock, sim = teleport_world
    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="prop",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is FAILURE
    assert "welded scenery" in action.feedback_message
    assert "motion: driven" in action.feedback_message


def test_set_entity_state_places_a_driven_prop_and_the_solver_leaves_it_there(teleport_world):
    """`motion: driven` is placeable, and that is the point of it.

    A trial that reveals an obstacle mid-run has to write its pose, and only two kinds of body
    can take one. A free body's pose is the solver's from the next step: placed intersecting other
    geometry it is launched out of the scene, and reached by the robot it is pushed off the
    placement the experiment chose. A mocap body has no DOF -- it takes the pose and keeps it,
    which is what an obstacle AT a position means.
    """
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "driven_prop")

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="driven",
        pose={"position": {"x": 5.0, "y": -1.0, "z": 0.5}, "orientation": {"yaw": math.pi / 2}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS

    assert np.allclose(ctx.data.xpos[bid], [5.0, -1.0, 0.5], atol=1e-6)
    quat = ctx.data.xquat[bid]
    assert np.allclose(quat, [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], atol=1e-6)

    # Exactly, and after further stepping: a free body would have fallen by now, and one placed
    # inside something would have been pushed out of it.
    for _ in range(50):
        mujoco.mj_step(ctx.model, ctx.data)
    assert np.allclose(ctx.data.xpos[bid], [5.0, -1.0, 0.5], atol=1e-9)


def test_spawn_places_a_driven_prop_as_it_appears(teleport_world):
    """The other door onto the same write: a revealed obstacle is placed where the trial says."""
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "driven_prop")
    ctx.entities.get("driven").present = False

    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="driven",
        pose={"position": {"x": 1.0, "y": 1.0, "z": 0.5}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS
    assert np.allclose(ctx.data.xpos[bid], [1.0, 1.0, 0.5], atol=1e-6)


def test_set_entity_state_raises_on_an_unknown_entity(teleport_world):
    _ctx, clock, sim = teleport_world
    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="ghost",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"yaw": 0.0}},
    )
    with pytest.raises(ActionError, match="no entity"):
        action.update()


def test_set_entity_state_applies_roll_and_pitch(teleport_world):
    """A full orientation, the same one `SetEntityState` has always accepted.

    This action used to convert yaw itself and refuse roll or pitch, on the grounds that it places
    a wheeled base on its floor. That made the OSC verb the only place in the substrate where an
    orientation meant something narrower than everywhere else -- while the service behind it took
    a whole quaternion -- and it blocked aiming a sensor, which is a pose with no floor in it. The
    real constraint is whether the body has a free joint, and the simulator already reports that.
    """
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    roll, pitch, yaw = 0.3, -0.2, 1.1

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="robot",
        pose={
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
        },
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS

    # Compared against roqsim's own conversion rather than a hand-written quaternion: the point is
    # that this action no longer has a convention of its own, and pinning a literal here would put
    # a second one back into the tests.
    from roqsim.pose import rpy_to_quat

    assert np.allclose(ctx.data.xquat[bid], rpy_to_quat(roll, pitch, yaw), atol=1e-6)


def test_set_entity_state_accepts_a_quaternion_as_the_service_states_one(teleport_world):
    """The other spelling. `parse_pose` takes either, so a pose can be pasted from a
    `geometry_msgs/Pose` without being rewritten as Euler angles first."""
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    half = math.pi / 4

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="robot",
        pose={
            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
        },
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS
    assert np.allclose(ctx.data.xquat[bid], [math.cos(half), 0.0, 0.0, math.sin(half)], atol=1e-6)


def test_set_entity_state_refuses_a_malformed_pose_naming_the_key(teleport_world):
    """A key `pose` has no business carrying is refused by name, by the same parser a world
    document goes through -- rather than silently defaulting to the origin."""
    _ctx, clock, sim = teleport_world
    action = SetEntityState()
    with pytest.raises(ActionError, match="orientation"):
        action.execute(
            entity="robot",
            pose={"position": {"x": 0.0, "y": 0.0}, "orientation": {"jaw": 0.5}},
        )


def test_set_entity_state_applies_a_stated_twist(teleport_world):
    """A twist is part of the state, and a stated one has to arrive.

    `SetEntityState` carries a twist, the GETTER reports one, and both this action and the bridge
    behind it used to drop it while replying OK -- so a caller could read a velocity it could not
    set. Written in the scenario's own vocabulary (`velocity_6d` spells its halves
    `translational`/`angular`).
    """
    ctx, clock, sim = teleport_world
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    dof = ctx.model.jnt_dofadr[jid]

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 1.0}, "orientation": {"yaw": 0.0}},
        twist={
            "translational": {"x": 1.5, "y": -0.5, "z": 0.0},
            "angular": {"roll": 0.0, "pitch": 0.0, "yaw": 0.75},
        },
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS

    # atol as elsewhere here: `_step` drains the write and then steps physics once, so gravity has
    # already moved qvel[z] by one timestep's worth.
    assert np.allclose(ctx.data.qvel[dof : dof + 3], [1.5, -0.5, 0.0], atol=0.03)
    assert np.allclose(ctx.data.qvel[dof + 3 : dof + 6], [0.0, 0.0, 0.75], atol=0.03)


def test_set_entity_state_accepts_the_messages_spelling_of_a_twist(teleport_world):
    """`linear`/`x` as the message spells it, so a twist can be pasted from either vocabulary."""
    ctx, clock, sim = teleport_world
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    dof = ctx.model.jnt_dofadr[jid]

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 1.0}, "orientation": {"yaw": 0.0}},
        twist={"linear": {"x": 0.0, "y": 2.0, "z": 0.0}, "angular": {"z": -0.25}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS
    assert np.allclose(ctx.data.qvel[dof + 1], 2.0, atol=0.03)
    assert np.allclose(ctx.data.qvel[dof + 5], -0.25, atol=0.03)


def test_set_entity_state_with_no_twist_still_zeroes_the_velocity(teleport_world):
    """The default, and what every caller before this relied on: a body PUT somewhere is not still
    carrying the velocity it had."""
    ctx, clock, sim = teleport_world
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    dof = ctx.model.jnt_dofadr[jid]
    ctx.data.qvel[dof : dof + 6] = 3.0

    action = _start(
        SetEntityState(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.5}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS
    assert np.allclose(ctx.data.qvel[dof : dof + 6], 0.0, atol=0.03)


# -- spawn_entity -----------------------------------------------------------------------------------
#
# Presence AND pose, and the point of the action is that they are one transaction: a flip and a pose
# applied separately leave the entity perceivable for a step wherever the world compiled it.


def test_spawn_makes_the_entity_present_at_the_pose_asked_for(teleport_world):
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    entity = ctx.entities.get("robot")
    entity.present = False  # declared absent, as a world would leave it for a per-run spawn

    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 4.0, "y": -2.0, "z": 0.1}, "orientation": {"yaw": math.pi / 2}},
    )
    assert action.update() is RUNNING, "the flip is posted, not yet drained"
    assert entity.present is False, "and nothing has changed before it drains"
    _step(ctx, clock)
    assert action.update() is SUCCESS

    assert entity.present is True
    assert np.allclose(ctx.data.xpos[bid][:2], [4.0, -2.0], atol=1e-6)
    assert np.allclose(
        ctx.data.xquat[bid], [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], atol=1e-6
    )


def test_spawn_takes_a_full_orientation(teleport_world):
    """Roll and pitch too -- aiming a sensor is a pose with no floor in it."""
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    ctx.entities.get("robot").present = False
    roll, pitch, yaw = 0.2, 0.4, -0.6

    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="robot",
        pose={
            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
            "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
        },
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS

    from roqsim.pose import rpy_to_quat

    assert np.allclose(ctx.data.xquat[bid], rpy_to_quat(roll, pitch, yaw), atol=1e-6)


def test_spawn_zeroes_the_velocity_the_entity_had_while_absent(teleport_world):
    """An entity that has just appeared has no history.

    An absent free body is frozen rather than moved, so whatever velocity it carried is still in
    qvel -- and a trial that measures the thing it just spawned must not inherit it.
    """
    ctx, clock, sim = teleport_world
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    dof = ctx.model.jnt_dofadr[jid]
    ctx.entities.get("robot").present = False
    ctx.data.qvel[dof : dof + 6] = 1.0

    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.5}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is SUCCESS
    # atol as in the teleport test: `_step` drains the write and then steps physics once.
    assert np.allclose(ctx.data.qvel[dof : dof + 6], 0.0, atol=0.03)


def test_spawn_fails_the_trial_when_the_entity_cannot_be_placed(teleport_world):
    """A welded prop asked for a pose is a fact about the world the campaign chose."""
    ctx, clock, sim = teleport_world
    # Absent first, or the presence check refuses it before the weld is ever reached -- which is
    # its own test below.
    ctx.entities.get("prop").present = False
    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="prop",
        pose={"position": {"x": 1.0, "y": 1.0, "z": 0.0}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is FAILURE
    assert "welded scenery" in action.feedback_message
    assert "motion: driven" in action.feedback_message


def test_spawn_refuses_an_entity_that_is_already_present(teleport_world):
    """The same answer both transports give.

    `SpawnEntity` over ROS answers RESULT_OPERATION_FAILED for an entity already in the state asked
    for, and in-process answers the same way: `set_present` returns whether anything changed, and
    the outcome is built from that rather than from having called it. A scenario is written once
    and does not learn which shape it runs in, so the two must not disagree.
    """
    ctx, clock, sim = teleport_world
    assert ctx.entities.get("robot").present, "the fixture spawns it present"

    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 1.0, "y": 1.0, "z": 0.5}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is FAILURE
    assert "already present" in action.feedback_message


def test_a_refused_spawn_leaves_the_entity_where_it_was(teleport_world):
    """Refusing after moving it would be worse than not refusing.

    The presence check runs before the pose is written, so a call that reports failure has not also
    relocated the thing it refused to spawn.
    """
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    before = np.array(ctx.data.xpos[bid])

    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="robot",
        pose={"position": {"x": 9.0, "y": 9.0, "z": 0.5}, "orientation": {"yaw": 0.0}},
    )
    action.update()
    _step(ctx, clock)
    assert action.update() is FAILURE
    assert np.allclose(ctx.data.xpos[bid][:2], before[:2], atol=1e-6)


def test_spawn_raises_on_an_entity_the_world_never_declared(teleport_world):
    """Activation, not creation: there is nothing to make appear, and saying so beats inventing it."""
    _ctx, clock, sim = teleport_world
    action = _start(
        SpawnEntity(),
        sim,
        clock,
        entity="ghost",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"yaw": 0.0}},
    )
    with pytest.raises(ActionError, match="ACTIVATES what the world already declares"):
        action.update()


# -- set_sensor_override --------------------------------------------------------------------------
#
# The report channel's action, driven through the SAME access seam as set_model_override. A sensor
# publishes its handle under `sensor_fault:<address>` rather than `model_override:<name>`, which is
# the only thing that differs in-process -- so these tests are mostly about proving that, and about
# the two verdicts a scenario is allowed to act on.


def _sensor(ctx, fault, address="rig.lidar", nominal=None):
    """A lidar carrying a `fault:` block, configured far enough to publish its handle.

    Built directly rather than through an Engine: the action only ever touches the blackboard handle,
    so a full world would be scaffolding around the one seam under test.
    """
    from roqsim_sensors.plugins.lidar import LidarPlugin

    entity, _, label = address.rpartition(".")
    cfg = {"site": "sensor_site", "rays": 8, "exclude_body": "", "fault": dict(fault)}
    cfg.update(nominal or {})
    plugin = LidarPlugin(cfg, name=label, entity=entity or None, label=label)
    plugin.register_fault_endpoints(ctx, namespace="")
    return plugin


def _sensor_handle(ctx, address="rig.lidar"):
    from roqsim_sensors.live_config import blackboard_key

    return ctx.blackboard.get(blackboard_key(address))


def test_a_sensor_fault_is_applied_and_the_verdict_read_back(world):
    ctx, clock, sim = world
    _sensor(ctx, fault={"dropout_percent": 60.0}, nominal={"dropout_percent": 2.0})

    action = _start(
        SetSensorOverride(), sim, clock, instance="rig.lidar", active=True, require_landed=True
    )
    assert action.update() is RUNNING, "the write is queued, not yet applied"
    assert _sensor_handle(ctx).is_active() is False

    _step(ctx, clock)
    assert _sensor_handle(ctx).is_active() is True
    assert action.update() is SUCCESS
    assert "landed" in action.feedback_message


def test_a_sensor_fault_that_changed_nothing_fails_the_trial(world):
    """A `fault:` block restating the nominal leaves a run recorded as faulted that was not."""
    ctx, clock, sim = world
    _sensor(ctx, fault={"dropout_percent": 2.0}, nominal={"dropout_percent": 2.0})

    action = _start(
        SetSensorOverride(), sim, clock, instance="rig.lidar", active=True, require_landed=True
    )
    action.update()
    _step(ctx, clock)
    assert action.update() is FAILURE
    assert "no_effect" in action.feedback_message or "changed nothing" in action.feedback_message


def test_an_unknown_sensor_address_names_what_the_world_offers(world):
    """A bare `lidar` against an owned sensor is the mistake the address exists to prevent."""
    ctx, clock, sim = world
    _sensor(ctx, fault={"dropout_percent": 60.0})

    action = _start(
        SetSensorOverride(), sim, clock, instance="lidar", active=True, require_landed=True
    )
    with pytest.raises(Exception) as err:
        action.update()
    assert "rig.lidar" in str(err.value), "the refusal must name the address that does exist"


def test_an_empty_sensor_address_is_refused_at_execute(world):
    _, clock, sim = world
    action = SetSensorOverride()
    action.name = "set_sensor_override"
    action.setup(simulation=sim, clock=clock, action_name="set_sensor_override")
    with pytest.raises(Exception) as err:
        action.execute(instance="", active=True, require_landed=True)
    assert "COMPONENT ADDRESS" in str(err.value)


# -- entity_navigate ------------------------------------------------------------------------------
class FakeNavHandle:
    """A navigator's handle, with the sequence-number contract and nothing else.

    The action is written against that contract, not against a navigator, so this is what it should
    be tested against: a stub makes the preemption and stale-completion cases reachable, which they
    are not with a real mover without contriving a race.
    """

    def __init__(self):
        self.applied = 0
        self.finished = False
        self.routes: list[list] = []
        self.cancels = 0
        self.starts = 0
        self._next = 0

    def _stamp(self) -> int:
        self._next += 1
        return self._next

    def send_goals(self, goals) -> int:
        self.routes.append(list(goals))
        return self._stamp()

    def start(self) -> int:
        self.starts += 1
        return self._stamp()

    def cancel(self) -> int:
        self.cancels += 1
        return self._stamp()

    def status(self):
        return self.applied, self.finished, 0, 0.0

    def apply(self, seq: int, *, finished: bool = False) -> None:
        """What the physics thread does when a posted route lands."""
        self.applied, self.finished = seq, finished


def _nav(ctx, entity="parcel"):
    handle = FakeNavHandle()
    ctx.blackboard.set(f"nav:{entity}:handle", handle)
    return handle


def _pose(x, y, yaw=0.0):
    return {"position": {"x": x, "y": y, "z": 0.0}, "orientation": {"yaw": yaw}}


def test_a_route_is_sent_and_the_action_waits_for_arrival(world):
    ctx, clock, sim = world
    handle = _nav(ctx)
    action = _start(EntityNavigate(), sim, clock, entity="parcel", goal_poses=[_pose(2.0, 1.0)])

    assert action.update() == py_trees.common.Status.RUNNING
    assert handle.routes == [[(2.0, 1.0)]]

    handle.apply(1)  # applied, still driving
    assert action.update() == py_trees.common.Status.RUNNING

    handle.apply(1, finished=True)
    assert action.update() == py_trees.common.Status.SUCCESS


def test_it_does_not_report_an_arrival_before_the_route_is_even_applied(world):
    """The failure the sequence number exists to prevent.

    The navigator is idle and 'finished' when the route is queued -- it completed whatever it was
    doing before. An action watching that flag alone would succeed instantly, having driven nothing.
    """
    ctx, clock, sim = world
    handle = _nav(ctx)
    handle.apply(0, finished=True)  # a previous route's completion, still latched
    action = _start(EntityNavigate(), sim, clock, entity="parcel", goal_poses=[_pose(1.0, 0.0)])
    assert action.update() == py_trees.common.Status.RUNNING


def test_a_newer_route_preempts_this_one_and_fails_the_branch(world):
    ctx, clock, sim = world
    handle = _nav(ctx)
    action = _start(EntityNavigate(), sim, clock, entity="parcel", goal_poses=[_pose(1.0, 0.0)])
    action.update()
    handle.apply(7, finished=True)  # somebody else's route, and it finished
    assert action.update() == py_trees.common.Status.FAILURE


def test_success_on_acceptance_does_not_wait_for_arrival(world):
    """Fire-and-forget traffic: the scenario wants the mover moving, not to watch it."""
    ctx, clock, sim = world
    handle = _nav(ctx)
    action = _start(
        EntityNavigate(),
        sim,
        clock,
        entity="parcel",
        goal_poses=[_pose(5.0, 5.0)],
        success_on_acceptance=True,
    )
    action.update()
    handle.apply(1)  # applied but nowhere near arrived
    assert action.update() == py_trees.common.Status.SUCCESS


def test_start_runs_the_configured_route_and_sends_no_goals(world):
    ctx, clock, sim = world
    handle = _nav(ctx)
    action = _start(EntityNavigateStart(), sim, clock, entity="parcel")
    action.update()
    assert handle.starts == 1
    assert handle.routes == [], "it sent a route instead of starting the configured one"
    handle.apply(1, finished=True)
    assert action.update() == py_trees.common.Status.SUCCESS


def test_an_abandoned_branch_stops_the_mover(world):
    """Otherwise the opponent keeps driving across the robot's path long after the phase ended."""
    ctx, clock, sim = world
    handle = _nav(ctx)
    action = _start(EntityNavigate(), sim, clock, entity="parcel", goal_poses=[_pose(9.0, 9.0)])
    action.update()
    assert action.request_cancel() is True
    assert handle.cancels == 1


def test_an_entity_with_no_navigator_raises_and_says_what_the_world_offers(world):
    ctx, clock, sim = world
    _nav(ctx, entity="cart")  # a different entity can navigate
    action = _start(EntityNavigate(), sim, clock, entity="parcel", goal_poses=[_pose(1.0, 0.0)])
    with pytest.raises(ActionError, match="has no navigator"):
        action.update()


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
def test_a_goal_orientation_is_refused_rather_than_dropped(world, axis):
    """The navigator drives to a position; it has no final-heading control.

    Accepting an orientation and discarding it would let a scenario believe it had set where the
    mover ends up facing. Yaw is refused for the same reason as roll and pitch, and the message says
    that adding the capability is the fix rather than passing the value.
    """
    ctx, clock, sim = world
    _nav(ctx)
    action = EntityNavigate()
    action.setup(simulation=sim, clock=clock)
    goal = _pose(1.0, 0.0)
    goal["orientation"][axis] = 0.3
    with pytest.raises(ActionError, match="orientation is nonzero"):
        action.execute(entity="parcel", goal_poses=[goal])


# -- what an action says while it is waiting ---------------------------------------------------


def test_a_waiting_action_names_what_is_missing_when_the_call_can_say():
    """ "Still running" has two causes, and only one of them is progress.

    A queued write drains next step; a service nobody serves never answers. They look identical
    until the scenario's timeout fires, at which point the trial has spent its budget and reports
    only that it ran out -- which is exactly what a world serving no `sim_interfaces` looked like.
    """
    from scenario_execution_roqsim.access import PendingCall

    class _Silent(PendingCall):
        def poll(self):
            return None

    class _Explains(PendingCall):
        def poll(self):
            return None

        def pending_reason(self):
            return "the simulator is not advertising 'set_entity_state'"

    action = SetEntityState()
    assert action.waiting("setting state", _Silent()) is RUNNING
    assert action.feedback_message == "setting state", "progress needs no explanation"

    assert action.waiting("setting state", _Explains()) is RUNNING
    assert "not advertising 'set_entity_state'" in action.feedback_message


def test_waiting_without_a_call_is_unchanged():
    """The bare form still works: not every wait has a call behind it."""
    action = SetEntityState()
    assert action.waiting("waiting for the simulation") is RUNNING
    assert action.feedback_message == "waiting for the simulation"


def test_every_call_type_can_be_asked_why_it_is_waiting():
    """The contract is shared, so an action never has to know which kind of call it holds --
    which is how the nav call was missed when the question was first added."""
    from scenario_execution_roqsim.access import NavCall, OverrideCall, SpawnCall, TeleportCall

    for cls in (OverrideCall, TeleportCall, SpawnCall, NavCall):
        assert hasattr(cls, "pending_reason"), cls.__name__
