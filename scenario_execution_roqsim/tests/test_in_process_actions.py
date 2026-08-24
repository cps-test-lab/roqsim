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
from scenario_execution_roqsim.actions.entity_moved import EntityMoved  # noqa: E402
from scenario_execution_roqsim.actions.entity_rotated import EntityRotated  # noqa: E402
from scenario_execution_roqsim.actions.entity_teleport import EntityTeleport  # noqa: E402
from scenario_execution_roqsim.actions.set_model_override import SetModelOverride  # noqa: E402
from scenario_execution_roqsim.actions.set_sensor_override import SetSensorOverride  # noqa: E402

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
        EntityMoved(), sim, clock,
        entities=["parcel"], threshold=0.05, mode="z", dwell=0.0, require="all",
    )
    assert action.update() is RUNNING, "0.4 m above the floor is not 0.05 m of MOVEMENT"
    assert "parcel" in action.feedback_message


def test_it_succeeds_once_the_entity_has_actually_moved(world):
    """The crate slides down the ramp; the action fires when the displacement passes the threshold."""
    ctx, clock, sim = world
    action = _start(
        EntityMoved(), sim, clock,
        entities=["parcel"], threshold=0.05, mode="distance", dwell=0.0, require="all",
    )
    assert action.update() is RUNNING
    for _ in range(2000):
        _step(ctx, clock)
        if action.update() is SUCCESS:
            break
    assert action.update() is SUCCESS
    moved = float(np.linalg.norm(ctx.data.xpos[mujoco.mj_name2id(
        ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")] - np.array([0.0, 0.0, 0.4])))
    assert moved >= 0.05


def test_the_dwell_is_measured_on_the_runners_clock_and_restarts_on_a_dip(world):
    """A crossing flatters the result, so the condition must hold CONTINUOUSLY for `dwell`.

    Driven synthetically rather than by physics: what is under test is the dwell bookkeeping, and a
    real dip would take a scene built to bounce.
    """
    ctx, clock, sim = world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    action = _start(
        EntityMoved(), sim, clock,
        entities=["parcel"], threshold=0.05, mode="z", dwell=1.0, require="all",
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
        EntityMoved(), sim, clock,
        entities=["parcell"], threshold=0.05, mode="z", dwell=0.0, require="all",
    )
    with pytest.raises(ActionError, match="parcel"):
        action.update()


def test_a_welded_entity_raises_rather_than_waiting_forever(world):
    """Its pose is a compile-time constant, so "wait until it moves" can never be satisfied."""
    ctx, clock, sim = world
    ctx.entities.add(Entity(name="ramp_entity", kind="prop", body="world"))
    action = _start(
        EntityMoved(), sim, clock,
        entities=["ramp_entity"], threshold=0.05, mode="distance", dwell=0.0, require="all",
    )
    with pytest.raises(ActionError, match="welded to the world"):
        action.update()


def test_it_waits_rather_than_building_a_world(world):
    """The tree is set up before the first reset, so `context` is None and the action must wait."""
    _ctx, clock, _sim = world

    class NotBuilt:
        context = None

    action = _start(
        EntityMoved(), NotBuilt(), clock,
        entities=["parcel"], threshold=0.05, mode="z", dwell=0.0, require="all",
    )
    assert action.update() is RUNNING
    assert "simulation" in action.feedback_message


# -- entity_rotated -------------------------------------------------------------------------------
def test_rotation_fires_on_the_geodesic_angle(world):
    ctx, clock, sim = world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    action = _start(
        EntityRotated(), sim, clock,
        entities=["parcel"], angle=0.5, dwell=0.0, require="all",
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
    _step(ctx, clock, plugin, seconds=2.0)  # let the crate settle onto the ramp, so a contact exists

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


# -- entity_teleport --------------------------------------------------------------------------------
#
# A separate tiny scene: the shared `world` fixture's freejoint is unnamed (fine for entity_moved,
# which resolves the BODY), and set_entity_pose needs a named joint to write qpos through.
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
  </worldbody>
</mujoco>
"""


@pytest.fixture
def teleport_world():
    model = mujoco.MjModel.from_xml_string(TELEPORT_SCENE)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    ctx.entities.add(Entity(name="robot", kind="robot", body="robot", meta={"base_joint": "robot_free"}))
    # No `base_joint` in meta: exercises the "cannot be teleported" outcome (a static prop).
    ctx.entities.add(Entity(name="prop", kind="object", body="fixed_prop", meta={}))
    return ctx, FakeClock(), FakeSim(ctx)


def test_teleport_places_the_entity_and_zeroes_its_velocity(teleport_world):
    ctx, clock, sim = teleport_world
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    dof = ctx.model.jnt_dofadr[jid]
    ctx.data.qvel[dof:dof + 6] = 1.0  # nonzero, so the teleport's zeroing is actually exercised

    action = _start(
        EntityTeleport(), sim, clock, entity="robot",
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
    assert np.allclose(ctx.data.qvel[dof:dof + 6], 0.0, atol=0.03)


def test_teleport_fails_the_trial_rather_than_raise_when_the_entity_has_no_free_joint(teleport_world):
    """A static prop is a fact about the world the campaign chose, not a malformed call."""
    ctx, clock, sim = teleport_world
    action = _start(
        EntityTeleport(), sim, clock, entity="prop",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"yaw": 0.0}},
    )
    assert action.update() is RUNNING
    _step(ctx, clock)
    assert action.update() is FAILURE
    assert "no free joint" in action.feedback_message


def test_teleport_raises_on_an_unknown_entity(teleport_world):
    _ctx, clock, sim = teleport_world
    action = _start(
        EntityTeleport(), sim, clock, entity="ghost",
        pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"yaw": 0.0}},
    )
    with pytest.raises(ActionError, match="no entity"):
        action.update()


def test_teleport_rejects_nonzero_roll_or_pitch_at_execute():
    action = EntityTeleport()
    with pytest.raises(ActionError, match="roll or pitch"):
        action.execute(
            entity="robot",
            pose={"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"roll": 0.1, "yaw": 0.0}},
        )


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
