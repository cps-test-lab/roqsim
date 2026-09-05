"""Control-plane plugin: a subset of ros-simulation/simulation_interfaces.

Implements the interfaces most useful for scenario-driven testing of the M1 turtlebot world:
  * GetSimulatorFeatures   — advertise what is supported
  * GetEntities            — list entities from the registry
  * GetEntityState/SetEntityState — read/teleport an entity's free-joint body
  * SpawnEntity/DeleteEntity — make a compiled entity perceivable at initial_pose, or absent
  * GetSimulationState/SetSimulationState — play/pause/stop (standalone driver)
  * StepSimulation         — step N times while paused
  * ResetSimulation        — reset the world

Concurrency: services run on the bridge/executor thread. State changes go through the shared
:class:`roqsim.control.RunControl`; anything touching ``data`` goes through
:func:`roqsim_ros_bridge.physics.run_on_physics`, which posts to the physics thread and waits (see
docs/architecture.rst §7). Every mutating service *waits*: answering ``RESULT_OK`` before the change
has run makes a paused simulator indistinguishable from a working one.

Reuses the ``rclpy`` node created by :class:`~roqsim_ros_bridge.ros2_bridge.Ros2Bridge` when present
(looked up on the blackboard under ``ros2_node``); otherwise creates and spins its own.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from simulation_interfaces.msg import Result, SimulationState, SimulatorFeatures
from simulation_interfaces.srv import (
    DeleteEntity,
    GetEntities,
    GetEntityState,
    GetSimulationState,
    GetSimulatorFeatures,
    ResetSimulation,
    SetEntityState,
    SetSimulationState,
    SpawnEntity,
    StepSimulation,
)

from roqsim import control as ctl
from roqsim.plugin import Plugin
from roqsim.presence import set_present

from .physics import DEFAULT_TIMEOUT_S, run_on_physics

_STATE_TO_MSG = {
    ctl.STOPPED: SimulationState.STATE_STOPPED,
    ctl.PLAYING: SimulationState.STATE_PLAYING,
    ctl.PAUSED: SimulationState.STATE_PAUSED,
    ctl.QUITTING: SimulationState.STATE_QUITTING,
}
_MSG_TO_STATE = {v: k for k, v in _STATE_TO_MSG.items()}

#: How far a welded entity's pose may differ from a requested one and still count as satisfied.
#: Tight, because it exists to absorb float round-trips through the message, not to accept a pose
#: somewhere else.
_POSE_EPS = 1e-6

#: Frame names that mean the world frame. Empty is the service's own default for it; 'world' is
#: spelled out because that is the name the service description gives that frame.
_WORLD_FRAMES = ("", "world")


def _pose_of(req):
    """``initial_pose`` as ``(pos, quat)``, or ``None`` when the quaternion is not a rotation.

    ``SpawnEntity.srv`` states ``initial_pose`` unconditionally -- there is no "unset": a
    default-constructed request carries the origin and, because ``geometry_msgs/Quaternion``
    declares ``w 1``, the IDENTITY. So every request asks for a pose, and this reads the one it
    asks for rather than guessing which fields the caller meant to fill in.

    A zero-norm quaternion is the one thing that cannot be a rotation, and the service has a code
    saying so (``INVALID_POSE``), so it is reported rather than repaired.
    """
    p = req.initial_pose.pose.position
    o = req.initial_pose.pose.orientation
    norm = math.sqrt(o.w * o.w + o.x * o.x + o.y * o.y + o.z * o.z)
    if norm < _POSE_EPS:
        return None
    # Normalised because MuJoCo reads the free joint's quaternion as a unit one; a caller's
    # near-unit value would otherwise scale the body's orientation.
    return (p.x, p.y, p.z), (o.w / norm, o.x / norm, o.y / norm, o.z / norm)


def _already_at(state, pose) -> bool:
    """Is the body described by *state* already at *pose*?

    What an entity with no free joint is asked instead of "can you move there": it cannot, so the
    only honourable answer is whether it is there already. The quaternion is compared through its
    dot product because ``q`` and ``-q`` are the same rotation, which a component-wise test would
    call a mismatch.
    """
    pos, quat = pose
    at = state.get("pos")
    rot = state.get("quat")
    if at is None or rot is None:
        return False
    if any(abs(a - b) > _POSE_EPS for a, b in zip(at, pos, strict=True)):
        return False
    return abs(sum(a * b for a, b in zip(rot, quat, strict=True))) > 1.0 - _POSE_EPS


def _unsupported_spawn_request(req):
    """``(result_code, message)`` for a request this simulator cannot serve as asked, else ``None``.

    EVERY field of the request is either honoured or named here, and that is the point rather than
    tidiness: each one used to be read off the request and discarded under a ``RESULT_OK``, which
    is the failure this module is shaped against -- a trial that believes it spawned something. A
    field added to the service later must join one list or the other.

    The codes are the service's OWN extended ones where it defines a fitting one, which
    ``Result.msg`` asks of an implementation rather than leaving to taste; the generic
    ``RESULT_FEATURE_UNSUPPORTED`` is for a call option this simulator does not offer, which is
    what that code is for.

    Why none of these is a gap to fill: geometry, because the model is compiled once, which
    ``get_simulator_features`` already says by advertising no ``spawn_formats``; a namespace,
    because an entity's name is settled when the model compiles; renaming, because spawning here
    SELECTS an entity by name, so an existing name is the required case rather than the collision
    ``allow_renaming`` resolves; and a frame, because the service requires one the simulator
    knows, and this one knows only the world frame the empty default already names.
    """
    if getattr(req, "uri", "") or getattr(req, "resource_string", ""):
        return SpawnEntity.Response.UNSUPPORTED_FORMAT, (
            "this simulator spawns entities the world compiled and loads no geometry, so 'uri' / "
            "'resource_string' cannot be honoured (get_simulator_features advertises no "
            "spawn_formats). Declare the entity in the world instead."
        )
    if getattr(req, "entity_namespace", ""):
        return Result.RESULT_FEATURE_UNSUPPORTED, (
            "'entity_namespace' cannot be honoured: an entity's name is settled when the model "
            "compiles, so this simulator cannot place one under a namespace. Name it in the world."
        )
    if getattr(req, "allow_renaming", False):
        return Result.RESULT_FEATURE_UNSUPPORTED, (
            "'allow_renaming' cannot be honoured: spawning here selects the entity the world "
            "compiled under this name, so an existing name is what the request needs rather than "
            "a collision to rename around. Ask for the name you want."
        )
    frame_id = getattr(getattr(req.initial_pose, "header", None), "frame_id", "")
    if frame_id and frame_id not in _WORLD_FRAMES:
        return SpawnEntity.Response.INVALID_POSE, (
            f"initial_pose is stated in frame {frame_id!r}, which this simulator does not know. "
            "It places entities in the world frame, which the empty default already names."
        )
    return None


class SimInterfacesPlugin(Plugin):
    # Pure service surface over an already-built world -- no geometry, no state of its own -- so a
    # scene consumer (render, review, export) may drop it. See Plugin.transport_only.
    transport_only = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._ctx = None
        self._node: Node | None = None
        self._own_node = False
        self._executor = None
        self._thread = None
        self._we_inited_rclpy = False

    def configure(self, ctx) -> None:
        self._ctx = ctx
        node = ctx.blackboard.get("ros2_node")
        if node is None:
            if not rclpy.ok():
                rclpy.init()
                self._we_inited_rclpy = True
            node = Node(self.config.get("node_name", "roqsim_interfaces"))
            self._own_node = True
        self._node = node

        node.create_service(GetSimulatorFeatures, "get_simulator_features", self._get_features)
        node.create_service(GetEntities, "get_entities", self._get_entities)
        # Spawning here is ACTIVATION, not creation: the model is compiled once and never
        # rebuilt, so these two make an entity the world already carries perceivable or not.
        node.create_service(SpawnEntity, "spawn_entity", self._spawn_entity)
        node.create_service(DeleteEntity, "delete_entity", self._delete_entity)
        node.create_service(GetEntityState, "get_entity_state", self._get_entity_state)
        node.create_service(SetEntityState, "set_entity_state", self._set_entity_state)
        node.create_service(GetSimulationState, "get_simulation_state", self._get_sim_state)
        node.create_service(SetSimulationState, "set_simulation_state", self._set_sim_state)
        node.create_service(StepSimulation, "step_simulation", self._step)
        node.create_service(ResetSimulation, "reset_simulation", self._reset)

        if self._own_node:
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()

    def shutdown(self, ctx) -> None:
        if self._own_node:
            if self._executor is not None:
                self._executor.shutdown()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if self._node is not None:
                self._node.destroy_node()
            if self._we_inited_rclpy and rclpy.ok():
                rclpy.shutdown()

    # -- services -----------------------------------------------------------------------------
    def _get_features(self, req, resp):
        f = SimulatorFeatures()
        f.features = [
            # Both are served over presence, not creation -- see _spawn_entity. Advertised
            # because a caller's question is "can I make this entity appear", and here it can.
            SimulatorFeatures.SPAWNING,
            SimulatorFeatures.DELETING,
            SimulatorFeatures.ENTITY_STATE_GETTING,
            SimulatorFeatures.ENTITY_STATE_SETTING,
            SimulatorFeatures.SIMULATION_RESET,
            SimulatorFeatures.SIMULATION_RESET_STATE,
            SimulatorFeatures.SIMULATION_STATE_GETTING,
            SimulatorFeatures.SIMULATION_STATE_SETTING,
            SimulatorFeatures.SIMULATION_STATE_PAUSE,
            SimulatorFeatures.STEP_SIMULATION_SINGLE,
            SimulatorFeatures.STEP_SIMULATION_MULTIPLE,
        ]
        # No format: nothing is loaded. Spawning selects an entity the world compiled, so a
        # caller offering MJCF would be refused -- saying "mjcf" here would invite exactly that.
        f.spawn_formats = []
        f.custom_info = (
            "roqsim bridge (M1 subset); spawn/delete activate entities the world compiled, "
            "spawn placing a free-jointed one at initial_pose (world frame only) -- the model "
            "is never rebuilt at runtime"
        )
        resp.features = f
        return resp

    def _get_entities(self, req, resp):
        # Present ones only: an absent entity is compiled into the model but nothing can see
        # or touch it, so listing it would make this disagree with every sensor.
        resp.entities = self._ctx.entities.names(present_only=True)
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _spawn_entity(self, req, resp):
        """Make an entity the world compiled perceivable again, at the pose the request states.

        Not creation. roqsim does not recompile the model at runtime, so there is no body to add:
        a world declares everything a trial may bring in, and this selects one. A request for a
        name the world does not carry is refused rather than approximated -- the alternative is
        a trial that believes it spawned something.

        ``initial_pose`` is applied, in the same physics transaction as the presence flip, so the
        entity is never perceivable at a pose nobody asked for. The service states that pose
        unconditionally and a default request carries the origin, so THAT is what a caller sending
        no pose asks for and what it gets; to bring an entity back where it was, state where that
        is. An entity the model welded cannot be moved at all, so for it the request succeeds only
        if it is already at the pose asked for, and is refused otherwise rather than appearing
        somewhere else under a RESULT_OK.
        """
        unsupported = _unsupported_spawn_request(req)
        if unsupported:
            code, message = unsupported
            resp.result = Result(result=code, error_message=message)
            return resp
        pose = _pose_of(req)
        if pose is None:
            resp.result = Result(
                result=SpawnEntity.Response.INVALID_POSE,
                error_message="initial_pose carries a zero-length quaternion, which is not a "
                "rotation. Send a unit quaternion; the identity is w=1.",
            )
            return resp
        return self._set_presence(req.name, True, resp, verb="spawn", pose=pose)

    def _delete_entity(self, req, resp):
        """Make an entity absent: invisible to sensors, untouchable, and unlisted.

        Its pose does not move. Parking it out of sight instead would leave a free body
        accelerating under gravity for as long as it stays away, so it would return with
        whatever velocity it had accumulated.
        """
        return self._set_presence(req.entity, False, resp, verb="delete")

    def _set_presence(self, name, present, resp, *, verb, pose=None):
        entity = self._ctx.entities.get(name)
        if entity is None:
            resp.result = Result(
                result=Result.RESULT_NOT_FOUND,
                error_message=(
                    f"no entity {name!r}. This simulator cannot create one: which entities "
                    "exist is settled when the model compiles, so the world must declare it."
                ),
            )
            return resp
        if bool(entity.present) == bool(present):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"entity {name!r} is already {'present' if present else 'absent'}",
            )
            return resp
        # Physics-thread only, like every other write to model/data -- and WAITED FOR, so RESULT_OK
        # means the entity really has appeared. Posting and answering OK immediately (which this did)
        # reports success before the flip has run, so a paused or stalled simulator accepts spawns
        # that never happen and the caller has no way to tell.
        #
        # Pose first, then presence, in ONE transaction: placing an entity that is already
        # perceivable would show it at the compiled pose for the steps in between.
        #
        # Each outcome is recorded POSITIVELY, never inferred from a flag that stayed unset:
        # run_on_physics sets its event in a `finally`, so a raising command still returns True,
        # and reading "no outcome" as one particular failure would explain an exception as a
        # missing free joint.
        outcome = {}

        def _apply(ctx):
            if pose is not None and not self._write_body(ctx, entity, pose[0], pose[1]):
                state = self._read_body(ctx, entity.body) if entity.body else {}
                if not _already_at(state, pose):
                    outcome["welded_at"] = state.get("pos")
                    return
            set_present(ctx, entity, present)
            outcome["done"] = True

        if not run_on_physics(self._ctx, _apply):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"the simulation did not apply {verb} {name!r} within "
                f"{DEFAULT_TIMEOUT_S} s (is it paused?)",
            )
            return resp
        if "welded_at" in outcome:
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=(
                    f"{verb} {name!r} asks for a pose the entity cannot take: the world compiled "
                    f"it without a free joint, welded at {outcome['welded_at']}. Ask for that "
                    "pose, or give it 'free: true' in the world so it can be placed."
                ),
            )
            return resp
        if not outcome.get("done"):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"{verb} {name!r} did not complete in the simulation; "
                "its log carries the reason.",
            )
            return resp
        if hasattr(resp, "entity_name"):
            resp.entity_name = name
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _get_entity_state(self, req, resp):
        entity = self._ctx.entities.get(req.entity)
        if entity is None or entity.body is None:
            resp.result = Result(
                result=Result.RESULT_NOT_FOUND, error_message=f"no entity {req.entity!r}"
            )
            return resp
        holder = {}
        run_on_physics(self._ctx, lambda c: holder.update(self._read_body(c, entity.body)))
        if not holder:
            resp.result = Result(result=Result.RESULT_OPERATION_FAILED)
            return resp
        s = resp.state
        s.pose.position.x, s.pose.position.y, s.pose.position.z = holder["pos"]
        (s.pose.orientation.w, s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z) = (
            holder["quat"]
        )
        s.twist.linear.x, s.twist.linear.y, s.twist.linear.z = holder["lin"]
        s.twist.angular.x, s.twist.angular.y, s.twist.angular.z = holder["ang"]
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _set_entity_state(self, req, resp):
        entity = self._ctx.entities.get(req.entity)
        if entity is None:
            resp.result = Result(
                result=Result.RESULT_NOT_FOUND, error_message=f"no entity {req.entity!r}"
            )
            return resp
        p = req.state.pose.position
        o = req.state.pose.orientation
        quat = (o.w, o.x, o.y, o.z) if any([o.w, o.x, o.y, o.z]) else (1.0, 0.0, 0.0, 0.0)
        ok = {}
        run_on_physics(
            self._ctx, lambda c: ok.update(done=self._write_body(c, entity, (p.x, p.y, p.z), quat))
        )
        resp.result = Result(
            result=Result.RESULT_OK if ok.get("done") else Result.RESULT_OPERATION_FAILED
        )
        return resp

    def _get_sim_state(self, req, resp):
        resp.state = SimulationState(state=_STATE_TO_MSG[self._ctx.control.state])
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _set_sim_state(self, req, resp):
        target = _MSG_TO_STATE.get(req.state.state)
        if target is None:
            resp.result = Result(result=Result.RESULT_OPERATION_FAILED)
            return resp
        self._ctx.control.set_state(target)
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _step(self, req, resp):
        if self._ctx.control.state != ctl.PAUSED:
            resp.result = Result(
                result=Result.RESULT_INCORRECT_STATE, error_message="step requires PAUSED state"
            )
            return resp
        self._ctx.control.request_steps(max(1, int(req.steps)))
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _reset(self, req, resp):
        self._ctx.control.request_reset()
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    # -- physics-thread helpers ---------------------------------------------------------------
    @staticmethod
    def _read_body(ctx, body_name):
        import mujoco

        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return {}
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(ctx.model, ctx.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        return {
            "pos": [float(v) for v in ctx.data.xpos[bid]],
            "quat": [float(v) for v in ctx.data.xquat[bid]],
            "ang": [float(v) for v in vel[:3]],
            "lin": [float(v) for v in vel[3:]],
        }

    @staticmethod
    def _write_body(ctx, entity, pos, quat) -> bool:
        import mujoco

        jname = entity.meta.get("base_joint")
        jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, jname) if jname else -1
        if jid < 0 or ctx.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            return False
        q = ctx.model.jnt_qposadr[jid]
        ctx.data.qpos[q : q + 3] = pos
        ctx.data.qpos[q + 3 : q + 7] = quat
        dof = ctx.model.jnt_dofadr[jid]
        ctx.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(ctx.model, ctx.data)
        return True

    def validate_config(self, config: dict) -> list[str]:
        return []
