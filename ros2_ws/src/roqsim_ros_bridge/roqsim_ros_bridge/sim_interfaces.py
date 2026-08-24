"""Control-plane plugin: a subset of ros-simulation/simulation_interfaces.

Implements the interfaces most useful for scenario-driven testing of the M1 turtlebot world:
  * GetSimulatorFeatures   — advertise what is supported
  * GetEntities            — list entities from the registry
  * GetEntityState/SetEntityState — read/teleport an entity's free-joint body
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
            "roqsim bridge (M1 subset); spawn/delete activate entities the world "
            "compiled -- the model is never rebuilt at runtime"
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
        """Make an entity the world compiled perceivable again.

        Not creation. roqsim does not recompile the model at runtime, so there is no body to add:
        a world declares everything a trial may bring in, and this selects one. A request for a
        name the world does not carry is refused rather than approximated -- the alternative is
        a trial that believes it spawned something.
        """
        return self._set_presence(req.name, True, resp, verb="spawn")

    def _delete_entity(self, req, resp):
        """Make an entity absent: invisible to sensors, untouchable, and unlisted.

        Its pose does not move. Parking it out of sight instead would leave a free body
        accelerating under gravity for as long as it stays away, so it would return with
        whatever velocity it had accumulated.
        """
        return self._set_presence(req.entity, False, resp, verb="delete")

    def _set_presence(self, name, present, resp, *, verb):
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
        if not run_on_physics(
            self._ctx, lambda ctx: set_present(ctx, ctx.entities.get(name), present)
        ):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"the simulation did not apply {verb} {name!r} within "
                f"{DEFAULT_TIMEOUT_S} s (is it paused?)",
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
