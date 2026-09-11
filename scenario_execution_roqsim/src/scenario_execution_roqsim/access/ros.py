# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The ROS backend: the simulator is another container, so the world is a service graph.

Nothing here is roqsim-specific except the fault endpoint. The pose comes from
``simulation_interfaces/GetEntityState``, a standard service keyed on the ENTITY name -- the same name
the in-process path resolves through ``ctx.entities`` -- so ``entity_moved`` works against any
simulator that serves it, not only this one.

**Service names are relative.** ``get_entity_state`` and ``<instance>/override`` are resolved by rclpy
against the scenario node's namespace, which is what the bridge itself does with its endpoints (a
relative name is scoped, an absolute one is left alone -- ``ros2_bridge._resolve_topic``). So the
default case needs no configuration, and a deployment that runs the bridge under a global namespace
runs the scenario node under the same one, as any ROS system does.

**Nothing blocks.** Every call is ``call_async`` with a done callback, at most one in flight per
target, following ``scenario_execution_ros.actions.ros_service_call``. A blocking call in a tick would
stall the tree; and the scenario node is spun by the runner's executor, so the future completes between
ticks without this module owning a thread.

``rclpy`` and ``simulation_interfaces`` are imported in ``__init__``, not at module scope: this module
is only imported when the ROS runner actually handed the action a node
(:func:`~scenario_execution_roqsim.access.select`), which is what keeps the package installable and
unit-testable in a plain venv.
"""

from __future__ import annotations

import numpy as np

from . import (
    AccessError,
    NavCall,
    NavOutcome,
    OverrideCall,
    OverrideOutcome,
    Pose,
    SpawnCall,
    SpawnOutcome,
    TeleportCall,
    TeleportOutcome,
    WorldAccess,
)


def _also_carries(offered, kind: str = "services") -> str:
    """`` -- the graph carries: a, b, c``, or empty when nothing can be listed.

    The ROS counterpart of the in-process refusal's "this world offers:". A name that is missing
    says which name; the names that ARE there are what show a prefix or a namespace that does not
    match, which is the mistake behind most of them.

    Appended to a REASON, never used to decide anything. When to stop waiting belongs to the
    scenario -- its own ``timeout()`` and the shape of its tree -- and a second clock in here
    would take that decision away from the author and hide it in a library constant.
    """
    names = sorted(offered or ())
    if not names:
        return ""
    listed = ", ".join(names[:20])
    more = f" (+{len(names) - 20} more)" if len(names) > 20 else ""
    return f" -- the graph carries these {kind}: {listed}{more}"


class _RosRoute(NavCall):
    """A navigation goal in flight: a ``NavigateThroughPoses`` or a ``StartRoute``.

    Three stages, none of which may block: wait for the server to appear (the stack and the simulator
    come up concurrently, so "not there yet" means wait, never fail), then for the goal to be
    accepted, then for the result. ``wait=False`` stops after acceptance.
    """

    #: How a world comes to serve the action, for the message while it is not there yet.
    SERVED_BY = "giving that entity a `navigator` component"

    def __init__(self, client, goal, name: str, *, wait: bool, offered=None, served_by=SERVED_BY):
        self._client, self._goal, self._name, self._wait = client, goal, name, wait
        self._offered = offered
        self._served_by = served_by
        self._send = None
        self._handle = None
        self._result = None

    def poll(self):
        if self._send is None:
            if not self._client.server_is_ready():
                return None  # the simulator may still be starting: waiting is not failing
            self._send = self._client.send_goal_async(self._goal)
            return None
        if self._handle is None:
            if not self._send.done():
                return None
            self._handle = self._send.result()
            if not self._handle.accepted:
                return NavOutcome(False, f"{self._name}: the navigator rejected the route")
            if not self._wait:
                return NavOutcome(True, "route accepted")
            self._result = self._handle.get_result_async()
            return None
        if self._result is None or not self._result.done():
            return None
        status = getattr(self._result.result(), "status", None)
        # 4 == STATUS_SUCCEEDED in action_msgs/GoalStatus; anything else is preemption or a give-up,
        # which is a trial fact rather than an authoring one.
        return (
            NavOutcome(True, "arrived")
            if status == 4
            else NavOutcome(False, f"{self._name}: route ended with status {status}")
        )

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel_goal_async()

    def pending_reason(self) -> str | None:
        if self._send is None and not self._client.server_is_ready():
            return (
                f"the simulator is not advertising the action server {self._name!r}. A world "
                f"serves it by {self._served_by}; without that this waits until the scenario's "
                "own timeout" + _also_carries(self._offered and self._offered(), "action servers")
            )
        return None


class RosAccess(WorldAccess):
    transport = "ROS"

    #: Where an entity's ground-truth pose comes from. Relative; see the module docstring.
    ENTITY_STATE_SERVICE = "get_entity_state"
    #: Where a teleport is sent. Same relative-naming rule as ENTITY_STATE_SERVICE.
    SET_ENTITY_STATE_SERVICE = "set_entity_state"
    #: Presence, in both directions. Same relative-naming rule.
    SPAWN_ENTITY_SERVICE = "spawn_entity"
    DELETE_ENTITY_SERVICE = "delete_entity"

    def __init__(self, node):
        try:
            from rclpy.callback_groups import ReentrantCallbackGroup
            from simulation_interfaces.msg import Result
            from simulation_interfaces.srv import (
                DeleteEntity,
                GetEntityState,
                SetEntityState,
                SpawnEntity,
            )
            from std_srvs.srv import SetBool
        except ImportError as err:  # pragma: no cover - only when ROS is genuinely absent
            raise AccessError(
                "this scenario is being run by the ROS runner, but the ROS interfaces this needs are "
                f"not importable ({err}). `simulation_interfaces` (the pose service) and `std_srvs` "
                "(the fault service) both come from the simulator's own ROS distribution -- source "
                "the workspace, or run the stepped runner instead."
            ) from None
        self._node = node
        self._result_ok = Result.RESULT_OK
        self._get_state_type = GetEntityState
        self._set_state_type = SetEntityState
        self._spawn_type = SpawnEntity
        self._delete_type = DeleteEntity
        self._set_bool_type = SetBool
        # nav2_msgs and geometry_msgs are imported lazily in `navigate`, not here: a world with no
        # navigator never needs nav2 installed, and requiring it at construction would make every
        # ROS scenario depend on the stack a subset of them command.
        self._nav_clients: dict = {}
        self._pose_stamped_type = None
        self._group = ReentrantCallbackGroup()
        self._state_client = node.create_client(
            GetEntityState, self.ENTITY_STATE_SERVICE, callback_group=self._group
        )
        self._set_state_client = node.create_client(
            SetEntityState, self.SET_ENTITY_STATE_SERVICE, callback_group=self._group
        )
        self._spawn_client = node.create_client(
            SpawnEntity, self.SPAWN_ENTITY_SERVICE, callback_group=self._group
        )
        self._delete_client = node.create_client(
            DeleteEntity, self.DELETE_ENTITY_SERVICE, callback_group=self._group
        )
        self._override_clients: dict[str, object] = {}
        #: entity -> (last known Pose | None, future in flight | None)
        self._poses: dict[str, Pose | None] = {}
        self._inflight: dict[str, object] = {}

    def ready(self) -> bool:
        # True regardless of whether the simulator is up: an unavailable service is handled per call
        # (the reply simply has not arrived), and refusing here would make a scenario that starts
        # before the simulator fail rather than wait -- which is the ordinary case in a ROS run.
        return True

    # -- the world ------------------------------------------------------------------------------
    def entity_pose(self, name: str) -> Pose | None:
        """The last pose the simulator reported, and a fresh request if none is in flight.

        Returns the LAST KNOWN pose rather than only fresh ones: the tree may tick faster than the
        round-trip, and a caller that saw ``None`` on those ticks would keep re-arming its own dwell.
        The cost is that a reported crossing can be one round-trip old, which is the resolution
        statement in the package docstring.
        """
        future = self._inflight.get(name)
        if future is None:
            self._request(name)
        elif future.done():
            self._inflight.pop(name, None)
            self._store(name, future)
            self._request(name)
        return self._poses.get(name)

    def _request(self, name: str) -> None:
        if not self._state_client.service_is_ready():
            # No simulator yet (or no `sim_interfaces` plugin in its world). Not an error here: the
            # caller waits. If it never appears, the scenario's own timeout() is what says so.
            return
        req = self._get_state_type.Request()
        req.entity = name
        self._inflight[name] = self._state_client.call_async(req)

    def _store(self, name: str, future) -> None:
        resp = future.result()
        if resp is None:  # pragma: no cover - a cancelled/failed call; try again next tick
            return
        if int(resp.result.result) != int(self._result_ok):
            raise AccessError(
                f"the simulator does not know an entity called {name!r} "
                f"({resp.result.error_message or 'GetEntityState reported ' + str(resp.result.result)})"
                ". The name is the world's `name:` for that spawn, the same one a stepped run "
                "resolves -- not a TF frame and not a body name."
            )
        p, q = resp.state.pose.position, resp.state.pose.orientation
        self._poses[name] = Pose(
            pos=np.array([p.x, p.y, p.z]),
            # (w, x, y, z), matching MuJoCo's xquat order and the order the bridge fills it in.
            quat=np.array([q.w, q.x, q.y, q.z]),
        )

    # -- the fault ------------------------------------------------------------------------------
    #: How long a graph listing is reused. A pending reason is rebuilt on every tick the action
    #: is RUNNING, and asking the graph each time would be a discovery round-trip per tick for a
    #: string nobody reads until the run ends. It is a diagnostic, so a slightly stale list is
    #: worth far more than an exact one that costs the tick.
    OFFERED_CACHE_S = 5.0

    def _advertised(self, fetch, slot: str):
        """A cached listing of what the graph carries, for a reason string. Never a decision."""
        import time

        now = time.monotonic()
        stamp, names = getattr(self, slot, (0.0, ()))
        if now - stamp < self.OFFERED_CACHE_S:
            return names
        try:
            names = tuple(name for name, _types in fetch())
        except Exception:  # noqa: BLE001 - a listing that fails must not break the reason
            names = ()
        setattr(self, slot, (now, names))
        return names

    def _advertised_services(self):
        return self._advertised(self._node.get_service_names_and_types, "_svc_cache")

    def _advertised_actions(self):
        from rclpy.action import get_action_names_and_types

        return self._advertised(lambda: get_action_names_and_types(node=self._node), "_act_cache")

    def apply_override(self, instance: str, active: bool, kind: str = "model") -> OverrideCall:
        """Call ``<instance>/override``. The REPLY is the outcome -- that is why it is a service.

        *kind* is accepted and unused: both channels scope their endpoint the same way, so over ROS
        there is nothing to distinguish. A component ADDRESS arrives dotted (``robot.lidar``) and is
        translated to slashes here, because a dot is not legal in a ROS name -- one translation, at
        the boundary that owns the naming.

        The bridge's ``std_srvs/SetBool`` handler barriers on the physics thread twice (once for the
        write, once for the step that verifies it) and answers with the plugin's own verdict in
        ``message``, and ``success = verified != "no_effect"``. So the two-phase wait this backend
        needs is exactly the service future, with no state to reconstruct on this side.
        """
        service = instance.replace(".", "/")
        client = self._override_clients.get(service)
        if client is None:
            client = self._node.create_client(
                self._set_bool_type, f"{service}/override", callback_group=self._group
            )
            self._override_clients[service] = client
        req = self._set_bool_type.Request()
        req.data = bool(active)
        return _RosCall(client, req, instance, bool(active), self._advertised_services)

    # -- navigation --------------------------------------------------------------------------------
    def navigate(self, name: str, goal_poses, *, wait: bool, action_name: str = "") -> NavCall:
        """Send an action goal to the simulator's own ``NavigateThroughPoses`` server.

        The same endpoint the in-process path reaches directly -- the bridge serves the navigator's
        goal endpoint as an action, so the two transports command one mover through one seam.

        The default name follows the convention the rest of this backend uses: the entity's own
        namespace, then the endpoint (``<entity>/navigate_through_poses``), exactly as
        ``apply_override`` reaches ``<instance>/override``. ``action_name`` is the escape hatch for a
        deployment that scoped it differently; leaving it empty is what keeps a scenario identical on
        both transports.

        Not ``simulation_interfaces``: that control plane has no navigation service, and inventing
        one there would put the same capability behind two different names.
        """
        if not goal_poses:
            raise AccessError(
                f"no goal poses for {name!r}: a route needs at least one. Running the route the "
                "entity was configured with is `start_route` (`entity_navigate_start`)."
            )
        try:
            from geometry_msgs.msg import PoseStamped  # noqa: PLC0415
            from nav2_msgs.action import NavigateThroughPoses  # noqa: PLC0415
            from rclpy.action import ActionClient  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - only without nav2 installed
            raise AccessError(
                f"entity_navigate over ROS needs nav2_msgs, which is not importable ({err}). The "
                "simulator serves its navigator as a nav2 NavigateThroughPoses action, so the "
                "scenario side needs those message types -- source a workspace with nav2, or run "
                "the stepped runner, where no ROS types are involved at all."
            ) from None
        self._pose_stamped_type = PoseStamped

        topic = action_name or f"{name}/navigate_through_poses"
        client = self._action_client(ActionClient, NavigateThroughPoses, topic)
        goal = NavigateThroughPoses.Goal()
        for point in goal_poses:
            pose = self._pose_stamped_type()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            # Identity orientation, always: the navigator drives to a position and stops facing the
            # way it arrived. `entity_navigate` refuses a goal orientation rather than sending one
            # that would be ignored at the far end, so there is never a heading to encode here.
            pose.pose.orientation.w = 1.0
            goal.poses.append(pose)
        return _RosRoute(client, goal, name, wait=wait, offered=self._advertised_actions)

    def start_route(self, name: str, *, wait: bool, action_name: str = "") -> NavCall:
        """Send a goal to the navigator's own ``StartRoute`` server, at ``<entity>/start_route``.

        A type of its own rather than an empty ``NavigateThroughPoses``: a nav2 goal with no poses
        has nowhere to go, and giving it a second meaning would leave one message meaning two
        things. The navigator serves it only when it has a configured route, so an entity without
        one leaves this waiting on a server that never appears, and the reason says so.
        """
        try:
            from rclpy.action import ActionClient  # noqa: PLC0415
            from roqsim_nav_interfaces.action import StartRoute  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - only without the interfaces built
            raise AccessError(
                f"entity_navigate_start over ROS needs roqsim_nav_interfaces, which is not "
                f"importable ({err}). The simulator serves a configured route as its StartRoute "
                "action, so the scenario side needs that message type -- source a workspace that "
                "built roqsim's ros2_ws, or run the stepped runner, where no ROS types are involved."
            ) from None
        topic = action_name or f"{name}/start_route"
        client = self._action_client(ActionClient, StartRoute, topic)
        return _RosRoute(
            client,
            StartRoute.Goal(),
            name,
            wait=wait,
            offered=self._advertised_actions,
            served_by="giving that entity a `navigator` component with a configured route (`goals:`)",
        )

    def _action_client(self, client_type, action_type, topic: str):
        """One client per (type, name), kept for the run: an action reached twice reuses it."""
        key = (action_type, topic)
        client = self._nav_clients.get(key)
        if client is None:
            client = client_type(self._node, action_type, topic, callback_group=self._group)
            self._nav_clients[key] = client
        return client

    # -- teleport ---------------------------------------------------------------------------------
    def set_entity_state(
        self, name: str, pos: np.ndarray, quat: np.ndarray, lin=None, ang=None
    ) -> TeleportCall:
        """Call ``set_entity_state``. The REPLY is the outcome, same shape as ``apply_override``."""
        req = self._set_state_type.Request()
        req.entity = name
        p, q = req.state.pose.position, req.state.pose.orientation
        p.x, p.y, p.z = (float(v) for v in pos)
        q.w, q.x, q.y, q.z = (float(v) for v in quat)
        # Stated even when zero: `EntityState` carries a twist unconditionally, so the request
        # says what the entity's velocity is to become rather than leaving it to the simulator.
        tl, ta = req.state.twist.linear, req.state.twist.angular
        tl.x, tl.y, tl.z = (float(v) for v in (lin if lin is not None else (0.0, 0.0, 0.0)))
        ta.x, ta.y, ta.z = (float(v) for v in (ang if ang is not None else (0.0, 0.0, 0.0)))
        return _RosTeleport(
            self._set_state_client, req, name, self._result_ok, self._advertised_services
        )

    def set_entity_presence(self, name: str, present: bool, pos=None, quat=None) -> SpawnCall:
        """``spawn_entity`` / ``delete_entity``. The REPLY is the outcome, as everywhere here."""
        if not present:
            if pos is not None or quat is not None:
                raise AccessError(
                    "making an entity absent takes no pose: DeleteEntity states none, and an "
                    "absent entity keeps the pose it had so it can come back where it was."
                )
            req = self._delete_type.Request()
            req.name = name
            return _RosSpawn(
                self._delete_client,
                req,
                name,
                self._result_ok,
                "delete_entity",
                self._advertised_services,
            )
        if pos is None or quat is None:
            raise AccessError(
                "spawning an entity needs a pose: SpawnEntity states `initial_pose` "
                "unconditionally, so sending none would ask for the origin rather than for "
                "wherever the entity is."
            )
        req = self._spawn_type.Request()
        req.name = name
        p, q = req.initial_pose.pose.position, req.initial_pose.pose.orientation
        p.x, p.y, p.z = (float(v) for v in pos)
        q.w, q.x, q.y, q.z = (float(v) for v in quat)
        return _RosSpawn(
            self._spawn_client,
            req,
            name,
            self._result_ok,
            "spawn_entity",
            self._advertised_services,
        )

    def teardown(self) -> None:
        for client in [
            self._state_client,
            self._set_state_client,
            self._spawn_client,
            self._delete_client,
            *self._override_clients.values(),
        ]:
            try:
                self._node.destroy_client(client)
            except Exception:  # noqa: BLE001 - teardown never fails a scenario
                pass


class _RosCall(OverrideCall):
    """One ``SetBool`` round-trip. Sent on the first poll, so nothing is in flight before it is due."""

    def __init__(self, client, request, instance: str, active: bool, offered=None):
        self._client = client
        self._request = request
        self._instance = instance
        self._active = active
        self._future = None
        self._offered = offered

    def poll(self) -> OverrideOutcome | None:
        if self._future is None:
            if not self._client.service_is_ready():
                # The simulator's bridge has not advertised it yet. Waiting beats failing: in a ROS
                # run the stack and the simulator come up concurrently.
                return None
            self._future = self._client.call_async(self._request)
            return None
        if not self._future.done():
            return None
        resp = self._future.result()
        if resp is None:  # pragma: no cover
            raise AccessError(f"the call to {self._instance}/override was dropped")
        verdict = str(resp.message or "")
        return OverrideOutcome(
            ok=bool(resp.success),
            verified=verdict,
            detail=f"{self._instance}/override replied {verdict!r}",
        )

    def pending_reason(self) -> str | None:
        if self._future is None and not self._client.service_is_ready():
            return (
                f"the simulator is not advertising {self._instance}/override. A world serves it "
                "by declaring the plugin this names, under exactly this label; without that "
                "this waits until the scenario's own timeout"
                + _also_carries(self._offered and self._offered())
            )
        return None


class _RosSpawn(SpawnCall):
    """One presence round-trip, spawn or delete. Sent on the first poll, mirroring ``_RosTeleport``."""

    def __init__(self, client, request, entity: str, result_ok: int, service: str, offered=None):
        self._client = client
        self._request = request
        self._entity = entity
        self._result_ok = result_ok
        self._service = service
        self._offered = offered
        self._future = None

    def pending_reason(self) -> str | None:
        if self._future is None and not self._client.service_is_ready():
            return (
                f"the simulator is not advertising {self._service!r}. A world serves it by "
                "declaring the `sim_interfaces` plugin; without that this waits until the "
                "scenario's own timeout" + _also_carries(self._offered and self._offered())
            )
        return None

    def poll(self) -> SpawnOutcome | None:
        if self._future is None:
            if not self._client.service_is_ready():
                return None
            self._future = self._client.call_async(self._request)
            return None
        if not self._future.done():
            return None
        resp = self._future.result()
        if resp is None:  # pragma: no cover
            raise AccessError(f"the call to {self._service} for {self._entity!r} was dropped")
        ok = int(resp.result.result) == int(self._result_ok)
        detail = (
            f"{self._service} {self._entity!r}"
            if ok
            else f"{self._service} for {self._entity!r} failed: "
            f"{resp.result.error_message or resp.result.result}"
        )
        return SpawnOutcome(ok=ok, detail=detail)


class _RosTeleport(TeleportCall):
    """One ``set_entity_state`` round-trip. Sent on the first poll, mirroring ``_RosCall``."""

    def __init__(self, client, request, entity: str, result_ok: int, offered=None):
        self._client = client
        self._request = request
        self._entity = entity
        self._result_ok = result_ok
        self._future = None
        self._offered = offered

    def pending_reason(self) -> str | None:
        if self._future is None and not self._client.service_is_ready():
            return (
                "the simulator is not advertising 'set_entity_state'. A world serves it by "
                "declaring the `sim_interfaces` plugin; without that this waits until the "
                "scenario's own timeout" + _also_carries(self._offered and self._offered())
            )
        return None

    def poll(self) -> TeleportOutcome | None:
        if self._future is None:
            if not self._client.service_is_ready():
                # No simulator yet, or no `sim_interfaces` plugin in its world. Waiting beats
                # failing: in a ROS run the stack and the simulator come up concurrently -- but the
                # caller is told WHICH service is missing, so a world that serves none is a
                # diagnosis rather than a trial that ran out of time.
                return None
            self._future = self._client.call_async(self._request)
            return None
        if not self._future.done():
            return None
        resp = self._future.result()
        if resp is None:  # pragma: no cover
            raise AccessError(f"the call to set_entity_state for {self._entity!r} was dropped")
        ok = int(resp.result.result) == int(self._result_ok)
        detail = (
            f"placed {self._entity!r}"
            if ok
            else f"set_entity_state for {self._entity!r} failed: "
            f"{resp.result.error_message or resp.result.result}"
        )
        return TeleportOutcome(ok=ok, detail=detail)
