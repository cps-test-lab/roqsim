"""ROS 2 action and service handlers for the bridge, keyed by type string.

The pub/sub side of the bridge is fully generic (type strings + converters in
:mod:`roqsim_ros_bridge.registry`); actions and services need a *policy* on top -- how a goal is
executed, what a reply means -- so each supported type registers a handler here. A producer declares
an ``in`` endpoint whose ros2 hint block carries ``action`` or ``service`` (the type string) and
``name`` (the relative action/service name); the bridge looks up the handler and serves an
``ActionServer`` or a service -- so a robot package never imports ROS and never depends on a specific
bridge. The *service* registry lives in :mod:`roqsim_ros_bridge.services`, which stays free of ROS
imports so its reply policy is testable without one; this module holds the actions.

Which of the three inbound kinds a producer wants is a question about the *interaction*, not about
taste: a **topic** is a stream with no answer, a **service** is a command whose outcome the caller
needs (and can therefore fail on), an **action** is a goal that takes time, reports feedback and can
be cancelled.

Handler contract (the service one is in :mod:`roqsim_ros_bridge.services`)::

    handler(goal_handle, ctx, on_payload, endpoint) -> result message

``goal_handle`` is rclpy's ServerGoalHandle (runs on the executor thread, may block for the goal's
duration); ``ctx`` is the :class:`~roqsim.context.SimContext` (use ``ctx.sim_time`` for pacing);
``on_payload`` is the bridge's inbound callback for the endpoint -- it marshals a neutral payload
onto the physics thread via ``ctx.post``, so handlers need no threading code; ``endpoint`` is the
:class:`~roqsim.context.Endpoint` being served, so a handler can reach its *producer's* state
generically (``ctx.blackboard.get(f"arm:{endpoint.owner}")``) instead of hardcoding a key.

A handler in another package reaches these registries through the
``roqsim_ros_bridge.extensions`` entry-point group -- see :mod:`roqsim_ros_bridge.extensions`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory, GripperCommand

from .extensions import EXTENSION_GROUP, load_extensions

logger = logging.getLogger(__name__)

# action-type-string -> handler(goal_handle, ctx, on_payload, endpoint) -> result message
ACTION_HANDLERS: dict[str, Callable[[Any, Any, Callable[[Any], None], Any], Any]] = {}


def action_handler(type_path: str):
    def register(fn):
        ACTION_HANDLERS[type_path] = fn
        return fn

    return register


def get_action_handler(type_path: str) -> Callable[[Any, Any, Callable[[Any], None], Any], Any]:
    load_extensions()  # a handler may live in another package (see the module docstring)
    fn = ACTION_HANDLERS.get(type_path)
    if fn is None:
        raise KeyError(
            f"no action handler registered for {type_path!r}; "
            f"known: {sorted(ACTION_HANDLERS)} (see roqsim_ros_bridge.actions; a handler in "
            f"another package must be advertised in the {EXTENSION_GROUP!r} entry-point group)"
        )
    return fn


def _sample(p0, v0, p1, v1, alpha: float, span: float) -> list[float]:
    """The trajectory between two waypoints at *alpha* in [0, 1].

    Cubic Hermite when the waypoints carry velocities, linear when they do not -- the same choice
    ros2_control's JointTrajectoryController makes. MoveIt's time parameterization does emit
    velocities, so the executed path matches the planned one in shape and not just at its knots;
    without them a straight line between waypoints is still far better than holding one until the
    next falls due.
    """
    a = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    if not v0 or not v1:
        return [q0 + (q1 - q0) * a for q0, q1 in zip(p0, p1, strict=False)]
    # Hermite basis. The velocity terms are scaled by the span because alpha is normalized time.
    h00 = 2 * a**3 - 3 * a**2 + 1
    h10 = a**3 - 2 * a**2 + a
    h01 = -2 * a**3 + 3 * a**2
    h11 = a**3 - a**2
    return [
        h00 * q0 + h10 * span * d0 + h01 * q1 + h11 * span * d1
        for q0, d0, q1, d1 in zip(p0, v0, p1, v1, strict=False)
    ]


def _dur_to_sec(d: DurationMsg) -> float:
    return float(d.sec) + float(d.nanosec) * 1e-9


@action_handler("control_msgs.action.FollowJointTrajectory")
def follow_joint_trajectory(goal_handle, ctx, on_payload, endpoint=None):
    """Follow a JointTrajectory by feeding each waypoint at its scheduled sim time.

    This is the interaction MoveIt2's ``moveit_simple_controller_manager`` drives to execute
    plans. Each due waypoint goes through ``on_payload`` as the neutral ``(names, positions)``
    payload the producer's ``write`` expects (e.g. ``ArmControllerPlugin.set_targets``).

    Feedback reports ``desired`` (the commanded waypoint) against ``actual`` (what the joints are
    really at, read back through the producer's state reader) and their difference as ``error``, the
    way a ros2_control JointTrajectoryController does. Reporting the command as both -- which this did
    -- makes the tracking error identically zero, hiding exactly the saturation or gain problem the
    feedback exists to expose.
    """
    traj = goal_handle.request.trajectory
    names = list(traj.joint_names)
    result = FollowJointTrajectory.Result()

    # The producer's joint-state reader, so `actual` is measured rather than assumed. Same key
    # convention as the gripper handler; absent, feedback falls back to reporting the command.
    reader = None
    if endpoint is not None:
        state_key = endpoint.backend.get("ros2", {}).get("arm_state_key", f"arm:{endpoint.owner}")
        handle = ctx.blackboard.get(state_key)
        reader = getattr(handle, "read_state", None)

    def measured(commanded: list[float]) -> list[float]:
        if reader is None:
            return commanded
        got_names, positions, *_ = reader()
        by_name = dict(zip(got_names, positions, strict=False))
        # A trajectory may name a subset, or joints this controller does not own.
        return [float(by_name.get(n, c)) for n, c in zip(names, commanded, strict=True)]

    start = ctx.sim_time
    feedback = FollowJointTrajectory.Feedback()
    feedback.joint_names = names
    # INTERPOLATE between waypoints, rather than holding each one until the next falls due.
    #
    # A zero-order hold makes the commanded position a staircase, and a stiff position servo chases
    # every step as a discontinuity: the arm jerks from waypoint to waypoint instead of moving, and
    # arrives at each one with an impulse. Watched in the viewer it is unmistakable, and it is not
    # cosmetic -- an impulsive approach knocks a small object over instead of closing on it, and the
    # effort a run reports is dominated by the steps rather than by the task.
    #
    # A real JointTrajectoryController does not do this: it evaluates the trajectory at its update
    # rate and commands a continuously varying setpoint. This does the same, which makes the executed
    # motion the trajectory MoveIt actually planned rather than a sampling of it.
    prev_t = start
    prev_pos = measured(list(traj.points[0].positions)) if traj.points else []
    prev_vel = [0.0] * len(names)
    for point in traj.points:
        target_t = start + _dur_to_sec(point.time_from_start)
        positions = list(point.positions)
        vels = list(point.velocities) if len(point.velocities) == len(names) else None
        span = target_t - prev_t
        last_fed = None
        # Feed the trajectory as it comes due, honouring cancellation.
        while ctx.sim_time < target_t:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return result
            now = ctx.sim_time
            # Once per physics step at most: the payload goes onto the physics thread's queue, and
            # posting faster than it steps only lengthens the queue.
            if span > 0.0 and now != last_fed:
                last_fed = now
                on_payload(
                    (
                        names,
                        _sample(prev_pos, prev_vel, positions, vels, (now - prev_t) / span, span),
                    )
                )
            time.sleep(0.002)
        on_payload((names, positions))
        prev_t, prev_pos = target_t, positions
        prev_vel = vels or [0.0] * len(names)
        actual = measured(positions)
        feedback.desired.positions = positions
        feedback.actual.positions = actual
        feedback.error.positions = [a - c for a, c in zip(actual, positions, strict=True)]
        goal_handle.publish_feedback(feedback)

    goal_handle.succeed()
    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
    return result


@action_handler("control_msgs.action.GripperCommand")
def gripper_command(goal_handle, ctx, on_payload, endpoint=None):
    """Drive a 1-DOF thing to a commanded position and report when it settles or stalls.

    This is the interaction MoveIt2's ``moveit_simple_controller_manager`` drives to open/close a
    gripper (controller ``type: GripperCommand``), but the policy is generic -- any producer with a
    single commandable position and a state reader reuses it (e.g. an automatic ``door``). The
    commanded ``position`` is sent to the producer (e.g. ``ArmControllerPlugin.set_gripper``,
    ``DoorPlugin.set_openness``) via ``on_payload`` as the neutral scalar payload its ``write``
    expects. We then watch the producer's state -- a ``() -> (position, velocity)`` reader on the
    blackboard -- and succeed once it reaches the target (``reached_goal``) or stops moving short of
    it (``stalled``, i.e. a gripper closed on an object / a door met an obstruction). The reader's
    blackboard key is the endpoint's ros2 ``state_key`` hint, defaulting to ``gripper:<owner>`` so
    existing arms are unchanged. Without a reader we wait a fixed settle time and report the command.
    """
    cmd = goal_handle.request.command
    target = float(cmd.position)
    on_payload(target)

    result = GripperCommand.Result()
    reader = None
    if endpoint is not None:
        state_key = endpoint.backend.get("ros2", {}).get("state_key", f"gripper:{endpoint.owner}")
        reader = ctx.blackboard.get(state_key)

    pos_tol = 0.005  # rad: close enough to call the goal reached
    vel_tol = 0.002  # rad/s: below this the fingers are considered stopped (stalled on an object)
    timeout = 5.0
    settle = 0.5  # min time before a low velocity counts as a stall (let motion start)

    start = ctx.sim_time
    position = target
    stalled = False
    reached = False
    feedback = GripperCommand.Feedback()
    while ctx.sim_time - start < timeout:
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.position = position
            return result
        if reader is not None:
            position, velocity = reader()
            reached = abs(position - target) <= pos_tol
            stalled = (ctx.sim_time - start) > settle and abs(velocity) <= vel_tol and not reached
            feedback.position = position
            feedback.stalled = stalled
            feedback.reached_goal = reached
            goal_handle.publish_feedback(feedback)
            if reached or stalled:
                break
        elif ctx.sim_time - start >= settle:
            reached = True
            break
        time.sleep(0.01)

    goal_handle.succeed()
    result.position = position
    result.stalled = stalled
    result.reached_goal = reached or stalled  # a stall on the object is a successful grasp
    return result
