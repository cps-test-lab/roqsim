"""The navigation actions, served by roqsim's own navigator.

This module is the whole ROS surface of ``roqsim_nav``. It registers its handlers into the bridge's
shared registry and is imported at bridge start-up through the ``roqsim_ros_bridge.extensions``
entry point -- so the core bridge never depends on ``nav2_msgs``, and ``roqsim_nav`` never imports
ROS. The navigator declares its goal endpoints with the action type named as a *string*; the bridge
resolves the string and finds what is registered here.

Three types: nav2's ``NavigateToPose`` and ``NavigateThroughPoses``, which send a route, and
``roqsim_nav_interfaces/StartRoute``, which releases the route the world configured. The last is a
type of its own rather than an empty nav2 goal: a nav2 goal with no poses has nowhere to go, and
giving it a meaning here alone would make one message mean two things. It is refused instead.

**One package serves every mover, and that is a correctness requirement rather than tidiness.**
``ACTION_HANDLERS[type] = fn`` overwrites silently and extension modules are imported in unspecified
order, so two packages registering ``NavigateThroughPoses`` would make which handler serves a goal
depend on install order, with nothing in the log. A pedestrian, a second robot and a navigating prop
are all driven through one ``NavHandle``, so one handler is all there is to register.

Goal execution is the same for every type: hand the request to the navigator, poll its progress in
*sim* time, publish feedback, and succeed when it reports arrival **under the sequence number this
goal was given**. That last part is what distinguishes our arrival from a stale one -- a navigator
that finished whatever it was doing before is already "finished" when a new goal is queued.
"""

from __future__ import annotations

import math
import time

from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from roqsim_nav_interfaces.action import StartRoute

from roqsim_ros_bridge.actions import action_handler

#: How often (wall clock) the handlers sample progress. The navigator's own pipeline runs at 20-60 Hz
#: in sim time; polling faster only burns the bridge's CPU.
_POLL_PERIOD = 0.02


def _yaw(q) -> float:
    """Yaw (rad) from a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _handle_for(ctx, endpoint):
    """The producing navigator's ``NavHandle``, resolved from the endpoint's owner.

    Keyed on the entity, so one handler serves any number of movers under one bridge with no
    configuration -- and so the name a ROS client uses is the same name a scenario uses.
    """
    handle = ctx.blackboard.get(f"nav:{endpoint.owner}:handle")
    if handle is None:
        raise KeyError(
            f"no navigator handle on the blackboard for endpoint owner {endpoint.owner!r}; "
            "is a 'navigator' component nested under that entity?"
        )
    return handle


def _duration(seconds: float):
    from builtin_interfaces.msg import Duration

    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def _through_poses_feedback(feedback, goals_left, dist_left, elapsed):
    """``NavigateThroughPoses`` and ``StartRoute`` report progress in the same three fields."""
    feedback.number_of_poses_remaining = int(goals_left)
    feedback.distance_remaining = float(dist_left)
    feedback.navigation_time = _duration(elapsed)


def _drive(goal_handle, ctx, endpoint, send, result, feedback, fill):
    """Hand the request to the navigator with ``send`` and block this thread until it resolves.

    Shared by every action type, which differ only in their message shapes and in what they ask the
    handle for: cancellation, preemption and completion are the same route either way. ``send``
    raising ``ValueError`` is the navigator refusing the request, and aborts the goal.
    """
    handle = _handle_for(ctx, endpoint)
    # Returns the sequence synchronously, before the route has been applied -- the change is
    # marshalled onto the physics thread. Holding the number is what lets us wait for *our* arrival.
    try:
        seq = send(handle)
    except ValueError:
        goal_handle.abort()
        return result
    start = ctx.sim_time
    while True:
        if goal_handle.is_cancel_requested:
            handle.cancel()
            goal_handle.canceled()
            return result

        applied, finished, goals_left, dist_left = handle.status()
        if applied > seq:  # a newer goal replaced ours before we finished
            goal_handle.abort()
            return result
        if applied == seq and finished:
            goal_handle.succeed()
            return result

        if applied == seq:  # our route is live -- report progress
            fill(feedback, goals_left, dist_left, ctx.sim_time - start)
            goal_handle.publish_feedback(feedback)
        time.sleep(_POLL_PERIOD)


@action_handler("nav2_msgs.action.NavigateToPose")
def navigate_to_pose(goal_handle, ctx, on_payload, endpoint=None):
    """Drive the mover to one pose."""

    def fill(feedback, _goals_left, dist_left, elapsed):
        feedback.distance_remaining = float(dist_left)
        feedback.navigation_time = _duration(elapsed)

    pose = goal_handle.request.pose
    poses = [(pose.pose.position.x, pose.pose.position.y, _yaw(pose.pose.orientation))]
    return _drive(
        goal_handle,
        ctx,
        endpoint,
        lambda handle: handle.send_goals(poses),
        NavigateToPose.Result(),
        NavigateToPose.Feedback(),
        fill,
    )


@action_handler("nav2_msgs.action.NavigateThroughPoses")
def navigate_through_poses(goal_handle, ctx, on_payload, endpoint=None):
    """Drive the mover through a list of poses. An empty list is refused by ``send_goals``."""
    poses = [
        (p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation))
        for p in goal_handle.request.poses
    ]
    return _drive(
        goal_handle,
        ctx,
        endpoint,
        lambda handle: handle.send_goals(poses),
        NavigateThroughPoses.Result(),
        NavigateThroughPoses.Feedback(),
        _through_poses_feedback,
    )


@action_handler("roqsim_nav_interfaces.action.StartRoute")
def start_route(goal_handle, ctx, on_payload, endpoint=None):
    """Run the route the mover was configured with, and finish when it does."""
    return _drive(
        goal_handle,
        ctx,
        endpoint,
        lambda handle: handle.start(),
        StartRoute.Result(),
        StartRoute.Feedback(),
        _through_poses_feedback,
    )
