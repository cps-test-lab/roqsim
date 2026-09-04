"""nav2's two navigation actions, served by roqsim's own navigator.

This module is the whole ROS surface of ``roqsim_nav``. It registers two handlers into the bridge's
shared registry and is imported at bridge start-up through the ``roqsim_ros_bridge.extensions``
entry point -- so the core bridge never depends on ``nav2_msgs``, and ``roqsim_nav`` never imports
ROS. The navigator declares its goal endpoints with the action type named as a *string*; the bridge
resolves the string and finds what is registered here.

**One package serves every mover, and that is a correctness requirement rather than tidiness.**
``ACTION_HANDLERS[type] = fn`` overwrites silently and extension modules are imported in unspecified
order, so two packages registering ``NavigateThroughPoses`` would make which handler serves a goal
depend on install order, with nothing in the log. A pedestrian, a second robot and a navigating prop
are all driven through one ``NavHandle``, so one handler is all there is to register.

Goal execution is the same for both types: hand the route to the navigator, poll its progress in
*sim* time, publish nav2's feedback, and succeed when it reports arrival **under the sequence number
this goal was given**. That last part is what distinguishes our arrival from a stale one -- a
navigator that finished whatever it was doing before is already "finished" when a new goal is queued.
"""

from __future__ import annotations

import math
import time

from nav2_msgs.action import NavigateThroughPoses, NavigateToPose

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


def _drive(goal_handle, ctx, endpoint, poses, result, feedback, fill):
    """Send ``poses`` and block this handler's thread until the route resolves.

    Shared by both action types, which differ only in their message shapes: the goal is the same
    route, and so are cancellation, preemption and completion.
    """
    handle = _handle_for(ctx, endpoint)
    if not poses:
        goal_handle.abort()
        return result

    # Returns the sequence synchronously, before the route has been applied -- the change is
    # marshalled onto the physics thread. Holding the number is what lets us wait for *our* arrival.
    seq = handle.send_goals(poses)
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
        poses,
        NavigateToPose.Result(),
        NavigateToPose.Feedback(),
        fill,
    )


@action_handler("nav2_msgs.action.NavigateThroughPoses")
def navigate_through_poses(goal_handle, ctx, on_payload, endpoint=None):
    """Drive the mover through a list of poses."""

    def fill(feedback, goals_left, dist_left, elapsed):
        feedback.number_of_poses_remaining = int(goals_left)
        feedback.distance_remaining = float(dist_left)
        feedback.navigation_time = _duration(elapsed)

    poses = [
        (p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation))
        for p in goal_handle.request.poses
    ]
    return _drive(
        goal_handle,
        ctx,
        endpoint,
        poses,
        NavigateThroughPoses.Result(),
        NavigateThroughPoses.Feedback(),
        fill,
    )
