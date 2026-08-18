"""nav2 ``NavigateThroughPoses`` served by a roqsim walker.

This module is the whole ROS surface of the walker: it registers one action handler into the shared
bridge registry (:mod:`roqsim_ros_bridge.actions`). It is imported by the bridge at start-up via
the ``roqsim_ros_bridge.extensions`` entry point declared in this package's ``setup.py`` -- so the
core bridge never depends on ``nav2_msgs``, and ``roqsim_walker`` never imports ROS.

The walker plugin declares an ``in`` endpoint whose ros2 hint block names this action type. The
bridge resolves the handler below and serves an ``ActionServer`` at ``<namespace>/<action_name>``
(default ``navigate_through_poses``).

Goal execution: the pose list is handed to the walker's nav stack as its route, replacing whatever
patrol it was on. The handler then polls the walker's progress in *sim* time, publishing nav2's
feedback (``number_of_poses_remaining``, ``distance_remaining``), and succeeds when the walker
reports it arrived. Cancelling stops the walker where it stands. When the route completes the walker
resumes its configured patrol, if it had one.
"""

from __future__ import annotations

import math
import time

from nav2_msgs.action import NavigateThroughPoses

from roqsim_ros_bridge.actions import action_handler

#: How often (wall clock) the handler samples walker progress. The walker's own nav pipeline runs at
#: 60 Hz in sim time; polling faster only burns CPU.
_POLL_PERIOD = 0.02


def _yaw(q) -> float:
    """Yaw (rad) from a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _handle_for(ctx, endpoint):
    """The producing walker's :class:`~roqsim_walker.plugins.walker.WalkerHandle`.

    Resolved from the endpoint's ``owner`` (the walker's entity name), so this handler works for any
    number of walkers under one bridge without configuration.
    """
    handle = ctx.blackboard.get(f"walker:{endpoint.owner}")
    if handle is None:
        raise KeyError(
            f"no walker handle on the blackboard for endpoint owner {endpoint.owner!r}; "
            "is the 'walker' plugin loaded for it?"
        )
    return handle


@action_handler("nav2_msgs.action.NavigateThroughPoses")
def navigate_through_poses(goal_handle, ctx, on_payload, endpoint=None):
    """Drive the walker through ``goal.poses``, reporting nav2-shaped feedback until it arrives."""
    handle = _handle_for(ctx, endpoint)
    result = NavigateThroughPoses.Result()

    poses = [
        (p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation))
        for p in goal_handle.request.poses
    ]
    if not poses:
        goal_handle.abort()
        return result

    # Stamp + apply the route. `send_route` marshals onto the physics thread and returns the route's
    # sequence number synchronously, so we can tell *our* completion from a stale or preempted one.
    seq = handle.send_route(poses)

    feedback = NavigateThroughPoses.Feedback()
    start = ctx.sim_time
    while True:
        if goal_handle.is_cancel_requested:
            handle.cancel_route()
            goal_handle.canceled()
            return result

        cur_seq, finished, goals_left, dist_left = handle.status()

        if cur_seq > seq:  # a newer goal replaced ours before we finished
            goal_handle.abort()
            return result
        if cur_seq == seq and finished:
            goal_handle.succeed()
            return result

        if cur_seq == seq:  # our route is live -- report progress
            feedback.number_of_poses_remaining = int(goals_left)
            feedback.distance_remaining = float(dist_left)
            feedback.navigation_time = _duration(ctx.sim_time - start)
            goal_handle.publish_feedback(feedback)
        time.sleep(_POLL_PERIOD)


def _duration(seconds: float):
    from builtin_interfaces.msg import Duration

    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))
