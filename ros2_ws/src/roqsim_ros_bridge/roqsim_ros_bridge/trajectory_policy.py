"""What a ros2_control JointTrajectoryController decides about a goal, without ROS.

The bridge's ``follow_joint_trajectory`` handler (:mod:`roqsim_ros_bridge.actions`) feeds a goal to an
arm; what it may *refuse*, what it *grades*, and which goal is the *current* one are decisions
ros2_controllers' JointTrajectoryController makes, and they live here, free of ROS imports, so they
can be tested without one.

- :func:`rejection` -- whether a goal is rejected outright, as the controller's goal callback
  rejects it: a goal naming fewer joints than the controller owns when partial goals are not
  allowed (and then a joint it does not own), a last point with a velocity when
  ``allow_nonzero_velocity_at_trajectory_end`` is false.
- :func:`path_tolerances` -- the per-joint tolerance between where the trajectory says a joint is
  and where it is, while the trajectory runs (``constraints.<joint>.trajectory``). A goal's own
  ``path_tolerance`` entries win, as they do upstream.
- :class:`CurrentGoal` -- one goal executes per controller: a new one replaces the running one, which
  ends with ``INVALID_GOAL`` and :data:`PREEMPTED`, as upstream's ``preempt_active_goal``.

The strings are upstream's (``joint_trajectory_controller.cpp``), since a stack may match on them.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Sequence

#: The result string of a goal replaced by a newer one.
PREEMPTED = "Current goal cancelled due to new incoming action."
#: The result string of a goal that reached its tolerance.
SUCCEEDED = "Goal successfully reached!"
#: The result string of a goal that left its path tolerance.
PATH_VIOLATED = "Aborted due to path tolerance violation"
#: What upstream counts as a zero final velocity: ``std::numeric_limits<float>::epsilon()``. A
#: planner's last point carries velocities of order 1e-17 rather than exact zeros.
ZERO_VELOCITY = 1.1920928955078125e-07


def rejection(names: Sequence[str], last_velocities: Sequence[float], hints: dict) -> str | None:
    """Why the controller would reject a goal naming *names* and ending at *last_velocities*,
    or ``None`` when it takes it. *hints* are the endpoint's (``joint_names``,
    ``allow_partial_joints_goal``, ``allow_nonzero_velocity_at_trajectory_end``); a hint left out
    keeps upstream's permissive default.
    """
    owned = hints.get("joint_names")
    # Checked for a controller that states its joints strictly (partial goals refused, as a
    # robot's driver configures it); one that allows partial goals keeps ignoring names it does
    # not own, as roqsim's arms always have.
    if owned and not hints.get("allow_partial_joints_goal", True):
        foreign = [n for n in names if n not in owned]
        if foreign:
            return f"Incoming joint {foreign[0]} doesn't match the controller's joints."
        if set(names) != set(owned):
            return (
                f"Joints on incoming trajectory don't match the controller joints: "
                f"{len(names)} given, {len(owned)} expected"
            )
    if not hints.get("allow_nonzero_velocity_at_trajectory_end", True):
        for name, v in zip(names, last_velocities, strict=False):
            if abs(v) > ZERO_VELOCITY:
                return f"Velocity of last trajectory point of joint {name} is not zero: {v:.15f}"
    return None


def path_tolerances(request_path_tolerance, hints: dict, names: Sequence[str]) -> dict[str, float]:
    """Per-joint path tolerance in force: the goal's own entries over the controller's
    ``path_tolerance`` hint. ``0`` disables a joint's check, as the action's convention has it."""
    configured = hints.get("path_tolerance", 0.0)
    if isinstance(configured, dict):
        tol = {n: float(configured.get(n, 0.0)) for n in names}
    else:
        tol = dict.fromkeys(names, float(configured or 0.0))
    for entry in request_path_tolerance or ():
        if entry.name in tol:
            tol[entry.name] = float(entry.position)
    return {n: v for n, v in tol.items() if v > 0.0}


def worst_violation(names, desired, actual, tol: dict[str, float]):
    """``(joint, error, allowed)`` for the joint furthest outside its tolerance, or ``None``."""
    worst = None
    for name, want, got in zip(names, desired, actual, strict=False):
        allowed = tol.get(name)
        if allowed is None:
            continue
        err = abs(got - want)
        if err > allowed and (worst is None or err - allowed > worst[1] - worst[2]):
            worst = (name, err, allowed)
    return worst


class CurrentGoal:
    """Which goal a controller is executing: the most recent one to start.

    A goal calls :meth:`start` and keeps the token; it asks :meth:`is_current` as it runs and ends
    itself once it is not. Thread-safe: goals execute on the executor's threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens = itertools.count(1)
        self._current: dict[object, int] = {}

    def start(self, key: object) -> int:
        with self._lock:
            token = next(self._tokens)
            self._current[key] = token
            return token

    def is_current(self, key: object, token: int) -> bool:
        with self._lock:
            return self._current.get(key) == token

    def finish(self, key: object, token: int) -> None:
        with self._lock:
            if self._current.get(key) == token:
                del self._current[key]


#: One per bridge process, keyed by the endpoint a goal is served for.
CURRENT_GOALS = CurrentGoal()
