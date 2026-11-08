"""What the GripperCommand handler decides about a goal, without ROS.

The bridge's ``gripper_command`` handler (:mod:`roqsim_ros_bridge.actions`) drives any producer with
one commandable position and a ``() -> (position, velocity)`` reader: an arm's gripper, an automatic
door. Two of its decisions belong to the producer, which states them in its endpoint's ``ros2``
hints, and the handler asks rather than guessing from a name or a kind:

- ``reach_tolerance`` -- how near the commanded position the measured one must be for
  ``reached_goal``, in the producer's unit. A gripper keeps ros2_control's tight
  :data:`DEFAULT_REACH_TOLERANCE`; a door states the band inside which it counts itself as arrived,
  so the result agrees with the door's own status.
- ``on_timeout`` -- what a goal still moving when its ``timeout`` runs out ends as. ``succeed``
  (the default) returns it succeeded with neither ``reached_goal`` nor ``stalled`` set, as MoveIt
  expects of a slow hand; ``abort`` aborts it, for a producer that computes
  its own bound from its travel, so a goal past it has gone wrong.

A hint of the wrong kind is refused (:class:`ValueError`) rather than read as its default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

#: Sim seconds a goal is given to reach its position or stall, unless the ``timeout`` hint says
#: otherwise; 0 waits however long it takes.
DEFAULT_GRIPPER_TIMEOUT = 5.0
#: How near the commanded position counts as reached, unless the ``reach_tolerance`` hint says
#: otherwise (rad for a gripper's joint).
DEFAULT_REACH_TOLERANCE = 0.005
#: What a goal still moving at its timeout may end as.
ON_TIMEOUT = ("succeed", "abort")
#: The handler's verdicts.
SUCCEED = "succeed"
ABORT = "abort"


@dataclass(frozen=True)
class GripperPolicy:
    timeout: float = DEFAULT_GRIPPER_TIMEOUT
    reach_tolerance: float = DEFAULT_REACH_TOLERANCE
    on_timeout: str = "succeed"

    @classmethod
    def from_hints(cls, hints: Mapping) -> GripperPolicy:
        """The policy a producer's ``ros2`` hints state; a hint that is not usable raises."""
        timeout = _number(hints, "timeout", DEFAULT_GRIPPER_TIMEOUT)
        if timeout < 0:
            raise ValueError(f"the `timeout` hint must be >= 0 (0 waits for ever), not {timeout!r}")
        tol = _number(hints, "reach_tolerance", DEFAULT_REACH_TOLERANCE)
        if not (tol > 0 and math.isfinite(tol)):
            raise ValueError(f"the `reach_tolerance` hint must be a finite number > 0, not {tol!r}")
        on_timeout = hints.get("on_timeout", "succeed")
        if on_timeout not in ON_TIMEOUT:
            raise ValueError(
                f"the `on_timeout` hint must be one of {', '.join(ON_TIMEOUT)}, not {on_timeout!r}"
            )
        return cls(timeout=timeout, reach_tolerance=tol, on_timeout=on_timeout)

    def reached(self, position: float, target: float) -> bool:
        """Whether the measured ``position`` counts as the commanded ``target``."""
        return abs(position - target) <= self.reach_tolerance

    def verdict(self, reached: bool, stalled: bool) -> str:
        """How a goal that was not cancelled ends: :data:`SUCCEED` or :data:`ABORT`.

        A goal that reached or stalled succeeds. One that did neither ran out of time while still
        moving, and ends as the producer's ``on_timeout`` says.
        """
        if reached or stalled:
            return SUCCEED
        return ABORT if self.on_timeout == "abort" else SUCCEED


def _number(hints: Mapping, key: str, default: float) -> float:
    value = hints.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise ValueError(f"the `{key}` hint must be a number, not {value!r}")
    return float(value)
