# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_rotated()``: succeed once named entities have turned by an angle.

A separate action from ``entity_moved`` rather than a sixth mode of it, and the reason is the
parameter, not the code: one ``threshold`` that means metres for five modes and radians for a sixth
cannot be read at the call site without resolving a sibling argument. Here the parameter is called
``angle`` and its unit is never in question.

The measure is the GEODESIC angle -- the single rotation that takes the starting orientation to the
current one, whatever axis that is about. Not roll/pitch/yaw differences, which are three numbers that
do not compose and which gimbal-lock; not a quaternion component, which is not an angle.
"""

from __future__ import annotations

from scenario_execution.actions.base_action import ActionError

from ..access import Pose
from ..displacement import rotation_angle
from .entity_condition import EntityCondition


class EntityRotated(EntityCondition):
    def __init__(self):
        super().__init__()
        self._angle = 0.0

    def execute(self, entities: list, angle: float, dwell: float, require):
        self._angle = float(angle)
        if self._angle <= 0.0:
            raise ActionError(
                f"entity_rotated: `angle` must be > 0 rad, got {self._angle}. It is a magnitude -- the "
                "geodesic angle is direction-free, so there is no negative case to express; use "
                "entity_moved for a signed quantity.",
                action=self,
            )
        # pi is the largest angle the measure can return: past half a turn the shorter way round is
        # the answer, which is what makes q and -q the same orientation.
        if self._angle > 3.141592653589793:
            raise ActionError(
                f"entity_rotated: `angle` must be <= pi rad, got {self._angle}. The geodesic angle is "
                "the SHORTEST rotation between two orientations, so it never exceeds pi and a larger "
                "threshold could never be met. Count turns with a different observable.",
                action=self,
            )
        self.start(entities, dwell, require)

    def measure(self, start: Pose, now: Pose) -> float:
        return rotation_angle(start.quat, now.quat)

    def holds(self, measured: float) -> bool:
        return measured >= self._angle

    def render(self, measured: float) -> str:
        import math

        return f"{math.degrees(measured):.0f}/{math.degrees(self._angle):.0f}deg"
