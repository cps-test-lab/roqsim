# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_moved()``: succeed once named entities have been displaced from where they started.

NET displacement, not path length. An entity that travels out and comes back has moved zero, which is
the opposite of ``osc.ros``' ``odometry_distance_traveled`` -- pick this one for "is it clear of the
table yet", that one for an odometry budget. Someone measuring a robot's progress on a curved path will
otherwise wonder why this number is smaller.

Works over both transports: in a stepped run the pose is read from ``data.xpos``, in a ROS run from
``simulation_interfaces/GetEntityState``, keyed by the same entity name either way. See
:mod:`scenario_execution_roqsim.access`.
"""

from __future__ import annotations

from scenario_execution.actions.base_action import ActionError

from ..access import Pose
from ..displacement import AXES, MAGNITUDE_MODES, MODES, displacement, satisfied
from .entity_condition import EntityCondition, enum_name


class EntityMoved(EntityCondition):
    def __init__(self):
        super().__init__()
        self._mode = "distance"
        self._threshold = 0.0

    def execute(self, entities: list, threshold: float, mode, dwell: float, require):
        self._mode = enum_name(mode)
        if self._mode not in MODES:
            raise ActionError(
                f"entity_moved: unknown mode {self._mode!r}; known: {', '.join(MODES)}.",
                action=self,
            )
        self._threshold = float(threshold)
        # Refused rather than allowed to mean "any movement at all": a zero threshold is satisfied by
        # solver noise on the first tick, and on an axis mode it does not even say which direction --
        # so it would fire on a parcel settling downward by a micron.
        if self._mode in MAGNITUDE_MODES and self._threshold <= 0.0:
            raise ActionError(
                f"entity_moved: `threshold` must be > 0 for mode {self._mode!r} (a magnitude), got "
                f"{self._threshold}.",
                action=self,
            )
        if self._mode in AXES and self._threshold == 0.0:
            raise ActionError(
                f"entity_moved: `threshold` must not be 0 for the axis mode {self._mode!r}. Its SIGN "
                "is what says which way: +0.05 means risen 5 cm, -0.05 means fallen 5 cm.",
                action=self,
            )
        self.start(entities, dwell, require)

    def measure(self, start: Pose, now: Pose) -> float:
        return displacement(self._mode, start.pos, now.pos)

    def holds(self, measured: float) -> bool:
        return satisfied(self._mode, self._threshold, measured)

    def render(self, measured: float) -> str:
        # Signed on an axis mode, because the sign is half the meaning there.
        fmt = "{:+.0f}" if self._mode in AXES else "{:.0f}"
        return f"{fmt.format(measured * 1000)}/{self._threshold * 1000:+.0f}mm {self._mode}"
