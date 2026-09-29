# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_near()`` / ``entity_near_position()``: succeed once an entity is within a distance of
another entity, or of a point.

The distance is between reference points -- the origins of the entities' bodies, as the core's
``sim/entities/<name>/pose`` reports them -- not between the nearest points of their geometry. It is
measured in the floor plane by default (``planar``), since "the robot reached the shelf" and "a
person came within a metre" are statements about the floor plan, and two entities of different
heights never share a z; ``spatial`` measures in 3D.
"""

from __future__ import annotations

from scenario_execution.actions.base_action import ActionError

from ..access import Pose
from ..geometry import DISTANCE_MODES, point_of, separation
from .entity_condition import enum_name
from .entity_where import WhereCondition


class _Near(WhereCondition):
    """What both share: the entity, the distance and how it is measured."""

    verb = ""

    def __init__(self):
        super().__init__()
        self._entity = ""
        self._distance = 0.0
        self._mode = "planar"

    def _configure(self, entity, distance, mode) -> None:
        self._entity = str(entity or "")
        if not self._entity:
            raise ActionError(
                f"{self.verb}: `entity` is empty. Name the entity to watch -- the world's `name:` "
                "for it.",
                action=self,
            )
        self._distance = float(distance)
        if self._distance <= 0.0:
            raise ActionError(
                f"{self.verb}: `distance` must be > 0 (metres), got {self._distance}.", action=self
            )
        self._mode = enum_name(mode)
        if self._mode not in DISTANCE_MODES:
            raise ActionError(
                f"{self.verb}: unknown mode {self._mode!r}; known: {', '.join(DISTANCE_MODES)}.",
                action=self,
            )

    def _judge(self, here, there, what: str) -> tuple[bool, str]:
        gap = separation(self._mode, here, there)
        summary = (
            f"{self._entity!r} {gap:.2f} m {self._mode} from {what} (near: <= {self._distance:g} m)"
        )
        return gap <= self._distance, summary


class EntityNear(_Near):
    verb = "entity_near"

    def __init__(self):
        super().__init__()
        self._target = ""

    def execute(self, entity: str, target: str, distance: float, mode):
        self._configure(entity, distance, mode)
        self._target = str(target or "")
        if not self._target:
            raise ActionError(
                "entity_near: `target` is empty. Name the other entity -- the world's `name:` for "
                "it; for a point, use entity_near_position.",
                action=self,
            )
        if self._target == self._entity:
            raise ActionError(
                f"entity_near: `target` is {self._entity!r}, the entity itself, which is always at "
                "distance 0 from itself.",
                action=self,
            )

    def names(self) -> list[str]:
        return [self._entity, self._target]

    def judge(self, poses: dict[str, Pose]) -> tuple[bool, str]:
        return self._judge(poses[self._entity].pos, poses[self._target].pos, repr(self._target))


class EntityNearPosition(_Near):
    verb = "entity_near_position"

    def __init__(self):
        super().__init__()
        self._point = (0.0, 0.0, 0.0)

    def execute(self, entity: str, position, distance: float, mode):
        self._configure(entity, distance, mode)
        self._point = point_of(position)

    def names(self) -> list[str]:
        return [self._entity]

    def judge(self, poses: dict[str, Pose]) -> tuple[bool, str]:
        what = f"({', '.join(f'{v:g}' for v in self._point)})"
        return self._judge(poses[self._entity].pos, self._point, what)
