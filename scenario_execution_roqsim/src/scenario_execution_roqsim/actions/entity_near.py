# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_near()``: succeed once an entity is within a distance of another entity or a position.

The distance is between reference points -- the origins of the entities' bodies, as the core's
``sim/entities/<name>/pose`` reports them -- not between the nearest points of their geometry. It is
measured in the floor plane by default (``planar``), since "the robot reached the shelf" and "a
person came within a metre" are statements about the floor plan, and two entities of different
heights never share a z; ``spatial`` measures in 3D.

The target is another entity, named by ``target``, or a point, given as ``position`` when ``target``
is empty.
"""

from __future__ import annotations

from scenario_execution.actions.base_action import ActionError

from ..access import Pose
from ..geometry import DISTANCE_MODES, point_of, separation
from .entity_condition import enum_name
from .entity_where import WhereCondition


class EntityNear(WhereCondition):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._target = ""
        self._point = (0.0, 0.0, 0.0)
        self._distance = 0.0
        self._mode = "planar"

    def execute(self, entity: str, target: str, position, distance: float, mode):
        self._entity = str(entity or "")
        if not self._entity:
            raise ActionError(
                "entity_near: `entity` is empty. Name the entity to watch -- the world's `name:` "
                "for it.",
                action=self,
            )
        self._target = str(target or "")
        self._point = point_of(position)
        if self._target and any(self._point):
            raise ActionError(
                f"entity_near: both `target` ({self._target!r}) and `position` {self._point} are "
                "given. Name an entity in `target`, or leave it empty and give `position`.",
                action=self,
            )
        if self._target == self._entity:
            raise ActionError(
                f"entity_near: `target` is {self._entity!r}, the entity itself, which is always at "
                "distance 0 from itself.",
                action=self,
            )
        self._distance = float(distance)
        if self._distance <= 0.0:
            raise ActionError(
                f"entity_near: `distance` must be > 0 (metres), got {self._distance}.", action=self
            )
        self._mode = enum_name(mode)
        if self._mode not in DISTANCE_MODES:
            raise ActionError(
                f"entity_near: unknown mode {self._mode!r}; known: {', '.join(DISTANCE_MODES)}.",
                action=self,
            )

    def names(self) -> list[str]:
        return [self._entity, self._target] if self._target else [self._entity]

    def judge(self, poses: dict[str, Pose]) -> tuple[bool, str]:
        there = poses[self._target].pos if self._target else self._point
        gap = separation(self._mode, poses[self._entity].pos, there)
        what = repr(self._target) if self._target else f"({', '.join(f'{v:g}' for v in there)})"
        summary = (
            f"{self._entity!r} {gap:.2f} m {self._mode} from {what} (near: <= {self._distance:g} m)"
        )
        return gap <= self._distance, summary
