# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_in_region()``: succeed once an entity's reference point is inside (or outside) a region.

The region is an area of the floor plan (:class:`~scenario_execution_roqsim.geometry.Region`): two
points are an axis-aligned box's opposite corners, three or more a polygon's vertices in order; z is
ignored. The reference point is the origin of the entity's body, as the core's
``sim/entities/<name>/pose`` reports it, so an entity counts as inside once that point is, however
much of its geometry is not. A point on the boundary is inside.

"Must never enter" is this action in a branch that fails the trial (``lib_osc/roqsim.osc``)::

    serial:
        entity_in_region(entity: 'robot', region: [...])
        emit fail
"""

from __future__ import annotations

from scenario_execution.actions.base_action import ActionError

from ..access import Pose
from ..geometry import Region, RegionError, point_of
from .entity_where import WhereCondition


class EntityInRegion(WhereCondition):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._region: Region | None = None
        self._outside = False

    def execute(self, entity: str, region, outside: bool):
        self._entity = str(entity or "")
        if not self._entity:
            raise ActionError(
                "entity_in_region: `entity` is empty. Name the entity to watch -- the world's "
                "`name:` for it.",
                action=self,
            )
        if not isinstance(region, (list, tuple)):
            raise ActionError(
                "entity_in_region: `region` must be a list of position_3d: a box's two opposite "
                f"corners, or a polygon's vertices in order. Got {region!r}.",
                action=self,
            )
        try:
            self._region = Region.from_points([point_of(p) for p in region])
        except RegionError as err:
            raise ActionError(f"entity_in_region: {err}", action=self) from None
        self._outside = bool(outside)

    def names(self) -> list[str]:
        return [self._entity]

    def judge(self, poses: dict[str, Pose]) -> tuple[bool, str]:
        at = poses[self._entity].pos
        inside = self._region.contains(at)
        holds = inside != self._outside
        summary = (
            f"{self._entity!r} at ({at[0]:.2f}, {at[1]:.2f}) is "
            f"{'inside' if inside else 'outside'} the {self._region.describe()}"
        )
        if not holds:
            summary += f", waiting for {'outside' if self._outside else 'inside'}"
        return holds, summary
