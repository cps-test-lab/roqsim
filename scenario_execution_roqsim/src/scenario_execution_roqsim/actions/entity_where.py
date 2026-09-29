# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""What ``entity_near`` and ``entity_in_region`` share: the entities' ground truth, every tick.

Each tick reads the core's ``sim/entities/<name>/pose`` of every entity the condition names -- on
both transports the same endpoint, so an entity welded to the world (a shelf) has a pose, and a name
the core does not serve is refused with the same text -- and succeeds on the first tick the
condition holds. Sampled like ``entity_moved``: in-process at every tick, over the control socket at
the tick period of the replies.

An absent entity (deleted at run time) is nowhere: the condition does not hold, and the action
waits, saying so, until it is spawned again. Bounding the wait is the scenario's (``timeout()``,
``until``).
"""

from __future__ import annotations

import py_trees

from ..access import AccessError, EntityAbsent, Pose
from ..base import SimAction


class WhereCondition(SimAction):
    """Succeeds on the first tick :meth:`judge` says the condition holds."""

    def names(self) -> list[str]:
        """The entities whose poses the condition reads."""
        raise NotImplementedError

    def judge(self, poses: dict[str, Pose]) -> tuple[bool, str]:
        """Whether the condition holds, and a message saying what was measured."""
        raise NotImplementedError

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            # As in entity_moved: the stepped runner builds the world on the first reset.
            return self.waiting("waiting for the simulation", self._access)
        poses = {}
        try:
            for name in self.names():
                poses[name] = self._access.ground_truth_pose(name)
        except EntityAbsent as err:
            return self.waiting(str(err))
        except AccessError as err:
            self.reraise(err)
        pending = [name for name, pose in poses.items() if pose is None]
        if pending:
            return self.waiting(f"waiting for {pending[0]!r}'s pose over {self.transport}")
        holds, summary = self.judge(poses)
        if holds:
            return self.satisfied(f"{summary} ({self.transport})")
        return self.waiting(summary)
