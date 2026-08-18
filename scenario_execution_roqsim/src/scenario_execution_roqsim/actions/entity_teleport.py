# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_teleport()``: place a free-jointed entity at a pose, and say whether it landed.

The counterpart of ``spawn_robot.pos``/``.yaw`` for a factor a MuJoCo compile cannot express: a
per-RUN pose (a campaign's randomly generated start pose) rather than a per-CONFIGURATION one. A
world's own YAML is resolved once per configuration, before any run of it starts, so it can carry
`sim: plugins.spawn_robot.pos` overrides that vary by configuration -- but nothing before compile
knows a value that is drawn once per repetition, and the substrate does not recompile mid-run
(architecture.rst). This action is the other half: it moves an already-spawned entity, over either
transport, exactly like ``simulation_interfaces/SetEntityState`` does for the ROS path.

Not a condition (contrast ``entity_moved``/``entity_rotated``): it is a WRITE with a verdict, same
shape as ``set_model_override``. **A teleport that did not land fails the trial**, because the
alternative is a row that claims the robot started somewhere the physics never put it -- exactly the
localisation-vs-physical-pose mismatch this action exists to prevent (a scenario that seeds nav2's
initial pose estimate at a point the robot's body never reached spends the whole trial recovering
from a belief error the campaign designer didn't intend to inject).
"""

from __future__ import annotations

import math

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation about Z only -- roll/pitch are ignored, matching a wheeled base."""
    half = yaw / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


class EntityTeleport(SimAction):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._pos = (0.0, 0.0, 0.0)
        self._quat = (1.0, 0.0, 0.0, 0.0)
        self._call = None

    def execute(self, entity: str, pose: dict):
        self._entity = str(entity)
        if not self._entity:
            raise ActionError(
                "entity_teleport: `entity` is empty. Name the world's `name:` for that spawn -- the "
                "same string simulation_interfaces resolves, not a body name and not a TF frame.",
                action=self,
            )
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})
        self._pos = (
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        )
        if float(orientation.get("roll", 0.0)) or float(orientation.get("pitch", 0.0)):
            raise ActionError(
                "entity_teleport: `pose.orientation` has a nonzero roll or pitch. Only yaw is "
                "applied -- this action places a wheeled base on the floor it was spawned on, not a "
                "body tumbling in free space.",
                action=self,
            )
        self._quat = _yaw_to_quat(float(orientation.get("yaw", 0.0)))
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active, so a
        #: teleport reached twice in one run fires twice rather than replaying the first outcome.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.set_entity_pose(self._entity, self._pos, self._quat)
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            return self.waiting(f"teleporting {self._entity!r} ({self.transport})")

        if not outcome.ok:
            return self.failed(
                f"could not place {self._entity!r} at {self._pos}: {outcome.detail}. The robot's "
                "physical pose and whatever nav2 believes about it are now inconsistent."
            )
        return self.satisfied(f"{self._entity!r} {outcome.detail}")
