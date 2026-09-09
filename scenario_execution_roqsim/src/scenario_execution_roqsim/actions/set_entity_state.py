# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``set_entity_state()``: put a free-jointed entity in a state, and say whether it landed.

A **full** pose, and not this action's own reading of one: ``pose`` is parsed by
:func:`roqsim.pose.parse_pose`, the same function a world document and
``simulation_interfaces/SetEntityState`` go through, so ``orientation`` accepts roll/pitch/yaw or a
quaternion and means the same thing here as everywhere else. What a body cannot do -- a weld with no
free joint -- is refused by the simulator, which knows; a floor is not this action's to assume.

The counterpart of ``spawn_robot.pos``/``.yaw`` for a factor a MuJoCo compile cannot express: a
per-RUN pose (a campaign's randomly generated start pose) rather than a per-CONFIGURATION one. A
world's own YAML is resolved once per configuration, before any run of it starts, so it can carry
`sim: plugins.spawn_robot.pos` overrides that vary by configuration -- but nothing before compile
knows a value that is drawn once per repetition, and the substrate does not recompile mid-run
(architecture.rst). This action is the other half: it moves an already-spawned entity, over either
transport, exactly like ``simulation_interfaces/SetEntityState`` does for the ROS path -- whose name and
shape this takes, because it is the same operation and there is no reason for two vocabularies.

Not a condition (contrast ``entity_moved``/``entity_rotated``): it is a WRITE with a verdict, same
shape as ``set_model_override``. **A write that did not land fails the trial**, because the
alternative is a row that claims the robot started somewhere the physics never put it -- exactly the
localisation-vs-physical-pose mismatch this action exists to prevent (a scenario that seeds nav2's
initial pose estimate at a point the robot's body never reached spends the whole trial recovering
from a belief error the campaign designer didn't intend to inject).
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from roqsim.pose import PoseError, parse_pose

from ..access import AccessError
from ..base import SimAction


def _twist_of(twist) -> tuple:
    """``((vx, vy, vz), (wx, wy, wz))`` from a ``velocity_6d``, or zeros.

    ``velocity_6d`` spells its halves ``translational``/``angular`` (see ``osc.types``), and the
    message spells them ``linear``/``angular`` -- both are accepted so a pose and a twist can be
    pasted from either the scenario's vocabulary or the service's.
    """
    twist = twist or {}
    linear = twist.get("translational") or twist.get("linear") or {}
    angular = twist.get("angular") or {}
    return (
        (float(linear.get("x", 0.0)), float(linear.get("y", 0.0)), float(linear.get("z", 0.0))),
        # roll/pitch/yaw is how `orientation_rate_3d` names its components; x/y/z is how the
        # message does. Same three numbers, and refusing one spelling would be arbitrary.
        (
            float(angular.get("roll", angular.get("x", 0.0))),
            float(angular.get("pitch", angular.get("y", 0.0))),
            float(angular.get("yaw", angular.get("z", 0.0))),
        ),
    )


class SetEntityState(SimAction):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._pos = (0.0, 0.0, 0.0)
        self._quat = (1.0, 0.0, 0.0, 0.0)
        self._lin = (0.0, 0.0, 0.0)
        self._ang = (0.0, 0.0, 0.0)
        self._call = None

    def execute(self, entity: str, pose: dict, twist: dict = None):
        self._entity = str(entity)
        if not self._entity:
            raise ActionError(
                "set_entity_state: `entity` is empty. Name the world's `name:` for that spawn -- the "
                "same string simulation_interfaces resolves, not a body name and not a TF frame.",
                action=self,
            )
        # Parsed by `roqsim.pose`, which is what a world document and the SetEntityState
        # service both go through. A full pose, therefore, and one convention: this action used
        # to convert yaw itself and refuse roll or pitch, which made the OSC verb the only place
        # in the substrate where an orientation meant something narrower than everywhere else --
        # and `rpy_to_quat`'s own docstring says why that is a bug waiting to happen. A pose a
        # body cannot take is refused by the simulator, naming the weld; a floor is not this
        # action's to assume.
        try:
            position, quat = parse_pose(pose)
        except PoseError as err:
            raise ActionError(f"set_entity_state: {err}", action=self) from None
        self._pos = (
            float(position[0]),
            float(position[1]),
            # `parse_pose` leaves z None when the document did not state one -- there it means
            # "the height the model rests at", which only a spawn can resolve. A teleport has no
            # model to ask, so an unstated z stays 0.0, exactly as this action has always read it.
            0.0 if position[2] is None else float(position[2]),
        )
        self._quat = tuple(float(v) for v in quat)
        # A twist is part of the state, and it defaults to zero: a body PUT somewhere is not still
        # carrying the velocity it had, which is what this has always done. What changes is that a
        # stated velocity now arrives instead of being dropped -- `SetEntityState` carries one, and
        # both this action and the bridge behind it used to ignore it while replying OK.
        self._lin, self._ang = _twist_of(twist)
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active, so a
        #: write reached twice in one run fires twice rather than replaying the first outcome.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.set_entity_state(
                    self._entity, self._pos, self._quat, self._lin, self._ang
                )
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            return self.waiting(f"setting {self._entity!r}'s state ({self.transport})", self._call)

        if not outcome.ok:
            return self.failed(
                f"could not place {self._entity!r} at {self._pos}: {outcome.detail}. The robot's "
                "physical pose and whatever nav2 believes about it are now inconsistent."
            )
        return self.satisfied(f"{self._entity!r} {outcome.detail}")
