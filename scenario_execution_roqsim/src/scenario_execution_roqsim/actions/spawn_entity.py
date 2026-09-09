# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``spawn_entity()``: make a declared entity perceivable, at the pose it appears at.

**Activation, not creation.** roqsim never recompiles the model at runtime, so a world declares
everything a trial may bring in and this selects one of them; a name the world does not carry is
refused rather than approximated. There is no ``uri`` here for that reason -- offering one would
invite a caller to send geometry nothing can load.

This is what a per-RUN start pose should go through. The pose is applied in the SAME transaction as
the presence flip, so the entity is never perceivable at a pose nobody asked for -- where setting a state
can only place it where the world compiled it and then move it, which is visible for a step and
leaves a free body accelerating under gravity in between.

Like every action here it runs over either transport (``roqsim.access``): in a stepped run the flip
is a posted callback on the physics thread, and in a ROS run it is ``simulation_interfaces``'
``SpawnEntity``. The scenario is written once and does not learn which shape it is in.

``scenario_execution_sim`` declares a ``spawn_entity`` too -- its own, ROS-only, ``uri``-shaped one.
That is not a clash to rename around: an action name is unique within a LIBRARY, and an invocation
binds to the implementation shipped by the package whose library declared it. A scenario importing
both libraries is the one ambiguous case, and scenario-execution reports it by name.

A spawn that did not land **fails the trial**, on the same grounds as ``set_entity_state``: the
alternative is a row that claims a robot started somewhere the physics never put it.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from roqsim.pose import PoseError, parse_pose

from ..access import AccessError
from ..base import SimAction


class SpawnEntity(SimAction):
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
                "spawn_entity: `entity` is empty. Name the world's `name:` for that spawn -- the "
                "same string simulation_interfaces resolves, not a body name and not a TF frame.",
                action=self,
            )
        # The same parser a world document and SetEntityState/SpawnEntity go through, so an
        # orientation means one thing everywhere and a full pose is expressible.
        try:
            position, quat = parse_pose(pose)
        except PoseError as err:
            raise ActionError(f"spawn_entity: {err}", action=self) from None
        self._pos = (
            float(position[0]),
            float(position[1]),
            # `parse_pose` leaves z None to mean "the height the model rests at". A spawn is the
            # one caller that could resolve it, but only the compiled model knows -- so 0.0, and
            # a world whose robot rests higher states its z.
            0.0 if position[2] is None else float(position[2]),
        )
        self._quat = tuple(float(v) for v in quat)
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.set_entity_presence(
                    self._entity, True, self._pos, self._quat
                )
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            return self.waiting(f"spawning {self._entity!r} ({self.transport})")

        if not outcome.ok:
            return self.failed(
                f"could not spawn {self._entity!r} at {self._pos}: {outcome.detail}. Nothing the "
                "trial goes on to measure would be about the entity it asked for."
            )
        return self.satisfied(f"{self._entity!r} {outcome.detail}")
