# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``delete_entity()``: make a declared entity absent -- unseen, untouchable, unlisted -- where it is.

**Absence, not removal.** roqsim never recompiles the model at runtime, so an entity cannot be taken
out of it; this is ``spawn_entity`` run the other way. What goes is everything that could perceive or
touch the entity: no sensor sees it, nothing collides with it, and the control plane stops listing
it. Its pose stays where it was, so a later ``spawn_entity`` can bring it back -- and a free body is
frozen rather than left to fall while it is away (``roqsim.presence``).

Like every action here it runs over either transport (``roqsim.access``): in a stepped run the flip
is a posted callback on the physics thread, and in a ROS run it is ``simulation_interfaces``'
``DeleteEntity``. The scenario is written once and does not learn which shape it is in.

``scenario_execution_sim`` declares a ``delete_entity`` too, for simulators that destroy entities.
As with ``spawn_entity``, an invocation binds to the implementation from the library the scenario
imported, so importing ``osc.roqsim`` is what selects this one.

A delete that did not land **fails the trial**: an entity still present is still an obstacle, and
every row measured after this point would be about a world the scenario did not ask for. Deleting an
entity that is already absent is refused for the same reason both transports refuse it -- the caller
asked for a change and none happened.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction


class DeleteEntity(SimAction):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._call = None

    def execute(self, entity: str):
        self._entity = str(entity)
        if not self._entity:
            raise ActionError(
                "delete_entity: `entity` is empty. Name the world's `name:` for that entry -- the "
                "same string simulation_interfaces resolves, not a body name and not a TF frame.",
                action=self,
            )
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.set_entity_presence(self._entity, False)
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            return self.waiting(f"deleting {self._entity!r} ({self.transport})", self._call)

        if not outcome.ok:
            return self.failed(
                f"could not delete {self._entity!r}: {outcome.detail}. It is still in the world, "
                "so nothing the trial goes on to measure would be about the world it asked for."
            )
        return self.satisfied(f"{self._entity!r} {outcome.detail}")
