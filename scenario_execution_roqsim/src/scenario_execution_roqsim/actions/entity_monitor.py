# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_monitor()``: keep a scenario variable equal to a value a plugin publishes about an entity.

The simulator's counterpart of ``osc.ros``'s ``topic_monitor``. A plugin that knows something about
an entity -- a ``force_limit`` that tripped, a ``clearance`` distance, a trial plugin's own
``resolved`` -- registers it as an ``out`` endpoint on that entity, and this writes one field of it
into a variable on every tick a reading arrives::

    var tripped: bool = false
    do parallel:
        entity_monitor(entity: 'ur5e', value: 'force_limit.tripped', target_variable: tripped)
        serial:
            wait tripped == true
            emit end

Conditions are then plain OpenSCENARIO over the variable (``wait``, ``until``, comparisons with
parameters, combined expressions), so this action decides nothing itself.

``value`` is ``<endpoint>.<field>`` as the world names it, never a topic; a bare ``<endpoint>``
means the field its ROS publication carries. In-process a reading is taken on every tick; over the
control socket a read is a round-trip, so the variable follows at the tick period of the replies.

It never succeeds: it runs until its branch is ended (the scenario's ``emit end``, an ``until``,
the other branch of a ``one_of``). Until the first reading arrives the variable keeps its declared
default. An entity, endpoint or field that does not exist, and a field that is not a single number,
flag or string, raise with the same text on both transports -- authoring errors, see
:mod:`scenario_execution_roqsim.base`.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError
from scenario_execution.model.types import VariableReference

from ..access import AccessError
from ..base import SimAction


class EntityMonitor(SimAction):
    """Writes ``<endpoint>.<field>`` of ``entity`` into ``target_variable``; RUNNING until ended."""

    def __init__(self):
        # The variable is handed over as a reference to write to, not resolved to its value.
        super().__init__(resolve_variable_reference_arguments_in_execute=False)
        self._entity = ""
        self._report = ""
        self._field = ""
        self._target = None
        self._call = None

    def execute(self, entity: str, value: str, target_variable: object):
        # None is what the parser hands over for an omitted required argument, so each is refused
        # by name rather than read as an entity called 'None'.
        if not entity:
            raise ActionError(
                "entity_monitor: `entity` is empty. Name the entity whose plugin publishes the "
                "value -- the world's `name:` for it, e.g. entity: 'ur5e'.",
                action=self,
            )
        value = str(value or "")
        self._report, _, self._field = value.partition(".")
        if not self._report:
            raise ActionError(
                f"entity_monitor: `value` {value!r} names no endpoint. Write '<endpoint>.<field>' "
                "(value: 'force_limit.tripped'), or a bare '<endpoint>' for the field it publishes.",
                action=self,
            )
        if not isinstance(target_variable, VariableReference):
            raise ActionError(
                "entity_monitor: `target_variable` must name a variable (a `var` of the scenario "
                f"or of an actor), got {target_variable!r}.",
                action=self,
            )
        self._entity = str(entity)
        self._target = target_variable
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            # As in entity_moved: the stepped runner builds the world on the first reset.
            return self.waiting("waiting for the simulation", self._access)

        name = f"{self._entity}.{self._report}"
        try:
            if self._call is None:
                self._call = self._access.entity_report(self._entity, self._report, self._field)
            reading = self._call.poll()
        except AccessError as err:
            self.reraise(err)
        if reading is None:
            return self.waiting(f"waiting for {name} over {self.transport}", self._call)

        self._target.set_value(reading.value)
        shown = f"{name}.{reading.field}" if reading.field else name
        return self.waiting(f"{shown} = {reading.value!r} -> {self._target.ref} ({self.transport})")
