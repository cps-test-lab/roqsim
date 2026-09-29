# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_call()``: send any command a world's plugins declare, and say whether it landed.

One action for every command endpoint -- a fault switched on (``model_override``'s ``override``), a
sensor degraded (a ``fault:`` block's ``override``), a tare, a gripper opened -- addressed as the
world names it: the entity that owns the endpoint and the endpoint's name, or the component that
declares it (``robot.lidar``) where one entity has two of that name::

    entity_call(entity: 'grip_fault', command: 'override', value: 'true')
    entity_call(entity: 'robot.lidar', command: 'override', value: 'false')
    entity_call(entity: 'ur5e', command: 'force_torque/tare')

``value`` is JSON (a bare word is a string); an endpoint declared with typed parameters takes a
mapping of their names, or, where it declares exactly one, a bare value for it (``'true'`` for
``override(data: bool)``). The world owns what a command does and how much -- a fault's ``to:``, a
sensor's ``fault:`` values -- so those stay campaign factors, and this owns only WHEN.

**A command that did not land fails the trial** (``require_verified``). Where the endpoint names one
that confirms it, the outcome carries that endpoint's value as recorded in the step that applied the
command, and a verdict of ``no_effect`` fails: a row that claims a fault which never happened is an
unfaulted outcome wearing a faulted label, which is worse than a failed run. ``untested`` (nothing to
verify, e.g. a restore) is not that. A confirmation that could not be read -- the simulator was
paused -- fails too, since nothing says the command did anything.

FAILURE rather than a raise for everything about THIS trial -- a producer's refusal, no outcome in
time, a verdict of ``no_effect`` -- because an exception from ``update()`` is not caught by the tick
loop and the run would leave no result row. An endpoint that does not exist is an authoring error
and raises. See :mod:`scenario_execution_roqsim.base`.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError, parse_value
from ..base import SimAction

#: The verdict for "the write landed and changed nothing". Compared by VALUE rather than imported
#: from the plugins that report it: that import pulls MuJoCo into the behaviour-tree build, which
#: happens before any world is compiled.
_NO_EFFECT = "no_effect"


class EntityCall(SimAction):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._command = ""
        self._value = None
        self._require_verified = True
        self._call = None

    def execute(self, entity: str, command: str, value: str = "", require_verified: bool = True):
        self._entity, self._command = str(entity or ""), str(command or "")
        if not self._entity or not self._command:
            raise ActionError(
                "entity_call: `entity` and `command` are both required -- the entity that owns "
                "the command (or the component that declares it) and the command's name. "
                "`roqsim endpoints` lists a running world's.",
                action=self,
            )
        self._value = parse_value(value) if value != "" else None
        self._require_verified = bool(require_verified)
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active, so an
        #: action reached twice in one run sends twice rather than replaying the first outcome.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation", self._access)

        name = f"{self._entity}/{self._command}"
        try:
            if self._call is None:
                self._call = self._access.call_endpoint(self._entity, self._command, self._value)
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            return self.waiting(f"calling {name} ({self.transport})", self._call)
        if not outcome.ok:
            return self.failed(
                f"{name} did not apply: {outcome.detail}. Whatever it was to change in this trial "
                "did not happen."
            )
        if self._require_verified and not outcome.confirmed:
            return self.failed(
                f"{name} applied but could not be confirmed: {outcome.detail or 'no verdict'}. "
                "Nothing says it changed anything, so this trial must not be scored as if it did."
            )
        if self._require_verified and outcome.verified == _NO_EFFECT:
            return self.failed(
                f"{name} applied and changed nothing: its confirmation reports 'no_effect' "
                f"({outcome.confirmation!r}). For a model_override, MuJoCo takes friction from the "
                "geom with the higher `priority`, and at equal priority the element-wise maximum of "
                "the two -- `roqsim scenes describe <world> --overridable '<glob>'` shows each "
                "candidate's value; for a sensor fault, every key already held the faulted value."
            )
        verdict = f", verdict {outcome.verified!r}" if outcome.verified else ""
        return self.satisfied(f"{name} applied{verdict}")
