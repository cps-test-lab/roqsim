# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_reports()``: succeed once a value a plugin publishes about an entity compares as expected.

The way a scenario reads a trial's outcome. A plugin that knows something about an entity -- a
``force_limit`` that tripped, a ``contact`` monitor, a trial plugin's own ``resolved`` -- registers it
as an ``out`` endpoint on that entity, and this waits on one field of it::

    entity_reports(entity: 'ur5e', report: 'force_limit.tripped', expected_value: 'True')
    emit end

``report`` is ``<endpoint>.<field>`` as the world names it, never a topic; a bare ``<endpoint>``
means the field its ROS publication carries, so the short form compares the same value on both
transports. The comparison arguments are shaped like ``osc.ros``'s ``check_data``: ``expected_value``
is a Python literal (``ast.literal_eval``), ``comparison_operator`` one of the ``operator`` module's
six, and ``fail_if_bad_comparison`` turns a comparison that does not hold into FAILURE instead of
waiting. ``dwell`` is ``entity_moved``'s: the comparison must hold continuously for that much sim time,
and restarts when it stops holding.

Works over both transports -- see :meth:`scenario_execution_roqsim.access.WorldAccess.entity_report`.
In a stepped run every field of the report is readable; over ROS only the published one travels.
"""

from __future__ import annotations

import operator
from ast import literal_eval

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction
from .entity_condition import enum_name

#: ``comparison_operator``'s members, as ``osc.ros``'s ``check_data`` maps them.
OPERATORS = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
}
_SYMBOLS = {"lt": "<", "le": "<=", "eq": "==", "ne": "!=", "ge": ">=", "gt": ">"}


class EntityReports(SimAction):
    """Succeeds once ``<report>.<field>`` of ``entity`` compares as expected, for ``dwell`` seconds."""

    def __init__(self):
        super().__init__()
        self._entity = ""
        self._report = ""
        self._field = ""
        self._expected = None
        self._op = "eq"
        self._dwell = 0.0
        self._fail_if_bad = False
        self._call = None
        #: When the comparison last started holding, in sim time; None while it does not.
        self._since: float | None = None

    def execute(
        self,
        entity: str,
        report: str,
        expected_value: str,
        comparison_operator,
        dwell: float,
        fail_if_bad_comparison: bool,
    ):
        # None is what the parser hands over for an omitted required argument, so each of the
        # three is refused by name rather than read as a report called 'None'.
        if not entity:
            raise ActionError(
                "entity_reports: `entity` is empty. Name the entity whose plugin publishes the "
                "report -- the world's `name:` for it, e.g. entity: 'ur5e'.",
                action=self,
            )
        report = str(report or "")
        self._report, _, self._field = report.partition(".")
        if not self._report:
            raise ActionError(
                f"entity_reports: `report` {report!r} names no report. Write '<report>.<field>' "
                "(report: 'force_limit.tripped'), or a bare '<report>' for the field it publishes.",
                action=self,
            )
        if not isinstance(expected_value, str):
            raise ActionError(
                "entity_reports: `expected_value` must be a string holding a Python literal "
                f"('True', '0.5', \"'resolved'\"), got {expected_value!r}.",
                action=self,
            )
        try:
            self._expected = literal_eval(expected_value)
        except (ValueError, SyntaxError) as err:
            raise ActionError(
                f"entity_reports: cannot read `expected_value` {expected_value!r} as a Python "
                f"literal ({err}). A string is quoted inside the string: expected_value: "
                "\"'resolved'\".",
                action=self,
            ) from None
        self._op = enum_name(comparison_operator)
        if self._op not in OPERATORS:
            raise ActionError(
                f"entity_reports: unknown `comparison_operator` {self._op!r}; known: "
                f"{', '.join(OPERATORS)}.",
                action=self,
            )
        self._dwell = float(dwell)
        if self._dwell < 0.0:
            raise ActionError(
                f"entity_reports: `dwell` must be >= 0, got {self._dwell}.", action=self
            )
        self._entity = str(entity)
        self._fail_if_bad = bool(fail_if_bad_comparison)
        self._call = None
        self._since = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            # As in entity_moved: the stepped runner builds the world on the first reset.
            return self.waiting("waiting for the simulation")

        name = f"{self._entity}.{self._report}"
        try:
            if self._call is None:
                self._call = self._access.entity_report(self._entity, self._report, self._field)
            reading = self._call.poll()
        except AccessError as err:
            self.reraise(err)
        if reading is None:
            return self.waiting(f"waiting for {name} over {self.transport}", self._call)

        shown = f"{name}.{reading.field}" if reading.field else name
        try:
            holds = bool(OPERATORS[self._op](reading.value, self._expected))
        except TypeError as err:
            # The literal cannot be compared with what the report holds (a string against a
            # number): no value the run could produce would change that, so it is an authoring error.
            raise ActionError(
                f"entity_reports: {shown} is {reading.value!r} ({type(reading.value).__name__}), "
                f"which cannot be compared {_SYMBOLS[self._op]} {self._expected!r} "
                f"({type(self._expected).__name__}): {err}",
                action=self,
            ) from None
        comparison = f"{shown} = {reading.value!r} (want {_SYMBOLS[self._op]} {self._expected!r})"

        if not holds:
            # Restarts the dwell, as entity_moved's does: a crossing that does not hold is not one.
            self._since = None
            if self._fail_if_bad:
                return self.failed(f"{comparison} ({self.transport})")
            return self.waiting(comparison)

        if self._since is None:
            self._since = self.now
        held = self.now - self._since
        if held < self._dwell:
            return self.waiting(f"{comparison}; held {held:.2f} s of {self._dwell:.2f} s")
        return self.satisfied(
            comparison + (f" for {held:.2f} s" if self._dwell else "") + f" ({self.transport})"
        )
