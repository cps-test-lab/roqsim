# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``zone_entered()``, ``zone_left()``, ``zone_dwelled()``: wait on a ``zone_trigger``.

The OpenSCENARIO forms of :meth:`roqsim.scenario.ScenarioPlugin.entered`, ``left`` and ``dwelled``,
with the same meaning. ``zone`` is the zone's address in the world (``gate``, or ``press.gate`` nested
under an entity), and ``pattern`` an fnmatch of entity names::

    zone_entered(zone: 'gate', pattern: 'box_*')
    zone_dwelled(zone: 'gate', since: 0.0)
    zone_left(zone: 'gate')

They read who is inside now, so ``zone_left`` succeeds at once on an empty zone. With ``since`` (s, sim
time, not negative; declared without a default, so a scenario that leaves it out passes ``None``)
``zone_entered`` and ``zone_dwelled`` also count an entity that has come and gone
since then, so a scene that was busy cannot miss one passing through. ``zone_dwelled`` on a zone whose
``dwell`` is 0, which reports no dwelling, raises, as an authoring error.
"""

from __future__ import annotations

import fnmatch

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction


class _ZoneWait(SimAction):
    def __init__(self):
        super().__init__()
        self._zone = ""
        self._pattern = "*"
        self._since: float | None = None
        self._call = None

    def _configure(self, zone: str, pattern: str, since: float | None) -> None:
        self._zone = str(zone or "")
        if not self._zone:
            raise ActionError(
                f"{type(self).__name__}: `zone` is required -- the zone's address in the world",
                action=self,
            )
        if since is not None and float(since) < 0.0:
            raise ActionError(
                f"{type(self).__name__}: `since` is {float(since):g}, a sim time before the run; leave "
                "`since` out for who is inside now",
                action=self,
            )
        self._pattern = str(pattern or "*")
        self._since = None if since is None else float(since)
        self._call = None

    def _matching(self, names) -> list[str]:
        return [n for n in names if fnmatch.fnmatchcase(n, self._pattern)]

    def _decide(self, occupancy: dict) -> py_trees.common.Status:
        raise NotImplementedError

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.not_ready()
        try:
            if self._call is None:
                self._call = self._access.entity_report(self._zone, "occupancy", whole=True)
            reading = self._call.poll()
        except AccessError as err:
            self.reraise(err)
        if reading is None:
            return self.waiting(f"reading zone {self._zone} ({self.transport})")
        return self._decide(reading.value)

    def _since_then(self, names, times) -> list[str]:
        return [
            n
            for n, t in zip(names, times, strict=True)
            if self._since is not None and t >= self._since
        ]


class ZoneEntered(_ZoneWait):
    def execute(self, zone: str, pattern: str = "*", since: float | None = None):
        self._configure(zone, pattern, since)

    def _decide(self, occupancy: dict) -> py_trees.common.Status:
        seen = self._since_then(occupancy["entered"], occupancy["entered_at"])
        found = self._matching([*occupancy["inside"], *seen])
        if found:
            return self.satisfied(f"{found[0]} entered zone {self._zone}")
        return self.waiting(f"waiting for {self._pattern!r} in zone {self._zone}")


class ZoneLeft(_ZoneWait):
    def execute(self, zone: str, pattern: str = "*"):
        self._configure(zone, pattern, None)

    def _decide(self, occupancy: dict) -> py_trees.common.Status:
        found = self._matching(occupancy["inside"])
        if not found:
            return self.satisfied(f"zone {self._zone} is empty of {self._pattern!r}")
        return self.waiting(f"waiting for {found[0]} to leave zone {self._zone}")


class ZoneDwelled(_ZoneWait):
    def execute(self, zone: str, pattern: str = "*", since: float | None = None):
        self._configure(zone, pattern, since)

    def _decide(self, occupancy: dict) -> py_trees.common.Status:
        if not occupancy["dwell"] > 0:
            raise ActionError(
                f"zone {self._zone} has dwell 0, which reports no dwelling; give it a dwell to "
                "wait on one",
                action=self,
            )
        seen = self._since_then(occupancy["dwelled"], occupancy["dwelled_at"])
        found = self._matching([*occupancy["dwelling"], *seen])
        if found:
            return self.satisfied(f"{found[0]} dwelled in zone {self._zone}")
        return self.waiting(f"waiting for {self._pattern!r} to dwell in zone {self._zone}")
