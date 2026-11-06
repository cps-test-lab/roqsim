# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``machine_run()``: run a program of a machine and wait until that start has ended.

The OpenSCENARIO form of :meth:`roqsim.scenario.ScenarioPlugin.run`, with the same outcome::

    machine_run(machine: 'press', program: 'stroke')
    machine_run(machine: 'press', program: 'stroke', args: 'depth=0.1, show=hole')

``args`` is ``name=value`` pairs, each value JSON or a bare word, since an OpenSCENARIO string cannot
hold the double quotes a JSON mapping needs; a list of numbers is one value. It waits on the start's own number, through the machine's ``status``, so a later start is not taken
for this one. With ``check`` the machine's error, or another start following this one before it
ended, fails the trial; without it the action succeeds whatever the program did. A branch abandoned
while it waits (a ``one_of``, a ``timeout``) cancels this start by its number, so the machine is free
for the next start and a start another client made is left alone. Abandoned before the start's reply
came, it cancels the start by the number the reply names
(:meth:`~scenario_execution_roqsim.access.WorldAccess.cancel_start`).

A machine that does not exist raises, as an authoring error; a refused start (the machine is busy, an
unknown program or argument) is a FAILURE. See :mod:`scenario_execution_roqsim.base`.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError, parse_value
from ..base import SimAction


class MachineRun(SimAction):
    def __init__(self):
        super().__init__()
        self._machine = ""
        self._program = ""
        self._args = {}
        self._check = True
        self._start = None
        self._status = None
        self._seq = None
        self._ended = False

    def execute(self, machine: str, program: str, args: str = "", check: bool = True):
        self._machine, self._program = str(machine or ""), str(program or "")
        if not self._machine or not self._program:
            raise ActionError(
                "machine_run: `machine` and `program` are both required -- the machine's name in "
                "the world and the program to run. `roqsim plugins describe <machine>` lists them.",
                action=self,
            )
        self._args = {}
        for pair in self._pairs(str(args or "")):
            key, equals, value = pair.partition("=")
            if not equals or not key.strip():
                raise ActionError(
                    f"machine_run: `args` is name=value pairs, and {pair!r} is not one", action=self
                )
            self._args[key.strip()] = parse_value(value.strip())
        self._check = bool(check)
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active, so an
        #: action reached twice in one run starts the program twice.
        self._start = self._status = self._seq = None
        self._ended = False

    def _pairs(self, args: str) -> list[str]:
        """``args`` split at the commas outside any brackets, so a list is one value."""
        pairs, current, depth = [], "", 0
        for char in args:
            depth += (char in "[{") - (char in "]}")
            if char == "," and depth == 0:
                pairs.append(current)
                current = ""
            else:
                current += char
        if depth != 0:
            raise ActionError(f"machine_run: `args` {args!r} has unbalanced brackets", action=self)
        return [pair.strip() for pair in [*pairs, current] if pair.strip()]

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.not_ready()
        name = f"{self._machine}.{self._program}"
        try:
            if self._start is None:
                self._start = self._access.call_endpoint(
                    self._machine, "start", {"program": self._program, "args": self._args}
                )
            if self._seq is None:
                outcome = self._start.poll()
                if outcome is None:
                    return self.waiting(f"starting {name} ({self.transport})", self._start)
                if not outcome.ok:
                    self._ended = True
                    return self.failed(f"{name} was not started: {outcome.detail}")
                self._seq = int(outcome.result)
                self._status = self._access.entity_report(self._machine, "status", whole=True)
            reading = self._status.poll()
        except AccessError as err:
            self.reraise(err)
        if reading is None:
            return self.waiting(f"running {name} (seq {self._seq})")
        status = reading.value
        if status["seq"] > self._seq:
            self._ended = True
            if self._check:
                return self.failed(
                    f"{name} (seq {self._seq}) was followed by seq {status['seq']} before it ended"
                )
            return self.satisfied(f"{name} (seq {self._seq}) was followed by seq {status['seq']}")
        if status["seq"] < self._seq or not status["finished"]:
            return self.waiting(f"running {name} (seq {self._seq})")
        self._ended = True
        if status["error"] and self._check:
            return self.failed(f"{name} (seq {self._seq}): {status['error']}")
        return self.satisfied(f"{name} (seq {self._seq}) ended: {status['error'] or 'done'}")

    def request_cancel(self) -> bool:
        """Cancel this start when the branch this sits in is abandoned, or it runs on behind the
        scene and the machine's next start is refused as busy. By its number, so a start another
        client made is never cancelled: one that ended or was refused has nothing to cancel. It
        cancels once."""
        if self._start is None or self._ended:
            return True
        self._ended = True
        if self._seq is None:
            outcome = self._start.poll()
            if outcome is None:
                self._access.cancel_start(self._machine, self._start)
                return True
            if not outcome.ok:
                return True  # refused: nothing of this action's to cancel
            self._seq = int(outcome.result)
        self._access.call_endpoint(self._machine, "cancel", {"seq": self._seq})
        return True
