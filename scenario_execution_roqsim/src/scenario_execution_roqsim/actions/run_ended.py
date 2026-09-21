# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``run_ended()``: succeed once the simulation has asked that the run end.

A trial plugin that knows the trial is over -- the peg is seated, the goal was reached, the episode
failed on its own criterion -- says so with :meth:`~roqsim.context.SimContext.request_stop`. Under
``roqsim sim`` that ends the run. Under scenario-execution the loop belongs to the tree, and the
request was a flag nobody read: a stepped scenario had to ``wait elapsed(N)`` past the slowest cell,
so every faster trial idled the difference out, and a trial that overran N was cut and mis-read as a
timeout. This action is the tree reading the flag, so the scenario ends where the trial does::

    import osc.roqsim

    scenario one_trial:
        timeout(60s)              # still the backstop, for a trial that never resolves
        do serial:
            run_ended()

Works over both transports: in a stepped run it reads ``ctx.stop_requested``; in a ROS run it polls
``simulation_interfaces/GetSimulationState`` for ``STATE_QUITTING``, which ``roqsim sim`` sets when it
honours the request. See :mod:`scenario_execution_roqsim.access`.

It never FAILS. Which outcome the trial reached is the plugin's to record and the analysis's to
grade; this only says that it reached one. A trial that never asks to stop is the scenario's own
``timeout()`` to catch.
"""

from __future__ import annotations

import py_trees

from ..access import AccessError
from ..base import SimAction


class RunEnded(SimAction):
    def __init__(self):
        super().__init__()
        self._call = None

    def execute(self):
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")
        try:
            if self._call is None:
                self._call = self._access.watch_stop()
            reason = self._call.poll()
        except AccessError as err:
            self.reraise(err)
        if reason is None:
            return self.waiting(f"waiting for the run to end ({self.transport})", self._call)
        return self.satisfied(f"run ended: {reason}")
