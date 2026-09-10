# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``sim_command()``: press a button a plugin declared, whatever plugin that is.

The generic door onto the simulator. Every other action in this package names one capability and
is written three times over -- an abstract method, an in-process implementation, a ROS
implementation -- so a plugin's new command costs four files here and makes this package name that
plugin. This one names no capability. It resolves whatever the world's plugins registered as an
inbound endpoint with no argument, so a plugin becomes reachable from a scenario by declaring one
and nothing here changes.

The registry it resolves through is the one the ROS bridge already serves from
(``ctx.interface``), which is what makes a command reachable on **both** transports at once: in a
stepped run the endpoint's ``write`` is posted to the physics thread, and over ROS the bridge is
already advertising it as ``<owner>/<command>``.

**A command, not a query.** It carries nothing and answers only whether the simulator ran it --
there is no verdict to branch on, because inventing one would mean this package knowing what each
plugin does, which is the coupling it exists to avoid. A plugin whose effect a scenario must check
publishes that as its own endpoint, and the scenario reads it with the condition actions.

Fails the trial when the world declares no such command, on the same terms as an entity that was
never spawned: a scenario naming something the world does not have is a scenario written against a
different world, and continuing would measure a run in which the command never happened.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction


class SimCommand(SimAction):
    def __init__(self):
        super().__init__()
        self._command = ""
        self._call = None

    def execute(self, command: str):
        self._command = str(command)
        if not self._command:
            raise ActionError(
                "sim_command: `command` is empty. A command is addressed as the simulator "
                "advertises it -- the producer's scope and the endpoint's name, e.g. 'ft/tare'. "
                "`roqsim scenes describe <world>` lists what a world offers.",
                action=self,
            )
        # Cleared here, not in __init__: `execute` runs each time the action becomes active, so an
        # action reached twice in one run fires twice rather than replaying the first outcome.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.send_command(self._command)
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            return self.waiting(f"sending {self._command!r} ({self.transport})", self._call)

        if not outcome.ok:
            return self.failed(
                f"the simulator refused {self._command!r}: {outcome.detail}. "
                "The command did not happen, so this trial ran without whatever it was for."
            )
        return self.satisfied(outcome.detail)
