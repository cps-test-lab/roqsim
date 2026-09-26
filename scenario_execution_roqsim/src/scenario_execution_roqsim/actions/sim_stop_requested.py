# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``sim_stop_requested()``: succeed once something in the world has asked for the run to end.

A plugin that knows the trial is over -- a goal reached, a protective stop tripped
(``roqsim_sensors``' ``force_limit`` with ``stop_run``) -- calls ``ctx.request_stop(reason)``.
``roqsim sim`` owns its loop and leaves it on the request. In a scenario-execution run the scenario
owns the loop instead, and the request would only be logged; this action is how a scenario honours
it. It stays RUNNING until the request, so it is the wait itself -- invoked as an action, not after
``wait``, which takes an event condition and refuses an action::

    do parallel:
        serial:
            ...                         # the trial
            emit end
        serial:
            sim_stop_requested()        # end on the world's own verdict
            emit end                    # or `emit fail`, where a stop means a failed trial

It reports the request's reason in its feedback, so the tree snapshot a run leaves says why it
ended. Stepped runner only: over ROS nothing carries the request (see
:meth:`~scenario_execution_roqsim.access.ros.RosAccess.stop_request`), and asking raises rather
than waiting on something that can never arrive.
"""

from __future__ import annotations

import py_trees

from ..access import AccessError
from ..base import SimAction


class SimStopRequested(SimAction):
    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")
        try:
            request = self._access.stop_request()
        except AccessError as err:
            self.reraise(err)
        if not request.requested:
            return self.waiting(f"no stop requested ({self.transport})")
        return self.satisfied(
            f"stop requested: {request.reason or '(no reason given)'}; seen at sim {self.now:.3f} s"
        )
