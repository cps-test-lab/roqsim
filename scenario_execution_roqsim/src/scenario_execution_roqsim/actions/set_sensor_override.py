# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``set_sensor_override()``: degrade a sensor mid-run, and say whether it landed.

The sensor-report twin of :mod:`~scenario_execution_roqsim.actions.set_model_override`. That one
switches a ``model_override`` instance -- a change to the PHYSICS. This one switches the ``fault:``
block a sensor carries in its own config -- a change to what the sensor REPORTS. roqsim keeps the two
apart deliberately (``roqsim.plugins.model_override``: "a perturbation of a reported value is sensor
noise and belongs in a sensor's own config"), and the split survives up here: two actions, one per
channel, rather than one verb that has to be told which world it is talking about.

Like its twin it owns only WHEN. Severity is the world's configured ``fault:`` value, so sweeping how
bad the fault gets is an ordinary campaign factor (``components.robot.lidar.fault.dropout_percent``)
and is deterministic per cell; what crosses the wire is one bit.

``instance`` is a COMPONENT ADDRESS -- ``robot.lidar``, the dotted path of labels a world publishes,
not the bare plugin ref. A robot may carry two lidars, and a bare ``lidar`` would name neither.

**A fault that did not land fails the trial** (``require_landed``), for the reason the twin gives: a
row that claims a fault which never happened is an unfaulted outcome wearing a faulted label, which
is worse than a failed run. A *restore* is not that -- there is nothing to verify when values are put
back -- and reports ``untested`` by construction.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction

#: The verdict for "the write landed and changed nothing". Compared BY VALUE rather than imported
#: from ``roqsim_sensors.live_config``: that import pulls MuJoCo into the behaviour-tree build, which
#: happens before any world is compiled. Same reasoning as set_model_override's copy of it.
_NO_EFFECT = "no_effect"


class SetSensorOverride(SimAction):
    def __init__(self):
        super().__init__()
        self._instance = ""
        self._active = True
        self._require_landed = True
        self._call = None

    def execute(self, instance: str, active: bool, require_landed: bool):
        self._instance = str(instance)
        if not self._instance:
            raise ActionError(
                "set_sensor_override: `instance` is empty. Name the sensor by its COMPONENT "
                "ADDRESS -- the dotted path of labels from the top of the world document, e.g. "
                "'robot.lidar'. `roqsim scenes describe <world>` lists the addresses a world "
                "publishes.",
                action=self,
            )
        self._active = bool(active)
        self._require_landed = bool(require_landed)
        #: Cleared here, not in __init__: `execute` runs each time the action becomes active, so an
        #: action reached twice in one run fires twice rather than replaying the first outcome.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.apply_override(
                    self._instance, self._active, kind="sensor"
                )
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            # In-process: the write is queued and lands in the next `pre_step`. Over ROS: the service
            # future has not completed. Either way one tick, and neither may be waited on by blocking.
            return self.waiting(
                f"{'applying' if self._active else 'restoring'} the fault on {self._instance!r} "
                f"({self.transport})"
            )

        what = "applied" if self._active else "restored"
        if not outcome.ok:
            return self.failed(
                f"the simulator did not {what[:-1]} the fault on {self._instance!r}: "
                f"{outcome.detail}. The fault was not injected, so this trial's outcome would be a "
                "nominal one carrying a faulted label."
            )
        if self._active and self._require_landed and outcome.verified == _NO_EFFECT:
            return self.failed(
                f"the fault on {self._instance!r} was applied and changed nothing: every key in its "
                f"`fault:` block already held the value the fault sets ({outcome.detail}). Either "
                "the block restates the nominal config, or a campaign override moved the nominal "
                "onto it."
            )
        verdict = f", verdict {outcome.verified!r}" if outcome.verified else ""
        return self.satisfied(f"{self._instance!r} {what} {outcome.detail}{verdict}")
