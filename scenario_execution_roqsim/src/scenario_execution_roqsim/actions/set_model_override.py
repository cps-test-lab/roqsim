# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``set_model_override()``: apply or restore a world's runtime fault, and say whether it landed.

``model_override`` (see :mod:`roqsim.plugins.model_override`) changes named model values -- pad friction,
a collision mask, an actuator's force limit, a mass -- while a run is in progress, and restores them
exactly. It deliberately owns no trigger: a fault's timing is the experiment's independent variable,
and severity is the world's configured ``to:`` value, so both stay campaign factors instead of message
payloads. This action is the trigger, and it is the ONLY thing that crosses the wire: one bit.

Composes with ``entity_moved`` rather than fusing with it. Its predecessor in ``tiago_pick`` did
both jobs plus a never-succeed composition policy, and three would-be reusers wanted the condition
alone, one wanted a different effect, and one wanted the opposite terminal policy.

**A fault that did not land fails the trial** (``require_landed``), because the alternative is a row
that claims a fault which never happened -- an unfaulted outcome wearing a faulted label, which is
worse than a failed run. Two things are NOT that:

``untested``
    nothing was touching the selected geoms, so there was nothing to verify. Usually means the object
    was already gone, which is a result.
a restore
    ``active: false`` puts back saved values; there is no contact to verify, and the plugin reports
    ``untested`` by construction.

FAILURE rather than a raise, deliberately: an exception from ``update()`` is not caught by the tick
loop, so the run would die with a traceback and leave no ``test.xml`` and no result row -- a campaign
cell that reads as "never scheduled". See :mod:`scenario_execution_roqsim.base`.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction

#: The plugin's verdict for "the write landed in the model and changed nothing". Compared by VALUE
#: rather than imported from ``roqsim.plugins.model_override``: that import pulls MuJoCo into the
#: behaviour-tree build, which happens before any world is compiled.
_NO_EFFECT = "no_effect"


class SetModelOverride(SimAction):
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
                "set_model_override: `instance` is empty. Name the world's `model_override` instance "
                "by its `name:` -- the same string its blackboard handle and its ROS endpoints are "
                "scoped by (`model_override:grip_fault`, `/grip_fault/override`).",
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
                self._call = self._access.apply_override(self._instance, self._active)
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            # In-process: the write is queued and lands in the next `pre_step`. Over ROS: the service
            # future has not completed. Either way one tick, and neither may be waited on by blocking.
            return self.waiting(
                f"{'applying' if self._active else 'restoring'} {self._instance!r} "
                f"({self.transport})"
            )

        what = "applied" if self._active else "restored"
        if not outcome.ok:
            return self.failed(
                f"the simulator did not {what[:-1]} {self._instance!r}: {outcome.detail}. The fault "
                "was not injected, so this trial's outcome would be a nominal one carrying a faulted "
                "label."
            )
        if self._active and self._require_landed and outcome.verified == _NO_EFFECT:
            return self.failed(
                f"fault {self._instance!r} was applied and changed nothing: the plugin read the "
                f"APPLIED contact back and reports 'no_effect' ({outcome.detail}). Its selection does "
                "not govern the contact -- MuJoCo takes friction from the geom with the higher "
                "`priority`, and at equal priority the element-wise maximum of the two. "
                "`roqsim scenes describe <world> --overridable '<glob>'` reports each candidate's current "
                "value and priority."
            )
        verdict = f", verdict {outcome.verified!r}" if outcome.verified else ""
        return self.satisfied(f"{self._instance!r} {what} {outcome.detail}{verdict}")
