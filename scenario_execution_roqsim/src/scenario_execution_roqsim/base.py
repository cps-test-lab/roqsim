# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""What every action here shares: a transport, a clock, and how it reports failure.

Three rules the subclasses inherit rather than restate, each learned the hard way:

**An authoring error raises; a runtime verdict is returned.** ``ActionError`` from ``update()`` is not
caught by the tick loop -- ``scenario_execution_base`` catches only ``KeyboardInterrupt`` -- so the run
dies with a traceback, no ``test.xml`` and no result row. For a campaign that is the worst possible
failure shape: the cell reads as "no data", indistinguishable from one that was never scheduled.
Measured today on an unrelated crash: 0 passed, 0 failed, nothing to read. So an unknown entity or a
missing plugin (which no run could ever recover from) raises, while "the fault did not land" -- a fact
about this trial -- returns FAILURE, which reaches ``on_scenario_shutdown`` with the message and the
last tree snapshot attached.

**Resolve the world every tick, cache nothing across resets.** The engine is built lazily and a
re-``reset()`` with different overrides replaces the model, the blackboard handles and the body ids. A
handle looked up in ``setup()`` would be a stale pointer into a torn-down world.

**These actions cannot run under ``remote()``.** The remote server is handed neither ``simulation`` nor
``node``, so there is nothing to reach; the modifier re-instantiates an action by entry-point name on
another machine, which is exactly what a simulation-reading action cannot survive.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError, BaseAction

from .access import AccessError, clock_of, select


class SimAction(BaseAction):
    """Base for an action that reads or drives an roqsim simulation over either transport."""

    def __init__(self):
        super().__init__()
        self._access = None
        self._clock = None

    def setup(self, **kwargs):
        """Pick the transport from what the runner offered. See :func:`access.select`."""
        try:
            self._access = select(kwargs, what=f"{self.__class__.__name__}")
        except AccessError as err:
            raise ActionError(str(err), action=self) from None
        self._clock = clock_of(kwargs)
        if self._clock is None:
            raise ActionError(
                f"{self.__class__.__name__} needs the runner's clock and got none: the stepped "
                "runner passes `clock`, the ROS runner `sim_clock`. Without one a dwell would have "
                "to use wall time, which does not agree with timeout() or with any recorded "
                "timestamp -- and under `pacing: asap` differs from sim time by orders of magnitude.",
                action=self,
            )

    def shutdown(self):
        if self._access is not None:
            self._access.teardown()

    # -- helpers for the subclasses --------------------------------------------------------------
    @property
    def now(self) -> float:
        """Sim-time seconds, from the runner's clock rather than from the transport."""
        return float(self._clock.now())

    @property
    def transport(self) -> str:
        return getattr(self._access, "transport", "unknown")

    def waiting(self, message: str) -> py_trees.common.Status:
        self.feedback_message = message  # pylint: disable=attribute-defined-outside-init
        return py_trees.common.Status.RUNNING

    def failed(self, message: str) -> py_trees.common.Status:
        """A statement about THIS TRIAL. Returned, so the run still produces a result row."""
        self.feedback_message = message  # pylint: disable=attribute-defined-outside-init
        return py_trees.common.Status.FAILURE

    def satisfied(self, message: str) -> py_trees.common.Status:
        self.feedback_message = message  # pylint: disable=attribute-defined-outside-init
        return py_trees.common.Status.SUCCESS

    def reraise(self, err: AccessError):
        """An AccessError is always an authoring error -- see the module docstring."""
        raise ActionError(
            f"{self.__class__.__name__} ({self.transport}): {err}", action=self
        ) from None
