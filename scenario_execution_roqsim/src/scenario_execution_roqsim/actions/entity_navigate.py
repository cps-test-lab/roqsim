# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``entity_navigate()``: send an entity somewhere, and wait until it gets there.

**This drives the apparatus, not the subject.** The mover is the *simulator's* -- a ``navigator``
component on that entity plans and drives it internally, with no bridge, no localisation and no
external stack -- so it is the traffic a trial puts around the robot under test. ``osc.nav2``'s
``nav_to_pose`` is the other thing entirely: it commands a real nav2 stack over ROS, and that stack
is what the experiment is measuring.

Keeping them as two actions rather than one with a flag is the point. They differ in what they prove:
a goal reached through nav2 says something about nav2, and a goal reached through this says only that
the scenario's traffic arrived on cue.

With no ``goal_poses``, this **starts the route the entity was configured with** rather than sending
a new one. That is what lets a world own an opponent's trajectory -- identical in every repetition,
and visible in a campaign's config diff -- while the scenario owns only its timing.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError
from ..base import SimAction


class EntityNavigate(SimAction):
    def __init__(self):
        super().__init__()
        self._entity = ""
        self._poses: list[tuple[float, float, float]] = []
        self._wait = True
        self._action_name = ""
        self._call = None

    def execute(  # noqa: D102 - the OSC signature is documented in roqsim.osc
        self,
        entity: str,
        goal_poses=None,
        success_on_acceptance: bool = False,
        action_name: str = "",
    ) -> None:
        if not entity:
            raise ActionError("entity_navigate: `entity` is required", action=self)
        self._entity = entity
        self._action_name = action_name or ""
        self._wait = not bool(success_on_acceptance)
        self._poses = []
        for i, pose in enumerate(goal_poses or []):
            position = (pose or {}).get("position") or {}
            orientation = (pose or {}).get("orientation") or {}
            if any(float(orientation.get(k, 0.0)) for k in ("roll", "pitch", "yaw")):
                raise ActionError(
                    f"entity_navigate: goal_poses[{i}].orientation is nonzero. The navigator drives "
                    "to a POSITION and stops facing the way it arrived -- it has no final-heading "
                    "control, so an orientation here would be accepted and then quietly dropped. "
                    "Give positions only; if the mover must end up facing a particular way, that is "
                    "a capability to add to the navigator rather than a value to pass and ignore.",
                    action=self,
                )
            self._poses.append((float(position.get("x", 0.0)), float(position.get("y", 0.0))))
        #: Cleared here rather than in ``__init__``: ``execute`` runs each time the action becomes
        #: active, so an action reached twice in one run sends the route twice instead of replaying
        #: the first outcome.
        self._call = None

    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            return self.waiting("waiting for the simulation")

        try:
            if self._call is None:
                self._call = self._access.navigate(
                    self._entity, self._poses, wait=self._wait, action_name=self._action_name
                )
            outcome = self._call.poll()
        except AccessError as err:
            self.reraise(err)

        if outcome is None:
            where = f"through {len(self._poses)} pose(s)" if self._poses else "on its own route"
            return self.waiting(f"navigating {self._entity!r} {where} ({self.transport})")

        if not outcome.ok:
            return self.failed(f"{self._entity!r} did not arrive: {outcome.detail}")
        return self.satisfied(f"{self._entity!r} {outcome.detail}")

    def request_cancel(self) -> bool:
        """Stop the mover when the branch this sits in is abandoned.

        Without this a ``one_of`` losing its race, or a ``timeout`` firing, would leave the opponent
        driving to a goal the scenario has given up on -- still crossing the robot under test's path,
        minutes after the phase that wanted it there ended. None of the other actions here needs a
        cancel because none of them leaves anything running behind it; this one does.
        """
        if self._call is not None:
            self._call.cancel()
        return True


class EntityNavigateStart(EntityNavigate):
    """``entity_navigate_start()``: run the route the entity was configured with.

    The same machinery with no poses of its own -- one implementation, so the sequence-number
    polling, the transports and the cancel path cannot drift between the two actions.
    """

    def execute(  # noqa: D102 - the OSC signature is documented in roqsim.osc
        self, entity: str, success_on_acceptance: bool = False, action_name: str = ""
    ) -> None:
        super().execute(
            entity,
            goal_poses=None,
            success_on_acceptance=success_on_acceptance,
            action_name=action_name,
        )
