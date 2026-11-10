# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``humanoid_<move>()``: start one of a humanoid's moves and wait until that start has ended.

A humanoid's moves (``roqsim_motion``: a walker with hands, or a humanoid robot) are the programs of
its ``humanoid`` machine, so each action here is :class:`~.machine_run.MachineRun` with the move's
parameters typed: it starts the move through the humanoid's ``start`` endpoint with a token of its
own, waits on that start's number through ``status``, fails the trial on the move's error (unless
``check: false``) and cancels its own start when its branch is abandoned::

    humanoid_approach(humanoid: 'worker', target: 'cabinet/drawer_0/grip', standoff: 0.65)
    humanoid_grasp(humanoid: 'worker', hand: 'right', target: 'cabinet/drawer_0/grip')
    humanoid_pull(humanoid: 'worker', hand: 'right', distance: 0.35)

The same scene in Python is ``yield from self.run(worker, "grasp", hand="right", target=...)``.
An optional parameter left out is not sent, so the move's own default applies: one with a default
is sent only when it differs from it, and one declared without a default (``facing``,
``max_force``) is ``None`` when a scenario leaves it out -- scenario-execution resolves a parameter
the invocation does not give and the action does not default to ``None``, as its own ``log(msg)``
relies on -- so unset is unset, never a magic number.
"""

from __future__ import annotations

import uuid

from scenario_execution.actions.base_action import ActionError

from .machine_run import MachineRun


class HumanoidMove(MachineRun):
    """Start ``MOVE`` on a humanoid with the given arguments; wait for that start."""

    MOVE = ""

    def _begin(self, humanoid: str, args: dict, check: bool) -> None:
        if not humanoid:
            raise ActionError(
                f"humanoid_{self.MOVE}: `humanoid` is required -- the name in the world of the walker "
                f"or robot whose `humanoid` component moves it",
                action=self,
            )
        self._machine, self._program = str(humanoid), self.MOVE
        self._args = {k: v for k, v in args.items() if v is not None}
        self._check = bool(check)
        # Cleared here, not in __init__: an action reached twice in one run starts the move twice.
        self._start = self._status = self._seq = None
        self._token = uuid.uuid4().hex
        self._ended = False


def _unset_or(value, what: str, action, check) -> float | None:
    """A parameter declared without a default: ``None`` when left out, else the number, refused
    where *check* says why."""
    if value is None:
        return None
    value = float(value)
    reason = check(value)
    if reason:
        raise ActionError(f"{what} is {value:g}: {reason}", action=action)
    return value


class HumanoidApproach(HumanoidMove):
    MOVE = "approach"

    def execute(
        self,
        humanoid: str,
        target: str,
        standoff: float = 0.6,
        facing: float | None = None,
        check: bool = True,
    ):
        facing = _unset_or(
            facing,
            "humanoid_approach: `facing`",
            self,
            lambda v: (
                "1000 meant unset; leave `facing` out to face the target's front"
                if v == 1000.0
                else ""
            ),
        )
        self._begin(
            humanoid, {"target": target, "standoff": float(standoff), "facing": facing}, check
        )


class HumanoidReach(HumanoidMove):
    MOVE = "reach"

    def execute(
        self,
        humanoid: str,
        hand: str,
        target: str,
        relative: str = "",
        orient: bool = True,
        pose: str = "open",
        check: bool = True,
    ):
        # `relative`, as `offset` is a word of OpenSCENARIO's own
        try:
            values = [float(v) for v in str(relative or "").split(",") if v.strip()]
        except ValueError:
            values = None
        if values is None or len(values) not in (0, 3, 6):
            raise ActionError(
                f"humanoid_reach: `relative` is 'x, y, z' or 'x, y, z, roll, pitch, yaw', not {relative!r}",
                action=self,
            )
        offset = values or None
        self._begin(
            humanoid,
            {
                "hand": hand,
                "target": target,
                "offset": offset,
                "orient": bool(orient),
                "pose": pose,
            },
            check,
        )


class HumanoidPress(HumanoidMove):
    MOVE = "press"

    def execute(
        self,
        humanoid: str,
        hand: str,
        target: str,
        max_force: float | None = None,
        check: bool = True,
    ):
        max_force = _unset_or(
            max_force,
            "humanoid_press: `max_force`",
            self,
            lambda v: (
                "a cap is above 0; leave `max_force` out for the humanoid's own "
                "`press: {max_force}`"
                if v <= 0.0
                else ""
            ),
        )
        self._begin(humanoid, {"hand": hand, "target": target, "max_force": max_force}, check)


class HumanoidGrasp(HumanoidMove):
    MOVE = "grasp"

    def execute(
        self, humanoid: str, hand: str, target: str, grip: str = "power", check: bool = True
    ):
        self._begin(humanoid, {"hand": hand, "target": target, "grip": grip}, check)


class HumanoidRelease(HumanoidMove):
    MOVE = "release"

    def execute(self, humanoid: str, hand: str = "both", check: bool = True):
        self._begin(humanoid, {"hand": hand}, check)


class HumanoidPull(HumanoidMove):
    MOVE = "pull"

    def execute(
        self, humanoid: str, hand: str, distance: float, speed: float = 0.2, check: bool = True
    ):
        self._begin(
            humanoid, {"hand": hand, "distance": float(distance), "speed": float(speed)}, check
        )


class HumanoidPush(HumanoidPull):
    MOVE = "push"


class HumanoidCarry(HumanoidMove):
    MOVE = "carry"

    def execute(self, humanoid: str, to: str, standoff: float = 0.6, check: bool = True):
        self._begin(humanoid, {"to": to, "standoff": float(standoff)}, check)


class HumanoidPlace(HumanoidMove):
    MOVE = "place"

    def execute(self, humanoid: str, at: str, release: bool = True, check: bool = True):
        self._begin(humanoid, {"at": at, "release": bool(release)}, check)


class HumanoidPlay(HumanoidMove):
    MOVE = "play"

    def execute(
        self,
        humanoid: str,
        clip: str,
        joints: str = "",
        weight: float = 1.0,
        loop: bool = False,
        fade: float = 0.3,
        speed: float = 1.0,
        check: bool = True,
    ):
        names = [j.strip() for j in str(joints or "").split(",") if j.strip()]
        self._begin(
            humanoid,
            {
                "clip": clip,
                "joints": names or None,
                "weight": float(weight),
                "loop": bool(loop),
                "fade": float(fade),
                "speed": float(speed),
            },
            check,
        )


class HumanoidStopClip(HumanoidMove):
    MOVE = "stop_clip"

    def execute(self, humanoid: str, clip: str = "", check: bool = True):
        self._begin(humanoid, {"clip": clip}, check)


class HumanoidLookAt(HumanoidMove):
    MOVE = "look_at"

    def execute(self, humanoid: str, target: str = "", check: bool = True):
        self._begin(humanoid, {"target": target}, check)
