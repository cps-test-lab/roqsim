# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Baseline, dwell and quantifier over a list of entities. The part both entity conditions share.

Subclasses supply only the measured quantity: ``entity_moved`` a displacement in metres,
``entity_rotated`` an angle in radians. Everything else -- where the baseline comes from, what a
continuous dwell means, how a list is quantified, what the feedback message says -- is here, once,
because those are the parts that are easy to get subtly wrong and pointless to get wrong twice.
"""

from __future__ import annotations

import py_trees
from scenario_execution.actions.base_action import ActionError

from ..access import AccessError, Pose
from ..base import SimAction


def enum_name(value) -> str:
    """The member NAME of an OSC enum argument.

    An enum reaches an action as ``(member_name, numeric_value)`` -- not as the bare name, and not as
    the ``int`` the framework's own type hints claim (``get_comparison_operator`` indexes ``[0]``).
    A plain string is accepted too, so a test or another driver can call the action directly.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value[0])
    except (TypeError, IndexError, KeyError):
        return str(value)


class EntityCondition(SimAction):
    """Succeeds once the measured quantity has met its threshold for ``dwell`` seconds."""

    def __init__(self):
        super().__init__()
        self._entities: list[str] = []
        self._dwell = 0.0
        self._require = "all"
        #: Pose per entity at the moment this action STARTED -- see `_capture`.
        self._baseline: dict[str, Pose] = {}
        #: When the aggregate predicate last became true, in sim time; None while it is false.
        self._since: float | None = None

    # -- to be called from a subclass's execute() -------------------------------------------------
    def start(self, entities, dwell, require) -> None:
        self._entities = [str(e) for e in (entities or [])]
        if not self._entities:
            raise ActionError(
                f"{self.__class__.__name__}: `entities` is empty. Name at least one entity; a single "
                "one is a one-element list (`entities: ['parcel']`), because OSC has no union type "
                "and an empty list literal is a grammar error.",
                action=self,
            )
        self._dwell = float(dwell)
        if self._dwell < 0.0:
            raise ActionError(
                f"{self.__class__.__name__}: `dwell` must be >= 0, got {self._dwell}.", action=self
            )
        self._require = enum_name(require)
        if self._require not in ("all", "any"):
            raise ActionError(
                f"{self.__class__.__name__}: unknown `require` {self._require!r}; use "
                "entity_quantifier!all or entity_quantifier!any.",
                action=self,
            )
        self._baseline = {}
        self._since = None

    # -- what a subclass supplies -----------------------------------------------------------------
    def measure(self, start: Pose, now: Pose) -> float:
        raise NotImplementedError

    def holds(self, measured: float) -> bool:
        raise NotImplementedError

    def render(self, measured: float) -> str:
        raise NotImplementedError

    # -- the tick -------------------------------------------------------------------------------
    def update(self) -> py_trees.common.Status:
        if not self._access.ready():
            # The stepped runner builds the world on the first reset, and the tree is set up before
            # that. Waiting is correct; asking for the world here would compile one as a side effect.
            return self.waiting("waiting for the simulation")

        try:
            poses = {}
            for name in self._entities:
                pose = self._access.entity_pose(name)
                if pose is None:
                    # Over ROS a pose is a round-trip. Not knowing YET is not the same as not being
                    # resolvable -- that one raises.
                    return self.waiting(f"waiting for {name!r}'s pose over {self.transport}")
                poses[name] = pose
        except AccessError as err:
            self.reraise(err)

        if not self._baseline:
            # Captured on the first tick where EVERY pose is known, not in setup() and not at reset:
            # "moved since this action started" is the composable meaning -- it is what makes the
            # action usable in the middle of a `serial:` and what removes any dependence on a
            # world-side plugin having captured a reference pose for us.
            self._baseline = poses

        measured = {n: self.measure(self._baseline[n], p) for n, p in poses.items()}
        hits = [n for n, m in measured.items() if self.holds(m)]
        quorum = len(self._entities) if self._require == "all" else 1
        summary = ", ".join(f"{n} {self.render(measured[n])}" for n in self._entities[:3])
        if len(self._entities) > 3:
            summary += f", +{len(self._entities) - 3} more"

        if len(hits) < quorum:
            # The dwell is over the AGGREGATE predicate and restarts whenever it stops holding: a
            # threshold CROSSING flatters the result (measured elsewhere in this repo: a parcel
            # crossed 50 mm, success was recorded, and 0.95 s later it was back on the table), and
            # per-entity dwells would let two entities each satisfy it at different times.
            self._since = None
            return self.waiting(f"{summary} ({self._require} of {len(self._entities)})")

        if self._since is None:
            self._since = self.now
        held = self.now - self._since
        if held < self._dwell:
            return self.waiting(f"{summary}; held {held:.2f} s of {self._dwell:.2f} s")
        return self.satisfied(
            summary + (f" for {held:.2f} s" if self._dwell else "") + f" ({self.transport})"
        )
