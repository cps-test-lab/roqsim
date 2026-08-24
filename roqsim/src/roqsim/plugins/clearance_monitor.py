# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation plugin: how close an entity came to something it should not touch.

The companion to :mod:`roqsim.plugins.contact_monitor`, and deliberately its opposite
number. That plugin answers *did it touch* -- the right failure criterion, and a poor thing
to optimise, because a bit gives every non-touching configuration the same score and leaves
a search nothing to climb. This one answers *how close did it come*, continuously, over the
same geometry. A verdict and a gradient, one owner each.

**Why the simulator and not an evaluator afterwards.** A clearance derived from recorded
poses is worse in three ways that are not fixable downstream:

* The closest approach happens *between* pose samples. A trial recording at 30 Hz while a
  robot passes at 0.3 m/s misses a centimetre of approach between frames, and the faster
  the pass the more it misses -- so the metric is least accurate exactly where it matters.
* It has to model footprints. A radius around a base is a calibration constant sitting
  between the geometry and the result, which is the objection ``contact_monitor``'s
  docstring raises against proximity proxies. ``mj_geomDistance`` measures the real shapes.
* An articulated obstacle's nearest part is a limb, not its origin. A walker reduced to a
  point with a radius cannot report the arm that actually came close.

**It never ends a trial.** ``contact_monitor`` latches and is what a scenario fails on;
this observes and does not. Two plugins reporting the same failure by different rules is
how a trial starts disagreeing with itself about whether it failed -- and a clearance
threshold is precisely the tunable number the contact oracle exists to avoid. A scenario
that *wants* to stop on a near-miss can still read this endpoint and decide for itself; the
difference is that the threshold is then the experiment's, stated in the experiment.

Config::

    clearance_monitor:
      body: ""               # base body override; default: the entity's registered base
      ignore: [floor]        # geom NAMES that never count (default: ['floor'])
      ignore_prefixes: []    # geom name prefixes that never count
      distmax: 5.0           # [m] cutoff: beyond this the distance is not computed
      compute_rate_hz: 200.0 # how often the distance is MEASURED
      rate_hz: 30.0          # how often the endpoint is PUBLISHED

Endpoint ``clearance`` (out) reads a :class:`ClearanceReport`:
``(current, minimum, at_time, geom, saturated)`` -- ``minimum`` is the closest approach
since the last reset and ``geom`` names what it was to, so a near-miss is attributable
rather than merely flagged.

``compute_rate_hz`` is separate from ``rate_hz`` because they answer different questions.
Publishing is cheap; measuring is a distance query per (watched geom, candidate geom) pair,
which on a nav world costs about a quarter of the step budget if done every physics step --
against a budget the simulator may already be over. Poses reach a consumer at ~30 Hz, so
measuring at 200 Hz resolves ~1.5 mm at walking pace: far finer than anything downstream can
use, at a fraction of the cost. Raise it if a fast pass matters more than throughput; the
trade is stated here rather than hidden in a default.

``distmax`` is the other performance knob and it is a real one: the cost is a distance query
per (watched geom, candidate geom) pair, and the cutoff lets MuJoCo reject far pairs
cheaply. Beyond it the report reads ``current == distmax`` with ``saturated`` true, which
says "at least this far" rather than offering a number that looks measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco

from ..context import Endpoint, SimContext
from ..plugin import Plugin

_log = logging.getLogger(__name__)


@dataclass
class ClearanceReport:
    """Neutral payload for the ``clearance`` endpoint."""

    current: float  # distance now [m]; <= 0 while overlapping
    minimum: float  # closest approach since the last reset
    at_time: float  # sim time of that closest approach; -1.0 before the first step
    geom: str  # what the closest approach was to ("" until measured)
    saturated: bool  # current is the distmax cutoff, not a measured distance


class ClearanceMonitorPlugin(Plugin):
    """Minimum distance from a watched subtree to everything it is not allowed to hit."""

    parallel_safe = True  # post_step reads geometry and writes only its own state
    requires_owner = True  # it watches an entity, so it must be nested under one

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.body = self.config.get("body", "")
        self.ignore = list(self.config.get("ignore", ["floor"]))
        self.ignore_prefixes = list(self.config.get("ignore_prefixes", []))
        self.distmax = float(self.config.get("distmax", 5.0))
        self.compute_rate_hz = float(self.config.get("compute_rate_hz", 200.0))
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self._next_due = 0.0
        self._ctx: SimContext | None = None
        self._watched: list[int] = []
        self._candidates: list[int] = []
        self._report = ClearanceReport(float("inf"), float("inf"), -1.0, "", False)

    # -- validation --------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("distmax", 5.0)) <= 0:
            errors.append("'distmax' must be > 0")
        if float(config.get("compute_rate_hz", 200.0)) <= 0:
            errors.append("'compute_rate_hz' must be > 0")
        for key in ("ignore", "ignore_prefixes"):
            if key in config and not isinstance(config[key], list):
                errors.append(f"'{key}' must be a list of strings")
        return errors

    # -- lifecycle ---------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        model = ctx.model
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        body_name = (
            (prefix + self.body)
            if self.body
            else (entity.body if entity and entity.body else prefix + "base_link")
        )
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if root < 0:
            # Loudly, for the same reason contact_monitor refuses: a monitor watching nothing
            # reports "never close to anything" forever, which is indistinguishable from a
            # clean run and would quietly pass every trial in a campaign.
            raise RuntimeError(f"clearance_monitor: base body {body_name!r} not found")

        subtree = [
            gid
            for gid in range(model.ngeom)
            if self._in_subtree(model, int(model.geom_bodyid[gid]), root)
        ]
        # Only geoms that can actually collide. `mj_geomDistance` is pure geometry and
        # ignores collision masks, so without this a render-only geom would be reported as
        # clearance to something the robot passes straight through -- and articulated
        # obstacles carry several (a pedestrian model here has six visual-only geoms beside
        # fifteen solid ones). The same rule applies on the watched side: a decorative geom
        # sticking out of a robot is not what would touch anything.
        self._watched = [gid for gid in subtree if self._collidable(model, gid)]
        if not self._watched:
            raise RuntimeError(
                f"clearance_monitor: body {body_name!r} and its subtree carry no collidable "
                f"geoms to watch ({len(subtree)} geom(s), all visual-only)"
            )

        watched = set(self._watched)
        ignored = set()
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name in self.ignore or any(name.startswith(p) for p in self.ignore_prefixes):
                ignored.add(gid)
        self._candidates = [
            gid
            for gid in range(model.ngeom)
            if gid not in watched
            and gid not in ignored
            # Could this geom contact any watched geom? MuJoCo pairs on
            # (contype1 & conaffinity2) | (contype2 & conaffinity1); a candidate that
            # matches no watched geom cannot be hit and is not clearance.
            and any(self._can_pair(model, gid, w) for w in self._watched)
        ]
        if not self._candidates:
            raise RuntimeError(
                "clearance_monitor: nothing left to measure against -- every geom outside "
                f"{body_name!r} is ignored, so this would report a constant distance forever"
            )
        missing = [
            n for n in self.ignore if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) < 0
        ]
        if missing:
            _log.warning("clearance_monitor: ignore names match no geom: %s", ", ".join(missing))

        # Keyed on the ADDRESS, so a world with two monitored entities gives two distinct
        # handles ("robot.clearance_monitor", "forklift.clearance_monitor") rather than
        # colliding on a class name.
        ctx.blackboard.set(f"clearance:{self.address}", self.read_state)
        ctx.interface.add(
            Endpoint(
                name="clearance",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._report,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        # Float32 and the CURRENT distance: a running minimum is derivable
                        # from a recorded series, while the series is not derivable from a
                        # minimum. Publishing the reducible half would throw away the shape
                        # of the approach, which is what a reader wants to see.
                        "type": "std_msgs.msg.Float32",
                        "field": "current",
                        "topic": self.topic_override("clearance") or "clearance",
                    }
                },
            )
        )
        _log.info(
            "clearance_monitor: watching %d geom(s) of %r against %d candidate(s), distmax %.2f m",
            len(self._watched),
            body_name,
            len(self._candidates),
            self.distmax,
        )

    def read_state(self) -> ClearanceReport:
        """The latest report; the blackboard handle hands this to an in-process consumer.

        A callable rather than the report, because ``post_step`` REPLACES it each step and a
        consumer holding the object would keep reading the first one it saw.
        """
        return self._report

    def on_reset(self, ctx: SimContext) -> None:
        # A trial's clearance is that trial's. Carrying the minimum across a reset would
        # report the previous run's near-miss as this one's -- and on a packed job, where one
        # process serves several trials, every trial after the first would inherit it.
        self._report = ClearanceReport(float("inf"), float("inf"), -1.0, "", False)
        # A reset is a new trial and therefore a new schedule; leaving the old due time in
        # place would skip the first measurement of the run by however far the clock moved.
        self._next_due = 0.0

    def post_step(self, ctx: SimContext) -> None:
        model, data = ctx.model, ctx.data
        if float(data.time) < self._next_due:
            return
        self._next_due = float(data.time) + 1.0 / self.compute_rate_hz
        best = self.distmax
        best_geom = -1
        for watched in self._watched:
            for candidate in self._candidates:
                if not self._can_pair(model, watched, candidate):
                    continue
                distance = mujoco.mj_geomDistance(model, data, watched, candidate, best, None)
                # MuJoCo returns distmax when the pair is farther than the cutoff it was
                # given; passing `best` narrows the cutoff as the scan improves, so later
                # pairs are rejected sooner.
                if distance < best:
                    best, best_geom = distance, candidate

        saturated = best_geom < 0
        name = (
            ""
            if saturated
            else (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, best_geom) or f"geom{best_geom}"
            )
        )
        report = self._report
        if best < report.minimum:
            self._report = ClearanceReport(best, best, float(data.time), name, saturated)
        else:
            self._report = ClearanceReport(
                best, report.minimum, report.at_time, report.geom, saturated
            )

    @staticmethod
    def _collidable(model, gid: int) -> bool:
        """Whether a geom participates in contacts at all (not render-only)."""
        return bool(int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))

    @staticmethod
    def _can_pair(model, a: int, b: int) -> bool:
        """MuJoCo's own pairing rule -- the test for "could these two ever touch"."""
        return bool(
            (int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
            or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))
        )

    @staticmethod
    def _in_subtree(model, body_id: int, root: int) -> bool:
        while body_id > 0:
            if body_id == root:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == root
