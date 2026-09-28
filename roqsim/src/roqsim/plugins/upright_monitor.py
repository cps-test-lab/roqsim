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

"""Observation plugin: did a body a trial drives in the plane stop being in the plane?

A trial that pushes something around a floor -- a pedestrian, an opponent robot, a cart --
assumes throughout that the thing is still standing on that floor. Nothing enforced it, and
when the assumption broke the run did not fail: it carried on producing positions, distances
and clearances about a body that was lying on its side, or airborne, or halfway through the
ground plane. The numbers stay plausible, which is what makes this worth a plugin rather than
a note. A trial nobody flagged is a trial someone will average.

The mechanism it catches most often is a drive force applied at a tall body's centre of mass
while friction holds its base: the pair is a couple, the body tips, and the tipping is correct
physics about a model that was wrong. Which body, and why it was wrong, is the experiment's to
fix -- :mod:`roqsim_walker`'s mocap pedestrian is the answer for pedestrians, and a low centre
of mass or a planar joint for anything hand-rolled. What the substrate owes is that nobody
finds out afterwards.

**It measures the pose, not the mechanism**, which is what makes it worth having on a body no
solver integrates. A mocap body cannot topple -- there are no degrees of freedom to topple with
-- so on one that nothing mishandles this stays quiet for the length of the run. But its pose is
written by something, and what is written can be wrong: a gait, a navigation output or a scenario
placing an entity can put a body through the floor or lay it flat, and ``xpos``/``xmat`` report
that exactly as they report a fall. So a driven prop and a walker are covered too, against the
failure that is actually available to them.

Nothing about a body's ``motion:`` is consulted, and no plugin that moves things needs to know
this exists. That is deliberate: it is why one monitor serves a free body, a mocap prop and a
walker, and why the packages that drive them need no dependency on it.

Config::

    upright_monitor:
      # The entity watched is the one this entry is NESTED UNDER -- there is no key for it, and
      # declaring it at the top of a document is refused (`requires_owner`).
      body: ""               # base body override; default: the entity's registered base body
      max_tilt_deg: 30.0     # degrees the body's own +z may lean from world +z
      max_rise_m: 0.10       # metres its height may depart from where it settled, either way
      settle_s: 0.5          # sim time to let the body come to rest before judging it, and the
                             #   moment its reference height is taken
      namespace: ""          # transport scope for the endpoint
      latch: true            # once fallen, stay fallen until on_reset (a trial is failed, not
                             #   un-failed -- the same rule contact_monitor follows)
      rate_hz: 30.0          # endpoint publish rate

Endpoint ``upright`` (out) reads an :class:`UprightReport`. The ROS 2 backend hint publishes
``upright`` as a ``std_msgs/Bool`` on ``upright`` (relative, so two namespaced entities get
``/a/upright`` and ``/b/upright``); a consumer wanting the detail reads the fields.

**The reference height is where the body SETTLED, not where it was spawned**, which is what
``settle_s`` buys. A body spawned twenty centimetres above the floor drops onto it, and measuring
against the spawn would call that drop a departure -- flagging every world whose author did not
place a body to the millimetre, which is a monitor people turn off. Nothing is judged and no
reference is taken until ``settle_s``; it is measured per episode, so a repetition compares
against its own start.

The cost is that a body already broken at t=0 is not reported for half a second. That is the right
trade: the verdict latches, so it is reported a moment later rather than not at all, whereas a
false positive on a correct world is reported forever. Set ``settle_s: 0`` where a trial starts in
contact and the first instant matters.

**Both thresholds are departures, not limits.** ``max_rise_m`` is symmetric: a body sinking through
the floor has left the plane exactly as much as one taking off, and a run where the ground gave way
is no more usable than one where the pedestrian flew. ``max_tilt_deg`` is the angle between the
body's own +z and the world's, so it says nothing about yaw -- a body turning on the spot is doing
what a planar trial expects.

**Not a failure criterion for a robot that is meant to tip.** A quadruped mid-gait, an aerial
vehicle banking, an arm's wrist -- all of these leave the plane on purpose. This watches an entity
because somebody nested it under one; nothing infers that an entity should be upright.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..context import Endpoint, SimContext
from ..plugin import Plugin

_log = logging.getLogger(__name__)


@dataclass
class UprightReport:
    """Neutral payload for the ``upright`` endpoint.

    ``upright`` is the verdict and ``worst_tilt_deg`` / ``worst_rise_m`` are the gradient beside
    it, carried here rather than in a second plugin because both come out of one pose read: the
    split ``contact_monitor`` and ``clearance_monitor`` make exists because a clearance costs a
    geometry query, and this costs nothing.

    ``first_time`` is the simulation time at which the body first left the plane since reset
    (``-1.0`` while it has not), and ``reason`` is ``"tilt"`` or ``"height"`` -- which of the two
    it was, because they point at different mistakes: a tilt is usually where the drive is
    applied, a height is usually contact or a joint that does not constrain what it looks like it
    constrains.
    """

    upright: bool
    first_time: float
    tilt_deg: float
    rise_m: float
    worst_tilt_deg: float
    worst_rise_m: float
    reason: str


class UprightMonitorPlugin(Plugin):
    parallel_safe = True  # post_step reads xmat/xpos and writes its own state

    # It watches an ENTITY, so it must be nested under the entry that provides one -- the same
    # reason contact_monitor declares it: at the top of a document `self.entity` is None and the
    # base body would resolve by accident or not at all.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.watched = self.entity
        self.body = self.config.get("body", "")
        self.max_tilt_deg = float(self.config.get("max_tilt_deg", 30.0))
        self.max_rise_m = float(self.config.get("max_rise_m", 0.10))
        self.settle_s = float(self.config.get("settle_s", 0.5))
        self.latch = bool(self.config.get("latch", True))
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self._ctx: SimContext | None = None
        self._bid = -1
        #: Height the body settled at, captured on the first step at or after ``settle_s``.
        #: ``None`` means "not yet captured", which is distinct from 0.0 -- a body legitimately
        #: settling at z=0 would otherwise re-capture its reference every step and never report
        #: a departure.
        self._reference_z: float | None = None
        self._report = _clean()

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if not 0.0 < float(config.get("max_tilt_deg", 30.0)) <= 180.0:
            errors.append("'max_tilt_deg' must be in (0, 180]")
        if float(config.get("settle_s", 0.5)) < 0:
            errors.append("'settle_s' must be >= 0: it is a sim time, not an offset")
        if float(config.get("max_rise_m", 0.10)) <= 0:
            errors.append(
                "'max_rise_m' must be > 0: it is how far the body may depart from where it "
                "settled, in either direction, not a height limit"
            )
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        entity = ctx.entities.get(self.watched)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        body_name = (prefix + self.body) if self.body else (entity.body if entity else "")
        self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self._bid < 0:
            raise RuntimeError(
                f"upright_monitor[{self.label}]: body {body_name!r} was not found. It defaults to "
                f"the entity's registered base body; name another with `body:` if this entity's "
                f"uprightness is carried by a different one."
            )

        # Keyed on the address, like contact_monitor's: two monitors in one world writing to one
        # key would report the second entity's verdict under the first entity's name.
        ctx.blackboard.set(f"upright:{self.address}", self.read_state)

        ctx.interface.add(
            Endpoint(
                name="upright",
                direction="out",
                owner=self.watched,
                namespace=ns,
                read=lambda: self._report,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Bool",
                        # The report is a structure and Bool carries one field, so the endpoint
                        # says WHICH -- the same convention contact_monitor uses.
                        "field": "upright",
                        "topic": self.topic_override("upright") or "upright",
                    }
                },
            )
        )
        _log.info(
            "upright_monitor: watching %r (tilt <= %.1f deg, height within %.3f m)",
            body_name,
            self.max_tilt_deg,
            self.max_rise_m,
        )

    def read_state(self) -> UprightReport:
        """The latest report. A callable, because ``post_step`` REPLACES the report each step."""
        return self._report

    def on_reset(self, ctx: SimContext) -> None:
        self._reference_z = None
        self._report = _clean()

    def post_step(self, ctx: SimContext) -> None:
        data = ctx.data
        if float(data.time) < self.settle_s:
            # Not yet: a body still dropping onto the floor has not left a plane it has not
            # reached. Nothing is measured either -- the worst-so-far would otherwise carry the
            # settling transient into a metric about the trial.
            return
        z = float(data.xpos[self._bid][2])
        if self._reference_z is None:
            self._reference_z = z

        # Third column of the rotation matrix IS the body's own +z in world coordinates, so the
        # dot with world +z is that column's z component -- no trigonometry on Euler angles, which
        # would have to pick a convention and would be singular somewhere.
        up_z = float(np.array(data.xmat[self._bid]).reshape(3, 3)[2, 2])
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
        rise_m = z - self._reference_z

        worst_tilt = max(self._report.worst_tilt_deg, tilt_deg)
        # Signed, and compared by magnitude: the worst departure is the biggest one either
        # way, and reporting its absolute value would lose which way a body went -- through
        # the floor and off it are different mistakes.
        worst_rise = max(self._report.worst_rise_m, rise_m, key=abs)

        reason = ""
        if tilt_deg > self.max_tilt_deg:
            reason = "tilt"
        elif abs(rise_m) > self.max_rise_m:
            reason = "height"

        if reason and self._report.first_time < 0.0:
            _log.info(
                "upright_monitor: %r left the plane (%s: tilt %.1f deg, height %+.3f m) at "
                "t=%.3f s",
                self.watched,
                reason,
                tilt_deg,
                rise_m,
                data.time,
            )
            self._report = UprightReport(
                False, float(data.time), tilt_deg, rise_m, worst_tilt, worst_rise, reason
            )
            return

        fallen = bool(reason) or (self.latch and self._report.first_time >= 0.0)
        self._report = UprightReport(
            not fallen,
            self._report.first_time,
            tilt_deg,
            rise_m,
            worst_tilt,
            worst_rise,
            reason or self._report.reason,
        )


def _clean() -> UprightReport:
    """The report before anything has been measured: upright, and nothing seen yet."""
    return UprightReport(True, -1.0, 0.0, 0.0, 0.0, 0.0, "")
