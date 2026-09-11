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

"""Observation plugin: stop when a measured wrench exceeds what the task allows.

Every arm family has this and each calls it something else -- a protective stop, a safety stop, a
collision reflex with per-axis thresholds. The capability is the same one and it is named for the
capability here, with the robot's own word for it supplied by ``reports_as``, the way controller
names come from a manifest rather than from the substrate.

**It is not a controller.** On a real arm this comes from the controller box and is surfaced
through the vendor's status interface, not through the controller manager. Simulating it as "the
controller deactivated itself" would be a fiction on the wrong service, and a results table's
failure-mode column would not mean what it says. So it reports a safety state and stops the run,
and deactivating controllers is an EFFECT of tripping rather than the mechanism.

It is also not part of ``arm_controller``: that plugin is the single writer of the arm's actuators,
and a limit is a property of the task rather than of the mechanism that drives the joints. Two
experiments on one arm disagree about what "too hard" is; neither disagrees about how a joint is
commanded.

Config -- a component of the entity whose wrench is watched::

    force_limit:
      ft: ft                   # blackboard key suffix of the force_torque sensor (`ft:<key>`)
      max_force: 40.0          # N; magnitude of the measured force, 0 disables
      max_torque: 0.0          # Nm; magnitude of the measured torque, 0 disables
      settle_s: 0.0            # ignore the first seconds, while a reset transient decays
      latch: true              # once tripped, stay tripped until reset (a trial is failed, not un-failed)
      stop_run: true           # ask the driver to end the run, as a real stop ends the motion
      release_controllers: true  # deactivate the controllers driving the arm, the way a stop does
      reports_as: protective_stop  # the word this robot's own interface uses
      rate_hz: 30.0

Endpoint ``force_limit`` (out) reads a :class:`LimitReport`:
``(tripped, reason, at_time, force, torque)`` -- ``at_time`` is the simulation time of the first
trip (``-1.0`` if none) and ``force``/``torque`` are the magnitudes that caused it, so a failure is
attributable rather than merely flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.controllers import registry_for
from roqsim.plugin import Plugin


@dataclass
class LimitReport:
    """What the monitor saw. ``reason`` is empty until it trips."""

    tripped: bool = False
    reason: str = ""
    at_time: float = -1.0
    force: float = 0.0
    torque: float = 0.0


class ForceLimitPlugin(Plugin):
    #: Watches one entity's sensor, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.watched = self.entity
        self.ft_key = self.config.get("ft", "ft")
        self.max_force = float(self.config.get("max_force", 0.0))
        self.max_torque = float(self.config.get("max_torque", 0.0))
        self.settle_s = float(self.config.get("settle_s", 0.0))
        self.latch = bool(self.config.get("latch", True))
        self.stop_run = bool(self.config.get("stop_run", True))
        self.release_controllers = bool(self.config.get("release_controllers", True))
        self.reports_as = str(self.config.get("reports_as", "protective_stop"))
        self.rate_hz = float(self.config.get("rate_hz", 30.0))

        self._ctx: SimContext | None = None
        self._ft = None
        self._report = LimitReport()

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("max_force", 0.0)) < 0 or float(config.get("max_torque", 0.0)) < 0:
            errors.append("'max_force' and 'max_torque' must be >= 0 (0 disables that axis)")
        if not float(config.get("max_force", 0.0)) and not float(config.get("max_torque", 0.0)):
            # A monitor watching nothing reports "nothing exceeded" forever, which reads in the
            # results exactly like a trial that stayed within its limits.
            errors.append(
                "one of 'max_force' or 'max_torque' must be set: a limit of zero on both watches "
                "nothing, and a trial that could never trip is indistinguishable from a safe one"
            )
        if float(config.get("settle_s", 0.0)) < 0:
            errors.append("'settle_s' must be >= 0")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        entity = ctx.entities.get(self.watched)
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        self._ft = ctx.blackboard.get(f"ft:{self.ft_key}")
        if self._ft is None:
            raise RuntimeError(
                f"force_limit[{self.label}]: no force_torque sensor at 'ft:{self.ft_key}'. This "
                f"watches a MEASURED wrench, the same one the task's controller closes its loop "
                f"around -- add a `force_torque` plugin (its `name` is the key) before this one."
            )

        ctx.blackboard.set(f"force_limit:{self.address}", self.read_state)
        ctx.interface.add(
            Endpoint(
                name="force_limit",
                direction="out",
                owner=self.watched,
                namespace=ns,
                read=lambda: self._report,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Bool",
                        "field": "tripped",
                        "topic": self.topic_override("force_limit") or "force_limit",
                    }
                },
            )
        )

    def read_state(self) -> LimitReport:
        return self._report

    def on_reset(self, ctx: SimContext) -> None:
        # A trip carried into the next episode is a measurement of the previous one, and
        # repetitions of a trial would not be repetitions.
        self._report = LimitReport()

    def post_step(self, ctx: SimContext) -> None:
        if self._report.tripped and self.latch:
            return
        if ctx.sim_time < self.settle_s:
            return

        force, torque = self._ft.read()
        f, t = float(np.linalg.norm(force)), float(np.linalg.norm(torque))
        self._report.force, self._report.torque = f, t

        reason = ""
        if self.max_force and f > self.max_force:
            reason = f"{self.reports_as}: force {f:.1f} N exceeds {self.max_force:.1f} N"
        elif self.max_torque and t > self.max_torque:
            reason = f"{self.reports_as}: torque {t:.2f} Nm exceeds {self.max_torque:.2f} Nm"
        if not reason:
            if not self.latch:
                self._report.tripped, self._report.reason = False, ""
            return

        first = not self._report.tripped
        self._report.tripped, self._report.reason = True, reason
        if first:
            self._report.at_time = float(ctx.sim_time)
            if self.release_controllers:
                self._release(ctx)
            if self.stop_run:
                ctx.request_stop(reason)

    def _release(self, ctx: SimContext) -> None:
        """Deactivate whatever is driving this robot, which is what a stop does to the motion.

        An EFFECT of the stop and not the mechanism: the controllers stay loaded, exactly as they
        do on the real arm, and what a results table records is the safety state rather than a
        switch nobody asked for.
        """
        registry = registry_for(ctx)
        entity = ctx.entities.get(self.watched)
        ns = (entity.meta.get("namespace", "") if entity else "") or ""
        driving = [c.name for c in registry.all(ns) if c.claimed_interfaces]
        if driving:
            registry.switch(deactivate=driving, namespace=ns, sim_time=ctx.sim_time)
