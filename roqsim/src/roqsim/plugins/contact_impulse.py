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

"""Observation plugin: how hard a watched entity was hit, integrated over the run.

The third question about one contact. :mod:`roqsim.plugins.contact_monitor` answers *did it touch*
and :mod:`roqsim.plugins.clearance_monitor` answers *how close did it come*; neither says whether
the touch was a brush or a crash, and a bit cannot be graded by severity. Beside them the substrate
has a six-axis wrench (``force_torque``), but that measures at a named site on an arm -- what is
transmitted through one cut of one kinematic chain -- so a chassis striking a wall is not a
measurement it can take.

**Why it is integrated here and not downstream.** The impulse a contact delivers is
:math:`\\int F\\,dt`, and the integrand is only fully resolved at the physics step. A free 10 kg body
meeting a wall is in contact for about 19 steps at MuJoCo's default 2 ms timestep -- under 40 ms --
so at a 30 Hz publish rate the whole collision falls in one sample, at 5 Hz it can fall between two
and be reported as nothing at all, and no trapezoid over those samples recovers the area. The
accumulation is therefore on the physics thread, every step, for the same reason
:mod:`roqsim.plugins.energy_monitor` accumulates power there: a rate-limited sample integrates a
different signal depending on who was listening.

**It counts exactly what ``contact_monitor`` counts.** The geometry rule is that plugin's own, not a
second one: every contact with exactly one side in the watched entity's kinematic subtree and
neither side in ``ignore`` / ``ignore_prefixes``. Two observables over one geometry can then never
disagree about which contacts they are describing -- one says whether it happened, the other how
hard. The one rule they do not share is ``contact_monitor``'s ``min_force``, and this plugin has no
equivalent **on purpose**: a force threshold chosen to reject numerical grazing is exactly the
calibration constant an impulse metric exists to avoid, and grazing contributes to an integral in
proportion to how weak and how brief it is -- which is to say almost nothing. Configuring one here
is refused rather than ignored, so a block copied from ``contact_monitor`` cannot leave a reader
believing a filter was applied. Where the two must agree contact for contact, run
``contact_monitor`` at ``min_force: 0``.

**What the normal force is summed over.** Within one step the normal components of all qualifying
contacts are added as magnitudes, not as vectors. A robot wedged between two walls is loaded by
both, and a vector sum would report it as touching nothing; the quantity here is the total normal
load the entity is carrying, and it never cancels.

Config::

    contact_impulse:
      # The entity watched is the one this entry is NESTED UNDER -- there is no key for it, and
      # declaring it at the top of a document is refused (`requires_owner`).
      body: ""               # base body override; default: the entity's registered base body
      namespace: ""          # transport scope for the endpoint
      ignore: [floor]        # geom NAMES that never count (default: ['floor'])
      ignore_prefixes: []    # geom name prefixes that never count (e.g. ['ground'])
      reset_on_spawn: true   # spawning the watched entity restarts the integral
      rate_hz: 30.0          # endpoint publish rate -- of the RUNNING TOTAL, not of the integrand

Endpoint ``contact_impulse`` (out) reads a :class:`ContactImpulseReport`:
``(impulse_ns, peak_normal_n, contact_time_s, normal_n, count, peak_time, peak_geom_a,
peak_geom_b)``. The three totals run from the last reset; ``normal_n`` and ``count`` are the current
step's, so the report says what it is integrating as well as what it has integrated. ``peak_time``
is the sim time of the largest single-step load (``-1.0`` if nothing was touched) and the two geom
names are that step's strongest single contact, so a severity figure is attributable rather than
merely large.

``contact_time_s`` is the time a qualifying contact **existed**, which is ``contact_monitor``'s
notion of touching and is longer than the time force was transmitted: MuJoCo goes on listing a pair
while the two geoms still overlap on the way apart, and those steps carry a zero normal force. The
impulse is unaffected -- a zero integrand adds nothing -- and the alternative, a duration that
switched off before the monitor's ``in_contact`` did, would be the second rule this plugin exists
not to have. The ROS 2 backend hint publishes ``impulse_ns`` as a ``std_msgs/Float64`` on
``contact_impulse``; a consumer that wants the rest reads the report through the blackboard handle
``contact_impulse:<address>``.

**Publishing is rate-limited and the integral is not.** ``rate_hz`` decides how often the running
total leaves the plugin, and no value of it can lose a contact: the total a slow endpoint publishes
is the same total a fast one publishes, just later. There is deliberately no ``compute_rate_hz``
knob of the kind ``clearance_monitor`` and ``contact_location`` offer -- decimating this
computation would drop the samples the integral is made of, and the number would then be a function
of the knob.

**It never ends a trial**, the line ``clearance_monitor`` and ``energy_monitor`` draw as well. What
counts as too hard is the experiment's threshold, stated in the experiment; a scenario reads the
endpoint and decides for itself.

**What a reset does.** ``on_reset`` -- ``ResetSimulation`` with ``SCOPE_STATE``, and what runs
between the trials one process serves -- zeroes all three totals. Without it the second cell of a
campaign starts with the first cell's collisions on its bill. ``reset_on_spawn`` (default true)
does the same when the watched entity gains presence, because an entity that was not in the world a
moment ago carries no history and because ``contact_monitor`` restarts there too -- one of the two
carrying a contact the other had forgotten is exactly the disagreement this plugin is shaped to
avoid. A ``SetEntityState`` is a pose and a twist, and resets nothing.

Untouched, the report reads ``impulse_ns = 0.0``, ``peak_normal_n = 0.0``, ``contact_time_s = 0.0``
and ``peak_time = -1.0`` -- a measured zero, which is what "nothing was hit" is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from ..context import Endpoint, SimContext
from ..plugin import Plugin

_log = logging.getLogger(__name__)


@dataclass
class ContactImpulseReport:
    """Neutral payload for the ``contact_impulse`` endpoint."""

    impulse_ns: float = 0.0  # integral of the summed normal force since reset [N s]
    peak_normal_n: float = 0.0  # largest summed normal force of any one step since reset [N]
    contact_time_s: float = 0.0  # sim time spent with at least one qualifying contact [s]
    normal_n: float = 0.0  # summed normal force this step [N] -- the integrand right now
    count: int = 0  # qualifying contacts this step
    peak_time: float = -1.0  # sim time of the peak; -1.0 while nothing has been touched
    peak_geom_a: str = ""  # strongest single contact at the peak step ("" until one happens)
    peak_geom_b: str = ""


class ContactImpulsePlugin(Plugin):
    """See the module docstring."""

    parallel_safe = False  # post_step accumulates state

    #: It watches an ENTITY, so it must be nested under the entry that provides one -- the same
    #: reason ``contact_monitor`` declares it: at the top of a document ``self.entity`` is None and
    #: the base body would fall back to a bare "base_link", resolving by accident for a robot that
    #: happens to use that name and failing obscurely for one that does not.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.body = self.config.get("body", "")
        self.ignore = list(self.config.get("ignore", ["floor"]))
        self.ignore_prefixes = list(self.config.get("ignore_prefixes", []))
        self.reset_on_spawn = bool(self.config.get("reset_on_spawn", True))
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self._ctx: SimContext | None = None
        self._watched: np.ndarray | None = None  # per-geom mask; see configure()
        self._ignored: np.ndarray | None = None
        # Reused across steps: mj_contactForce writes into it, and a 6-vector allocated per contact
        # per step is a cost this plugin adds to every step of every run that carries it.
        self._force_scratch = np.zeros(6)
        self._report = ContactImpulseReport()
        self._entity = None
        self._was_present = True
        self._last_time = 0.0

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if "min_force" in config:
            # Refused rather than ignored: a block copied from contact_monitor would otherwise read
            # as though grazing had been filtered out of the integral, and the number would be
            # quoted as if it had.
            errors.append(
                "'min_force' is not a key of contact_impulse: a force threshold is the "
                "calibration constant an impulse metric exists to avoid, and a weak brief "
                "contact contributes to the integral in proportion to how weak and how brief it "
                "is. Filter downstream on the impulse itself, or use contact_monitor for a verdict."
            )
        for key in ("ignore", "ignore_prefixes"):
            if key in config and not isinstance(config[key], list):
                errors.append(f"'{key}' must be a list of strings")
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        model = ctx.model
        entity = ctx.entities.get(self.robot)
        self._entity = entity
        self._was_present = bool(getattr(entity, "present", True)) if entity else True
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        body_name = (
            (prefix + self.body)
            if self.body
            else (entity.body if entity and entity.body else prefix + "base_link")
        )
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if root < 0:
            # Fail loudly: a meter watching nothing reports a zero impulse forever, which reads as
            # a trial that touched nothing gently and would be averaged in as one.
            raise RuntimeError(f"contact_impulse: base body {body_name!r} not found")

        # Boolean masks indexed by geom id, so the per-step filter is one vectorised lookup over
        # the contact array. A world's contacts are dominated by pairs the watched entity is not in
        # -- props on the floor, a crowd's feet -- and touching each of them from Python costs more
        # than the physics step that produced them.
        self._watched = np.zeros(model.ngeom, dtype=bool)
        for gid in range(model.ngeom):
            if self._in_subtree(model, int(model.geom_bodyid[gid]), root):
                self._watched[gid] = True
        if not self._watched.any():
            raise RuntimeError(
                f"contact_impulse: body {body_name!r} and its subtree carry no geoms to watch"
            )

        self._ignored = np.zeros(model.ngeom, dtype=bool)
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name in self.ignore or any(name.startswith(p) for p in self.ignore_prefixes):
                self._ignored[gid] = True
        missing = [
            n for n in self.ignore if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) < 0
        ]
        if missing:
            # Not fatal (a world may legitimately have no `floor` geom), but never silent: an
            # unmatched ignore entry is how the weight a robot rests on the ground with becomes
            # the largest impulse of the trial.
            _log.warning(
                "contact_impulse: ignore entry has no matching geom: %s", ", ".join(missing)
            )

        # `read` rather than the report, because post_step REPLACES it each step -- a consumer
        # holding the dataclass would read one frozen step forever. Keyed on the ADDRESS, since
        # `self.name` falls back to the class name and two unnamed instances in one world would
        # write to a single key, reporting one robot's impulse as another's.
        ctx.blackboard.set(f"contact_impulse:{self.address}", self.read)
        ctx.interface.add(
            Endpoint(
                name="contact_impulse",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Float64",
                        # The report is a structure and Float64 carries one field, so the endpoint
                        # says WHICH rather than the bridge holding a converter that knows this
                        # plugin's attribute names. The other fields stay readable in-process.
                        "field": "impulse_ns",
                        "topic": self.topic_override("contact_impulse") or "contact_impulse",
                    }
                },
            )
        )
        _log.info(
            "contact_impulse: watching %d geoms of %r, ignoring %d",
            int(self._watched.sum()),
            body_name,
            int(self._ignored.sum()),
        )

    def read(self) -> ContactImpulseReport:
        """The report as it stands. What the blackboard handle hands an in-process consumer."""
        return self._report

    @staticmethod
    def _in_subtree(model, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return body == root

    def on_reset(self, ctx: SimContext) -> None:
        self._report = ContactImpulseReport()
        self._was_present = bool(getattr(self._entity, "present", True)) if self._entity else True
        self._last_time = ctx.sim_time

    def _became_present(self) -> bool:
        """Has the watched entity been SPAWNED since the last step -- gained presence?"""
        if self._entity is None:
            return False
        now = bool(getattr(self._entity, "present", True))
        appeared = now and not self._was_present
        self._was_present = now
        return appeared and self.reset_on_spawn

    # -- the integral --------------------------------------------------------------------------
    def post_step(self, ctx: SimContext) -> None:
        if self._became_present():
            # A re-spawned entity carries no history, and contact_monitor restarts here too: one
            # of the two keeping a contact the other has forgotten is the disagreement this plugin
            # is shaped to avoid.
            self._report = ContactImpulseReport()

        dt = ctx.sim_time - self._last_time
        self._last_time = ctx.sim_time
        data, model = ctx.data, ctx.model

        n = data.ncon
        total = 0.0
        strongest = 0.0
        geoms = ("", "")
        hits = 0
        if n:
            con = data.contact
            g1, g2 = con.geom1[:n], con.geom2[:n]
            # Exactly one side watched: neither is nothing to do with this entity, both is a
            # self-contact. The same predicate contact_monitor applies, so the two plugins cannot
            # count different things.
            keep = (self._watched[g1] ^ self._watched[g2]) & ~(
                self._ignored[g1] | self._ignored[g2]
            )
            for i in np.flatnonzero(keep):
                # Only for the handful that survived the filter: this is a C call per contact.
                mujoco.mj_contactForce(model, data, int(i), self._force_scratch)
                normal = abs(float(self._force_scratch[0]))
                total += normal
                hits += 1
                if normal > strongest:
                    strongest = normal
                    geoms = (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g1[i]))
                        or f"geom{int(g1[i])}",
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g2[i]))
                        or f"geom{int(g2[i])}",
                    )

        report = self._report
        impulse = report.impulse_ns
        contact_time = report.contact_time_s
        if dt > 0.0:
            # Against elapsed SIM time rather than a fixed timestep, so a replay over recorded
            # samples accumulates the same way at its own spacing (see roqsim.recording).
            impulse += total * dt
            if hits:
                contact_time += dt

        peak, peak_time = report.peak_normal_n, report.peak_time
        peak_a, peak_b = report.peak_geom_a, report.peak_geom_b
        if hits and total > peak:
            peak, peak_time = total, float(data.time)
            peak_a, peak_b = geoms

        self._report = ContactImpulseReport(
            impulse_ns=impulse,
            peak_normal_n=peak,
            contact_time_s=contact_time,
            normal_n=total,
            count=hits,
            peak_time=peak_time,
            peak_geom_a=peak_a,
            peak_geom_b=peak_b,
        )
