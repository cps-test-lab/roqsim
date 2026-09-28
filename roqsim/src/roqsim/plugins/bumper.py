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

"""Observation plugin: a bumper -- which ZONE of a body is being pushed, every step.

A bumper on a real robot is a shell with a few switches behind it: it does not report a position,
it reports which of its zones is depressed. That is the observable a base's safety stack consumes
(a Create 3 stops on ``bump_front_center``, a vacuum turns away from ``bump_left``), and it is what
this plugin produces from the contacts MuJoCo already solves.

The sibling of :mod:`roqsim.plugins.contact_location`, and separate for the same reason that one
is separate from ``contact_monitor``: it is a different question. ``contact_location`` answers
"where, as a point in my frame" for a tactile controller; this answers "which switch" for a stack
that has a switch name per zone and nothing else. Both read the same contacts and neither latches.

A zone is a **bearing sector** of the watched body's own frame: the bearing of a contact's position
from the body's origin, in the body's horizontal plane, decides the zone. That is the rule the
Create 3's simulator uses to zone a bumper, and it is the right one for any shell that wraps a base:
a switch on a bumper shell is, physically, a range of bearings. A shell that is not radially
arranged (a bumper bar on a flat front) declares one zone spanning its width and gets a single
switch, which is what such a bar is.

Config::

    bumper:
      # The entity watched is the one this entry is NESTED UNDER -- there is no key for it, and
      # declaring it at the top of a document is refused (`requires_owner`).
      body: ""               # base body override; default: the entity's registered base body
      namespace: ""          # transport scope for the endpoints
      zones:                 # bearing sectors, radians in the base frame, counter-clockwise from
        front: [-0.52, 0.52] #   +x: {<zone>: [from, to]}. A sector with from > to wraps through
        left: [0.52, 1.57]   #   +/-pi, so a rear zone is `[2.6, -2.6]`.
      geoms: []              # geom NAMES that ARE the bumper (default: the entity's whole subtree,
                             #   like contact_monitor, its flexes included). A real bumper is one
                             #   shell; a contact on the chassis roof presses no switch, so a model
                             #   that names its shell geoms lists them here.
      geom_prefixes: []      # geom name prefixes that are the bumper
      ignore: [floor]        # geom or flex NAMES that never count (default: ['floor'])
      ignore_prefixes: []    # geom or flex name prefixes that never count (e.g. ['ground'])
      min_force: 1.0         # N; contacts below this normal force are ignored (numerical grazing)
      rate_hz: 62.0          # endpoint publish rate

One ``out`` endpoint per zone, named ``bumper/<zone>``, reads a ``bool``: is that zone pressed this
step. The ROS 2 backend hint publishes each as a ``std_msgs/Bool`` on ``bumper/<zone>`` (relative,
so it is scoped by the entity's namespace). A stack that wants a vendor's message assembles it from
these in its own adapter node -- a bumper switch is a bool on every robot, and the vendor's
envelope around it is the stack's business, not the simulator's.

**Read it through the blackboard, not the endpoint, inside a control loop.**
``ctx.blackboard.get(f"bumper:{address}")`` returns a callable giving the current
:class:`BumperReading`, the same convention ``contact_monitor`` and ``contact_location`` use.

Not latched, on purpose: a bumper releases when the robot backs off, and a stack that needs "it
bumped at some point" reads ``contact_monitor``. Where a contact falls into no declared zone
(behind a robot with a front bumper only), nothing is pressed -- exactly as a shell that is not
there reports nothing, and as ``contact_monitor`` still reports the collision.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from ..contact_scope import ContactScope, resolve_contact_scope
from ..context import Endpoint, SimContext
from ..plugin import Plugin

_log = logging.getLogger(__name__)


@dataclass
class BumperReading:
    """Neutral payload behind the blackboard handle: every zone, this step."""

    pressed: dict[str, bool]  # zone -> pressed
    any_pressed: bool
    time: float  # sim time of this reading


def _in_sector(bearing: float, lo: float, hi: float) -> bool:
    """Is *bearing* (radians, in (-pi, pi]) inside the sector from *lo* to *hi* counter-clockwise?

    A sector with ``lo > hi`` wraps through +/-pi, which is how a rear zone is spelled.
    """
    if lo <= hi:
        return lo <= bearing <= hi
    return bearing >= lo or bearing <= hi


class BumperPlugin(Plugin):
    parallel_safe = True  # post_step only reads data.contact and writes its own state
    # It watches an ENTITY, so it must be nested under the entry that provides one -- same reason
    # as contact_monitor: at the top of a document `self.entity` is None and the base body would
    # fall back to a bare "base_link", resolving by accident for some robots and obscurely failing
    # for the rest.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.body = self.config.get("body", "")
        # Tolerant of a malformed entry here: validate_config reports it, and a constructor that
        # raised first would hide that report behind a traceback.
        self.zones: dict[str, tuple[float, float]] = {}
        for k, v in (self.config.get("zones") or {}).items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                self.zones[str(k)] = (float(v[0]), float(v[1]))
        self.geoms = list(self.config.get("geoms", []))
        self.geom_prefixes = list(self.config.get("geom_prefixes", []))
        self.ignore = list(self.config.get("ignore", ["floor"]))
        self.ignore_prefixes = list(self.config.get("ignore_prefixes", []))
        self.min_force = float(self.config.get("min_force", 1.0))
        self.rate_hz = float(self.config.get("rate_hz", 62.0))
        # Reused across steps: mj_contactForce writes into it, and allocating a 6-vector per
        # contact per step is exactly the kind of cost this plugin must not add.
        self._force_scratch = np.zeros(6)
        self._ctx: SimContext | None = None
        self._scope: ContactScope | None = None
        self._root = -1
        self._reading = BumperReading(dict.fromkeys(self.zones, False), False, 0.0)

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        zones = config.get("zones")
        if not isinstance(zones, dict) or not zones:
            errors.append(
                "'zones' must be a non-empty mapping of zone name -> [from, to] bearing (rad)"
            )
        else:
            for name, sector in zones.items():
                if (
                    not isinstance(sector, (list, tuple))
                    or len(sector) != 2
                    or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) for v in sector
                    )
                ):
                    errors.append(f"zones[{name!r}] must be [from, to], two bearings in radians")
                    continue
                lo, hi = float(sector[0]), float(sector[1])
                if not (-math.pi <= lo <= math.pi and -math.pi <= hi <= math.pi):
                    errors.append(f"zones[{name!r}]: bearings must be within [-pi, pi]")
                if lo == hi:
                    errors.append(f"zones[{name!r}]: an empty sector (from == to) presses nothing")
                if "/" in str(name) or not str(name):
                    errors.append(f"zones[{name!r}]: a zone name is one topic segment")
        if float(config.get("rate_hz", 62.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("min_force", 1.0)) < 0:
            errors.append("'min_force' must be >= 0")
        for key in ("geoms", "geom_prefixes", "ignore", "ignore_prefixes"):
            if key in config and not isinstance(config[key], list):
                errors.append(f"'{key}' must be a list of strings")
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        model = ctx.model
        entity = ctx.entities.get(self.robot)
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        # Which contacts are this entity's: the rule shared with every other contact observable,
        # so the bumper and contact_monitor can never disagree about whose contact it was.
        scope = resolve_contact_scope(
            model,
            entity,
            plugin="bumper",
            body=self.body,
            ignore=self.ignore,
            ignore_prefixes=self.ignore_prefixes,
        )
        self._root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scope.body)
        if self.geoms or self.geom_prefixes:
            # Narrow the subtree to the shell. Names are the model's own, before any spawn prefix,
            # like every other geom name a manifest states. A named geom that is not in the model,
            # or not on this entity, is a bumper watching nothing -- which reports "released"
            # forever and reads like a robot that never hit anything.
            prefix = entity.meta.get("prefix", "") if entity else ""
            for name in self.geoms:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
                if gid < 0:
                    raise RuntimeError(f"bumper: geom {prefix + name!r} not found")
                if not scope.watched[gid]:
                    raise RuntimeError(
                        f"bumper: geom {prefix + name!r} is not part of {scope.body!r}'s subtree"
                    )
            shell = np.zeros(model.ngeom, dtype=bool)
            for gid in np.flatnonzero(scope.watched):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
                bare = name.removeprefix(prefix)
                if bare in self.geoms or any(bare.startswith(p) for p in self.geom_prefixes):
                    shell[gid] = True
            if not shell.any():
                raise RuntimeError("bumper: 'geoms'/'geom_prefixes' matched no geom of the entity")
            # The shell is the named geoms and nothing else, so no flex of the entity presses it.
            scope = replace(scope, watched=shell, watched_flex=np.zeros_like(scope.watched_flex))
        self._scope = scope

        ctx.blackboard.set(f"bumper:{self.address}", self.read_state)
        for zone in self.zones:
            ctx.interface.add(
                Endpoint(
                    name=f"bumper/{zone}",
                    direction="out",
                    owner=self.robot,
                    namespace=ns,
                    read=lambda z=zone: self._reading.pressed[z],
                    rate_hz=self.rate_hz,
                    # Cheap to read, but there are as many of these as zones and only a safety
                    # stack listens: nothing is published until something subscribes.
                    lazy=True,
                    backend={
                        "ros2": {
                            "type": "std_msgs.msg.Bool",
                            "topic": self.topic_override(f"bumper/{zone}") or f"bumper/{zone}",
                        }
                    },
                )
            )
        _log.info(
            "bumper: %d zones on %r over %d geoms",
            len(self.zones),
            scope.body,
            int(scope.watched.sum()),
        )

    def read_state(self) -> BumperReading:
        """The current reading. A callable, not the dataclass: ``post_step`` REPLACES it each step,
        so a consumer holding the object would read one frozen step forever."""
        return self._reading

    def on_reset(self, ctx: SimContext) -> None:
        self._reading = BumperReading(dict.fromkeys(self.zones, False), False, 0.0)

    def post_step(self, ctx: SimContext) -> None:
        data, model = ctx.data, ctx.model
        idx = self._scope.indices(data)
        pressed = dict.fromkeys(self.zones, False)
        if idx.size:
            # Into the watched body's own frame, so a zone is a property of the shell and not of
            # where the robot happens to stand. xmat is row-major 3x3; the transpose is a column
            # read, written out to keep this allocation-free.
            ox, oy, oz = data.xpos[self._root].tolist()
            m = data.xmat[self._root].tolist()
            force = self._force_scratch
            con = data.contact
            for i in idx:
                i = int(i)
                if self.min_force > 0:
                    mujoco.mj_contactForce(model, data, i, force)
                    if abs(float(force[0])) < self.min_force:
                        continue
                px, py, pz = con.pos[i].tolist()
                dx, dy, dz = px - ox, py - oy, pz - oz
                bx = m[0] * dx + m[3] * dy + m[6] * dz
                by = m[1] * dx + m[4] * dy + m[7] * dz
                bearing = math.atan2(by, bx)
                for zone, (lo, hi) in self.zones.items():
                    if not pressed[zone] and _in_sector(bearing, lo, hi):
                        pressed[zone] = True
        self._reading = BumperReading(pressed, any(pressed.values()), float(data.time))
