"""Observation plugin: report **where** a watched entity is being touched, every step.

The sibling of :mod:`roqsim.plugins.contact_monitor`, which answers "did it touch" and deliberately
nothing more. This one answers "where, right now" -- the signal a tactile skin gives a controller,
and the one a bumper-driven behaviour needs in its control loop rather than at the end of a trial.

The two are separate plugins because they are separate questions with different lifetimes.
``contact_monitor`` latches the FIRST qualifying contact and keeps it, because a trial that touched
something is failed and not un-failed. A tactile reading is the opposite: it is only useful while it
is current, it is replaced every step, and a stale one is worse than none.

What a real capacitive skin reports, and what this reproduces, is a contact **region** rather than a
point: one taxel firing is a point contact, several adjacent ones a line. MuJoCo already computes a
position for every contact it solves, so the region is the set of simultaneous qualifying contact
positions, and the classification is their count -- no taxel grid is modelled, and none needs to be.
A skin's spatial resolution enters as ``merge_radius`` instead: two solver contacts closer together
than one sensing area cannot be told apart by the real device either.

Config::

    contact_location:
      # The entity watched is the one this entry is NESTED UNDER -- there is no key for it, and
      # declaring it at the top of a document is refused (`requires_owner`).
      body: ""               # base body override; default: the entity's registered base body
      namespace: ""          # transport scope for the endpoint
      ignore: [floor]        # geom NAMES that never count (default: ['floor'])
      ignore_prefixes: []    # geom name prefixes that never count (e.g. ['ground'])
      min_force: 1.0         # N; contacts below this normal force are ignored (numerical grazing)
      merge_radius: 0.02     # m; contacts closer than this count as one sensing area
      frame: base            # 'base' -> report in the watched body's frame; 'world' -> world frame
      compute_rate_hz: 0.0   # how often the reading is COMPUTED; 0 = every physics step
      rate_hz: 30.0          # how often the endpoint is PUBLISHED

Endpoint ``contact_location`` (out) reads a :class:`ContactLocation`:
``(in_contact, kind, x, y, z, extent, count, time)``. ``kind`` is ``"none"``, ``"point"`` or
``"line"``; ``x/y/z`` is the region's centre and ``extent`` the distance between its two furthest
members (0.0 for a point), so a consumer gets both the location and how spread out it is. The ROS 2
backend hint publishes the centre as a ``geometry_msgs/PointStamped`` on ``contact_location``.

**Read it through the blackboard, not the endpoint, inside a control loop.** The endpoint is
rate-limited for logging; ``ctx.blackboard.get(f"contact_location:{address}")`` returns a callable
giving the current reading, the same convention ``contact_monitor`` and ``force_torque`` use.

Cost. The per-step work is a vectorised pass over ``data.contact`` -- a mask lookup per contact and
an xor -- and a contact force query only for the handful that survive it. That matters because a
world's contacts are overwhelmingly pairs the robot is not in (props resting on the floor, a
crowd's feet), and touching each of them from Python costs more than the physics step that produced
them. ``compute_rate_hz`` decimates the whole computation for the case where even that is too much;
it defaults to 0, meaning every step, because unlike a latching monitor this plugin cannot recover a
contact it did not look at -- one shorter than the interval is simply missed.

Frames. ``frame: base`` (the default) rotates the position into the watched body's own frame, which
is what a controller reasoning about "contact on my left" wants and what makes a reading independent
of where the robot happens to be standing. ``frame: world`` leaves it in world coordinates.
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
class ContactLocation:
    """Neutral payload for the ``contact_location`` endpoint."""

    in_contact: bool
    kind: str  # "none" | "point" | "line"
    x: float  # region centre, in the configured frame
    y: float
    z: float
    extent: float  # m between the two furthest members; 0.0 for a point contact
    count: int  # distinct sensing areas in contact this step (after merge_radius)
    time: float  # sim time of this reading


class ContactLocationPlugin(Plugin):
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
        self.ignore = list(self.config.get("ignore", ["floor"]))
        self.ignore_prefixes = list(self.config.get("ignore_prefixes", []))
        self.min_force = float(self.config.get("min_force", 1.0))
        self.merge_radius = float(self.config.get("merge_radius", 0.02))
        self.frame = str(self.config.get("frame", "base"))
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        # 0 = every physics step, which is the default because the scan is a vectorised pass over
        # the contact array and costs a few per cent of a step. Raise the interval only if a
        # contact-heavy world makes even that matter; the trade is that a contact shorter than one
        # interval can be missed entirely, which a latching monitor cannot do to you.
        self.compute_rate_hz = float(self.config.get("compute_rate_hz", 0.0))
        self._period = 1.0 / self.compute_rate_hz if self.compute_rate_hz > 0 else 0.0
        self._accum = 0.0
        # Reused across steps: mj_contactForce writes into it, and allocating a 6-vector per
        # contact per step is exactly the kind of cost this plugin must not add.
        self._force_scratch = np.zeros(6)
        self._ctx: SimContext | None = None
        self._watched: np.ndarray | None = None  # per-geom mask; see configure()
        self._ignored: np.ndarray | None = None
        self._root = -1
        self._reading = ContactLocation(False, "none", 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("min_force", 1.0)) < 0:
            errors.append("'min_force' must be >= 0")
        if float(config.get("merge_radius", 0.02)) < 0:
            errors.append("'merge_radius' must be >= 0")
        if config.get("frame", "base") not in ("base", "world"):
            errors.append("'frame' must be 'base' or 'world'")
        for key in ("ignore", "ignore_prefixes"):
            if key in config and not isinstance(config[key], list):
                errors.append(f"'{key}' must be a list of strings")
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
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
        self._root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self._root < 0:
            # Fail loudly: a sensor watching nothing reports "no contact" forever, which a
            # controller cannot distinguish from open space and would drive straight through.
            raise RuntimeError(f"contact_location: base body {body_name!r} not found")

        # Boolean masks indexed by geom id, not Python sets: the per-step filter is then one
        # vectorised lookup over the contact array instead of a membership test per contact.
        self._watched = np.zeros(model.ngeom, dtype=bool)
        for gid in range(model.ngeom):
            if self._in_subtree(model, int(model.geom_bodyid[gid]), self._root):
                self._watched[gid] = True
        if not self._watched.any():
            raise RuntimeError(
                f"contact_location: body {body_name!r} and its subtree carry no geoms to watch"
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
            # Not fatal, never silent: an unmatched ignore entry is how the ground plane starts
            # reading as a permanent contact under the robot.
            _log.warning(
                "contact_location: ignore entry has no matching geom: %s", ", ".join(missing)
            )

        ctx.blackboard.set(f"contact_location:{self.address}", self.read_state)
        ctx.interface.add(
            Endpoint(
                name="contact_location",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._reading,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.PointStamped",
                        "topic": self.topic_override("contact_location") or "contact_location",
                    }
                },
            )
        )
        _log.info(
            "contact_location: watching %d geoms of %r in the %s frame, ignoring %d",
            int(self._watched.sum()),
            body_name,
            self.frame,
            int(self._ignored.sum()),
        )

    def read_state(self) -> ContactLocation:
        """The current reading. A callable, not the dataclass: ``post_step`` REPLACES it each step,
        so a consumer holding the object would read one frozen step forever."""
        return self._reading

    @staticmethod
    def _in_subtree(model, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return body == root

    def on_reset(self, ctx: SimContext) -> None:
        self._reading = ContactLocation(False, "none", 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    def post_step(self, ctx: SimContext) -> None:
        data, model = ctx.data, ctx.model

        # Decimate the whole computation, not just the publish. See `compute_rate_hz`.
        if self._period > 0.0:
            self._accum += model.opt.timestep
            if self._accum < self._period:
                return
            self._accum = 0.0

        n = data.ncon
        if n == 0:
            self._reading = ContactLocation(False, "none", 0.0, 0.0, 0.0, 0.0, 0, float(data.time))
            return

        # Filter the whole contact array at once. A world's contacts are dominated by pairs the
        # robot is not in -- props resting on the floor, a crowd's feet -- and touching each of
        # them from Python costs more than the physics step that produced them. `data.contact` is
        # a struct of arrays, so the geom test is two mask lookups and an xor.
        con = data.contact
        g1, g2 = con.geom1[:n], con.geom2[:n]
        w1, w2 = self._watched[g1], self._watched[g2]
        # Exactly one side watched: neither is nothing to do with us, both is a self-contact.
        keep = (w1 ^ w2) & ~(self._ignored[g1] | self._ignored[g2])
        idx = np.flatnonzero(keep)

        if idx.size and self.min_force > 0:
            # Only now, and only for the handful that survived: this is a C call per contact.
            force = self._force_scratch
            strong = []
            for i in idx:
                mujoco.mj_contactForce(model, data, int(i), force)
                if abs(float(force[0])) >= self.min_force:
                    strong.append(i)
            idx = np.asarray(strong, dtype=int)

        if not idx.size:
            self._reading = ContactLocation(False, "none", 0.0, 0.0, 0.0, 0.0, 0, float(data.time))
            return
        # `.tolist()` once, rather than iterating the array: looping a numpy array yields a numpy
        # scalar per element, and at these sizes that conversion is most of the remaining cost.
        points = con.pos[idx].tolist()

        # Merge solver contacts a real sensing area could not tell apart. MuJoCo often solves
        # several contacts across one flat face touching another; a skin with 2 cm taxels reports
        # that as one area, and counting them separately would classify every flat push as a line.
        #
        # Kept as plain arithmetic on floats rather than a numpy call per pair: after the filter
        # above there are typically one to a handful of contacts, and at that size numpy's
        # per-call overhead is most of the cost of the whole plugin.
        r2 = self.merge_radius * self.merge_radius
        areas: list[tuple[float, float, float]] = []
        for px, py, pz in points:
            for ax, ay, az in areas:
                if (ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2 <= r2:
                    break
            else:
                areas.append((px, py, pz))

        k = len(areas)
        cx = sum(a[0] for a in areas) / k
        cy = sum(a[1] for a in areas) / k
        cz = sum(a[2] for a in areas) / k
        # Widest separation in the set: the region's extent, and what distinguishes a corner touch
        # from a whole face resting against something. O(k^2) over the MERGED areas, which is the
        # sensing-area count and not the solver's contact count.
        extent = 0.0
        for i in range(k):
            ax, ay, az = areas[i]
            for j in range(i + 1, k):
                bx, by, bz = areas[j]
                d2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
                if d2 > extent:
                    extent = d2
        extent = math.sqrt(extent)
        # The rule a taxel array implements: one sensing area in contact is a point, more than one
        # is a line.
        kind = "point" if k == 1 else "line"

        if self.frame == "base":
            # Into the watched body's own frame: what a controller reasoning about "contact on my
            # left" needs, and what makes the reading independent of where the robot is standing.
            # xmat is row-major 3x3, so the transpose is a column read -- written out rather than
            # built as a matrix, to keep this allocation-free.
            ox, oy, oz = data.xpos[self._root].tolist()
            m = data.xmat[self._root].tolist()
            dx, dy, dz = cx - ox, cy - oy, cz - oz
            cx = m[0] * dx + m[3] * dy + m[6] * dz
            cy = m[1] * dx + m[4] * dy + m[7] * dz
            cz = m[2] * dx + m[5] * dy + m[8] * dz

        self._reading = ContactLocation(
            True, kind, float(cx), float(cy), float(cz), extent, k, float(data.time)
        )
