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
      rate_hz: 30.0          # endpoint publish rate

Endpoint ``contact_location`` (out) reads a :class:`ContactLocation`:
``(in_contact, kind, x, y, z, extent, count, time)``. ``kind`` is ``"none"``, ``"point"`` or
``"line"``; ``x/y/z`` is the region's centre and ``extent`` the distance between its two furthest
members (0.0 for a point), so a consumer gets both the location and how spread out it is. The ROS 2
backend hint publishes the centre as a ``geometry_msgs/PointStamped`` on ``contact_location``.

**Read it through the blackboard, not the endpoint, inside a control loop.** The endpoint is
rate-limited for logging; ``ctx.blackboard.get(f"contact_location:{address}")`` returns a callable
giving the current reading at full step rate, the same convention ``contact_monitor`` and
``force_torque`` use.

Frames. ``frame: base`` (the default) rotates the position into the watched body's own frame, which
is what a controller reasoning about "contact on my left" wants and what makes a reading independent
of where the robot happens to be standing. ``frame: world`` leaves it in world coordinates.
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
        self._ctx: SimContext | None = None
        self._watched: set[int] = set()
        self._ignored: set[int] = set()
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

        self._watched = {
            gid
            for gid in range(model.ngeom)
            if self._in_subtree(model, int(model.geom_bodyid[gid]), self._root)
        }
        if not self._watched:
            raise RuntimeError(
                f"contact_location: body {body_name!r} and its subtree carry no geoms to watch"
            )

        self._ignored = set()
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name in self.ignore or any(name.startswith(p) for p in self.ignore_prefixes):
                self._ignored.add(gid)
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
            len(self._watched),
            body_name,
            self.frame,
            len(self._ignored),
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
        force = np.zeros(6)
        points: list[np.ndarray] = []
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._watched) == (g2 in self._watched):
                continue  # neither side watched, or a self-contact
            if g1 in self._ignored or g2 in self._ignored:
                continue
            if self.min_force > 0:
                mujoco.mj_contactForce(model, data, i, force)
                if abs(float(force[0])) < self.min_force:
                    continue
            points.append(np.array(c.pos, dtype=float))

        if not points:
            self._reading = ContactLocation(False, "none", 0.0, 0.0, 0.0, 0.0, 0, float(data.time))
            return

        # Merge solver contacts a real sensing area could not tell apart. MuJoCo often solves
        # several contacts across one flat face touching another; a skin with 2 cm taxels reports
        # that as one area, and counting them separately would classify every flat push as a line.
        areas: list[np.ndarray] = []
        for p in points:
            for a in areas:
                if float(np.linalg.norm(a - p)) <= self.merge_radius:
                    break
            else:
                areas.append(p)

        arr = np.array(areas)
        centre = arr.mean(axis=0)
        if len(arr) > 1:
            # Widest separation in the set: the region's extent, and what distinguishes a corner
            # touch from a whole face resting against something.
            d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            extent = float(d.max())
        else:
            extent = 0.0
        # The rule a taxel array implements: one sensing area in contact is a point, more than one
        # is a line.
        kind = "point" if len(arr) == 1 else "line"

        if self.frame == "base":
            # Into the watched body's own frame: what a controller reasoning about "contact on my
            # left" needs, and what makes the reading independent of where the robot is standing.
            origin = np.array(data.xpos[self._root], dtype=float)
            rot = np.array(data.xmat[self._root], dtype=float).reshape(3, 3)
            centre = rot.T @ (centre - origin)

        self._reading = ContactLocation(
            True,
            kind,
            float(centre[0]),
            float(centre[1]),
            float(centre[2]),
            extent,
            len(arr),
            float(data.time),
        )
