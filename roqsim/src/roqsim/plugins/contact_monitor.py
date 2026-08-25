"""Observation plugin: report when an entity touches something it should not.

The substrate's first contact-derived signal. MuJoCo already computes every contact each step; this
plugin turns the subset that matters into an observable, so a scenario or an evaluator can end a
trial on a collision instead of inferring one from a proximity threshold.

Why a threshold is not good enough: a clearance-based proxy puts a tunable number between the
simulator and the result. For a navigation experiment whose failure criterion *is* "did it hit
something", the honest signal is a real contact force, which needs no calibration and no arguing
about where the footprint ends.

What counts as a collision is defined by exclusion, not by enumeration: every contact involving one
of the watched entity's bodies counts, EXCEPT contacts against a geom in ``ignore`` (by default the
ground plane, which a wheeled robot touches continuously by design). Listing what a robot may touch
is short and stable; listing what it may not is neither.

Config::

    contact_monitor:
      # The entity watched is the one this entry is NESTED UNDER -- there is no key for it, and
      # declaring it at the top of a document is refused (`requires_owner`).
      body: ""               # base body override; default: the entity's registered base body
      namespace: ""          # transport scope for the endpoint
      ignore: [floor]        # geom NAMES that never count as a collision (default: ['floor'])
      ignore_prefixes: []    # geom name prefixes that never count (e.g. ['ground'])
      min_force: 1.0         # N; contacts below this normal force are ignored (numerical grazing)
      latch: true            # once true, stay true until on_reset (a trial is failed, not un-failed)
      rate_hz: 30.0          # endpoint publish rate

Endpoint ``contact`` (out) reads a :class:`ContactReport`:
``(in_contact, first_time, count, geom_a, geom_b)`` -- ``first_time`` is the simulation time of the
first qualifying contact since reset (``-1.0`` if none), and ``geom_a``/``geom_b`` name the geoms of
that first contact, so a failure is attributable rather than just flagged. The ROS 2 backend hint
publishes ``in_contact`` as a ``std_msgs/Bool`` on ``collision`` (relative, so it is scoped by the
entity's namespace: two namespaced robots get ``/a/collision`` and ``/b/collision``); a bridge that
wants the detail reads the fields directly.

The watched set is the entity's **kinematic subtree**: for a mobile base that is the chassis plus its
wheels, so a wheel clipping a box counts exactly as much as the bumper does.
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
class ContactReport:
    """Neutral payload for the ``contact`` endpoint."""

    in_contact: bool
    first_time: float  # sim time of the first qualifying contact since reset; -1.0 if none
    count: int  # qualifying contacts in the most recent step
    geom_a: str  # geoms of the FIRST qualifying contact ("" until one happens)
    geom_b: str


class ContactMonitorPlugin(Plugin):
    parallel_safe = True  # post_step only reads data.contact and writes its own state
    # It watches an ENTITY, so it must be nested under the entry that provides one. Declared
    # rather than left implicit: at the top of a document `self.entity` is None, and the base
    # body fell back to a bare "base_link" -- resolving by accident when a robot happened to
    # use that name, and failing obscurely when one did not.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.body = self.config.get("body", "")
        self.ignore = list(self.config.get("ignore", ["floor"]))
        self.ignore_prefixes = list(self.config.get("ignore_prefixes", []))
        self.min_force = float(self.config.get("min_force", 1.0))
        self.latch = bool(self.config.get("latch", True))
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self._ctx: SimContext | None = None
        self._watched: set[int] = set()  # geom ids belonging to the watched subtree
        self._ignored: set[int] = set()  # geom ids that never count
        self._report = ContactReport(False, -1.0, 0, "", "")

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("min_force", 1.0)) < 0:
            errors.append("'min_force' must be >= 0")
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
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if root < 0:
            # Fail loudly: a monitor watching nothing would report "no collisions" forever, which
            # is indistinguishable from a clean run and would silently pass every trial.
            raise RuntimeError(f"contact_monitor: base body {body_name!r} not found")

        self._watched = {
            gid
            for gid in range(model.ngeom)
            if self._in_subtree(model, int(model.geom_bodyid[gid]), root)
        }
        if not self._watched:
            raise RuntimeError(
                f"contact_monitor: body {body_name!r} and its subtree carry no geoms to watch"
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
            # Not fatal (a world may legitimately have no `floor` geom), but never silent: an
            # unmatched ignore entry is how a ground plane starts counting as a collision.
            _log.warning(
                "contact_monitor: ignore entry has no matching geom: %s", ", ".join(missing)
            )

        # The same report, for a driver in this process: an `.osc` action, a test, another plugin.
        # A consumer would otherwise have to find this instance in `engine.plugins` and match it by
        # class name -- which is what the handle convention exists to retire (architecture.rst §12).
        # `read_state` rather than the report itself, because the report is REPLACED each step.
        # Keyed on the ADDRESS. `self.name` falls back to the class name, so two unnamed
        # monitors in one world wrote to a single key and the second silently replaced the
        # first -- one robot's collisions reported as another's.
        ctx.blackboard.set(f"contact:{self.address}", self.read_state)

        ctx.interface.add(
            Endpoint(
                name="contact",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._report,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Bool",
                        # The report is a structure and Bool carries one field, so the endpoint says
                        # WHICH -- rather than the bridge holding a converter that knows this
                        # plugin's attribute names. The other fields stay readable in-process.
                        "field": "in_contact",
                        "topic": self.topic_override("contact") or "collision",
                    }
                },
            )
        )
        _log.info(
            "contact_monitor: watching %d geoms of %r, ignoring %d",
            len(self._watched),
            body_name,
            len(self._ignored),
        )

    def read_state(self) -> ContactReport:
        """The latest report. What the blackboard handle hands an in-process consumer.

        A callable rather than the report object, because ``post_step`` REPLACES the report each
        step -- a consumer holding the dataclass would read one frozen step forever.
        """
        return self._report

    @staticmethod
    def _in_subtree(model, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return body == root

    def on_reset(self, ctx: SimContext) -> None:
        self._report = ContactReport(False, -1.0, 0, "", "")

    def post_step(self, ctx: SimContext) -> None:
        data, model = ctx.data, ctx.model
        hits = 0
        first: tuple[str, str] | None = None
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._watched) == (g2 in self._watched):
                continue  # neither side watched, or a self-contact: not an external collision
            if g1 in self._ignored or g2 in self._ignored:
                continue
            if self.min_force > 0:
                mujoco.mj_contactForce(model, data, i, force)
                if abs(float(force[0])) < self.min_force:
                    continue
            hits += 1
            if first is None:
                first = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}",
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}",
                )

        if hits and self._report.first_time < 0.0:
            self._report = ContactReport(True, float(data.time), hits, first[0], first[1])
            _log.info(
                "contact_monitor: %r collided (%s <-> %s) at t=%.3f s",
                self.robot,
                first[0],
                first[1],
                data.time,
            )
        else:
            in_contact = bool(hits) or (self.latch and self._report.first_time >= 0.0)
            self._report = ContactReport(
                in_contact,
                self._report.first_time,
                hits,
                self._report.geom_a,
                self._report.geom_b,
            )
