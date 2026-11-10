# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""World plugin: presence in a region -- who entered a box, who dwelt there, who left.

A light barrier at a belt's end, the space before an automatic door, the die of a press: each is a
box in which a scene needs to know who is there. This plugin reports it and decides nothing. What a
presence leads to -- a belt stopping, a door opening -- is a scenario's (``docs/architecture.rst``,
"Machines and scenarios"); a machine that works on what is in its tool reads the zone nested under
it.

**The box.** ``size`` gives its full extents, as a ``box``'s, a conveyor's and a door's do;
``pose`` its centre and orientation: in the world, or, with ``body``, in that body's frame, so the zone rides on it (a barrier on a moving carriage, a die
zone on a press). Nested under an entity, ``body`` is named after that entity's prefix. An entity is
inside when its ``point`` is: ``com`` (default) the centre of mass of its root body; ``origin`` the
root body's frame, for a body whose mass is not where it stands (a walker's mocap pelvis).

**Who counts.** ``entities`` filters by name pattern (fnmatch, ``["box_*", "walker_*"]``); without it
every present entity with a body counts, except the states of :mod:`part_state`. The entity the
zone is nested under never counts, whatever ``entities`` says. An entity counts only while its root
body is present (:func:`roqsim.presence.body_present`), so one inside an absent entity does not. While
the body a zone rides is absent, or the entity it is nested under is, the zone reports no one. A
pattern that matches no entity at the first reset is logged once as a warning, not refused.

**What it reports.** Each step, on the physics thread: an ``entered`` event with the entity, a
``dwelled`` event once it has been inside for ``dwell`` seconds, and a ``left`` event with how long
it was inside. A :class:`ZoneHandle` under ``zone:<address>`` reads the same: who is inside, in the
order they entered, since when, who dwelt, the counts since the episode began. ``occupied`` (a
``bool``, ``std_msgs/Bool`` on ROS) is the barrier's signal for a stack or an OpenSCENARIO monitor;
``occupancy`` lists who is inside and who has been this episode.

Config::

    zone_trigger:
      body: ""                  # a body the zone rides on; the pose is then in its frame
      pose:                     # the box's centre (roqsim.pose), world or `body` frame
        position: {x: 1.2, y: 0.0, z: 0.95}
        orientation: {yaw: 0.0}
      size: [0.04, 0.6, 0.2]    # full extents (m), as a box's
      entities: [box_1]         # entity name patterns that count (fnmatch, e.g. box_*); default all
      point: com                # com: the root body's centre of mass | origin: its frame
      dwell: 1.0                # seconds inside before `dwelled` (0: never)
"""

from __future__ import annotations

import fnmatch
import logging
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors
from roqsim.presence import body_present

_log = logging.getLogger(__name__)

_KEYS = {"body", "pose", "size", "entities", "point", "dwell"}


@dataclass(frozen=True)
class ZoneCounts:
    """How many entities entered and left since the episode began."""

    entered: int = 0
    left: int = 0


@dataclass(frozen=True)
class ZoneOccupancy:
    """Who is inside a zone, and who has been this episode.

    Attributes:
        inside: the entities inside, in the order they entered
        count: how many
        dwelling: the entities inside that have been for the zone's ``dwell``
        dwell: seconds inside before an entity counts as dwelling; 0 means never
        entered: every entity that entered, in order, once per entry
        entered_at: when each of ``entered`` did (s, sim time)
        dwelled: every entity that reached the zone's ``dwell``, in order
        dwelled_at: when each of ``dwelled`` did (s, sim time)
    """

    inside: list[str] = field(default_factory=list)
    count: int = 0
    dwelling: list[str] = field(default_factory=list)
    dwell: float = 0.0
    entered: list[str] = field(default_factory=list)
    entered_at: list[float] = field(default_factory=list)
    dwelled: list[str] = field(default_factory=list)
    dwelled_at: list[float] = field(default_factory=list)


class ZoneHandle:
    """Published by ``zone_trigger`` under ``zone:<address>``. Latched in ``post_step``; safe to read
    from any thread."""

    def __init__(self, name: str, dwell: float = 0.0):
        self.name = name
        #: Seconds inside before an entity counts as dwelling; 0 means never.
        self.dwell = float(dwell)
        self._inside: dict[str, float] = {}
        self._dwelt: list[str] = []
        self._counts = ZoneCounts()
        #: ``(sim time, entity)`` of each entry and each dwell this episode.
        self._entries: list[tuple[float, str]] = []
        self._dwells: list[tuple[float, str]] = []

    def inside(self) -> list[str]:
        """The entities inside now, in the order they entered."""
        return list(self._inside)

    def entered_at(self, entity: str) -> float | None:
        """When ``entity`` entered (s, sim time), or ``None`` when it is not inside."""
        return self._inside.get(entity)

    def dwelling(self) -> list[str]:
        """The entities inside that have been for the zone's ``dwell``."""
        return list(self._dwelt)

    def entered_since(self, when: float) -> list[str]:
        """The entities that entered at or after ``when`` (s, sim time), inside or gone since, each
        once, in the order they first did."""
        return list(dict.fromkeys(name for t, name in self._entries if t >= when))

    def dwelled_since(self, when: float) -> list[str]:
        """The entities that reached the zone's ``dwell`` at or after ``when``, inside or gone since,
        each once, in the order they first did."""
        return list(dict.fromkeys(name for t, name in self._dwells if t >= when))

    @property
    def occupied(self) -> bool:
        return bool(self._inside)

    @property
    def counts(self) -> ZoneCounts:
        return self._counts

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ZoneHandle({self.name!r}, inside={self.inside()})"


class ZoneTriggerPlugin(Plugin):
    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.pos, self.quat = config_pose(self.config)
        self.half = np.asarray(self.config.get("size") or [0.0, 0.0, 0.0], dtype=float) / 2.0
        self.patterns = list(self.config.get("entities") or [])
        self.point = str(self.config.get("point", "com"))
        self.dwell = float(self.config.get("dwell", 0.0) or 0.0)
        self.handle = ZoneHandle(self.address, dwell=self.dwell)
        self._ctx: SimContext | None = None
        self._bid = -1
        self._warned = False
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(self.quat, dtype=float))
        self._rot = rot.reshape(3, 3)

    def validate_config(self, config: dict) -> list[str]:
        errors = config_pose_errors(config, "zone_trigger")
        unknown = set(config) - _KEYS - {"namespace", "topics", "qos"}
        if unknown:
            errors.append(
                f"zone_trigger: unknown key(s) {sorted(unknown)}; it takes {sorted(_KEYS)}"
            )
        size = config.get("size")
        if (
            not isinstance(size, list)
            or len(size) != 3
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in size
            )
        ):
            errors.append("zone_trigger: 'size' is three positive full extents (m), as a box's")
        if str(config.get("point", "com")) not in ("com", "origin"):
            errors.append("zone_trigger: 'point' is com or origin")
        if not isinstance(config.get("body", ""), str):
            errors.append("zone_trigger: 'body' is a body's name")
        entities = config.get("entities")
        if entities is not None and not (
            isinstance(entities, list) and entities and all(isinstance(p, str) for p in entities)
        ):
            errors.append("zone_trigger: 'entities' is a list of entity name patterns, not empty")
        dwell = config.get("dwell", 0.0)
        if (
            isinstance(dwell, bool)
            or not isinstance(dwell, (int, float))
            or not math.isfinite(dwell)
            or dwell < 0
        ):
            errors.append("zone_trigger: 'dwell' is seconds, 0 or more")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        body = self.config.get("body") or ""
        if body:
            prefix = ""
            owner = ctx.entities.get(self.entity) if self.entity else None
            if owner is not None and isinstance(owner.meta, dict):
                prefix = str(owner.meta.get("prefix", ""))
            name = prefix + body
            self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if self._bid < 0:
                bodies = [
                    mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    for i in range(ctx.model.nbody)
                ]
                own = [b for b in bodies if b and b.startswith(prefix)]
                raise RuntimeError(
                    f"zone_trigger {self.address}: no body {name!r} to ride on"
                    + (f" (the bodies: {', '.join(own)})" if own else "")
                )
        ctx.blackboard.set(f"zone:{self.address}", self.handle)

    def on_reset(self, ctx: SimContext) -> None:
        self.handle._inside = {}
        self.handle._dwelt = []
        self.handle._counts = ZoneCounts()
        self.handle._entries = []
        self.handle._dwells = []
        if not self._warned:
            self._warned = True
            names = ctx.entities.names()
            for pattern in self.patterns:
                if not any(fnmatch.fnmatchcase(n, pattern) for n in names):
                    _log.warning(
                        "zone_trigger %s: 'entities' pattern %r matches no entity of the world",
                        self.address,
                        pattern,
                    )

    def _frame(self, ctx: SimContext) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(self.pos, dtype=float)
        if self._bid < 0:
            return pos, self._rot
        body_rot = ctx.data.xmat[self._bid].reshape(3, 3)
        return ctx.data.xpos[self._bid] + body_rot @ pos, body_rot @ self._rot

    def _candidates(self, ctx: SimContext):
        for e in ctx.entities.all(present_only=True):
            if not e.body or e.kind == "look" or e.name == self.entity:
                continue
            if self.patterns and not any(fnmatch.fnmatchcase(e.name, p) for p in self.patterns):
                continue
            yield e

    def _reporting(self, ctx: SimContext) -> bool:
        """Whether the zone is there to report: the body it rides and the entity it is nested under
        are both present, through every entity around them."""
        if self._bid >= 0 and not body_present(ctx, self._bid):
            return False
        owner = ctx.entities.get(self.entity) if self.entity else None
        if owner is None:
            return True
        if not owner.present:
            return False
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, owner.body or "")
        return bid < 0 or body_present(ctx, bid)

    def post_step(self, ctx: SimContext) -> None:
        centre, rot = self._frame(ctx)
        now = ctx.sim_time
        h = self.handle
        inside_now = []
        for e in self._candidates(ctx) if self._reporting(ctx) else ():
            bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, e.body)
            if bid < 0 or not body_present(ctx, bid):
                continue
            point = ctx.data.xipos[bid] if self.point == "com" else ctx.data.xpos[bid]
            if np.all(np.abs(rot.T @ (point - centre)) <= self.half):
                inside_now.append(e.name)
        inside = dict(h._inside)
        dwelt = list(h._dwelt)
        entered = left = 0
        for name in list(inside):
            if name not in inside_now:
                since = inside.pop(name)
                if name in dwelt:
                    dwelt.remove(name)
                left += 1
                ctx.events.emit(self.address, "left", entity=name, inside_s=now - since)
        for name in inside_now:
            if name not in inside:
                inside[name] = now
                entered += 1
                h._entries.append((now, name))
                ctx.events.emit(self.address, "entered", entity=name)
            elif self.dwell > 0 and name not in dwelt and now - inside[name] >= self.dwell - 1e-9:
                dwelt.append(name)
                h._dwells.append((now, name))
                ctx.events.emit(self.address, "dwelled", entity=name, after=now - inside[name])
        h._inside, h._dwelt = inside, dwelt
        if entered or left:
            c = h._counts
            h._counts = ZoneCounts(c.entered + entered, c.left + left)

    def _occupied_topic(self) -> str:
        """The zone's own topic, relative to a namespace it is under: the entity that holds the
        namespace names it, so only a zone with none spells out the entities around it."""
        scoped = self.entity and self.endpoint_namespace(self._ctx)
        return f"{self.label if scoped else self.address.replace('.', '/')}/occupied"

    @property
    def endpoint_owner(self) -> str:
        """The endpoints belong to this zone: an entity may carry several."""
        return self.address

    def endpoint_namespace(self, ctx: SimContext, owner: str | None = None) -> str:
        """The namespace of the entity the zone is nested under, the zone's own being no entity."""
        return super().endpoint_namespace(ctx, self.entity or owner)

    @endpoint.out(rate=10.0, ros2=lambda self: {"topic": self._occupied_topic()})
    def occupied(self) -> bool:
        """Whether an entity that counts is inside the zone."""
        return self.handle.occupied

    @endpoint.out(rate=10.0, ros2=None)
    def occupancy(self) -> ZoneOccupancy:
        """Who is inside the zone, and who has been this episode."""
        h, inside = self.handle, self.handle.inside()
        return ZoneOccupancy(
            inside=inside,
            count=len(inside),
            dwelling=h.dwelling(),
            dwell=h.dwell,
            entered=[name for _, name in h._entries],
            entered_at=[t for t, _ in h._entries],
            dwelled=[name for _, name in h._dwells],
            dwelled_at=[t for t, _ in h._dwells],
        )
