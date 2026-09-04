"""Stopping for what the planner cannot see.

The planner's grid holds the world's **static walls and nothing else** -- the robot under test, a
pedestrian, another navigated prop are all deliberately absent from it, because a grid rasterized
once cannot represent something that moves. So a mover following a planned path needs a second,
faster layer that looks where it is actually going.

The invariant, and it is the complement of the grid's:

    **Caution sees exactly what the planner cannot.** A hit counts as a blocker when it belongs to
    neither the mover itself nor to static geometry -- that is, when its body has degrees of freedom
    or is a mocap body. Those are precisely the bodies :func:`~roqsim_nav.obstacles.wall_polygons`
    refuses to rasterize. A wall ahead at a corner is the planner's problem and never this layer's,
    or a mover would stop dead every time its path hugged one.

The response is to **stop**, never to re-route. A stopped mover is still on its path and resumes on
the waypoint it was heading for; a re-routing one has quietly changed the trajectory an experiment
was holding fixed. ``on_blocked`` exists so that a future ``replan`` is a new value rather than a
config break.

Cost matters here, because this runs on the same thread as everything else: it is a handful of rays
at the navigator's tick rate, and none at all when the mover is not moving. :mod:`roqsim.raycast` is
single-threaded on purpose (its docstring documents an ``mjData`` stack race that corrupts an
allocator invariant and crashes much later), so the ray count is a budget rather than a free knob.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim import raycast

from .control import STOPPED

#: Values ``on_blocked`` accepts. Only one is implemented; the other is named so that adding it
#: later is a new value rather than a change of shape.
ON_BLOCKED = ("stop",)
FUTURE_ON_BLOCKED = ("replan",)


def subtree_geoms(model, body_id: int) -> set[int]:
    """Every geom of ``body_id`` and its descendants.

    Needed because ``mj_multiRay``'s ``bodyexclude`` takes a single body: a robot is a base link plus
    wheels plus sensor mounts, so excluding only its root would leave it seeing its own wheels a few
    centimetres ahead and concluding it was permanently blocked.
    """
    bodies = {int(body_id)}
    for b in range(model.nbody):
        parent = b
        while parent > 0:
            if parent in bodies:
                bodies.add(b)
                break
            parent = int(model.body_parentid[parent])
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}


def is_dynamic_body(model, body_id: int) -> bool:
    """Whether this body is something the planner's grid could not have contained.

    The two ways to move in MuJoCo without being a wall: carry degrees of freedom (a robot, a free
    prop), or be a mocap body written each step (a walker's limb, a navigated prop).
    """
    return int(model.body_weldid[body_id]) != 0 or int(model.body_mocapid[body_id]) >= 0


class CautionProbe:
    """A forward clearance check over the corridor a mover is about to occupy.

    ``lookahead`` is how much clear corridor it needs, ``width`` the corridor's width (normally the
    mover's footprint), ``rays`` how finely that width is sampled. ``clear_time`` is how long the way
    must stay clear before moving off again, so a robot crossing tangentially does not make the mover
    stutter forward into its path.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        # Set by the navigator from its `traffic:` policy, not written in a world directly.
        self.enabled = bool(cfg.get("enabled", True))
        self.on_blocked = str(cfg.get("on_blocked", "stop"))
        self.lookahead = float(cfg.get("lookahead", 1.2))
        self.width = float(cfg.get("width", 0.6))
        self.rays = int(cfg.get("rays", 5))
        #: Height above the floor to look at, like a scanning plane -- NOT the mover's body origin.
        #: The origin means something different on every platform: a TurtleBot's is 3 cm up, an
        #: omni base's is a centimetre BELOW the floor, and a walker's pelvis is at 0.9 m. Casting
        #: from the walker's origin sent its rays straight over the top of a 0.88 m robot, which it
        #: then walked into and knocked over. A fixed height sees the things that stand on floors.
        self.height = float(cfg.get("height", 0.30))
        self.clear_time = float(cfg.get("clear_time", 0.5))
        self.ignore = tuple(cfg.get("ignore") or ())
        self._own: set[int] = set()
        self._ignored: set[int] = set()
        self._resolved = False
        self._radius: float | None = None
        self._clear_since: float | None = None
        #: Last verdict, for status reporting and tests.
        self.blocked = False

    @staticmethod
    def validate(cfg: dict | None) -> list[str]:
        cfg = cfg or {}
        errors = []
        mode = cfg.get("on_blocked", "stop")
        if mode in FUTURE_ON_BLOCKED:
            errors.append(
                f"'caution.on_blocked: {mode}' is not implemented; only "
                f"{', '.join(ON_BLOCKED)} is. Traffic is handled by stopping."
            )
        elif mode not in ON_BLOCKED:
            errors.append(f"'caution.on_blocked' must be one of: {', '.join(ON_BLOCKED)}")
        for key in ("lookahead", "width", "rays", "height"):
            if key in cfg and float(cfg[key]) <= 0:
                errors.append(f"'caution.{key}' must be > 0")
        if "clear_time" in cfg and float(cfg["clear_time"]) < 0:
            errors.append("'caution.clear_time' must be >= 0")
        return errors

    def attach(self, ctx, entity) -> None:
        """Learn which geoms are the mover's own.

        Only its own: the entities named in ``ignore`` may be declared *after* this one in the
        document, and components are configured with their owner rather than at the end, so half of
        them would not exist yet. They are resolved in :meth:`resolve_ignored` instead.
        """
        model = ctx.model
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        self._own = subtree_geoms(model, bid) if bid >= 0 else set()

    def resolve_ignored(self, ctx) -> None:
        """Resolve ``ignore`` once every entity in the world has registered itself.

        Called from the navigator's ``on_reset``, which is the first point at which the whole
        document has been configured. Idempotent, so the repeat on every later episode is free.
        """
        if self._resolved:
            return
        model = ctx.model
        ignored: set[int] = set()
        for name in self.ignore:
            other = ctx.entities.get(name)
            if other is None or not other.body:
                known = ", ".join(sorted(e.name for e in ctx.entities.all())) or "(none)"
                raise RuntimeError(
                    f"caution.ignore names {name!r}, which is not an entity in this world. "
                    f"This world has: {known}"
                )
            obid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other.body)
            if obid >= 0:
                ignored |= subtree_geoms(model, obid)
        self._ignored = ignored
        self._resolved = True

    def reset(self) -> None:
        self._clear_since = None
        self.blocked = False

    def check(self, ctx, origin_xy, z: float, pref_vel) -> bool:
        """Whether the mover should hold still this tick.

        Returns ``True`` while blocked, including during the ``clear_time`` settle after the way
        opens again.
        """
        if not self.enabled:
            return False
        speed = float(math.hypot(pref_vel[0], pref_vel[1]))
        if speed < STOPPED:
            # Nothing to run into if we are not going anywhere, and no direction to cast along.
            self.blocked = False
            self._clear_since = None
            return False

        heading = math.atan2(float(pref_vel[1]), float(pref_vel[0]))
        hit = self._nearest_hit(ctx, origin_xy, self.height, heading)
        now = float(ctx.sim_time)
        if hit is not None and hit < self.lookahead:
            self._clear_since = None
            self.blocked = True
            return True
        if self._clear_since is None:
            self._clear_since = now
        self.blocked = (now - self._clear_since) < self.clear_time
        return self.blocked

    def footprint_radius(self, ctx, origin_xy) -> float:
        """The mover's own circumscribed xy radius, measured once from its geoms.

        Yaw-invariant (a bounding sphere per geom), so a rotating body never reports a radius below
        its true extent. Measured rather than configured: a world author should not have to restate
        geometry the model already carries.
        """
        if self._radius is None:
            model, data = ctx.model, ctx.data
            origin = np.asarray(origin_xy, dtype=float)
            self._radius = 0.0
            for g in self._own:
                d = float(np.linalg.norm(data.geom_xpos[g][:2] - origin))
                self._radius = max(self._radius, d + float(model.geom_rbound[g]))
        return self._radius

    def _nearest_hit(self, ctx, origin_xy, z: float, heading: float) -> float | None:
        """Distance to the nearest blocking hit ahead of the mover's own footprint, or ``None``.

        **The ray starts at the front of the footprint, not at the body origin, and that is load
        bearing.** ``mj_multiRay`` reports only the *nearest* hit per ray, so a ray cast from inside
        the mover's own box hits its own far face first -- and discarding that by geom id does not
        reveal what is behind it, it just yields "nothing there". The probe then stayed silent until
        an obstacle came closer than the mover's own half-width, which is well inside the distance it
        was supposed to stop at. (:mod:`roqsim.raycast` documents the same trap for GPU backends:
        "rejecting a hit does not reveal what is behind it".)

        Starting at the footprint edge also gives ``lookahead`` the meaning a world author expects:
        clear space in front of the mover, not from its centre.
        """
        model, data = ctx.model, ctx.data
        half = self.width / 2.0
        # Rays fan across the corridor's width at `lookahead`, so the swept band is the footprint
        # rather than a single centre line -- a robot clipping the edge is still a blocker.
        spread = math.atan2(half, max(self.lookahead, 1e-6))
        offsets = [0.0] if self.rays < 2 else np.linspace(-spread, spread, self.rays)
        dirs = np.array(
            [[math.cos(heading + a), math.sin(heading + a), 0.0] for a in offsets], dtype=float
        )
        nose = self.footprint_radius(ctx, origin_xy) + 1e-3
        origin = np.array(
            [
                float(origin_xy[0]) + math.cos(heading) * nose,
                float(origin_xy[1]) + math.sin(heading) * nose,
                float(z),
            ],
            dtype=float,
        )
        # The default geomgroup mask excludes entities made absent, so a prop nothing can collide
        # with does not stop the mover -- the same rule every other raycaster here follows.
        hits = raycast.cast(model, data, origin, dirs, cutoff=self.lookahead)
        nearest = None
        for dist, gid in zip(hits.dist, hits.geomid, strict=False):
            dist, gid = float(dist), int(gid)
            if dist < 0 or gid < 0 or gid in self._own or gid in self._ignored:
                continue
            if not is_dynamic_body(model, int(model.geom_bodyid[gid])):
                continue  # a wall: the planner's business, not ours
            if nearest is None or dist < nearest:
                nearest = dist
        return nearest
