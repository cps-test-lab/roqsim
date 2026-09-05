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

The response is to **stop** by default. A stopped mover is still on its path and resumes on the
waypoint it was heading for; a re-routing one has quietly changed the trajectory an experiment was
holding fixed, so which of the two you want is the experiment's call and ``on_blocked`` is where you
say it.

``on_blocked: replan`` records the blocker's position and routes around it. What makes that more than
a re-plan of the same path is that the record **persists and decays**: the planner's grid is static,
so a mover that merely re-planned would compute the identical path and drive back into the same
obstacle. A remembered blockage is temporary on purpose -- an obstacle that moved on must stop being
a wall, or a mover would inherit a map of everything that ever got in its way.

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

#: How far above the bottom of a mover's obstacle band to scan. Off the boundary itself, so that a
#: geom whose extent starts exactly at the band's edge is hit rather than grazed.
_SCAN_MARGIN = 0.05

#: Values ``on_blocked`` accepts. ``stop`` holds position and keeps the path; ``replan`` remembers
#: the blocker for ``forget_after`` seconds and routes around it.
ON_BLOCKED = ("stop", "replan")


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
        #: Width of the corridor swept ahead. It is deliberately NOT the mover's measured
        #: footprint: this width also decides how early the mover stops for traffic, so deriving it
        #: from geometry would mean a wider robot silently acquiring more cautious manners, and on a
        #: base of half a metre's radius it turns encounters that resolve by steering into
        #: stand-offs. Set it to the body plus the clearance the mover should keep.
        self.width = float(cfg.get("width", 0.6))
        self.rays = int(cfg.get("rays", 5))
        #: Height above the floor to look at, like a scanning plane -- NOT the mover's body origin.
        #: The origin means something different on every platform: a TurtleBot's is 3 cm up, an
        #: omni base's is a centimetre BELOW the floor, and a walker's pelvis is at 0.9 m. Casting
        #: from the walker's origin sent its rays straight over the top of a 0.88 m robot.
        #:
        #: The height is taken from the bottom of the mover's own ``obstacle_height`` band, the same
        #: declaration the planner rasterizes: scan where the things that stand on floors actually
        #: are. Any *fixed* height is a guess about other people's robots and will be wrong for some
        #: of them -- 0.30 m looked reasonable and was above the roof of a TurtleBot 4, whose
        #: collision geometry stops at 0.25 m, so every probe in a world of them saw nothing at all.
        #: Scanning low cannot have that failure: a mover that cannot pass over something must have
        #: geometry near the floor, or it would not be an obstacle.
        band = cfg.get("band") or (0.1, 1.8)
        self.height = float(cfg.get("height", 0.0)) or float(band[0]) + _SCAN_MARGIN
        self.clear_time = float(cfg.get("clear_time", 0.5))
        #: How long a blockage is treated as traffic that will clear. While the mover is inside
        #: this window it holds AND its progress clock is rebased, so waiting for a robot to pass
        #: never talks it into a recovery. Past it, the blockage is treated as something that is not
        #: going to move, the clock runs, and recovery becomes reachable -- which is the only way out
        #: of a pocket whose exit the mover must drive toward an obstacle to reach.
        self.yield_time = float(cfg.get("yield_time", 3.0))
        #: `replan` only: how long a remembered blockage keeps steering the planner away. Long
        #: enough to get round what is there, short enough that a mover forgets an obstacle which
        #: has since moved on rather than treating the world as permanently narrower.
        self.forget_after = float(cfg.get("forget_after", 5.0))
        #: Radius of the disc a remembered blockage marks out. Defaults to the corridor's own half
        #: width, so a mark is as wide as the gap the mover just failed to fit through.
        #: Radius of the disc a remembered blockage marks out, defaulting to the corridor's own
        #: half width -- a mark as wide as the gap the mover just failed to fit through.
        self.blockage_radius = float(cfg.get("blockage_radius", 0.0)) or self.width / 2.0
        self.ignore = tuple(cfg.get("ignore") or ())
        self._own: set[int] = set()
        self._ignored: set[int] = set()
        self._resolved = False
        self._radius: float | None = None
        self._clear_since: float | None = None
        self._blocked_since: float | None = None
        self._was_blocked = False
        #: Last verdict, for status reporting and tests.
        self.blocked = False
        #: World xy of the NEAREST blocking hit of the last :meth:`check`, or ``None``. What
        #: recovery backs away from -- without it a wedged mover reverses along its own heading,
        #: which is only the right way out when it drove in head-on.
        self.blocker_xy: np.ndarray | None = None
        #: World xy of EVERY blocking hit of the last check, one per ray that found something.
        #: `replan` marks all of them, and that is what lets it discover an obstacle wider than a
        #: single ray: one point tells a planner where the mover stopped, a fan of them outlines how
        #: far the thing that stopped it extends across the corridor.
        self.blocker_points: list = []

    @staticmethod
    def validate(cfg: dict | None) -> list[str]:
        cfg = cfg or {}
        errors = []
        mode = cfg.get("on_blocked", "stop")
        if mode not in ON_BLOCKED:
            errors.append(f"'caution.on_blocked' must be one of: {', '.join(ON_BLOCKED)}")
        for key in ("lookahead", "width", "rays", "height", "blockage_radius"):
            if key in cfg and float(cfg[key]) <= 0:
                errors.append(f"'caution.{key}' must be > 0")
        for key in ("clear_time", "forget_after", "yield_time"):
            if key in cfg and float(cfg[key]) < 0:
                errors.append(f"'caution.{key}' must be >= 0")
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
        self._blocked_since = None
        self._was_blocked = False
        self.blocked = False
        self.blocker_xy = None
        self.blocker_points = []

    def blocked_for(self, now: float) -> float:
        """How long the way has been blocked without a break, in seconds. ``0.0`` when clear."""
        return 0.0 if self._blocked_since is None else max(0.0, float(now) - self._blocked_since)

    def yielding(self, now: float) -> bool:
        """Whether this block should still be read as traffic rather than as an obstacle."""
        return self.blocked_for(now) < self.yield_time

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
            self._blocked_since = None
            self.blocker_xy = None
            self.blocker_points = []
            return False

        heading = math.atan2(float(pref_vel[1]), float(pref_vel[0]))
        nearest, points = self._hits(ctx, origin_xy, self.height, heading)
        now = float(ctx.sim_time)
        if nearest is not None and nearest[0] < self.lookahead:
            self._clear_since = None
            if self._blocked_since is None:
                self._blocked_since = now
            self._was_blocked = True
            self.blocked = True
            self.blocker_xy = nearest[1]
            self.blocker_points = points
            return True
        self.blocker_xy = None
        self.blocker_points = []
        if self._clear_since is None:
            # Never blocked, so there is nothing to settle after. Without this a mover holds for
            # `clear_time` at the start of every episode -- the timer would start from a cold
            # `None` and read as "the way only just cleared" when in fact it was never shut.
            self._clear_since = now
            if not self._was_blocked:
                self.blocked = False
                return False
        self.blocked = (now - self._clear_since) < self.clear_time
        if not self.blocked:
            # Only once the way is open AND has stayed open: a clear ray during the settle is not
            # the same as being through, and treating it as such re-arms the yield window on every
            # flicker -- which is exactly the state a mover stuttering against an obstacle is in.
            self._blocked_since = None
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

    def _hits(self, ctx, origin_xy, z: float, heading: float):
        """``((distance, world_xy) | None, [world_xy, ...])`` for the corridor ahead.

        The first is the nearest blocking hit; the second is every one of them, one per ray that
        found something.

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
        # PARALLEL rays offset across the corridor, not a fan diverging from one point. A fan only
        # spans the full width at `lookahead` and pinches to nothing at the nose, so it is blind to
        # exactly what a body scrapes against: something beside the mover's shoulder, a few
        # centimetres ahead. Offsetting instead sweeps the true rectangle the body is about to
        # occupy, at every distance along it.
        nose = self.footprint_radius(ctx, origin_xy) + 1e-3
        ahead = np.array([math.cos(heading), math.sin(heading)])
        side = np.array([-ahead[1], ahead[0]])
        lateral = [0.0] if self.rays < 2 else np.linspace(-half, half, self.rays)
        base = np.asarray(origin_xy, dtype=float) + ahead * nose
        origins = np.array([[*(base + side * o), float(z)] for o in lateral], dtype=float)
        direction = np.array([[ahead[0], ahead[1], 0.0]], dtype=float)
        # The default geomgroup mask excludes entities made absent, so a prop nothing can collide
        # with does not stop the mover -- the same rule every other raycaster here follows.
        # One `mj_multiRay` per origin -- `cast_many`'s docstring records that the API takes a
        # single origin, so a swept corridor is irreducibly one call per lateral offset. At five
        # rays on a 20 Hz tick that is well inside the budget, and it is only paid while moving.
        hits = raycast.cast_many(model, data, origins, direction, cutoff=self.lookahead)
        nearest, points = None, []
        for i, (dist, gid) in enumerate(
            zip(hits.dist.reshape(-1), hits.geomid.reshape(-1), strict=False)
        ):
            dist, gid = float(dist), int(gid)
            if dist < 0 or gid < 0 or gid in self._own or gid in self._ignored:
                continue
            if not is_dynamic_body(model, int(model.geom_bodyid[gid])):
                continue  # a wall: the planner's business, not ours
            point = origins[i][:2] + ahead * dist
            points.append(point)
            if nearest is None or dist < nearest[0]:
                nearest = (dist, point)
        return nearest, points
