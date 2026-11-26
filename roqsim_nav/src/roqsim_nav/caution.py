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
holding fixed, so which of the two you want is the experiment's call and ``reroute`` is where you
say it.

``reroute`` records the blocker's position and plans around it. What makes that more than
a re-plan of the same path is that the record **persists and decays**: the planner's grid is static,
so a mover that merely re-planned would compute the identical path and drive back into the same
obstacle. A remembered blockage is temporary on purpose -- an obstacle that moved on must stop being
a wall, or a mover would inherit a map of everything that ever got in its way.

**Setting off again is judged against the body, not the rays.** The rays are a cheap, early answer
to "is something coming into my way", and they are blind in two places a held mover's body is not:
at every height but one, and between the mover's front and the ray origins, which sit on its
circumscribed radius (a walker's is its pelvis capsule's bounding sphere, 0.34 m, where its body
ends at 0.16 m). A robot whose base is lower than the scan plane shows the rays only its tower,
narrower than their spacing; one standing beside the mover's front shows them nothing. Either can
stand in the mover's way while every ray reads clear, and a mover that sets off on the rays alone
walks into it. So before a held mover sets off, :meth:`CautionProbe._sweep` also asks whether the
corridor it held for is empty, filled in rather than sampled: ``width`` across, the same width it
stops for, from its body's front to the end of the probe's reach, and from the bottom of its own
body to its top (the floor, for a mover standing on it). Nothing that moves may reach into it, each
thing measured by an extent never smaller than itself, so a mover that passes this check sets off
with at least the clearance it stops for; what moves into the way afterwards is for the rays, as
before. The check runs only to end a hold of the rays, so a mover they never held moves exactly as
it did. A robot that drives across a mover's way and on is held for until its body, not only the
part the rays see, has left the corridor.

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
from roqsim.presence import entity_geoms, subtree_geom_ids

from .control import STOPPED

#: How far above the bottom of a mover's obstacle band to scan. Off the boundary itself, so that a
#: geom whose extent starts exactly at the band's edge is hit rather than grazed.
_SCAN_MARGIN = 0.05

#: The probe's own tuning, as it appears in a navigator's ``avoidance:`` block. Named so the
#: navigator can tell a probe setting from one of the three capabilities and refuse a typo in
#: either, rather than silently accepting a key nothing reads.
_TUNING = (
    "lookahead",
    "width",
    "rays",
    "height",
    "clear_time",
    "yield_time",
    "forget_after",
    "blockage_radius",
    "ignore",
)


def subtree_geoms(model, body_id: int) -> set[int]:
    """Every geom of ``body_id`` and its descendants.

    Needed because ``mj_multiRay``'s ``bodyexclude`` takes a single body: a robot is a base link plus
    wheels plus sensor mounts, so excluding only its root would leave it seeing its own wheels a few
    centimetres ahead and concluding it was permanently blocked.
    """
    return set(subtree_geom_ids(model, body_id))


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
        # Set by the navigator from its `avoidance: {stop: ...}`, not written in a world directly.
        self.enabled = bool(cfg.get("enabled", True))
        self.reroute = bool(cfg.get("reroute", False))
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
        #: from the walker's origin sends its rays straight over the top of a 0.88 m robot.
        #:
        #: The height is taken from the bottom of the mover's own ``obstacle_height`` band, the same
        #: declaration the planner rasterizes: scan where the things that stand on floors actually
        #: are. Any *fixed* height is a guess about other people's robots and will be wrong for some
        #: of them -- 0.30 m looks reasonable and is above the roof of a TurtleBot 4, whose
        #: collision geometry stops at 0.25 m, so every probe in a world of them would see nothing.
        #: Scanning low cannot have that failure: a mover that cannot pass over something must have
        #: geometry near the floor, or it would not be an obstacle.
        band = cfg.get("band") or (0.1, 1.8)
        height = cfg.get("height")
        self.height = float(band[0]) + _SCAN_MARGIN if height is None else float(height)
        self.clear_time = float(cfg.get("clear_time", 0.5))
        #: How long a blockage is treated as traffic that will clear. While the mover is inside
        #: this window it holds AND its progress clock is rebased, so waiting for a robot to pass
        #: never talks it into a recovery. Past it, the blockage is treated as something that is not
        #: going to move, the clock runs, and recovery becomes reachable -- which is the only way out
        #: of a pocket whose exit the mover must drive toward an obstacle to reach.
        self.yield_time = float(cfg.get("yield_time", 3.0))
        #: `reroute` only: how long a remembered blockage keeps steering the planner away. Long
        #: enough to get round what is there, short enough that a mover forgets an obstacle which
        #: has since moved on rather than treating the world as permanently narrower.
        self.forget_after = float(cfg.get("forget_after", 5.0))
        #: Radius of the disc a remembered blockage marks out, defaulting to the corridor's own
        #: half width -- a mark as wide as the gap the mover just failed to fit through.
        radius = cfg.get("blockage_radius")
        self.blockage_radius = self.width / 2.0 if radius is None else float(radius)
        self.ignore = tuple(cfg.get("ignore") or ())
        self._own: set[int] = set()
        self._footprint: set[int] = set()
        self._ignored: set[int] = set()
        self._resolved = False
        self._radius: float | None = None
        self._clear_since: float | None = None
        self._blocked_since: float | None = None
        self._was_blocked = False
        #: The mover's own geoms, as an array for :meth:`_sweep`; which of them collide is read
        #: when it runs, because presence switches a geom's collision off and on at runtime.
        self._body = np.zeros(0, dtype=int)
        #: Geoms of everything else that moves and is not ignored: what the sweep looks for. Built on
        #: first use, after ``ignore`` is resolved.
        self._others: np.ndarray | None = None
        #: Whether the last check held the mover for the filled-in corridor rather than for a ray.
        #: While it does, the corridor is looked at every tick, so the settle runs from the moment
        #: it cleared rather than from whichever tick it happened to be looked at.
        self._sweep_held = False
        #: Whether a hold the rays began is still to be ended: from the rays' hit until a settle
        #: ends with the filled-in corridor (:meth:`_sweep`) clear. A settle that the corridor
        #: restarted is looked at again when it ends, so something that left the corridor and came
        #: back meanwhile holds the mover again. The settle a mover keeps once at the start of a
        #: move is no hold of the rays, and is not looked at this way.
        self._ray_held = False
        #: Last verdict, for status reporting and tests.
        self.blocked = False
        #: World xy of the NEAREST blocking hit of the last :meth:`check`, or ``None``. What
        #: recovery backs away from -- without it a wedged mover reverses along its own heading,
        #: which is only the right way out when it drove in head-on.
        self.blocker_xy: np.ndarray | None = None
        #: World xy of EVERY blocking hit of the last check, one per ray that found something.
        #: `reroute` marks all of them, and that is what lets it discover an obstacle wider than a
        #: single ray: one point tells a planner where the mover stopped, a fan of them outlines how
        #: far the thing that stopped it extends across the corridor.
        self.blocker_points: list = []

    #: What this probe reads out of an ``avoidance:`` block, for the navigator to validate against.
    KEYS = _TUNING

    @staticmethod
    def validate(cfg: dict | None) -> list[str]:
        """The ranges of the tuning in *cfg*. A value that is not a number is the schema's to report."""
        numbers = {
            k: v
            for k, v in (cfg or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        errors = []
        for key in ("lookahead", "width", "rays", "height", "blockage_radius"):
            if key in numbers and numbers[key] <= 0:
                errors.append(f"'avoidance.{key}' must be > 0")
        for key in ("clear_time", "forget_after", "yield_time"):
            if key in numbers and numbers[key] < 0:
                errors.append(f"'avoidance.{key}' must be >= 0")
        return errors

    def attach(self, ctx, entity) -> None:
        """Learn which geoms are the mover's own.

        Only its own: the entities named in ``ignore`` may be declared *after* this one in the
        document, and components are configured with their owner rather than at the end, so half of
        them would not exist yet. They are resolved in :meth:`resolve_ignored` instead.
        """
        # The whole mover: a walker's limbs are roots of their own (Entity.bodies), and a probe
        # that saw them would find its own arm in the way.
        self._own = set(entity_geoms(ctx.model, entity, strict=False))
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body or "")
        self._footprint = subtree_geoms(ctx.model, bid) if bid >= 0 else set()
        self._body = np.array(sorted(self._own), dtype=int)
        self._others = None

    def resolve_ignored(self, ctx) -> None:
        """Resolve ``ignore`` once every entity in the world has registered itself.

        Called from the navigator's ``on_reset``, which is the first point at which the whole
        document has been configured. Idempotent, so the repeat on every later run is free.
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
            # The whole entity, so `ignore: [walker]` takes a walker's limbs with its pelvis.
            ignored |= set(entity_geoms(model, other, strict=False))
        self._ignored = ignored
        self._others = None
        self._resolved = True

    def reset(self) -> None:
        self._clear_since = None
        self._blocked_since = None
        self._was_blocked = False
        self._sweep_held = False
        self._ray_held = False
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
        opens again. A settle after a hold of the rays also ends only with the filled-in corridor
        (:meth:`_sweep`) clear: while something reaches into it, the mover holds, and the settle
        runs again from the moment it is clear.
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
            self._ray_held = True
            self.blocked = True
            self.blocker_xy = nearest[1]
            self.blocker_points = points
            return True
        self.blocker_xy = None
        self.blocker_points = []
        if self._clear_since is None:
            # Never blocked, so there is nothing to settle after. Without this a mover holds for
            # `clear_time` at the start of every run -- the timer would start from a cold
            # `None` and read as "the way only just cleared" when in fact it was never shut.
            self._clear_since = now
            if not self._was_blocked:
                self.blocked = False
                return False
        self.blocked = (now - self._clear_since) < self.clear_time
        if self._sweep_held or (self._ray_held and not self.blocked):
            # About to set off after the rays held it, or held for the corridor already: the rays
            # have said it is open, and they cannot see everything the body would meet (the module
            # docstring). Only then: a mover under way, or one the rays never held, is not looked
            # at this way, so it walks as before.
            points = self._sweep(ctx, origin_xy, heading)
            self._sweep_held = bool(points)
            if points:
                self._clear_since = now
                self.blocked = True
                self.blocker_points = points
                self.blocker_xy = min(
                    points, key=lambda p: float(np.linalg.norm(p - np.asarray(origin_xy)))
                )
                return True
            if not self.blocked:
                self._ray_held = False
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
        geometry the model already carries. Measured over the root body's subtree: the other roots
        an entity owns (``Entity.bodies``, a walker's limbs) are its own to the probe's rays but not
        part of the footprint.
        """
        if self._radius is None:
            model, data = ctx.model, ctx.data
            origin = np.asarray(origin_xy, dtype=float)
            self._radius = 0.0
            for g in self._footprint:
                d = float(np.linalg.norm(data.geom_xpos[g][:2] - origin))
                self._radius = max(self._radius, d + float(model.geom_rbound[g]))
        return self._radius

    def _sweep(self, ctx, origin_xy, heading: float) -> list:
        """World xy of what lies in the corridor the mover holds for; ``[]`` if none.

        One point per geom found, where its extent comes nearest the mover's centre.

        The corridor is the one the rays sample, filled in: in the mover's heading frame, ``width``
        across (the body plus the clearance it keeps, as for stopping), from the front of the
        mover's collidable geoms to the end of the probe's reach (its radius plus ``lookahead``),
        and in height everything those geoms span, from their bottom (the floor, for a mover
        standing on it) up. From its front, not its centre: what stands on the mover, or beside its
        flank, is not ahead of it. Setting off is judged by the same width as stopping, so a mover
        does not leave a hold to pass closer to something than it would have stopped for.

        What is looked for is every collidable geom of a body with degrees of freedom or a mocap
        body -- the bodies the rays stop for -- other than the mover's own, the ignored ones and what
        the mover holds (a body an active weld or connect ties to one of its own, which moves with
        it). Each is taken by its extent along the frame's three axes, exact for a sphere, capsule,
        cylinder and box and the bounding box for any other shape, which can find one in the
        corridor that is not, never miss one that is. It counts when it reaches into the corridor
        by more than a millimetre, so that what rests flush on the mover is not in its way.

        A mover none of whose geoms can collide meets nothing, and is never held by this.
        """
        model, data = ctx.model, ctx.data
        body = self._body[(model.geom_contype[self._body] | model.geom_conaffinity[self._body]) > 0]
        others = self._candidates(model)
        if body.size == 0 or others.size == 0:
            return []
        others = others[
            ((model.geom_contype[others] | model.geom_conaffinity[others]) > 0)
            & (raycast.VISIBLE_GROUPS[model.geom_group[others]] > 0)
        ]
        held = _held_bodies(model, data, self._body)
        if held:
            others = others[~np.isin(model.geom_bodyid[others], list(held))]
        if others.size == 0:
            return []
        cos, sin = math.cos(heading), math.sin(heading)
        axes = np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])
        origin = np.array([float(origin_xy[0]), float(origin_xy[1]), 0.0])
        own_c, own_r = _extents(model, data, body, axes, origin)
        half = self.width / 2.0
        own_lo, own_hi = own_c - own_r, own_c + own_r
        reach = self.footprint_radius(ctx, origin_xy) + self.lookahead
        lo = np.array([own_hi[:, 0].max(), -half, own_lo[:, 2].min()])
        hi = np.array([reach, half, own_hi[:, 2].max()])
        c, r = _extents(model, data, others, axes, origin)
        meets = np.all((c + r > lo + _TOUCH) & (c - r < hi - _TOUCH), axis=1)
        # Where each is nearest the mover, not its centre: what `reroute` marks and recovery backs
        # away from. A long barrier's centre is not where the mover meets it.
        near = np.clip(np.zeros(3), c[meets] - r[meets], c[meets] + r[meets])
        return [origin[:2] + p[0] * axes[0, :2] + p[1] * axes[1, :2] for p in near]

    def _candidates(self, model) -> np.ndarray:
        """Every geom of a body that moves, other than the mover's own and the ignored ones."""
        if self._others is None:
            skip = self._own | self._ignored
            self._others = np.array(
                [
                    g
                    for g in range(model.ngeom)
                    if g not in skip and is_dynamic_body(model, int(model.geom_bodyid[g]))
                ],
                dtype=int,
            )
        return self._others

    def _hits(self, ctx, origin_xy, z: float, heading: float):
        """``((distance, world_xy) | None, [world_xy, ...])`` for the corridor ahead.

        The first is the nearest blocking hit; the second is every one of them, one per ray that
        found something.

        **The ray starts at the front of the footprint, not at the body origin, and that is load
        bearing.** ``mj_multiRay`` reports only the *nearest* hit per ray, so a ray cast from inside
        the mover's own box hits its own far face first -- and discarding that by geom id does not
        reveal what is behind it, it just yields "nothing there". Such a probe stays silent until
        an obstacle comes closer than the mover's own half-width, which is well inside the distance
        it is supposed to stop at. (:mod:`roqsim.raycast` documents the same trap for GPU backends:
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


#: How far two extents must overlap to meet: a millimetre, so that what rests flush on a mover's top
#: is not counted as in its way.
_TOUCH = 1e-3


def _held_bodies(model, data, own_geoms) -> set[int]:
    """Bodies an active weld or connect ties to one of the mover's own bodies: what it holds."""
    if model.neq == 0:
        return set()
    own = {int(model.geom_bodyid[g]) for g in own_geoms}
    held = set()
    for i in np.flatnonzero(data.eq_active):
        if int(model.eq_type[i]) not in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            continue
        if int(model.eq_objtype[i]) == mujoco.mjtObj.mjOBJ_SITE:
            a = int(model.site_bodyid[model.eq_obj1id[i]])
            b = int(model.site_bodyid[model.eq_obj2id[i]])
        else:
            a, b = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        if a in own and b not in own:
            held.add(b)
        elif b in own and a not in own:
            held.add(a)
    # A held body takes what hangs from it along: a tote's lid, a tray's load.
    roots = {int(model.body_rootid[b]) for b in held}
    return {b for b in range(model.nbody) if int(model.body_rootid[b]) in roots and b != 0}


def _extents(model, data, geoms, axes, origin):
    """Centre and half extent of each geom along ``axes``, measured from ``origin``.

    Exact for a sphere, a capsule, a cylinder and a box: a capsule reaches its half length times how
    far its axis leans onto the frame's axis, plus its radius; a cylinder its half length so, plus its
    radius times how far it leans off; a box the sum of its half sizes weighted by how far each of its
    axes leans on. Any other shape is taken as its bounding box in its own frame (``geom_aabb``) the
    same way a box is, which can be larger than it, never smaller.
    """
    rot = data.geom_xmat[geoms].reshape(-1, 3, 3)
    size = model.geom_size[geoms]
    gtype = model.geom_type[geoms]
    aabb = model.geom_aabb[geoms]
    centre = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rot, aabb[:, :3])
    lean = np.abs(np.einsum("kj,nji->nki", axes, rot))  # (n, frame axis, geom axis)
    half = np.einsum("nki,ni->nk", lean, aabb[:, 3:])
    sphere = gtype == mujoco.mjtGeom.mjGEOM_SPHERE
    half[sphere] = size[sphere, :1]
    capsule = gtype == mujoco.mjtGeom.mjGEOM_CAPSULE
    half[capsule] = lean[capsule, :, 2] * size[capsule, 1:2] + size[capsule, :1]
    cyl = gtype == mujoco.mjtGeom.mjGEOM_CYLINDER
    off = np.sqrt(np.clip(1.0 - lean[cyl, :, 2] ** 2, 0.0, 1.0))
    half[cyl] = lean[cyl, :, 2] * size[cyl, 1:2] + off * size[cyl, :1]
    box = gtype == mujoco.mjtGeom.mjGEOM_BOX
    half[box] = np.einsum("nki,ni->nk", lean[box], size[box])
    return (centre - origin) @ axes.T, half
