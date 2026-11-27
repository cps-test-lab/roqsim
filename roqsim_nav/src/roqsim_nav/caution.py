"""Stopping for what the planner cannot see.

The planner's grid holds the world's **static walls and nothing else** -- the robot under test, a
pedestrian, another navigated prop are all deliberately absent from it, because a grid rasterized
once cannot represent something that moves. So a mover following a planned path needs a second,
faster layer that looks where it is actually going.

The invariant, and it is the complement of the grid's:

    **Caution sees exactly what the planner cannot.** A geom counts when it belongs to neither the
    mover itself nor to static geometry -- that is, when its body has degrees of freedom or is a
    mocap body. Those are precisely the bodies :func:`~roqsim_nav.obstacles.wall_polygons` refuses
    to rasterize. A wall ahead at a corner is the planner's problem and never this layer's, or a
    mover would stop dead every time its path hugged one.

The response is to **stop** by default. A stopped mover is still on its path and resumes on the
waypoint it was heading for; a re-routing one has quietly changed the trajectory an experiment was
holding fixed, so which of the two you want is the experiment's call and ``reroute`` is where you
say it.

``reroute`` records the blocker's position and plans around it. What makes that more than
a re-plan of the same path is that the record **persists and decays**: the planner's grid is static,
so a mover that merely re-planned would compute the identical path and drive back into the same
obstacle. A remembered blockage is temporary on purpose -- an obstacle that moved on must stop being
a wall, or a mover would inherit a map of everything that ever got in its way.

**What is in the way is judged against the body, filled in, every tick.** The probe looks at the
corridor the mover's body is about to occupy (:meth:`CautionProbe._ahead`): ``width`` across, from
its body's front to ``lookahead`` beyond it, from the bottom of its body to its top, laid along the
path it is about to walk -- bent where the path turns, not straight along its heading. A thing that
moves counts when any of its geoms, visual ones included, reaches up into the mover's own obstacle
band (``obstacle_height``, the band its planner rasterizes walls in): a robot's tower is part of the
robot, and what sees the tower sees the whole robot, while a mat the mover walks over is not in its
way. Where it is in the corridor is where its body is -- its collidable geoms, each taken by its
extent along the corridor's axes: a robot's base, lower than the band's floor; a walker's legs and
hips, not its swinging arms. A thing nothing of which collides is where its visible geoms are.
What is in the way is what reaches past the body's front: a thing already inside the body that
reaches past it is in the way at once, and one beside the body -- a walker abreast, a shoulder
touching -- is not.
What the mover holds (a body an active weld or connect ties to one of its own) is never in its way.

Because the same corridor decides holding and setting off, a held mover sets off only once nothing
that moves reaches into it, with at least the clearance it stops for.

Cost matters here, because this runs on the same thread as everything else: per tick it is one
vectorised pass over the geoms of what moves for each leg of the corridor, and none at all when the
mover is not moving.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim import raycast
from roqsim.presence import entity_geoms, subtree_geom_ids

from .control import STOPPED

#: The probe's own tuning, as it appears in a navigator's ``avoidance:`` block. Named so the
#: navigator can tell a probe setting from one of the three capabilities and refuse a typo in
#: either, rather than silently accepting a key nothing reads.
_TUNING = (
    "lookahead",
    "width",
    "clear_time",
    "yield_time",
    "forget_after",
    "blockage_radius",
    "ignore",
)

#: Keys the probe read when it looked along rays, and why nothing reads them now: a world that sets
#: one is refused with this, rather than having it silently ignored.
RETIRED = {
    "rays": "the probe no longer samples its corridor with rays; it fills the corridor in, so "
    "there is no ray count to set",
    "height": "the probe no longer scans at one height; it looks at the whole height of the mover's "
    "body, and counts what reaches into its obstacle_height band",
}

#: How many legs of the path ahead the corridor follows at most.
_MAX_LEGS = 4

#: Where across the corridor, as fractions of its half width, a thing in it reports points.
_ACROSS = np.linspace(-1.0, 1.0, 5)

#: How far back a thing's velocity is measured: its own parts move as it walks (a walker's legs),
#: and a velocity taken over one tick would read a stride as a dash.
_VELOCITY_WINDOW = 0.3

#: What moves along the mover's way at least this share of its speed is going away as fast as the
#: mover comes, and is not in its way: walkers side by side, or one a step behind another.
_ALONG = 0.8


def subtree_geoms(model, body_id: int) -> set[int]:
    """Every geom of ``body_id`` and its descendants.

    A mover's footprint is its root body's subtree: a robot is a base link plus wheels plus sensor
    mounts, and a footprint of its root alone would leave its wheels outside it.
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

    ``lookahead`` is how much clear corridor it needs ahead of its body, ``width`` the corridor's
    width (normally the mover's footprint plus the clearance it keeps). ``clear_time`` is how long
    the way must stay clear before moving off again, so a robot crossing tangentially does not make
    the mover stutter forward into its path.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        # Set by the navigator from its `avoidance: {stop: ...}`, not written in a world directly.
        self.enabled = bool(cfg.get("enabled", True))
        self.reroute = bool(cfg.get("reroute", False))
        self.lookahead = float(cfg.get("lookahead", 1.2))
        #: Width of the corridor. It is deliberately NOT the mover's measured footprint: this width
        #: also decides how early the mover stops for traffic, so deriving it from geometry would
        #: mean a wider robot silently acquiring more cautious manners, and on a base of half a
        #: metre's radius it turns encounters that resolve by steering into stand-offs. Set it to
        #: the body plus the clearance the mover should keep.
        self.width = float(cfg.get("width", 0.6))
        #: The mover's own obstacle band, the declaration its planner rasterizes walls in: a thing
        #: is in its way when any of its geoms reaches above the band's floor.
        band = cfg.get("band") or (0.1, 1.8)
        self.band = (float(band[0]), float(band[1]))
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
        #: The mover's own geoms, as an array; which of them collide is read when it runs, because
        #: presence switches a geom's collision off and on at runtime.
        self._body = np.zeros(0, dtype=int)
        #: Geoms of everything else that moves and is not ignored, and for each the thing it belongs
        #: to (an entity, or the root body of what is no entity): what the corridor looks for. Built
        #: on first use, after ``ignore`` is resolved.
        self._others: np.ndarray | None = None
        self._owner: np.ndarray | None = None
        self._entity_of: dict[int, int] = {}
        #: The root body of each thing, where its velocity is measured, and where each thing that
        #: moves was: thing -> [(t, xy), ...].
        self._owner_body: dict[int, int] = {}
        self._track: dict[int, list] = {}
        #: The mover's body's extent in the frame of its root body (lo, hi), measured the first time
        #: it looks and kept (:meth:`_front_along`).
        self._front: tuple | None = None
        self._root: int | None = None
        #: Last verdict, for status reporting and tests.
        self.blocked = False
        #: The distance from the mover's front to the nearest thing in its corridor along its path,
        #: of the last :meth:`check`, or ``None`` when nothing is within its reach.
        self.gap: float | None = None
        #: World xy of the NEAREST thing in the way of the last :meth:`check`, or ``None``. What
        #: recovery backs away from -- without it a wedged mover reverses along its own heading,
        #: which is only the right way out when it drove in head-on.
        self.blocker_xy: np.ndarray | None = None
        #: World xy of EVERY geom in the way of the last check, where each comes nearest the
        #: mover. `reroute` marks all of them, which outlines how far the thing that stopped it
        #: extends across the corridor.
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
        for key in ("lookahead", "width", "blockage_radius"):
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
        self._root = bid
        self._body = np.array(sorted(self._own), dtype=int)
        self._others = None
        self._front = None

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
        # Which thing each geom belongs to: a walker's limbs are roots of their own, and are one
        # walker for what counts as in the way.
        self._entity_of = {}
        self._owner_body = {}
        for k, other in enumerate(ctx.entities.all()):
            if other.body:
                for g in entity_geoms(model, other, strict=False):
                    self._entity_of.setdefault(int(g), k)
                self._owner_body[k] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other.body)
        self._others = None
        self._resolved = True

    def reset(self) -> None:
        self._clear_since = None
        self._blocked_since = None
        self._was_blocked = False
        self.blocked = False
        self.gap = None
        self.blocker_xy = None
        self.blocker_points = []
        self._front = None  # measured again at the run's first look
        self._track = {}

    def leg_ended(self) -> None:
        """The mover stands with nowhere to go -- a dwell, a goal reached -- so what a hold left is
        over: its next leg starts as a mover that was never held, with no settle. Called by its
        navigator, which knows the difference between that and a tick an avoidance model leaves
        without speed, which ends no hold."""
        self._was_blocked = False
        self._clear_since = None

    def blocked_for(self, now: float) -> float:
        """How long the way has been blocked without a break, in seconds. ``0.0`` when clear."""
        return 0.0 if self._blocked_since is None else max(0.0, float(now) - self._blocked_since)

    def yielding(self, now: float) -> bool:
        """Whether this block should still be read as traffic rather than as an obstacle."""
        return self.blocked_for(now) < self.yield_time

    def check(self, ctx, origin_xy, z: float, pref_vel, path=None) -> bool:
        """Whether the mover should hold still this tick.

        ``path`` is the route ahead as world xy points, the next first; the corridor is laid along
        it, and straight along ``pref_vel`` without one. Returns ``True`` while something is in
        the corridor, and during the ``clear_time`` settle after it has left.
        """
        if not self.enabled:
            return False
        speed = float(math.hypot(pref_vel[0], pref_vel[1]))
        if speed < STOPPED:
            # Nothing to run into if we are not going anywhere, and no direction to look along.
            self.blocked = False
            self._clear_since = None
            self._blocked_since = None
            self.gap = None
            self.blocker_xy = None
            self.blocker_points = []
            return False

        heading = math.atan2(float(pref_vel[1]), float(pref_vel[0]))
        gap, points = self._ahead(
            ctx, origin_xy, heading, path, self.lookahead, now=float(ctx.sim_time), speed=speed
        )
        self.gap = gap
        now = float(ctx.sim_time)
        if gap is not None:
            self._clear_since = None
            if self._blocked_since is None:
                self._blocked_since = now
            self._was_blocked = True
            self.blocked = True
            self.blocker_points = points
            self.blocker_xy = min(
                points, key=lambda p: float(np.linalg.norm(p - np.asarray(origin_xy)))
            )
            return True
        self.blocker_xy = None
        self.blocker_points = []
        if not self._was_blocked:
            # Never blocked, so there is nothing to settle after, on this tick or any other until
            # it is. Without this a mover holds for `clear_time` at the start of every run -- the
            # timer would start on its first tick and read as "the way only just cleared" from the
            # second, when in fact it was never shut.
            self._clear_since = now
            self.blocked = False
            return False
        if self._clear_since is None:
            self._clear_since = now
        self.blocked = (now - self._clear_since) < self.clear_time
        if not self.blocked:
            # Only once the way is open AND has stayed open: a clear tick during the settle is not
            # the same as being through, and treating it as such re-arms the yield window on every
            # flicker -- which is exactly the state a mover stuttering against an obstacle is in.
            self._blocked_since = None
        return self.blocked

    def footprint_radius(self, ctx, origin_xy) -> float:
        """The mover's own circumscribed xy radius, measured once from its geoms.

        Yaw-invariant (a bounding sphere per geom), so a rotating body never reports a radius below
        its true extent. Measured rather than configured: a world author should not have to restate
        geometry the model already carries. Measured over the root body's subtree: the other roots
        an entity owns (``Entity.bodies``, a walker's limbs) are its own to the probe but not part
        of the footprint.
        """
        if self._radius is None:
            model, data = ctx.model, ctx.data
            origin = np.asarray(origin_xy, dtype=float)
            self._radius = 0.0
            for g in self._footprint:
                d = float(np.linalg.norm(data.geom_xpos[g][:2] - origin))
                self._radius = max(self._radius, d + float(model.geom_rbound[g]))
        return self._radius

    def _ahead(
        self, ctx, origin_xy, heading: float, path, reach: float, now: float = 0.0, speed=None
    ):
        """``(gap, points)``: what lies in the corridor ahead, within ``reach`` of the front.

        ``gap`` is the distance along the corridor from the mover's front to the nearest thing in
        it (0 for one already inside the body that reaches past its front), ``None`` when there is
        none; ``points``
        is the world xy where each geom found comes nearest the mover, and where it spans the
        corridor at five points across it.

        The corridor is ``width`` across and spans the height of the mover's own body, from its
        bottom (the floor, for a mover standing on it) to its top. Its first leg runs from the
        front of the mover's body along ``heading``, or toward the first point of ``path`` when
        that is within a right angle of the heading; it then follows ``path`` point to point, up to
        :data:`_MAX_LEGS` legs, until it is ``reach`` long. Each geom of what moves is taken by its
        extent along each leg's axes (:func:`_extents`), which is exact for a sphere, capsule,
        cylinder and box and the bounding box for any other shape, so it can find one in the
        corridor that is not, never miss one that is. A geom counts when it reaches into a leg by
        more than a millimetre and its thing reaches above the floor of the mover's obstacle band.
        """
        model, data = ctx.model, ctx.data
        own = self._body
        body = own[(model.geom_contype[own] | model.geom_conaffinity[own]) > 0]
        if body.size == 0:
            body = own
        others, owners = self._candidates(model)
        if body.size == 0 or others.size == 0:
            return None, []
        visible = raycast.VISIBLE_GROUPS[model.geom_group[others]] > 0
        others, owners = others[visible], owners[visible]
        held = _held_bodies(model, data, self._body)
        if held:
            keep = ~np.isin(model.geom_bodyid[others], list(held))
            others, owners = others[keep], owners[keep]
        if others.size == 0:
            return None, []
        start = np.array([float(origin_xy[0]), float(origin_xy[1])])
        floor = np.array([start[0], start[1], 0.0])
        half = self.width / 2.0
        # The things that reach into the band: a thing below its floor is walked or driven over.
        world = np.eye(3)
        wc, wr = _extents(model, data, others, world, floor)
        tops = {}
        for o, top in zip(owners.tolist(), (wc[:, 2] + wr[:, 2]).tolist(), strict=True):
            tops[o] = max(tops.get(o, -np.inf), top)
        counts = np.array([tops[o] > self.band[0] + _TOUCH for o in owners.tolist()])
        # Where each thing in the band is, every tick, for its velocity: where its root body is, not
        # the middle of its geoms, which a walker's swinging arms move about.
        velocity = {}
        for o in set(owners[counts].tolist()):
            velocity[o] = self._velocity(o, data.xpos[self._owner_body[o]][:2], now)
        for o in list(self._track):
            if o not in velocity:
                del self._track[o]
        others, owners = others[counts], owners[counts]
        # Where a thing is in the corridor is where its body is, what can collide: a robot's base,
        # a walker's legs and hips, not its swinging arms. A thing nothing of which collides is
        # where its visible geoms are.
        solid = (model.geom_contype[others] | model.geom_conaffinity[others]) > 0
        has_solid = {o for o, sd in zip(owners.tolist(), solid.tolist(), strict=True) if sd}
        keep = solid | ~np.isin(owners, list(has_solid))
        others, owners = others[keep], owners[keep]
        if others.size == 0:
            return None, []
        legs = self._legs(start, heading, path)
        best, points = None, []
        covered = 0.0  # corridor length behind the leg being looked at, from the mover's front
        for k, (a, b) in enumerate(legs):
            d = b - a
            length = float(np.linalg.norm(d))
            if length < 1e-9:
                continue
            cos, sin = d[0] / length, d[1] / length
            axes = np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])
            o3 = np.array([a[0], a[1], 0.0])
            c, r = _extents(model, data, others, axes, o3)
            if k == 0:
                front, z_lo, z_hi = self._front_along(model, data, body, start, heading_of(axes))
                # From the body's front: what is in the way of its motion ahead. A thing already
                # inside the body that reaches past its front is in its way at once (its gap is
                # 0); one beside it, a walker abreast with a touching shoulder, is not ahead of it.
                lo_x, end = front, min(length, front + reach)
            else:
                lo_x, end = 0.0, min(length, reach - covered)
            if end > lo_x:
                lo = np.array([lo_x, -half, z_lo])
                hi = np.array([end, half, z_hi])
                meets = np.all((c + r > lo + _TOUCH) & (c - r < hi - _TOUCH), axis=1)
                meets &= ~_apart(model, data, others, a, axes, lo, hi)
            else:
                meets = np.zeros(len(others), dtype=bool)
            if speed is not None and meets.any():
                # What goes along the way as fast as the mover comes is not in its way.
                along = np.array([velocity[o] @ axes[0, :2] for o in owners.tolist()])
                meets &= along < _ALONG * float(speed)
            for i in np.flatnonzero(meets):
                dist = covered + max(lo_x, float(c[i, 0] - r[i, 0])) - lo_x
                near = np.clip(np.array([lo_x, 0.0, 0.0]), c[i] - r[i], c[i] + r[i])
                # Where it comes nearest, and across the corridor as far as it spans it: a long
                # barrier outlines itself by the points it shows the corridor, which is what
                # `reroute` marks.
                across = _ACROSS * half
                span = across[(across >= c[i, 1] - r[i, 1]) & (across <= c[i, 1] + r[i, 1])]
                for y in sorted({float(near[1]), *span.tolist()}):
                    points.append(a + near[0] * axes[0, :2] + y * axes[1, :2])
                if best is None or dist < best:
                    best = dist
            covered += end - lo_x
            if covered >= reach - 1e-9:
                break
        return best, points

    def _front_along(self, model, data, body, start, heading: float):
        """``(front, bottom, top)``: how far the mover's body reaches ahead of its centre along
        ``heading``, and its bottom and top.

        Its extent is measured once, in the frame of its own root body, and kept: a walker's front
        moves by a stride as it walks and stands, and a corridor that started where its foot
        happens to be would open and close as it settles. The front along a heading is then where
        that extent reaches along it, turned with the body: a body that is not round in plan (a
        long cart, a robot not yet facing its first goal) has the front of the side it is going.
        """
        yaw = self._yaw(data)
        if self._front is None:
            c, s_ = math.cos(yaw), math.sin(yaw)
            axes = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])
            own_c, own_r = _extents(model, data, body, axes, np.array([start[0], start[1], 0.0]))
            lo, hi = (own_c - own_r).min(axis=0), (own_c + own_r).max(axis=0)
            self._front = (lo, hi)
        lo, hi = self._front
        turn = heading - yaw
        u = np.array([math.cos(turn), math.sin(turn)])
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        return float((corners @ u).max()), float(lo[2]), float(hi[2])

    def _yaw(self, data) -> float:
        """The heading of the mover's root body in the plane, 0 where it has none."""
        if self._root is None or self._root < 0:
            return 0.0
        m = data.xmat[self._root].reshape(3, 3)
        return math.atan2(float(m[1, 0]), float(m[0, 0]))

    def _velocity(self, owner: int, xy, now: float) -> np.ndarray:
        """A thing's planar velocity, from where it was over the last :data:`_VELOCITY_WINDOW`."""
        seen = self._track.setdefault(owner, [])
        seen.append((now, np.array(xy, dtype=float)))  # a copy: `xy` may be a view of mjData
        while len(seen) > 2 and now - seen[1][0] >= _VELOCITY_WINDOW:
            seen.pop(0)
        t0, p0 = seen[0]
        return (seen[-1][1] - p0) / (now - t0) if now - t0 > 1e-6 else np.zeros(2)

    def _legs(self, start, heading: float, path) -> list:
        """The legs of the corridor: from ``start`` along the path ahead, or along ``heading``."""
        ahead = np.array([math.cos(heading), math.sin(heading)])
        pts = [np.asarray(p, dtype=float)[:2] for p in (path or [])]
        legs = []
        if pts:
            first = pts[0] - start
            dist = float(np.linalg.norm(first))
            if dist > 1e-6 and float(first @ ahead) / dist > 0.0:
                legs.append((start, pts[0]))
                for p, q in zip(pts[:-1], pts[1:], strict=False):
                    if len(legs) >= _MAX_LEGS:
                        break
                    legs.append((p, q))
        if not legs:
            legs = [(start, start + ahead * 1e3)]
        else:
            # Past the path's last point the corridor runs on along its last leg, so a mover
            # arriving at its goal still looks at what stands just beyond it.
            p, q = legs[-1]
            d = q - p
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                legs.append((q, q + d / n * 1e3))
        return legs

    def _candidates(self, model):
        """Every geom of a thing that moves, other than the mover's own and the ignored ones, and
        the thing each belongs to."""
        if self._others is None:
            skip = self._own | self._ignored
            geoms = [
                g
                for g in range(model.ngeom)
                if g not in skip and is_dynamic_body(model, int(model.geom_bodyid[g]))
            ]
            self._others = np.array(geoms, dtype=int)
            # Not an entity's: one thing per root body, numbered past the entities.
            base = 1 + max(self._entity_of.values(), default=-1)
            self._owner = np.array(
                [
                    self._entity_of.get(g, base + int(model.body_rootid[model.geom_bodyid[g]]))
                    for g in geoms
                ],
                dtype=int,
            )
            for g, o in zip(geoms, self._owner.tolist(), strict=True):
                self._owner_body.setdefault(o, int(model.body_rootid[model.geom_bodyid[g]]))
        return self._others, self._owner


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


def heading_of(axes) -> float:
    """The heading of a frame's x axis in the plane."""
    return math.atan2(float(axes[0, 1]), float(axes[0, 0]))


def _apart(model, data, geoms, start, axes, lo, hi) -> np.ndarray:
    """Which ``geoms`` are clear of the box ``lo``..``hi`` in the frame ``axes`` at ``start``.

    :func:`_extents` measures a geom along the frame's axes, which is exact for one lined up with
    them and, for one turned against them, the box around it in that frame -- a long barrier at 45
    degrees to the corridor fills a square there that reaches far beside it. This looks along the
    geom's own horizontal axes as well, the other axes that can separate two boxes seen from above:
    the box's plan view against the geom's (its ``geom_aabb`` in its own frame, turned into the
    plane). A geom it finds clear along any of them is clear of the box.
    """
    rot = data.geom_xmat[geoms].reshape(-1, 3, 3)
    aabb = model.geom_aabb[geoms]
    centre = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rot, aabb[:, :3])
    size = aabb[:, 3:]
    box_c = np.asarray(start, dtype=float)[:2] + (
        (lo[0] + hi[0]) / 2.0 * axes[0, :2] + (lo[1] + hi[1]) / 2.0 * axes[1, :2]
    )
    box_h = np.array([(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0])
    apart = np.zeros(len(geoms), dtype=bool)
    for k in range(2):  # the geom's own x and y axes, in the plane
        u = rot[:, :2, k]
        n = np.linalg.norm(u, axis=1)
        ok = n > 0.3  # an axis that stands up shows the plane no edge to separate along
        u = np.where(ok[:, None], u / np.maximum(n, 1e-9)[:, None], 0.0)
        # the geom's reach along u: its three half sizes, each leaning onto u
        reach = np.sum(np.abs(np.einsum("nji,nj->ni", rot[:, :2, :], u)) * size, axis=1)
        box_r = box_h[0] * np.abs(u @ axes[0, :2]) + box_h[1] * np.abs(u @ axes[1, :2])
        gap = np.abs(np.einsum("ni,ni->n", centre[:, :2] - box_c, u)) - reach - box_r
        apart |= ok & (gap > -_TOUCH)
    return apart


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
