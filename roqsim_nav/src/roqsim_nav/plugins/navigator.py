"""Controller plugin: move an entity along a route, inside the simulator.

The counterpart of an external nav2 stack, for the movers a trial needs but is not measuring -- a
second robot in the aisle, a cart crossing a junction, a pedestrian, a pallet that goes somewhere
rather than along a fixed polyline. It plans with A* over a grid rasterized from the model's own wall
geoms (no map file, no localisation), follows it with a behaviour tree, and needs no bridge, no ROS
and no external stack -- so the only thing on the wire is the robot under test.

**It builds no geometry.** It moves the entity of the entry it is nested under, and *how* that entity
moves is its ``output``: a wheeled base takes a velocity command through the ``RobotHandle`` its
drive plugin published (so ``diff_drive`` still does its own inverse kinematics, acceleration ramp
and odometry -- the wheels really turn), a mocap prop takes a written pose. Outputs are resolved from
an entry-point group, so nothing here knows what embodiments exist.

Config -- a component of the entry that provides the entity it moves, since ownership is where the
entry sits rather than a config key::

    navigator:
      output: auto            # auto | drive | mocap | ... | module:Class | file.py:Class
      speed: 0.5              # m/s the route is followed at (REQUIRED, > 0)
      goals:                  # the route, world metres
        - [4.0, 3.0]          #   [x, y] or [x, y, yaw]
        - [0.0, 3.0]
      dwell: 0.0              # seconds to stand still on reaching a goal: one number, `[lo, hi]`
                              #   for a random pause, or a list of either -- one per route point,
                              #   the mover's own start included, so it lines up with `goals`
                              #   preceded by where the mover began
      route_mode: plan        # plan -> A* between the points; exact -> the points ARE the path
      tracker: waypoint       # waypoint -> steer at the goal, advance within `arrival_radius`
                              # pure_pursuit -> steer at a carrot `lookahead` along the route and
                              #   advance on crossing a goal; bounds cross-track error by the
                              #   lookahead instead of by the arrival radius, which is what a
                              #   non-holonomic base asked to follow a given path needs
      lookahead: 0.6          # m; pure_pursuit only
      autostart: true         # false -> hold at the first point until started
      loop: false             # cycle the route forever rather than stopping at the last point
      arrival_radius: 0.25

      # -- what it does about what the plan did not contain -----------------------------------
      # Three independent capabilities, not a ladder. See AVOIDANCE_KEYS for why.
      avoidance:
        stop: true            # look ahead and hold until the way is clear
        steer: none           # none | give_way | orca | module:Class -- which model gives way
        reroute: false        # remember what stopped it and plan around it (needs `stop`)
        params: {}            # per-agent keys the chosen model accepts, checked at load

        # the probe's own tuning, in the same block
        lookahead: 1.2        # m of clear corridor needed, measured from the mover's FRONT
        width: 0.6            # m of corridor swept: the body, plus the clearance it should keep
        rays: 5               # how finely that width is sampled
        height: 0             # m above the floor to scan; 0 -> just above obstacle_height's floor
        clear_time: 0.5       # s the way must stay open before setting off again
        yield_time: 3.0       # s a blockage reads as traffic before recovery may engage
        forget_after: 5.0     # s a remembered blockage keeps steering the planner (reroute only)
        blockage_radius: 0    # m of the disc a blockage marks; 0 -> half the corridor width
        ignore: []            # entities this mover never stops for

      # -- output: drive ---------------------------------------------------------------------
      kinematics: auto        # auto | unicycle | holonomic | ackermann (auto asks the output)
      heading_gain: 2.0       # rad/s of yaw command per rad of heading error
      max_angular_vel: 1.5    # rad/s cap BEFORE the drive plugin's own clip
      turn_in_place: 0.8      # rad of heading error above which a unicycle base pivots instead
      min_speed: 0.15         # m/s an ackermann base is never commanded below (it cannot pivot)
      face: travel            # holonomic only: travel | hold

      # -- output: mocap / walker ------------------------------------------------------------
      yaw_rate: 3.0           # rad/s the body is re-faced at (0 = snap)

      # -- planning --------------------------------------------------------------------------
      obstacle_height: [0.05, 0.6]  # z band a geom must span to be a wall FOR THIS MOVER
      resolution: 0.05              # m per planner grid cell
      planner:  {inflation_radius: 0.35, waypoint_radius: 0.3}
      recovery: {enabled: true, stuck_time: 1.5, backup_time: 0.5, max_recovery: 4}
      update_hz: 20.0               # nav pipeline rate; physics steps far faster

``obstacle_height`` is per mover on purpose: a 0.4 m pallet is not stopped by a ceiling beam that
blocks a walker, so "what counts as a wall" is a property of the thing navigating, not of the world.
Movers that agree on it share one rasterized grid (see :mod:`roqsim_nav.grid`).
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.kinematics import body_twist
from roqsim.plugin import Plugin

from .._resolve import RegistryError
from ..avoidance import NO_AGENT, SERVICE_KEY, resolve_model, service_for
from ..avoidance import available as avoidance_models
from ..behavior import NavCore, NavParams, build_tree
from ..caution import CautionProbe
from ..control import LAWS
from ..grid import DEFAULT_RESOLUTION, build_grid, grid_key
from ..handle import NavHandle, Sequencer
from ..outputs import available, resolve_output
from ..planner import GridPlanner
from ..state import NavState

logger = logging.getLogger(__name__)

#: Tried in order by ``output: auto``. Cheapest and most specific first: an entity with a drive
#: publishes a handle, and that is a stronger signal than merely having a mocap body.
_AUTO_ORDER = ("drive", "mocap")

ROUTE_MODES = ("plan", "exact")

#: The three things a mover can do about what its planner could not know about -- another robot, a
#: pedestrian, a driven prop. They are **independent capabilities, not a ladder**, and each is one
#: question with one answer:
#:
#:   ``stop``     look ahead and hold until the way is clear.
#:   ``steer``    which shared model gives way for it, or ``none`` to never deviate.
#:   ``reroute``  remember what stopped it and plan around it. Needs ``stop`` to have something to
#:                remember, and is the only one of the three that can change the planned path.
#:
#: A ladder was the obvious shape and it does not fit: a walker steers without ever stopping, which
#: is how every existing pedestrian world behaves, and an ordered scale cannot say that. Keeping the
#: three separate also means there is no combination table to learn -- each key stands alone.
#:
#: Not stopping is a decision rather than an oversight. An opponent that must be in the same place
#: at the same time in every repetition should not stop, because stopping for the robot under test
#: makes its trajectory a function of that robot's behaviour. One sharing a corridor should. And a
#: mover that does neither is not thereby harmless: a mocap body has no degrees of freedom, so the
#: solver treats it as immovable and it shoves anything free it touches, however politely that thing
#: stopped for it.
AVOIDANCE_KEYS = ("steer", "stop", "reroute", "params")

#: Keys that used to say this, and where each went. Refused by name rather than quietly ignored: a
#: world that still spells it the old way is asking for behaviour it will not get.
_RETIRED_KEYS = {
    "traffic": "`traffic: respect` is `avoidance: {stop: true}` (the default); `traffic: ignore` "
    "is `avoidance: {stop: false}`.",
    "caution": "Its tuning keys (lookahead, width, rays, height, clear_time, ignore, ...) are now "
    "keys of `avoidance:` directly, and `caution.on_blocked: replan` is `avoidance: {reroute: true}`.",
}

#: How the planned path is turned into motion.
#:
#: ``waypoint`` steers at the active goal and advances when within ``arrival_radius`` -- the
#: pedestrian stack's historical follower, and the default. A corner is rounded by about that radius,
#: because the mover aims at the *end* of the leg rather than at the path.
#:
#: ``pure_pursuit`` steers at a carrot a fixed ``lookahead`` along the route and advances on arc
#: progress rather than proximity. It exists for a base whose **steering is constrained** -- a car,
#: or a differential base with an acceleration limit -- where aiming at a distant goal overshoots and
#: oscillates, and where a bounded commanded curvature is what keeps the wheels tracking.
#:
#: **It is not an improvement for a pose-written body, and the numbers say so.** Measured on a mocap
#: mover round a right-angle corner, pure pursuit cuts the corner by roughly its lookahead
#: (0.27 m at 0.15, 0.85 m at 1.0) where the waypoint follower cuts by its arrival radius (0.04 m at
#: 0.05), and neither re-converges to a displaced path faster than the other. A body whose pose is
#: written has no steering to constrain, so the simpler follower is already doing the best available
#: thing. Reach for this when the embodiment has wheels, not by default.
TRACKERS = ("waypoint", "pure_pursuit")


def _validate_avoidance(config: dict, base_dir) -> list[str]:
    """Check the `avoidance:` block: its three capabilities, its model, and the probe's tuning."""
    spec = config.get("avoidance")
    if spec is None:
        return []
    if not isinstance(spec, dict):
        return [
            "'avoidance' is a block, not a value. It carries three independent capabilities -- "
            "`stop` (look ahead and hold), `steer` (which model gives way for this mover, or none) "
            "and `reroute` (remember a blockage and plan around it) -- plus the probe's tuning. "
            "For example: avoidance: {steer: give_way, stop: true, lookahead: 0.6}"
        ]
    errors = []
    for key in ("stop", "reroute"):
        if key in spec and not isinstance(spec[key], bool):
            errors.append(f"'avoidance.{key}' must be true or false")
    if spec.get("reroute") and not spec.get("stop", True):
        errors.append(
            "'avoidance.reroute' needs 'stop': a blockage is only ever discovered by looking ahead "
            "and stopping for it, so there is nothing to plan around without it."
        )
    steer = spec.get("steer", "none")
    if isinstance(steer, bool):
        errors.append(
            f"'avoidance.steer' names a model, not a yes/no -- there is more than one and 'yes' "
            f"does not say which. Use 'none' or one of: "
            f"{', '.join(avoidance_models()) or '(none registered)'}."
        )
    elif not isinstance(steer, str):
        errors.append("'avoidance.steer' must be 'none' or a model name")
    elif steer != "none":
        try:
            cls = resolve_model(steer, base_dir)
        except RegistryError as exc:
            errors.append(str(exc))
        else:
            schema = set(getattr(cls, "params_schema", ()) or ())
            params = spec.get("params") or {}
            unknown = sorted(set(params) - schema) if schema else []
            if unknown:
                errors.append(
                    f"avoidance model {steer!r} does not accept {', '.join(unknown)}. It accepts: "
                    f"{', '.join(sorted(schema))}. (A key left over from another model is refused "
                    f"rather than ignored.)"
                )
    probe = {k: v for k, v in spec.items() if k not in AVOIDANCE_KEYS}
    unknown = sorted(set(probe) - set(CautionProbe.KEYS))
    if unknown:
        errors.append(
            f"'avoidance' does not accept {', '.join(unknown)}. Its keys are "
            f"{', '.join(AVOIDANCE_KEYS)} plus the probe's: {', '.join(CautionProbe.KEYS)}."
        )
    return errors + CautionProbe.validate(probe)


def _dwell_pair(d) -> tuple[float, float]:
    """One dwell entry: ``s`` seconds, or ``[lo, hi]`` for a random pause, as a ``(lo, hi)`` pair."""
    if isinstance(d, (list, tuple)):
        if len(d) != 2:
            raise ValueError(f"a random pause is [lo, hi], got {list(d)!r}")
        lo, hi = float(d[0]), float(d[1])
        if lo > hi:
            raise ValueError(f"[lo, hi] must be ordered, got [{lo}, {hi}]")
    else:
        lo = hi = float(d)
    if lo < 0.0:
        raise ValueError(f"a dwell cannot be negative, got {lo}")
    return lo, hi


def _dwell_list(spec, n: int) -> list[tuple[float, float]]:
    """`n` dwell pairs from a scalar, a ``[lo, hi]`` pair, or one entry per route point.

    A bare ``[lo, hi]`` is ambiguous against a two-point route's per-point list, and is read as the
    random pause -- which is the form a world writes far more often. Spell the per-point case with
    nested entries (``[[0, 3], [1, 2]]``) to say the other thing.
    """
    if isinstance(spec, (list, tuple)) and spec and all(isinstance(e, (list, tuple)) for e in spec):
        if len(spec) != n:
            raise ValueError(f"one dwell per route point: expected {n}, got {len(spec)}")
        return [_dwell_pair(e) for e in spec]
    return [_dwell_pair(spec)] * n


class NavigatorPlugin(Plugin):
    #: It drives an entity somebody else provided and builds nothing, so it belongs inside that
    #: entity's ``components:`` block -- in every output mode, which is why this is a class
    #: attribute it can honestly carry.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._output = None
        self._core: NavCore | None = None
        self._tree = None
        self._state: NavState | None = None
        self._accum = 0.0
        self._period = 0.0
        self._started = False
        self._caution: CautionProbe | None = None
        self._bid = -1
        self._seq = Sequencer()
        self._plan_pending = True
        self._radius: float | None = None
        self._avoid = None
        self._agent = NO_AGENT
        self._yields = False
        self._configured_goals: list[tuple] = []
        self._commanded = False
        self._pose_snapshot = (0.0, 0.0, 0.0)

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        speed = config.get("speed")
        if speed is None:
            errors.append("'speed' is required (m/s the route is followed at)")
        else:
            try:
                if float(speed) < 0.0:
                    # Zero is allowed and means "does not move": a mover parked as scenery that a
                    # scenario may later send somewhere, or a walker placed purely to be measured
                    # against. Negative is not a slower speed, it is a mistake.
                    errors.append("'speed' must be >= 0 (0 means it does not move)")
            except (TypeError, ValueError):
                errors.append("'speed' must be a number (m/s)")

        mode = config.get("route_mode", "plan")
        if mode not in ROUTE_MODES:
            errors.append(f"'route_mode' must be one of: {', '.join(ROUTE_MODES)}")

        tracker = config.get("tracker", "waypoint")
        if tracker not in TRACKERS:
            errors.append(f"'tracker' must be one of: {', '.join(TRACKERS)}")
        if config.get("lookahead") is not None:
            if float(config["lookahead"]) <= 0.0:
                errors.append("'lookahead' must be > 0")
            if tracker != "pure_pursuit":
                errors.append(
                    "'lookahead' has no meaning with tracker: waypoint -- it is the pure-pursuit "
                    "carrot distance"
                )

        for i, g in enumerate(config.get("goals") or []):
            if not (isinstance(g, (list, tuple)) and 2 <= len(g) <= 3):
                errors.append(f"goals[{i}] must be [x, y] or [x, y, yaw] in world metres")
        try:
            _dwell_list(config.get("dwell", 0.0), 1 + len(config.get("goals") or []))
        except (TypeError, ValueError) as exc:
            errors.append(f"'dwell': {exc}")
        if config.get("loop") and not (config.get("goals") or []):
            # The route is the mover's start plus its goals, so ONE goal is already a two-point
            # shuttle -- which is exactly what a two-waypoint patrol is. Counting goals rather than
            # route points refused that.
            errors.append("'loop: true' needs at least one goal to cycle through")

        kin = config.get("kinematics", "auto")
        if kin != "auto" and kin not in LAWS:
            errors.append(f"'kinematics' must be 'auto' or one of: {', '.join(sorted(LAWS))}")
        if kin != "auto" and config.get("output") in ("mocap",):
            # Silently ignoring a key is the failure a validator exists to prevent: a pose is
            # written where the path says, so there is no base geometry to shape a command for.
            errors.append(
                "'kinematics' has no meaning with output: mocap -- the pose is written directly, "
                "not steered"
            )

        band = config.get("obstacle_height")
        if band is not None:
            ok = isinstance(band, (list, tuple)) and len(band) == 2
            if not ok or float(band[0]) >= float(band[1]):
                errors.append("'obstacle_height' must be [z_lo, z_hi] with z_lo < z_hi")

        for key in ("arrival_radius", "resolution", "update_hz", "heading_gain", "max_angular_vel"):
            value = config.get(key)
            if value is not None and float(value) <= 0.0:
                errors.append(f"'{key}' must be > 0")

        for gone, guidance in _RETIRED_KEYS.items():
            if gone in config:
                errors.append(f"'{gone}' has moved into the `avoidance:` block. {guidance}")
        errors += _validate_avoidance(config, self.base_dir)
        avoid_spec = config.get("avoidance")
        avoid_spec = avoid_spec if isinstance(avoid_spec, dict) else {}
        if avoid_spec.get("reroute") and config.get("route_mode") == "exact":
            errors.append(
                "'avoidance.reroute' cannot be used with 'route_mode: exact'. An exact route IS the "
                "given polyline, so planning around something means not walking it; stopping is the "
                "only response that keeps that promise."
            )

        if (config.get("recovery") or {}).get("enabled") and mode == "exact":
            # Backing up and re-planning is, by definition, leaving the path that was given.
            errors.append(
                "'recovery' cannot be enabled with route_mode: exact -- recovery re-plans, and "
                "'exact' promises the given polyline. Traffic is handled by stopping instead."
            )
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def build(self, spec, ctx: SimContext) -> None:
        """Nothing: this plugin builds no geometry. (Said out loud so nobody goes looking.)"""

    def configure(self, ctx: SimContext) -> None:
        cfg = self.config
        entity = ctx.entities.get(self.entity)
        if entity is None:
            raise RuntimeError(
                f"navigator {self.address!r}: no entity {self.entity!r}. It must be nested under an "
                f"entry that provides one (spawn_robot, spawn_model, walker)."
            )
        if ctx.blackboard.get(f"nav:{self.entity}") is not None:
            raise RuntimeError(
                f"navigator {self.address!r}: entity {self.entity!r} already has a navigator. Two "
                f"would fight over one body every step."
            )
        self._output = self._resolve_output(ctx, entity)

        # Ground truth, not odometry: see NavOutput.pose.
        x, y, yaw = self._output.pose(ctx)
        # Read per tick rather than cached here: `data.xpos` is not populated until the first
        # forward pass, so caching it in `configure` yields 0.0 -- and a ray cast along z = 0 grazes
        # the bottom face of everything standing on the floor, which detects obstacles late and
        # erratically instead of not at all. Reading it each tick is one array index and is always
        # right, including for a body whose height changes.
        self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        goals = [tuple(g)[:2] for g in (cfg.get("goals") or [])]
        waypoints = np.asarray([(x, y), *goals] if goals else [(x, y)], dtype=float)
        # `dwell` is indexed by `goal_idx`, which indexes `waypoints` -- so it is built to that
        # length here rather than to the length of `goals`, and the mover's own start point gets an
        # entry like every other route point.
        dwell = _dwell_list(cfg.get("dwell", 0.0), len(waypoints))

        self._state = NavState(
            name=self.entity,
            waypoints=waypoints,
            dwell=dwell,
            speed=float(cfg.get("speed", 0.5)),
            loop=bool(cfg.get("loop", False)),
            arrival_radius=float(cfg.get("arrival_radius", 0.25)),
            pos=np.array([x, y], dtype=float),
            yaw=yaw,
        )

        # The planner is built on the first tick, not here. `configure` runs before any presence has
        # been applied, so a grid rasterized now contains every obstacle the world COMPILED --
        # including the ones compiled in precisely so they can appear mid-trial, which are absent at
        # the start. The mover would then route around a thing nothing can see or touch. Deferring to
        # the first tick puts it after every plugin's `on_reset`, which is where presence is settled.
        planner = None

        params = NavParams.from_spec(cfg, float(cfg.get("radius", 0.3)))
        rng = ctx.rng_for(f"navigator:{self.entity}")
        self._core = NavCore(self._state, planner, params, uniform=rng.uniform)
        self._tree = build_tree(
            self._core,
            recovery=bool((cfg.get("recovery") or {}).get("enabled", True)),
            lookahead=(
                float(cfg.get("lookahead", 0.6))
                if cfg.get("tracker", "waypoint") == "pure_pursuit"
                else None
            ),
        )

        avoid = cfg.get("avoidance") or {}
        # `avoidance.steer` names the model that gives way for this mover, or 'none' to never
        # deviate. There is no world-level entry to declare: the model appears when a mover asks to
        # be steered, and the first to ask fixes it for the world.
        steer = str(avoid.get("steer", "none"))
        self._yields = steer != "none"
        if self._yields:
            spec = {"model": steer, **(avoid.get("params") or {})}
            # Created here, joined at reset. Configure order is deterministic and every configure
            # precedes every reset, so a mover that only opts OUT still finds the model at reset
            # time and can register as something the others must avoid.
            service_for(ctx, spec, self.base_dir)

        self._caution = CautionProbe(
            {
                # The probe's tuning lives in the same block as the capabilities it serves, so it is
                # passed through whole; the navigator has already refused any key neither reads.
                **{k: v for k, v in avoid.items() if k not in AVOIDANCE_KEYS},
                "enabled": bool(avoid.get("stop", True)),
                "reroute": bool(avoid.get("reroute", False)),
                # The probe scans inside the same band the planner rasterizes, so "what counts as an
                # obstacle" is declared once for this mover rather than twice.
                "band": cfg.get("obstacle_height") or (0.1, 1.8),
            }
        )
        self._caution.attach(ctx, entity)

        # Joining the avoidance model is deferred to on_reset, not done here: components are
        # configured with their owner, so a world that declares its `avoidance:` entry *after* its
        # movers would configure every navigator before the model existed. They would each find
        # nothing, fall back silently to no avoidance, and the world would behave differently for
        # the order of two lines in a file -- with nothing in the log to say so.

        rate = cfg.get("update_hz")
        if rate is None:
            rate = self._output.update_hz or 20.0
        self._period = 1.0 / float(rate)
        self._started = bool(cfg.get("autostart", True))

        self._configured_goals = list(goals)
        # Kept, not just applied once: `st.dwell` is indexed by `goal_idx` against
        # `st.waypoints`, so every route rebuild below has to resize it too.
        self._dwell_spec = cfg.get("dwell", 0.0)
        self._ctx = ctx
        self._declare_endpoints(ctx, entity)
        ctx.blackboard.set(f"nav:{self.entity}", self)
        ctx.blackboard.set(
            f"nav:{self.entity}:handle",
            NavHandle(
                name=self.entity,
                send_goals=self.send_goals,
                start=self.start,
                cancel=self.cancel,
                status=self.status,
                pose=lambda: self._pose_snapshot,
            ),
        )

    def _resolve_output(self, ctx, entity):
        """The embodiment, named or probed. ``auto`` reports every reason it tried, not the first."""
        ref = self.config.get("output", "auto")
        if ref != "auto":
            output = resolve_output(ref, self.base_dir)(self.config)
            output.attach(ctx, entity)
            return output
        reasons = []
        for name in _AUTO_ORDER:
            try:
                output = resolve_output(name, self.base_dir)(self.config)
                output.attach(ctx, entity)
                return output
            except RegistryError as exc:
                # RegistryError, not just OutputUnavailable: a candidate that is not installed here
                # is as unavailable as one that cannot drive this entity, and probing must degrade
                # over what the environment actually has rather than dying on the first gap.
                reasons.append(f"  {name}: {exc}")
        raise RuntimeError(
            f"navigator {self.address!r}: no output can move entity {self.entity!r}.\n"
            + "\n".join(reasons)
            + f"\nRegistered outputs: {', '.join(available()) or '(none)'}. Name one explicitly "
            f"with `output:` if it is not probed automatically."
        )

    def _planner(self, ctx, waypoints) -> GridPlanner | None:
        """The world's shared grid, wrapped in this mover's own inflation.

        One raster per (resolution, height band) rather than one per navigator: building it walks
        every geom in the model and rasterizes every wall, which is wasted work repeated for each
        mover in a scene. ``GridPlanner`` memoises inflation per radius on top, so movers of
        different sizes still share the raster underneath.
        """
        cfg = self.config
        band = cfg.get("obstacle_height") or (0.1, 1.8)
        z_lo, z_hi = float(band[0]), float(band[1])
        resolution = float(cfg.get("resolution", DEFAULT_RESOLUTION))
        resting = self._parked_props(ctx)
        key = f"{grid_key(resolution, z_lo, z_hi, resting)}:{ctx.episode}"
        grid = ctx.blackboard.get(key)
        if grid is None:
            mujoco.mj_forward(ctx.model, ctx.data)  # geom_xpos must be valid to read footprints
            grid = build_grid(
                ctx.model,
                ctx.data,
                extra_points=waypoints,
                z_lo=z_lo,
                z_hi=z_hi,
                resolution=resolution,
                resting_roots=resting,
            )
            ctx.blackboard.set(key, grid if grid is not None else False)
        elif grid is False:
            grid = None
        if grid is None:
            # Not an error: with no static geometry the behaviour tree walks straight legs.
            logger.info("navigator %s: no walls to plan around; legs are straight", self.entity)
            return None
        # Inflated by the mover's own footprint, so the path it plans is one it fits through.
        return GridPlanner(grid, NavParams.from_spec(cfg, self.radius(ctx)).inflation_radius)

    def _parked_props(self, ctx) -> tuple[int, ...]:
        """Weld roots of the free props nobody is driving, for the grid to treat as walls.

        A prop with a free joint and no navigator is scenery that happens to be movable -- a crate
        somebody left in a doorway. Planning through it and then stopping in front of it wastes the
        whole route, so while it is standing still it is a wall.

        Robots are NOT in this set even when they are motionless, and that is the distinction that
        matters: the subject stands still at spawn because nothing is driving it *yet*. Baking it
        into the grid would make every opponent plan around where the robot under test was parked at
        t = 0, for the whole episode. Traffic is the business of caution and avoidance, which can
        watch it move; the grid only gets what will still be true in a minute.
        """
        model = ctx.model
        roots = set()
        for entity in ctx.entities.all():
            if entity.kind != "object" or not entity.body:
                continue
            if ctx.blackboard.get(f"nav:{entity.name}:handle") is not None:
                continue  # driven, so it is traffic rather than scenery
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
            if bid >= 0 and int(model.body_weldid[bid]) != 0:
                roots.add(int(model.body_weldid[bid]))
        return tuple(sorted(roots))

    def on_reset(self, ctx: SimContext) -> None:
        """The owner's ``on_reset`` ran first (owners flatten before their components), so the body
        is already back at its spawn pose -- read it rather than assuming the configured start."""
        x, y, yaw = self._output.pose(ctx)
        self._state.pos = np.array([x, y], dtype=float)
        self._state.yaw = yaw
        self._state.waypoints[0] = (x, y)
        self._core.reset()
        # First point at which every entity in the document has registered, so `caution.ignore` can
        # name one declared after this mover.
        self._caution.resolve_ignored(ctx)
        self._caution.reset()
        self._accum = 0.0
        self._started = bool(self.config.get("autostart", True))
        # Episode N must not inherit episode N-1's route, nor its completion latch.
        st = self._state
        st.waypoints = np.asarray([(x, y), *self._configured_goals], dtype=float)
        st.dwell = _dwell_list(self._dwell_spec, len(st.waypoints))
        st.loop = bool(self.config.get("loop", False))
        self._seq.apply(0)
        # Re-plan next tick: an episode may make a different set of obstacles present, and a grid
        # carried over from the last one would route around whichever were present then.
        self._plan_pending = True
        self._join_avoidance(ctx)
        if self._avoid is not None:
            self._avoid.ensure_reset(ctx)
        self._output.stop(ctx)

    def pre_step(self, ctx: SimContext) -> None:
        # Ahead of the decimation gate, and only ever true once per episode: the route must be
        # planned on the FIRST tick rather than a decimation period later, so that a route the
        # planner cannot solve surfaces at the start of the episode instead of whenever the mover
        # happens to first be ticked. One boolean test on the hot path, False ever after.
        if self._plan_pending:
            self._plan_pending = False
            if self.config.get("route_mode", "plan") == "plan":
                self._core.planner = self._planner(ctx, self._state.waypoints)

        # Decimate, before reading anything else: at 20 Hz inside a 500 Hz loop this hook is a float
        # comparison on 24 steps out of 25, and it shares the one thread with the stack under test.
        self._accum += ctx.dt
        if self._accum < self._period:
            return
        step_dt, self._accum = self._accum, 0.0

        if not self._started:
            return
        if ctx.manual_control and self._output.kinematics != "holonomic":
            return  # the viewer's sliders own the actuators

        x, y, yaw = self._output.pose(ctx)
        self._state.pos = np.array([x, y], dtype=float)
        self._state.yaw = yaw
        # Rebound, never mutated: `NavHandle.pose` reads this from another thread, and a tuple swap
        # is atomic where writing into a shared array is not.
        self._pose_snapshot = (x, y, yaw)

        # Order: plan, then AVOID, then decide whether to stop. Caution used to run first, against
        # the raw preferred velocity, and that made steering impossible: it stopped the mover, a
        # stopped mover has no velocity to steer, so the avoidance model saw nothing to shape and two
        # movers meeting head-on stopped nose to nose however good the model was. Stopping is the
        # fallback for what steering cannot clear, so it has to judge the steered velocity.
        # The blocker is last tick's, which is what there is: caution runs after the tree, because
        # it has to judge the velocity avoidance produced rather than the raw one. A tick of lag is
        # nothing next to the seconds recovery waits for before it acts.
        self._core.observe(ctx.sim_time, self._state.pos, self._caution.blocker_xy)
        self._tree.tick()
        if self._state.done and not self._seq.finished:
            self._seq.finish()
            self._resume_configured_route()

        wanted = self._core.pref_vel
        if self._avoid is not None:
            entity = ctx.entities.get(self.entity)
            self._ensure_solved(ctx)
            self._avoid.submit(
                self._agent,
                self._state.pos,
                self._velocity(ctx),
                wanted,
                present=bool(entity.present) if entity is not None else True,
            )
            wanted = self._avoid.result(self._agent)

        if self._caution.check(ctx, self._state.pos, 0.0, wanted):
            # Hold: do NOT advance along the path. Whether the progress clock is rebased with it
            # depends on what is in the way. Traffic will move, so waiting for it must never look
            # like being stuck -- that is what would turn "wait for the robot to pass" into "back up
            # and re-route around it". Something that has not moved in `yield_time` will not move,
            # and then the opposite is true: the clock has to run, or recovery can never engage. A
            # mover in a pocket needs it, because the way out is toward the obstacle beside it, so
            # every plan it makes is refused by this same probe until it has backed off.
            #
            # Without `reroute` the clock never runs at all, whatever is in the way: that mover's
            # promise is that being blocked cannot change its path, and a recovery is a change of
            # path -- letting one fire after a while would turn "it stopped" into "it went round",
            # which is the failure this whole layer exists to prevent.
            if not self._caution.reroute or self._caution.yielding(ctx.sim_time):
                self._core.forget_progress()
            self._remember_blockage(ctx)
            # `hold`, not `stop`: time is passing and the mover is still in the scene, so an
            # embodiment that is watched can settle -- a walker stands rather than freezing
            # mid-stride -- and turns toward the way it will leave.
            self._output.hold(ctx, step_dt, wanted)
            return

        self._output.emit(ctx, wanted, yaw, step_dt)

    def _remember_blockage(self, ctx) -> None:
        """Under ``avoidance.reroute``, mark what stopped us and plan around it.

        Only when asked for: the default is to hold position and keep the path, because a mover that
        re-routes has changed the trajectory an experiment may have been holding fixed.

        Dropping ``path`` is what makes the mark take effect -- the behaviour tree re-plans whenever
        there is no path, and the next plan is the one that sees the mark. The mover still holds this
        tick; it leaves on the next one, along a route that goes round.
        """
        probe = self._caution
        if not probe.reroute or not probe.blocker_points:
            return
        planner = self._core.planner
        if planner is None:
            return  # nothing to route around with: straight legs, and caution is the only answer
        expires = float(ctx.sim_time) + probe.forget_after
        # EVERY hit, not just the nearest: the rays fan across the corridor, so a wall in front
        # produces several hits spread along it, and marking the lot outlines how far the obstacle
        # reaches across the way. Marking only the nearest point leaves a disc narrower than the
        # thing it stands for -- the next plan rounds the disc, drives into the same wall half a
        # metre along, and reports a point already inside the mark, so nothing new is learnt and the
        # mover holds there for good.
        fresh = [
            planner.add_blockage(p, probe.blockage_radius, expires) for p in probe.blocker_points
        ]
        if any(fresh):
            self._state.path = None
            self._state.path_idx = 0

    def _resume_configured_route(self) -> None:
        """After a commanded route finishes, go back to the route the world configured.

        A patrolling opponent that is sent somewhere should resume patrolling when it gets there,
        rather than standing wherever the last goal left it -- which is what the pedestrian stack
        always did, and is the sensible behaviour for a cart on a loop too. The completion latch is
        left alone: the caller that sent the route still has to be able to observe its arrival, and
        restoring the patrol clears ``done``.
        """
        if not self._configured_goals or self._commanded is False:
            return
        self._commanded = False
        st = self._state
        st.waypoints = np.asarray([tuple(st.pos), *self._configured_goals], dtype=float)
        st.dwell = _dwell_list(self._dwell_spec, len(st.waypoints))
        st.loop = bool(self.config.get("loop", False))
        self._core.reset()

    #: Action types a bridge may serve this navigator's goal endpoint as. Named as STRINGS, so this
    #: package imports nothing ROS and a world that declares no bridge needs no nav2 installed --
    #: the bridge resolves the name and finds its handler.
    ACTIONS = {
        "navigate_to_pose": "nav2_msgs.action.NavigateToPose",
        "navigate_through_poses": "nav2_msgs.action.NavigateThroughPoses",
    }

    def _declare_endpoints(self, ctx: SimContext, entity) -> None:
        """Declare the goal interface as backend-neutral ``in`` endpoints.

        One ``write`` for both: the neutral payload is a list of points either way, and a single goal
        is a one-element list. The two exist as separate endpoints only because a ROS client picks an
        action type, and nav2 has two.

        ``goal_endpoint: false`` declares none, so a bridge needs no handler -- and therefore no
        nav2_msgs -- for a mover that is only ever commanded in-process. Declaring an endpoint no
        handler serves is a hard error at bridge start-up, by design, so this is not a formality.
        """
        cfg = self.config
        if not cfg.get("goal_endpoint", True):
            return
        namespace = cfg.get("namespace") or (entity.meta or {}).get("namespace", "")
        names = cfg.get("action_names") or {}
        # `action_name` (singular) is the walker's legacy spelling for the through-poses name.
        legacy = cfg.get("action_name")
        for endpoint, action_type in self.ACTIONS.items():
            if endpoint not in (cfg.get("actions") or self.ACTIONS):
                continue
            name = names.get(endpoint) or (
                legacy if legacy and endpoint == "navigate_through_poses" else endpoint
            )
            ctx.interface.add(
                Endpoint(
                    name=endpoint,
                    direction="in",
                    owner=self.entity,
                    namespace=namespace,
                    write=self._write_goals,
                    backend={"ros2": {"action": action_type, "name": name}},
                )
            )

    def _write_goals(self, poses) -> None:
        """Endpoint ``write``: the bridge has already marshalled this onto the physics thread."""
        route = [(float(p[0]), float(p[1])) for p in poses]
        if route:
            self._apply_goals(route, self._seq.next())
        else:
            self._apply_start(self._seq.next())

    def radius(self, ctx) -> float:
        """This mover's footprint radius: configured, or MEASURED from its own geometry.

        Measured by default, because a declared one is a number a world has to get right about a
        robot it did not build -- and getting it wrong is quiet. An mpo_500 is 0.64 m across the
        diagonal; declaring the 0.35 that looks right made the avoidance model believe two of them
        cleared at 0.70 m when they need 1.29, and they ground past each other in contact for the
        whole pass. The model already computes this for the caution probe, so the default costs
        nothing and cannot be wrong.
        """
        if self._radius is None:
            # `geom_xpos` is only refreshed by a forward pass, and this runs during `on_reset` --
            # after the owner has written the body's new pose but before anything has stepped.
            # Measuring without this read the PREVIOUS positions: a walker's skeleton still parked
            # where the model compiled it, which came out as a 4.2 m radius and was then cached for
            # the run.
            mujoco.mj_forward(ctx.model, ctx.data)
            configured = self.config.get("radius")
            self._radius = (
                float(configured)
                if configured is not None
                else self._caution.footprint_radius(ctx, self._state.pos)
            )
        return self._radius

    def _join_avoidance(self, ctx) -> None:
        """Join the world's avoidance model, once every plugin has configured. Idempotent.

        Deferred to ``on_reset`` rather than done in ``configure`` so that a mover declared before
        another still shares its model: the first to arrive creates it, and order does not decide
        who gets one.
        """
        if self._agent != NO_AGENT:
            return
        # EVERY mover joins, including one that never gives way. Opting out of yielding is not
        # opting out of existing: a mover the others cannot see is one they drive into, and
        # "one stops, the other goes around it" needs the stopping one to be there to go around.
        self._avoid = ctx.blackboard.get(SERVICE_KEY)
        if self._avoid is None:
            return  # nobody in this world asked for avoidance at all
        cfg = self.config
        self._agent = self._avoid.add_agent(
            self.entity,
            radius=self.radius(ctx),
            max_speed=float(cfg.get("max_speed", max(1.0, 2.0 * self._state.speed))),
            # Apparatus yields; the subject does not. Opting out still occupies an agent, so others
            # go round this mover -- it simply never gives way itself, which is what keeps a
            # strictly reproducible opponent reproducible.
            yields=self._yields,
            params=cfg.get("params") or {},
        )

    def _ensure_solved(self, ctx) -> None:
        """Make sure this step's avoidance solve has happened, whoever gets here first."""
        ensure = getattr(self._avoid, "ensure_solved", None)
        if ensure is not None:
            ensure(ctx)

    def _velocity(self, ctx) -> np.ndarray:
        """Ground-truth planar velocity, for the avoidance model's reciprocity.

        Read from the model rather than from the last command: reciprocal avoidance works because
        agents react to what the others are *doing*, and a mover reporting its intention instead
        would be avoided as though it had already achieved it -- which is how a pair of them end up
        both giving way to a manoeuvre neither has made.
        """
        if self._bid < 0:
            return np.zeros(2)
        twist = body_twist(ctx.model, ctx.data, self._bid)
        return np.array([twist.linear[0], twist.linear[1]], dtype=float)

    # -- runtime control -----------------------------------------------------------------------
    # Every method here may be called from any thread. None of them touches `model`/`data`: they
    # stamp a sequence number and marshal the actual change onto the physics thread with `ctx.post`,
    # which is the substrate's single-writer rule.
    @property
    def started(self) -> bool:
        """Whether the configured route is running. ``autostart: false`` holds it until started."""
        return self._started

    def start(self) -> int:
        """Release a route that was armed but held (``autostart: false``).

        The route was planned at ``configure``, not here, so a route that cannot be planned fails at
        load rather than halfway through a trial when a scenario finally triggers it. Starting an
        already-running route is a no-op that returns the live sequence, so a caller can trigger
        unconditionally without having to know whether it already ran.
        """
        if self._started:
            return self._seq.applied
        seq = self._seq.next()
        self._ctx.post(lambda ctx: self._apply_start(seq))
        return seq

    def send_goals(self, goals) -> int:
        """Replace the route with ``goals`` and run it. Returns its sequence number immediately."""
        route = [tuple(float(v) for v in g)[:2] for g in goals]
        if not route:
            raise ValueError("send_goals needs at least one goal")
        seq = self._seq.next()
        self._ctx.post(lambda ctx: self._apply_goals(route, seq))
        return seq

    def cancel(self) -> int:
        """Stop where it stands. The configured route is not resumed -- a cancel is not an arrival."""
        seq = self._seq.next()
        self._ctx.post(lambda ctx: self._apply_cancel(seq))
        return seq

    def status(self) -> tuple[int, bool, int, float]:
        """``(applied sequence, finished, goals remaining, distance remaining)``.

        Read without a lock: every field is a plain int/float/bool latched on the physics thread, and
        a caller that sees a half-updated pair polls again a moment later.
        """
        if self._seq.finished:
            # Nothing remains OF THE ROUTE that finished. The mover may well be busy again -- it
            # resumes its configured patrol on arrival -- but reporting that patrol's remaining
            # goals to a caller polling for its own completion would answer a different question.
            return self._seq.applied, True, 0, 0.0
        goals_left, dist_left = self._core.remaining()
        return self._seq.applied, self._seq.finished, goals_left, dist_left

    # -- applied on the physics thread -----------------------------------------------------------
    def _apply_start(self, seq: int) -> None:
        self._started = True
        self._seq.apply(seq)

    def _apply_goals(self, route, seq: int) -> None:
        x, y, _yaw = self._output.pose(self._ctx)
        st = self._state
        st.waypoints = np.asarray([(x, y), *route], dtype=float)
        # A commanded route is a one-off goal sequence, not the configured patrol, so
        # it carries no dwell -- and clearing it also keeps `dwell` the same length as
        # `waypoints`, which `goal_idx` indexes both of.
        st.dwell = None
        st.loop = False  # a commanded route runs once; looping is a property of the configured one
        self._commanded = True
        self._core.reset()
        self._caution.reset()
        self._started = True
        self._seq.apply(seq)

    def _apply_cancel(self, seq: int) -> None:
        x, y, _yaw = self._output.pose(self._ctx)
        self._state.waypoints = np.asarray([(x, y)], dtype=float)
        self._state.dwell = None
        self._core.reset()
        self._state.done = True
        self._output.stop(self._ctx)
        self._seq.apply(seq)
        self._seq.finish()

    def shutdown(self, ctx: SimContext) -> None:
        if self._output is not None:
            self._output.stop(ctx)
