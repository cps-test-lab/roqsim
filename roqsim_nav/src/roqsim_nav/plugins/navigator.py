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
from ..avoidance import NO_AGENT, NullAvoidance
from ..behavior import NavCore, NavParams, build_tree
from ..caution import CautionProbe
from ..control import LAWS
from ..grid import DEFAULT_RESOLUTION, build_grid, grid_key
from ..handle import NavHandle, Sequencer
from ..outputs import available, resolve_output
from ..planner import GridPlanner
from ..state import NavState
from .avoidance import MODEL_KEY

logger = logging.getLogger(__name__)

#: Tried in order by ``output: auto``. Cheapest and most specific first: an entity with a drive
#: publishes a handle, and that is a stronger signal than merely having a mocap body.
_AUTO_ORDER = ("drive", "mocap")

ROUTE_MODES = ("plan", "exact")

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
        self._avoid = NullAvoidance()
        self._agent = NO_AGENT
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

        errors += CautionProbe.validate(config.get("caution"))

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

        self._state = NavState(
            name=self.entity,
            waypoints=waypoints,
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

        self._caution = CautionProbe(cfg.get("caution"))
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
        key = f"{grid_key(resolution, z_lo, z_hi)}:{ctx.episode}"
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
            )
            ctx.blackboard.set(key, grid if grid is not None else False)
        elif grid is False:
            grid = None
        if grid is None:
            # Not an error: with no static geometry the behaviour tree walks straight legs.
            logger.info("navigator %s: no walls to plan around; legs are straight", self.entity)
            return None
        return GridPlanner(
            grid, NavParams.from_spec(cfg, float(cfg.get("radius", 0.3))).inflation_radius
        )

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
        st.loop = bool(self.config.get("loop", False))
        self._seq.apply(0)
        # Re-plan next tick: an episode may make a different set of obstacles present, and a grid
        # carried over from the last one would route around whichever were present then.
        self._plan_pending = True
        self._join_avoidance(ctx)
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

        # Caution runs against the velocity we WOULD command, which is last tick's -- the current one
        # is not known until the tree ticks, and ticking it is exactly what must not happen while
        # blocked. At 20 Hz that is a 50 ms old heading for a mover on a planned path, which does not
        # change direction abruptly.
        z = float(ctx.data.xpos[self._bid][2]) if self._bid >= 0 else 0.0
        if self._caution.check(ctx, self._state.pos, z, self._core.pref_vel):
            # Hold everything: do NOT tick the tree, so `FollowPath` cannot advance a waypoint and
            # `EnsurePath` cannot re-plan. The mover resumes on the leg it was already on.
            #
            # Rebasing the progress clock is what keeps yielding from decaying into re-routing. The
            # recovery branch declares a stall after `stuck_time` of no movement, and a mover waiting
            # for the robot to pass is not moving -- so without this, "wait for it to go by" turns
            # itself into "back up and plan around it" after a second and a half, silently defeating
            # the stop. It matters because recovery is ON by default.
            self._core.observe(ctx.sim_time, self._state.pos, None)
            self._core.forget_progress()
            self._output.stop(ctx)
            return

        self._core.observe(ctx.sim_time, self._state.pos, None)
        self._tree.tick()
        if self._state.done and not self._seq.finished:
            self._seq.finish()
            self._resume_configured_route()
        # Submit what we want, then execute what the model says we may. `result` precedes this tick's
        # `submit` by one step -- the plugin solves at the top of the step -- which is 2 ms and well
        # under the nav period. Doing it this way is what makes plugin order in the world irrelevant.
        entity = ctx.entities.get(self.entity)
        self._ensure_solved(ctx)
        self._avoid.submit(
            self._agent,
            self._state.pos,
            self._velocity(ctx),
            self._core.pref_vel,
            present=bool(entity.present) if entity is not None else True,
        )
        self._output.emit(ctx, self._avoid.result(self._agent), yaw, step_dt)

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

    def _join_avoidance(self, ctx) -> None:
        """Join the world's avoidance model, once every plugin has configured. Idempotent."""
        if self._agent != NO_AGENT:
            return
        self._avoid = ctx.blackboard.get(MODEL_KEY) or NullAvoidance()
        cfg = self.config
        self._agent = self._avoid.add_agent(
            self.entity,
            radius=float(cfg.get("radius", 0.3)),
            max_speed=float(cfg.get("max_speed", max(1.0, 2.0 * self._state.speed))),
            # Apparatus yields; the subject does not. Opting out still occupies an agent, so others
            # go round this mover -- it simply never gives way itself, which is what keeps a
            # strictly reproducible opponent reproducible.
            yields=bool(cfg.get("avoidance", False)),
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
        st.loop = False  # a commanded route runs once; looping is a property of the configured one
        self._commanded = True
        self._core.reset()
        self._caution.reset()
        self._started = True
        self._seq.apply(seq)

    def _apply_cancel(self, seq: int) -> None:
        x, y, _yaw = self._output.pose(self._ctx)
        self._state.waypoints = np.asarray([(x, y)], dtype=float)
        self._core.reset()
        self._state.done = True
        self._output.stop(self._ctx)
        self._seq.apply(seq)
        self._seq.finish()

    def shutdown(self, ctx: SimContext) -> None:
        if self._output is not None:
            self._output.stop(ctx)
