"""Walkers that follow a route, yield to the robot (ORCA), and animate (mocap).

Ported from our earlier in-house nav prototype's ``mujoco_nav.pedestrian.controller``, with two additions for roqsim:
a runtime **goal route** interface (:meth:`WalkerController.set_route`, driving the
``NavigateThroughPoses`` endpoint) and a per-walker **avoidance toggle**.

Each walker is a kinematic articulated **humanoid** (a flat set of mocap bodies, see
:mod:`roqsim_walker.humanoid`). Two decoupled layers:

* **Nav root** -- three sub-layers, slow to fast:

  - **global plan** -- A* on an occupancy grid rasterized from the model's wall geoms
    (:mod:`~roqsim_walker.nav.planner`) routes a wall-safe path between the walker's goals;
  - **behaviour** -- a py-trees tree (:mod:`~roqsim_walker.nav.behavior`) follows that path,
    advances goals, and runs a *stuck recovery* (back up, then replan) when progress stalls;
  - **local avoidance** -- ORCA turns the behaviour's preferred velocity into a collision-free one.
    The **robot is inserted as a non-yielding agent** (its ORCA state is overwritten from ground
    truth every step, so ORCA never moves it); wall footprints become static obstacles, and other
    mocap props are pinned as immovable agents.

* **Body pose** -- a motion :class:`~roqsim_walker.motion.Clip` blendspace (idle/short/walk/run
  on a speed axis; turn + strafe on a direction axis), phased by *distance travelled* so feet don't
  slide, then written to every mocap body via forward kinematics.

**Avoidance** is per-walker (``avoidance: true``). The shared ORCA sim is built when *any* walker
enables it; a walker with ``avoidance: false`` still occupies an ORCA agent (so others steer around
it) but integrates its own preferred velocity directly instead of ORCA's. With no ``rvo2`` installed
every walker falls back to plain path following without collision avoidance.

**Routes.** A walker patrols its configured ``waypoints`` until :meth:`set_route` supplies a goal
route (from an interface such as nav2's ``NavigateThroughPoses``). The route overrides the patrol;
on arrival the walker resumes patrolling from its nearest patrol waypoint, or stands if none was
configured.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from roqsim_walker.humanoid import (
    JOINT_NAMES,
    forward_kinematics,
    quat_rotate,
    to_skeleton,
)
from roqsim_walker.motion import (
    Clip,
    blend_quats,
    procedural_idle,
    procedural_walk,
    smoothstep,
)
from roqsim_walker.nav import obstacles
from roqsim_walker.nav.behavior import NavCore, NavParams, build_tree
from roqsim_walker.nav.occupancy import OccupancyGrid
from roqsim_walker.nav.planner import GridPlanner

logger = logging.getLogger(__name__)

_BLOCKER_RANGE = 1.5  # m, how near another agent must be to bias a backup dir

# Animation smoothing (CARLA-style): ease the body, never snap it.
_SPEED_TAU = 0.15  # s, low-pass on the speed that drives the gait/blend
_MAX_TURN_RATE = 4.0  # rad/s cap on how fast the body re-faces its heading
# Rate the (expensive) walker pipeline runs at. The physics loop calls update() far more often
# (~500 Hz), so we decimate -- nav/ORCA/animation look smooth at ~60 Hz.
_UPDATE_HZ = 60.0
# Speed-axis blend windows (m/s) for CARLA's idle->short->walk->run order.
_IDLE_SHORT = (0.05, 0.25)  # idle -> short shuffle
_SHORT_WALK = (0.35, 0.90)  # short -> walk
_WALK_RUN = (1.70, 2.80)  # walk -> run
# Direction axis -- two parts of CARLA's BS_GEN3 ``Direction`` parameter:
#  (1) turn-in-place / lean: heading-rate (rad/s) for a full turn blend; when standing the in-place
#      turn clip is played through and fully takes over, when walking it is capped (``_TURN_MAX``)
#      so it only flavours (leans) the gait.
#  (2) strafe: when travel is not aligned with facing, blend the matching side / back walk clip by
#      the travel-vs-facing angle (optional clips; inert if absent).
_TURN_REF = 2.5
_TURN_MAX = 0.45
_TURN_STRIDE = math.radians(60.0)  # body heading change per turn-clip cycle (sets step cadence)
# Strafe blend windows on |travel - facing| (rad).
_STRAFE_FWD = math.radians(25.0)
_STRAFE_SIDE = math.radians(90.0)
_STRAFE_BACK = math.radians(150.0)


def _rest_foot_z(skel):
    """Lower-ankle height at this walker's rest pose -- the target a grounded clip's lowest foot
    should reach so the planted sole touches the floor."""
    ident = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    poses = forward_kinematics([0.0, 0.0, skel.root_height], 0.0, ident, skeleton=skel)
    return min(poses["ankle_l"][0][2], poses["ankle_r"][0][2])


def _ground_clip(clip, skel):
    """Lower a clip's ``root_z`` so its lowest foot over the whole cycle just reaches the rest stance
    height -- the planted foot touches the floor instead of hovering. Some retargeted clips (notably
    run) never fully plant, leaving the character floating several cm."""
    rest = _rest_foot_z(skel)
    lows = [
        min(
            forward_kinematics(
                [0.0, 0.0, skel.root_height + clip.root_z[i]],
                0.0,
                {n: clip.joint_rot[i, j] for j, n in enumerate(JOINT_NAMES)},
                skeleton=skel,
            )[a][0][2]
            for a in ("ankle_l", "ankle_r")
        )
        for i in range(clip.num_frames)
    ]
    clip.root_z = clip.root_z - (float(min(lows)) - rest)
    return clip


def _foot_rest(skel):
    """Rest-pose heel(ankle) and toe-tip heights per foot -- the z's at which that foot's sole
    touches the floor, used to ground each frame to the real contact."""
    ident = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    poses = forward_kinematics([0.0, 0.0, skel.root_height], 0.0, ident, skeleton=skel)
    tip = np.array(skel.foot_tip)
    rest = {}
    for s in ("l", "r"):
        tp, tq = poses[f"toe_{s}"]
        rest[s] = (float(poses[f"ankle_{s}"][0][2]), float((tp + quat_rotate(tq, tip))[2]))
    return rest


def _foot_ground(poses, st):
    """Vertical-only foot grounding: shift the whole body so the **lowest shoe-sole point** -- the
    measured heel/toe of either foot -- sits exactly on the floor. Tracking the real sole (not the
    ankle/toe joint, which is several cm above it) removes the residual hover."""
    tip = np.array(st.skeleton.foot_tip)
    lows = []
    for s in ("l", "r"):
        ap, aq = poses[f"ankle_{s}"]
        tp, tq = poses[f"toe_{s}"]
        if st.sole:  # exact: measured sole offsets
            heel = float((ap + quat_rotate(aq, np.array(st.sole[s]["heel"])))[2])
            toe = float((tp + quat_rotate(tq, np.array(st.sole[s]["toe"])))[2])
            lows.append(min(heel, toe))
        else:  # fallback: ankle/toe-tip joints
            toetip = float((tp + quat_rotate(tq, tip))[2])
            ra, rt = st.foot_rest[s]
            lows.append(min(ap[2] - ra, toetip - rt))
    c = min(lows)
    if abs(c) > 1e-6:
        for nm, (p, q) in poses.items():
            poses[nm] = (p - np.array([0.0, 0.0, c]), q)
    return poses


def _wp_xy(p) -> tuple[float, float]:
    """World (x, y) of a waypoint entry: ``[x, y]``, ``[x, y, dwell]``, or ``{pos: [x, y], dwell:
    ...}``."""
    if isinstance(p, dict):
        return float(p["pos"][0]), float(p["pos"][1])
    return float(p[0]), float(p[1])


def _dwell_range(d) -> tuple[float, float]:
    """Coerce a dwell spec (seconds, or ``[lo, hi]`` for a random pause) to a tuple."""
    if isinstance(d, (list, tuple)):
        return (float(d[0]), float(d[1]))
    return (float(d), float(d))


def _wp_dwell(p, default) -> tuple[float, float]:
    """Per-waypoint dwell ``(lo, hi)`` seconds: from the entry's ``dwell`` (dict or 3rd element),
    else the walker's default."""
    if isinstance(p, dict):
        return _dwell_range(p.get("dwell", default))
    if len(p) >= 3:
        return _dwell_range(p[2])
    return _dwell_range(default)


def _heading(a, b) -> float:
    return math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))


def _approach_angle(cur, target, max_step) -> float:
    """Move ``cur`` toward ``target`` by at most ``max_step`` (shortest way)."""
    d = math.atan2(math.sin(target - cur), math.cos(target - cur))
    if abs(d) <= max_step:
        return target
    return cur + math.copysign(max_step, d)


@dataclass
class _Walker:
    name: str
    waypoints: np.ndarray  # (N, 2) -- the active route (patrol or commanded goals)
    speed: float
    loop: bool
    arrival_radius: float
    radius: float
    max_speed: float
    neighbor_dist: float
    time_horizon: float
    walk: Clip
    idle: Clip
    run: Clip
    mocap: dict  # part name -> mocap id
    skeleton: object = None  # this walker's per-rig bone table (humanoid.Skeleton)
    avoid: bool = False  # yields to the robot / other agents via ORCA
    short: Clip = None  # slow-shuffle gait (idle->walk transition)
    turn_l: Clip = None  # in-place turn-left / turn-right (lean into turns)
    turn_r: Clip = None
    walk_l: Clip = None  # strafe-left / strafe-right / back-pedal (direction axis);
    walk_r: Clip = None  # optional -- inert when the clips are absent
    walk_back: Clip = None
    foot_rest: dict = None  # per-foot rest heel/toe-tip z (grounding fallback)
    sole: dict = None  # per-foot measured shoe-sole offsets -> exact grounding
    dwell: list = None  # per-waypoint (lo, hi) seconds to stand-idle on arrival
    # -- patrol backup, restored when a commanded route finishes ---------------------------------
    patrol_wps: np.ndarray = None
    patrol_dwell: list = None
    patrol_loop: bool = True
    route_active: bool = False
    route_seq: int = 0  # generation of the last applied route (0 = none)
    # Latched on route completion. ``done`` cannot serve: restoring the patrol clears it, so an
    # interface polling ``status()`` would never observe the arrival it is waiting for.
    route_finished: bool = False
    # -- runtime ---------------------------------------------------------------------------------
    pos: np.ndarray = field(default=None)
    goal_idx: int = 1  # index of the targeted waypoint (a goal)
    path: list = None  # current planned sub-path to that goal (world xy)
    path_idx: int = 0  # index of the next path point to chase
    yaw: float = 0.0
    phase: float = 0.0  # walk-clip phase (cycles, advanced by distance)
    phase_run: float = 0.0
    phase_short: float = 0.0
    phase_turn: float = 0.0  # turn-in-place phase (cycles, advanced by |heading change|)
    t_idle: float = 0.0  # idle-clip clock (s, advanced by time)
    disp_speed: float = 0.0  # low-passed body speed (drives gait + blend)
    pref_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    done: bool = False  # finished a non-looping route -> stand
    agent: int = None  # ORCA agent id


class WalkerController:
    """Drives every walker humanoid each physics step: nav root + clip body.

    ``specs`` is a list of walker spec dicts (see :meth:`_make_state` for the keys); they come from
    the ``walker`` plugin's YAML config merged with its resolved blueprint.
    """

    def __init__(
        self, model, data, specs, robot_body="base_link", robot_radius=0.25, update_hz=None
    ):
        self.model, self.data = model, data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        self._robot_bid = bid if bid >= 0 else None
        specs = specs or []
        # One obstacle source for both layers: the model's static wall geoms.
        mujoco.mj_forward(model, data)  # ensure geom_xpos is valid
        self._wall_polys = obstacles.wall_polygons(model, data)
        self._grid = self._build_grid(self._wall_polys, specs)
        if self._grid is not None:
            logger.info(
                "walker planner grid %dx%d from %d wall footprint(s)",
                self._grid.width,
                self._grid.height,
                len(self._wall_polys),
            )
        self._states = [self._make_state(s) for s in specs]
        self._by_name = {st.name: st for st in self._states}
        self._navs = {
            st.name: self._make_nav(st, s) for st, s in zip(self._states, specs, strict=True)
        }
        self._trees = {name: build_tree(core) for name, core in self._navs.items()}
        self._sim = None
        self._robot_agent = None
        self._obstacle_agents = []  # (orca agent, body id) for mocap props
        self._warned = False
        # Decimate the whole pipeline to this rate. Runs no faster than the viewer needs it: a walker
        # publishes its bones at 30 Hz, so a higher rate just computes frames that are dropped before
        # publish. Overridable per-world via the ``update_hz`` config (default keeps the 60 Hz look).
        self._period = 1.0 / (update_hz or _UPDATE_HZ)
        self._accum = 0.0  # sim time since last tick
        if any(st.avoid for st in self._states):
            self._build_orca(robot_radius)
        self.reset()  # pose humanoids at start

    # -- setup ---------------------------------------------------------------------------------
    def _build_grid(self, wall_polys, specs):
        """An occupancy grid covering the walls and every waypoint, or ``None`` when there are no
        walls to plan around (-> straight-line goals)."""
        if not wall_polys:
            return None
        pts = [p for poly in wall_polys for p in poly]
        for spec in specs:
            pts.extend(_wp_xy(p) for p in self._spec_waypoints(spec))
        arr = np.asarray(pts, dtype=float)
        bounds = (arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max())
        return OccupancyGrid.from_polygons(wall_polys, bounds=bounds)

    @staticmethod
    def _spec_waypoints(spec) -> list:
        """The spec's patrol waypoints, or a single spawn point when it is goal-driven only."""
        wps = list(spec.get("waypoints") or [])
        if wps:
            return wps
        return [list(spec.get("pos") or (0.0, 0.0))]

    def _make_nav(self, st, spec) -> NavCore:
        params = NavParams.from_spec(spec, st.radius)
        planner = (
            GridPlanner(self._grid, params.inflation_radius) if self._grid is not None else None
        )
        return NavCore(st, planner, params)

    def _make_state(self, spec) -> _Walker:
        default_dwell = spec.get("dwell", 0.0)
        raw = self._spec_waypoints(spec)
        wps = np.array([_wp_xy(p) for p in raw], dtype=float)
        dwell = [_wp_dwell(p, default_dwell) for p in raw]
        orca = spec.get("orca") or {}
        speed = float(spec.get("speed", 1.0))
        skel = to_skeleton(spec.get("skeleton"))
        loop = bool(spec.get("loop", True)) and len(wps) > 1
        st = _Walker(
            name=spec["name"],
            waypoints=wps,
            speed=speed,
            loop=loop,
            arrival_radius=float(spec.get("arrival_radius", 0.25)),
            radius=float(orca.get("radius", 0.26)),
            max_speed=float(orca.get("max_speed", round(max(speed * 1.5, 1.0), 3))),
            neighbor_dist=float(orca.get("neighbor_dist", 4.0)),
            time_horizon=float(orca.get("time_horizon", 3.0)),
            avoid=bool(spec.get("avoidance", False)),
            walk=self._load_clip(spec, "walk", procedural_walk, skel),
            idle=self._load_clip(spec, "idle", procedural_idle, skel),
            run=self._load_clip(spec, "run", procedural_walk, skel),
            short=self._load_clip(spec, "short", procedural_walk, skel),
            turn_l=self._opt_clip(spec, "turn_l", skel),
            turn_r=self._opt_clip(spec, "turn_r", skel),
            walk_l=self._opt_clip(spec, "walk_l", skel),
            walk_r=self._opt_clip(spec, "walk_r", skel),
            walk_back=self._opt_clip(spec, "walk_back", skel),
            mocap=self._mocap_ids(spec["name"]),
            skeleton=skel,
            foot_rest=_foot_rest(skel),
            sole=spec.get("sole"),
            dwell=dwell,
            patrol_wps=wps.copy(),
            patrol_dwell=list(dwell),
            patrol_loop=loop,
        )
        st.pos = wps[0].copy()
        st.yaw = _heading(wps[0], wps[1]) if len(wps) > 1 else 0.0
        return st

    def _load_clip(self, spec, kind, fallback, skel) -> Clip:
        path = (spec.get("motion") or {}).get(kind)
        if path:
            try:
                return _ground_clip(Clip.load(path), skel)
            except Exception as e:  # noqa: BLE001 -- fall back, log why
                logger.warning(
                    "walker %s: could not load %s clip %s (%s); using procedural",
                    spec.get("name"),
                    kind,
                    path,
                    e,
                )
        return _ground_clip(fallback(), skel)

    def _opt_clip(self, spec, kind, skel):
        """Load an optional clip (e.g. a turn) -> grounded Clip or None."""
        path = (spec.get("motion") or {}).get(kind)
        if not path:
            return None
        try:
            return _ground_clip(Clip.load(path), skel)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "walker %s: could not load %s clip %s (%s)", spec.get("name"), kind, path, e
            )
            return None

    def _mocap_ids(self, name) -> dict:
        ids = {}
        for part in JOINT_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{part}")
            if bid < 0:
                raise ValueError(
                    f"humanoid body {name}/{part} not in model "
                    "(was build_humanoid run before compile?)"
                )
            ids[part] = int(self.model.body_mocapid[bid])
        return ids

    def _build_orca(self, robot_radius):
        try:
            import rvo2
        except ImportError:
            logger.warning(
                "walker: avoidance requested but 'rvo2' is not installed; walkers navigate "
                "WITHOUT collision avoidance. Install with: pip install 'roqsim_walker[avoidance]'"
            )
            return  # fallback handled in update()
        dt = float(self.model.opt.timestep)
        self._sim = rvo2.PyRVOSimulator(dt, 4.0, 10, 3.0, 2.0, 0.26, 1.5)
        for st in self._states:
            st.agent = self._sim.addAgent(
                (float(st.pos[0]), float(st.pos[1])),
                st.neighbor_dist,
                10,
                st.time_horizon,
                2.0,
                st.radius,
                st.max_speed,
                (0.0, 0.0),
            )
        if self._robot_bid is not None:
            rx, ry, _, _ = self._robot_state()
            self._robot_agent = self._sim.addAgent(
                (rx, ry), 4.0, 10, 1.0, 2.0, float(robot_radius), 2.0, (0.0, 0.0)
            )
        self._add_dynamic_obstacles()
        self._add_static_obstacles()

    def _add_dynamic_obstacles(self):
        """Insert each runtime-teleported mocap prop as an immovable ORCA agent, the same way the
        robot is: its position is overwritten from ground truth every step, so ORCA never moves it
        and walkers steer around it."""
        walker_mocapids = {mid for st in self._states for mid in st.mocap.values()}
        for bid, radius in obstacles.dynamic_obstacle_bodies(self.model, walker_mocapids):
            px, py = float(self.data.xpos[bid][0]), float(self.data.xpos[bid][1])
            agent = self._sim.addAgent((px, py), 4.0, 10, 1.0, 2.0, float(radius), 0.0, (0.0, 0.0))
            self._obstacle_agents.append((agent, bid))
        if self._obstacle_agents:
            logger.info("ORCA: %d dynamic obstacle agent(s)", len(self._obstacle_agents))

    def _add_static_obstacles(self):
        """Give ORCA the model's wall footprints (CCW = solid, agents stay outside) so local
        avoidance can't push a human through a wall -- the same polygons the planner grid was built
        from."""
        added = 0
        for poly in self._wall_polys:
            if len(poly) >= 2:
                self._sim.addObstacle(poly)
                added += 1
        if added:
            logger.info("ORCA: %d wall obstacle(s)", added)
            self._sim.processObstacles()

    # -- goal-route interface (called on the physics thread) -----------------------------------
    def set_route(self, name: str, poses, seq: int) -> None:
        """Send ``name`` through ``poses`` (a list of ``(x, y)`` or ``(x, y, yaw)``), overriding its
        patrol. ``seq`` stamps this route so an interface can tell when it was applied and whether a
        later goal has superseded it."""
        st = self._by_name.get(name)
        if st is None:
            logger.warning("set_route: no walker named %r", name)
            return
        pts = [(float(p[0]), float(p[1])) for p in poses]
        if not pts:
            self.cancel_route(name, seq)
            return
        # waypoints[0] is the start (the walker's current pose); goals follow, matching the patrol
        # convention that NavCore.reset() targets index 1.
        st.waypoints = np.array([tuple(st.pos), *pts], dtype=float)
        st.dwell = [(0.0, 0.0)] * len(st.waypoints)
        st.loop = False
        st.route_active = True
        st.route_finished = False
        st.route_seq = int(seq)
        self._navs[name].reset()

    def cancel_route(self, name: str, seq: int = 0) -> None:
        """Stop ``name`` where it is and drop any active route (it stands until told otherwise)."""
        st = self._by_name.get(name)
        if st is None:
            return
        st.waypoints = np.array([tuple(st.pos)], dtype=float)
        st.dwell = [(0.0, 0.0)]
        st.loop = False
        st.route_active = False
        st.route_finished = True
        st.route_seq = int(seq)
        self._navs[name].reset()
        st.done = True
        st.pref_vel = np.zeros(2)

    def status(self, name: str) -> tuple[int, bool, int, float]:
        """``(route_seq, finished, goals_remaining, distance_remaining)`` -- a cheap, lock-free
        snapshot an interface thread can poll for action feedback."""
        st = self._by_name.get(name)
        if st is None:
            return (0, True, 0, 0.0)
        if st.route_finished:
            return (int(st.route_seq), True, 0, 0.0)
        goals, dist = self._navs[name].remaining()
        return (int(st.route_seq), False, goals, float(dist))

    def _restore_patrol(self, st) -> None:
        """A commanded route finished: resume the configured patrol from the nearest waypoint, or
        stand still when the walker had no patrol."""
        st.route_active = False
        if st.patrol_wps is None or len(st.patrol_wps) <= 1:
            return  # goal-driven only: stay put (st.done stays True)
        st.waypoints = st.patrol_wps.copy()
        st.dwell = list(st.patrol_dwell)
        st.loop = st.patrol_loop
        core = self._navs[st.name]
        core.reset()  # clears done/path; sets goal_idx = 1
        # Re-enter the loop at whichever patrol waypoint is closest to where the route left us.
        d = np.linalg.norm(st.waypoints - st.pos, axis=1)
        st.goal_idx = int(np.argmin(d))

    # -- per-step ------------------------------------------------------------------------------
    def update(self, dt):
        # The physics loop calls us every step (~500 Hz); decimate the whole walker pipeline to
        # _UPDATE_HZ by accumulating elapsed time and only doing the real work once a full period
        # has built up. Mocap bodies hold their last pose in between (~16 ms at 60 Hz).
        self._accum += dt
        if self._accum < self._period:
            return
        step_dt = self._accum
        self._accum = 0.0
        t = float(self.data.time)

        if self._sim is None:
            if not self._warned and any(st.avoid for st in self._states):
                logger.warning("rvo2 unavailable; walkers navigate WITHOUT collision avoidance")
                self._warned = True
            for st in self._states:
                pref = self._navigate(st, t)
                self._animate(st, st.pos + pref * step_dt, step_dt)
                self._finish_route(st)
            return

        # Pin the non-yielding agents (robot, mocap props) to ground truth so ORCA never moves them.
        if self._robot_agent is not None:
            rx, ry, vx, vy = self._robot_state()
            self._sim.setAgentPosition(self._robot_agent, (rx, ry))
            self._sim.setAgentVelocity(self._robot_agent, (vx, vy))
            self._sim.setAgentPrefVelocity(self._robot_agent, (vx, vy))
        for agent, bid in self._obstacle_agents:
            ox, oy = float(self.data.xpos[bid][0]), float(self.data.xpos[bid][1])
            self._sim.setAgentPosition(agent, (ox, oy))
            self._sim.setAgentVelocity(agent, (0.0, 0.0))
            self._sim.setAgentPrefVelocity(agent, (0.0, 0.0))

        prefs = {}
        for st in self._states:
            pref = self._navigate(st, t)
            prefs[st.name] = pref
            if st.avoid:
                self._sim.setAgentPrefVelocity(st.agent, (float(pref[0]), float(pref[1])))
            else:
                # Not yielding: hold ORCA's copy of this walker at its own integrated pose, so peers
                # still steer around it but it never gets pushed.
                nxt = st.pos + pref * step_dt
                self._sim.setAgentPosition(st.agent, (float(nxt[0]), float(nxt[1])))
                self._sim.setAgentVelocity(st.agent, (float(pref[0]), float(pref[1])))
                self._sim.setAgentPrefVelocity(st.agent, (float(pref[0]), float(pref[1])))
        self._sim.setTimeStep(step_dt)  # ORCA integrates by this, not by update()'s arg
        self._sim.doStep()
        for st in self._states:
            if st.avoid:
                px, py = self._sim.getAgentPosition(st.agent)
                new_pos = np.array([px, py])
            else:
                new_pos = st.pos + prefs[st.name] * step_dt
            self._animate(st, new_pos, step_dt)
            self._finish_route(st)

    def _finish_route(self, st) -> None:
        if st.route_active and st.done:
            st.route_finished = True  # latch before _restore_patrol clears ``done``
            self._restore_patrol(st)

    def _navigate(self, st, t) -> np.ndarray:
        """Tick one walker's behaviour tree and return its preferred velocity."""
        core = self._navs[st.name]
        core.observe(t, st.pos, self._nearest_blocker(st))
        self._trees[st.name].tick()
        st.pref_vel = np.asarray(core.pref_vel, dtype=float)  # desired heading source
        return core.pref_vel

    def _nearest_blocker(self, st):
        """World xy of the closest other agent (robot or peer) within ``_BLOCKER_RANGE``, or ``None``
        -- used to bias the backup direction."""
        best, best_d = None, _BLOCKER_RANGE
        if self._robot_bid is not None:
            rx, ry, _, _ = self._robot_state()
            d = math.hypot(rx - st.pos[0], ry - st.pos[1])
            if d < best_d:
                best, best_d = (rx, ry), d
        for other in self._states:
            if other is st:
                continue
            d = math.hypot(other.pos[0] - st.pos[0], other.pos[1] - st.pos[1])
            if d < best_d:
                best, best_d = (other.pos[0], other.pos[1]), d
        return best

    def _animate(self, st, new_pos, dt):
        """Ease the body toward the new nav position along CARLA's 2D blendspace: a speed axis
        (idle->short->walk->run) and a direction axis (turn-in-place when stopped / lean when moving,
        plus strafe) -- then FK + write the pose."""
        delta = new_pos - st.pos
        dist = float(np.linalg.norm(delta))
        raw_speed = dist / max(dt, 1e-9)
        st.disp_speed += (1.0 - math.exp(-dt / _SPEED_TAU)) * (raw_speed - st.disp_speed)
        # Face where the walker *wants* to go (nav preferred velocity), not the instantaneous ORCA
        # push: avoidance side-steps then become a strafe/lean (direction axis) rather than the body
        # whipping around, and a walker that is blocked (~0 displacement) still turns in place to
        # face its goal.
        pref = st.pref_vel
        if float(np.hypot(pref[0], pref[1])) > 1e-3:
            target = math.atan2(float(pref[1]), float(pref[0]))
        elif dist > 1e-4:
            target = math.atan2(float(delta[1]), float(delta[0]))
        else:
            target = st.yaw
        new_yaw = _approach_angle(st.yaw, target, _MAX_TURN_RATE * dt)
        d_yaw = math.atan2(math.sin(new_yaw - st.yaw), math.cos(new_yaw - st.yaw))
        yaw_rate = d_yaw / max(dt, 1e-9)
        st.yaw = new_yaw
        st.phase += dist / st.walk.stride_len  # gait phases by distance (no slide)
        st.phase_run += dist / st.run.stride_len
        st.phase_short += dist / st.short.stride_len
        st.t_idle += dt  # idle by time
        # -- speed axis: idle -> short -> walk -> run (each takes over in turn) --
        qi, zi = st.idle.sample_array(st.t_idle / st.idle.duration)
        qs, zs = st.short.sample_array(st.phase_short)
        qw, zw = st.walk.sample_array(st.phase)
        qr, zr = st.run.sample_array(st.phase_run)
        w_short = smoothstep(st.disp_speed, *_IDLE_SHORT)
        w_walk = smoothstep(st.disp_speed, *_SHORT_WALK)
        w_run = smoothstep(st.disp_speed, *_WALK_RUN)
        q = blend_quats(qi, qs, w_short)
        root_z = (1 - w_short) * zi + w_short * zs
        q = blend_quats(q, qw, w_walk)
        root_z = (1 - w_walk) * root_z + w_walk * zw
        q = blend_quats(q, qr, w_run)
        root_z = (1 - w_run) * root_z + w_run * zr
        q = self._direction_axis(st, q, d_yaw, yaw_rate, w_short, w_walk, delta, dist)
        joint_rot = {name: q[j] for j, name in enumerate(JOINT_NAMES)}
        poses = forward_kinematics(
            [new_pos[0], new_pos[1], st.skeleton.root_height + root_z],
            st.yaw,
            joint_rot,
            skeleton=st.skeleton,
        )
        self._write(st, _foot_ground(poses, st))
        st.pos = new_pos

    def _direction_axis(self, st, q, d_yaw, yaw_rate, w_short, w_walk, delta, dist):
        """CARLA's BS_GEN3 ``Direction`` parameter, in two parts.

        **Turn:** the in-place turn clip, played through (its cycle advanced by the actual heading
        change so the steps don't slide). When the walker is standing (``w_short`` ~ 0) it fully
        takes over -> a real turn-in-place; when walking it is capped to ``_TURN_MAX`` so it only
        leans the gait into the turn.

        **Strafe:** when travel is not aligned with facing (the walker re-faced toward its goal while
        ORCA pushes it sideways), blend the matching side / back walk clip by the travel-vs-facing
        angle. These clips are optional; absent, the body simply translates along the small residual
        deviation.
        """
        turning = min(abs(yaw_rate) / _TURN_REF, 1.0)
        turn = st.turn_l if d_yaw > 0 else st.turn_r
        if turn is not None and turning > 1e-3:
            st.phase_turn += abs(d_yaw) / _TURN_STRIDE
            w_turn = turning * ((1.0 - w_short) + w_short * _TURN_MAX)
            q = blend_quats(q, turn.sample_array(st.phase_turn)[0], w_turn)
        if dist > 1e-4 and w_walk > 1e-3:
            travel = math.atan2(float(delta[1]), float(delta[0]))
            direction = math.atan2(math.sin(travel - st.yaw), math.cos(travel - st.yaw))
            a = abs(direction)
            side = st.walk_l if direction > 0 else st.walk_r
            if side is not None:
                w_side = smoothstep(a, _STRAFE_FWD, _STRAFE_SIDE) * w_walk
                if w_side > 1e-3:
                    q = blend_quats(q, side.sample_array(st.phase)[0], w_side)
            if st.walk_back is not None:
                w_back = smoothstep(a, _STRAFE_SIDE, _STRAFE_BACK) * w_walk
                if w_back > 1e-3:
                    q = blend_quats(q, st.walk_back.sample_array(st.phase)[0], w_back)
        return q

    def _write(self, st, poses):
        d = self.data
        for part, (pos, quat) in poses.items():  # per-limb collision rides the joints
            mid = st.mocap[part]
            d.mocap_pos[mid] = pos
            d.mocap_quat[mid] = quat

    # -- lifecycle -----------------------------------------------------------------------------
    def reset(self):
        """Return every walker to its patrol start and pose it standing."""
        self._accum = 0.0  # restart decimation clock
        for st in self._states:
            st.waypoints = st.patrol_wps.copy()
            st.dwell = list(st.patrol_dwell)
            st.loop = st.patrol_loop
            st.route_active = False
            st.route_finished = False
            st.route_seq = 0
            st.pos = st.waypoints[0].copy()
            st.yaw = _heading(st.waypoints[0], st.waypoints[1]) if len(st.waypoints) > 1 else st.yaw
            st.phase = 0.0
            st.phase_run = 0.0
            st.phase_short = 0.0
            st.phase_turn = 0.0
            st.t_idle = 0.0
            st.disp_speed = 0.0
            st.pref_vel = np.zeros(2)
            self._navs[st.name].reset()  # goal/path/recovery state
            if self._sim is not None and st.agent is not None:
                self._sim.setAgentPosition(st.agent, (float(st.pos[0]), float(st.pos[1])))
                self._sim.setAgentVelocity(st.agent, (0.0, 0.0))
            joint_rot, root_z = st.idle.sample(0.0)
            poses = forward_kinematics(
                [st.pos[0], st.pos[1], st.skeleton.root_height + root_z],
                st.yaw,
                joint_rot,
                skeleton=st.skeleton,
            )
            self._write(st, _foot_ground(poses, st))

    # -- helpers -------------------------------------------------------------------------------
    def _robot_state(self):
        bid = self._robot_bid
        pos = self.data.xpos[bid]
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        return float(pos[0]), float(pos[1]), float(vel[3]), float(vel[4])
