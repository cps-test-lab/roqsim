"""Scene + controller plugin: a kinematic pedestrian that patrols a route or is driven to goals.

Ported from our earlier in-house nav prototype's pedestrian stack (``humanoid`` + ``pedestrian.controller``), split into the
roqsim plugin model: this plugin builds one walker's mocap bodies + skin into the ``MjSpec``,
registers it as an ``Entity(kind='pedestrian')``, and declares a backend-neutral goal
:class:`~roqsim.context.Endpoint` any bridge can serve (the ROS 2 bridge serves it as
``nav2_msgs/NavigateThroughPoses`` -- see ``roqsim_walker_ros``).

Config::

    walker:
      walker: MaleVisitorWalk  # blueprint folder under models/people/ (required)
      namespace: ""            # transport scope for the goal endpoint
      outfit: B                # clothing variant: a letter, or {pants: C, jacket: A}
      skin: true               # false -> capsule visuals instead of the character mesh
      speed: 1.2               # m/s; past ~1.7 the run clip blends in
      pos: [0.0, 0.0]          # spawn, used when `waypoints` is empty (goal-driven only)
      waypoints:               # patrol route; the walker starts at waypoints[0]
        - [-2.5, -2.5]
        - [ 2.5, -2.5, [3, 6]] # optional per-waypoint dwell: secs, or [lo, hi] random pause
      loop: true               # cycle the patrol forever
      dwell: 0.0               # default dwell applied to every waypoint
      arrival_radius: 0.25
      avoidance: false         # true -> ORCA local avoidance (needs the [avoidance] extra)
      robot_body: base_link    # body the walker yields to (default: the robot entity's base)
      robot_radius: 0.25
      goal_endpoint: true      # false -> patrol only; declares no goal endpoint, so a bridge needs
                               #   no handler for it (a patrol-only world drops the nav2_msgs dep)
      action_name: navigate_through_poses   # relative action name of the goal endpoint
      orca: {neighbor_dist: 4.0, time_horizon: 3.0, radius: 0.26, max_speed: 1.6}
      planner: {inflation_radius: 0.3, waypoint_radius: 0.3}
      recovery: {stuck_time: 1.5, backup_time: 0.5, max_recovery: 4}
      motion: {walk: /abs/walk.npz}         # override a resolved locomotion clip

Several ``walker`` plugins may coexist: they share one :class:`~roqsim_walker.nav.controller.
WalkerController` (so ORCA sees every walker, the robot and any mocap props in one simulation). The
first instance to initialise owns the per-step tick; the rest only contribute their spec.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass

import mujoco

from roqsim.context import Endpoint, Entity, SimContext
from roqsim.plugin import Plugin
from roqsim_walker.blueprint import BlueprintError, resolve_walker
from roqsim_walker.humanoid import JOINT_NAMES, build_humanoid
from roqsim_walker.nav.controller import WalkerController

# Blackboard keys for the state shared by every ``walker`` instance in a world.
_SPECS_KEY = "walker:_specs"
_CONTROLLER_KEY = "walker:_controller"
_DRIVER_KEY = "walker:_driver"


@dataclass
class WalkerHandle:
    """Published by :class:`WalkerPlugin` under ``walker:<name>``; consumed by goal-driven
    interfaces (the ROS 2 ``NavigateThroughPoses`` action handler) and in-process drivers.

    ``send_route`` is thread-safe: it stamps the route with a monotonically increasing sequence
    number, marshals the change onto the physics thread, and returns that number immediately. Poll
    :attr:`status` until it reports the same sequence as *finished* to know the walker arrived; a
    larger sequence means a newer goal preempted this one.
    """

    name: str
    send_route: Callable[[list], int]  # poses [(x, y[, yaw]), ...] -> route sequence number
    cancel_route: Callable[[], int]  # -> route sequence number
    status: Callable[[], tuple[int, bool, int, float]]  # (seq, finished, goals_left, dist_left)


class WalkerPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.walker_name = self.address
        self.blueprint = self.config.get("walker")
        self._spec: dict = {}
        self._ctx: SimContext | None = None
        self._seq = itertools.count(1)  # route sequence numbers (1, 2, 3, ...)
        self._seq_lock = threading.Lock()

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if not config.get("walker"):
            errors.append("'walker' is required (a blueprint folder under models/people/)")
        else:
            try:
                resolve_walker(config["walker"], outfit=config.get("outfit"))
            except BlueprintError as exc:
                errors.append(str(exc))
        for i, wp in enumerate(config.get("waypoints") or []):
            ok = (isinstance(wp, dict) and len(wp.get("pos", ())) == 2) or (
                isinstance(wp, (list, tuple)) and len(wp) >= 2
            )
            if not ok:
                errors.append(f"waypoints[{i}] must be [x, y], [x, y, dwell] or {{pos: [x, y]}}")
        if not config.get("waypoints") and len(config.get("pos", [0.0, 0.0])) != 2:
            errors.append("'pos' must be [x, y]")
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        """Inject this walker's mocap bodies (one per skeleton joint) + its character skin."""
        cfg = self.config
        blueprint = resolve_walker(
            self.blueprint, outfit=cfg.get("outfit"), motion=cfg.get("motion")
        )
        use_skin = cfg.get("skin", True)
        kw = {}
        if cfg.get("rgba") is not None:
            kw["rgba"] = tuple(cfg["rgba"])
        build_humanoid(
            spec,
            name=self.walker_name,
            mesh=blueprint["mesh"] if use_skin else None,
            materials=blueprint["materials"] if use_skin else None,
            tpose=blueprint["tpose"],
            flip=blueprint["flip"],
            skeleton=blueprint["skeleton"],
            collision=blueprint["collision"],
            **kw,
        )

        # The controller spec = this plugin's config + everything the blueprint resolved.
        self._spec = {
            **{
                k: cfg[k]
                for k in (
                    "speed",
                    "loop",
                    "dwell",
                    "arrival_radius",
                    "avoidance",
                    "orca",
                    "planner",
                    "recovery",
                    "waypoints",
                    "pos",
                )
                if k in cfg
            },
            "name": self.walker_name,
            "skeleton": blueprint["skeleton"],
            "sole": blueprint["sole"],
            "motion": blueprint["motion"],
        }
        specs = ctx.blackboard.get(_SPECS_KEY) or []
        specs.append(self._spec)
        ctx.blackboard.set(_SPECS_KEY, specs)

    def configure(self, ctx: SimContext) -> None:
        """Register the entity, the blackboard handle and the goal endpoint.

        The shared controller is *not* built here: it needs the robot's entity, which a
        ``spawn_robot`` plugin listed after this one has yet to register. It is created lazily on the
        first ``on_reset``/``pre_step``, by which point every plugin has configured.
        """
        self._ctx = ctx
        ns = self.config.get("namespace", "")
        ctx.entities.add(
            Entity(
                name=self.walker_name,
                kind="pedestrian",
                body=f"{self.walker_name}/pelvis",
                meta={
                    "walker": self.blueprint,
                    "namespace": ns,
                    "waypoints": list(self.config.get("waypoints") or []),
                    "avoidance": bool(self.config.get("avoidance", False)),
                },
            )
        )
        ctx.blackboard.set(
            f"walker:{self.walker_name}",
            WalkerHandle(
                name=self.walker_name,
                send_route=self.send_route,
                cancel_route=self.cancel_route,
                status=self.status,
            ),
        )

        # Publish the walker's live bone poses so a viewer can animate its skinned mesh. The walker is
        # mocap-driven (no MuJoCo joints, so nothing to put on /joint_states); instead each of the 17
        # skeleton bodies is broadcast as its own transform on /tf. A bridge with a TFMessage converter
        # (the ROS 2 bridge has one) bundles them into one message per tick. The bones are world-frame
        # (mocap bodies are world children), so each is a flat child of the world/map frame.
        self._body_ids = [
            (name, mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in (f"{self.walker_name}/{j}" for j in JOINT_NAMES)
        ]
        ctx.interface.add(
            Endpoint(
                name="body_poses",
                direction="out",
                owner=self.walker_name,
                namespace=ns,
                read=self.read_body_poses,
                rate_hz=30.0,
                backend={
                    "ros2": {
                        "type": "tf2_msgs.msg.TFMessage",
                        "topic": "/tf",  # absolute: the shared TF topic, never namespaced
                        "frame_id": "map",
                    }
                },
            )
        )

        # Goal route as a backend-neutral *action* endpoint: the hint names the action type as a
        # string; a bridge with a handler for it (see roqsim_walker_ros.actions) runs the goal and
        # feeds the pose list through ``write`` as a neutral [(x, y, yaw), ...] payload.
        #
        # A patrol-only world sets `goal_endpoint: false` so no endpoint is declared -- the bridge
        # then needs no handler (and no nav2_msgs) for this walker. Declaring the endpoint without a
        # handler installed is a hard error at bridge start-up, by design.
        if not self.config.get("goal_endpoint", True):
            return
        action = self.config.get("action_name", "navigate_through_poses")
        ctx.interface.add(
            Endpoint(
                name="navigate_through_poses",
                direction="in",
                owner=self.walker_name,
                namespace=ns,
                write=self._write_route,
                backend={
                    "ros2": {
                        "action": "nav2_msgs.action.NavigateThroughPoses",
                        "name": action,
                    }
                },
            )
        )

    def read_body_poses(self):
        """Endpoint ``read`` (physics thread): ``[(frame, pos[3], quat[4]), ...]`` for the 17 bones.

        World transforms straight from ``data.xpos``/``xquat`` (mocap bodies are world children).
        ``frame`` is the body name (== the exported scene body name), so the viewer binds each
        transform to its bone node by name. ``quat`` is MuJoCo (w, x, y, z)."""
        d = self._ctx.data
        return [(name, d.xpos[bid], d.xquat[bid]) for name, bid in self._body_ids if bid >= 0]

    def on_reset(self, ctx: SimContext) -> None:
        ctrl = self._ensure_controller(ctx)
        if self._is_driver(ctx):
            ctrl.reset()  # mj_resetData parked the mocap bodies; re-pose every walker

    def pre_step(self, ctx: SimContext) -> None:
        ctrl = self._ensure_controller(ctx)
        if self._is_driver(ctx):
            ctrl.update(ctx.dt)

    # -- shared controller ---------------------------------------------------------------------
    def _ensure_controller(self, ctx: SimContext) -> WalkerController:
        """Build the one controller shared by every ``walker`` plugin, on first use."""
        ctrl = ctx.blackboard.get(_CONTROLLER_KEY)
        if ctrl is not None:
            return ctrl
        ctrl = WalkerController(
            ctx.model,
            ctx.data,
            ctx.blackboard.get(_SPECS_KEY) or [],
            robot_body=self._robot_body(ctx),
            robot_radius=float(self.config.get("robot_radius", 0.25)),
            update_hz=self.config.get(
                "update_hz"
            ),  # None -> controller default; lower it to save CPU
        )
        ctx.blackboard.set(_CONTROLLER_KEY, ctrl)
        ctx.blackboard.set(_DRIVER_KEY, id(self))  # this instance owns the per-step tick
        return ctrl

    def _robot_body(self, ctx: SimContext) -> str:
        """The body walkers yield to: an explicit ``robot_body``, else the first robot entity's
        base body, else the conventional ``base_link`` (absent -> no robot agent)."""
        explicit = self.config.get("robot_body")
        if explicit:
            return explicit
        for entity in ctx.entities.all():
            if entity.kind == "robot" and entity.body:
                return entity.body
        return "base_link"

    def _is_driver(self, ctx: SimContext) -> bool:
        return ctx.blackboard.get(_DRIVER_KEY) == id(self)

    def _controller(self) -> WalkerController | None:
        return self._ctx.blackboard.get(_CONTROLLER_KEY) if self._ctx else None

    def _next_seq(self) -> int:
        with self._seq_lock:
            return next(self._seq)

    # -- goal interface ------------------------------------------------------------------------
    def _write_route(self, poses) -> None:
        """Endpoint ``write``: already marshalled onto the physics thread by the bridge."""
        ctrl = self._controller()
        if ctrl is not None:
            ctrl.set_route(self.walker_name, poses, self._next_seq())

    def send_route(self, poses) -> int:
        """Thread-safe: stamp a route, post it to the physics thread, return its sequence number."""
        seq = self._next_seq()
        ctx = self._ctx
        if ctx is None:
            return seq
        ctx.post(lambda c, p=list(poses), s=seq: self._apply(c, "set_route", p, s))
        return seq

    def cancel_route(self) -> int:
        seq = self._next_seq()
        ctx = self._ctx
        if ctx is None:
            return seq
        ctx.post(lambda c, s=seq: self._apply(c, "cancel_route", None, s))
        return seq

    def _apply(self, ctx: SimContext, op: str, poses, seq: int) -> None:
        ctrl = ctx.blackboard.get(_CONTROLLER_KEY)
        if ctrl is None:
            return
        if op == "set_route":
            ctrl.set_route(self.walker_name, poses, seq)
        else:
            ctrl.cancel_route(self.walker_name, seq)

    def status(self) -> tuple[int, bool, int, float]:
        """``(route_seq, finished, goals_remaining, distance_remaining)``; safe to poll off-thread."""
        ctrl = self._controller()
        if ctrl is None:
            return (0, False, 0, 0.0)
        return ctrl.status(self.walker_name)
