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
      avoidance: false         # true -> the shared local model gives way for it. A walker
                               #   steers or does nothing; it has never looked ahead, so this
                               #   never makes it stop. Write a `navigator` with
                               #   `avoidance: {stop: true}` for one that should.
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
import numpy as np

from roqsim.config import PluginSpec
from roqsim.context import Endpoint, Entity, SimContext
from roqsim.plugin import Plugin, PluginError
from roqsim_nav.avoidance import DEFAULT_MODEL
from roqsim_walker.blueprint import BlueprintError, resolve_walker
from roqsim_walker.humanoid import JOINT_NAMES, build_humanoid, forward_kinematics
from roqsim_walker.nav.controller import (
    _foot_ground as foot_ground,
)
from roqsim_walker.nav.controller import (
    _heading,
    make_anim_state,
    write_pose,
)
from roqsim_walker.output import STATE_KEY

# Blackboard keys for the state shared by every ``walker`` instance in a world.
_SPECS_KEY = "walker:_specs"


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

    # -- expansion -----------------------------------------------------------------------------
    #: Legacy ``walker:`` keys and where they now live on the nested ``navigator``. The walker used
    #: to own navigation itself; it now owns the body, and one navigator serves it, a robot and a
    #: prop alike. Mapping the old keys here rather than asking worlds to be rewritten is what makes
    #: that a refactor instead of a breaking change.
    _NAV_KEYS = {
        "speed": "speed",
        "loop": "loop",
        "arrival_radius": "arrival_radius",
        "avoidance": "avoidance",
        "planner": "planner",
        "recovery": "recovery",
        "update_hz": "update_hz",
        "goal_endpoint": "goal_endpoint",
        "action_name": "action_name",
        "namespace": "namespace",
    }

    @classmethod
    def expand(cls, spec, world, base_dir):
        """Give this walker a ``navigator`` component, unless the world already wrote one.

        The same mechanism ``spawn_robot`` uses to attach a model manifest's controllers. A world
        that says nothing about navigation still gets exactly what it always got; a world that wants
        the navigator's newer options -- ``route_mode``, ``autostart``, a different tracker -- writes
        the component itself and this steps aside.
        """
        cfg = spec.config or {}
        if any(child.ref == "navigator" for child in spec.children):
            if any(k in cfg for k in ("speed", "waypoints", "loop")):
                raise PluginError(
                    f"walker {spec.name!r} configures navigation both in its own block and in a "
                    f"nested `navigator`. Put it in one place -- the navigator, for anything new."
                )
            # NOTHING, not this spec: `expand` contributes entries *beside* the one it was called
            # for, and the caller keeps that one. Returning it here builds the humanoid twice and
            # MuJoCo refuses the duplicate body names.
            return []

        nav = {dst: cfg[src] for src, dst in cls._NAV_KEYS.items() if src in cfg}
        nav["output"] = "walker"
        # `waypoints` become the navigator's `goals`, minus the first: a walker starts *at* its first
        # waypoint, and the navigator's route already begins wherever the body is.
        raw = cfg.get("waypoints") or []
        wps = [list(w)[:2] if not isinstance(w, dict) else list(w["pos"])[:2] for w in raw]
        if len(wps) > 1:
            nav["goals"] = wps[1:]
        elif wps:
            nav["goals"] = wps
        # ...and the per-waypoint dwell travels with them. It cannot ride along as a third element
        # of a goal, because there that position is the goal's YAW -- so it goes as the navigator's
        # own `dwell`, one entry per route point in the same order. Truncating the waypoints to
        # (x, y) and stopping there dropped every pause a world asked for, silently: the walker
        # still validated `[x, y, dwell]` and still documented it, and the crowd simply never
        # stopped walking.
        default = cfg.get("dwell", 0.0)
        dwells = [
            (w.get("dwell", default) if isinstance(w, dict) else (w[2] if len(w) > 2 else default))
            for w in raw
        ]
        if dwells and dwells != [0.0] * len(dwells):
            # The navigator's route is the mover's start followed by `goals`, and a walker starts at
            # its first waypoint -- so the two lists already line up entry for entry. The values are
            # passed through as written and coerced there, so the dwell format has one owner.
            nav["dwell"] = [list(d) if isinstance(d, (list, tuple)) else d for d in dwells]
        if isinstance(nav.get("avoidance"), bool):
            # A walker's own block has always spelled this as a yes/no. The navigator names a model
            # instead, because there is more than one and "yes" does not say which -- so the legacy
            # spelling is translated here rather than a world being asked to change.
            #
            # `stop: false` is the load-bearing half. A walker has never looked ahead: it gives way
            # through the local model or not at all, and `avoidance: true` has always meant "steers,
            # never stops". Letting it acquire a forward probe here would change how every existing
            # pedestrian world behaves, which is exactly what this compatibility path exists to
            # prevent -- and it is why the three capabilities are independent rather than a ladder.
            nav["avoidance"] = {
                "steer": DEFAULT_MODEL if nav["avoidance"] else "none",
                "stop": False,
            }
        if "goals" not in nav:
            # A goal-driven-only walker has no patrol, so there is nothing to cycle through.
            nav.pop("loop", None)
        nav.setdefault("speed", 1.0)
        # Same reason, for a walker that named no avoidance at all. It is a default worth knowing
        # about rather than one to rely on: a walker is a mocap body, so the solver treats it as
        # immovable and it will shove anything free it walks into, however politely that thing
        # stopped. A world that puts a walker in a room with a robot should write a `navigator` for
        # it and ask for `avoidance: {stop: true}`.
        nav.setdefault("avoidance", {"stop": False})
        # A walker's own footprint, so it presents the same disc to avoidance it always did.
        nav.setdefault("radius", float((cfg.get("orca") or {}).get("radius", 0.26)))
        if (cfg.get("orca") or {}).get("max_speed") is not None:
            nav["max_speed"] = float(cfg["orca"]["max_speed"])
        # Returned as an ADDITIONAL spec, not a rewritten owner: `expand` contributes entries
        # beside the one it was called for (the caller keeps that one), so returning the walker
        # again builds its skeleton twice and MuJoCo refuses the duplicate body names.
        return [
            PluginSpec(ref="navigator", name=None, config=nav, children=[], entity=spec.address)
        ]

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
        """Register the entity, build this walker's animation state, and declare its endpoints.

        Navigation is not here any more: a nested ``navigator`` owns it (see :meth:`expand`), and
        this plugin owns the body it moves -- the mocap skeleton, the resolved motion clips, the
        blendspace state they are sampled into. The ``walker`` output reads that state from the
        blackboard, which is the seam that lets one navigator serve a pedestrian, a robot and a prop.
        """
        self._ctx = ctx
        self._anim = make_anim_state(ctx.model, self._spec)
        states = ctx.blackboard.get(STATE_KEY) or {}
        states[self.walker_name] = self._anim
        ctx.blackboard.set(STATE_KEY, states)
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

        # The goal endpoint is NOT declared here any more: the nested `navigator` declares it, for
        # both nav2 action types, from one place. Two declarations of the same capability would mean
        # two handlers racing to register for one type in the bridge, where the loser is silently
        # overwritten. `goal_endpoint` and `action_name` still work in this block -- `expand` passes
        # them through -- so a world reads the same as it always did.

    def read_body_poses(self):
        """Endpoint ``read`` (physics thread): ``[(frame, pos[3], quat[4]), ...]`` for the 17 bones.

        World transforms straight from ``data.xpos``/``xquat`` (mocap bodies are world children).
        ``frame`` is the body name (== the exported scene body name), so the viewer binds each
        transform to its bone node by name. ``quat`` is MuJoCo (w, x, y, z)."""
        d = self._ctx.data
        return [(name, d.xpos[bid], d.xquat[bid]) for name, bid in self._body_ids if bid >= 0]

    def on_reset(self, ctx: SimContext) -> None:
        """Put the body back at its start, before the navigator's own reset reads it.

        ``mj_resetData`` parks every mocap body at the origin, so the skeleton has to be re-posed
        whatever else happens. Doing it here rather than in the navigator is what makes the ordering
        work: components reset after their owner, so the navigator reads a body that is already home
        rather than one still standing where the last episode left it.
        """
        st = self._anim
        st.pos = st.patrol_wps[0].copy()
        st.yaw = _heading(st.patrol_wps[0], st.patrol_wps[1]) if len(st.patrol_wps) > 1 else st.yaw
        st.phase = st.phase_run = st.phase_short = st.phase_turn = 0.0
        st.t_idle = 0.0
        st.disp_speed = 0.0
        st.pref_vel = np.zeros(2)
        # Posed straight from the idle clip at phase 0, not through the blendspace: running one
        # animation frame here would advance the idle clock and settle the body a couple of
        # millimetres off the pose every previous episode started from.
        joint_rot, root_z = st.idle.sample(0.0)
        poses = forward_kinematics(
            [st.pos[0], st.pos[1], st.skeleton.root_height + root_z],
            st.yaw,
            joint_rot,
            skeleton=st.skeleton,
        )
        write_pose(ctx.data, st, foot_ground(poses, st))

    def _next_seq(self) -> int:
        with self._seq_lock:
            return next(self._seq)

    # -- goal interface ------------------------------------------------------------------------
    # Delegated to the navigator's own handle, so a route sent to a walker and one sent to a robot
    # take the same path through the simulator. ``WalkerHandle`` survives as a thin alias: the ROS
    # action handler and any downstream code that looks up ``walker:<name>`` keeps working, and the
    # names, the action type and the sequence-number contract are all unchanged.
    def _nav(self):
        return self._ctx.blackboard.get(f"nav:{self.walker_name}:handle") if self._ctx else None

    def _write_route(self, poses) -> None:
        """Endpoint ``write``: already marshalled onto the physics thread by the bridge."""
        self.send_route(poses)

    def send_route(self, poses) -> int:
        handle = self._nav()
        return handle.send_goals(list(poses)) if handle is not None else self._next_seq()

    def cancel_route(self) -> int:
        handle = self._nav()
        return handle.cancel() if handle is not None else self._next_seq()

    def status(self) -> tuple[int, bool, int, float]:
        """``(route_seq, finished, goals_remaining, distance_remaining)``; safe to poll off-thread."""
        handle = self._nav()
        return handle.status() if handle is not None else (0, False, 0, 0.0)
