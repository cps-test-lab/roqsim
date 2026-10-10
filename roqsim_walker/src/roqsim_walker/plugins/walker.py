"""Scene plugin: a kinematic pedestrian, moved by the navigator nested under it.

The pedestrian stack in the roqsim plugin model: this plugin builds one walker's mocap bodies + skin
into the ``MjSpec``, registers it as an ``Entity(kind='pedestrian')``, and publishes the animation
state the ``walker`` output of ``roqsim_nav`` moves. Where it goes is the navigator's, as for every
mover roqsim navigates, and so is its goal endpoint (served over ROS 2 by ``roqsim_nav_ros``).

Config::

    walker:
      walker: MaleVisitorWalk  # blueprint folder under models/people/ (required)
      namespace: ""            # transport scope of the walker's endpoints and its navigator's
      outfit: B                # clothing variant: a letter, or {pants: C, jacket: A}
      skin: true               # false -> capsule visuals instead of the character mesh
      rgba: [r, g, b, a]       # colour of the capsule visuals (default: the humanoid's own)
      pose:                    # where the walker stands at the start and after every reset:
        position: {x: 0.0, y: 0.0}  #   a world pose as SpawnEntity states one, with no z -- a
        orientation: {yaw: 0.0}     #   walker stands on the floor -- and a heading only
      motion: {walk: /abs/walk.npz}  # override a resolved locomotion clip

A walker with no nested ``navigator`` gets one that is goal-driven only: ``output: walker``,
``speed: 1.0``, its 0.26 m disc as ``radius``, and ``avoidance: {stop: false}``, since a walker does
not look ahead. A patrol, a pause, avoidance or a goal endpoint's name is stated on a navigator
nested under it::

    - walker: {walker: MaleVisitorWalk, pose: {position: {x: -2.0, y: -2.0}}}
      name: pedestrian
      components:
        - navigator:
            output: walker
            speed: 1.2
            goals: [[2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
            loop: true
            dwell: [[0, 0], [2, 4], [0, 0], [1, 3]]   # per route point, the start first
            avoidance: {steer: give_way, stop: false}
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.config import PluginSpec
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import PoseError, parse_pose, pose_spelling, rpy_to_quat, yaw_of
from roqsim.types import Transform, Transforms
from roqsim_walker.animation import (
    _foot_ground as foot_ground,
)
from roqsim_walker.animation import make_anim_state, write_pose
from roqsim_walker.blueprint import BlueprintError, resolve_walker
from roqsim_walker.humanoid import JOINT_NAMES, build_humanoid, forward_kinematics
from roqsim_walker.output import STATE_KEY


def start_of(config: dict) -> tuple[float, float, float] | None:
    """``(x, y, yaw)`` a walker's ``pose`` places it at, or ``None`` when it states none.

    Raises :class:`~roqsim.pose.PoseError` for a pose a walker cannot take: one with a ``z`` (it
    stands on the floor, where its soles put it) or one that is not a heading.
    """
    if "pose" not in config:
        return None
    pos, quat = parse_pose(config["pose"])
    if pos[2] is not None:
        raise PoseError(
            "'pose.position.z' is not read -- a walker stands on the floor, at the height its "
            "soles put it; state x and y"
        )
    yaw = yaw_of(quat)
    flat = rpy_to_quat(0.0, 0.0, yaw)
    if abs(sum(a * b for a, b in zip(quat, flat, strict=True))) < 1.0 - 1e-9:
        raise PoseError(
            "'pose.orientation' tilts the walker -- a walker stands upright, so its orientation "
            "is a heading: orientation: {yaw: ...}"
        )
    return pos[0], pos[1], yaw


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
    #: The navigator a walker gets when the world nests none: goal-driven only, its own disc, and
    #: an avoidance that never stops it, since a walker does not look ahead.
    DEFAULT_NAVIGATOR = {
        "output": "walker",
        "speed": 1.0,
        "radius": 0.26,
        "avoidance": {"stop": False},
    }

    #: Keys a walker's own block refuses: each is the navigator's, and where it lives there.
    NAVIGATION_KEYS = {
        "speed": "'speed'",
        "waypoints": "'goals', with the walker starting at its own 'pose'",
        "loop": "'loop'",
        "dwell": "'dwell', one entry per route point with the start first",
        "arrival_radius": "'arrival_radius'",
        "avoidance": "'avoidance': {steer: give_way} for true, {steer: none} for false",
        "goal_endpoint": "'goal_endpoint'",
        "action_name": "'action_names': {navigate_through_poses: ...}",
        "orca": "'radius' and 'max_speed'",
        "planner": "'planner'",
        "recovery": "'recovery'",
        "update_hz": "'update_hz'",
    }

    @classmethod
    def expand(cls, spec, world, base_dir):
        """Give this walker the default ``navigator`` component, unless the world nests one.

        The same mechanism ``spawn_robot`` uses to attach a model manifest's controllers.
        """
        if any(child.ref == "navigator" for child in spec.children):
            # NOTHING, not this spec: `expand` contributes entries *beside* the one it was called
            # for, and the caller keeps that one. Returning it here builds the humanoid twice and
            # MuJoCo refuses the duplicate body names.
            return []
        config = {**cls.DEFAULT_NAVIGATOR, "avoidance": dict(cls.DEFAULT_NAVIGATOR["avoidance"])}
        return [
            PluginSpec(ref="navigator", name=None, config=config, children=[], entity=spec.address)
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
        for key, there in self.NAVIGATION_KEYS.items():
            if key in config:
                errors.append(
                    f"walker: {key!r} is not read -- where a walker goes is its navigator's: state "
                    f"it as the nested navigator's {there} (components: [{{navigator: {{output: "
                    f"walker, ...}}}}])"
                )
        if "pos" in config:
            errors.append(
                "walker: 'pos' is not read -- a walker's start is stated as 'pose', a world pose "
                "as SpawnEntity states one (x and y, and a heading): "
                f"{pose_spelling(config['pos'])}"
            )
        try:
            start_of(config)
        except PoseError as exc:
            errors.append(f"walker: {exc}")
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

        # What the animation state is built from: where the walker starts + what the blueprint
        # resolved.
        self._spec = {
            **({"start": start} if (start := start_of(cfg)) is not None else {}),
            "name": self.walker_name,
            "skeleton": blueprint["skeleton"],
            "sole": blueprint["sole"],
            "motion": blueprint["motion"],
        }

    def configure(self, ctx: SimContext) -> None:
        """Register the entity, build this walker's animation state, and declare its endpoints.

        Navigation is not here: a nested ``navigator`` owns it (see :meth:`expand`), and
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
                meta={"walker": self.blueprint, "namespace": ns},
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

        self._body_ids = [
            (name, mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in (f"{self.walker_name}/{j}" for j in JOINT_NAMES)
        ]

        # The goal endpoint is NOT declared here: the nested `navigator` declares it, for
        # both nav2 action types, from one place. Two declarations of the same capability would mean
        # two handlers racing to register for one type in the bridge, where the loser is silently
        # overwritten.

    @property
    def endpoint_owner(self) -> str:
        """The pedestrian entity this plugin registers."""
        return self.walker_name

    # The walker is mocap-driven (no MuJoCo joints, so nothing to put on /joint_states); its 17
    # skeleton bodies go out as transforms on the shared, never-namespaced /tf, one message per tick,
    # so a viewer animates the skinned mesh. Mocap bodies are world children, so each bone is a flat
    # child of the map frame.
    @endpoint.out(rate=30.0, ros2={"topic": "/tf"})
    def body_poses(self) -> Transforms:
        """The 17 bones' world poses, each child frame named after its body."""
        d = self._ctx.data
        return Transforms(
            [
                Transform("", name, d.xpos[bid], d.xquat[bid])
                for name, bid in self._body_ids
                if bid >= 0
            ]
        )

    def on_reset(self, ctx: SimContext) -> None:
        """Put the body back at its start, before the navigator's own reset reads it.

        ``mj_resetData`` parks every mocap body at the origin, so the skeleton has to be re-posed
        whatever else happens. Doing it here rather than in the navigator is what makes the ordering
        work: components reset after their owner, so the navigator reads a body that is already home
        rather than one still standing where the last episode left it.
        """
        st = self._anim
        st.pos = st.start[:2].copy()
        st.yaw = float(st.start[2])
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
