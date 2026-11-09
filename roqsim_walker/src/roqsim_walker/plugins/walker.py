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
      hands: false             # true -> wrists that turn, five fingers a hand, grip poses, and
                               # the moves of a nested `humanoid` (roqsim_motion); false is the
                               # seventeen-body walker, with nothing of the hands built or computed
      rgba: [r, g, b, a]       # colour of the capsule visuals (default: the humanoid's own)
      pose:                    # where the walker stands at the start and after every reset:
        position: {x: 0.0, y: 0.0}  #   a world pose as SpawnEntity states one, with no z -- a
        orientation: {yaw: 0.0}     #   walker stands on the floor -- and a heading only
      motion: {walk: /abs/walk.npz}  # a locomotion clip of its own: a clip's name or a .npz
      layers:                  # clips played over the gait from the start (roqsim_walker.layers):
        - {clip: wave, joints: [arm_r], loop: true, weight: 1.0, fade: 0.3, speed: 1.0}
      body_lag: 0.0            # s: the body follows its nav root as a damped spring of this time
                               # constant and its turns speed up and slow down over twice it, so
                               # it starts, turns and stops smoothly (0: on the root, turning at once)

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

import math
import numbers
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
from roqsim_walker.animation import HandReach, make_anim_state, write_pose
from roqsim_walker.animation import (
    _foot_ground as foot_ground,
)
from roqsim_walker.blueprint import BlueprintError, resolve_clip, resolve_walker
from roqsim_walker.humanoid import JOINT_NAMES, build_humanoid, forward_kinematics
from roqsim_walker.layers import Layers, check
from roqsim_walker.motion import Clip
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


def _point(hand: str, point) -> np.ndarray | None:
    """``point`` as a world point for the ``hand`` hand: ``None``, or 3 finite numbers."""
    if point is None:
        return None
    try:
        values = list(point)
    except TypeError:
        values = None
    if (
        isinstance(point, (str, bytes))
        or values is None
        or len(values) != 3
        or not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values)
        or not all(math.isfinite(v) for v in values)
    ):
        raise ValueError(
            f"reach: the {hand} hand's point is a world point of 3 finite numbers, or None to "
            f"let it go back to the walk -- not {point!r}"
        )
    return np.array(values, dtype=float)


@dataclass
class WalkerHandle:
    """The walker's body, published by :class:`WalkerPlugin` under ``walker:<name>``: its hands, the
    clips it plays over the gait and where it looks. Where it goes is its navigator's, through the
    navigator's handle (``nav:<name>:handle``) as for every mover."""

    name: str
    #: ``reach(left=None, right=None)`` puts each hand given a world point (3 floats) on it -- re-set
    #: it as the points move -- the spine bending as far as they need; a hand given ``None`` goes
    #: back to the walk, so ``reach()`` lets go with both. Each arm eases onto its point and back
    #: over the gait on its own (:mod:`roqsim_walker.reach`). A point that is not 3 finite numbers
    #: raises :class:`ValueError`. Call it on the physics thread.
    reach: Callable[..., None] | None = None
    #: the wrists' world positions now: ``(left, right)``
    hands: Callable[[], tuple] | None = None
    #: ``play(clip, joints=("all",), weight=1.0, loop=False, fade=0.3, speed=1.0)`` plays a clip over
    #: the gait on some joints and returns the layer's id; ``stop_clip(clip="")`` fades layers out;
    #: ``clip_ended(id)``, ``playing()``; ``look_at(point)`` turns the head to a world point or a
    #: callable giving one, ``None`` back (:class:`roqsim_walker.layers.Layers`). On the physics
    #: thread.
    play: Callable[..., int] | None = None
    stop_clip: Callable[..., list] | None = None
    clip_ended: Callable[[int], bool] | None = None
    playing: Callable[[], list] | None = None
    look_at: Callable[..., None] | None = None


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
        self._layouts = None  # each hand's layout, with `hands: true` (set in build)
        self.body = None  # the motion core's body, with `hands: true` (set in configure)

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
        if not isinstance(config.get("hands", False), bool):
            errors.append(f"walker: 'hands' is true or false, not {config['hands']!r}")
        errors += self._layer_errors(config)
        lag = config.get("body_lag", 0.0)
        if (
            not isinstance(lag, (int, float))
            or isinstance(lag, bool)
            or not math.isfinite(lag)
            or lag < 0
        ):
            errors.append(
                f"walker: 'body_lag' is a time constant in seconds, 0 or more, not {lag!r}"
            )
        return errors

    #: The keys of one ``layers:`` entry, and what ``play`` takes them as.
    LAYER_KEYS = ("clip", "joints", "weight", "loop", "fade", "speed")

    def _layer_errors(self, config: dict) -> list[str]:
        """What is wrong with ``layers:``: each a mapping with a ``clip`` the walker has and can
        read, joints it has, and numbers in range."""
        layers = config.get("layers", [])
        if not isinstance(layers, list):
            return [f"walker: 'layers' is a list of clips to play, not {layers!r}"]
        errors = []
        for i, entry in enumerate(layers):
            where = f"walker: layers[{i}]"
            if not isinstance(entry, dict) or "clip" not in entry:
                errors.append(f"{where} is a mapping with a 'clip', not {entry!r}")
                continue
            unknown = sorted(set(entry) - set(self.LAYER_KEYS))
            if unknown:
                errors.append(
                    f"{where}: {', '.join(map(repr, unknown))} not read; a layer takes "
                    f"{', '.join(self.LAYER_KEYS)}"
                )
                continue
            try:
                Clip.load(resolve_clip(config.get("walker", ""), entry["clip"]))
                check(**{k: v for k, v in entry.items() if k != "clip"})
            except (BlueprintError, ValueError) as exc:
                errors.append(f"{where}: {exc}")
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
        self._layouts = None
        fingers = blueprint["hand_weights"] if use_skin else None
        if cfg.get("hands", False):
            from roqsim_walker.handling import layouts_for
            from roqsim_walker.humanoid import to_skeleton

            self._layouts = layouts_for(
                to_skeleton(blueprint["skeleton"]),
                blueprint["hands"],
                mesh=blueprint["mesh"] if fingers else None,
                tpose=blueprint["tpose"],
                flip=blueprint["flip"],
                finger_weights=fingers,
            )
            kw["hands"] = self._layouts
            kw["finger_weights"] = fingers
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
            "body_lag": float(cfg.get("body_lag", 0.0)),
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
        self.motion = Layers(self._anim, self.blueprint, ctx)
        self.body = None
        if self._layouts is not None:
            from roqsim_walker.handling import WalkerBody

            self.body = WalkerBody(self.walker_name, self._anim, ctx.model, self._layouts)
            self.body._data = ctx.data
            self._anim.rig = self.body
            ctx.blackboard.set(f"humanoid_body:{self.walker_name}", self.body)
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
                    **({"hands": True} if self.body is not None else {}),
                },
            )
        )
        ctx.blackboard.set(
            f"walker:{self.walker_name}",
            WalkerHandle(
                name=self.walker_name,
                reach=self.reach,
                hands=self.hands,
                play=self.motion.play,
                stop_clip=self.motion.stop,
                clip_ended=self.motion.ended,
                playing=self.motion.playing,
                look_at=self.motion.look_at,
            ),
        )
        if self.body is not None:
            self.body.motion = self.motion

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
        st.body_pos = st.start[:2].copy()
        st.body_vel = np.zeros(2)
        st.yaw_rate = 0.0
        st.ground, st.ground_vel = None, 0.0
        st.reach = {side: HandReach() for side in st.reach}
        st.reach_bend = st.reach_bend_rate = 0.0
        st.reach_need = (frozenset(), 0.0)
        st.disp_speed = 0.0
        st.pref_vel = np.zeros(2)
        if self.body is not None:
            self.body.reset()
        self.motion.reset()
        for entry in self.config.get("layers", []):  # what it plays from the start
            self.motion.play(**{"loop": True, **entry})
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
        if self.body is not None:
            self.body.write_fingers(ctx.data)

    # -- the body's interface (where it goes is its navigator's: nav:<name>:handle) ---------------
    def reach(self, left=None, right=None) -> None:
        """The left and right hand each on a world point, or back to the walk with ``None``.

        Raises :class:`ValueError` for a point that is not 3 finite numbers, and
        :class:`RuntimeError` while a move of the walker's ``humanoid`` runs (the move places the
        hands), before either hand changes."""
        targets = {"l": _point("left", left), "r": _point("right", right)}
        moves = self._ctx.blackboard.get(f"machine:{self.walker_name}") if self._ctx else None
        if moves is not None and moves.busy:
            raise RuntimeError(
                f"walker {self.walker_name}: reach refused while its move "
                f"{moves.status().program!r} runs; cancel it, or wait for it to end"
            )
        for side, target in targets.items():
            if self.body is not None:
                # Hands on: the motion core places the wrist, as the two-bone reach does with them off.
                if target is None:
                    self.body.free_hand(side)
                else:
                    self.body.set_hand(side, target, frame="wrist")
            else:
                self._anim.reach[side].target = target

    def hands(self) -> tuple:
        d = self._ctx.data
        return tuple(d.mocap_pos[self._anim.mocap[f"wrist_{s}"]].copy() for s in ("l", "r"))

    # -- motion layers and the look, on the control socket and `entity_call` ---------------------
    # Off ROS: none of them has a message of its own, and a ROS client moves a walker through its
    # navigator's goal actions.
    @endpoint.command(ros2=None)
    def play(
        self,
        clip: str,
        joints: list[str] | None = None,
        weight: float = 1.0,
        loop: bool = False,
        fade: float = 0.3,
        speed: float = 1.0,
    ) -> int:
        """Play a clip over the gait on some joints; the layer's id.

        Args:
            clip: a clip of the walker's anim set by name, or a path to a .npz
            joints: joint groups (all, upper, arms, arm_l, arm_r, head, spine, pelvis, legs, leg_l,
                leg_r) and joint names; all by default
            weight: 0 to 1, how much of the pose under it the clip takes
            loop: play it round and round until stopped, rather than once
            fade: s it fades in and out over
            speed: times the clip's own pace
        """
        return self.motion.play(clip, joints or ["all"], weight, loop, fade, speed)

    @endpoint.command(ros2=None)
    def stop_clip(self, clip: str = "") -> list[int]:
        """Fade out the layers playing a clip, or every layer; the ids stopped.

        Args:
            clip: the clip's name as it was played, or empty for every layer
        """
        return self.motion.stop(clip)

    @endpoint.command(ros2=None)
    def look_at(self, target: str = "") -> None:
        """Turn the head toward a target, following it as it moves; empty turns it back.

        Args:
            target: a site, body or entity of the world by name, or empty
        """
        self.motion.look_at(target)

    @endpoint.out(rate=10.0, ros2=None)
    def clips(self) -> list[dict]:
        """The clips playing over the gait: each layer's id, clip, joints, loop, time, weight and
        whether it is stopping."""
        return self.motion.playing()
