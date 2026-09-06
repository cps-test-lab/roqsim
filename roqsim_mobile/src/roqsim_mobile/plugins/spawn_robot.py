"""Scene plugin: attach a robot MJCF into the world and own its base spawn pose.

Ported from our earlier in-house nav prototype's ``Robot`` scene assembly, but split into the plugin model: this plugin
only *places* the robot (build + initial pose). Its kinematics/odometry live in a controller plugin
(e.g. :mod:`roqsim.plugins.diff_drive`), which finds this robot via the entity registry.

Config::

    spawn_robot:
      model: turtlebot4     # bundled model name, filename, or absolute path
      namespace: ""         # optional transport scope; the robot's endpoints inherit it
      prefix: ""            # MJCF name prefix (use distinct prefixes for >1 robot)
      pos: [0.0, 0.0]       # world XY spawn ([x, y, z] to override the model's rest height)
      yaw: 0.0              # spawn heading (rad)
      base_joint: base_free # free joint used to place the base
      present: true         # false: compiled in, but absent until it is spawned

``name:`` is the entry's reserved SIBLING, not one of the keys above: it labels the entry and names
the entity this spawn registers (default: the plugin ref). Components nested under the entry attach
to that entity by position, and are addressed ``<name>.<label>``.

The plugin registers an ``Entity(kind='robot')`` whose ``meta`` carries ``prefix``, ``model``, and
``initial_pose`` so controller/sensor/bridge plugins can resolve the right (prefixed) names.

``present: false`` compiles the robot in and starts it **absent** -- nothing sees or touches it, and
the control plane does not list it -- until ``SpawnEntity`` brings it in at the pose that call
states (see :mod:`roqsim.presence`). The declared value is restored on ``on_reset``, so what one
trial spawned does not carry into the next.

Absence hides the robot's BODY, not its software: its controller and sensor plugins keep running,
so an absent robot still publishes and still responds to a twist -- and being out of the contact
set, a twist drives it through walls. For a start pose decided per run, move the robot with
``SetEntityState`` instead; absence is for a machine that is not meant to be in the trial yet.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Entity, SimContext
from roqsim.manifest import expand_manifest
from roqsim.models import ModelError, apply_assets, resolve_model
from roqsim.plugin import Plugin
from roqsim.presence import set_present


def _keyframe_base_z(spec: mujoco.MjSpec, base_joint: str) -> float | None:
    """The base's resting height, as the model's own keyframe states it -- or ``None``.

    A wheeled model is authored with ``base_link`` at the origin and its wheels hanging below it, so
    the compiled rest height of the base free joint is 0 and spawning at it buries the robot by a
    wheel radius. What the model actually knows about its own stance is in its ``<key>``: every
    model in this package states one, from a couple of millimetres of tyre squash to the Warthog's
    0.288 m. That height is read here because :func:`_strip_keyframes` is about to discard the rest
    of it, and a robot dropped into the floor is not merely ugly -- it is ejected on the first step,
    so the trial begins with a pose and a velocity nobody asked for.

    The address is resolved by compiling a copy rather than assuming the free joint comes first: a
    keyframe's qpos is the whole robot's, and reading the wrong three numbers out of it would put
    a wheel angle in the z of the spawn.
    """
    if not spec.keys:
        return None
    key = next((k for k in spec.keys if k.name == "home"), spec.keys[0])
    model = spec.copy().compile()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_joint)
    if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        return None
    adr = int(model.jnt_qposadr[jid])
    qpos = list(key.qpos)
    return float(qpos[adr + 2]) if len(qpos) > adr + 2 else None


def _strip_keyframes(spec: mujoco.MjSpec) -> None:
    """Drop a model's keyframes before it is attached into the world.

    A spawned robot's pose is owned by spawn_robot (the base free-joint qpos) and its controller
    (which sets the joint stance in ``on_reset``), never by a baked ``<keyframe>``. MuJoCo cannot
    merge a model-level keyframe on attach and warns ("nkey: parent has 0, child has 1, keeping
    parent value"); the keyframe is also meaningless once the robot is one body among many in a
    composed world (its qpos no longer matches the model layout). Clearing both the key elements and
    the ``<size nkey>`` reservation keeps the attach clean.
    """
    for key in list(spec.keys):
        spec.delete(key)
    spec.nkey = 0


class SpawnRobotPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    @classmethod
    def expand(cls, spec, world, base_dir):
        """Inject the robot model's default controller/sensor plugins (its ``<model>.manifest.yaml``).

        Keeps robot-intrinsic plugins (diff_drive, lidar) out of the world YAML: they ship with the
        model and become components of this entry, so they attach by position rather than by a
        wiring key. Off with ``default_plugins: false``; a component nested here with the same
        LABEL merges into the injected default rather than running beside it.
        """
        return expand_manifest(spec, world, base_dir=base_dir)

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot_name = self.address
        self.prefix = self.config.get("prefix", "")
        pos = self.config.get("pos", [0.0, 0.0])
        self.initial_pose = (float(pos[0]), float(pos[1]), float(self.config.get("yaw", 0.0)))
        #: An explicit z in ``pos``, which a world states when the ground under the spawn is not at
        #: z=0 -- a height field, a ramp, a shelf. Without one the model's own rest height is used.
        self.spawn_z = float(pos[2]) if len(pos) > 2 else None
        #: Read from the model's keyframe in :meth:`build`; None for a model that states no stance.
        self.rest_z: float | None = None
        self.base_joint = self.prefix + self.config.get("base_joint", "base_free")
        self.present = bool(self.config.get("present", True))

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if not config.get("model"):
            errors.append("'model' is required")
        else:
            try:
                resolve_model(config["model"], base_dir=self.base_dir)
            except ModelError as exc:
                errors.append(str(exc))
        pos = config.get("pos", [0.0, 0.0])
        if len(pos) not in (2, 3):
            errors.append("'pos' must be [x, y], or [x, y, z] to override the model's rest height")
        if "present" in config and not isinstance(config["present"], bool):
            errors.append("'present' must be true or false")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        asset = resolve_model(self.config["model"], base_dir=self.base_dir)
        child = mujoco.MjSpec.from_file(str(asset.path))
        # Resolve mesh/texture refs to absolute paths across the model's asset dirs (own package plus
        # any borrowed via the manifest's `assets:`), so compilation does not depend on CWD.
        apply_assets(child, asset)
        self.rest_z = _keyframe_base_z(child, self.config.get("base_joint", "base_free"))
        _strip_keyframes(child)
        frame = spec.worldbody.add_frame()
        spec.attach(child, prefix=self.prefix, frame=frame)

    def _resolve_base_body(self, ctx: SimContext) -> str:
        """The robot's root body: ``<prefix>base_link`` if the model has one, else the body owning
        the base joint.

        ``base_link`` is a ROS convention, not a guarantee. The base joint holds whatever a model
        calls its root, so it is the rule that works for every family; where ``base_link`` exists,
        both name the same body. If neither resolves the entity would name a body absent from the
        compiled model, so this raises here rather than at the first use of that name.
        """
        named = self.prefix + "base_link"
        if mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, named) >= 0:
            return named
        jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, self.base_joint)
        if jid >= 0:
            resolved = mujoco.mj_id2name(
                ctx.model, mujoco.mjtObj.mjOBJ_BODY, int(ctx.model.jnt_bodyid[jid])
            )
            if resolved:
                return resolved
        raise RuntimeError(
            f"spawn_robot ({self.robot_name}): cannot resolve a root body -- neither {named!r} nor "
            f"a body owning the base joint {self.base_joint!r} exists in the compiled model"
        )

    def configure(self, ctx: SimContext) -> None:
        base_body = self._resolve_base_body(ctx)
        ctx.entities.add(
            Entity(
                name=self.robot_name,
                kind="robot",
                body=base_body,
                meta={
                    "prefix": self.prefix,
                    "model": self.config["model"],
                    "initial_pose": self.initial_pose,
                    "base_joint": self.base_joint,
                    # Inherited by the robot's endpoint-producing plugins (diff_drive, lidar), so
                    # the manifest-injected defaults need no namespace plumbing of their own.
                    "namespace": self.config.get("namespace", ""),
                },
            )
        )
        self._apply_initial_pose(ctx)
        self._apply_declared_presence(ctx)

    def on_reset(self, ctx: SimContext) -> None:
        self._apply_initial_pose(ctx)
        self._apply_declared_presence(ctx)
        mujoco.mj_forward(ctx.model, ctx.data)

    def _apply_declared_presence(self, ctx: SimContext) -> None:
        """Put the robot back to the presence the world declared.

        Run at configure AND at every reset, because presence lives in ``model`` while
        ``mj_resetData`` restores ``data``: a robot a trial spawned is still present when the next
        one begins.
        """
        set_present(ctx, ctx.entities.get(self.robot_name), self.present)

    def _apply_initial_pose(self, ctx: SimContext) -> None:
        x, y, yaw = self.initial_pose
        jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, self.base_joint)
        if jid < 0:
            ctx.logger.warning(
                "spawn_robot %s: base joint %r not found; pose not applied",
                self.robot_name,
                self.base_joint,
            )
            return
        q = ctx.model.jnt_qposadr[jid]
        ctx.data.qpos[q] = x
        ctx.data.qpos[q + 1] = y
        # z: the world's if it states one, else the model's own stance. Only a model that states
        # neither keeps the compiled qpos0, which for a base_link authored at the origin is 0.
        z = self.spawn_z if self.spawn_z is not None else self.rest_z
        if z is not None:
            ctx.data.qpos[q + 2] = z
        h = yaw / 2.0
        ctx.data.qpos[q + 3 : q + 7] = (np.cos(h), 0.0, 0.0, np.sin(h))
