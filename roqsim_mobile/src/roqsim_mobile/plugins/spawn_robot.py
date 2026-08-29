"""Scene plugin: attach a robot MJCF into the world and own its base spawn pose.

Ported from our earlier in-house nav prototype's ``Robot`` scene assembly, but split into the plugin model: this plugin
only *places* the robot (build + initial pose). Its kinematics/odometry live in a controller plugin
(e.g. :mod:`roqsim.plugins.diff_drive`), which finds this robot via the entity registry.

Config::

    spawn_robot:
      model: turtlebot4     # bundled model name, filename, or absolute path
      namespace: ""         # optional transport scope; the robot's endpoints inherit it
      prefix: ""            # MJCF name prefix (use distinct prefixes for >1 robot)
      pos: [0.0, 0.0]       # world XY spawn
      yaw: 0.0              # spawn heading (rad)
      base_joint: base_free # free joint used to place the base

``name:`` is the entry's reserved SIBLING, not one of the keys above: it labels the entry and names
the entity this spawn registers (default: the plugin ref). Components nested under the entry attach
to that entity by position, and are addressed ``<name>.<label>``.

The plugin registers an ``Entity(kind='robot')`` whose ``meta`` carries ``prefix``, ``model``, and
``initial_pose`` so controller/sensor/bridge plugins can resolve the right (prefixed) names.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Entity, SimContext
from roqsim.manifest import expand_manifest
from roqsim.models import ModelError, apply_assets, resolve_model
from roqsim.plugin import Plugin


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
        self.base_joint = self.prefix + self.config.get("base_joint", "base_free")

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
        if len(pos) != 2:
            errors.append("'pos' must be [x, y]")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        asset = resolve_model(self.config["model"], base_dir=self.base_dir)
        child = mujoco.MjSpec.from_file(str(asset.path))
        # Resolve mesh/texture refs to absolute paths across the model's asset dirs (own package plus
        # any borrowed via the manifest's `assets:`), so compilation does not depend on CWD.
        apply_assets(child, asset)
        _strip_keyframes(child)
        frame = spec.worldbody.add_frame()
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        base_body = self.prefix + "base_link"
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

    def on_reset(self, ctx: SimContext) -> None:
        self._apply_initial_pose(ctx)
        mujoco.mj_forward(ctx.model, ctx.data)

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
        # qpos[q+2] keeps the compiled rest height.
        h = yaw / 2.0
        ctx.data.qpos[q + 3 : q + 7] = (np.cos(h), 0.0, 0.0, np.sin(h))
