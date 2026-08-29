"""Scene plugin: place a static model (a prop) into the world from the world YAML, at a fixed pose.

The generic, family-free counterpart to ``spawn_robot``/``spawn_arm``/``spawn_sensor``: it resolves
any ``roqsim.models`` entry -- notably the reusable props in ``roqsim_assets`` (an office table,
a chair, a fire extinguisher) -- and attaches it at a mount pose. By default there is no free joint, so
the model is welded in place (static scenery); it is the world-YAML alternative to ``<include>``-ing the
prop's MJCF into a baked scene. The prop's own MJCF is used unchanged. Set ``free: true`` for a prop
physics should move (a box to be picked up).

Config::

    spawn_model:
      model: industrial_table        # bundled model name, filename, or absolute path
      prefix: ""                     # MJCF name prefix (use distinct prefixes for >1 of the model)
      pos: [0.0, 0.0, 0.0]           # [x, y] or [x, y, z]
      rpy: [0.0, 0.0, 0.0]           # orientation as roll/pitch/yaw (rad)
      scale: 1.0                     # uniform geometric scale factor (see below)
      free: false                    # give the prop a free joint: a movable body, not scenery
      mass: 0.5                      # override the root body's total geom mass (kg)
      friction: [1.2, 0.005, 0.0001] # override the root body's geom friction (or a single sliding val)
      publish_tf: false              # publish the root body's world pose as TF (see below)
      tf_rate: 30.0                  # publish_tf: dynamic -- stream rate (Hz)

``name:`` is the entry's reserved SIBLING, not one of the keys above: it labels the entry and names
the entity this spawn registers (default: the plugin ref). Written *inside* the config block it is
silently inert, and anything addressing the prop by the name you chose then resolves to nothing.

``mass`` and ``friction`` exist so the two properties that decide whether a grasp holds are world-YAML
keys, and therefore ordinary campaign factors -- a sweep over payload or surface friction needs no new
variation plugin, just ``ParameterVariationList`` against these. ``mass`` rescales the root body's geom
masses in proportion, keeping the mass distribution of a multi-geom prop; ``friction`` accepts a single
sliding coefficient or the full ``[sliding, torsional, rolling]`` triple. Both are refused when the prop
has nothing to scale, rather than silently doing nothing.

``free: true`` adds a ``<freejoint/>`` to the prop's root body, turning it from welded scenery into a
body physics moves -- a box a robot can pick up. It also registers the joint as the entity's
``base_joint``, which is what lets ``simulation_interfaces``' ``SetEntityState`` teleport or re-seat it
(the service rejects any entity without one), and what ``on_reset`` uses to put it back at its spawn
pose between episodes instead of leaving it wherever the last run dropped it.

The prop must end up with mass. MuJoCo derives it from geom volume x density (default 1000), so an
ordinary prop is fine as-is; a geom carrying ``density="0"`` -- the convention for visual-only
decoration, used throughout the robot models here -- is not, and MuJoCo rejects a massless moving body
at compile time rather than simulating it.

Pair it with ``publish_tf: dynamic``: nothing else publishes a free body's pose.

``scale`` resizes the prop at spawn time, so one asset serves every size a scene needs instead of the
library carrying a folder per size (a 2.9 m wall screen and a 1.2 m one are the same model). It is
applied to the loaded child spec before attach, and covers the whole geometry -- mesh scales, and the
positions/sizes of bodies, geoms, sites, lights and joints -- so a prop built from primitives or from
several offset parts scales as one piece rather than coming apart. Two spawns of one model at
different scales stay distinct assets: the dedup key in :mod:`roqsim.assets` includes ``mesh.scale``.

It is deliberately a single **uniform** factor. Non-uniform scaling is ill-defined for a sphere or a
capsule and silently shears any child body that is rotated relative to its parent, so a prop that must
be stretched on one axis needs a purpose-built plugin instead (as ``door`` does for its leaf).
``scale`` is geometry only -- it does not touch mass or inertia, which is why it suits the static
scenery this plugin places (there is no free joint) and not a dynamic body.

Unlike the ``spawn_*`` plugins for robots/sensors this does **not** pull in a model manifest -- a prop
is inert geometry with no intrinsic controller or sensors. Place several by listing the plugin
multiple times with distinct ``prefix`` (and ``name``).

``publish_tf`` puts the spawned root body's world pose on TF so a viewer binds the scene node by name
(``child_frame_id`` == the exported body name) and a TF-tree consumer (rviz, an rso_web_backend federation)
gets the frame. It has no effect on the baked web scene, which already seats the body at its spawn pose.

  - ``false`` (default): a static prop already seated by the baked scene needs no TF.
  - ``dynamic`` (or ``true``): stream the live world pose on the relative ``tf`` topic at ``tf_rate``.
    For a **free body** (a ``<freejoint/>`` prop the robot moves) -- nothing else publishes its pose, so
    a name-binding viewer would otherwise freeze it at the spawn pose. The ros2_bridge's ``gt`` config
    maps the relative topic to ``/gt/tf``; a multi-robot gateway federates it under the robot's scope.
  - ``static``: publish the world pose **once** on the latched ``/tf_static`` (a welded prop's frame for
    the TF tree). The pose is model-fixed, so one ``mj_forward`` at configure resolves it.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Endpoint, Entity, SimContext
from roqsim.models import ModelError, apply_assets, resolve_model
from roqsim.plugin import Plugin

_TF_MODES = (False, "dynamic", "static")


def _scale_spec(spec: mujoco.MjSpec, factor: float) -> None:
    """Scale every geometric quantity in ``spec`` by ``factor`` (uniform), in place.

    Mesh scales alone are not enough: a prop may be built from primitives (the glass door's chrome
    rail is a ``<geom type="box">``) or from parts offset inside their body, and scaling only the
    meshes would shrink those parts while leaving their offsets and sizes untouched -- the prop would
    come apart rather than get smaller. Inertial properties are left alone; see the module docstring.
    """
    for mesh in spec.meshes:
        mesh.scale = [c * factor for c in mesh.scale]
    for body in spec.bodies:
        body.pos = [c * factor for c in body.pos]
        body.ipos = [c * factor for c in body.ipos]
    for geom in spec.geoms:
        geom.pos = [c * factor for c in geom.pos]
        geom.size = [c * factor for c in geom.size]  # unused (zero) for mesh geoms
    for site in spec.sites:
        site.pos = [c * factor for c in site.pos]
        site.size = [c * factor for c in site.size]
    for light in spec.lights:
        light.pos = [c * factor for c in light.pos]
    for joint in spec.joints:
        joint.pos = [c * factor for c in joint.pos]


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """(w, x, y, z) quaternion from roll/pitch/yaw (rad), fixed-axis XYZ (ROS/URDF convention)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


class SpawnModelPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.model_ref = self.config.get("model", "")
        self.prefix = self.config.get("prefix", "")
        self.entity_name = self.address
        pos = self.config.get("pos", [0.0, 0.0, 0.0])
        self.pos = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        rpy = self.config.get("rpy", [0.0, 0.0, 0.0])
        self.quat = _rpy_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        self.scale = float(self.config.get("scale", 1.0))
        self.free = bool(self.config.get("free", False))
        self.mass = self.config.get("mass")
        friction = self.config.get("friction")
        if friction is not None and not isinstance(friction, (list, tuple)):
            friction = [friction]  # a bare number means the sliding coefficient
        self.friction = [float(v) for v in friction] if friction is not None else None
        self._base_joint = ""
        self._spawn_qpos: list[float] = []
        # publish_tf: false | "dynamic" | "static"; `true` is an alias for "dynamic".
        mode = self.config.get("publish_tf", False)
        self.publish_tf = "dynamic" if mode is True else mode
        self.tf_rate = float(self.config.get("tf_rate", 30.0))
        self._root_body = ""
        self._body_frame = ""
        self._body_id = -1
        self._ctx: SimContext | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if not config.get("model"):
            errors.append("'model' is required")
        else:
            try:
                resolve_model(config["model"])
            except ModelError as exc:
                errors.append(str(exc))
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")
        scale = config.get("scale", 1.0)
        if isinstance(scale, (list, tuple)):
            errors.append(
                "'scale' must be a single uniform factor, not a per-axis list -- non-uniform "
                "scaling shears rotated child bodies and is undefined for spheres/capsules"
            )
        else:
            try:
                if float(scale) <= 0.0:
                    errors.append("'scale' must be > 0")
            except (TypeError, ValueError):
                errors.append("'scale' must be a number")
        mode = config.get("publish_tf", False)
        if ("dynamic" if mode is True else mode) not in _TF_MODES:
            errors.append("'publish_tf' must be false, 'dynamic' (or true), or 'static'")
        if (mass := config.get("mass")) is not None:
            try:
                if float(mass) <= 0.0:
                    errors.append("'mass' must be > 0")
            except (TypeError, ValueError):
                errors.append("'mass' must be a number (kg)")
        friction = config.get("friction")
        if friction is not None:
            values = friction if isinstance(friction, (list, tuple)) else [friction]
            if not 1 <= len(values) <= 3:
                errors.append("'friction' must be a number or [sliding, torsional, rolling]")
            elif any(float(v) < 0.0 for v in values):
                errors.append("'friction' components must be >= 0")
        if config.get("free") and ("dynamic" if mode is True else mode) == "static":
            # A latched one-shot pose for a body that moves is a frame frozen at the spawn pose.
            errors.append(
                "'publish_tf: static' contradicts 'free: true' -- a movable body's pose is not "
                "model-fixed; use publish_tf: dynamic"
            )
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        asset = resolve_model(self.model_ref)
        try:
            child = mujoco.MjSpec.from_file(str(asset.path))
        except ValueError as exc:
            # MuJoCo raises a bare "could not decode content" that names neither the file nor the
            # cause. Point at the resolved path and the likely reason (not an MJCF, or malformed XML).
            raise ModelError(
                f"spawn_model {self.model_ref!r}: MuJoCo could not parse {asset.path} as MJCF "
                f"({exc}). Check that this file is a MuJoCo XML model (a valid <mujoco> document), "
                f"not a world YAML, a directory, or another format."
            ) from exc
        # Resolve mesh/texture refs to absolute paths across the model's asset dirs (the prop's own
        # folder is included as a fallback), so compilation does not depend on CWD.
        apply_assets(child, asset)
        if self.scale != 1.0:
            _scale_spec(child, self.scale)
        # Best-effort root body name (for the entity's pose lookups); props are named after their file.
        bodies = getattr(child.worldbody, "bodies", [])
        self._root_body = bodies[0].name if bodies else asset.path.stem
        if self.mass is not None or self.friction is not None:
            self._apply_physics_overrides(bodies, asset)
        if self.free:
            self._add_freejoint(child, bodies, asset)
        if not self.entity_name:
            self.entity_name = asset.path.stem
        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def _apply_physics_overrides(self, bodies, asset) -> None:
        """Rescale root-body geom mass and/or set geom friction, so both are campaign factors."""
        if not bodies:
            raise ModelError(
                f"spawn_model {self.model_ref!r}: mass/friction override needs a root body, but "
                f"{asset.path} declares none."
            )
        geoms = list(bodies[0].geoms)
        if self.mass is not None:
            total = sum(float(getattr(g, "mass", 0.0) or 0.0) for g in geoms)
            if total <= 0.0:
                raise ModelError(
                    f"spawn_model {self.model_ref!r}: mass override needs the prop's geoms to declare "
                    f"mass to rescale (a density-only or massless prop has no distribution to keep). "
                    f"Set mass on the geoms in {asset.path}, or drop the override."
                )
            factor = float(self.mass) / total
            for g in geoms:
                g.mass = float(g.mass) * factor
        if self.friction is not None:
            for g in geoms:
                # MuJoCo's geom friction is [sliding, torsional, rolling]; keep the prop's own value
                # for any component the world did not name.
                g.friction = [*self.friction, *list(g.friction)[len(self.friction) :]]

    def _add_freejoint(self, child: mujoco.MjSpec, bodies, asset) -> None:
        """Make the prop's root body a free body, refusing the cases that go silently wrong."""
        if not bodies:
            raise ModelError(
                f"spawn_model {self.model_ref!r}: free: true needs a root body to attach the free "
                f"joint to, but {asset.path} declares none (its geoms sit directly on worldbody)."
            )
        root = bodies[0]
        if any(getattr(j, "type", None) is not None for j in getattr(root, "joints", [])):
            raise ModelError(
                f"spawn_model {self.model_ref!r}: free: true, but {asset.path} already gives its root "
                f"body a joint. Spawn it without `free` -- the prop defines its own articulation."
            )
        root.add_freejoint(name="free")
        self._base_joint = f"{self.prefix}free"

    def configure(self, ctx: SimContext) -> None:
        self._body_frame = self.prefix + self._root_body
        meta = {"prefix": self.prefix, "model": self.model_ref}
        if self.free:
            # simulation_interfaces' SetEntityState only accepts an entity whose base_joint is a free
            # joint, so without this a movable prop could not be teleported or reset between episodes.
            meta["base_joint"] = self._base_joint
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="object" if self.free else "prop",
                body=self._body_frame,
                meta=meta,
            )
        )
        if self.free:
            jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, self._base_joint)
            if jid < 0:
                raise RuntimeError(f"spawn_model: free joint {self._base_joint!r} not found")
            adr = int(ctx.model.jnt_qposadr[jid])
            self._spawn_qpos = [adr, *(float(v) for v in ctx.model.qpos0[adr : adr + 7])]
        if not self.publish_tf:
            return

        self._ctx = ctx
        self._body_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, self._body_frame)
        if self._body_id < 0:
            raise RuntimeError(f"spawn_model: body {self._body_frame!r} not found for publish_tf")

        if self.publish_tf == "dynamic":
            # Stream the live world pose as a one-entry TF payload. child_frame_id == the exported body
            # name so a name-binding viewer animates the node; the relative `tf` topic lets the bridge's
            # gt namespace map it to /gt/tf and a gateway federate it. Nothing else publishes a free
            # body's pose, so without this a viewer freezes it at the baked spawn pose.
            ctx.interface.add(
                Endpoint(
                    name=f"{self.entity_name}_pose",
                    direction="out",
                    owner=self.entity_name,
                    read=self.read_pose,
                    rate_hz=self.tf_rate,
                    backend={
                        "ros2": {
                            "type": "tf2_msgs.msg.TFMessage",
                            "topic": "tf",
                            "frame_id": "world",
                        }
                    },
                )
            )
        else:  # static: a welded prop's frame, published once on the latched /tf_static.
            # A welded body's world pose is model-fixed, so one mj_forward (configure runs before the
            # engine's) resolves data.xpos/xquat. read is a no-op -- the bridge sends the static_tf hint
            # once at bind and never streams.
            mujoco.mj_forward(ctx.model, ctx.data)
            ctx.interface.add(
                Endpoint(
                    name=f"{self.entity_name}_pose",
                    direction="out",
                    owner=self.entity_name,
                    read=lambda: None,
                    backend={
                        "ros2": {
                            "type": "tf2_msgs.msg.TFMessage",
                            "topic": "tf",
                            "frame_id": self._body_frame,
                            "static_tf": {
                                "parent": "world",
                                "translation": ctx.data.xpos[self._body_id].tolist(),
                                "rotation": ctx.data.xquat[self._body_id].tolist(),
                            },
                        }
                    },
                )
            )

    def on_reset(self, ctx: SimContext) -> None:
        """Re-seat a free prop at its spawn pose, and stop it moving.

        Only for ``free: true``: a welded prop cannot drift, but a movable one is wherever the last
        episode left it -- on the floor, if the robot knocked it off the table. Repetitions of a trial
        would then not be repetitions. ``mj_resetData`` restores ``qpos0``, which already holds the
        spawn pose, but velocity is cleared per-joint here so a prop reset mid-flight does not keep it.
        """
        if not self.free or not self._spawn_qpos:
            return
        adr, *pose = self._spawn_qpos
        adr = int(adr)
        ctx.data.qpos[adr : adr + 7] = pose
        jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, self._base_joint)
        dofadr = int(ctx.model.jnt_dofadr[jid])
        ctx.data.qvel[dofadr : dofadr + 6] = 0.0

    def read_pose(self):
        """Endpoint ``read`` (physics thread): the root body's world pose as a one-entry TF payload
        ``[(frame, pos[3], quat_wxyz[4])]``. ``frame`` is the MuJoCo body name (== the exported scene
        body name) so a viewer binds the transform to its node by name."""
        d = self._ctx.data
        return [(self._body_frame, d.xpos[self._body_id], d.xquat[self._body_id])]
