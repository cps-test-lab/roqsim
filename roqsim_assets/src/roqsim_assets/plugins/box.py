"""Scene plugin: a **parametric** rectangular box obstacle placed from the world YAML.

The plainest possible piece of scenery, and the one the library was missing. Every other prop here
is a specific object (a shelf, a workbench, a duct); ``spawn_model`` places a modelled asset but
scales it only **uniformly**, so it cannot make a 0.4 x 0.4 x 0.8 m pillar out of a 0.67 x 0.53 x
0.22 m carton. Navigation experiments are full of anonymous boxes whose only meaningful properties
are *where* and *how big*, and until now the only way to place one was to bake it into a scene.

Baking is the wrong home for an obstacle the robot is not supposed to know about. A scene is what the
occupancy grid is generated from, so a box baked into the scene lands in the map -- and an experiment
about **unknown** static obstacles then has none. Declaring the box in the world YAML instead keeps
map and world structurally separate rather than relying on someone regenerating the grid at the right
moment.

Config::

    box:
      prefix: ""           # MJCF name prefix (use distinct prefixes for >1 box)
      pos: [x, y]          # centre in world metres; [x, y] sits the box ON the floor,
                           #   [x, y, z] places its CENTRE at z (REQUIRED)
      size: [0.4, 0.4, 0.8]  # full extents (not half-extents), metres (REQUIRED)
      yaw: 0.0             # rotation about z, radians
      color: [r,g,b,a]     # default a light warehouse grey; alpha optional
      collide: true        # false -> visual only (raycast still sees it; nothing bumps into it)
      friction: 1.0        # sliding friction, or the full [sliding, torsional, rolling] triple
      free: false          # give the box a free joint: movable, and TELEPORTABLE (see below)

``size`` is deliberately **full extents**, not MuJoCo half-extents: a world file describes a 0.4 m
box, and halving it in your head is exactly the kind of silent factor-of-two a scene should not ask
of its author.

By default the box is welded scenery -- static, with no free joint -- like every other plugin in this
package. ``free: true`` gives it a free joint, which buys two things: physics can move it, and
``simulation_interfaces``' ``SetEntityState`` can **teleport** it (that service rejects any entity
without a free ``base_joint``).

Teleporting is how an obstacle *appears* mid-trial. roqsim never recompiles the model at runtime, so
there is no spawning: a box that must show up on cue is compiled in at build time, parked somewhere
harmless (below the floor, say), and moved into place by the scenario when the moment comes. Between
episodes ``on_reset`` puts it back at its declared pose, so a trial never inherits the previous
trial's obstacle position.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_GREY_RGBA = [0.86, 0.86, 0.83, 1.0]  # pale warehouse carton
_ROOT_BODY = "box"


class BoxPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.size = self._vec3(self.config.get("size"), (0.4, 0.4, 0.4))
        self.pos = self._pos(self.config.get("pos"), self.size[2])
        self.yaw = self._float(self.config.get("yaw"), 0.0)
        self.color = self._rgba(self.config.get("color")) or _GREY_RGBA
        self.collide = bool(self.config.get("collide", True))
        self.friction = self._friction(self.config.get("friction"))
        self.free = bool(self.config.get("free", False))
        self._base_joint = ""
        self._spawn_qpos: list[float] | None = None

    # -- config coercion -------------------------------------------------------------------------
    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _vec3(value, default: tuple[float, float, float]) -> tuple[float, float, float]:
        try:
            if len(value) >= 3:
                return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            pass
        return default

    def _pos(self, value, height: float) -> tuple[float, float, float]:
        """``[x, y]`` sits the box on the floor; ``[x, y, z]`` places its centre at z."""
        try:
            if len(value) >= 3:
                return float(value[0]), float(value[1]), float(value[2])
            if len(value) == 2:
                return float(value[0]), float(value[1]), height / 2.0
        except (TypeError, ValueError):
            pass
        return 0.0, 0.0, height / 2.0

    @staticmethod
    def _rgba(value) -> list[float] | None:
        if value is None:
            return None
        try:
            rgba = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if len(rgba) == 3:
            rgba.append(1.0)
        return rgba if len(rgba) == 4 else None

    @staticmethod
    def _friction(value) -> list[float]:
        if value is None:
            return [1.0, 0.005, 0.0001]
        try:
            if isinstance(value, (int, float)):
                return [float(value), 0.005, 0.0001]
            triple = [float(v) for v in value]
        except (TypeError, ValueError):
            return [1.0, 0.005, 0.0001]
        return triple if len(triple) == 3 else [1.0, 0.005, 0.0001]

    # -- validation ------------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        for key in ("pos", "size"):
            if key not in config:
                errors.append(f"'{key}' is required")
        if "pos" in config:
            try:
                if len(config["pos"]) not in (2, 3):
                    errors.append("'pos' must be [x, y] or [x, y, z] in world metres")
            except TypeError:
                errors.append("'pos' must be [x, y] or [x, y, z] in world metres")
        if "size" in config:
            try:
                extents = [float(v) for v in config["size"]]
            except (TypeError, ValueError):
                errors.append("'size' must be three numbers: full extents [x, y, z] in metres")
            else:
                if len(extents) != 3:
                    errors.append("'size' must be three numbers: full extents [x, y, z] in metres")
                elif any(v <= 0 for v in extents):
                    errors.append(f"'size' must be positive in every axis, got {extents}")
        if "color" in config and self._rgba(config["color"]) is None:
            errors.append("'color' must be [r, g, b] or [r, g, b, a] numbers")
        if "collide" in config and not isinstance(config["collide"], bool):
            errors.append("'collide' must be a boolean")
        if "free" in config and not isinstance(config["free"], bool):
            errors.append("'free' must be a boolean")
        return errors

    # -- lifecycle -------------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        mat = child.add_material()
        mat.name = "box_surface"
        mat.rgba = self.color

        body = child.worldbody.add_body()
        body.name = _ROOT_BODY

        g = body.add_geom()
        g.name = "box"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [self.size[0] / 2, self.size[1] / 2, self.size[2] / 2]  # MuJoCo wants half-extents
        g.material = mat.name
        g.friction = self.friction
        if not self.collide:
            g.contype = 0
            g.conaffinity = 0

        if self.free:
            # A free joint is what makes the box MOVABLE -- and, more to the point, what lets
            # simulation_interfaces' SetEntityState teleport it (the service rejects any entity
            # without a free base_joint). roqsim forbids mid-run recompilation, so a scenario that
            # needs an obstacle to APPEAR must compile it in up front, park it out of the way and
            # teleport it in on cue. Naming and meta follow spawn_model's convention so both kinds
            # of prop are teleported by the same code path.
            body.add_freejoint(name="free")
            self._base_joint = f"{self.prefix}free"

        frame = spec.worldbody.add_frame()
        frame.pos = list(self.pos)
        frame.quat = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        meta = {
            "prefix": self.prefix,
            "size": list(self.size),
            "pos": list(self.pos),
            "yaw": self.yaw,
        }
        if self.free:
            meta["base_joint"] = self._base_joint
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="object" if self.free else "prop",
                body=self.prefix + _ROOT_BODY,
                meta=meta,
            )
        )
        if self.free:
            jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, self._base_joint)
            if jid < 0:
                raise RuntimeError(f"box: free joint {self._base_joint!r} not found")
            adr = int(ctx.model.jnt_qposadr[jid])
            # Remember the spawn pose so on_reset re-seats the box between episodes instead of
            # leaving it wherever the last trial pushed (or teleported) it.
            self._spawn_qpos = [adr, *(float(v) for v in ctx.model.qpos0[adr : adr + 7])]

    def on_reset(self, ctx: SimContext) -> None:
        if self._spawn_qpos is None:
            return
        adr = int(self._spawn_qpos[0])
        ctx.data.qpos[adr : adr + 7] = self._spawn_qpos[1:]
        jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, self._base_joint)
        dof = int(ctx.model.jnt_dofadr[jid])
        ctx.data.qvel[dof : dof + 6] = 0.0
