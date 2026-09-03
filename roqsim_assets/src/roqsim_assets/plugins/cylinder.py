"""Scene plugin: a **parametric** cylindrical obstacle placed from the world YAML.

The round sibling of ``box``. Between them they cover the two shapes anonymous scenery actually
takes: a carton and a post. Poles, bollards, pillars, tree trunks, traffic cones and the cylinder
fields that procedurally generated navigation benchmarks are built from are all this object, and
until now the only way to place one was to bake it into a scene or to find a modelled asset with the
right proportions -- ``spawn_model`` scales uniformly, so it cannot turn a 0.3 m drum into a
0.075 m post without shrinking its height too.

The distinction from ``box`` is not cosmetic where clearance is the experiment. Two diagonally
adjacent 0.15 m boxes touch at their corners and leave no gap; two tangent 0.15 m cylinders on the
same pitch leave one. A benchmark whose difficulty is defined by how tightly the robot must squeeze
measures something different if its round obstacles are squared off.

Like every plugin in this package the cylinder is **welded scenery** -- static, no free joint -- and
is declared in the world YAML rather than baked into the scene, so it stays out of the occupancy grid
the map is generated from. For something physics should move, use ``spawn_model`` with ``free: true``.

Config::

    cylinder:
      prefix: ""           # MJCF name prefix (use distinct prefixes for >1 cylinder)
      pos: [x, y]          # centre in world metres; [x, y] stands the cylinder ON the floor,
                           #   [x, y, z] places its CENTRE at z (REQUIRED)
      radius: 0.075        # metres (REQUIRED)
      height: 0.5          # full height, metres (REQUIRED)
      color: [r,g,b,a]     # default a light warehouse grey; alpha optional
      collide: true        # false -> visual only (raycast still sees it; nothing bumps into it)
      friction: 1.0        # sliding friction, or the full [sliding, torsional, rolling] triple

``height`` is the **full** height and ``radius`` is a true radius, matching how a world file talks
about a post. MuJoCo's cylinder geom wants ``[radius, half-height]``; that conversion happens here so
it never has to happen in your head -- the same reason ``box`` takes full extents.
"""

from __future__ import annotations

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_GREY_RGBA = [0.86, 0.86, 0.83, 1.0]  # pale warehouse grey, as box
_ROOT_BODY = "cylinder"


class CylinderPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.radius = self._float(self.config.get("radius"), 0.2)
        self.height = self._float(self.config.get("height"), 0.5)
        self.pos = self._pos(self.config.get("pos"), self.height)
        self.color = self._rgba(self.config.get("color")) or _GREY_RGBA
        self.collide = bool(self.config.get("collide", True))
        self.friction = self._friction(self.config.get("friction"))

    # -- config coercion -------------------------------------------------------------------------
    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _pos(self, value, height: float) -> tuple[float, float, float]:
        """``[x, y]`` stands the cylinder on the floor; ``[x, y, z]`` places its centre at z."""
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
        for key in ("pos", "radius", "height"):
            if key not in config:
                errors.append(f"'{key}' is required")
        if "pos" in config:
            try:
                if len(config["pos"]) not in (2, 3):
                    errors.append("'pos' must be [x, y] or [x, y, z] in world metres")
            except TypeError:
                errors.append("'pos' must be [x, y] or [x, y, z] in world metres")
        for key in ("radius", "height"):
            if key in config:
                try:
                    value = float(config[key])
                except (TypeError, ValueError):
                    errors.append(f"'{key}' must be a number in metres")
                else:
                    if value <= 0:
                        errors.append(f"'{key}' must be positive, got {value}")
        if "color" in config and self._rgba(config["color"]) is None:
            errors.append("'color' must be [r, g, b] or [r, g, b, a] numbers")
        if "collide" in config and not isinstance(config["collide"], bool):
            errors.append("'collide' must be a boolean")
        return errors

    # -- lifecycle -------------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        mat = child.add_material()
        mat.name = "cylinder_surface"
        mat.rgba = self.color

        body = child.worldbody.add_body()
        body.name = _ROOT_BODY

        g = body.add_geom()
        g.name = "cylinder"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [self.radius, self.height / 2, 0.0]  # MuJoCo wants [radius, half-height]
        g.material = mat.name
        g.friction = self.friction
        if not self.collide:
            g.contype = 0
            g.conaffinity = 0

        frame = spec.worldbody.add_frame()
        frame.pos = list(self.pos)
        # A cylinder is rotationally symmetric about its own axis, so unlike `box` there is no `yaw`:
        # there would be nothing for it to do.
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + _ROOT_BODY,
                meta={
                    "prefix": self.prefix,
                    "radius": self.radius,
                    "height": self.height,
                    "pos": list(self.pos),
                },
            )
        )
