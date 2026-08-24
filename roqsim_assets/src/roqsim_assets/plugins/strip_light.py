"""Scene plugin: a **parametric** linear ceiling luminaire -- the LED batten, optionally lit.

The long white line under an office soffit. Two things in one, because in a scene they are one thing:
the *fixture* (a slim box you can see) and, with ``emit: true``, an actual MuJoCo light under it, so
the room is lit from where the light visibly comes from instead of from a single lamp at the plan's
centre.

The light is added to the **worldbody**, not to the fixture: the core ``ceiling`` plugin deletes
ceiling *geoms* by height, so opening the roof for a top-down view takes the batten out of the
picture but leaves the room lit -- which is what a top-down view wants.

Geometry (all metres). ``pos`` is the fixture's centre, ``yaw`` runs it along a direction, ``length``
is along that direction and ``width`` across it.

Config::

    strip_light:
      name: strip         # entity name (default 'strip')
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 batten)
      pos: [x, y, z]      # fixture centre, world (z = the underside of the ceiling it hangs on)
      yaw: 0.0            # direction of the run, rad
      length: 2.4         # along yaw, m
      width: 0.09         # across it, m
      height: 0.06        # how deep the fixture hangs, m
      color: [r,g,b,a]    # fixture colour (default white)
      emission: 0.6       # 0..1 self-illumination, so the batten reads as lit, not just white
      emit: false         # also add a real light below the fixture
      diffuse: [0.25, 0.25, 0.25]   # its colour/intensity (only with emit)
      cutoff: 80.0        # its spot half-angle in degrees (only with emit)

Keep ``emit`` for the few battens that should actually light the scene: MuJoCo caps a model at 100
lights, and every extra one costs render time for a room that is already lit.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_FIXTURE_RGBA = [0.97, 0.97, 0.95, 1.0]
_DIFFUSE = [0.25, 0.25, 0.25]


class StripLightPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "strip_light"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        pos = self.config.get("pos", [0.0, 0.0, 3.5])
        self.pos = [float(v) for v in pos] if len(pos) == 3 else [0.0, 0.0, 3.5]
        self.yaw = self._float(self.config.get("yaw"), 0.0)
        self.length = self._float(self.config.get("length"), 2.4)
        self.width = self._float(self.config.get("width"), 0.09)
        self.height = self._float(self.config.get("height"), 0.06)
        self.color = self._rgba(self.config.get("color")) or _FIXTURE_RGBA
        self.emission = self._float(self.config.get("emission"), 0.6)
        self.emit = bool(self.config.get("emit", False))
        self.diffuse = self._rgba(self.config.get("diffuse")) or _DIFFUSE
        self.cutoff = self._float(self.config.get("cutoff"), 80.0)

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

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

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if len(config.get("pos", [0, 0, 0])) != 3:
            errors.append("'pos' must be [x, y, z] -- a luminaire needs its mounting height")
        for key in ("length", "width", "height"):
            if key in config:
                try:
                    if float(config[key]) <= 0:
                        errors.append(f"'{key}' must be > 0")
                except (TypeError, ValueError):
                    errors.append(f"'{key}' must be a number > 0")
        if "emit" in config and not isinstance(config["emit"], bool):
            errors.append("'emit' must be a boolean")
        if "emission" in config:
            try:
                if not 0.0 <= float(config["emission"]) <= 1.0:
                    errors.append("'emission' must be between 0 and 1")
            except (TypeError, ValueError):
                errors.append("'emission' must be a number between 0 and 1")
        for key in ("color", "diffuse"):
            if key in config and self._rgba(config[key]) is None:
                errors.append(f"'{key}' must be [r, g, b] or [r, g, b, a] numbers")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        mat = child.add_material()
        mat.name = "luminaire"
        mat.rgba = self.color
        mat.emission = self.emission  # a lamp that is merely white reads as a paper strip
        mat.specular = 0.2

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY
        g = body.add_geom()
        g.name = "fixture"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [self.length / 2, self.width / 2, self.height / 2]
        g.pos = [0.0, 0.0, -self.height / 2]  # hangs below the mounting height
        g.material = mat.name
        g.contype = 0
        g.conaffinity = 0

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        spec.attach(child, prefix=self.prefix, frame=frame)

        if self.emit:
            light = spec.worldbody.add_light()
            light.name = f"{self.prefix}{self._ROOT_BODY}_light"
            light.pos = [self.pos[0], self.pos[1], self.pos[2] - self.height]
            light.dir = [0.0, 0.0, -1.0]
            light.diffuse = self.diffuse[:3]
            light.exponent = 0.0  # flat across the cone; MuJoCo's default 10 makes a hotspot
            light.cutoff = self.cutoff

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                meta={"prefix": self.prefix, "length": self.length, "emit": self.emit},
            )
        )
