"""Scene plugin: a **parametric** field of flat acoustic ceiling panels under a soffit.

The white rectangles that hang a few centimetres under an exposed concrete deck in an open-plan
office. Parametric for the same reason ``shelf`` is: a room's panel field is a rectangle, a panel
size and a spacing, and freezing one MJCF per room would mean a new model file every time a wall
moves. One entry covers a whole room.

Geometry (all metres). ``area`` is the rectangle to cover, in world coordinates; panels are laid on
a regular grid centred in it and only those that fit **entirely inside** are emitted, so a field
never pokes through the walls it was measured against. ``z`` is the soffit (the ceiling's underside)
and each panel hangs ``drop`` below it -- a panel is drawn where a real one is, not inside the slab.

Everything this plugin emits sits above head height, which is what lets the core ``ceiling`` plugin
(``roqsim``) take the whole ceiling out by height for a top-down view: list ``ceiling`` *after* this one
in the world's plugins, since build order is list order and it can only delete what already exists.

Config::

    ceiling_panels:
      name: panels        # entity name (default 'panels')
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 field)
      area: [x0, y0, x1, y1]  # world rectangle to cover (REQUIRED)
      z: 3.5              # soffit height, m -- panels hang below this
      drop: 0.04          # gap between soffit and panel top, m
      panel: [1.8, 0.6]   # panel size [x, y] before `yaw`, m
      pitch: [2.6, 1.6]   # grid spacing [x, y], m (>= panel, or the panels would overlap)
      thickness: 0.04     # panel thickness, m
      yaw: 0.0            # rotation of the whole field about its centre, rad
      color: [r,g,b,a]    # panel colour (default near-white); alpha optional
      emission: 0.35      # 0..1 self-illumination -- a panel faces away from every lamp in the room
                          #   (they hang at the same height), so without it white panels render as
                          #   dark grey slabs. This is what makes them read as the white they are.

Panels are visual only (no contacts): they are out of reach of anything driving on the floor, and a
convex collider per panel would be pure cost. The lidar raycaster still sees them -- ``mj_multiRay``
ignores contype -- so an upward-looking sensor reads the panelled ceiling, as it should.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_PANEL_RGBA = [0.95, 0.95, 0.94, 1.0]
_PANEL_EMISSION = 0.35


class CeilingPanelsPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "ceiling_panels"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        area = self.config.get("area", [0.0, 0.0, 1.0, 1.0])
        self.area = [float(v) for v in area] if len(area) == 4 else [0.0, 0.0, 1.0, 1.0]
        self.z = self._float(self.config.get("z"), 3.5)
        self.drop = self._float(self.config.get("drop"), 0.04)
        self.thickness = self._float(self.config.get("thickness"), 0.04)
        self.yaw = self._float(self.config.get("yaw"), 0.0)
        self.panel = self._pair(self.config.get("panel"), (1.8, 0.6))
        self.pitch = self._pair(self.config.get("pitch"), (2.6, 1.6))
        self.color = self._rgba(self.config.get("color")) or _PANEL_RGBA
        self.emission = self._float(self.config.get("emission"), _PANEL_EMISSION)

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _pair(value, default: tuple[float, float]) -> tuple[float, float]:
        try:
            if len(value) == 2:
                return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            pass
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
        if "area" not in config:
            errors.append("'area' is required: [x0, y0, x1, y1] of the rectangle to cover")
        elif len(config["area"]) != 4:
            errors.append("'area' must be [x0, y0, x1, y1]")
        else:
            x0, y0, x1, y1 = (self._float(v, 0.0) for v in config["area"])
            if x1 <= x0 or y1 <= y0:
                errors.append("'area' must have x1 > x0 and y1 > y0")
        for key in ("z", "drop", "thickness"):
            if key in config:
                try:
                    float(config[key])
                except (TypeError, ValueError):
                    errors.append(f"'{key}' must be a number")
        for key in ("panel", "pitch"):
            if key in config and len(config[key]) != 2:
                errors.append(f"'{key}' must be [x, y] in metres")
        panel = self._pair(config.get("panel", self.panel), self.panel)
        pitch = self._pair(config.get("pitch", self.pitch), self.pitch)
        for axis, p, q in (("x", panel[0], pitch[0]), ("y", panel[1], pitch[1])):
            if p <= 0:
                errors.append(f"'panel' {axis} must be > 0")
            elif q < p:
                errors.append(
                    f"'pitch' {axis} ({q:g} m) is smaller than 'panel' ({p:g} m): overlap"
                )
        if "color" in config and self._rgba(config["color"]) is None:
            errors.append("'color' must be [r, g, b] or [r, g, b, a] numbers")
        if "emission" in config:
            try:
                if not 0.0 <= float(config["emission"]) <= 1.0:
                    errors.append("'emission' must be between 0 and 1")
            except (TypeError, ValueError):
                errors.append("'emission' must be a number between 0 and 1")
        return errors

    def _centres(self) -> list[tuple[float, float]]:
        """Panel centres in the field's own (yaw-aligned) frame, centred on the area's centre.

        A grid big enough to cover the area's diagonal is laid out and then filtered by whether the
        panel's four corners all land inside the area -- rather than counting rows from the area's
        span. Containment is the actual requirement (a panel must not overhang the room), it is what
        makes a yawed field behave, and it costs nothing at this scale.
        """
        x0, y0, x1, y1 = self.area
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        reach = math.hypot(x1 - x0, y1 - y0) / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hw, hd = self.panel[0] / 2, self.panel[1] / 2
        out = []
        for i in range(-int(reach // self.pitch[0]) - 1, int(reach // self.pitch[0]) + 2):
            for j in range(-int(reach // self.pitch[1]) - 1, int(reach // self.pitch[1]) + 2):
                u, v = i * self.pitch[0], j * self.pitch[1]
                corners = [
                    (cx + (u + du) * c - (v + dv) * s, cy + (u + du) * s + (v + dv) * c)
                    for du in (-hw, hw)
                    for dv in (-hd, hd)
                ]
                if all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in corners):
                    out.append((u, v))
        return out

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        mat = child.add_material()
        mat.name = "acoustic_panel"
        mat.rgba = self.color
        mat.emission = self.emission
        mat.specular = 0.1
        mat.shininess = 0.1

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        top = -self.drop  # relative to the soffit, which the attach frame puts at z
        centres = self._centres()
        for i, (u, v) in enumerate(centres, start=1):
            g = body.add_geom()
            g.name = f"panel_{i:03d}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [self.panel[0] / 2, self.panel[1] / 2, self.thickness / 2]
            g.pos = [u, v, top - self.thickness / 2]
            g.material = mat.name
            g.contype = 0
            g.conaffinity = 0
        self.count = len(centres)

        x0, y0, x1, y1 = self.area
        frame = spec.worldbody.add_frame()
        frame.pos = [(x0 + x1) / 2, (y0 + y1) / 2, self.z]
        frame.quat = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                meta={"prefix": self.prefix, "panels": getattr(self, "count", 0), "z": self.z},
            )
        )
