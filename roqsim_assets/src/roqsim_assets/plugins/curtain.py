"""Scene plugin: a pleated curtain hung from a bar -- visual scenery, matte by default.

A backdrop behind a cell: what a camera sees the cell against. The cloth is a zigzag of thin panels,
``pleats`` folds across ``width``, each fold ``pleat_depth`` deep, from the bar down to ``hem``
above the floor; the bar is a rod on clamps at ``height``. Nothing of it collides (cloth that a
robot brushes does not stop it), and the default colour is a matte black, as a photographic
backdrop is, so a renderer draws it without highlights.

Origin: the curtain's centre on the floor, under the bar; the cloth spans X, its folds in Y, and
the side it is seen from is -Y. A pose places and turns it.

Config::

    curtain:
      pose: {position: {x: 0.0, y: 1.0}, orientation: {yaw: 0.0}}
      width: 3.0          # m, along X
      height: 2.0         # m, the bar above the floor
      hem: 0.02           # m, the cloth's lower edge above the floor
      pleats: 24          # folds across the width
      pleat_depth: 0.06   # m, front to back of a fold
      rgba: [0.025, 0.025, 0.03, 1.0]
      bar: true           # the rod and its clamps
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors
from roqsim.schema import Field

_BAR_RGBA = [0.15, 0.15, 0.16, 1.0]
_CLAMP_RGBA = [0.08, 0.08, 0.08, 1.0]
_CLOTH_T = 0.002


class CurtainPlugin(Plugin):
    provides_entity = True
    _ROOT_BODY = "curtain"

    CONFIG_SCHEMA = {
        "prefix": Field(str, default="", doc="MJCF name prefix; distinct per curtain"),
        "pose": Field(dict, doc="the curtain's centre on the floor, under the bar"),
        "width": Field(float, default=3.0, minimum=0.1, unit="m", doc="along X"),
        "height": Field(float, default=2.0, minimum=0.1, unit="m", doc="the bar above the floor"),
        "hem": Field(
            float, default=0.02, minimum=0.0, unit="m", doc="the cloth's lower edge above the floor"
        ),
        "pleats": Field(int, default=24, minimum=1, doc="folds across the width"),
        "pleat_depth": Field(
            float, default=0.06, minimum=0.0, unit="m", doc="front to back of one fold"
        ),
        "rgba": Field(list, default=[0.025, 0.025, 0.03, 1.0], length=4, doc="the cloth's colour"),
        "bar": Field(bool, default=True, doc="the rod and its clamps"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.pos, self.quat = config_pose(self.config)

    def validate_config(self, config: dict) -> list[str]:
        errors = config_pose_errors(config, f"curtain {self.address}")
        try:
            s = self.settings_for(config)
            if s.hem >= s.height:
                errors.append(f"curtain {self.address}: 'hem' must be under 'height'")
        except (TypeError, ValueError):
            pass  # the schema reports the shape
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        s = self.settings
        child = mujoco.MjSpec()
        cloth = child.add_material()
        cloth.name = "curtain_cloth"
        cloth.rgba = [float(v) for v in s.rgba]
        cloth.specular = 0.0
        cloth.shininess = 0.0
        cloth.reflectance = 0.0
        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        # Two panels per fold, each across half a fold and tilted by the depth it spans.
        n = 2 * s.pleats
        run = s.width / n
        rise = s.pleat_depth
        length = math.hypot(run, rise)
        angle = math.atan2(rise, run)
        z0, z1 = s.hem, s.height - 0.03
        for i in range(n):
            x = -s.width / 2 + (i + 0.5) * run
            sign = 1.0 if i % 2 == 0 else -1.0
            g = body.add_geom()
            g.name = f"cloth_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [length / 2, _CLOTH_T / 2, (z1 - z0) / 2]
            g.pos = [x, 0.0, (z0 + z1) / 2]
            half = sign * angle / 2
            g.quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
            g.material = cloth.name
            g.contype = 0
            g.conaffinity = 0

        if s.bar:
            bar = body.add_geom()
            bar.name = "bar"
            bar.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            bar.size = [0.012, s.width / 2 + 0.05, 0.0]
            bar.pos = [0.0, 0.0, s.height]
            bar.quat = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]  # axis along X
            bar.rgba = _BAR_RGBA
            bar.contype = 0
            bar.conaffinity = 0
            for i in range(s.pleats + 1):
                c = body.add_geom()
                c.name = f"clamp_{i}"
                c.type = mujoco.mjtGeom.mjGEOM_BOX
                c.size = [0.012, 0.015, 0.03]
                c.pos = [-s.width / 2 + i * s.width / s.pleats, 0.0, s.height - 0.03]
                c.rgba = _CLAMP_RGBA
                c.contype = 0
                c.conaffinity = 0

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=s.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        s = self.settings
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=s.prefix + self._ROOT_BODY,
                meta={"width": s.width, "height": s.height},
            )
        )
