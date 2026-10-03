"""Scene plugin: a **parametric** four-legged table, every dimension config.

A plain work table for the places a scene needs one at a given height and size: a hand-over stand at
a mobile robot's transfer height, a bench a mobile manipulator picks from, a table a conveyor stands
on. The library's mesh tables (``industrial_table``, ``office_table``) have one size each, and scaling
one changes its legs and its top together; here the top's size, its height and the legs are separate
keys. Built from primitive boxes at build time and welded in place (static scenery, no free joint),
like ``shelf`` and ``workbench``.

Geometry (all metres), origin at the footprint's centre on the floor, so a pose of (x y z) drops the
table exactly there; the top's long side runs along X.

  - Top: ``width`` x ``depth`` x ``top_thickness``, its upper surface exactly at ``height``.
  - Legs: four square legs of side ``leg``, from the floor to the top's underside, their outer faces
    ``leg_inset`` in from the top's edges.
  - Apron: with ``apron: true`` (default), a rail of ``apron_height`` under each edge between the legs,
    stiffening the look and giving a mobile base a face its scanner sees at a plausible height;
    ``apron: false`` leaves the space between the legs open, so a base or a tote can pass under.

Config::

    table:
      prefix: ""            # MJCF name prefix (distinct prefixes for >1 table)
      pose:                 # world placement, a geometry_msgs/Pose (roqsim.pose); omitted
        position: {x: 0.0, y: 0.0, z: 0.0}          #   components are 0
        orientation: {roll: 0.0, pitch: 0.0, yaw: 0.0}
      height: 0.75          # top surface above the floor, m
      width: 1.20           # top along X, m
      depth: 0.80           # top along Y, m
      top_thickness: 0.03   # m
      leg: 0.05             # square leg side, m
      leg_inset: 0.03       # leg outer face in from the top's edge, m
      apron: true           # rails under the edges between the legs
      apron_height: 0.08    # m
      top_rgba: [0.55, 0.56, 0.56, 1.0]
      frame_rgba: [0.72, 0.73, 0.75, 1.0]
      friction: [1.0, 0.005, 0.0001]   # the top's sliding / torsional / rolling friction

Everything collides: the top is what is set down on, the legs and the apron are what a base or an arm
can strike. The top's friction is config because what a scene sets on it -- a tote, a carton, a part
-- is held or slides by it.
"""

from __future__ import annotations

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors

_DEFAULTS = {
    "height": 0.75,
    "width": 1.20,
    "depth": 0.80,
    "top_thickness": 0.03,
    "leg": 0.05,
    "leg_inset": 0.03,
    "apron_height": 0.08,
}
_TOP_RGBA = [0.55, 0.56, 0.56, 1.0]  # mid grey, darker than the frame so the top reads as a top
_FRAME_RGBA = [0.72, 0.73, 0.75, 1.0]  # anodised aluminium
_FRICTION = [1.0, 0.005, 0.0001]


class TablePlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a ``components:`` block.
    provides_entity = True
    _ROOT_BODY = "table"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.pos, self.quat = config_pose(self.config)
        # Bad values are tolerated here (kept as the default) so validate_config reports them with a
        # message rather than construction crashing.
        for key, default in _DEFAULTS.items():
            setattr(self, key, self._float(self.config.get(key), default))
        self.apron = bool(self.config.get("apron", True))
        self.top_rgba = self._vector(self.config.get("top_rgba"), _TOP_RGBA, 4)
        self.frame_rgba = self._vector(self.config.get("frame_rgba"), _FRAME_RGBA, 4)
        self.friction = self._vector(self.config.get("friction"), _FRICTION, 3)

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _vector(value, default: list[float], n: int) -> list[float]:
        try:
            v = [float(x) for x in value]
        except (TypeError, ValueError):
            return list(default)
        return v if len(v) == n else list(default)

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        for key in _DEFAULTS:
            if key not in config:
                continue
            try:
                if float(config[key]) <= 0:
                    errors.append(f"'{key}' must be > 0 (m)")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number > 0 (m)")
        if "leg_inset" in config:
            try:
                if float(config["leg_inset"]) < 0:
                    errors = [e for e in errors if not e.startswith("'leg_inset'")]
                    errors.append("'leg_inset' must be >= 0 (m)")
            except (TypeError, ValueError):
                pass
        g = {k: self._float(config.get(k), getattr(self, k)) for k in _DEFAULTS}
        if g["top_thickness"] >= g["height"]:
            errors.append(f"'top_thickness' {g['top_thickness']:g} m must be less than 'height' {g['height']:g} m")
        for side in ("width", "depth"):
            if 2 * (g["leg_inset"] + g["leg"]) > g[side]:
                errors.append(
                    f"'{side}' {g[side]:g} m leaves no room for two legs of {g['leg']:g} m "
                    f"inset {g['leg_inset']:g} m"
                )
        if config.get("apron", True) and g["apron_height"] > g["height"] - g["top_thickness"]:
            errors.append("'apron_height' must fit under the top")
        for key, n in (("top_rgba", 4), ("frame_rgba", 4), ("friction", 3)):
            if key in config:
                try:
                    ok = len([float(x) for x in config[key]]) == n
                except (TypeError, ValueError):
                    ok = False
                if not ok:
                    errors.append(f"'{key}' must be a list of {n} numbers")
        errors += config_pose_errors(config, "table")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        top_mat = child.add_material()
        top_mat.name = "table_top"
        top_mat.rgba = self.top_rgba
        top_mat.specular, top_mat.shininess = 0.2, 0.15
        frame_mat = child.add_material()
        frame_mat.name = "table_frame"
        frame_mat.rgba = self.frame_rgba
        frame_mat.specular, frame_mat.shininess = 0.6, 0.5

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY
        under = self.height - self.top_thickness

        top = self._box(
            body, "top", [self.width / 2, self.depth / 2, self.top_thickness / 2],
            [0.0, 0.0, self.height - self.top_thickness / 2], top_mat.name,
        )
        top.friction = self.friction

        lx = self.width / 2 - self.leg_inset - self.leg / 2
        ly = self.depth / 2 - self.leg_inset - self.leg / 2
        for sx, sy, tag in ((-1, -1, "rl"), (1, -1, "fl"), (1, 1, "fr"), (-1, 1, "rr")):
            self._box(
                body, f"leg_{tag}", [self.leg / 2, self.leg / 2, under / 2],
                [sx * lx, sy * ly, under / 2], frame_mat.name,
            )

        if self.apron:
            z = under - self.apron_height / 2
            half_t = self.leg / 4  # a rail half as thick as a leg, flush with the legs' outer faces
            span_x = lx - self.leg / 2  # between the legs' inner faces
            span_y = ly - self.leg / 2
            for sy, tag in ((-1, "y_neg"), (1, "y_pos")):
                self._box(
                    body, f"apron_{tag}", [span_x, half_t, self.apron_height / 2],
                    [0.0, sy * (ly + self.leg / 2 - half_t), z], frame_mat.name,
                )
            for sx, tag in ((-1, "x_neg"), (1, "x_pos")):
                self._box(
                    body, f"apron_{tag}", [half_t, span_y, self.apron_height / 2],
                    [sx * (lx + self.leg / 2 - half_t), 0.0, z], frame_mat.name,
                )

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    @staticmethod
    def _box(body, name, half, pos, material):
        g = body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(half)
        g.pos = list(pos)
        g.material = material
        return g

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                # The top's height and size are what a trial is laid out against, so they belong in
                # the entity record a run's ground truth carries.
                meta={"prefix": self.prefix, "height": self.height, "width": self.width, "depth": self.depth},
            )
        )
