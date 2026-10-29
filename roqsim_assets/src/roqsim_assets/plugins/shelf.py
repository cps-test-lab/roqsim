"""Scene plugin: a **parametric** chipboard shelf built from primitive boxes at build time.

Instead of frozen mesh geometry, the shelf is generated from box geoms so its dimensions -- most
usefully the **number of layers** -- are config, not vertex data. ``layers: 8`` gives an eight-board
shelf; ``depth: 0.40`` a half-depth one. Like ``spawn_model`` the prop is welded in place (static
scenery, no free joint).

Geometry (all metres): ``layers`` horizontal boards (box half-extents ``[depth/2, width/2,
thickness/2]``) evenly stacked in Z between the base-board height and the top, plus four vertical corner
uprights spanning the full height. The boards are a plain chipboard colour and the uprights grey; the
shelf needs no asset beyond this module.

Config::

    shelf:
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 shelf)
      pose:                 # world placement, a geometry_msgs/Pose (roqsim.pose); omitted
        position: {x: 0.0, y: 0.0, z: 0.0}          #   components are 0
        orientation: {roll: 0.0, pitch: 0.0, yaw: 0.0}   #   or a quaternion x/y/z/w
      layers: 5           # number of boards (int >= 2; default 5)
      width: 1.51         # board size along Y, m   (default 1.51)
      depth: 0.80         # board size along X, m   (default 0.80; half-depth => 0.40)
      height: 2.00        # overall shelf height, m (default 2.00)
      thickness: 0.02     # board thickness, m      (default 0.02)
      leg: 0.04           # square cross-section of a corner upright, m (default 0.04)
"""

from __future__ import annotations

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors

# Height (m) of the bottom board's centre above the floor: a base board raised clear of the floor.
_Z0 = 0.08

_FRAME_RGBA = [0.72, 0.72, 0.74, 1.0]  # neutral grey uprights
_CHIPBOARD_RGBA = [0.82, 0.70, 0.52, 1.0]  # plain chipboard tone for the boards


class ShelfPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "shelf"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.pos, self.quat = config_pose(self.config)
        # Geometry. Bad values are tolerated here (kept as the default) so validate_config can report
        # them with a friendly message rather than crashing during construction.
        self.layers = self._int(self.config.get("layers"), 5)
        self.width = self._float(self.config.get("width"), 1.51)
        self.depth = self._float(self.config.get("depth"), 0.80)
        self.height = self._float(self.config.get("height"), 2.00)
        self.thickness = self._float(self.config.get("thickness"), 0.02)
        self.leg = self._float(self.config.get("leg"), 0.04)
        self._body_frame = ""

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value, default: int) -> int:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return default
        return int(f) if f == int(f) else default  # keep a non-integer as default -> flagged below

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if "layers" in config:
            layers = config["layers"]
            if not isinstance(layers, int) or isinstance(layers, bool) or layers < 2:
                errors.append("'layers' must be an integer >= 2")
        for key in ("width", "depth", "height", "thickness", "leg"):
            if key not in config:
                continue
            try:
                if float(config[key]) <= 0:
                    errors.append(f"'{key}' must be > 0")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number > 0")
        # Boards must fit inside the shelf height with room to stack (the top board's centre sits at
        # height - thickness/2, the bottom at _Z0; they must not cross).
        try:
            layers = int(config.get("layers", self.layers))
            thickness = float(config.get("thickness", self.thickness))
            height = float(config.get("height", self.height))
            if layers >= 2 and thickness > 0 and height - thickness / 2 <= _Z0:
                errors.append(f"'height' too small: must exceed {_Z0 + thickness / 2:.3f} m")
            if layers * thickness >= height:
                errors.append("boards do not fit: 'thickness' * 'layers' must be < 'height'")
        except (TypeError, ValueError):
            pass  # already reported by the per-key checks above
        errors += config_pose_errors(config, "shelf")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        chipboard, frame_mat = self._materials(child)

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        # Boards: evenly spaced centres from the base-board height (_Z0) to just under the top.
        z_top = self.height - self.thickness / 2
        spacing = (z_top - _Z0) / (self.layers - 1)
        half = [self.depth / 2, self.width / 2, self.thickness / 2]
        for i in range(self.layers):
            g = body.add_geom()
            g.name = f"board_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = half
            g.pos = [0.0, 0.0, _Z0 + i * spacing]
            g.material = chipboard

        # Four corner uprights spanning the full height, seated just inside the board footprint.
        lx = self.depth / 2 - self.leg / 2
        ly = self.width / 2 - self.leg / 2
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            g = body.add_geom()
            g.name = f"leg_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [self.leg / 2, self.leg / 2, self.height / 2]
            g.pos = [sx * lx, sy * ly, self.height / 2]
            g.material = frame_mat

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def _materials(self, child: mujoco.MjSpec) -> tuple[str, str]:
        """Add the chipboard + grey-frame materials to the child spec; return their names."""
        frame_mat = child.add_material()
        frame_mat.name = "shelf_frame"
        frame_mat.rgba = _FRAME_RGBA

        chipboard = child.add_material()
        chipboard.name = "shelf_chipboard"
        chipboard.rgba = _CHIPBOARD_RGBA
        return "shelf_chipboard", "shelf_frame"

    def configure(self, ctx: SimContext) -> None:
        self._body_frame = self.prefix + self._ROOT_BODY
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self._body_frame,
                meta={"prefix": self.prefix, "layers": self.layers},
            )
        )
