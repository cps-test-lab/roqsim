"""Scene plugin: a **parametric** chipboard shelf built from primitive boxes at build time.

The customizable counterpart to the baked ``free_chipboard_shelf`` / ``free_chipboard_shelf_half_depth``
mesh models: instead of frozen geometry, the shelf is generated from box geoms so its dimensions -- most
usefully the **number of layers** -- are config, not vertex data. ``layers: 8`` gives an eight-board
shelf; ``depth: 0.40`` reproduces the half-depth variant without a separate model. Like ``spawn_model``
the prop is welded in place (static scenery, no free joint).

Geometry (all metres): ``layers`` horizontal boards (box half-extents ``[depth/2, width/2,
thickness/2]``) evenly stacked in Z between the base-board height and the top, plus four vertical corner
uprights spanning the full height. The boards reuse the real chipboard colour map from the mesh model's
``textures/`` (grey uprights, matching the mesh's frame material); if that model can't be resolved the
boards fall back to a plain wood ``rgba`` and a warning is logged.

Config::

    shelf:
      name: shelf         # the entry's OWN key, not the config's: names the entity (default 'shelf')
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 shelf)
      pos: [0.0, 0.0, 0.0]  # [x, y] or [x, y, z] world placement
      rpy: [0.0, 0.0, 0.0]  # orientation as roll/pitch/yaw (rad)
      layers: 5           # number of boards (int >= 2; default 5, matching the mesh model)
      width: 1.51         # board size along Y, m   (default 1.51)
      depth: 0.80         # board size along X, m   (default 0.80; half-depth => 0.40)
      height: 2.00        # overall shelf height, m (default 2.00)
      thickness: 0.02     # board thickness, m      (default 0.02)
      leg: 0.04           # square cross-section of a corner upright, m (default 0.04)

Defaults reproduce the baked ``free_chipboard_shelf`` (5 boards, ~0.80 x 1.51 x 2.0 m) so it is a
drop-in, parameterised replacement.
"""

from __future__ import annotations

import logging
import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.models import ModelError, resolve_model
from roqsim.plugin import Plugin

logger = logging.getLogger("roqsim_assets.shelf")

# Height (m) of the bottom board's centre above the floor. Lifted from the baked mesh (base board at
# z~=0.077..0.082), so the parametric shelf seats its lowest board where the mesh model does.
_Z0 = 0.08

# The mesh model whose chipboard colour map + frame colour the boards reuse, and the relative texture
# it ships. Kept in sync with free_chipboard_shelf.xml.
_TEXTURE_MODEL = "free_chipboard_shelf"
_TEXTURE_REL = "textures/Material.001_diffuse.png"
_FRAME_RGBA = [0.72, 0.72, 0.74, 1.0]  # neutral grey, copied from the mesh model's frame material
_WOOD_FALLBACK_RGBA = [
    0.82,
    0.70,
    0.52,
    1.0,
]  # plain chipboard tone when the texture can't be found
_TEXTURE_TILE_M = 1.0  # real-world metres one chipboard texture tile spans (via texuniform)


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


class ShelfPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "free_chipboard_shelf"  # matches the mesh model's root body name

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        # Pose parsing tolerates malformed input (falls back to the origin / identity) so a bad length
        # is reported by validate_config with a friendly message rather than crashing construction.
        pos = self.config.get("pos", [0.0, 0.0, 0.0])
        if len(pos) in (2, 3):
            self.pos = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        else:
            self.pos = [0.0, 0.0, 0.0]
        rpy = self.config.get("rpy", [0.0, 0.0, 0.0])
        if len(rpy) == 3:
            self.quat = _rpy_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        else:
            self.quat = [1.0, 0.0, 0.0, 0.0]
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
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")
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
        """Add the chipboard + grey-frame materials to the child spec; return their names.

        The boards reuse the real chipboard texture shipped with the ``free_chipboard_shelf`` mesh
        model (resolved via the model provider, so it works cross-package). If that model or its
        texture cannot be found the boards fall back to a plain wood colour -- logged, not fatal, so a
        missing optional asset never sinks the whole run.
        """
        frame_mat = child.add_material()
        frame_mat.name = "shelf_frame"
        frame_mat.rgba = _FRAME_RGBA

        chipboard = child.add_material()
        chipboard.name = "shelf_chipboard"

        texture_path = None
        try:
            asset = resolve_model(_TEXTURE_MODEL)
            candidate = asset.path.parent / _TEXTURE_REL
            if candidate.is_file():
                texture_path = str(candidate)
            else:
                logger.warning(
                    "shelf: chipboard texture %s not found; using a plain wood colour", candidate
                )
        except ModelError as exc:
            logger.warning(
                "shelf: model %r unresolved (%s); using a plain wood colour for the boards",
                _TEXTURE_MODEL,
                exc,
            )

        if texture_path is None:
            chipboard.rgba = _WOOD_FALLBACK_RGBA
            return "shelf_chipboard", "shelf_frame"

        tex = child.add_texture()
        tex.name = "shelf_chipboard_tex"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.file = texture_path
        chipboard.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "shelf_chipboard_tex"
        chipboard.texuniform = True  # texrepeat is repetitions-per-metre (real-world scale)
        chipboard.texrepeat = [1.0 / _TEXTURE_TILE_M, 1.0 / _TEXTURE_TILE_M]
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
