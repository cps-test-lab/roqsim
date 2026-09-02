"""Scene plugin: a **parametric** fixed window -- a glazed pane in a slim frame, built from boxes.

The counterpart to the ``door`` plugin for openings that are glazed rather than hung: a floorplan
opening with ``leaf: false`` in the generator's doors-map is cut (with its lintel) but gets no leaf, and
this plugin fills it. Parametric for the same reason ``shelf`` is -- a wall's openings are rarely all
one size, and a per-width MJCF model per wall is exactly the frozen geometry the plugin pattern exists
to avoid. Like ``spawn_model`` the window is welded in place (static scenery, no free joint).

Geometry (all metres). Placed by the **opening centre** (``pos``, like ``door``) and the wall direction
(``rpy`` yaw): the pane spans ``width`` along the wall, ``height`` up from the floor, and ``depth``
through it. Two stiles run the full height at the sides, a head and a sill span between them, and the
glass fills what is left inside a uniform ``frame`` border.

Defaults are sized to sit **beside a door** from this library: ``height: 2.06`` is the outer height of
the ``door_frame`` casing (a 2 m opening plus its 6 cm head trim), so a window and a door on the same
wall top out level, and ``depth: 0.1`` is the floorplan generator's default wall thickness, so the frame
fills the reveal. To butt one against a door, place it ``width / 2 + 0.518`` m from the door's opening
centre (0.518 m is ``door_frame``'s outer half-width, so its jamb lands flush against the stile).

Openings are cut from the **floor** up, so this is a floor-to-head unit: there is no sill height to
configure, and asking for one means changing how the generator cuts openings, not this plugin.

Config::

    window:
      name: window        # the entry's OWN key, not the config's: names the entity (default 'window')
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 window)
      pos: [0.0, 0.0, 0.0]  # opening CENTRE, [x, y] or [x, y, z] world placement
      rpy: [0.0, 0.0, 0.0]  # orientation; yaw aligns the pane with its wall
      width: 0.94         # opening width along the wall, m (default 0.94)
      height: 2.06        # overall height from the floor, m (default 2.06 -- door_frame's outer height)
      depth: 0.10         # thickness through the wall, m (default 0.10 -- the default wall thickness)
      frame: 0.05         # frame border width, m (default 0.05); glass fills the rest
      glass: 0.012        # glass thickness, m (default 0.012)
      color: [r,g,b,a]    # frame colour (default a neutral grey); alpha optional
      glass_color: [r,g,b,a]  # glass colour, alpha < 1 to see through it

Both parts collide: the wall around the opening stops a robot, and the glass must too, or a robot would
drive through the window.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

# Defaults that make a window line up with this library's door unit -- see the module docstring.
_DOOR_MATCHED_H = 2.06
_WALL_THICKNESS = 0.10
_FRAME_RGBA = [0.62, 0.62, 0.64, 1.0]
_GLASS_RGBA = [0.60, 0.75, 0.85, 0.35]


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


class WindowPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "window"

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
        self.quat = (
            _rpy_to_quat(*(float(v) for v in rpy)) if len(rpy) == 3 else [1.0, 0.0, 0.0, 0.0]
        )
        # Geometry. Bad values are tolerated here (kept as the default) so validate_config reports them
        # with a friendly message rather than crashing construction.
        self.width = self._float(self.config.get("width"), 0.94)
        self.height = self._float(self.config.get("height"), _DOOR_MATCHED_H)
        self.depth = self._float(self.config.get("depth"), _WALL_THICKNESS)
        self.frame = self._float(self.config.get("frame"), 0.05)
        self.glass = self._float(self.config.get("glass"), 0.012)
        self.color = self._rgba(self.config.get("color")) or _FRAME_RGBA
        self.glass_color = self._rgba(self.config.get("glass_color")) or _GLASS_RGBA

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _rgba(value) -> list[float] | None:
        """An ``[r, g, b]`` / ``[r, g, b, a]`` config colour as rgba; ``None`` if unset or malformed."""
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
        for key in ("width", "height", "depth", "frame", "glass"):
            if key not in config:
                continue
            try:
                if float(config[key]) <= 0:
                    errors.append(f"'{key}' must be > 0")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number > 0")
        # The frame border is taken off both sides of both axes, so anything at or past half the
        # smaller extent leaves no glass -- a frame filling its own opening is a config error, not a
        # window.
        border = self._float(config.get("frame", self.frame), self.frame)
        for key in ("width", "height"):
            extent = self._float(config.get(key, getattr(self, key)), getattr(self, key))
            if border > 0 and extent > 0 and 2 * border >= extent:
                errors.append(f"'frame' ({border:g} m) leaves no glass in a {extent:g} m '{key}'")
        for key in ("color", "glass_color"):
            if key in config and self._rgba(config[key]) is None:
                errors.append(f"'{key}' must be [r, g, b] or [r, g, b, a] numbers")
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        frame_mat = child.add_material()
        frame_mat.name = "window_frame"
        frame_mat.rgba = self.color
        frame_mat.specular = 0.4
        frame_mat.shininess = 0.4

        glass_mat = child.add_material()
        glass_mat.name = "window_glass"
        glass_mat.rgba = self.glass_color
        glass_mat.specular = 0.8
        glass_mat.shininess = 0.9

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        b, d = self.frame, self.depth / 2
        inner_w = self.width - 2 * b  # glass span across the opening
        inner_h = self.height - 2 * b

        # Two stiles the full height at the sides, then head and sill spanning between them, so the
        # border is uniform and the corners are not doubled up.
        for name, x in (("stile_n", -(self.width - b) / 2), ("stile_p", (self.width - b) / 2)):
            g = body.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [b / 2, d, self.height / 2]
            g.pos = [x, 0.0, self.height / 2]
            g.material = frame_mat.name
        for name, z in (("sill", b / 2), ("head", self.height - b / 2)):
            g = body.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [inner_w / 2, d, b / 2]
            g.pos = [0.0, 0.0, z]
            g.material = frame_mat.name

        g = body.add_geom()
        g.name = "glass"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [inner_w / 2, self.glass / 2, inner_h / 2]
        g.pos = [0.0, 0.0, self.height / 2]
        g.material = glass_mat.name

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                meta={"prefix": self.prefix, "width": self.width, "height": self.height},
            )
        )
