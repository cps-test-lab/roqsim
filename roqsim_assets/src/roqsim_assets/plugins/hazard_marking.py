"""Scene plugin: a **parametric** floor marking in diagonal warning stripes -- yellow and black.

The striped rectangle that says "this floor is kept for something": a set-down place for a pallet,
a hand-over position between two vehicles, a keep-clear zone before a door. Either the outline of
the rectangle alone (``band``), which is how a set-down place is painted so the load stands on bare
floor inside it, or the whole rectangle (``band: 0``).

It is paint: a couple of millimetres thick and without contact, so a wheel rolls over it without
a bump and nothing rests on it. A camera sees it, and a ray aimed at the floor returns from the
paint as it would from the floor, that thickness nearer.

The stripes are not a texture. Each stripe is cut to the marked area and all stripes of one colour
are one mesh, so the edges are sharp at any distance and the stripes run through the corners of an
outline without a seam.

Geometry (all metres). ``pose`` places the marking: its position is the rectangle's centre on the
floor, its yaw turns the rectangle; ``length`` is along that direction and ``width`` across it.
Paint lies flat, so a pose that tilts it is refused.

Config::

    hazard_marking:
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 marking)
      pose:               # a geometry_msgs/Pose in the world (roqsim.pose):
        position: {x: 0.0, y: 0.0, z: 0.0}   #   the centre, z the floor it is painted on
        orientation: {yaw: 0.0}              #   direction of 'length', rad; no roll or pitch
      length: 1.4         # along yaw, m, over the outer edges
      width: 1.0          # across it, m, over the outer edges
      band: 0.1           # width of the striped outline, m; 0 stripes the whole rectangle
      stripe: 0.1         # width of one stripe, m, measured across the stripe
      angle: 45.0         # the stripes' slant against 'length', degrees, in (0, 180)
      colors:             # the two paints, each [r, g, b] or [r, g, b, a]
        - [0.98, 0.80, 0.05]
        - [0.05, 0.05, 0.05]
      thickness: 0.002    # how far the paint stands above the floor, m
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors, yaw_of

_YELLOW = [0.98, 0.80, 0.05, 1.0]
_BLACK = [0.05, 0.05, 0.05, 1.0]

Point = tuple[float, float]


def _clip(polygon: list[Point], normal: Point, offset: float) -> list[Point]:
    """The part of a convex ``polygon`` where ``normal . p <= offset`` (Sutherland-Hodgman)."""
    out: list[Point] = []
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        da = normal[0] * a[0] + normal[1] * a[1] - offset
        db = normal[0] * b[0] + normal[1] * b[1] - offset
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _area(polygon: list[Point]) -> float:
    return 0.5 * sum(
        a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    )


def marked_rectangles(
    length: float, width: float, band: float
) -> list[tuple[float, float, float, float]]:
    """The painted area as ``(x0, y0, x1, y1)`` rectangles about the marking's centre.

    One rectangle when the whole area is striped; four that meet without overlap when only the
    outline is: the two long sides at full length and the two short sides between them.
    """
    hx, hy = length / 2, width / 2
    if band <= 0 or 2 * band >= min(length, width):
        return [(-hx, -hy, hx, hy)]
    return [
        (-hx, -hy, hx, -hy + band),
        (-hx, hy - band, hx, hy),
        (-hx, -hy + band, -hx + band, hy - band),
        (hx - band, -hy + band, hx, hy - band),
    ]


def stripe_polygons(
    rectangles: list[tuple[float, float, float, float]], stripe: float, angle: float
) -> tuple[list[list[Point]], list[list[Point]]]:
    """Cut ``rectangles`` into stripes; return the polygons of the first and of the second colour.

    A stripe is the set of points whose distance along the stripes' normal lies in one
    ``stripe``-wide interval. The intervals are counted from the marking's centre for every
    rectangle alike, which is what carries a stripe through a corner of an outline.
    """
    normal = (-math.sin(angle), math.cos(angle))
    first: list[list[Point]] = []
    second: list[list[Point]] = []
    for x0, y0, x1, y1 in rectangles:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        reach = [normal[0] * x + normal[1] * y for x, y in corners]
        for k in range(math.floor(min(reach) / stripe), math.ceil(max(reach) / stripe)):
            piece = _clip(corners, normal, (k + 1) * stripe)
            piece = _clip(piece, (-normal[0], -normal[1]), -k * stripe)
            if len(piece) >= 3 and _area(piece) > 1e-10:
                (first if k % 2 == 0 else second).append(piece)
    return first, second


def _prisms(polygons: list[list[Point]], thickness: float) -> tuple[list[float], list[int]]:
    """One closed mesh of every polygon extruded from z = 0 to ``thickness``."""
    verts: list[float] = []
    faces: list[int] = []
    for polygon in polygons:
        n = len(polygon)
        base = len(verts) // 3
        for z in (0.0, thickness):
            for x, y in polygon:
                verts += [x, y, z]
        for i in range(1, n - 1):
            faces += [base + n, base + n + i, base + n + i + 1]  # top, facing up
            faces += [base, base + i + 1, base + i]  # bottom, facing down
        for i in range(n):
            j = (i + 1) % n
            faces += [base + i, base + j, base + n + j]
            faces += [base + i, base + n + j, base + n + i]
    return verts, faces


class HazardMarkingPlugin(Plugin):
    #: Registers an entity, so its label names that entity and a scenario can ask where it is.
    provides_entity = True
    _ROOT_BODY = "hazard_marking"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.pos, quat = config_pose(self.config)
        self.yaw = yaw_of(quat)
        self.length = self._float(self.config.get("length"), 1.4)
        self.width = self._float(self.config.get("width"), 1.0)
        self.band = self._float(self.config.get("band"), 0.1)
        self.stripe = self._float(self.config.get("stripe"), 0.1)
        self.angle = math.radians(self._float(self.config.get("angle"), 45.0))
        self.thickness = self._float(self.config.get("thickness"), 0.002)
        colors = self.config.get("colors")
        if isinstance(colors, (list, tuple)) and len(colors) == 2:
            self.colors = [
                self._rgba(c) or d for c, d in zip(colors, (_YELLOW, _BLACK), strict=True)
            ]
        else:
            self.colors = [_YELLOW, _BLACK]

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
        errors += config_pose_errors(config, "hazard_marking")
        if "yaw" in config:
            errors.append(
                "hazard_marking: 'yaw' is not read -- the rectangle's direction is the yaw of "
                "'pose': pose: {orientation: {yaw: ...}}"
            )
        _, quat = config_pose(config)
        heading = [math.cos(yaw_of(quat) / 2), 0.0, 0.0, math.sin(yaw_of(quat) / 2)]
        if abs(abs(sum(a * b for a, b in zip(quat, heading, strict=True))) - 1.0) > 1e-9:
            errors.append(
                "hazard_marking: 'pose' tilts the marking; paint lies flat on the floor, so its "
                "orientation is a yaw alone"
            )
        for key in ("length", "width", "stripe", "thickness"):
            if key in config:
                try:
                    if float(config[key]) <= 0:
                        errors.append(f"'{key}' must be > 0")
                except (TypeError, ValueError):
                    errors.append(f"'{key}' must be a number > 0")
        if "band" in config:
            try:
                if float(config["band"]) < 0:
                    errors.append("'band' must be >= 0 (0 stripes the whole rectangle)")
            except (TypeError, ValueError):
                errors.append("'band' must be a number >= 0")
        if "angle" in config:
            try:
                if not 0.0 < float(config["angle"]) < 180.0:
                    errors.append("'angle' must be between 0 and 180 degrees, both excluded")
            except (TypeError, ValueError):
                errors.append("'angle' must be a number of degrees between 0 and 180")
        if "colors" in config:
            colors = config["colors"]
            if not (
                isinstance(colors, (list, tuple))
                and len(colors) == 2
                and all(self._rgba(c) is not None for c in colors)
            ):
                errors.append("'colors' must be two colours, each [r, g, b] or [r, g, b, a]")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        rectangles = marked_rectangles(self.length, self.width, self.band)
        for name, rgba, polygons in zip(
            ("first", "second"),
            self.colors,
            stripe_polygons(rectangles, self.stripe, self.angle),
            strict=True,
        ):
            if not polygons:
                continue
            mat = child.add_material()
            mat.name = f"paint_{name}"
            mat.rgba = rgba
            mat.specular = 0.1

            verts, faces = _prisms(polygons, self.thickness)
            mesh = child.add_mesh()
            mesh.name = f"stripes_{name}"
            mesh.uservert = verts
            mesh.userface = faces

            g = body.add_geom()
            g.name = f"stripes_{name}"
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = mesh.name
            g.material = mat.name
            g.contype = 0
            g.conaffinity = 0

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                meta={
                    "prefix": self.prefix,
                    "length": self.length,
                    "width": self.width,
                    "band": self.band,
                },
            )
        )
