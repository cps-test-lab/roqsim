"""Scene plugin: a **parametric** floor marking -- a line, a rectangle's outline or a filled area, painted
in one colour, solid or dashed, or in yellow-and-black warning stripes.

The paint that lays out a hall: a lane's edge drawn through, a walkway's edge, a bay's outline in
dashes (``solid``, ``dashed``), and the striped rectangle that says "this floor is kept for
something" -- a set-down place for a pallet, a hand-over position between two vehicles, a
keep-clear zone before a door (``hazard``). Either the outline of the rectangle alone (``band``),
which is how a set-down place or a bay is painted so the load stands on bare floor inside it, or the
whole rectangle (``band: 0``), which is how a line is painted: a long, narrow rectangle.

It is paint: half a millimetre thick and without contact, so a wheel rolls over it without a bump
and nothing rests on it. A camera sees it, and a ray aimed at the floor returns from the paint as
it would from the floor, that thickness nearer.

It is drawn as paint on a floor looks, not as a sign: a traffic yellow and an anthracite instead of
pure yellow and black, warning stripes in proportion to the band as marking tape has them, and the
sheen of a trowelled floor, so a lamp's highlight runs across floor and paint alike. The coat
covers: a colour given with an alpha below 1 is paint worn thin, and the floor shows through it.

Neither stripes nor dashes are a texture. Each piece is cut to the marked area and all pieces of one
colour are one mesh, so the edges are sharp at any distance and warning stripes run through the
corners of an outline without a seam. A dashed marking cuts each painted side along its length into
``dash``-long pieces with ``dash`` between them, starting at the side's end.

Geometry (all metres). ``pose`` places the marking: its position is the rectangle's centre on the
floor, its yaw turns the rectangle; ``length`` is along that direction and ``width`` across it.
Paint lies flat, so a pose that tilts it is refused.

``pattern`` defaults to ``solid``. ``color`` belongs to ``solid`` and ``dashed``, ``dash`` to
``dashed``, ``stripe`` / ``angle`` / ``colors`` to ``hazard``; a key of another pattern than the one
the marking has is refused, so a setting is never silently left unread.

Config::

    floor_marking:
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 marking)
      pose:               # a geometry_msgs/Pose in the world (roqsim.pose):
        position: {x: 0.0, y: 0.0, z: 0.0}   #   the centre, z the floor it is painted on
        orientation: {yaw: 0.0}              #   direction of 'length', rad; no roll or pitch
      length: 1.4         # along yaw, m, over the outer edges
      width: 1.0          # across it, m, over the outer edges
      band: 0.05          # width of the painted outline, m; 0 paints the whole rectangle
      pattern: solid      # solid | dashed | hazard (warning stripes)
      color: [0.90, 0.66, 0.06]  # solid, dashed: the paint, [r, g, b] or [r, g, b, a]
      dash: 0.5           # dashed: a dash's length and the gap after it, m
      stripe: 0.035       # hazard: width of one stripe, m, measured across the stripe; by default
                          #   0.7 of 'band', and 0.1 where the whole rectangle is striped
      angle: 45.0         # hazard: the stripes' slant against 'length', degrees, in (0, 180)
      colors:             # hazard: the two paints, each [r, g, b] or [r, g, b, a]
        - [0.90, 0.66, 0.06]
        - [0.13, 0.13, 0.14]
      thickness: 0.0005   # how far the paint stands above the floor, m
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors, yaw_of
from roqsim.schema import INJECTED_KEYS

_YELLOW = [0.90, 0.66, 0.06]
_BLACK = [0.13, 0.13, 0.14]
#: A stripe's width as a share of the band's, as marking tape is printed.
_STRIPE_PER_BAND = 0.7
_FILLED_STRIPE = 0.1

#: The keys each pattern reads beyond the shared ones; a key of another pattern is refused.
PATTERN_KEYS = {
    "solid": frozenset({"color"}),
    "dashed": frozenset({"color", "dash"}),
    "hazard": frozenset({"stripe", "angle", "colors"}),
}
_SHARED_KEYS = frozenset({"prefix", "pose", "length", "width", "band", "pattern", "thickness"})
#: Refused with a reason of their own: 'pose' carries the yaw, and roqsim.pose names 'pos'/'rpy'.
_REFUSED_KEYS = frozenset({"yaw", "pos", "rpy"})

Point = tuple[float, float]
Rect = tuple[float, float, float, float]


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


def _corners(rect: Rect) -> list[Point]:
    x0, y0, x1, y1 = rect
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def marked_rectangles(length: float, width: float, band: float) -> list[Rect]:
    """The painted area as ``(x0, y0, x1, y1)`` rectangles about the marking's centre.

    One rectangle when the whole area is painted; four that meet without overlap when only the
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


def dash_rectangles(rectangles: list[Rect], dash: float) -> list[Rect]:
    """Cut each rectangle along its longer side into ``dash``-long pieces with ``dash`` between.

    The pieces start at the side's lower end; the last one is cut short where the side ends.
    """
    out: list[Rect] = []
    for x0, y0, x1, y1 in rectangles:
        along_x = (x1 - x0) >= (y1 - y0)
        a0, a1 = (x0, x1) if along_x else (y0, y1)
        k = 0
        while a0 + 2 * k * dash < a1 - 1e-9:
            s0, s1 = a0 + 2 * k * dash, min(a0 + (2 * k + 1) * dash, a1)
            out.append((s0, y0, s1, y1) if along_x else (x0, s0, x1, s1))
            k += 1
    return out


def stripe_polygons(
    rectangles: list[Rect], stripe: float, angle: float
) -> tuple[list[list[Point]], list[list[Point]]]:
    """Cut ``rectangles`` into stripes; return the polygons of the first and of the second colour.

    A stripe is the set of points whose distance along the stripes' normal lies in one
    ``stripe``-wide interval. The intervals are counted from the marking's centre for every
    rectangle alike, which is what carries a stripe through a corner of an outline.
    """
    normal = (-math.sin(angle), math.cos(angle))
    first: list[list[Point]] = []
    second: list[list[Point]] = []
    for rect in rectangles:
        corners = _corners(rect)
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


class FloorMarkingPlugin(Plugin):
    #: Registers an entity, so its label names that entity and a scenario can ask where it is.
    provides_entity = True
    _ROOT_BODY = "floor_marking"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.pos, quat = config_pose(self.config)
        self.yaw = yaw_of(quat)
        self.pattern = self.config.get("pattern", "solid")
        self.length = self._float(self.config.get("length"), 1.4)
        self.width = self._float(self.config.get("width"), 1.0)
        self.band = self._float(self.config.get("band"), 0.05)
        self.thickness = self._float(self.config.get("thickness"), 0.0005)
        self.dash = self._float(self.config.get("dash"), 0.5)
        outlined = len(marked_rectangles(self.length, self.width, self.band)) > 1
        self.stripe = self._float(
            self.config.get("stripe"),
            _STRIPE_PER_BAND * self.band if outlined else _FILLED_STRIPE,
        )
        self.angle = math.radians(self._float(self.config.get("angle"), 45.0))
        if self.pattern == "hazard":
            colors = self.config.get("colors")
            if not (isinstance(colors, (list, tuple)) and len(colors) == 2):
                colors = (_YELLOW, _BLACK)
            self.colors = [
                self._rgba(c) or self._rgba(d)
                for c, d in zip(colors, (_YELLOW, _BLACK), strict=True)
            ]
        else:
            self.colors = [self._rgba(self.config.get("color")) or self._rgba(_YELLOW)]

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
        pattern = config.get("pattern", "solid")
        if pattern not in PATTERN_KEYS:
            errors.append(
                f"floor_marking: 'pattern' is one of {', '.join(PATTERN_KEYS)}, not {pattern!r}"
            )
        else:
            for key in sorted(set(config) - _SHARED_KEYS - PATTERN_KEYS[pattern]):
                owners = [p for p, keys in PATTERN_KEYS.items() if key in keys]
                if owners:
                    errors.append(
                        f"floor_marking: '{key}' is read by pattern {' and '.join(owners)} only; "
                        f"this marking's pattern is {pattern}"
                    )
                elif key not in _REFUSED_KEYS and key not in INJECTED_KEYS:
                    errors.append(f"floor_marking: unknown key {key!r}")
        errors += config_pose_errors(config, "floor_marking")
        if "yaw" in config:
            errors.append(
                "floor_marking: 'yaw' is not read -- the rectangle's direction is the yaw of "
                "'pose': pose: {orientation: {yaw: ...}}"
            )
        _, quat = config_pose(config)
        heading = [math.cos(yaw_of(quat) / 2), 0.0, 0.0, math.sin(yaw_of(quat) / 2)]
        if abs(abs(sum(a * b for a, b in zip(quat, heading, strict=True))) - 1.0) > 1e-9:
            errors.append(
                "floor_marking: 'pose' tilts the marking; paint lies flat on the floor, so its "
                "orientation is a yaw alone"
            )
        for key in ("length", "width", "stripe", "dash", "thickness"):
            if key in config:
                try:
                    if float(config[key]) <= 0:
                        errors.append(f"floor_marking: '{key}' must be > 0")
                except (TypeError, ValueError):
                    errors.append(f"floor_marking: '{key}' must be a number > 0")
        if "band" in config:
            try:
                if float(config["band"]) < 0:
                    errors.append(
                        "floor_marking: 'band' must be >= 0 (0 paints the whole rectangle)"
                    )
            except (TypeError, ValueError):
                errors.append("floor_marking: 'band' must be a number >= 0")
        if "angle" in config:
            try:
                if not 0.0 < float(config["angle"]) < 180.0:
                    errors.append(
                        "floor_marking: 'angle' must be between 0 and 180 degrees, both excluded"
                    )
            except (TypeError, ValueError):
                errors.append(
                    "floor_marking: 'angle' must be a number of degrees between 0 and 180"
                )
        if "color" in config and self._rgba(config["color"]) is None:
            errors.append("floor_marking: 'color' must be [r, g, b] or [r, g, b, a]")
        if "colors" in config:
            colors = config["colors"]
            if not (
                isinstance(colors, (list, tuple))
                and len(colors) == 2
                and all(self._rgba(c) is not None for c in colors)
            ):
                errors.append(
                    "floor_marking: 'colors' must be two colours, each [r, g, b] or [r, g, b, a]"
                )
        return errors

    def paint_polygons(self) -> list[list[list[Point]]]:
        """The painted polygons, one list per colour of :attr:`colors`, about the marking's centre."""
        rectangles = marked_rectangles(self.length, self.width, self.band)
        if self.pattern == "hazard":
            return list(stripe_polygons(rectangles, self.stripe, self.angle))
        if self.pattern == "dashed":
            rectangles = dash_rectangles(rectangles, self.dash)
        return [[_corners(r) for r in rectangles]]

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        for i, (rgba, polygons) in enumerate(zip(self.colors, self.paint_polygons(), strict=True)):
            if not polygons:
                continue
            mat = child.add_material()
            mat.name = f"paint_{i}"
            mat.rgba = rgba
            # A trowelled floor's sheen, so a lamp's highlight runs across floor and paint alike.
            mat.specular = 0.3
            mat.shininess = 0.6

            verts, faces = _prisms(polygons, self.thickness)
            mesh = child.add_mesh()
            mesh.name = f"paint_{i}"
            mesh.uservert = verts
            mesh.userface = faces

            g = body.add_geom()
            g.name = f"paint_{i}"
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
                    "pattern": self.pattern,
                    "length": self.length,
                    "width": self.width,
                    "band": self.band,
                },
            )
        )
