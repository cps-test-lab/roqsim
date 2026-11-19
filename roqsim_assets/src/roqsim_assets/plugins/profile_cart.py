"""Scene plugin: a **parametric** aluminium profile cart with a T-slot plate top, on castors.

The kind of mobile cart a robot cell is built on: four profile posts on castors, a frame under the
top, a top of extruded aluminium T-slot plates, a shelf with a pull-out drawer below it and a framed
panel across the front. Like ``workbench`` it is built from primitives at build time and welded in
place (static scenery, no free joint and no castor joints: the cart stands still while a cell runs).

**The slots are collision geometry, not a texture.** The top is a slab with lands between the slot
openings, and under each opening the T's wider head, so something slid across the slots meets the
slot's edges and something small drops into one. With ``slots.direction: y`` (the default) the slots
run front to back and are spaced along the cart's length; ``x`` turns them. The slots are laid out
symmetrically about the centre: ``count`` slots at ``pitch``, by default as many as fit with half a
pitch of land at each edge (``round(length / pitch)``).

Geometry (all metres), origin at the footprint centre on the floor (min z == 0, the wheels), so a pose
of (x y z) drops the cart there. The cart's length runs along X, its depth along Y; the front -- the
drawer and the panel -- faces **-Y**, as ``workbench``'s operator side does.

  - Top: ``top.length`` x ``top.depth`` x ``top.thickness``, its upper face exactly at ``top.height``.
  - Frame: profile posts (``frame.profile`` square) at the footprint's corners, from the castors to
    under the top, a rail round the top's underside and one round the bottom.
  - Castors: one per post, a wheel of ``castors.wheel`` diameter; ``castors.height`` is the floor to
    the bottom frame's underside.
  - Shelf: a board at ``shelf.height`` (its upper face) between the posts.
  - Drawer: an open tray standing on the shelf, ``drawer.height`` high, pulled out by
    ``drawer.open`` toward the front.
  - Front panel: a framed panel across the front between the shelf and the top frame, fixed to the
    frame or, with ``front_panel.carried_by: drawer``, the drawer's front, out with it by ``drawer.open``.

Config::

    profile_cart:
      prefix: ""            # MJCF name prefix (distinct prefixes for >1 cart)
      pose:                 # world placement, a geometry_msgs/Pose (roqsim.pose)
        position: {x: 0.0, y: 0.0, z: 0.0}
        orientation: {yaw: 0.0}
      top: {length: 1.20, depth: 0.80, height: 0.85, thickness: 0.025}
      slots: {pitch: 0.04, width: 0.008, depth: 0.010, head_width: 0.016, neck: 0.004,
              direction: y, count: null}
      footprint: null       # [length, depth] over the posts; default: the top's
      frame: {profile: 0.045}
      castors: {height: 0.125, wheel: 0.10}
      shelf: {enabled: true, height: 0.30}
      drawer: {enabled: true, height: 0.20, open: 0.0}
      front_panel: {enabled: true, carried_by: frame}   # or drawer, the drawer's front
      edge_covers: true     # caps over the slots' open ends, front and back

What collides: the top (slab, lands, the T-heads' walls), posts, frame rails, shelf, drawer and the
wheels. What does not: the edge covers, the front panel and its frame, the castor forks and brakes --
trim a robot in this cell plans no contact against.
"""

from __future__ import annotations

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors
from roqsim.schema import Field

_ALU_RGBA = [0.74, 0.75, 0.77, 1.0]  # anodised profile and plate
_COVER_RGBA = [0.78, 0.79, 0.80, 1.0]  # grey plastic end caps
_DARK_RGBA = [0.12, 0.12, 0.13, 1.0]  # wheels, forks, brakes
_BOARD_RGBA = [0.86, 0.86, 0.84, 1.0]  # shelf board, laminated
_PANEL_RGBA = [0.95, 0.95, 0.94, 1.0]  # the front panel's face

TOP = {
    "length": Field(float, default=1.20, minimum=0.2, unit="m", doc="along the cart's X"),
    "depth": Field(float, default=0.80, minimum=0.2, unit="m", doc="along the cart's Y"),
    "height": Field(
        float, default=0.85, minimum=0.3, unit="m", doc="its upper face above the floor"
    ),
    "thickness": Field(
        float, default=0.025, minimum=0.005, unit="m", doc="the plate, face to underside"
    ),
}
SLOTS = {
    "pitch": Field(float, default=0.04, minimum=0.005, unit="m", doc="slot centre to slot centre"),
    "width": Field(float, default=0.008, minimum=0.001, unit="m", doc="the opening at the surface"),
    "depth": Field(float, default=0.010, minimum=0.001, unit="m", doc="surface to the T's bottom"),
    "head_width": Field(
        float, default=0.016, minimum=0.001, unit="m", doc="the T's inner width under the lips"
    ),
    "neck": Field(float, default=0.004, minimum=0.0005, unit="m", doc="the lips' thickness"),
    "direction": Field(
        str, default="y", choices=("x", "y"), doc="the axis the slots run along; y: front to back"
    ),
    "count": Field(int, minimum=1, doc="how many slots; default: round(across / pitch)"),
}
FRAME = {
    "profile": Field(
        float, default=0.045, minimum=0.01, unit="m", doc="the posts' and rails' section"
    ),
}
CASTORS = {
    "height": Field(
        float, default=0.125, minimum=0.02, unit="m", doc="floor to the bottom frame's underside"
    ),
    "wheel": Field(float, default=0.10, minimum=0.01, unit="m", doc="wheel diameter"),
}
SHELF = {
    "enabled": Field(bool, default=True, doc="a shelf board between the posts"),
    "height": Field(
        float, default=0.30, minimum=0.0, unit="m", doc="its upper face above the floor"
    ),
}
DRAWER = {
    "enabled": Field(bool, default=True, doc="a tray on the shelf (needs the shelf)"),
    "height": Field(float, default=0.20, minimum=0.02, unit="m", doc="the tray's outer height"),
    "open": Field(float, default=0.0, minimum=0.0, unit="m", doc="how far it is pulled out, to -Y"),
}
FRONT_PANEL = {
    "enabled": Field(bool, default=True, doc="a framed panel across the front, under the top"),
    "carried_by": Field(
        str,
        default="frame",
        choices=("frame", "drawer"),
        doc="what carries it: the frame (fixed), or the drawer, as its front (moves with drawer.open)",
    ),
}

_BOARD_T = 0.018
_TRAY_WALL = 0.012


class ProfileCartPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a ``components:`` block.
    provides_entity = True
    _ROOT_BODY = "profile_cart"

    CONFIG_SCHEMA = {
        "prefix": Field(str, default="", doc="MJCF name prefix; distinct per cart"),
        "pose": Field(dict, doc="the footprint centre on the floor: a geometry_msgs/Pose"),
        "top": Field(dict, schema=TOP, doc="the T-slot plate top"),
        "slots": Field(dict, schema=SLOTS, doc="the T-slots in the top, collision geometry"),
        "footprint": Field(
            list, length=2, unit="m", doc="[length, depth] over the posts; default: the top's"
        ),
        "frame": Field(dict, schema=FRAME, doc="posts and rails"),
        "castors": Field(dict, schema=CASTORS, doc="one under each post"),
        "shelf": Field(dict, schema=SHELF, doc="the shelf board"),
        "drawer": Field(dict, schema=DRAWER, doc="the pull-out tray on the shelf"),
        "front_panel": Field(dict, schema=FRONT_PANEL, doc="the framed front panel, visual"),
        "edge_covers": Field(bool, default=True, doc="caps over the slots' open ends, visual"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.pos, self.quat = config_pose(self.config)

    # -- derived geometry ---------------------------------------------------------------------

    @staticmethod
    def _footprint(s) -> tuple[float, float]:
        fp = s.footprint
        if fp is None:
            return s.top.length, s.top.depth
        return float(fp[0]), float(fp[1])

    @staticmethod
    def _slot_layout(s) -> tuple[float, list[float]]:
        """The extent across the slots and the slot centres along it, symmetric about 0."""
        across = s.top.length if s.slots.direction == "y" else s.top.depth
        n = s.slots.count if s.slots.count is not None else max(1, round(across / s.slots.pitch))
        first = -(n - 1) * s.slots.pitch / 2
        return across, [first + i * s.slots.pitch for i in range(n)]

    def slot_centres(self) -> list[float]:
        """The slot centres across the slots, in the cart's frame (X for ``direction: y``)."""
        return self._slot_layout(self.settings)[1]

    def validate_config(self, config: dict) -> list[str]:
        where = f"profile_cart {self.address}"
        errors = config_pose_errors(config, where)
        try:
            s = self.settings_for(config)
            top, slots, frame = s.top, s.slots, s.frame
            castors, shelf, drawer = s.castors, s.shelf, s.drawer
            fl, fd = self._footprint(s)
        except (TypeError, ValueError, AttributeError):
            return errors  # the schema reports the shape
        if not slots.width < slots.head_width < slots.pitch:
            errors.append(f"{where}: slots need width < head_width < pitch")
        if not slots.neck < slots.depth < top.thickness:
            errors.append(f"{where}: slots need neck < depth < the top's thickness")
        across, centres = self._slot_layout(s)
        if centres[-1] + slots.head_width / 2 > across / 2:
            errors.append(
                f"{where}: {len(centres)} slots at {slots.pitch:g} m do not fit in {across:g} m"
            )
        if fl <= 2 * frame.profile or fd <= 2 * frame.profile:
            errors.append(f"{where}: the footprint must be wider than two profiles each way")
        if fl > top.length + 1e-9 or fd > top.depth + 1e-9:
            errors.append(f"{where}: the footprint may not stick out from under the top")
        frame_bottom = castors.height + frame.profile
        under_top = top.height - top.thickness - frame.profile
        if castors.wheel >= castors.height:
            errors.append(f"{where}: castors.wheel must be under castors.height (the fork)")
        if under_top <= frame_bottom:
            errors.append(f"{where}: top.height leaves no room for the frame above the castors")
        if shelf.enabled and not frame_bottom <= shelf.height - _BOARD_T < under_top:
            errors.append(
                f"{where}: shelf.height must lie between the bottom frame ({frame_bottom:g} m) and "
                f"the top frame ({under_top:g} m)"
            )
        if s.front_panel.enabled and s.front_panel.carried_by == "drawer" and not drawer.enabled:
            errors.append(f"{where}: front_panel.carried_by: drawer needs the drawer")
        if drawer.enabled:
            if not shelf.enabled:
                errors.append(f"{where}: the drawer stands on the shelf; enable the shelf")
            elif shelf.height + drawer.height > under_top:
                errors.append(
                    f"{where}: the drawer does not fit between the shelf and the top frame"
                )
            if drawer.open >= fd - 2 * frame.profile:
                errors.append(f"{where}: drawer.open must be under the drawer's own depth")
        return errors

    # -- build --------------------------------------------------------------------------------

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        s = self.settings
        child = mujoco.MjSpec()
        mats = self._materials(child)
        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        self._add_top(body, mats, s)
        self._add_frame(body, mats, s)
        self._add_castors(body, mats, s)
        if s.shelf.enabled:
            self._add_shelf(body, mats, s)
            if s.drawer.enabled:
                self._add_drawer(body, mats, s)
        if s.front_panel.enabled:
            self._add_front_panel(body, mats, s)
        site = body.add_site()
        site.name = "top_surface"
        site.pos = [0.0, 0.0, s.top.height]

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=s.prefix, frame=frame)

    @staticmethod
    def _materials(child: mujoco.MjSpec) -> dict[str, str]:
        names = {}
        for role, rgba, specular, shininess in (
            ("alu", _ALU_RGBA, 0.5, 0.4),
            ("cover", _COVER_RGBA, 0.2, 0.2),
            ("dark", _DARK_RGBA, 0.2, 0.2),
            ("board", _BOARD_RGBA, 0.1, 0.1),
            ("panel", _PANEL_RGBA, 0.05, 0.05),
        ):
            mat = child.add_material()
            mat.name = f"profile_cart_{role}"
            mat.rgba = rgba
            mat.specular = specular
            mat.shininess = shininess
            names[role] = mat.name
        return names

    @staticmethod
    def _box(body, name, half, pos, material, *, collide=True):
        g = body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(half)
        g.pos = list(pos)
        g.material = material
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    def _add_top(self, body, mats, s) -> None:
        """The plate: a slab under the slots, a lip land between each two openings, and under the
        lips the core between the T-heads. Every piece runs the slots' full length."""
        top, slots = s.top, s.slots
        h = top.height
        across, centres = self._slot_layout(s)
        along = top.depth if slots.direction == "y" else top.length

        def place(name, a0, a1, z0, z1):
            # a span [a0, a1] across the slots and [z0, z1] in height, the slots' full length along
            half_a, mid_a = (a1 - a0) / 2, (a1 + a0) / 2
            if half_a <= 1e-9:
                return
            if slots.direction == "y":
                half, pos = [half_a, along / 2, (z1 - z0) / 2], [mid_a, 0.0, (z1 + z0) / 2]
            else:
                half, pos = [along / 2, half_a, (z1 - z0) / 2], [0.0, mid_a, (z1 + z0) / 2]
            self._box(body, name, half, pos, mats["alu"])

        place("top_slab", -across / 2, across / 2, h - top.thickness, h - slots.depth)
        lips = [-across / 2] + [
            x for c in centres for x in (c - slots.width / 2, c + slots.width / 2)
        ]
        lips.append(across / 2)
        cores = [-across / 2] + [
            x for c in centres for x in (c - slots.head_width / 2, c + slots.head_width / 2)
        ]
        cores.append(across / 2)
        for i in range(len(centres) + 1):
            place(f"top_land_{i}", lips[2 * i], lips[2 * i + 1], h - slots.neck, h)
            place(f"top_core_{i}", cores[2 * i], cores[2 * i + 1], h - slots.depth, h - slots.neck)

        if s.edge_covers:
            # Caps over the slots' open ends: the two edges the slots run out at.
            t = 0.004
            cz = h - top.thickness / 2
            if slots.direction == "y":
                for sign, tag in ((-1, "front"), (1, "back")):
                    self._box(
                        body,
                        f"edge_cover_{tag}",
                        [top.length / 2, t / 2, top.thickness / 2],
                        [0.0, sign * (top.depth / 2 + t / 2), cz],
                        mats["cover"],
                        collide=False,
                    )
            else:
                for sign, tag in ((-1, "left"), (1, "right")):
                    self._box(
                        body,
                        f"edge_cover_{tag}",
                        [t / 2, top.depth / 2, top.thickness / 2],
                        [sign * (top.length / 2 + t / 2), 0.0, cz],
                        mats["cover"],
                        collide=False,
                    )

    def _add_frame(self, body, mats, s) -> None:
        fl, fd = self._footprint(s)
        p = s.frame.profile
        z_bottom = s.castors.height
        z_under_top = s.top.height - s.top.thickness
        px, py = fl / 2 - p / 2, fd / 2 - p / 2
        for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            self._box(
                body,
                f"post_{i}",
                [p / 2, p / 2, (z_under_top - z_bottom) / 2],
                [sx * px, sy * py, (z_under_top + z_bottom) / 2],
                mats["alu"],
            )
        for tag, z in (("top", z_under_top - p / 2), ("bottom", z_bottom + p / 2)):
            for sign, side in ((-1, "front"), (1, "back")):
                self._box(
                    body,
                    f"rail_{tag}_{side}",
                    [px - p / 2, p / 2, p / 2],
                    [0.0, sign * py, z],
                    mats["alu"],
                )
            for sign, side in ((-1, "left"), (1, "right")):
                self._box(
                    body,
                    f"rail_{tag}_{side}",
                    [p / 2, py - p / 2, p / 2],
                    [sign * px, 0.0, z],
                    mats["alu"],
                )

    def _add_castors(self, body, mats, s) -> None:
        fl, fd = self._footprint(s)
        p = s.frame.profile
        r = s.castors.wheel / 2
        hc = s.castors.height
        px, py = fl / 2 - p / 2, fd / 2 - p / 2
        for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            x, y = sx * px, sy * py
            w = body.add_geom()
            w.name = f"castor_wheel_{i}"
            w.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            w.size = [r, 0.016, 0.0]
            w.pos = [x, y, r]
            w.quat = [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]  # axis along X
            w.material = mats["dark"]
            # The fork and its plate, from the wheel's axle to the post: trim.
            self._box(
                body,
                f"castor_fork_{i}",
                [0.006, 0.024, (hc - r) / 2],
                [x, y, (hc + r) / 2],
                mats["dark"],
                collide=False,
            )
            self._box(
                body,
                f"castor_plate_{i}",
                [p / 2, p / 2, 0.004],
                [x, y, hc - 0.004],
                mats["alu"],
                collide=False,
            )
            self._box(
                body,
                f"castor_brake_{i}",
                [0.015, 0.02, 0.006],
                [x, y - sy * (r + 0.01), r + 0.02],
                mats["dark"],
                collide=False,
            )

    def _shelf_span(self, s) -> tuple[float, float]:
        fl, fd = self._footprint(s)
        p = s.frame.profile
        return fl / 2 - p, fd / 2 - p

    def _add_shelf(self, body, mats, s) -> None:
        hx, hy = self._shelf_span(s)
        z = s.shelf.height
        self._box(
            body, "shelf", [hx, hy, _BOARD_T / 2], [0.0, 0.0, z - _BOARD_T / 2], mats["board"]
        )

    def _add_drawer(self, body, mats, s) -> None:
        """An open tray on the shelf, its front at the cart's front, ``open`` pulled out to -Y."""
        hx, hy = self._shelf_span(s)
        hx -= 0.01
        hy -= 0.01
        hz = s.drawer.height / 2
        y0 = -s.drawer.open
        z0 = s.shelf.height
        w = _TRAY_WALL
        for name, half, pos in (
            ("drawer_floor", [hx, hy, w / 2], [0.0, y0, z0 + w / 2]),
            ("drawer_front", [hx, w / 2, hz], [0.0, y0 - hy + w / 2, z0 + hz]),
            ("drawer_back", [hx, w / 2, hz], [0.0, y0 + hy - w / 2, z0 + hz]),
            ("drawer_left", [w / 2, hy, hz], [-hx + w / 2, y0, z0 + hz]),
            ("drawer_right", [w / 2, hy, hz], [hx - w / 2, y0, z0 + hz]),
        ):
            self._box(body, name, half, pos, mats["board"])

    def _add_front_panel(self, body, mats, s) -> None:
        """A panel in a profile frame across the front, from the shelf (or the bottom frame) up to
        the top frame. Carried by the frame it stays put; as the drawer's front (``carried_by: drawer``) it
        comes out with the drawer by ``drawer.open``."""
        fl, fd = self._footprint(s)
        p = s.frame.profile
        z0 = s.shelf.height if s.shelf.enabled else s.castors.height + p
        z1 = s.top.height - s.top.thickness - p
        drawer_out = s.drawer.open if s.front_panel.carried_by == "drawer" else 0.0
        y = -fd / 2 - 0.012 - drawer_out
        half_x = fl / 2 - p
        fw = 0.02  # the panel frame's profile
        self._box(
            body,
            "front_panel",
            [half_x - fw, 0.003, (z1 - z0) / 2 - fw],
            [0.0, y, (z0 + z1) / 2],
            mats["panel"],
            collide=False,
        )
        for name, half, pos in (
            ("front_panel_frame_top", [half_x, 0.01, fw / 2], [0.0, y, z1 - fw / 2]),
            ("front_panel_frame_bottom", [half_x, 0.01, fw / 2], [0.0, y, z0 + fw / 2]),
            (
                "front_panel_frame_left",
                [fw / 2, 0.01, (z1 - z0) / 2],
                [-half_x + fw / 2, y, (z0 + z1) / 2],
            ),
            (
                "front_panel_frame_right",
                [fw / 2, 0.01, (z1 - z0) / 2],
                [half_x - fw / 2, y, (z0 + z1) / 2],
            ),
        ):
            self._box(body, name, half, pos, mats["alu"], collide=False)

    def configure(self, ctx: SimContext) -> None:
        s = self.settings
        fl, fd = self._footprint(s)
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=s.prefix + self._ROOT_BODY,
                meta={
                    "prefix": s.prefix,
                    "top": {
                        "length": s.top.length,
                        "depth": s.top.depth,
                        "height": s.top.height,
                    },
                    "slots": {
                        "pitch": s.slots.pitch,
                        "direction": s.slots.direction,
                        "centres": self.slot_centres(),
                    },
                    "footprint": [fl, fd],
                },
            )
        )
