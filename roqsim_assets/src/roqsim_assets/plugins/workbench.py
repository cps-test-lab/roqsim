"""Scene plugin: a **parametric** ESD assembly workbench with an electric lift column.

Modelled after a MiniTec "Tisch Elektrisch 300" workstation (datasheet DE118552_1): a 2.00 x 0.70 m
melamine worktop on two telescopic lift columns, a drawer cabinet slung under the top, a perforated
tool panel and a monitor arm on two uprights, and an overhead light frame cantilevered over the bench.
Like ``shelf`` / ``window`` it is built from primitive boxes at build time and welded in place (static
scenery, no free joint).

It is parametric for one reason above the others: the real bench is **height-adjustable**, and the
worktop height is the parameter a workstation experiment actually varies (reach, occlusion, whether a
mobile base can drive under the top). ``height`` is therefore config, restricted to the column's real
stroke -- 0.695 m (lowest) to 0.995 m (highest) -- so the two catalogue settings are ``height: 0.695``
and ``height: 0.995``, and anything between them is a valid column position. A campaign sweeps it with
an ordinary ``ParameterVariationList`` against ``plugins.workbench.height``; no variation plugin needed.

Geometry (all metres), origin at the base footprint centre on the floor (min z == 0), so a pose of
(x y z) drops the bench exactly there. The operator stands on **-Y** facing +Y: the tool panel and
uprights are at the back (+Y), the drawer fronts and the power strip face the operator, and "left" is
-X (the operator's left).

  - Worktop: ``width`` x ``depth``, 25 mm melamine, top surface exactly at ``height``.
  - Frame: two adjustable-foot base rails inset 0.15 m from the ends, a longitudinal brace, one
    two-stage lift column per side (the upper stage is what grows with ``height``), and the top frame
    rails the plate sits on.
  - Cabinet: 0.50 x 0.60 x 0.36 m, three drawers (50/100/150 mm), flush under the worktop.
  - Uprights: 45 x 45 profiles standing on the base rails, up to a 2.00 m overhead frame carrying a
    light bar. These are **floor-referenced** -- they do not move with the worktop.
  - Tool panel (0.456 m high) and monitor mount are **worktop-referenced**: 0.20 m and 0.40 m above the
    top, the catalogue's "Höhe der Position". They bolt into the uprights' profile slot, so the chosen
    worktop height decides where on the fixed uprights they sit -- which keeps the reach ergonomics of
    the real bench at either column setting instead of pinning them to one.

Config::

    workbench:
      name: workbench       # entity name (default 'workbench')
      prefix: ""            # MJCF name prefix (distinct prefixes for >1 bench)
      pos: [0.0, 0.0, 0.0]  # [x, y] or [x, y, z] world placement
      rpy: [0.0, 0.0, 0.0]  # orientation as roll/pitch/yaw (rad)
      height: 0.695         # worktop height, m -- the lift column's stroke, 0.695 .. 0.995
      width: 2.00           # worktop size along X, m (default 2.00)
      depth: 0.70           # worktop size along Y, m (default 0.70)
      cabinet: left         # drawer cabinet side: 'left' | 'right' | 'none'
      superstructure: true  # uprights + tool panel + monitor arm + overhead light frame

The structure collides -- worktop, frame, columns, cabinet, uprights, tool panel and the overhead beams
are all things a robot or an arm can hit. The trim does not: adjustable feet pads, drawer fronts and
handles, the power strip, the monitor arm and the light bar are visual, so a bench in a navigation
scene does not cost contacts for decoration.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

# The electric column's real stroke (datasheet: 695 mm nominal, 695-995 mm adjustable). Outside it the
# bench is no longer this product, so it is refused rather than silently clamped.
_HEIGHT_MIN = 0.695
_HEIGHT_MAX = 0.995

_WIDTH = 2.00
_DEPTH = 0.70
_TOP_T = 0.025  # melamine worktop, 25 mm
_FOOT_H = 0.065  # articulated foot (Gelenkfuß M10 D80), 65 mm
_PROFILE = 0.045  # 45 x 45 aluminium profile
_RAIL_W = 0.090  # 45 x 90 profile, flat side across X
_FRAME_INSET = 0.15  # base rail / column centre, in from the worktop end
_COL_W = 0.150  # lift column cross-section across X
_COL_D = 0.300  # ... and along Y (the "300" of the dimension drawing)
_COL_STAGE1_TOP = 0.55  # fixed lower stage; the upper stage telescopes to the top frame
_CAB_W = 0.50
_CAB_D = 0.60
_CAB_H = 0.36
_DRAWERS = (0.15, 0.10, 0.05)  # bottom-up: 150 / 100 / 50 mm fronts
_DRAWER_GAP = 0.015
_PANEL_H = 0.456  # perforated tool panel (Werkzeughalterplatte 2000)
_PANEL_T = 0.010
_PANEL_ABOVE_TOP = 0.20  # "Höhe der Position: 200 mm", measured from the worktop
_MONITOR_ABOVE_TOP = 0.40  # "Höhe der Position: 400 mm"
_MONITOR_REACH = 0.35  # arm extension over the bench (catalogue reach 640 mm, arm folded)
_UPRIGHT_TOP = 2.00  # overhead frame's top face above the floor
_OVERHEAD_REACH = 0.60  # cantilever of the light frame forward over the bench
_STRIP_L = 0.55  # 8-way power strip under the front rail

_ALU_RGBA = [0.72, 0.73, 0.75, 1.0]  # profile, anodised E6/EV1
# Melamine worktop, mid grey (~RAL 7037 Staubgrau). Deliberately DARKER than the anodised alu frame
# and the steel cabinet: on the real bench the ESD top is the one grey surface that reads as a
# worktop rather than as structure, and at RAL 7035 it rendered indistinguishable from white.
_TOP_RGBA = [0.55, 0.56, 0.56, 1.0]
_STEEL_RGBA = [0.80, 0.81, 0.81, 1.0]  # cabinet, light grey RAL 7035
_PANEL_RGBA = [0.70, 0.71, 0.72, 1.0]  # powder-coated steel, grey
_DARK_RGBA = [0.25, 0.25, 0.27, 1.0]  # handles, foot pads, monitor arm
_LIGHT_RGBA = [0.95, 0.95, 0.88, 1.0]  # the luminaire's diffuser

_CABINET_SIDES = ("left", "right", "none")


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


class WorkbenchPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "workbench"

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
        self.height = self._float(self.config.get("height"), _HEIGHT_MIN)
        self.width = self._float(self.config.get("width"), _WIDTH)
        self.depth = self._float(self.config.get("depth"), _DEPTH)
        self.cabinet = self._side(self.config.get("cabinet"))
        self.superstructure = bool(self.config.get("superstructure", True))

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _side(value) -> str:
        """The cabinet side as one of ``_CABINET_SIDES``; anything else stays for validate_config."""
        if value is None:
            return "left"
        if value is False:
            return "none"  # `cabinet: false` reads naturally in YAML for "no cabinet"
        return str(value).lower() if str(value).lower() in _CABINET_SIDES else "left"

    # -- derived geometry ---------------------------------------------------------------------
    # Everything the build reads, in one place, so a changed width/depth moves the whole bench
    # consistently instead of only the parts that happen to reference it.

    @property
    def _frame_x(self) -> float:
        """Column / base-rail centre, inset from the worktop end."""
        return self.width / 2 - _FRAME_INSET

    @property
    def _upright_y(self) -> float:
        """Upright centre: immediately behind the worktop's rear edge, so the two never intersect."""
        return self.depth / 2 + _PROFILE / 2

    @property
    def _base_top(self) -> float:
        return _FOOT_H + _PROFILE

    @property
    def _cab_depth(self) -> float:
        return min(_CAB_D, self.depth - 0.10)

    @property
    def _cab_x(self) -> float:
        """Cabinet centre, tucked against the inner face of its side's column."""
        offset = self._frame_x - _COL_W / 2 - _CAB_W / 2
        return -offset if self.cabinet == "left" else offset

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if "height" in config:
            try:
                h = float(config["height"])
            except (TypeError, ValueError):
                errors.append("'height' must be a number (m)")
            else:
                if not _HEIGHT_MIN - 1e-9 <= h <= _HEIGHT_MAX + 1e-9:
                    errors.append(
                        f"'height' {h:g} m is outside the lift column's stroke "
                        f"{_HEIGHT_MIN:g}..{_HEIGHT_MAX:g} m"
                    )
        for key in ("width", "depth"):
            if key not in config:
                continue
            try:
                if float(config[key]) <= 0:
                    errors.append(f"'{key}' must be > 0")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number > 0")
        width = self._float(config.get("width", self.width), self.width)
        depth = self._float(config.get("depth", self.depth), self.depth)
        # The frame is built inward from the ends, so a too-small top leaves no room for the columns
        # (and, with a cabinet, none for the cabinet inside them).
        min_width = 2 * (_FRAME_INSET + _COL_W / 2)
        if 0 < width < min_width:
            errors.append(f"'width' must be >= {min_width:g} m to fit the lift columns")
        if 0 < depth < _COL_D + 0.10:
            errors.append(f"'depth' must be >= {_COL_D + 0.10:g} m to fit the lift columns")
        side = config.get("cabinet", self.cabinet)
        side = "none" if side is False else str(side).lower()
        if side not in _CABINET_SIDES:
            errors.append("'cabinet' must be 'left', 'right' or 'none'")
        elif side != "none":
            min_cab_width = 2 * (_FRAME_INSET + _COL_W / 2 + _CAB_W)
            if 0 < width < min_cab_width:
                errors.append(
                    f"'width' must be >= {min_cab_width:g} m for a drawer cabinet "
                    f"(or set cabinet: none)"
                )
            if 0 < depth < _CAB_D / 2 + 0.10:
                errors.append(f"'depth' must be >= {_CAB_D / 2 + 0.10:g} m for a drawer cabinet")
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        mats = self._materials(child)

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        self._add_base(body, mats)
        self._add_columns(body, mats)
        self._add_top(body, mats)
        if self.cabinet != "none":
            self._add_cabinet(body, mats)
        self._add_power_strip(body, mats)
        if self.superstructure:
            self._add_uprights(body, mats)
            self._add_tool_panel(body, mats)
            self._add_monitor_arm(body, mats)
            self._add_overhead(body, mats)

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    # -- geometry helpers ---------------------------------------------------------------------

    @staticmethod
    def _materials(child: mujoco.MjSpec) -> dict[str, str]:
        """Add the bench's flat materials to the child spec; return name-by-role."""
        specs = (
            ("alu", _ALU_RGBA, 0.6, 0.5, 0.0),
            ("top", _TOP_RGBA, 0.2, 0.15, 0.0),
            ("steel", _STEEL_RGBA, 0.3, 0.2, 0.0),
            ("panel", _PANEL_RGBA, 0.3, 0.25, 0.0),
            ("dark", _DARK_RGBA, 0.4, 0.3, 0.0),
            ("light", _LIGHT_RGBA, 0.1, 0.1, 0.6),
        )
        names = {}
        for role, rgba, specular, shininess, emission in specs:
            mat = child.add_material()
            mat.name = f"workbench_{role}"
            mat.rgba = rgba
            mat.specular = specular
            mat.shininess = shininess
            mat.emission = emission
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

    @staticmethod
    def _cylinder(body, name, radius, half_h, pos, material, *, collide=True):
        g = body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [radius, half_h, 0.0]
        g.pos = list(pos)
        g.material = material
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    # -- parts --------------------------------------------------------------------------------

    def _add_base(self, body, mats) -> None:
        """Two foot rails on articulated feet, plus the longitudinal brace between them.

        The rails run the full worktop depth *plus* the upright profile, so the superstructure stands
        on them rather than needing a second footprint behind the bench.
        """
        rail_y = _PROFILE / 2
        rail_half_y = (self.depth + _PROFILE) / 2
        z = self._base_top - _PROFILE / 2
        for sign, side in ((-1, "l"), (1, "r")):
            x = sign * self._frame_x
            self._box(
                body,
                f"base_rail_{side}",
                [_RAIL_W / 2, rail_half_y, _PROFILE / 2],
                [x, rail_y, z],
                mats["alu"],
            )
            for tag, y in (
                ("f", -self.depth / 2 + 0.06),
                ("b", self.depth / 2 + _PROFILE - 0.06),
            ):
                self._cylinder(
                    body, f"foot_pad_{side}{tag}", 0.04, 0.006, [x, y, 0.006], mats["dark"]
                )
                self._cylinder(
                    body,
                    f"foot_spindle_{side}{tag}",
                    0.008,
                    (_FOOT_H - 0.012) / 2,
                    [x, y, 0.012 + (_FOOT_H - 0.012) / 2],
                    mats["alu"],
                    collide=False,
                )
        self._box(
            body,
            "base_brace",
            [self._frame_x - _RAIL_W / 2, _PROFILE / 2, _PROFILE / 2],
            [0.0, 0.0, z],
            mats["alu"],
        )

    def _add_columns(self, body, mats) -> None:
        """One two-stage lift column per side. The upper stage carries the whole height adjustment --
        it is the part that telescopes, so a taller bench is a longer upper stage, not a floating one.
        """
        col_d = min(_COL_D, self.depth - 0.10)
        upper_top = self.height - _TOP_T - _PROFILE  # underside of the top frame rails
        # Stages overlap by 0.05 m so the column never shows a seam, whatever the extension.
        upper_bottom = min(_COL_STAGE1_TOP - 0.05, upper_top - 0.02)
        for sign, side in ((-1, "l"), (1, "r")):
            x = sign * self._frame_x
            self._box(
                body,
                f"column_lower_{side}",
                [_COL_W / 2, col_d / 2, (_COL_STAGE1_TOP - self._base_top) / 2],
                [x, 0.0, (_COL_STAGE1_TOP + self._base_top) / 2],
                mats["alu"],
            )
            self._box(
                body,
                f"column_upper_{side}",
                [_COL_W / 2 - 0.013, col_d / 2 - 0.02, (upper_top - upper_bottom) / 2],
                [x, 0.0, (upper_top + upper_bottom) / 2],
                mats["alu"],
            )

    def _add_top(self, body, mats) -> None:
        """The worktop plate and the frame rails it rests on. The plate's TOP face is at ``height``."""
        rail_z = self.height - _TOP_T - _PROFILE / 2
        for sign, side in ((-1, "l"), (1, "r")):
            self._box(
                body,
                f"top_rail_{side}",
                [_RAIL_W / 2, self.depth / 2, _PROFILE / 2],
                [sign * self._frame_x, 0.0, rail_z],
                mats["alu"],
            )
        for sign, side in ((-1, "front"), (1, "back")):
            self._box(
                body,
                f"apron_{side}",
                [self._frame_x, _PROFILE / 2, _PROFILE / 2],
                [0.0, sign * (self.depth / 2 - 0.05), rail_z],
                mats["alu"],
            )
        self._box(
            body,
            "worktop",
            [self.width / 2, self.depth / 2, _TOP_T / 2],
            [0.0, 0.0, self.height - _TOP_T / 2],
            mats["top"],
        )

    def _add_cabinet(self, body, mats) -> None:
        """Drawer cabinet flush under the worktop, fronts (50/100/150 mm) facing the operator (-Y)."""
        depth = self._cab_depth
        top = self.height - _TOP_T
        self._box(
            body,
            "cabinet",
            [_CAB_W / 2, depth / 2, _CAB_H / 2],
            [self._cab_x, 0.0, top - _CAB_H / 2],
            mats["steel"],
        )
        face_y = -depth / 2 - 0.005
        z = top - _CAB_H + _DRAWER_GAP
        for i, front_h in enumerate(_DRAWERS):
            self._box(
                body,
                f"drawer_{i}",
                [_CAB_W / 2 - 0.01, 0.005, front_h / 2],
                [self._cab_x, face_y, z + front_h / 2],
                mats["steel"],
                collide=False,
            )
            self._box(
                body,
                f"drawer_handle_{i}",
                [_CAB_W / 2 - 0.03, 0.008, 0.012],
                [self._cab_x, face_y - 0.008, z + front_h - 0.02],
                mats["dark"],
                collide=False,
            )
            z += front_h + _DRAWER_GAP

    def _add_power_strip(self, body, mats) -> None:
        """8-way strip under the front apron, on the side the cabinet does not occupy."""
        x = -self._cab_x if self.cabinet != "none" else self.width / 4
        self._box(
            body,
            "power_strip",
            [_STRIP_L / 2, 0.025, 0.0225],
            [x, -self.depth / 2 + 0.05, self.height - _TOP_T - _PROFILE - 0.0225],
            mats["alu"],
            collide=False,
        )

    def _add_uprights(self, body, mats) -> None:
        """The two profile uprights, standing on the base rails behind the worktop. Floor-referenced:
        they do not move with the worktop, the accessories bolted to them do."""
        for sign, side in ((-1, "l"), (1, "r")):
            self._box(
                body,
                f"upright_{side}",
                [_PROFILE / 2, _PROFILE / 2, (_UPRIGHT_TOP - self._base_top) / 2],
                [sign * self._frame_x, self._upright_y, (_UPRIGHT_TOP + self._base_top) / 2],
                mats["alu"],
            )

    def _add_tool_panel(self, body, mats) -> None:
        """Perforated tool panel on the operator side of the uprights, ``_PANEL_ABOVE_TOP`` above the
        worktop (the catalogue's mounting position, hence relative to the selected height)."""
        self._box(
            body,
            "tool_panel",
            [self.width / 2 - 0.012, _PANEL_T / 2, _PANEL_H / 2],
            [
                0.0,
                self.depth / 2 - _PANEL_T / 2,
                self.height + _PANEL_ABOVE_TOP + _PANEL_H / 2,
            ],
            mats["panel"],
        )

    def _add_monitor_arm(self, body, mats) -> None:
        """Monitor arm on the right upright: wall plate, two segments, VESA plate. Visual only -- the
        real arm is a compliant, hand-positioned linkage, not something to plan contacts against."""
        z = self.height + _MONITOR_ABOVE_TOP
        x = self._frame_x
        y0 = self.depth / 2 - 0.01
        self._box(
            body, "monitor_plate", [0.03, 0.01, 0.06], [x, y0, z], mats["dark"], collide=False
        )
        self._box(
            body,
            "monitor_arm_1",
            [0.02, _MONITOR_REACH / 2, 0.02],
            [x, y0 - _MONITOR_REACH / 2, z],
            mats["dark"],
            collide=False,
        )
        self._box(
            body,
            "monitor_arm_2",
            [0.14, 0.02, 0.02],
            [x - 0.14, y0 - _MONITOR_REACH, z],
            mats["dark"],
            collide=False,
        )
        self._box(
            body,
            "monitor_mount",
            [0.05, 0.01, 0.05],
            [x - 0.28, y0 - _MONITOR_REACH, z],
            mats["dark"],
            collide=False,
        )

    def _add_overhead(self, body, mats) -> None:
        """Cantilevered light frame at the top of the uprights: a rear beam, two side arms reaching
        forward over the bench, a front beam, and the luminaire slung under them."""
        reach = min(_OVERHEAD_REACH, self.depth - 0.05)
        z = _UPRIGHT_TOP - _PROFILE / 2
        y_back = self._upright_y
        y_front = y_back - reach
        for name, y in (("overhead_back", y_back), ("overhead_front", y_front)):
            self._box(
                body,
                name,
                [self._frame_x + _PROFILE / 2, _PROFILE / 2, _PROFILE / 2],
                [0.0, y, z],
                mats["alu"],
            )
        for sign, side in ((-1, "l"), (1, "r")):
            self._box(
                body,
                f"overhead_arm_{side}",
                [_PROFILE / 2, reach / 2, _PROFILE / 2],
                [sign * self._frame_x, y_back - reach / 2, z],
                mats["alu"],
            )
        self._box(
            body,
            "luminaire",
            [self._frame_x, 0.05, 0.03],
            [0.0, y_back - reach / 2, z - _PROFILE / 2 - 0.03],
            mats["light"],
            collide=False,
        )

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                meta={
                    "prefix": self.prefix,
                    # The selected column position is what a workstation experiment varies, so it
                    # belongs in the entity record a run's ground truth carries.
                    "height": self.height,
                    "width": self.width,
                    "depth": self.depth,
                },
            )
        )
