"""Scene plugin: a **parametric** artificial palm-like tree -- trunk, arching fronds, fruit bunches.

The prop class the asset library was missing entirely: foliage. It is not a decorative plant. A palm's
crown is a *thin, radially arranged, self-occluding* obstacle set, which is a different planning
problem from the boxes and shelves the other assets build -- narrow passages between fronds, a target
tucked under the crown, and a target that a depth sensor sees only partially. That is exactly what
makes palm-like trees a distinct motion-planning case (a real oil palm's morphology is why the
harvesting literature treats it separately), and it is what this prop exists to reproduce.

"Artificial" is literal: this models the *lab prop* -- a moulded plastic palm on a straight pole with
fake fruit bunches wired to it -- not a botanical oil palm. So the trunk is a plain cylinder, the
fronds are flat blades, and the bunches are sphere clusters at poses the world states.

Geometry (all metres), origin at the trunk axis on the floor (min z == 0), so a pose of (x y z) stands
the tree exactly there. Everything is built from primitives at build time and welded in place (static
scenery, no free joint), like ``shelf`` / ``workbench``.

  - Pot: a short cylinder the prop stands in, ``pot_height`` tall.
  - Trunk: one cylinder of ``trunk_radius`` from the pot rim to ``trunk_height``.
  - Crown: ``fronds`` blades spaced evenly in yaw around the trunk at ``crown_height``, each two
    segments -- an inner one rising at ``frond_pitch_deg``, an outer one arching over by a further
    ``frond_droop_deg``. Two segments rather than one because a straight blade leaves either no
    passage under the crown (shallow) or none through it (steep); the arch is what creates the gap a
    planner has to find. Alternate fronds are offset in height by ``frond_tier`` so the crown is two
    staggered tiers, as a real crown is, instead of a single flat disc.
  - Fruit bunches: for each entry in ``bunches``, a cluster of ``fruits`` spheres packed around
    ``pos`` (relative to the trunk base). Each bunch also gets a **site** at its centre, named
    ``<prefix>bunch_<i>``, so a task can read where the target is instead of repeating the number.

Config::

    palm_tree:
      name: palm            # entity name (default 'palm')
      prefix: ""            # MJCF name prefix (distinct prefixes for >1 tree)
      pos: [0.0, 0.0, 0.0]  # [x, y] or [x, y, z] world placement of the trunk base
      rpy: [0.0, 0.0, 0.0]  # orientation as roll/pitch/yaw (rad)
      trunk_height: 1.60    # m, top of the pole above the floor
      trunk_radius: 0.030   # m
      pot_height: 0.16      # m (0 for a pole with no pot)
      pot_radius: 0.085     # m
      crown_height: 0.95    # m, where the fronds spring from the trunk
      fronds: 8             # blades in the crown
      frond_length: 0.45    # m, tip to trunk (split over the two segments)
      frond_width: 0.10     # m
      frond_pitch_deg: 35   # inner segment's rise above horizontal
      frond_droop_deg: 45   # extra downward turn of the outer segment
      frond_tier: 0.10      # m, height offset between alternating fronds
      bunches:              # fruit bunches; pos is relative to the trunk base
        - pos: [-0.13, 0.02, 0.88]
          radius: 0.075
          fruits: 9

The trunk, the fronds and the fruit all **collide** -- every one of them is something an arm can hit,
and the fronds in particular are the obstacle the experiment is about. The pot collides too (it is a
solid object at floor level). Nothing here is decoration, so nothing here is contact-free.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_TRUNK_H = 1.60
_TRUNK_R = 0.030
_POT_H = 0.16
_POT_R = 0.085
_CROWN_H = 0.95
_FRONDS = 8
_FROND_L = 0.45
_FROND_W = 0.10
_FROND_T = 0.006  # blade thickness; a moulded plastic frond is a few mm
_PITCH_DEG = 35.0
_DROOP_DEG = 45.0
_FROND_TIER = 0.10
_FRUIT_R = 0.026  # a single oil-palm fruitlet on the prop
_BUNCH_R = 0.075

_TRUNK_RGBA = [0.42, 0.31, 0.20, 1.0]  # brown pole
_POT_RGBA = [0.12, 0.12, 0.13, 1.0]  # black plastic pot
_FROND_RGBA = [0.16, 0.45, 0.16, 1.0]  # plastic palm green
_FRUIT_RGBA = [0.30, 0.12, 0.06, 1.0]  # ripe bunch, dark red-brown


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


def _euler_zy_quat(yaw: float, pitch: float) -> list[float]:
    """Quaternion for a blade rotated by ``yaw`` about z then pitched by ``pitch`` about its own y.

    Pitch is applied in the rotated frame (intrinsic), which is what "this frond points outward along
    its own axis and rises" means; composing the two extrinsically would tilt the crown sideways.
    """
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # q = q_z(yaw) * q_y(pitch)
    return [cy * cp, -sy * sp, cy * sp, sy * cp]


class PalmTreePlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "palm_tree"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        pos = self.config.get("pos", [0.0, 0.0, 0.0])
        if len(pos) in (2, 3):
            self.pos = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        else:
            self.pos = [0.0, 0.0, 0.0]
        rpy = self.config.get("rpy", [0.0, 0.0, 0.0])
        self.quat = (
            _rpy_to_quat(*(float(v) for v in rpy)) if len(rpy) == 3 else [1.0, 0.0, 0.0, 0.0]
        )
        # Bad values are tolerated here (kept as the default) so validate_config reports them with a
        # friendly message rather than crashing construction.
        self.trunk_height = self._float(self.config.get("trunk_height"), _TRUNK_H)
        self.trunk_radius = self._float(self.config.get("trunk_radius"), _TRUNK_R)
        self.pot_height = self._float(self.config.get("pot_height"), _POT_H)
        self.pot_radius = self._float(self.config.get("pot_radius"), _POT_R)
        self.crown_height = self._float(self.config.get("crown_height"), _CROWN_H)
        self.fronds = self._int(self.config.get("fronds"), _FRONDS)
        self.frond_length = self._float(self.config.get("frond_length"), _FROND_L)
        self.frond_width = self._float(self.config.get("frond_width"), _FROND_W)
        self.frond_pitch = self._float(self.config.get("frond_pitch_deg"), _PITCH_DEG)
        self.frond_droop = self._float(self.config.get("frond_droop_deg"), _DROOP_DEG)
        self.frond_tier = self._float(self.config.get("frond_tier"), _FROND_TIER)
        self.bunches = list(self.config.get("bunches") or [])

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        for key, minimum in (
            ("trunk_height", 0.0),
            ("trunk_radius", 0.0),
            ("pot_radius", 0.0),
            ("crown_height", 0.0),
            ("frond_length", 0.0),
            ("frond_width", 0.0),
        ):
            if key not in config:
                continue
            try:
                if float(config[key]) <= minimum:
                    errors.append(f"'{key}' must be > {minimum:g}")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number")
        if "pot_height" in config:
            try:
                if float(config["pot_height"]) < 0:
                    errors.append("'pot_height' must be >= 0 (0 for a pole with no pot)")
            except (TypeError, ValueError):
                errors.append("'pot_height' must be a number")
        if "fronds" in config:
            try:
                if int(config["fronds"]) < 0:
                    errors.append("'fronds' must be >= 0")
            except (TypeError, ValueError):
                errors.append("'fronds' must be an integer >= 0")
        crown = self._float(config.get("crown_height", self.crown_height), self.crown_height)
        trunk = self._float(config.get("trunk_height", self.trunk_height), self.trunk_height)
        # A crown above the pole tip would float, which reads as a modelling slip rather than a choice.
        if crown > trunk:
            errors.append(f"'crown_height' {crown:g} must be <= 'trunk_height' {trunk:g}")
        for i, bunch in enumerate(self.bunches):
            if not isinstance(bunch, dict) or len(bunch.get("pos", [])) != 3:
                errors.append(f"bunches[{i}]: 'pos' must be [x, y, z] relative to the trunk base")
                continue
            if self._float(bunch.get("radius"), _BUNCH_R) <= 0:
                errors.append(f"bunches[{i}]: 'radius' must be > 0")
            if self._int(bunch.get("fruits"), 9) < 1:
                errors.append(f"bunches[{i}]: 'fruits' must be >= 1")
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

        self._add_pot(body, mats)
        self._add_trunk(body, mats)
        self._add_crown(body, mats)
        self._add_bunches(body, mats)

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        # Publish the bunch centres in WORLD coordinates. A benchmark that plans to a fruit bunch
        # should read the target from the scene rather than carry its own copy of the number -- the
        # two drifting apart is how a "goal at the bunch" silently becomes a goal in mid-air.
        centres = []
        for bunch in self.bunches:
            local = [float(v) for v in bunch["pos"]]
            centres.append([self.pos[i] + local[i] for i in range(3)])
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="scenery",
                meta={
                    "prefix": self.prefix,
                    "body": f"{self.prefix}{self._ROOT_BODY}",
                    "bunch_sites": [f"{self.prefix}bunch_{i}" for i in range(len(self.bunches))],
                    "bunch_world_pos": centres,
                },
            )
        )

    # -- geometry helpers ---------------------------------------------------------------------

    @staticmethod
    def _materials(child: mujoco.MjSpec) -> dict[str, str]:
        names = {}
        for role, rgba, specular, shininess in (
            ("trunk", _TRUNK_RGBA, 0.1, 0.1),
            ("pot", _POT_RGBA, 0.3, 0.2),
            ("frond", _FROND_RGBA, 0.25, 0.3),
            ("fruit", _FRUIT_RGBA, 0.35, 0.4),
        ):
            mat = child.add_material()
            mat.name = f"palm_{role}"
            mat.rgba = rgba
            mat.specular = specular
            mat.shininess = shininess
            names[role] = mat.name
        return names

    @staticmethod
    def _cylinder(body, name, radius, half_h, pos, material):
        g = body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [radius, half_h, 0.0]
        g.pos = list(pos)
        g.material = material
        return g

    # -- parts --------------------------------------------------------------------------------

    def _add_pot(self, body, mats) -> None:
        if self.pot_height <= 0:
            return
        self._cylinder(
            body,
            "pot",
            self.pot_radius,
            self.pot_height / 2,
            [0, 0, self.pot_height / 2],
            mats["pot"],
        )

    def _add_trunk(self, body, mats) -> None:
        """One cylinder from the pot rim to the pole tip.

        The pole starts INSIDE the pot (at z=0, not at the rim) so there is no seam a depth sensor can
        see through -- the octomap would otherwise show a gap in the trunk at pot height.
        """
        top = self.trunk_height
        self._cylinder(body, "trunk", self.trunk_radius, top / 2, [0, 0, top / 2], mats["trunk"])

    def _add_crown(self, body, mats) -> None:
        seg = self.frond_length / 2
        half_w = self.frond_width / 2
        for i in range(self.fronds):
            yaw = 2 * math.pi * i / self.fronds
            # Alternating tiers: a real crown is not a flat disc, and a flat one would leave a single
            # planar barrier rather than the staggered gaps the experiment's "narrow passage" needs.
            base_z = self.crown_height + (self.frond_tier if i % 2 else 0.0)
            pitch_in = math.radians(self.frond_pitch)
            pitch_out = math.radians(self.frond_pitch - self.frond_droop)
            # Inner segment: from the trunk surface, rising.
            r0 = self.trunk_radius
            mid_in_r = r0 + seg / 2 * math.cos(pitch_in)
            mid_in_z = base_z + seg / 2 * math.sin(pitch_in)
            end_in_r = r0 + seg * math.cos(pitch_in)
            end_in_z = base_z + seg * math.sin(pitch_in)
            # Outer segment: continues from the inner tip, arching over.
            mid_out_r = end_in_r + seg / 2 * math.cos(pitch_out)
            mid_out_z = end_in_z + seg / 2 * math.sin(pitch_out)
            for tag, (mr, mz, pitch) in (
                ("in", (mid_in_r, mid_in_z, pitch_in)),
                ("out", (mid_out_r, mid_out_z, pitch_out)),
            ):
                g = body.add_geom()
                g.name = f"frond_{i}_{tag}"
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = [seg / 2, half_w, _FROND_T / 2]
                g.pos = [mr * math.cos(yaw), mr * math.sin(yaw), mz]
                # -pitch: a MuJoCo y-rotation lifts the box's +x end when the angle is negative.
                g.quat = _euler_zy_quat(yaw, -pitch)
                g.material = mats["frond"]

    def _add_bunches(self, body, mats) -> None:
        """Each bunch: fruitlets packed on two rings around its centre, plus a site at that centre.

        A sphere cluster rather than one big sphere because the bunch's *surface* is what a depth
        sensor maps and what the arm has to approach between -- a smooth ball would map as a
        smooth ball, and the approach-from-below pose the experiment uses would have nothing to
        thread.
        """
        for i, bunch in enumerate(self.bunches):
            cx, cy, cz = (float(v) for v in bunch["pos"])
            radius = self._float(bunch.get("radius"), _BUNCH_R)
            fruits = self._int(bunch.get("fruits"), 9)
            fruit_r = self._float(bunch.get("fruit_radius"), _FRUIT_R)
            for k in range(fruits):
                # Two rings (upper/lower) plus a centre fruitlet, spiralled so they interlock.
                ring = k % 2
                ang = 2 * math.pi * k / max(1, fruits)
                rr = radius * (0.55 if ring else 0.85)
                dz = radius * (0.35 if ring else -0.25)
                g = body.add_geom()
                g.name = f"bunch_{i}_fruit_{k}"
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [fruit_r, 0.0, 0.0]
                g.pos = [cx + rr * math.cos(ang), cy + rr * math.sin(ang), cz + dz]
                g.material = mats["fruit"]
            s = body.add_site()
            s.name = f"bunch_{i}"
            s.pos = [cx, cy, cz]
            s.size = [0.005, 0.005, 0.005]
