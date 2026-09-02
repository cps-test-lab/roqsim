"""Scene plugin: a **parametric** round ventilation duct run under a ceiling.

The spiral-seam steel tube that runs across an exposed soffit in an office or a hall, with the odd
branch dropping out of it into a diffuser. Parametric because a run is a line between two points and
a list of branch offsets -- a per-room MJCF would freeze exactly the two numbers that change.

Geometry (all metres). The run is a cylinder from ``start`` to ``end`` at height ``z`` (its axis).
Each entry in ``branches`` is a distance **along the run from ``start``** where a tee drops out:
a short vertical tube of ``branch_radius`` hanging ``branch_length`` below the main tube, closed by a
flat ``diffuser_radius`` disc -- the round outlet you see from the floor. An empty list is a bare run.

Like ``ceiling_panels`` everything here sits above head height, so the core ``ceiling`` plugin can
delete the lot by height for a top-down view -- list ``ceiling`` after this one, build order is list
order.

Config::

    duct:
      name: duct          # the entry's OWN key, not the config's: names the entity (default 'duct')
      prefix: ""          # MJCF name prefix (distinct prefixes for >1 run)
      start: [x, y]       # run start, world (REQUIRED)
      end: [x, y]         # run end, world (REQUIRED)
      z: 3.2              # height of the run's AXIS, m
      radius: 0.14        # main tube radius, m
      branches: []        # distances along the run, m, where a drop + diffuser hangs
      branch_length: 0.40 # how far a drop hangs below the main tube, m
      branch_radius: 0.08 # drop tube radius, m
      diffuser_radius: 0.16   # disc at the bottom of a drop, m (0 = no disc)
      color: [r,g,b,a]    # galvanised steel by default; alpha optional

Visual only (no contacts), for the same reason the panels are: nothing driving on the floor reaches
a duct, and the raycaster sees it regardless.
"""

from __future__ import annotations

import math

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_STEEL_RGBA = [0.72, 0.73, 0.75, 1.0]
_DISC_T = 0.02  # diffuser disc thickness


class DuctPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "duct"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.start = self._xy(self.config.get("start"), (0.0, 0.0))
        self.end = self._xy(self.config.get("end"), (1.0, 0.0))
        self.z = self._float(self.config.get("z"), 3.2)
        self.radius = self._float(self.config.get("radius"), 0.14)
        self.branch_length = self._float(self.config.get("branch_length"), 0.40)
        self.branch_radius = self._float(self.config.get("branch_radius"), 0.08)
        self.diffuser_radius = self._float(self.config.get("diffuser_radius"), 0.16)
        self.color = self._rgba(self.config.get("color")) or _STEEL_RGBA
        try:
            self.branches = [float(t) for t in (self.config.get("branches") or [])]
        except (TypeError, ValueError):
            self.branches = []

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _xy(value, default: tuple[float, float]) -> tuple[float, float]:
        try:
            if len(value) >= 2:
                return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            pass
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

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        for key in ("start", "end"):
            if key not in config:
                errors.append(f"'{key}' is required: [x, y] in world metres")
            elif len(config[key]) < 2:
                errors.append(f"'{key}' must be [x, y]")
        if self.length <= 0:
            errors.append("'start' and 'end' are the same point: a run needs a length")
        for key in ("radius", "branch_radius", "branch_length", "diffuser_radius", "z"):
            if key in config:
                try:
                    value = float(config[key])
                except (TypeError, ValueError):
                    errors.append(f"'{key}' must be a number")
                    continue
                if key != "z" and value < 0:
                    errors.append(f"'{key}' must be >= 0")
        if "branches" in config:
            try:
                offsets = [float(t) for t in config["branches"]]
            except (TypeError, ValueError):
                errors.append("'branches' must be a list of distances along the run, in metres")
            else:
                outside = [t for t in offsets if t < 0 or t > self.length]
                if outside:
                    errors.append(
                        f"'branches' {outside} fall outside the {self.length:.2f} m run "
                        f"(distances are measured from 'start')"
                    )
        if "color" in config and self._rgba(config["color"]) is None:
            errors.append("'color' must be [r, g, b] or [r, g, b, a] numbers")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        mat = child.add_material()
        mat.name = "duct_steel"
        mat.rgba = self.color
        mat.specular = 0.6
        mat.shininess = 0.5

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY

        # The child is built along +x from the run's start; the attach frame below carries the yaw.
        # A MuJoCo cylinder is along its local z, so the run is turned 90 deg about y.
        length = self.length
        g = body.add_geom()
        g.name = "run"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [self.radius, length / 2, 0.0]
        g.pos = [length / 2, 0.0, 0.0]
        g.quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        g.material = mat.name
        g.contype = 0
        g.conaffinity = 0

        for i, t in enumerate(self.branches, start=1):
            drop = body.add_geom()
            drop.name = f"drop_{i:02d}"
            drop.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            drop.size = [self.branch_radius, self.branch_length / 2, 0.0]
            drop.pos = [t, 0.0, -self.branch_length / 2]
            drop.material = mat.name
            drop.contype = 0
            drop.conaffinity = 0
            if self.diffuser_radius > 0:
                disc = body.add_geom()
                disc.name = f"diffuser_{i:02d}"
                disc.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                disc.size = [self.diffuser_radius, _DISC_T / 2, 0.0]
                disc.pos = [t, 0.0, -self.branch_length - _DISC_T / 2]
                disc.material = mat.name
                disc.contype = 0
                disc.conaffinity = 0

        yaw = math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        frame = spec.worldbody.add_frame()
        frame.pos = [self.start[0], self.start[1], self.z]
        frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="prop",
                body=self.prefix + self._ROOT_BODY,
                meta={"prefix": self.prefix, "length": self.length, "z": self.z},
            )
        )
