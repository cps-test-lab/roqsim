"""Scene + controller plugin: carry a prop along a prescribed planar path at a fixed speed.

The generic "a thing in the scene moves along a path I hand it" capability. `conveyor` moves objects
along ONE axis by belt friction and `walker` follows waypoints but builds a skinned *person*; neither
covers "this object must trace this 2-D polyline at this speed", which is what a dynamic-manipulation
benchmark needs.

Mechanism: an XY stage. The plugin builds a carrier plate on two nested slide joints (x, then y) and
forces their velocities every ``pre_step`` so the plate tracks the path. Anything resting on the plate
is carried **by friction**, exactly as on the physical rig -- the object is never teleported, so its
contact with the plate (and with a gripper closing on it) stays physical. This mirrors ``conveyor``,
which forces its belt slab's joint velocity rather than scripting the package's pose.

It is also what the hardware is: the rig this was written for (DGBench, Burgess-Limerick et al. 2022) is
a belt-driven **CoreXY** stage executing g-code, i.e. two orthogonal prescribed axes.

Config::

    prop_trajectory:
      name: object_stage         # the entry's OWN key, not the config's: names the entity (default 'prop_trajectory')
      prefix: ""                 # MJCF name prefix (distinct prefixes for >1 stage)
      path: trajectories/t1.csv  # 2-column "x,y" CSV, resolved relative to the world YAML
      units: mm                  # mm | m -- the CSV's units (default mm)
      speed: 0.03                # m/s along the path (arc length), constant
      origin: [2.0, 1.0, 0.6]    # world pose of CSV point (0,0); z is the plate's TOP surface
      plate: [0.07, 0.07, 0.006] # carrier plate half-extents (x, y, z)
      friction: 1.0              # plate friction (a carried object must not slide off)
      loop: false                # restart at the path start when the end is reached
      start_index: 0             # begin at this CSV row (phase offset within one path)
      travel: [0.15, 0.15]       # +/- soft limits on each axis (m); the path is clamped into them

The plate is driven, not simulated: its joints have no actuator and are velocity-forced, so the stage
is infinitely stiff and its motion is exactly the commanded path regardless of the load it carries. That
is the right model for a stepper-driven gantry and the wrong one for a compliant conveyor.

``Props carried by the plate need their own free joint`` -- spawn them with
``spawn_model: {..., free: true}`` and place them just above the plate's top surface.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np

from roqsim.context import Endpoint, Entity, SimContext
from roqsim.plugin import Plugin


class PropTrajectoryPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    @classmethod
    def expand(cls, spec, world, base_dir):
        """Record the world file's directory so ``path`` resolves relative to the WORLD, not the CWD.

        ``base_dir`` is handed only to ``expand`` (the config-load hook), never to ``build``, so this is
        where a plugin that reads a data file has to capture it. Resolving relative to the world file is
        what lets an experiment keep its trajectories beside its worlds and still run from anywhere --
        a CWD-relative path silently becomes "file not found" the moment a campaign runs from elsewhere.
        """
        spec.config.setdefault("_base_dir", str(base_dir))
        return []

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if not config.get("path"):
            errors.append("prop_trajectory: `path` (a 2-column x,y CSV) is required")
        if config.get("units", "mm") not in ("mm", "m"):
            errors.append("prop_trajectory: `units` must be 'mm' or 'm'")
        if float(config.get("speed", 0.03)) <= 0.0:
            errors.append("prop_trajectory: `speed` must be > 0")
        plate = config.get("plate", [0.07, 0.07, 0.006])
        if len(plate) != 3:
            errors.append("prop_trajectory: `plate` must be [hx, hy, hz]")
        return errors

    def sources(self) -> list:
        """The trajectory CSV, resolved the same way ``_load_path`` resolves it.

        Relative to the WORLD file (``_base_dir``, captured in ``expand``), not the CWD -- so a
        caller staging this world elsewhere gets the file the run would have read.
        """
        raw = self.config.get("path")
        if not raw:
            return []
        path = Path(raw)
        if not path.is_absolute():
            path = Path(self.config.get("_base_dir") or ".") / path
        return [str(path.resolve())]

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.speed = float(self.config.get("speed", 0.03))
        self.origin = [float(v) for v in self.config.get("origin", [0.0, 0.0, 0.0])]
        self.plate = [float(v) for v in self.config.get("plate", [0.07, 0.07, 0.006])]
        self.friction = float(self.config.get("friction", 1.0))
        self.loop = bool(self.config.get("loop", False))
        self.start_index = int(self.config.get("start_index", 0))
        self.travel = [float(v) for v in self.config.get("travel", [0.15, 0.15])]
        self._pts: np.ndarray = np.zeros((0, 2))
        self._cum: np.ndarray = np.zeros(0)  # cumulative arc length, metres
        self._s = 0.0  # distance travelled along the path
        self._qadr = (-1, -1)
        self._dadr = (-1, -1)
        self._done = False

    # -- path ----------------------------------------------------------------------------------
    def _load_path(self, base_dir: Path | None) -> None:
        raw = Path(self.config["path"])
        path = raw if raw.is_absolute() else Path(base_dir or ".") / raw
        if not path.exists():
            # Fail loudly: a silently missing trajectory would make the dynamic condition quietly
            # identical to the static one, i.e. a different experiment reported as this one.
            raise RuntimeError(f"prop_trajectory: path file not found: {path}")
        pts = []
        with path.open() as fh:
            for row in csv.reader(fh):
                if len(row) < 2 or row[0].lstrip().startswith("#"):
                    continue
                try:
                    pts.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue  # header line
        if len(pts) < 2:
            raise RuntimeError(f"prop_trajectory: {path} has fewer than 2 usable points")
        arr = np.asarray(pts, dtype=float)
        if self.config.get("units", "mm") == "mm":
            arr /= 1000.0
        arr = arr[self.start_index :] if self.start_index < len(arr) else arr[-2:]
        arr -= arr[0]  # the path starts at the origin, whatever row we began on
        arr[:, 0] = np.clip(arr[:, 0], -self.travel[0], self.travel[0])
        arr[:, 1] = np.clip(arr[:, 1], -self.travel[1], self.travel[1])
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        self._pts = arr
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])

    def _target(self, s: float) -> np.ndarray:
        """Position at arc length ``s``, linearly interpolated between path points."""
        total = float(self._cum[-1])
        if total <= 0.0:
            return self._pts[0]
        if s >= total:
            if not self.loop:
                self._done = True
                return self._pts[-1]
            s = s % total
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        i = max(0, min(i, len(self._pts) - 2))
        span = self._cum[i + 1] - self._cum[i]
        f = 0.0 if span <= 0 else (s - self._cum[i]) / span
        return self._pts[i] + f * (self._pts[i + 1] - self._pts[i])

    # -- lifecycle -----------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        self._load_path(self.config.get("_base_dir"))
        p = self.prefix
        hx, hy, hz = self.plate
        # Two nested slide joints = an XY stage. `origin` z is the plate's TOP, so the body sits hz
        # lower: a carried object's rest height is what the world YAML actually cares about.
        xb = spec.worldbody.add_body(
            name=f"{p}stage_x", pos=[self.origin[0], self.origin[1], self.origin[2] - hz]
        )
        jx = xb.add_joint(name=f"{p}stage_x_joint", type=mujoco.mjtJoint.mjJNT_SLIDE)
        jx.axis = [1, 0, 0]
        jx.range = [-self.travel[0] * 1.5, self.travel[0] * 1.5]
        # A rail with mass: MuJoCo rejects a massless moving body, and the stage is force-driven
        # anyway so the value only has to be physical, not exact.
        xb.add_geom(
            name=f"{p}stage_x_rail",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, hz * 0.5],
            pos=[0, 0, -hz * 1.6],
            rgba=[0.25, 0.25, 0.28, 1.0],
            contype=0,
            conaffinity=0,
            mass=0.5,
        )
        yb = xb.add_body(name=f"{p}stage", pos=[0, 0, 0])
        jy = yb.add_joint(name=f"{p}stage_y_joint", type=mujoco.mjtJoint.mjJNT_SLIDE)
        jy.axis = [0, 1, 0]
        jy.range = [-self.travel[1] * 1.5, self.travel[1] * 1.5]
        plate = yb.add_geom(
            name=f"{p}stage_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, hz],
            pos=[0, 0, 0],
            rgba=[0.55, 0.55, 0.58, 1.0],
            mass=1.0,
        )
        plate.friction = [self.friction, 0.005, 0.0001]
        plate.condim = 4  # torsional friction: a carried box must not spin off the plate

    def configure(self, ctx: SimContext) -> None:
        m = ctx.model
        p = self.prefix
        jx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}stage_x_joint")
        jy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}stage_y_joint")
        if jx < 0 or jy < 0:
            raise RuntimeError("prop_trajectory: stage joints missing (build did not run?)")
        self._qadr = (int(m.jnt_qposadr[jx]), int(m.jnt_qposadr[jy]))
        self._dadr = (int(m.jnt_dofadr[jx]), int(m.jnt_dofadr[jy]))
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="object",
                body=f"{p}stage",
                meta={"prefix": p, "path": str(self.config["path"]), "speed": self.speed},
            )
        )
        # Expose the stage's progress so a trial can be gated on it (and so a run's log records what
        # the object actually did, not just what it was asked to do).
        ctx.interface.add(
            Endpoint(
                name="stage_progress",
                direction="out",
                owner=self.entity_name,
                namespace=self.config.get("namespace", ""),
                read=self.read_progress,
                rate_hz=30.0,
                backend={"ros2": {"type": "std_msgs.msg.Float64", "topic": "stage_progress"}},
            )
        )
        self.on_reset(ctx)

    def read_progress(self):
        """(arc length travelled [m], total path length [m], finished?)."""
        return float(self._s), float(self._cum[-1] if len(self._cum) else 0.0), bool(self._done)

    def on_reset(self, ctx: SimContext) -> None:
        self._s = 0.0
        self._done = False
        start = self._target(0.0)
        ctx.data.qpos[self._qadr[0]] = float(start[0])
        ctx.data.qpos[self._qadr[1]] = float(start[1])
        ctx.data.qvel[self._dadr[0]] = 0.0
        ctx.data.qvel[self._dadr[1]] = 0.0

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            return
        dt = float(ctx.model.opt.timestep)
        self._s += self.speed * dt
        tgt = self._target(self._s)
        cur = np.array([ctx.data.qpos[self._qadr[0]], ctx.data.qpos[self._qadr[1]]], dtype=float)
        # Force velocity (not qpos) so the plate pushes what it carries through ordinary contact
        # dynamics, the same way `conveyor` drives its belt slab. Writing qpos directly would move the
        # plate through a carried object instead of taking it along.
        v = (tgt - cur) / dt
        ctx.data.qvel[self._dadr[0]] = float(v[0])
        ctx.data.qvel[self._dadr[1]] = float(v[1])
