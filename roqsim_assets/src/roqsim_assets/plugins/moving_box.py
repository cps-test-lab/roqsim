"""Scene plugin: a **moving** rectangular box obstacle, driven kinematically from the world YAML.

The dynamic counterpart of :mod:`roqsim_assets.plugins.box`. `box` is welded scenery; this one is a
**mocap** body whose pose the plugin writes every step, so it crosses a corridor at a constant speed
without physics ever pushing it around. Being mocap also means it has infinite effective mass: a robot
that drives into it is stopped by it, which is what an obstacle is for.

Why this exists as its own plugin rather than a flag on `box`: the substrate had exactly one
kinematically driven mover, ``roqsim_walker``'s ``walker``, and that one is a *pedestrian* — a humanoid
blueprint with locomotion clips and ORCA avoidance. A great many navigation papers instead put
anonymous boxes in the robot's way and state only a speed (this plugin was written to reconstruct a
maze paper whose four cubes "move randomly with fixed velocity"), and
dressing such a cube as a person changes both what the lidar sees and what the costmap inflates.

Two motion modes, and they are mutually exclusive:

``waypoints`` — a fixed route, the reproducible default. Deterministic, inspectable, diffable.

``random_walk`` — a seeded random walk for papers that specify only "moves randomly". The box drives
straight until it is about to hit something, then picks a new heading. Obstacles are found by
**ray-casting the compiled model**, so it respects whatever geometry the world contains (baked scene,
`box` props, other movers) without a map file, and it needs no wall list to be configured. The seed is
part of the config, so a campaign varies the motion by varying the seed and every trial replays
exactly.

Config::

    moving_box:
      prefix: "cube1_"          # MJCF name prefix (use distinct prefixes for >1 mover)
      size: [0.3, 0.3, 0.3]     # full extents, metres (REQUIRED)
      pos: [x, y]               # start; [x, y] sits it ON the floor, [x, y, z] sets its CENTRE
      yaw: 0.0                  # rotation about z, radians (constant; the box does not turn)
      speed: 0.1                # m/s along the route (REQUIRED, > 0)
      color: [r, g, b, a]       # default a light warehouse grey; alpha optional
      collide: true             # false -> nothing bumps into it (a raycast still sees it)
      friction: 1.0             # sliding friction, or the full [sliding, torsional, rolling] triple

      # -- mode A: a fixed route -----------------------------------------------------------------
      waypoints: [[2.0, 1.0], [2.0, -3.0]]   # world metres; the box starts at `pos`
      loop: true                # true -> cycle the route forever; false -> stop at the last point
      ping_pong: false          # true -> reverse at the end instead of jumping back to the start

      # -- mode B: a seeded random walk -----------------------------------------------------------
      random_walk:
        seed: 1                 # REQUIRED — an unseeded random obstacle is not an experiment
        clearance: 0.5          # m of free space required ahead; below it, a new heading is picked
        bounds: [x0, y0, x1, y1]   # optional axis-aligned box the centre must stay inside
        turn_deg: [60, 300]     # heading change sampled uniformly from this range, in degrees

``pos`` is where the box is at ``on_reset``, every episode: a trial never inherits the previous
trial's obstacle position, and neither does the RNG (it is re-seeded), so repetition N of a cell sees
the same obstacle motion however many trials ran before it.
"""

from __future__ import annotations

import math
import random

import mujoco
import numpy as np

from roqsim import raycast
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

_GREY_RGBA = [0.86, 0.86, 0.83, 1.0]  # pale warehouse carton, as `box`
_ROOT_BODY = "moving_box"
_DEFAULT_TURN_DEG = (60.0, 300.0)


#: Effectively "no culling", matching the `mj_ray` this plugin used to call.
_NO_CUTOFF = 1e6


class MovingBoxPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.size = self._vec3(self.config.get("size"), (0.3, 0.3, 0.3))
        self.pos = self._pos(self.config.get("pos"), self.size[2])
        self.yaw = self._float(self.config.get("yaw"), 0.0)
        self.speed = self._float(self.config.get("speed"), 0.1)
        self.color = self._rgba(self.config.get("color")) or _GREY_RGBA
        self.collide = bool(self.config.get("collide", True))
        self.friction = self._friction(self.config.get("friction"))

        self.waypoints = self._waypoints(self.config.get("waypoints"))
        self.loop = bool(self.config.get("loop", True))
        self.ping_pong = bool(self.config.get("ping_pong", False))

        walk = self.config.get("random_walk") or None
        self.random_walk = dict(walk) if isinstance(walk, dict) else None

        self._body_name = self.prefix + _ROOT_BODY
        self._mocapid = -1
        self._geom_ids: list[int] = []
        # motion state (reset by on_reset)
        self._xy = np.array(self.pos[:2], dtype=float)
        self._target = 0
        self._dir = 1
        self._heading = 0.0
        self._rng: random.Random | None = None
        self._done = False

    # -- config coercion (same shapes as `box`, so the two props read alike) ----------------------
    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _vec3(value, default):
        try:
            if len(value) >= 3:
                return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            pass
        return default

    def _pos(self, value, height: float):
        try:
            if len(value) >= 3:
                return float(value[0]), float(value[1]), float(value[2])
            if len(value) == 2:
                return float(value[0]), float(value[1]), height / 2.0
        except (TypeError, ValueError):
            pass
        return 0.0, 0.0, height / 2.0

    @staticmethod
    def _rgba(value):
        if value is None:
            return None
        try:
            rgba = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if len(rgba) == 3:
            rgba.append(1.0)
        return rgba if len(rgba) == 4 else None

    @staticmethod
    def _friction(value):
        if value is None:
            return [1.0, 0.005, 0.0001]
        try:
            if isinstance(value, (int, float)):
                return [float(value), 0.005, 0.0001]
            triple = [float(v) for v in value]
        except (TypeError, ValueError):
            return [1.0, 0.005, 0.0001]
        return triple if len(triple) == 3 else [1.0, 0.005, 0.0001]

    @staticmethod
    def _waypoints(value) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        if not value:
            return out
        try:
            for wp in value:
                out.append((float(wp[0]), float(wp[1])))
        except (TypeError, ValueError, IndexError):
            return []
        return out

    # -- validation ------------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        for key in ("pos", "size", "speed"):
            if key not in config:
                errors.append(f"'{key}' is required")
        if "size" in config:
            try:
                extents = [float(v) for v in config["size"]]
            except (TypeError, ValueError):
                errors.append("'size' must be three numbers: full extents [x, y, z] in metres")
            else:
                if len(extents) != 3 or any(v <= 0 for v in extents):
                    errors.append(f"'size' must be three positive numbers, got {extents}")
        if "pos" in config:
            try:
                if len(config["pos"]) not in (2, 3):
                    errors.append("'pos' must be [x, y] or [x, y, z] in world metres")
            except TypeError:
                errors.append("'pos' must be [x, y] or [x, y, z] in world metres")
        if "speed" in config and self._float(config["speed"], -1.0) <= 0.0:
            errors.append("'speed' must be a positive number of m/s")

        has_wp = bool(config.get("waypoints"))
        walk = config.get("random_walk")
        # Fail loudly rather than picking a motion for the author: a silently-chosen obstacle
        # trajectory is exactly the kind of invisible experiment change this substrate refuses.
        if has_wp and walk:
            errors.append("give either 'waypoints' or 'random_walk', not both")
        if not has_wp and not walk:
            errors.append("a mover needs a motion: give 'waypoints' or 'random_walk'")
        if has_wp and not self._waypoints(config["waypoints"]):
            errors.append("'waypoints' must be a list of [x, y] pairs in world metres")
        if walk is not None:
            if not isinstance(walk, dict):
                errors.append("'random_walk' must be a mapping")
            elif "seed" not in walk:
                errors.append(
                    "'random_walk' requires a 'seed' — an unseeded obstacle is not reproducible"
                )
            else:
                try:
                    int(walk["seed"])
                except (TypeError, ValueError):
                    errors.append("'random_walk.seed' must be an integer")
                if "bounds" in walk:
                    try:
                        b = [float(v) for v in walk["bounds"]]
                        if len(b) != 4 or b[0] >= b[2] or b[1] >= b[3]:
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(
                            "'random_walk.bounds' must be [x0, y0, x1, y1] with x0<x1, y0<y1"
                        )
        for key in ("collide", "loop", "ping_pong"):
            if key in config and not isinstance(config[key], bool):
                errors.append(f"'{key}' must be a boolean")
        return errors

    # -- lifecycle -------------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()

        mat = child.add_material()
        mat.name = "moving_box_surface"
        mat.rgba = self.color

        body = child.worldbody.add_body()
        body.name = _ROOT_BODY
        # mocap, not a free joint: the plugin owns this body's pose, and physics may not move it.
        # A free-jointed box would be shoved aside by the robot it is supposed to obstruct, and its
        # commanded motion would fight the solver instead of defining the experiment.
        body.mocap = True

        g = body.add_geom()
        g.name = "moving_box"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [self.size[0] / 2, self.size[1] / 2, self.size[2] / 2]  # MuJoCo wants half-extents
        g.material = mat.name
        g.friction = self.friction
        if not self.collide:
            g.contype = 0
            g.conaffinity = 0

        frame = spec.worldbody.add_frame()
        frame.pos = list(self.pos)
        frame.quat = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, self._body_name)
        if bid < 0:
            raise RuntimeError(f"moving_box: body {self._body_name!r} not found")
        self._mocapid = int(ctx.model.body_mocapid[bid])
        if self._mocapid < 0:
            raise RuntimeError(
                f"moving_box: body {self._body_name!r} is not a mocap body — cannot drive it"
            )
        # The mover's own geoms, so the look-ahead ray never reports the box hitting itself.
        self._geom_ids = [g for g in range(ctx.model.ngeom) if int(ctx.model.geom_bodyid[g]) == bid]
        ctx.entities.add(
            Entity(
                name=self.entity_name,
                kind="object",
                body=self._body_name,
                meta={
                    "prefix": self.prefix,
                    "size": list(self.size),
                    "pos": list(self.pos),
                    "yaw": self.yaw,
                    "speed": self.speed,
                    "motion": "random_walk" if self.random_walk else "waypoints",
                },
            )
        )
        self.on_reset(ctx)

    def on_reset(self, ctx: SimContext) -> None:
        """Re-seat the box AND re-seed its RNG, so repetition N never inherits trial N-1's state."""
        self._xy = np.array(self.pos[:2], dtype=float)
        self._target = 0
        self._dir = 1
        self._done = False
        if self.random_walk is not None:
            self._rng = random.Random(int(self.random_walk["seed"]))
            self._heading = self._rng.uniform(-math.pi, math.pi)
        if ctx.data is not None and self._mocapid >= 0:
            ctx.data.mocap_pos[self._mocapid] = [self._xy[0], self._xy[1], self.pos[2]]
            ctx.data.mocap_quat[self._mocapid] = [
                math.cos(self.yaw / 2),
                0.0,
                0.0,
                math.sin(self.yaw / 2),
            ]

    def pre_step(self, ctx: SimContext) -> None:
        if self._mocapid < 0 or self._done:
            return
        dt = float(ctx.model.opt.timestep)
        step = self.speed * dt
        if self.random_walk is not None:
            self._advance_random(ctx, step)
        else:
            self._advance_route(step)
        ctx.data.mocap_pos[self._mocapid] = [self._xy[0], self._xy[1], self.pos[2]]

    # -- motion ----------------------------------------------------------------------------------
    def _advance_route(self, step: float) -> None:
        """Walk the polyline at constant speed, consuming as many segments as one step spans."""
        remaining = step
        guard = 0
        while remaining > 1e-12 and not self._done:
            guard += 1
            if guard > len(self.waypoints) + 2:  # a zero-length route would spin here
                return
            tgt = np.array(self.waypoints[self._target], dtype=float)
            delta = tgt - self._xy
            dist = float(np.linalg.norm(delta))
            if dist <= remaining:
                self._xy = tgt
                remaining -= dist
                self._next_waypoint()
            else:
                self._xy = self._xy + delta / dist * remaining
                remaining = 0.0

    def _next_waypoint(self) -> None:
        if self.ping_pong:
            if self._target + self._dir >= len(self.waypoints) or self._target + self._dir < 0:
                self._dir *= -1
            self._target += self._dir
            return
        self._target += 1
        if self._target >= len(self.waypoints):
            if self.loop:
                self._target = 0
            else:
                self._target = len(self.waypoints) - 1
                self._done = True  # parked on the last waypoint for the rest of the trial

    def _advance_random(self, ctx: SimContext, step: float) -> None:
        """Drive straight; on an imminent obstacle (or bound), pick a new heading and try again."""
        walk = self.random_walk or {}
        clearance = self._float(walk.get("clearance"), 0.5)
        lo, hi = self._turn_range(walk.get("turn_deg"))
        for _ in range(12):  # bounded: a box in a dead end must not spin forever inside one step
            nxt = self._xy + step * np.array([math.cos(self._heading), math.sin(self._heading)])
            if self._is_free(ctx, nxt, clearance) and self._in_bounds(walk.get("bounds"), nxt):
                self._xy = nxt
                return
            assert self._rng is not None
            self._heading = _wrap(self._heading + math.radians(self._rng.uniform(lo, hi)))
        # Every sampled heading was blocked: hold position this step rather than tunnelling through
        # geometry. The next step re-samples, so a boxed-in mover idles instead of teleporting out.

    def _turn_range(self, value) -> tuple[float, float]:
        try:
            lo, hi = float(value[0]), float(value[1])
            return (lo, hi) if lo <= hi else (hi, lo)
        except (TypeError, ValueError, IndexError):
            return _DEFAULT_TURN_DEG

    def _is_free(self, ctx: SimContext, xy: np.ndarray, clearance: float) -> bool:
        """Ray-cast the compiled model ahead of the box; True if nothing is within `clearance`.

        Casting from the box's centre at its own mid-height means the mover sees whatever the model
        contains — baked walls, other props, the robot — with no map file and no configured wall
        list. The mover's own geoms are excluded by id.

        Through `roqsim.raycast.cast`, so an entity that has been made *absent* is not an obstacle:
        a prop nothing can collide with should not make the mover turn. This was the last raycaster
        in the tree that passed `geomgroup=None` and saw absent entities.
        """
        direction = np.array([math.cos(self._heading), math.sin(self._heading), 0.0])
        origin = np.array([xy[0], xy[1], self.pos[2]])
        # Half the diagonal, so the corner leading the way is what has to fit, not the centre.
        half = 0.5 * math.hypot(self.size[0], self.size[1])
        # Deliberately generous: `cutoff` culls geoms beyond it, and `mj_ray` (which this replaced)
        # has no cutoff at all, so a large value keeps the predicate identical rather than making
        # the answer depend on a culling distance.
        hits = raycast.cast(ctx.model, ctx.data, origin, direction, cutoff=_NO_CUTOFF)
        dist = float(hits.dist[0])
        if dist < 0:
            return True
        if int(hits.geomid[0]) in self._geom_ids:
            return True
        return dist > clearance + half

    @staticmethod
    def _in_bounds(bounds, xy: np.ndarray) -> bool:
        if not bounds:
            return True
        try:
            x0, y0, x1, y1 = (float(v) for v in bounds)
        except (TypeError, ValueError):
            return True
        return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
