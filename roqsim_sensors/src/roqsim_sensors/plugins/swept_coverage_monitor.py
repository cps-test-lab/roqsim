# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation plugin: the union of a *moving* sensor's field of view over a run.

The moving counterpart of :mod:`roqsim_sensors.plugins.sensor_coverage_probe`, and deliberately its
opposite number. That plugin answers *what does this world's fixed sensor set observe* -- once, at
``configure``, from a mount that never moves. This one answers *what did a sensor carried through the
world ever observe* -- re-posing the same field of view at the mount's live pose each tick and
OR-accumulating the result over a fixed sample set. A layout question and a trajectory question, one
owner each, over one geometry stack: both go through :func:`roqsim_sensors.coverage.engine.coverage`,
so a swept figure and a static one are the same measurement asked at different times.

**Why the simulator and not an evaluator afterwards.** A swept union reconstructed from recorded poses
is worse in two ways that cannot be fixed downstream. It needs a model of the field of view outside
the simulator -- a cone or a radius standing in for the real angular sector, with no line of sight at
all -- so a cell behind a wall counts as covered because the sensor pointed at it. And it is sampled
at the recording's rate rather than the sensor's, which for a sensor swinging on a moving base is the
rate that decides the answer. Here the membership test is the same range -> angular -> raycast gate a
static coverage estimate uses, against the same geometry, with occluders in place.

**The union is a lower bound, and that is the honest reading.** The field of view is evaluated at
discrete instants (``compute_rate_hz``), so whatever it swept *between* two evaluations is not
counted. Raising the rate raises the figure and the cost together; the plugin never interpolates,
because an interpolated FoV has no line-of-sight test behind it and would report coverage through
walls. Two further conservative edges, both in the same direction: the sample set is built once from
the world's initial state, so a cell the mount's own body occupies at ``t=0`` is classified as
occupied and never sampled (the sampler's own reasoning -- points wrongly dropped only make coverage
look worse, never better); and a raycast is excluded from the mount body alone, so the rest of a
carrier's structure occludes exactly as any other geometry does.

**It never ends a trial.** Like ``clearance_monitor``, this observes; a scenario that wants to stop
once a coverage target is reached reads the endpoint and decides, with the threshold stated in the
experiment rather than in the substrate.

Config::

    swept_coverage_monitor:
      type: ""               # REQUIRED: sensor type, i.e. which coverage adapter builds the FoV
                             #   (any of `roqsim sensors coverage catalog`'s types)
      # The mount, exactly one of these three. Names are resolved with the owning entity's
      # `prefix` when this entry is nested under a spawn, so `camera: oakd_rgb` finds a
      # prefixed robot's camera.
      camera: ""             # a MuJoCo <camera>: pose AND intrinsics from the compiled model
      site: ""               # a MuJoCo <site>: pose from the model (a lidar's scan site)
      body: ""               # a MuJoCo <body>, plus the offset/rpy below
      offset: [0, 0, 0]      # sensor position in the mount frame  -- `body:` only
      rpy: [0, 0, 0]         # sensor orientation in the mount frame -- `body:` only
      config: {}             # FoV parameters for the adapter (fovy/width/height/far, range_max, ...)

      sample:                # the fixed point set the union is accumulated over
        volume: true         #   free-interior grid at `resolution`, one layer per height
        objects: false       #   object surfaces as well (labelled per geom)
        resolution: 0.25     #   grid pitch [m]; also the cell the area figure is derived from
        heights: [0.5]       #   world z of the grid layers
        per_object: 64       #   surface points per object

      regions: ""            # optional named regions (a JSON path or an inline spec / floorplan)
      region_names: []       #   subset of those regions to keep
      restrict: false        #   sample only inside the regions' union

      compute_rate_hz: 5.0   # how often the FoV is EVALUATED and the union grown
      rate_hz: 2.0           # how often the endpoint is PUBLISHED
      out: ""                # optional directory for a report.json written at shutdown

Endpoint ``coverage`` (out) reads a :class:`SweptCoverageReport`: the covered fraction of the sample
set, the covered and sampled **areas** with the cell they are derived from, how many evaluations went
into the union, and the mean number of evaluations a point was seen in -- so a revisit figure falls
out of the same accumulator as a coverage one. A :class:`SweptCoverageReader` on the blackboard under
``swept_coverage:<address>`` hands an in-process consumer the sample points and the per-point visit
counts themselves.

``compute_rate_hz`` is separate from ``rate_hz`` for the reason ``clearance_monitor`` separates them:
publishing is cheap and evaluating is not. One evaluation is a batched raycast over every sample point
that passed the range and angular gates, so the cost scales with the sample set; a coarse grid at a
few Hz is cheap, a 0.1 m grid over three heights at every physics step is not. The trade is stated
here rather than hidden in a default.

The area figure is derived from the volume grid only, because only a grid point stands for a known
piece of ground: ``covered_area_m2`` counts the distinct ``resolution``-sized xy cells covered at any
height, so several height layers do not multiply the area. With ``volume: false`` there is no grid and
both areas read ``-1.0`` -- the "this cannot be reported" convention, not a plausible-looking zero.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

from ..coverage import sampling
from ..coverage.adapters import PlacedSensor, build_fov
from ..coverage.engine import coverage
from ..coverage.report import build_report

_log = logging.getLogger(__name__)

#: The three mount keys, in the order they are reported. Exactly one may be set.
_MOUNT_KEYS = ("camera", "site", "body")

#: Areas are unreportable without a volume grid; ``-1.0`` says so rather than reading as zero.
UNKNOWN_AREA = -1.0


def _triple(value) -> np.ndarray:
    """A 3-vector from config, or zeros when it is not one -- ``validate_config`` reports the why."""
    try:
        arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(3)
    return arr.reshape(3) if arr.size == 3 else np.zeros(3)


@dataclass
class SweptCoverageReport:
    """Neutral payload for the ``coverage`` endpoint: the union as it stands."""

    fraction: float = 0.0  # of the sample set, covered by at least one evaluation
    n_points: int = 0  # size of the fixed sample set
    n_covered: int = 0  # how many of them have ever been covered
    covered_area_m2: float = UNKNOWN_AREA  # distinct grid cells covered x cell_area_m2
    sampled_area_m2: float = UNKNOWN_AREA  # the denominator the fraction of area is taken over
    cell_area_m2: float = 0.0  # resolution^2, so the two areas are reconstructible
    n_evaluations: int = 0  # FoV evaluations folded into the union since the last reset
    mean_visits: float = 0.0  # evaluations a point was covered in, averaged over the sample set
    sim_time: float = 0.0  # when this report was assembled


@dataclass
class SweptCoverageReader:
    """Blackboard handle under ``swept_coverage:<address>``; every call runs on the physics thread.

    ``points`` and ``visits`` are the accumulator itself rather than a summary of it, because the
    figure a consumer wants is rarely the scalar: a coverage-over-time curve, a revisit histogram and
    a map of what was missed all come from the same two arrays, and none of them is derivable from the
    fraction. ``visits`` is returned as a copy -- the live array is added to every evaluation, and a
    consumer holding it would watch its own snapshot change underneath it.
    """

    name: str
    read: Callable[[], SweptCoverageReport]
    points: Callable[[], np.ndarray]  # (P, 3) the fixed sample set, in world coordinates
    visits: Callable[[], np.ndarray]  # (P,) int -- evaluations each point was covered in


class SweptCoverageMonitorPlugin(Plugin):
    """See the module docstring."""

    #: ``post_step`` reads ``data`` and writes only this instance's own accumulator -- the condition
    #: ``contact_monitor`` and ``clearance_monitor`` declare this on. ``mj_multiRay`` fills
    #: caller-owned buffers and mutates nothing, and unlike ``sensor_coverage_probe`` there is no
    #: render here, so nothing forces this onto one thread. Declaring it False would be the expensive
    #: mistake: on a many-core lane it holds back every other parallel-safe observer in the world.
    parallel_safe = True

    #: NOT `requires_owner`: the mount is named explicitly, so there is no default to resolve from an
    #: entity and therefore none to resolve wrongly -- and a sensor may ride something that registers
    #: no entity at all (a moving prop, a gantry body in the world MJCF). Nesting it under a spawn is
    #: still the usual case and is what supplies the name `prefix`.
    requires_owner = False

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.sensor_type = str(self.config.get("type", ""))
        self.camera = str(self.config.get("camera", ""))
        self.site = str(self.config.get("site", ""))
        self.body = str(self.config.get("body", ""))
        # Same reason as `sample` below: read tolerantly so validate_config is the one that reports.
        self.offset = _triple(self.config.get("offset"))
        self.rpy = _triple(self.config.get("rpy"))
        self.sensor_config = dict(self.config.get("config") or {})

        # Construction runs BEFORE validate_config, so a mistyped block has to survive being read
        # here to be reported there -- otherwise the one hook that collects every problem into a
        # single report is bypassed by an AttributeError from the first one.
        sample = self.config.get("sample")
        sample = sample if isinstance(sample, dict) else {}
        self.sample_volume = bool(sample.get("volume", True))
        self.sample_objects = bool(sample.get("objects", False))
        self.resolution = float(sample.get("resolution", 0.25))
        heights = sample.get("heights", [0.5])
        self.heights = (
            tuple(float(h) for h in heights) if isinstance(heights, (list, tuple)) else ()
        )
        self.per_object = int(sample.get("per_object", 64))

        self.regions_spec = self.config.get("regions") or ""
        self.region_names = self.config.get("region_names") or []
        self.restrict = bool(self.config.get("restrict", False))

        self.compute_rate_hz = float(self.config.get("compute_rate_hz", 5.0))
        self.rate_hz = float(self.config.get("rate_hz", 2.0))
        self.out = str(self.config.get("out", ""))

        self._ctx: SimContext | None = None
        self._fov = None  # the SensorFov, built once and re-posed each evaluation
        self._mount_pose: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
        self._local_pos = np.zeros(3)  # sensor origin in the mount frame
        self._local_rot = np.eye(3)  # sensor rotation in the mount frame
        self._points = np.zeros((0, 3))
        self._labels = np.zeros(0, dtype=int)
        self._label_names: list[str] = []
        self._regions: list = []
        self._visits = np.zeros(0, dtype=np.int64)
        self._cell_of = np.zeros(0, dtype=np.int64)  # grid point -> distinct xy cell index
        self._grid_idx = np.zeros(0, dtype=np.int64)  # which sample points are grid points
        self._n_cells = 0
        self._mount_desc = ""  # filled once the mount resolves; used in the log and the report
        self._evaluations = 0
        self._next_due = 0.0

    # -- validation --------------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if not str(config.get("type", "")):
            # There is no sensible default: which adapter builds the FoV decides the whole geometry,
            # so guessing one would report coverage for a device the world does not carry.
            from ..coverage.adapters import registered_types

            errors.append(f"'type' is required; known sensor types: {registered_types()}")
        mounted = [key for key in _MOUNT_KEYS if str(config.get(key, ""))]
        if len(mounted) != 1:
            errors.append(
                f"exactly one of {list(_MOUNT_KEYS)} names the mount, got {mounted or 'none'}"
            )
        for key in ("compute_rate_hz", "rate_hz"):
            default = 5.0 if key == "compute_rate_hz" else 2.0
            if float(config.get(key, default)) <= 0:
                errors.append(f"'{key}' must be > 0")
        sample = config.get("sample") or {}
        if not isinstance(sample, dict):
            # Reported here rather than left to blow up in __init__: this hook exists to collect
            # every problem into one report, and an AttributeError from a mistyped block escapes it.
            errors.append("'sample' must be a mapping of volume/objects/resolution/heights")
            return errors
        if not (sample.get("volume", True) or sample.get("objects", False)):
            errors.append("'sample' must enable at least one of volume/objects")
        if float(sample.get("resolution", 0.25)) <= 0:
            errors.append("sample 'resolution' must be > 0")
        heights = sample.get("heights", [0.5])
        if not isinstance(heights, (list, tuple)) or not heights:
            errors.append("sample 'heights' must be a non-empty list of world z values")
        for key in ("offset", "rpy"):
            value = config.get(key)
            if value is None:
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                errors.append(f"'{key}' must be 3 numbers")
        if config.get("region_names") is not None and not isinstance(config["region_names"], list):
            errors.append("'region_names' must be a list of region names")
        return errors

    # -- lifecycle ---------------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        model, data = ctx.model, ctx.data
        # The sample set and the mount pose are both read out of `data`, and neither exists until a
        # forward pass has populated the world's geom/camera/site poses.
        mujoco.mj_forward(model, data)

        entity = ctx.entities.get(self.entity) if self.entity else None
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        self._build_fov(model, data, prefix)
        self._build_points(model, data)

        self._visits = np.zeros(len(self._points), dtype=np.int64)
        ctx.blackboard.set(
            f"swept_coverage:{self.address}",
            SweptCoverageReader(
                name=self.label,
                read=self.read,
                points=lambda: self._points,
                visits=lambda: self._visits.copy(),
            ),
        )
        ctx.interface.add(
            Endpoint(
                name="coverage",
                direction="out",
                owner=self.entity or "",
                namespace=ns,
                read=self.read,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        # Float32 and the running FRACTION: the series is the coverage-over-time
                        # curve a reader wants, and the final value is the last sample of it. The
                        # areas and the visit counts stay readable in-process, where an array can go.
                        "type": "std_msgs.msg.Float32",
                        "field": "fraction",
                        "topic": self.topic_override("coverage") or "coverage_fraction",
                    }
                },
            )
        )
        _log.info(
            "swept_coverage_monitor[%s]: %s FoV on %s, %d sample point(s), %d grid cell(s), "
            "evaluating at %.1f Hz",
            self.label,
            self.sensor_type,
            self._mount_desc,
            len(self._points),
            self._n_cells,
            self.compute_rate_hz,
        )

    # -- the mount ---------------------------------------------------------------------------------

    def _build_fov(self, model, data, prefix: str) -> None:
        """Resolve the mount and build the field of view once, at its current pose.

        Built once and re-posed per evaluation rather than rebuilt: an adapter re-reads the MJCF
        intrinsics (and, for a lidar type, re-instantiates the plugin whose defaults it borrows) on
        every call, and none of that changes while a sensor moves. What changes is the pose, and the
        pose is two array reads.
        """
        mounted = [key for key in _MOUNT_KEYS if getattr(self, key)]
        if len(mounted) != 1:
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: exactly one of {list(_MOUNT_KEYS)} names "
                f"the mount, got {mounted or 'none'}"
            )
        kind = mounted[0]
        name = prefix + getattr(self, kind)
        obj = {
            "camera": mujoco.mjtObj.mjOBJ_CAMERA,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "body": mujoco.mjtObj.mjOBJ_BODY,
        }[kind]
        mount_id = mujoco.mj_name2id(model, obj, name)
        if mount_id < 0:
            # Loudly, for the reason contact_monitor and clearance_monitor refuse: a monitor whose
            # mount does not exist would report zero coverage for the whole run, which is
            # indistinguishable from a sensor that saw nothing and would quietly pass a campaign.
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: {kind} {name!r} not found in the compiled "
                f"world. Names are resolved with the owning entity's prefix "
                f"({prefix!r}); check the spelling, or nest this entry under the spawn that "
                f"carries the sensor."
            )
        self._mount_desc = f"{kind} {name!r}"

        if kind == "camera":
            placed = PlacedSensor(self.sensor_type, cam_id=mount_id, config=self.sensor_config)
            self._mount_pose = lambda i=mount_id: (
                self._ctx.data.cam_xpos[i],
                self._ctx.data.cam_xmat[i].reshape(3, 3),
            )
        elif kind == "site":
            placed = PlacedSensor(self.sensor_type, site_id=mount_id, config=self.sensor_config)
            self._mount_pose = lambda i=mount_id: (
                self._ctx.data.site_xpos[i],
                self._ctx.data.site_xmat[i].reshape(3, 3),
            )
        else:
            # A body mount is the adapter's *hypothetical* placement form: offset/rpy are read in the
            # mount frame, so the adapter's own conventions (a camera's optical axis, a lidar's
            # boresight) are inherited rather than restated here.
            placed = PlacedSensor(
                self.sensor_type, pos=self.offset, rpy=self.rpy, config=self.sensor_config
            )
            self._mount_pose = lambda i=mount_id: (
                self._ctx.data.xpos[i],
                self._ctx.data.xmat[i].reshape(3, 3),
            )
        placed.label = self.label

        try:
            self._fov = build_fov(model, data, placed)
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: cannot build a {self.sensor_type!r} field "
                f"of view on a {kind} mount: {exc}"
            ) from exc

        # The sensor's pose in the mount frame. For a camera/site mount the adapter already resolved
        # the frame itself, so the sensor sits at its origin; for a body mount it is the configured
        # offset and the rotation the adapter derived from `rpy`.
        if kind == "body":
            self._local_pos = np.asarray(self._fov.origin, dtype=np.float64).copy()
            self._local_rot = np.asarray(self._fov.rot, dtype=np.float64).copy()
            # A hypothetical placement excludes nothing, but a mounted sensor's origin sits inside
            # the housing it is bolted to -- the same correction the adapters make for a camera/site
            # mount via SensorFov.body_exclude.
            self._fov.body_exclude = mount_id
        else:
            self._local_pos = np.zeros(3)
            self._local_rot = np.eye(3)

    def _repose(self) -> None:
        """Move the field of view to the mount's pose as it stands. Physics thread only."""
        pos, mat = self._mount_pose()
        self._fov.origin = np.asarray(pos, dtype=np.float64) + mat @ self._local_pos
        self._fov.rot = mat @ self._local_rot

    # -- the sample set ----------------------------------------------------------------------------

    def _build_points(self, model, data) -> None:
        points, labels, names = sampling.sample_set(
            model,
            data,
            volume=self.sample_volume,
            objects=self.sample_objects,
            resolution=self.resolution,
            heights=self.heights,
            per_object=self.per_object,
        )
        if len(points) == 0:
            # A coverage fraction over zero points is the failure this refuses: `0/0` reported as a
            # number, or a fraction of 1.0 over an empty set, either of which reads as a finished
            # measurement. An enclosed room is what the volume sampler needs; a world with no walls
            # yields no interior points.
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: the sample set is empty, so there is "
                f"nothing to accumulate coverage over. The volume sampler keeps only enclosed "
                f"free-space points, so an unwalled world yields none at these heights "
                f"({list(self.heights)}); enable 'objects', add heights inside the room, or "
                f"coarsen 'resolution'."
            )

        self._regions = self._load_regions()
        if self._regions and self.restrict:
            from ..coverage.regions import union_mask

            mask = union_mask(points, self._regions)
            if not mask.any():
                raise RuntimeError(
                    f"swept_coverage_monitor[{self.label}]: 'restrict' left no sample points -- "
                    f"none of the {len(points)} points fall inside regions "
                    f"{[r.name for r in self._regions]}."
                )
            points, labels = points[mask], labels[mask]

        self._points = np.ascontiguousarray(points)
        self._labels = labels
        self._label_names = names
        self._build_cells()

    def _load_regions(self) -> list:
        if not self.regions_spec:
            return []
        from ..coverage import regions as regionsmod

        spec = self.regions_spec
        if isinstance(spec, str):
            path = Path(spec)
            # A path in a world document names a file beside that document, not beside the CWD.
            spec = str(path if path.is_absolute() else (self.base_dir / path))
        regs = regionsmod.load_regions(spec)
        if self.region_names:
            regs = regionsmod.select(regs, self.region_names)
        if not regs:
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: 'regions' produced no regions, so a "
                f"per-region breakdown would be silently empty."
            )
        return regs

    def _build_cells(self) -> None:
        """Index the grid points by their xy cell, which is what makes the area figure an area.

        Only the volume grid takes part: a grid point stands for one ``resolution``-sized piece of
        ground, and an object-surface point stands for a piece of a surface with no footprint. Several
        height layers share one xy cell, so a column counts once however many heights were sampled.
        """
        self._grid_idx = np.nonzero(self._labels < 0)[0]
        if self._grid_idx.size == 0:
            self._cell_of = np.zeros(0, dtype=np.int64)
            self._n_cells = 0
            return
        cells = np.floor(self._points[self._grid_idx, :2] / self.resolution).astype(np.int64)
        unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
        self._cell_of = np.asarray(inverse).reshape(-1)
        self._n_cells = len(unique_cells)

    # -- the union ---------------------------------------------------------------------------------

    def on_reset(self, ctx: SimContext) -> None:
        # A trial's swept coverage is that trial's. Carried across a reset, trial 2 of a process
        # serving several would start from trial 1's union and report a sweep it never made.
        self._visits = np.zeros(len(self._points), dtype=np.int64)
        self._evaluations = 0
        # A reset is a new clock, so the old due time would skip the start of the run by however far
        # `data.time` had moved.
        self._next_due = 0.0

    def post_step(self, ctx: SimContext) -> None:
        if ctx.sim_time < self._next_due:
            return
        self._next_due = ctx.sim_time + 1.0 / self.compute_rate_hz
        self._repose()
        result = coverage(ctx.model, ctx.data, [self._fov], self._points)
        # The accumulation: a point covered in this evaluation gains a visit, and the union is
        # `visits > 0`. Monotone by construction -- nothing here ever subtracts.
        self._visits += result.by_sensor[:, 0]
        self._evaluations += 1

    def read(self) -> SweptCoverageReport:
        """The union as it stands. Runs on the physics thread."""
        covered = self._visits > 0
        n = len(self._points)
        covered_area = sampled_area = UNKNOWN_AREA
        cell_area = 0.0
        if self._n_cells:
            cell_area = self.resolution * self.resolution
            hit_cells = np.bincount(self._cell_of[covered[self._grid_idx]], minlength=self._n_cells)
            covered_area = float(np.count_nonzero(hit_cells) * cell_area)
            sampled_area = float(self._n_cells * cell_area)
        return SweptCoverageReport(
            fraction=float(np.mean(covered)) if n else 0.0,
            n_points=n,
            n_covered=int(covered.sum()),
            covered_area_m2=covered_area,
            sampled_area_m2=sampled_area,
            cell_area_m2=cell_area,
            n_evaluations=self._evaluations,
            mean_visits=float(np.mean(self._visits)) if n else 0.0,
            sim_time=self._ctx.sim_time if self._ctx else 0.0,
        )

    # -- the optional file -------------------------------------------------------------------------

    def shutdown(self, ctx: SimContext) -> None:
        if not self.out or self._fov is None:
            return
        report = self.build_report()
        out = Path(self.out)
        if not out.is_absolute():
            out = self.base_dir / out
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        _log.info(
            "swept_coverage_monitor[%s]: swept %.3f of %d point(s) over %d evaluation(s) -> %s",
            self.label,
            report["swept"]["fraction"],
            report["swept"]["n_points"],
            report["swept"]["n_evaluations"],
            out / "report.json",
        )

    def build_report(self) -> dict:
        """The accumulated union as the same report shape a static coverage estimate produces.

        The union is handed to :func:`~roqsim_sensors.coverage.report.build_report` as a one-sensor
        result, so ``uncovered_regions`` clusters exactly what the sweep never reached and
        ``per_object`` says which objects were never seen -- the two questions a swept figure raises.
        A ``swept`` block carries what a static estimate has no field for: the areas, the evaluation
        count and the visit distribution.
        """
        from ..coverage.engine import CoverageResult

        covered = self._visits > 0
        result = CoverageResult(
            points=self._points,
            counts=covered.astype(int),
            by_sensor=covered.reshape(-1, 1),
            fovs=[self._fov],
            labels=self._labels,
            label_names=self._label_names,
        )
        per_region = None
        if self._regions:
            from ..coverage.regions import per_region_coverage

            per_region = per_region_coverage(self._points, result.counts, self._regions)
        report = build_report(
            result,
            world=str((self._ctx.config.get("sim", {}) if self._ctx else {}).get("world", "")),
            gap_resolution=self.resolution,
            per_region=per_region,
        )
        current = self.read()
        report["swept"] = {
            "mount": self._mount_desc,
            "sensor_type": self.sensor_type,
            "fraction": current.fraction,
            "n_points": current.n_points,
            "n_covered": current.n_covered,
            "covered_area_m2": current.covered_area_m2,
            "sampled_area_m2": current.sampled_area_m2,
            "cell_area_m2": current.cell_area_m2,
            "n_evaluations": current.n_evaluations,
            "compute_rate_hz": self.compute_rate_hz,
            "mean_visits": current.mean_visits,
            "max_visits": int(self._visits.max()) if len(self._visits) else 0,
            "sim_time": current.sim_time,
        }
        return report
