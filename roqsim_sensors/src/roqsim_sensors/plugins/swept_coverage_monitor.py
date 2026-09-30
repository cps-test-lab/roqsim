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

Its keys are declared in :attr:`SweptCoverageMonitorPlugin.CONFIG_SCHEMA`. ``frame`` is the mount: a
frame path (:mod:`roqsim.paths`, resolved by :func:`roqsim.frames.resolve_frame`) -- an entity's root,
one of its bodies, sites, cameras, declared or device frames, or a body, site or camera of the world's
own MJCF by its MuJoCo name (``gantry``). Nested under an entity the path is relative to it (``.`` is
the entity, ``oakd/oakd_rgb`` its OAK-D's camera) and a leading ``/`` starts at the top of the world;
at the top of a world it starts with an entity's name, and the entry is declared after the entry that
spawns it. The field of view sits at the frame, moved by ``pose``, a ``geometry_msgs/Pose`` in the
frame's coordinates whose omitted components are 0. A camera frame gives a camera-type field of view
its intrinsics and its pose -- MuJoCo's camera frame, looking along -z with +y up -- with ``config``
overriding ``fovy``/``width``/``height`` and setting the range; on any other frame ``config`` states
the whole field of view, boresight along the frame's +x. ``type`` names the coverage adapter that
builds it (any of ``roqsim sensors coverage catalog``'s types); ``sample`` is the fixed point set the
union is accumulated over, and ``regions``/``region_names``/``restrict`` limit it to named regions.

Endpoint ``coverage`` (out) reads a :class:`SweptCoverageReport`; ROS carries its ``fraction`` as a
``std_msgs/Float32`` on ``coverage_fraction``. The report holds the covered fraction of the sample
set, the covered and sampled **areas** with the cell they are derived from, how many evaluations
went into the union, and the mean number of evaluations a point was seen in -- so a revisit figure
falls out of the same accumulator as a coverage one. A :class:`SweptCoverageReader` on the blackboard under
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
from typing import Annotated

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.endpoint import Unit
from roqsim.frames import Frame, resolve_frame
from roqsim.paths import PathError
from roqsim.plugin import Plugin
from roqsim.pose import PoseError, parse_pose
from roqsim.schema import Field
from roqsim.types import Duration

from ..coverage import sampling
from ..coverage.adapters import PlacedSensor, build_fov
from ..coverage.engine import coverage as evaluate_coverage
from ..coverage.report import build_report

_log = logging.getLogger(__name__)

#: Areas are unreportable without a volume grid; ``-1.0`` says so rather than reading as zero.
UNKNOWN_AREA = -1.0


#: An area, in square metres.
Area = Annotated[float, Unit("m^2")]


@dataclass
class SweptCoverageReport:
    """Neutral payload for the ``coverage`` endpoint: the union as it stands.

    Attributes:
        fraction: of the sample set, covered by at least one evaluation
        n_points: size of the fixed sample set
        n_covered: how many of them have ever been covered
        covered_area_m2: distinct grid cells covered times ``cell_area_m2``; -1.0 without a grid
        sampled_area_m2: the area the fraction of area is taken over; -1.0 without a grid
        cell_area_m2: the grid pitch squared, so the two areas are reconstructible
        n_evaluations: field-of-view evaluations folded into the union since the last reset
        mean_visits: evaluations a point was covered in, averaged over the sample set
        sim_time: when this report was assembled
    """

    fraction: float = 0.0
    n_points: int = 0
    n_covered: int = 0
    covered_area_m2: Area = UNKNOWN_AREA
    sampled_area_m2: Area = UNKNOWN_AREA
    cell_area_m2: Area = 0.0
    n_evaluations: int = 0
    mean_visits: float = 0.0
    sim_time: Duration = 0.0


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _carrier(model, frame: Frame) -> int:
    """The body *frame* rides on, for ``SensorFov.body_exclude``; ``-1`` for the world body."""
    if frame.kind in ("root", "body"):
        body = frame.index
    elif frame.is_camera:
        body = int(model.cam_bodyid[frame.index])
    else:
        body = int(model.site_bodyid[frame.index])
    return body if body > 0 else -1


def _pose_reader(ctx: SimContext, frame: Frame) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """``(position, rotation matrix)`` of *frame* in the world, read from ``data`` as it stands.

    The arrays :func:`roqsim.frames.frame_pose` reads, read directly: an entity's root is its body,
    and the monitor needs the pose at ``configure`` too, before the entity pose endpoints exist.
    """
    i = frame.index
    if frame.kind in ("root", "body"):
        return lambda: (ctx.data.xpos[i], ctx.data.xmat[i].reshape(3, 3))
    if frame.is_camera:
        return lambda: (ctx.data.cam_xpos[i], ctx.data.cam_xmat[i].reshape(3, 3))
    return lambda: (ctx.data.site_xpos[i], ctx.data.site_xmat[i].reshape(3, 3))


class SweptCoverageMonitorPlugin(Plugin):
    """See the module docstring."""

    #: ``post_step`` reads ``data`` and writes only this instance's own accumulator -- the condition
    #: ``contact_monitor`` and ``clearance_monitor`` declare this on. ``mj_multiRay`` fills
    #: caller-owned buffers and mutates nothing, and unlike ``sensor_coverage_probe`` there is no
    #: render here, so nothing forces this onto one thread. Declaring it False would be the expensive
    #: mistake: on a many-core lane it holds back every other parallel-safe observer in the world.
    parallel_safe = True

    #: NOT `requires_owner`: the mount is a frame path, which a top-level entry states absolutely --
    #: and a sensor may ride something that registers no entity at all (a gantry body in the world
    #: MJCF). Nesting it under a spawn is still the usual case, and makes the path relative to it.
    requires_owner = False

    #: Declared once, so ``roqsim plugins describe swept_coverage_monitor`` publishes the keys the
    #: checks run on. Everything but the two rates is read once, at configure.
    CONFIG_SCHEMA = {
        "type": Field(
            str,
            required=True,
            static=True,
            doc="sensor type: the coverage adapter that builds the field of view "
            "(any of `roqsim sensors coverage catalog`'s types)",
        ),
        "frame": Field(
            str,
            required=True,
            static=True,
            doc="mount: a frame path; relative to the owning entity when nested, '/' for absolute",
        ),
        "pose": Field(
            dict,
            default={},
            static=True,
            doc="the sensor in the frame's coordinates, a geometry_msgs/Pose (omitted components "
            "are 0)",
        ),
        "config": Field(
            dict,
            default={},
            static=True,
            doc="field-of-view parameters for the adapter (fovy/width/height/far, range_max, ...); "
            "on a camera frame, overrides of the camera's own intrinsics",
        ),
        "sample": Field(
            dict,
            schema={
                "volume": Field(
                    bool,
                    default=True,
                    doc="free-interior grid at 'resolution', one layer per height",
                ),
                "objects": Field(bool, default=False, doc="object surfaces, labelled per geom"),
                "resolution": Field(
                    float,
                    default=0.25,
                    unit="m",
                    doc="grid pitch; also the cell the area figure is derived from",
                ),
                "heights": Field(list, default=[0.5], unit="m", doc="world z of the grid layers"),
                "per_object": Field(int, default=64, minimum=1, doc="surface points per object"),
            },
            static=True,
            doc="the fixed point set the union is accumulated over",
        ),
        "regions": Field(
            (str, dict, list),
            default="",
            static=True,
            doc="named regions: a JSON path beside the world, an inline spec, or a floorplan",
        ),
        "region_names": Field(list, default=[], static=True, doc="subset of 'regions' to keep"),
        "restrict": Field(
            bool, default=False, static=True, doc="sample only inside the regions' union"
        ),
        "compute_rate_hz": Field(
            float, default=5.0, unit="Hz", doc="how often the field of view is evaluated"
        ),
        "rate_hz": Field(float, default=2.0, unit="Hz", doc="how often the endpoint publishes"),
        "out": Field(
            str,
            default="",
            static=True,
            doc="directory for a report.json written at shutdown, beside the world when relative",
        ),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        # The endpoint's publish rate is named by attribute; the rest is read at configure, after
        # the config was checked.
        self.rate_hz = self.settings.rate_hz
        self.sensor_type = ""
        self._ctx: SimContext | None = None
        self._fov = None  # the SensorFov, built once and re-posed each evaluation
        self._mount_pose: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
        self._local_pos = np.zeros(3)  # sensor origin in the mount frame
        self._local_rot = np.eye(3)  # sensor rotation in the mount frame
        self._fov_start = (np.zeros(3), np.eye(3))  # the FoV's pose at configure
        self._resolution = 0.25
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
        self._period = 0.2
        self._next_due = 0.0

    # -- validation --------------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        # Types, defaults and unknown keys come from CONFIG_SCHEMA; this checks what it cannot say.
        errors = self.validate_topics(config)
        settings = self.settings_for(config)
        if settings.type == "":
            # There is no sensible default: which adapter builds the FoV decides the whole geometry,
            # so guessing one would report coverage for a device the world does not carry.
            from ..coverage.adapters import registered_types

            errors.append(f"'type' is required; known sensor types: {registered_types()}")
        if settings.frame == "":
            errors.append("'frame' is required: the frame path the sensor is mounted on")
        if isinstance(settings.pose, dict):
            try:
                parse_pose(settings.pose, relative=True)
            except PoseError as exc:
                errors.append(str(exc))
        for key, value in (
            ("compute_rate_hz", settings.compute_rate_hz),
            ("rate_hz", settings.rate_hz),
        ):
            if _is_number(value) and value <= 0:
                errors.append(f"'{key}' must be > 0")
        if not isinstance(config.get("sample", {}), dict):
            return errors  # the schema reports the type; the block's rules need a mapping
        sample = settings.sample
        if not (sample.volume or sample.objects):
            errors.append("'sample' must enable at least one of volume/objects")
        if _is_number(sample.resolution) and sample.resolution <= 0:
            errors.append("sample 'resolution' must be > 0")
        if isinstance(sample.heights, list) and not sample.heights:
            errors.append("sample 'heights' must be a non-empty list of world z values")
        return errors

    # -- lifecycle ---------------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        settings = self.settings
        self.sensor_type = settings.type
        self._resolution = settings.sample.resolution
        self._period = 1.0 / settings.compute_rate_hz
        model, data = ctx.model, ctx.data
        # The sample set and the mount pose are both read out of `data`, and neither exists until a
        # forward pass has populated the world's geom/camera/site poses.
        mujoco.mj_forward(model, data)

        self._build_fov(ctx)
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
        _log.info(
            "swept_coverage_monitor[%s]: %s FoV on %s, %d sample point(s), %d grid cell(s), "
            "evaluating at %.1f Hz",
            self.label,
            self.sensor_type,
            self._mount_desc,
            len(self._points),
            self._n_cells,
            settings.compute_rate_hz,
        )

    # Float32 and the running FRACTION: the series is the coverage-over-time curve a reader wants,
    # and the final value is the last sample of it. The areas and the visit counts stay readable
    # in-process, where an array can go.
    @endpoint.out(
        rate="rate_hz",
        ros2={"type": "std_msgs.msg.Float32", "field": "fraction", "topic": "coverage_fraction"},
    )
    def coverage(self) -> SweptCoverageReport:
        """The union as it stands: the covered fraction and area, and how it was accumulated."""
        return self.read()

    # -- the mount ---------------------------------------------------------------------------------

    def _build_fov(self, ctx: SimContext) -> None:
        """Resolve the mount frame and build the field of view once, at its current pose.

        Built once and re-posed per evaluation rather than rebuilt: an adapter re-reads the MJCF
        intrinsics (and, for a lidar type, re-instantiates the plugin whose defaults it borrows) on
        every call, and none of that changes while a sensor moves. What changes is the pose, and the
        pose is two array reads.
        """
        settings = self.settings
        model, data = ctx.model, ctx.data
        try:
            frame = resolve_frame(ctx, settings.frame, within=self.entity or None)
        except PathError as exc:
            # Loudly, for the reason contact_monitor and clearance_monitor refuse: a monitor whose
            # mount does not exist would report zero coverage for the whole run, which is
            # indistinguishable from a sensor that saw nothing and would quietly pass a campaign.
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: 'frame' {settings.frame!r}: {exc}"
            ) from exc
        self._mount_desc = f"frame {frame.path!r}"
        self._mount_pose = _pose_reader(ctx, frame)

        if frame.is_camera:
            # The adapter reads the camera's intrinsics and places the field of view at the camera
            # frame itself, so the sensor sits at the frame's origin before `pose` moves it.
            placed = PlacedSensor(self.sensor_type, cam_id=frame.index, config=settings.config)
        else:
            # The adapter's hypothetical placement form, at the origin with no rotation, so its own
            # conventions (a camera's optical axis, a lidar's boresight) are inherited rather than
            # restated here; the frame's pose is applied on top below, as the adapter applies a
            # placement's own (rot = R @ base).
            placed = PlacedSensor(
                self.sensor_type, pos=np.zeros(3), rpy=np.zeros(3), config=settings.config
            )
        placed.label = self.label
        try:
            self._fov = build_fov(model, data, placed)
        except (KeyError, ValueError) as exc:
            where = "camera frame" if frame.is_camera else "frame"
            raise RuntimeError(
                f"swept_coverage_monitor[{self.label}]: cannot build a {self.sensor_type!r} field "
                f"of view on {where} {frame.path!r}: {exc}"
            ) from exc

        # The field of view in the frame's coordinates: its base pose there, moved by `pose`.
        if frame.is_camera:
            base_pos, base_rot = np.zeros(3), np.eye(3)
        else:
            base_pos = np.asarray(self._fov.origin, dtype=np.float64).copy()
            base_rot = np.asarray(self._fov.rot, dtype=np.float64).copy()
        position, quat = parse_pose(settings.pose, relative=True)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=np.float64))
        rotation = rotation.reshape(3, 3)
        self._local_pos = np.asarray(position, dtype=np.float64) + rotation @ base_pos
        self._local_rot = rotation @ base_rot
        # A mounted sensor's origin sits inside the housing it is bolted to, so the carrying body is
        # excluded from its raycasts -- unless that is the world body, whose geometry is the walls.
        self._fov.body_exclude = _carrier(model, frame)
        # Posed at the mount as configured, which is where a reset puts it back.
        self._repose()
        self._fov_start = (self._fov.origin.copy(), self._fov.rot.copy())

    def _repose(self) -> None:
        """Move the field of view to the mount's pose as it stands. Physics thread only."""
        pos, mat = self._mount_pose()
        self._fov.origin = np.asarray(pos, dtype=np.float64) + mat @ self._local_pos
        self._fov.rot = mat @ self._local_rot

    # -- the sample set ----------------------------------------------------------------------------

    def _build_points(self, model, data) -> None:
        sample = self.settings.sample
        heights = tuple(float(h) for h in sample.heights)
        points, labels, names = sampling.sample_set(
            model,
            data,
            volume=sample.volume,
            objects=sample.objects,
            resolution=sample.resolution,
            heights=heights,
            per_object=sample.per_object,
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
                f"({list(heights)}); enable 'objects', add heights inside the room, or "
                f"coarsen 'resolution'."
            )

        self._regions = self._load_regions()
        if self._regions and self.settings.restrict:
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
        settings = self.settings
        if not settings.regions:
            return []
        from ..coverage import regions as regionsmod

        spec = settings.regions
        if isinstance(spec, str):
            path = Path(spec)
            # A path in a world document names a file beside that document, not beside the CWD.
            spec = str(path if path.is_absolute() else (self.base_dir / path))
        regs = regionsmod.load_regions(spec)
        if settings.region_names:
            regs = regionsmod.select(regs, settings.region_names)
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
        cells = np.floor(self._points[self._grid_idx, :2] / self._resolution).astype(np.int64)
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
        if self._fov is not None:
            self._fov.origin, self._fov.rot = (a.copy() for a in self._fov_start)

    def post_step(self, ctx: SimContext) -> None:
        # A thousandth of a step short still counts: float drift in the summed clock would otherwise
        # push a period the timestep divides to the step after it.
        if ctx.sim_time < self._next_due - 1e-3 * ctx.model.opt.timestep:
            return
        self._next_due = ctx.sim_time + self._period
        self._repose()
        result = evaluate_coverage(ctx.model, ctx.data, [self._fov], self._points)
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
            cell_area = self._resolution * self._resolution
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
        if not self.settings.out or self._fov is None:
            return
        report = self.build_report()
        out = Path(self.settings.out)
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
            gap_resolution=self._resolution,
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
            "compute_rate_hz": self.settings.compute_rate_hz,
            "mean_visits": current.mean_visits,
            "max_visits": int(self._visits.max()) if len(self._visits) else 0,
            "sim_time": current.sim_time,
        }
        return report
