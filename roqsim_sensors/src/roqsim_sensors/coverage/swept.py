"""Swept coverage: the union of a moving sensor's field of view over a recorded run.

The static estimate asks what a sensor set sees from where it stands. A sensor carried through a
world asks what it ever saw on the way, and that is answered here from a recording
(:func:`roqsim.recording.open_recording`): each recorded sample restores the full MuJoCo state -- the
carrier where it was, and every moving occluder where it was -- the field of view is placed at the
mount frame's pose in that state, and the same range -> angular -> raycast gate as the static
estimate (:func:`~roqsim_sensors.coverage.engine.coverage`) is ORed into a union over one fixed
sample set (:func:`~roqsim_sensors.coverage.sampling.sample_set`). Because the run is already
recorded, the sensor, its mount frame and its range are chosen afterwards, and one run answers any
number of them.

**The mount is a frame path** (:func:`roqsim.frames.resolve_frame`): an entity's root, body, site,
camera or declared frame, or a body, site or camera of the world's own MJCF by its MuJoCo name
(``gantry``). ``pose`` offsets the sensor in that frame's coordinates. On a camera frame the field
of view is that camera's: its intrinsics come from the model and its axes are MuJoCo's camera frame
(looking along -z, +y up); the catalog's intrinsics for the type are not applied, and ``config``
overrides either. On any other frame the catalog entry and ``config`` state the field of view,
placed along the frame's +x as the adapter's hypothetical form places it.

**The union is a lower bound.** The field of view is evaluated at the recorded samples only, or at
a lower stated rate, and never interpolated between them: an interpolated field of view has no
line-of-sight test behind it and would report coverage through walls. Whatever the sensor swept
between two samples is not counted. The sample set is built once, from the state at the first
evaluated sample, so a cell occupied then -- by the carrier, or by an occluder -- is never sampled;
a point wrongly dropped only makes coverage look worse. A raycast excludes the body the frame rides
on, unless that is the world body, whose geometry is the walls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.frames import Frame, frame_pose, resolve_frame
from roqsim.paths import PathError
from roqsim.pose import PoseError, parse_pose

from .adapters import PlacedSensor, build_fov
from .catalog import CATALOG
from .engine import CoverageResult, coverage
from .fov import SensorFov

#: The keys a camera frame supplies itself; the catalog's values for them are not applied there.
CAMERA_INTRINSICS = ("fovy", "width", "height")

#: Areas are unreportable without a volume grid; ``-1.0`` says so rather than reading as zero.
UNKNOWN_AREA = -1.0

#: ``sample(model, data) -> (points, labels, label_names)``: the fixed point set, built once.
Sampler = Callable[[mujoco.MjModel, mujoco.MjData], tuple[np.ndarray, np.ndarray, list[str]]]


class SweptError(ValueError):
    """An input the sweep cannot be computed from: a frame, a window, a rate or a sample set."""


@dataclass
class SweptResult:
    """The accumulated union.

    Attributes:
        frame: the mount frame's path
        sensor_type: the catalog type the field of view was built for
        fov: the field of view, posed at the last evaluated sample
        points: (P, 3) the fixed sample set, world coordinates
        labels: (P,) object label per point, ``-1`` for a volume grid point
        label_names: label index -> object name
        visits: (P,) evaluations each point was covered in
        times: sim time of each evaluated sample
    """

    frame: str
    sensor_type: str
    fov: SensorFov
    points: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    visits: np.ndarray
    times: np.ndarray

    @property
    def covered(self) -> np.ndarray:
        return self.visits > 0

    def as_coverage(self) -> CoverageResult:
        """The union as a one-sensor coverage result, the shape the static report is built from."""
        covered = self.covered
        return CoverageResult(
            points=self.points,
            counts=covered.astype(int),
            by_sensor=covered.reshape(-1, 1),
            fovs=[self.fov],
            labels=self.labels,
            label_names=self.label_names,
        )

    def summary(self, resolution: float) -> dict:
        """The figures a static estimate has no field for: areas, evaluations, visit distribution.

        The areas come from the volume grid only, because only a grid point stands for a known piece
        of ground: ``covered_area_m2`` counts the distinct ``resolution``-sized xy cells covered at any
        height, so several height layers do not multiply the area. Without grid points both read
        ``-1.0``.
        """
        covered = self.covered
        n = len(self.points)
        grid = np.nonzero(self.labels < 0)[0]
        covered_area = sampled_area = UNKNOWN_AREA
        cell_area = 0.0
        if grid.size:
            cells = np.floor(self.points[grid, :2] / resolution).astype(np.int64)
            unique, inverse = np.unique(cells, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            cell_area = resolution * resolution
            hit = np.bincount(inverse[covered[grid]], minlength=len(unique))
            covered_area = float(np.count_nonzero(hit) * cell_area)
            sampled_area = float(len(unique) * cell_area)
        return {
            "frame": self.frame,
            "sensor_type": self.sensor_type,
            "fraction": float(np.mean(covered)) if n else 0.0,
            "n_points": n,
            "n_covered": int(covered.sum()),
            "covered_area_m2": covered_area,
            "sampled_area_m2": sampled_area,
            "cell_area_m2": cell_area,
            "n_evaluations": len(self.times),
            "from": float(self.times[0]) if len(self.times) else None,
            "to": float(self.times[-1]) if len(self.times) else None,
            "mean_visits": float(np.mean(self.visits)) if n else 0.0,
            "max_visits": int(self.visits.max()) if n else 0,
            # visits_histogram[v] points were covered in exactly v evaluations.
            "visits_histogram": np.bincount(self.visits).tolist() if n else [],
        }


def fov_config(sensor_type: str, config: dict | None, *, camera_frame: bool) -> dict:
    """The catalog entry's field-of-view config for *sensor_type*, overridden by *config*.

    On a camera frame the catalog's intrinsics are left out: the camera the frame names has its own.
    """
    if sensor_type not in CATALOG:
        raise SweptError(f"unknown sensor type {sensor_type!r}; catalog has {sorted(CATALOG)}")
    template = dict(CATALOG[sensor_type].fov_template)
    if camera_frame:
        for key in CAMERA_INTRINSICS:
            template.pop(key, None)
    template.update(config or {})
    return template


def select_samples(
    times: np.ndarray, fps: float, start: float | None, stop: float | None, rate: float | None
) -> np.ndarray:
    """Indices of the samples evaluated: those in ``[start, stop]``, thinned to *rate* if given.

    A rate above the recording's is refused, because the samples it would need do not exist and
    evaluating the same state twice would only inflate the visit counts.
    """
    t0, t1 = float(times[0]), float(times[-1])
    if rate is not None:
        if rate <= 0:
            raise SweptError(f"--rate {rate:g}: a rate must be > 0")
        if rate > fps * (1 + 1e-9):
            raise SweptError(
                f"--rate {rate:g} Hz is above the recording's {fps:g} samples per second; the "
                "union can be evaluated only at recorded states."
            )
    half = 0.5 / fps
    lo = -np.inf if start is None else start - 1e-9
    hi = np.inf if stop is None else stop + 1e-9
    window = np.nonzero((times >= lo) & (times <= hi))[0]
    if window.size == 0:
        raise SweptError(
            f"the window {_window(start, stop)} holds no sample: the recording spans "
            f"{t0:.3f}..{t1:.3f} s of sim time at {fps:g} samples per second ({len(times)} samples)."
        )
    if rate is None:
        return window
    period = 1.0 / rate
    chosen, due = [], -np.inf
    for i in window:
        if times[i] >= due - half:
            chosen.append(int(i))
            due = float(times[i]) + period
    return np.asarray(chosen, dtype=np.int64)


def _window(start: float | None, stop: float | None) -> str:
    first = "the start" if start is None else f"{start:g} s"
    last = "the end" if stop is None else f"{stop:g} s"
    return f"from {first} to {last}"


def _carrier(model, frame: Frame) -> int:
    """The body *frame* rides on, for ``SensorFov.body_exclude``; ``-1`` for the world body."""
    if frame.kind in ("root", "body"):
        body = frame.index
    elif frame.is_camera:
        body = int(model.cam_bodyid[frame.index])
    else:
        body = int(model.site_bodyid[frame.index])
    return body if body > 0 else -1


def _rotation(quat) -> np.ndarray:
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def swept_coverage(
    recording,
    *,
    frame: str,
    sensor_type: str,
    sample: Sampler,
    config: dict | None = None,
    pose: dict | None = None,
    start: float | None = None,
    stop: float | None = None,
    rate: float | None = None,
) -> SweptResult:
    """The union of the field of view on *frame* over *recording*'s samples in ``[start, stop]``.

    *recording* is an open :class:`roqsim.recording.Recording`; its world is rebuilt here. *sample*
    builds the fixed point set from the state at the first evaluated sample.
    """
    model, ctx = recording.build()
    if ctx is None:
        raise SweptError(
            f"{recording.path} is a recording of a mesh preview, which has no frames to mount on"
        )
    times = np.asarray(recording.times, dtype=np.float64)
    fps = float(recording.fps)
    chosen = select_samples(times, fps, start, stop, rate)
    try:
        offset_pos, offset_quat = parse_pose(pose or {}, relative=True)
    except PoseError as exc:
        raise SweptError(f"--pose: {exc}") from None

    first = recording.at(float(times[chosen[0]]))
    try:
        mount = resolve_frame(ctx, frame)
    except PathError as exc:
        raise SweptError(f"--frame {frame!r}: {exc}") from None

    config = fov_config(sensor_type, config, camera_frame=mount.is_camera)
    if mount.is_camera:
        # The adapter reads the camera's intrinsics and places the field of view at the camera frame
        # itself, so the sensor sits at the frame's origin before the offset moves it.
        placed = PlacedSensor(sensor_type, cam_id=mount.index, config=config, label=mount.path)
    else:
        # The adapter's hypothetical form, at the origin with no rotation, so its own conventions (a
        # camera's optical axis, a lidar's boresight) are inherited; the frame's pose goes on top.
        placed = PlacedSensor(
            sensor_type, pos=np.zeros(3), rpy=np.zeros(3), config=config, label=mount.path
        )
    try:
        fov = build_fov(model, first.data, placed)
    except (KeyError, ValueError) as exc:
        where = "camera frame" if mount.is_camera else "frame"
        raise SweptError(
            f"cannot build a {sensor_type!r} field of view on {where} {mount.path!r}: {exc}"
        ) from None
    if mount.is_camera:
        base_pos, base_rot = np.zeros(3), np.eye(3)
    else:
        base_pos = np.asarray(fov.origin, dtype=np.float64).copy()
        base_rot = np.asarray(fov.rot, dtype=np.float64).copy()
    offset_rot = _rotation(offset_quat)
    local_pos = np.asarray(offset_pos, dtype=np.float64) + offset_rot @ base_pos
    local_rot = offset_rot @ base_rot
    fov.body_exclude = _carrier(model, mount)

    points, labels, names = sample(model, first.data)
    if len(points) == 0:
        raise SweptError(
            "the sample set is empty, so there is nothing to accumulate coverage over. The volume "
            "sampler keeps only enclosed free-space points; check the sample heights and the world."
        )
    points = np.ascontiguousarray(points)
    visits = np.zeros(len(points), dtype=np.int64)
    evaluated = []
    for index in chosen:
        state = recording.at(float(times[index]))
        where = frame_pose(ctx, mount)
        if where is None:
            raise SweptError(
                f"--frame {mount.path!r} belongs to {mount.entity!r}, which is absent in the "
                "rebuilt world, so the frame has no pose to place the sensor at."
            )
        mat = _rotation(where.rotation)
        fov.origin = np.asarray(where.translation, dtype=np.float64) + mat @ local_pos
        fov.rot = mat @ local_rot
        visits += coverage(model, state.data, [fov], points).by_sensor[:, 0]
        evaluated.append(state.sim_time)
    return SweptResult(
        frame=mount.path,
        sensor_type=sensor_type,
        fov=fov,
        points=points,
        labels=np.asarray(labels),
        label_names=list(names),
        visits=visits,
        times=np.asarray(evaluated, dtype=np.float64),
    )
