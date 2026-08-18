"""The coverage computation: how many sensors observe each sample point.

For each sensor and each sample point the test is three gates, cheapest first:

1. **range** -- ``range_min <= ||p - origin|| <= range_max`` (pure numpy).
2. **angular FOV** -- :func:`~.fov.in_fov` in the sensor frame (pure numpy).
3. **line of sight** -- one batched ``mj_multiRay`` per sensor over the points that passed 1+2, using
   the same call pattern as the lidar plugins. A point is visible iff the nearest occluder is beyond it
   (or there is none).

Correctness notes:

* The raycast ``geomgroup`` mask includes only geom groups 0-3, which keeps an absent entity
  (:data:`roqsim.presence.ABSENT_GEOM_GROUP`, group 4) from occluding. It is **not** what keeps a sensor
  model's FOV-visualisation mesh out: those live in group 2 (``spawn_sensor.FOV_GEOM_GROUP``, chosen
  because the MuJoCo 3.x offscreen renderer drops large group-4/5 geoms) and are excluded because
  their alpha is 0 -- ``mj_ray`` skips a geom exactly when its resolved alpha is 0, whatever the mask
  says. Masking by group is still needed, since raycasts hit *visible* geometry regardless of contact
  flags.
* Sensors are evaluated against the *fixed* world only; they do not occlude one another (a real
  deployment's sensors are transparent to each other's sensing), so coverage is order-independent.
* Sample points are expected to sit in free space (volume points) or just outside a surface (object
  points, offset along the normal); that keeps the target's own geometry from being read as an occluder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .fov import SensorFov, in_fov


@dataclass
class CoverageResult:
    points: np.ndarray  # (P, 3) world sample points
    counts: np.ndarray  # (P,) int -- number of sensors that see each point
    by_sensor: np.ndarray  # (P, S) bool
    fovs: list[SensorFov]
    labels: np.ndarray | None = None  # (P,) int object-label index (surface points) or None
    label_names: list[str] = field(default_factory=list)  # index -> object name

    def coverage_fraction(self, k: int = 1) -> float:
        """Fraction of sample points seen by at least ``k`` sensors."""
        if self.counts.size == 0:
            return 0.0
        return float(np.mean(self.counts >= k))

    def counts_by_type(self) -> dict[str, np.ndarray]:
        """Per sensor type, a (P,) bool mask of points seen by at least one sensor of that type."""
        out: dict[str, np.ndarray] = {}
        for s, fov in enumerate(self.fovs):
            col = self.by_sensor[:, s]
            out[fov.sensor_type] = out.get(fov.sensor_type, np.zeros(len(self.points), bool)) | col
        return out

    def per_sensor_seen(self) -> np.ndarray:
        """(S,) count of points each sensor sees."""
        return self.by_sensor.sum(axis=0).astype(int)


def _geomgroup_mask(include_groups) -> np.ndarray:
    mask = np.zeros(6, dtype=np.uint8)
    for g in include_groups:
        if 0 <= int(g) < 6:
            mask[int(g)] = 1
    return mask


def coverage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fovs: list[SensorFov],
    points: np.ndarray,
    *,
    eps: float = 1e-3,
    include_groups=(0, 1, 2, 3),
    labels: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> CoverageResult:
    """Compute per-point sensor coverage of ``points`` in the compiled world (``model``/``data``)."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    n_points = len(points)
    n_sensors = len(fovs)
    by_sensor = np.zeros((n_points, n_sensors), dtype=bool)
    geomgroup = _geomgroup_mask(include_groups)

    for s, fov in enumerate(fovs):
        delta = points - fov.origin
        dist = np.linalg.norm(delta, axis=1)
        in_range = (dist >= fov.range_min) & (dist <= fov.range_max)
        if not in_range.any():
            continue
        angular = in_fov(fov, fov.to_local(points))
        candidate = in_range & angular
        idx = np.nonzero(candidate)[0]
        if idx.size == 0:
            continue

        dist_to = dist[idx]
        dirs = np.ascontiguousarray((delta[idx] / dist_to[:, None]).reshape(-1))
        geomid = np.full(idx.size, -1, dtype=np.int32)
        raydist = np.full(idx.size, -1.0, dtype=np.float64)
        cutoff = float(dist_to.max()) + 1.0
        mujoco.mj_multiRay(
            model,
            data,
            np.ascontiguousarray(fov.origin),
            dirs,
            geomgroup,
            1,  # flg_static: static geometry (walls, furniture) occludes
            # The sensor's own mount, which its origin sits inside -- see SensorFov.body_exclude.
            fov.body_exclude,
            geomid,
            raydist,
            None,
            idx.size,
            cutoff,
        )
        # Visible: nothing hit before the target (raydist < 0), or the nearest hit is at/behind it.
        visible = (raydist < 0.0) | (raydist >= dist_to - eps)
        by_sensor[idx, s] = visible

    counts = by_sensor.sum(axis=1).astype(int)
    return CoverageResult(
        points=points,
        counts=counts,
        by_sensor=by_sensor,
        fovs=list(fovs),
        labels=None if labels is None else np.asarray(labels).reshape(-1),
        label_names=list(label_names or []),
    )
