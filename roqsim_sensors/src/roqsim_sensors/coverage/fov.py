"""The single shared field-of-view definition + its membership test.

A camera frustum and a lidar dome are both "an angular sector between ``range_min`` and ``range_max``,
posed in the world" -- they differ only in the shape of the angular cross-section and in whether
``range_max`` is a real device property. :class:`SensorFov` encodes both honestly, and :func:`in_fov`
tests whether points fall inside the angular sector. This module knows nothing about any specific
sensor; the per-type extraction lives in :mod:`~roqsim_sensors.coverage.adapters`.

Frame conventions (must match the raycasters so membership and line-of-sight agree):

* ``CONE_BAND`` uses the lidar convention from ``livox_mid360._build_directions`` -- forward ``+x``,
  azimuth measured about ``+z``, elevation off the xy-plane
  (``dir = [cos(el)cos(az), cos(el)sin(az), sin(el)]``).
* ``FRUSTUM`` uses the MuJoCo camera frame (``data.cam_xmat``): the optical axis looks along ``-z``,
  ``+x`` points right, ``+y`` points up. Membership is a true pinhole projection, not a spherical band.

``rot`` is the world<-sensor rotation (its columns are the sensor's axes expressed in world), so a
world point maps to the sensor frame with ``local = (p - origin) @ rot``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from ..plugins.camera_common import Intrinsics

_TWO_PI = 2.0 * np.pi


class FovKind(enum.Enum):
    FRUSTUM = "frustum"  # camera: rectangular pinhole; membership by image-plane projection
    CONE_BAND = "cone_band"  # lidar: azimuth/elevation band; membership by (az, el) bounds


@dataclass
class SensorFov:
    """A posed field of view, common to every sensor type."""

    kind: FovKind
    origin: np.ndarray  # (3,) world position of the optical/ray origin
    rot: np.ndarray  # (3, 3) world<-sensor rotation
    h_fov: tuple[float, float]  # azimuth (min, max) [rad], sensor frame
    v_fov: tuple[float, float]  # elevation (min, max) [rad], sensor frame
    range_min: float
    range_max: float
    #: True when ``range_max`` is a real device limit (lidar); False when it is an analysis assumption
    #: (a camera has no physical far range -- an over-generous ``far`` inflates coverage, so keep the
    #: distinction visible rather than pretending the number is physics).
    range_is_physical: bool
    sensor_type: str
    intrinsics: Intrinsics | None = None  # FRUSTUM only (the honest rectangular boundary)
    label: str = ""
    #: MuJoCo body id whose geoms this sensor cannot see, or ``-1`` for none -- its own mount. A real
    #: device's lens sits on the outside of its housing; a MuJoCo ``<camera>`` sits at the *pose* the
    #: datasheet gives, which is millimetres BEHIND the housing geom modelling that face. So a
    #: visibility ray leaves the origin already inside the sensor's own body and is occluded by it
    #: immediately: the D435 mount's ``d435_front`` sits 4.3 mm ahead of its camera and blocked the
    #: whole central cone, which is why every ``spawn_sensor``-mounted camera under-reported (a lone
    #: camera in an empty room measured 0.000 coverage while its wide-angle fringe rays still got
    #: out). Passed to ``mj_multiRay``'s ``bodyexclude``, the same mechanism the ``lidar`` plugin's
    #: ``exclude_body`` uses for a robot's chassis -- this is that fix for the coverage engine.
    body_exclude: int = -1

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.rot = np.asarray(self.rot, dtype=np.float64).reshape(3, 3)
        if self.kind is FovKind.FRUSTUM and self.intrinsics is None:
            raise ValueError("FRUSTUM SensorFov requires intrinsics")

    def to_local(self, world_points: np.ndarray) -> np.ndarray:
        """World points (N, 3) expressed in this sensor's frame."""
        world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        return (world_points - self.origin) @ self.rot


def _azimuth_in_range(az: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Angular membership with wraparound. A span >= 2*pi means the full circle (always in range)."""
    span = hi - lo
    if span >= _TWO_PI - 1e-9:
        return np.ones_like(az, dtype=bool)
    rel = np.mod(az - lo, _TWO_PI)  # distance CCW from lo, in [0, 2*pi)
    return rel <= np.mod(span, _TWO_PI)


def in_fov(fov: SensorFov, local_points: np.ndarray) -> np.ndarray:
    """Boolean mask: which ``local_points`` (N, 3 in the sensor frame) lie inside the angular sector.

    This is the *angular* gate only -- range and line-of-sight are applied by the coverage engine.
    """
    local = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
    x, y, z = local[:, 0], local[:, 1], local[:, 2]

    if fov.kind is FovKind.CONE_BAND:
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.hypot(x, y))
        h_ok = _azimuth_in_range(az, fov.h_fov[0], fov.h_fov[1])
        v_ok = (el >= fov.v_fov[0]) & (el <= fov.v_fov[1])
        return h_ok & v_ok

    # FRUSTUM: pinhole projection into the image rectangle. The camera looks along -z, so a point is
    # in front only when z < 0; project and test the pixel bounds.
    intr = fov.intrinsics
    in_front = z < 0.0
    zz = np.where(in_front, z, -1.0)  # avoid divide-by-zero for points behind the camera
    u = intr.cx - intr.fx * x / zz
    v = intr.cy + intr.fy * y / zz
    return in_front & (u >= 0.0) & (u < intr.width) & (v >= 0.0) & (v < intr.height)
