"""Sensor plugin: Livox Mid-360 3D lidar via batched ray-casting (:func:`roqsim.raycast.cast`, no GL).

The Mid-360 is a 360deg (horizontal) x 59deg (vertical, -7deg..+52deg) dome-FoV 3D lidar that emits a
point cloud rather than a planar scan, so this is a sibling of the 2D :class:`~.lidar.LidarPlugin`
rather than a mode of it: it casts a spherical grid of rays each frame and exposes the hits as a
``cloud`` (``sensor_msgs/PointCloud2``) output endpoint for a transport plugin. Everything the two
share -- rate gate, range window, noise, mount TF, endpoint -- comes from
:class:`~.lidar_common.RayCastSensorPlugin`.

Fidelity note (a substrate artifact, not a spec of the device): the real Mid-360 sweeps a proprietary
*non-repetitive* rosette pattern that fills the FoV densely over time. The raycaster is deterministic,
so this plugin samples a uniform spherical grid over the same FoV bounds each frame. The default grid
(360 x 56 = 20160 rays at 10 Hz ~= 200k points/s) matches the device's point rate and FoV, not its
instantaneous pattern; for algorithms that depend on the true scan trajectory this is an
approximation, and that is deliberate and documented here rather than hidden.

20160 rays at 10 Hz is by far the most expensive sensor in the tree -- of the order of a whole CPU
core spent on raycasting alone in a mesh-walled indoor world, against ~2% for the 2D ``lidar``. That
cost is currently irreducible: see :mod:`roqsim.raycast` for why the obvious fix (threading the
batch) is unsafe, and what it would actually take.

Datasheet (https://www.livoxtech.com/mid-360): range 0.1 m (blind zone) .. 40 m @ 10% reflectivity
(70 m @ 80%), range precision 1sigma ~= 2 cm @ 10 m, 905 nm, 10 Hz frame rate, ~200k points/s.
Range noise is off by default (package convention -- opt in via ``range_stddev: 0.02`` for the 2 cm
figure), matching how ``lidar`` treats noise.

Config (in addition to ``lidar_common``'s shared keys)::

    livox_mid360:
      site: lidar                # site the rays are cast from
      frame_id: livox_frame      # defaults to `site`; Livox's own driver publishes `livox_frame`
      horizontal_rays: 360       # azimuth samples across the 360deg FoV (wraps, so none at 2pi)
      vertical_rays: 56          # elevation samples across the vertical FoV (inclusive endpoints)
      h_fov_min: 0.0
      h_fov_max: 6.283185307     # 2*pi (full 360deg)
      v_fov_min: -0.122173048    # -7 deg
      v_fov_max: 0.907571211     # +52 deg
      range_min: 0.1             # blind zone; nearer returns are dropped
      max_range: 40.0
"""

from __future__ import annotations

import math

import numpy as np

from .lidar_common import RayCastSensorPlugin
from .payloads import PointCloud

# Vertical FoV of the Mid-360, in radians -- see the datasheet reference in the module docstring.
_V_FOV_MIN = math.radians(-7.0)
_V_FOV_MAX = math.radians(52.0)


class LivoxMid360Plugin(RayCastSensorPlugin):
    ENDPOINT_NAME = "cloud"
    ROS_TYPE = "sensor_msgs.msg.PointCloud2"
    #: Livox's own ROS driver publishes the point cloud on this topic by default.
    DEFAULT_TOPIC = "livox/lidar"
    PLUGIN_LABEL = "livox_mid360"

    DEFAULT_SITE = "lidar"
    DEFAULT_RANGE_MIN = 0.1
    DEFAULT_MAX_RANGE = 40.0

    #: A point cloud lists real returns, so a blind-zone return is not a point (unlike a
    #: fixed-length ``LaserScan``, which clamps it to ``range_min`` to keep its slot).
    CLAMP_NEAR_RETURNS = False

    #: Azimuth spans a full 360deg dome and wraps, so the last sample is one step short of
    #: ``h_fov_max`` (no duplicate ray at 2*pi). A bounded, forward-facing FoV subclass (see
    #: :class:`~roqsim_sensors.plugins.seyond_robin_w1g.SeyondRobinW1GPlugin`) sets this ``False`` to
    #: use inclusive azimuth endpoints instead, like elevation.
    AZIMUTH_WRAPS = True

    DEFAULT_H_RAYS = 360
    DEFAULT_V_RAYS = 56
    DEFAULT_H_FOV_MIN = 0.0
    DEFAULT_H_FOV_MAX = 2.0 * math.pi
    DEFAULT_V_FOV_MIN = _V_FOV_MIN
    DEFAULT_V_FOV_MAX = _V_FOV_MAX

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.h_rays = int(self.config.get("horizontal_rays", self.DEFAULT_H_RAYS))
        self.v_rays = int(self.config.get("vertical_rays", self.DEFAULT_V_RAYS))
        self.h_fov_min = float(self.config.get("h_fov_min", self.DEFAULT_H_FOV_MIN))
        self.h_fov_max = float(self.config.get("h_fov_max", self.DEFAULT_H_FOV_MAX))
        self.v_fov_min = float(self.config.get("v_fov_min", self.DEFAULT_V_FOV_MIN))
        self.v_fov_max = float(self.config.get("v_fov_max", self.DEFAULT_V_FOV_MAX))

    @property
    def num_rays(self) -> int:
        return self.h_rays * self.v_rays

    def _validate_extra(self, config: dict) -> list[str]:
        errors = []
        if int(config.get("horizontal_rays", self.DEFAULT_H_RAYS)) <= 0:
            errors.append("'horizontal_rays' must be > 0")
        if int(config.get("vertical_rays", self.DEFAULT_V_RAYS)) <= 0:
            errors.append("'vertical_rays' must be > 0")
        v_min = float(config.get("v_fov_min", self.DEFAULT_V_FOV_MIN))
        v_max = float(config.get("v_fov_max", self.DEFAULT_V_FOV_MAX))
        if v_min > v_max:
            errors.append("'v_fov_min' must be <= 'v_fov_max'")
        return errors

    def _build_directions(self) -> np.ndarray:
        """Unit ray directions in the site frame: a spherical grid over the (azimuth, elevation) FoV.

        For the full 360deg dome (``AZIMUTH_WRAPS``) azimuth wraps (like the 2D lidar), so the last
        sample is one step short of ``h_fov_max`` -- no duplicate ray at 2*pi. For a bounded FoV
        (a forward-facing subclass) azimuth uses inclusive endpoints, like elevation. Elevation is
        always a bounded band, so its endpoints are inclusive.
        """
        if self.AZIMUTH_WRAPS:
            az = self.h_fov_min + np.arange(self.h_rays) * (
                (self.h_fov_max - self.h_fov_min) / self.h_rays
            )
        elif self.h_rays == 1:
            az = np.array([(self.h_fov_min + self.h_fov_max) * 0.5])
        else:
            az = np.linspace(self.h_fov_min, self.h_fov_max, self.h_rays)
        if self.v_rays == 1:
            el = np.array([(self.v_fov_min + self.v_fov_max) * 0.5])
        else:
            el = np.linspace(self.v_fov_min, self.v_fov_max, self.v_rays)
        EL, AZ = np.meshgrid(el, az, indexing="ij")  # (v_rays, h_rays)
        cos_el = np.cos(EL)
        return np.stack([cos_el * np.cos(AZ), cos_el * np.sin(AZ), np.sin(EL)], axis=-1).reshape(
            -1, 3
        )

    def _payload(self, dist: np.ndarray, valid: np.ndarray) -> PointCloud:
        # Points in the sensor frame: direction * range for each valid return. Frame-independent, so
        # the cloud needs no world transform -- the static TF places the sensor frame in the tree.
        points = self._local_dirs[valid] * dist[valid, None]
        return PointCloud(points=np.ascontiguousarray(points, dtype=np.float32))
