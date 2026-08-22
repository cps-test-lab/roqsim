"""Sensor plugin: 2D lidar via batched ray-casting (:func:`roqsim.raycast.cast`, no GL).

Ported from our earlier in-house nav prototype's ``Lidar``. Casts a horizontal fan from the robot's
``lidar`` site at ``rate_hz``, optionally applies sensor noise, and exposes the latest
:class:`~.payloads.LaserScan` via a ``scan`` output endpoint for a transport plugin.

The shared machinery -- the rate gate, the range window, the noise model, the static mount TF and the
endpoint -- lives in :class:`~.lidar_common.RayCastSensorPlugin`; this file is the fan pattern, the
``LaserScan`` payload, and the device defaults.

Config (in addition to ``lidar_common``'s ``robot``/``namespace``/``site``/``frame_id``/
``range_min``/``max_range``/``rate_hz``/``exclude_body``/``range_stddev``/``dropout_percent``/
``emit_static_tf``)::

    lidar:
      rays: 360
      angle_min: 0.0
      angle_max: 6.283185307     # 2*pi

``frame_id`` defaults to ``site``; set it when the robot's real description names the frame
differently (the TurtleBot 4's URDF calls it ``rplidar_link``).
"""

from __future__ import annotations

import math

import numpy as np

from .lidar_common import RayCastSensorPlugin
from .payloads import LaserScan


class LidarPlugin(RayCastSensorPlugin):
    ENDPOINT_NAME = "scan"
    ROS_TYPE = "sensor_msgs.msg.LaserScan"
    DEFAULT_TOPIC = "scan"
    PLUGIN_LABEL = "lidar"

    DEFAULT_SITE = "lidar"
    DEFAULT_RANGE_MIN = 0.164
    DEFAULT_MAX_RANGE = 20.0

    #: A ``LaserScan`` is a fixed-length array, so a blind-zone return keeps its slot at
    #: ``range_min`` rather than vanishing and shifting every later angle.
    CLAMP_NEAR_RETURNS = True

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self._num_rays = int(self.config.get("rays", 360))
        self.angle_min = float(self.config.get("angle_min", 0.0))
        self.angle_max = float(self.config.get("angle_max", 2.0 * math.pi))
        self._angle_increment = (self.angle_max - self.angle_min) / self._num_rays

    @property
    def num_rays(self) -> int:
        return self._num_rays

    def _validate_extra(self, config: dict) -> list[str]:
        return [] if int(config.get("rays", 360)) > 0 else ["'rays' must be > 0"]

    def _build_directions(self) -> np.ndarray:
        """A horizontal fan. Azimuth wraps, so the last sample is one step short of ``angle_max``
        -- no duplicate ray at 2*pi for a full sweep."""
        angles = self.angle_min + np.arange(self._num_rays) * self._angle_increment
        return np.stack([np.cos(angles), np.sin(angles), np.zeros(self._num_rays)], axis=1)

    def _payload(self, dist: np.ndarray, valid: np.ndarray) -> LaserScan:
        return LaserScan(
            ranges=np.where(valid, dist, np.inf),
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self._angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
        )
