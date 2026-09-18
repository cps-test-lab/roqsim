"""Sensor plugin: a ray-grid range sensor -- the ToF / IR proximity / ultrasonic / cliff model.

The small range sensors a base carries are not scanners. An IR proximity sensor, a ToF module, an
ultrasonic ranger or a downward cliff sensor illuminates a narrow cone and reports what is in it,
and a simulator models that cone as a small **grid** of rays: a few across by a few down, all
within the sensor's field of view. What a consumer does with the grid is the consumer's business --
a cliff detector asks whether the nearest return is farther than the floor should be, a
proximity sensor turns the nearest return into an intensity -- so the grid is published as it is,
and nothing here decides what a cliff or an obstacle is.

Casts ``h_rays x v_rays`` rays from its ``site`` through ``h_fov x v_fov`` at ``rate_hz`` and
publishes them as one :class:`~.payloads.LaserScan`, rows concatenated in row-major order (the top
row first, each row sweeping from ``-h_fov/2`` to ``+h_fov/2``). The ``angle_*`` fields describe the
horizontal sweep of one row, so a single-row sensor is an ordinary short ``LaserScan`` and a
multi-row one is a ``LaserScan`` whose ``ranges`` hold every row -- the layout the Create 3's
simulator publishes its 5x5 IR sensors in, and the one a consumer that takes the minimum over
``ranges`` (every proximity and cliff detector) reads unchanged. Every other key -- the detection
limits, ``too_close`` / ``no_return``, the noise model, the fault switch, the static mount TF -- is
:mod:`lidar`'s, inherited rather than restated.

Config (in addition to ``lidar``'s ``site``/``frame_id``/``rate_hz``/``range_min``/``max_range``/
``detection_min``/``detection_max``/``too_close``/``no_return``/noise/``emit_static_tf``/
``tf_parent`` keys)::

    range_sensor:
      site: cliff_front_left     # rays are cast along the site's +x axis
      h_rays: 1                  # rays across the horizontal field of view (>= 1)
      v_rays: 1                  # rays down the vertical field of view (>= 1)
      h_fov: 0.0873              # horizontal field of view, rad (5 deg); 0 with h_rays 1
      v_fov: 0.0                 # vertical field of view, rad; 0 with v_rays 1
      range_min: 0.0001          # published as the header's range_min
      max_range: 0.15            # published as the header's range_max
      rate_hz: 62.0

A ``lidar``'s ``rays``/``angle_min``/``angle_max`` are refused here: the layout is the grid's, and
a world that wants a fan declares a ``lidar``.

**Pointing it.** The site's ``+x`` is the boresight, as for every ray sensor here; a cliff sensor
is a site pitched towards the floor, a proximity sensor a site facing out through the shell. With
the boresight on the floor at a known standoff, a floor return reads the standoff and a hole reads
``no_return`` (``+inf`` by REP 117), which is exactly the comparison a cliff detector makes.
"""

from __future__ import annotations

import math

import numpy as np

from .lidar import LidarPlugin
from .payloads import LaserScan


class RangeSensorPlugin(LidarPlugin):
    ENDPOINT_NAME = "scan"
    ROS_TYPE = "sensor_msgs.msg.LaserScan"
    DEFAULT_TOPIC = "range"
    PLUGIN_LABEL = "range_sensor"

    DEFAULT_SITE = "range"
    DEFAULT_RANGE_MIN = 0.01
    DEFAULT_MAX_RANGE = 2.0
    DEFAULT_RATE_HZ = 62.0
    DEFAULT_H_RAYS = 1
    DEFAULT_V_RAYS = 1
    DEFAULT_H_FOV = 0.0
    DEFAULT_V_FOV = 0.0

    REFUSED_WRITES = {
        **LidarPlugin.REFUSED_WRITES,
        "h_rays": "it changes the LaserScan's length mid-run.",
        "v_rays": "see 'h_rays'.",
        "h_fov": "it changes which bearing each index means.",
        "v_fov": "see 'h_fov'.",
    }

    #: A fan's layout keys, refused so a grid cannot be half a fan.
    _FAN_KEYS = ("rays", "angle_min", "angle_max")

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.h_rays = max(int(self.config.get("h_rays", self.DEFAULT_H_RAYS)), 1)
        self.v_rays = max(int(self.config.get("v_rays", self.DEFAULT_V_RAYS)), 1)
        self.h_fov = float(self.config.get("h_fov", self.DEFAULT_H_FOV))
        self.v_fov = float(self.config.get("v_fov", self.DEFAULT_V_FOV))
        # One row's sweep, in the fan vocabulary the payload is built from: a single ray sweeps
        # nothing, which the fan spells as angle_max == angle_min and a zero increment.
        half_h = self.h_fov / 2.0 if self.h_rays > 1 else 0.0
        self._num_rays = self.h_rays
        self.angle_min = -half_h
        self.angle_max = half_h
        self._angle_increment = self.h_fov / (self.h_rays - 1) if self.h_rays > 1 else 0.0

    @property
    def num_rays(self) -> int:
        return self.h_rays * self.v_rays

    def _validate_extra(self, config: dict) -> list[str]:
        errors = []
        for key in self._FAN_KEYS:
            if key in config:
                errors.append(
                    f"'{key}' is a lidar fan's layout; a range_sensor is laid out by "
                    f"'h_rays'/'v_rays' and 'h_fov'/'v_fov'"
                )
        h_rays = int(config.get("h_rays", self.DEFAULT_H_RAYS))
        v_rays = int(config.get("v_rays", self.DEFAULT_V_RAYS))
        h_fov = float(config.get("h_fov", self.DEFAULT_H_FOV))
        v_fov = float(config.get("v_fov", self.DEFAULT_V_FOV))
        if h_rays < 1 or v_rays < 1:
            errors.append("'h_rays' and 'v_rays' must be >= 1")
        for rays, fov, axis in ((h_rays, h_fov, "h"), (v_rays, v_fov, "v")):
            if rays == 1 and fov != 0.0:
                errors.append(
                    f"a single ray has no field of view: '{axis}_fov' must be 0 with '{axis}_rays: 1'"
                )
            if rays > 1 and not 0.0 < fov < math.pi:
                errors.append(f"'{axis}_fov' must be in (0, pi) with '{axis}_rays' > 1")
        # The fan checks (detection limits, too_close/no_return) still apply; run them on the
        # config LidarPlugin would have seen, without the fan keys this class refuses.
        fan = {k: v for k, v in config.items() if k not in self._FAN_KEYS}
        return errors + super()._validate_extra(fan)

    def _build_directions(self) -> np.ndarray:
        """A grid of unit directions: rows down the vertical fov, each a sweep across the horizontal.

        Row-major, top row first, so ``ranges[r * h_rays + c]`` is row ``r`` column ``c``. Elevation
        is applied about the site's y axis and azimuth about z, so the boresight is ``+x``.
        """
        half_h = self.h_fov / 2.0 if self.h_rays > 1 else 0.0
        half_v = self.v_fov / 2.0 if self.v_rays > 1 else 0.0
        az = np.linspace(-half_h, half_h, self.h_rays)
        el = np.linspace(half_v, -half_v, self.v_rays)
        el_grid, az_grid = np.meshgrid(el, az, indexing="ij")
        cos_el = np.cos(el_grid)
        dirs = np.stack(
            [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)], axis=-1
        )
        return dirs.reshape(-1, 3)

    def _payload(self, dist: np.ndarray, valid: np.ndarray, near: np.ndarray) -> LaserScan:
        ranges = np.full(self.num_rays, self.no_return, dtype=np.float64)
        ranges[valid] = dist[valid]
        ranges[near] = dist[near] if self.too_close is None else self.too_close
        return LaserScan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self.angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
        )
