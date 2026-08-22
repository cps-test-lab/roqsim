"""Sensor plugin: Seyond Robin W1G forward-facing solid-state 3D lidar via batched ray-casting.

The Robin W1G is an automotive-grade solid-state lidar with a *bounded, forward-facing* FoV
(120deg horizontal x 70deg vertical), unlike the 360deg-dome Livox Mid-360. It shares the Mid-360's
substrate machinery -- a spherical grid of rays cast each frame, exposed as a ``cloud``
(``sensor_msgs/PointCloud2``) endpoint -- so this is a thin subclass of
:class:`~roqsim_sensors.plugins.livox_mid360.LivoxMid360Plugin` that only changes the FoV bounds, the
default grid/range, and (crucially) makes azimuth a *bounded band with inclusive endpoints* instead
of a wrapping 360deg sweep (``AZIMUTH_WRAPS = False``). Boresight is +x (the ``robin_w1g`` site's
forward axis); azimuth spans -60deg..+60deg about it, elevation -35deg..+35deg.

Fidelity note (a substrate artifact, not a spec of the device): the real W1G sweeps a proprietary
scan pattern that fills the FoV over time at its full 0.15deg x 0.36deg angular resolution
(~1.28M points/s single return @ 10 Hz). The raycaster is deterministic, so this plugin samples a
uniform ``horizontal_rays x vertical_rays`` grid over the same FoV bounds each frame. The default
192 x 112 = 21504 rays at 10 Hz keeps the per-frame cost in the Mid-360's range and matches the
device's FoV, not its instantaneous pattern or full point rate; for algorithms that depend on the
true scan trajectory or density this is an approximation, deliberate and documented here.

Datasheet (https://www.seyond.com/product/robin-w1g/): FoV 120deg x 70deg, angular resolution
0.15deg x 0.36deg, detection range 70 m (POD>90% @ 10% reflectivity, 10 Hz) / 150 m max, blind zone
0.1 m, range precision 1 cm (1sigma), 905 nm, 10-20 FPS. ``max_range`` defaults to the 70 m spec'd
detection range (raise it toward 150 for the maximum). Range noise is off by default (package
convention -- opt in via ``range_stddev: 0.01`` for the 1 cm precision), matching ``lidar``/Mid-360.

Config: the same keys as ``livox_mid360``; only the defaults below differ.
"""

from __future__ import annotations

import math

from .livox_mid360 import LivoxMid360Plugin

# Robin W1G FoV half-angles (radians) about the +x boresight -- see the datasheet reference above.
_H_FOV = math.radians(60.0)  # 120 deg horizontal total
_V_FOV = math.radians(35.0)  # 70 deg vertical total


class SeyondRobinW1GPlugin(LivoxMid360Plugin):
    #: Seyond's ROS driver publishes the point cloud on this topic by default.
    DEFAULT_TOPIC = "seyond/points"
    PLUGIN_LABEL = "seyond_robin_w1g"

    #: Bounded forward FoV: azimuth is a band with inclusive endpoints, not a wrapping 360deg sweep.
    AZIMUTH_WRAPS = False

    DEFAULT_SITE = "robin_w1g"
    DEFAULT_MAX_RANGE = 70.0
    DEFAULT_H_RAYS = 192
    DEFAULT_V_RAYS = 112
    DEFAULT_H_FOV_MIN = -_H_FOV
    DEFAULT_H_FOV_MAX = _H_FOV
    DEFAULT_V_FOV_MIN = -_V_FOV
    DEFAULT_V_FOV_MAX = _V_FOV
