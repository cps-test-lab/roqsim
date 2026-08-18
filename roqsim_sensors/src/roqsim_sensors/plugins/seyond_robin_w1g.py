"""Sensor plugin: Seyond Robin W1G forward-facing solid-state 3D lidar via batched ray-casting.

The Robin W1G is an automotive-grade solid-state lidar with a *bounded, forward-facing* FoV
(120deg horizontal x 70deg vertical), unlike the 360deg-dome Livox Mid-360. It shares the Mid-360's
substrate machinery -- a spherical grid of rays cast with ``mj_multiRay`` each frame, exposed as a
``cloud`` (``sensor_msgs/PointCloud2``) endpoint -- so this is a thin subclass of
:class:`~roqsim_sensors.plugins.livox_mid360.LivoxMid360Plugin` that only changes the FoV bounds,
the default grid/range, and (crucially) makes azimuth a *bounded band with inclusive endpoints*
instead of a wrapping 360deg sweep (``AZIMUTH_WRAPS = False``). Boresight is +x (the ``robin_w1g``
site's forward axis); azimuth spans -60deg..+60deg about it, elevation -35deg..+35deg.

Fidelity note (a substrate artifact, not a spec of the device): the real W1G sweeps a proprietary
scan pattern that fills the FoV over time at its full 0.15deg x 0.36deg angular resolution
(~1.28M points/s single return @ 10 Hz). ``mj_multiRay`` is a deterministic caster, so this plugin
samples a uniform ``horizontal_rays x vertical_rays`` grid over the same FoV bounds each frame. The
default 192 x 112 = 21504 rays at 10 Hz keeps the per-frame cost in the Mid-360's range and matches
the device's FoV, not its instantaneous pattern or full point rate; for algorithms that depend on the
true scan trajectory or density this is an approximation, deliberate and documented here.

Datasheet (https://www.seyond.com/product/robin-w1g/): FoV 120deg x 70deg, angular resolution
0.15deg x 0.36deg, detection range 70 m (POD>90% @ 10% reflectivity, 10 Hz) / 150 m max, blind zone
0.1 m, range precision 1 cm (1sigma), 905 nm, 10-20 FPS. ``max_range`` defaults to the 70 m spec'd
detection range (raise it toward 150 for the maximum). Range noise is off by default (package
convention -- opt in via ``range_stddev: 0.01`` for the 1 cm precision), matching ``lidar``/Mid-360.

Config (same keys as ``livox_mid360``; only the defaults differ)::

    seyond_robin_w1g:
      robot: robot
      namespace: ""              # transport scope (default: inherited from spawn's namespace)
      site: robin_w1g            # site the rays are cast from
      frame_id: seyond_lidar     # ROS frame the cloud is stamped in + child of the static mount TF;
                                 #   defaults to `site`. Seyond's own driver publishes `seyond_lidar`.
      horizontal_rays: 192       # azimuth samples across the 120deg FoV (inclusive endpoints)
      vertical_rays: 112         # elevation samples across the 70deg FoV (inclusive endpoints)
      h_fov_min: -1.047197551    # -60 deg
      h_fov_max: 1.047197551     # +60 deg
      v_fov_min: -0.610865238    # -35 deg
      v_fov_max: 0.610865238     # +35 deg
      range_min: 0.1             # blind zone; nearer returns are dropped
      max_range: 70.0            # spec'd detection range (150 m maximum)
      rate_hz: 10.0              # cloud rate: rays are cast (and published) at this rate, not per step
      exclude_body: base_link    # robot's own body, so rays skip the chassis
      range_stddev: 0.0          # optional Gaussian range noise (0 = off); device 1sigma ~= 0.01
      dropout_percent: 0.0       # optional: % of points dropped (-> no return) per frame (0..100)
      emit_static_tf: true       # publish base_link -> frame_id as a static TF (from the site)
"""

from __future__ import annotations

import math

from .livox_mid360 import LivoxMid360Plugin

# Robin W1G FoV half-angles (radians) about the +x boresight -- see the datasheet reference above.
_H_FOV = math.radians(60.0)  # 120 deg horizontal total
_V_FOV = math.radians(35.0)  # 70 deg vertical total


class SeyondRobinW1GPlugin(LivoxMid360Plugin):
    #: Bounded forward FoV: azimuth is a band with inclusive endpoints, not a wrapping 360deg sweep.
    AZIMUTH_WRAPS = False
    #: Seyond's ROS driver publishes the point cloud on this topic by default.
    DEFAULT_TOPIC = "seyond/points"
    PLUGIN_LABEL = "seyond_robin_w1g"

    def __init__(self, config=None, *, name=None):
        # Robin W1G defaults differ from the Mid-360's dome (360deg x 59deg, 40 m). Apply them unless
        # the world overrides -- the base class reads these same keys (zivid<-oakd pattern).
        cfg = dict(config or {})
        cfg.setdefault("site", "robin_w1g")
        cfg.setdefault("horizontal_rays", 192)
        cfg.setdefault("vertical_rays", 112)
        cfg.setdefault("h_fov_min", -_H_FOV)
        cfg.setdefault("h_fov_max", _H_FOV)
        cfg.setdefault("v_fov_min", -_V_FOV)
        cfg.setdefault("v_fov_max", _V_FOV)
        cfg.setdefault("range_min", 0.1)
        cfg.setdefault("max_range", 70.0)
        cfg.setdefault("rate_hz", 10.0)
        super().__init__(cfg, name=name)
