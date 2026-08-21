"""Sensor plugin: Intel RealSense D455 colour + depth + point cloud, via ``mujoco.Renderer``.

Topic/frame naming follows ``realsense-ros``'s conventions (``<ns>/camera/color/image_raw``,
``<ns>/camera/color/camera_info``, ``<ns>/camera/depth/image_rect_raw``,
``<ns>/camera/depth/camera_info``, ``<ns>/camera/depth/color/points``), so a world can point an
unmodified RealSense-based stack at it.

The depth and cloud paths are :class:`~roqsim_sensors.plugins.realsense_d435.RealsenseD435Plugin`'s,
inherited rather than copied -- the two cameras differ in optics and range, not in what a plugin has
to publish. Both are opt-in (``depth:``/``points:``), and ``points`` implies ``depth``; the cloud
reprojects every valid pixel, so it is not free.

What is D455-specific: a distinctly wider-FOV, longer-range camera (87 x 62 deg colour, 95 mm stereo
baseline, 0.6-6 m ideal range) against the D435's 69 x 42 deg and 0.3-3 m. The bundled ``d455`` model
(``models/d455.xml``) provides a ``d455_color`` camera with that wide FOV baked in, and the clip
range below is the D455's own -- which is what decides whether a wall 4 m away is a depth return or
a "no return". Attach it to a robot or mount it standalone with ``spawn_sensor``.

Config: see ``camera_common.CameraPlugin`` (``robot``/``arm``, ``camera``, ``width``/``height``,
``fovy``, ``rate_hz``, ``frame_id``, ``compressed``/``jpeg_quality``) plus
``depth``/``points``/``clip_near``/``clip_far``/
``depth_frame_id``. Defaults below match the D455's 640x400@30fps colour profile (the native
1280x800 colour aspect, downscaled to fit MuJoCo's default 640x480 offscreen buffer).
"""

from __future__ import annotations

from .realsense_d435 import RealsenseD435Plugin


class RealsenseD455Plugin(RealsenseD435Plugin):
    DEFAULT_CAMERA = "d455_color"
    DEFAULT_FRAME_ID = "camera_color_optical_frame"
    DEFAULT_TOPIC_PREFIX = "camera/color"
    DEFAULT_RATE_HZ = 30.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 400

    def __init__(self, config=None, *, name=None):
        cfg = dict(config or {})
        # The D455's own working range, not the D435's 0.28-3 m: min-Z ~0.6 m, ideal to ~6 m. Set
        # BEFORE super().__init__, whose setdefault would otherwise install the D435's numbers.
        cfg.setdefault("clip_near", 0.6)
        cfg.setdefault("clip_far", 6.0)
        super().__init__(cfg, name=name)
