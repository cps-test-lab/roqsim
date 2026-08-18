"""Sensor plugin: Intel RealSense D455 colour stream, via ``mujoco.Renderer`` (GL, offscreen).

RGB only -- no depth/IR yet (like :mod:`realsense_d415`; :mod:`realsense_d435` has an opt-in
depth + point-cloud path to copy when a D455 world needs one). Topic/frame naming
follows ``realsense-ros``'s conventions (``<ns>/camera/color/image_raw``,
``<ns>/camera/color/camera_info``, ``camera_color_optical_frame``), so a world can point an
unmodified RealSense-based stack at it.

Same code as the D435/D415 plugins (all three are ``CameraPlugin`` colour renderers); it exists as
its own plugin/model so a robot that carries a real D455 is faithful -- the D455 is a distinctly
wider-FOV, longer-range camera (87 x 62 deg colour, 95 mm stereo baseline, 0.6-6 m ideal range). The
bundled ``d455`` model (``models/d455.xml``) provides a ``d455_color`` camera with that wide FOV
baked in; attach it to a robot or mount it standalone with ``spawn_sensor``.

Config: see ``camera_common.CameraPlugin`` (``robot``/``arm``, ``camera``, ``width``/``height``,
``fovy``, ``rate_hz``, ``frame_id``). Defaults below match the D455's 640x400@30fps colour profile
(the native 1280x800 colour aspect, downscaled to fit MuJoCo's default 640x480 offscreen buffer).
"""

from __future__ import annotations

from .camera_common import CameraPlugin


class RealsenseD455Plugin(CameraPlugin):
    DEFAULT_CAMERA = "d455_color"
    DEFAULT_FRAME_ID = "camera_color_optical_frame"
    DEFAULT_TOPIC_PREFIX = "camera/color"
    DEFAULT_RATE_HZ = 30.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 400
