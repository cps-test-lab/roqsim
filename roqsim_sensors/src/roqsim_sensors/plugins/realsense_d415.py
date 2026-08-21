"""Sensor plugin: Intel RealSense D415 colour stream, via ``mujoco.Renderer`` (GL, offscreen).

RGB only -- no depth/IR yet (unlike :mod:`realsense_d435`, which grew an opt-in depth +
point-cloud path; add one here the same way once a D415 world needs it). Topic/frame naming follows
``realsense-ros``'s conventions (``<ns>/camera/color/image_raw``, ``<ns>/camera/color/camera_info``,
``camera_color_optical_frame``), so a world can point an unmodified RealSense-based stack at it.

Same code as the D435 plugin (both are ``CameraPlugin`` colour renderers); it exists as its own
plugin/model so a robot that carries a real D415 is faithful. The bundled
``d415`` model (``models/d415.xml``) provides a ``d415_color`` camera; attach it to a robot or mount
it standalone with ``spawn_sensor``.

Config: see ``camera_common.CameraPlugin`` (``robot``/``arm``, ``camera``, ``width``/``height``,
``fovy``, ``rate_hz``, ``frame_id``, ``compressed``/``jpeg_quality``). Defaults below match the
D415's 640x480@30fps colour profile.
"""

from __future__ import annotations

from .camera_common import CameraPlugin


class RealsenseD415Plugin(CameraPlugin):
    DEFAULT_CAMERA = "d415_color"
    DEFAULT_FRAME_ID = "camera_color_optical_frame"
    DEFAULT_TOPIC_PREFIX = "camera/color"
    DEFAULT_RATE_HZ = 30.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
