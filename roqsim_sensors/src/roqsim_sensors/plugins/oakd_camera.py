"""Sensor plugin: OAK-D Pro RGB-D camera via ``mujoco.Renderer`` (GL, offscreen).

Mirrors the Gazebo TurtleBot 4 ``rgbd_camera`` topic (a ``sensor_msgs/Image`` colour + depth pair,
each with its own ``CameraInfo`` -- ``rgbd_camera/camera_info`` and
``rgbd_camera/depth/camera_info``). Bundled with the ``oakd_pro`` device model, which
``turtlebot4.manifest.yaml`` in ``roqsim_mobile`` mounts, reading resolution/FOV from its
``oakd_rgb`` camera; the mount publishes the vendor chain to the frame the images are stamped in.
Depth renders through that one camera and is stamped in its optical frame: depthai-ros aligns stereo
depth to the RGB image by default (``i_align_depth``) and stamps it in
``<name>_rgb_camera_optical_frame``, so the device has no separate depth camera.

Config (in addition to ``camera_common.CameraPlugin``'s, and ``depth_camera.DepthCameraPlugin``'s
``clip_near``/``clip_far``/``depth_encoding``)::

    oakd_camera:
      camera: oakd_rgb
      clip_near: 0.3      # m; depth outside [clip_near, clip_far] reads as "no return" (inf)
      clip_far: 100.0     # m

On ``depth_encoding: 16UC1``: this camera's default 100 m ``clip_far`` is further than uint16
millimetres reach, so opting in means also lowering the range to the depth the world actually needs --
which the plugin says at load time rather than saturating at 65.5 m.
"""

from __future__ import annotations

from roqsim.context import SimContext

from .camera_common import join_topic
from .depth_camera import DepthCameraPlugin


class OakDCameraPlugin(DepthCameraPlugin):
    DEFAULT_CAMERA = "oakd_rgb"
    DEFAULT_FRAME_ID = "oakd_rgb_camera_optical_frame"
    DEFAULT_TOPIC_PREFIX = "rgbd_camera"
    DEFAULT_RATE_HZ = 10.0
    DEFAULT_WIDTH = 320
    DEFAULT_HEIGHT = 240

    def _configure_extra(self, ctx: SimContext, prefix: str) -> None:
        self._add_depth_endpoints(
            ctx, join_topic(self.DEFAULT_TOPIC_PREFIX, "depth/image_raw"), self.frame_id
        )
