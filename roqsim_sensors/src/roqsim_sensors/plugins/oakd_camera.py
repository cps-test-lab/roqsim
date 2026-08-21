"""Sensor plugin: OAK-D Pro RGB-D camera via ``mujoco.Renderer`` (GL, offscreen).

Ported from our earlier in-house nav prototype's ``Camera``/``CameraFrame`` (``mujoco_nav/camera.py``). Mirrors the Gazebo
TurtleBot 4 ``rgbd_camera`` topic (a ``sensor_msgs/Image`` colour + depth pair, each with its own
``CameraInfo`` -- ``rgbd_camera/camera_info`` and ``rgbd_camera/depth/camera_info``). Bundled as a default TurtleBot 4 sensor (see ``turtlebot4.manifest.yaml`` in
``roqsim_mobile``), reading resolution/FOV from the model's ``oakd_rgb`` camera.

Config (in addition to ``camera_common.CameraPlugin``'s, and ``depth_camera.DepthCameraPlugin``'s
``clip_near``/``clip_far``/``depth_encoding``)::

    oakd_camera:
      robot: robot
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

    def _configure_extra(self, ctx: SimContext, prefix: str, ns: str) -> None:
        self._add_depth_endpoints(
            ctx,
            ns,
            self.topic_override("depth")
            or join_topic(self.DEFAULT_TOPIC_PREFIX, "depth/image_raw"),
            self.frame_id,
        )
