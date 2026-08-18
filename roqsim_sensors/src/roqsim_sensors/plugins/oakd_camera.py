"""Sensor plugin: OAK-D Pro RGB-D camera via ``mujoco.Renderer`` (GL, offscreen).

Ported from our earlier in-house nav prototype's ``Camera``/``CameraFrame`` (``mujoco_nav/camera.py``). Mirrors the Gazebo
TurtleBot 4 ``rgbd_camera`` topic (a ``sensor_msgs/Image`` colour + depth pair with ``CameraInfo``
intrinsics). Bundled as a default TurtleBot 4 sensor (see ``turtlebot4.manifest.yaml`` in
``roqsim_mobile``), reading resolution/FOV from the model's ``oakd_rgb`` camera.

Config (in addition to ``camera_common.CameraPlugin``'s)::

    oakd_camera:
      robot: robot
      camera: oakd_rgb
      clip_near: 0.3      # m; depth outside [clip_near, clip_far] reads as "no return" (inf)
      clip_far: 100.0     # m
"""

from __future__ import annotations

import numpy as np

from roqsim.context import Endpoint, SimContext

from .camera_common import CameraPlugin, join_topic


class OakDCameraPlugin(CameraPlugin):
    DEFAULT_CAMERA = "oakd_rgb"
    DEFAULT_FRAME_ID = "oakd_rgb_camera_optical_frame"
    DEFAULT_TOPIC_PREFIX = "rgbd_camera"
    DEFAULT_RATE_HZ = 10.0
    DEFAULT_WIDTH = 320
    DEFAULT_HEIGHT = 240

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.clip_near = float(self.config.get("clip_near", 0.3))
        self.clip_far = float(self.config.get("clip_far", 100.0))
        self._depth: np.ndarray | None = None
        self._depth_ep: Endpoint | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = super().validate_config(config)
        if float(config.get("clip_near", 0.3)) < 0:
            errors.append("'clip_near' must be >= 0")
        if float(config.get("clip_far", 100.0)) <= float(config.get("clip_near", 0.3)):
            errors.append("'clip_far' must be > 'clip_near'")
        return errors

    def _configure_extra(self, ctx: SimContext, prefix: str, ns: str) -> None:
        self._depth_ep = Endpoint(
            name="depth",
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=lambda: self._depth,
            rate_hz=self.rate_hz,
            backend={
                "ros2": {
                    "type": "sensor_msgs.msg.Image",
                    "topic": self.topic_override("depth")
                    or join_topic(self.DEFAULT_TOPIC_PREFIX, "depth/image_raw"),
                    "frame_id": self.frame_id,
                    "encoding": "32FC1",
                }
            },
        )
        ctx.interface.add(self._depth_ep)
        # Gate the renderer on depth too: a consumer wanting only depth must still get frames.
        self._extra_outputs.append(self._depth_ep)

    def _capture_extra(self, ctx: SimContext, renderer) -> None:
        renderer.enable_depth_rendering()
        renderer.update_scene(ctx.data, camera=self._cam_id)
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()
        depth[(depth < self.clip_near) | (depth > self.clip_far)] = np.inf
        self._depth = depth
