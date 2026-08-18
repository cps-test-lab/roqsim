"""Sensor plugin: Zivid 3 XL250 structured-light 3D camera, via ``mujoco.Renderer`` (GL, offscreen).

The Zivid 3 XL250 is an industrial structured-light 3D camera: a single imaging sensor yields both a
colour 2D image and a coloured 3D point cloud (with a projector at a 250 mm baseline). This plugin
renders the colour image + a depth pass off the model's ``zivid_color`` camera -- the same RGB-D
shape as ``oakd_camera`` -- standing in for the sensor's coloured point cloud (a dedicated
PointCloud2 publisher is out of scope; colour + depth + ``CameraInfo`` carry the same information for
a consumer that reprojects). Defaults follow the datasheet:

* 39 deg square FOV, 480x480 (a downscaled square render of the sensor's square 2816x2816 frame --
  see the ``zivid_color`` camera in ``zivid.xml`` for why it is not a native mode;
  ``width``/``height``/``fovy`` config still override it).
* ``rate_hz`` 5 Hz -- the XL250's typical 3D capture time is 250-1500 ms, i.e. a few Hz at most.
* Depth clipped to the recommended-to-extended working range (``clip_near`` 1.3 m, ``clip_far`` 5 m);
  returns outside it read as "no return" (inf), mirroring the sensor's operating distance.

Topic/frame naming approximates the ``zivid-ros`` driver (``color/...``, ``depth/...``,
``zivid_optical_frame``); the exact driver names (e.g. ``color/image_color``, ``points/xyzrgba``) can
be set via the ``topics:`` override (see ``camera_common.CameraPlugin``). Config: see
``oakd_camera`` / ``camera_common.CameraPlugin`` (``robot``, ``camera``, ``width``/``height``,
``fovy``, ``rate_hz``, ``frame_id``, ``clip_near``, ``clip_far``).

References:

* https://www.zivid.com -- Zivid 3 XL250 datasheet (FOV, focus/working distance, capture time).
* https://github.com/zivid/zivid-ros -- ROS 2 driver topic layout and frame names.
"""

from __future__ import annotations

from roqsim.context import Endpoint, SimContext

from .camera_common import join_topic
from .oakd_camera import OakDCameraPlugin

#: Colour image/CameraInfo sit under `color/`; depth under its own `depth/` (as in zivid-ros), so
#: the two share no prefix -- the base class's single DEFAULT_TOPIC_PREFIX cannot express that, hence
#: the `_configure_extra` override below.
DEPTH_PREFIX = "depth"


class ZividPlugin(OakDCameraPlugin):
    DEFAULT_CAMERA = "zivid_color"
    DEFAULT_FRAME_ID = "zivid_optical_frame"
    DEFAULT_TOPIC_PREFIX = "color"
    DEFAULT_RATE_HZ = 5.0
    DEFAULT_WIDTH = 480
    DEFAULT_HEIGHT = 480

    def __init__(self, config=None, *, name=None):
        # Working-range defaults differ from the OAK-D's (0.3-100 m): the XL250 operates at
        # 1.3-5 m (recommended 1.5-4 m). Apply them unless the world overrides.
        cfg = dict(config or {})
        cfg.setdefault("clip_near", 1.3)
        cfg.setdefault("clip_far", 5.0)
        super().__init__(cfg, name=name)

    def _configure_extra(self, ctx: SimContext, prefix: str, ns: str) -> None:
        # Same depth endpoint as the OAK-D, but published under `depth/image_raw` (Zivid's own
        # namespace) rather than under the colour prefix.
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
                    "topic": self.topic_override("depth") or join_topic(DEPTH_PREFIX, "image_raw"),
                    "frame_id": self.frame_id,
                    "encoding": "32FC1",
                }
            },
        )
        ctx.interface.add(self._depth_ep)
