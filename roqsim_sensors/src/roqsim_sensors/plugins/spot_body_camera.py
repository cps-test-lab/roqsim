"""Sensor plugin: one body camera station of a Boston Dynamics Spot, as the ``spot_ros2`` driver publishes it.

A Spot carries five stations -- ``frontleft``, ``frontright``, ``left``, ``right``, ``back`` -- each a
projected-IR stereo pair with a fisheye imager. The robot's image service offers three sources per
station (``<station>_fisheye_image``, ``<station>_depth``, ``<station>_depth_in_visual_frame``; Spot SDK
``docs/python/understanding_spot_programming.md``), and the driver ``bdaiinstitute/spot_ros2``
(@ ``949feb7``) publishes them as::

    camera/<station>/image              sensor_msgs/Image      mono8 (greyscale bodies) or bgr8
    camera/<station>/camera_info        sensor_msgs/CameraInfo plumb_bob, D = 0, K from the pinhole
    camera/<station>/compressed         sensor_msgs/CompressedImage (off unless asked for)
    depth/<station>/image               sensor_msgs/Image      16UC1, millimetres, 0 = no return
    depth/<station>/camera_info
    depth_registered/<station>/image    depth in the fisheye image's frame and size
    depth_registered/<station>/camera_info

with QoS reliable + transient_local, depth 10, at 15 Hz (``spot_driver``:
``src/api/spot_image_sources.cpp``, ``src/images/images_middleware_handle.cpp``,
``src/conversions/decompress_images.cpp``, ``src/api/default_image_client.cpp``,
``include/spot_driver/api/middleware_handle_base.hpp``,
``include/spot_driver/interfaces/parameter_interface_base.hpp``). The fisheye image is stamped in
``<station>_fisheye`` and depth in ``<station>``, the frames of Spot's own tree (Spot SDK
``docs/concepts/geometry_and_frames.md``), which are already optical frames: z out of the lens, x and
y along the picture.

The images are rendered as rectilinear pinholes and published with D = 0, because that is what the
driver's ``camera_info`` says of the real images: it fills K from the image source's pinhole model and
publishes no distortion. Depth is in millimetres (``PIXEL_FORMAT_DEPTH_U16``, Spot SDK
``protos/bosdyn/api/image.proto``).

Bundled with the ``spot_body_camera`` device model, whose ``spot_fisheye`` and ``spot_depth`` cameras
carry each stream's lens.

Config (in addition to ``depth_camera.DepthCameraPlugin``'s)::

    spot_body_camera:
      station: frontleft     # the station's name, which every topic carries (required)
      depth_frame_id: <station>  # the frame depth is stamped in: the station's own
      color_encoding: mono8  # bgr8 for a Spot with colour body cameras (spot_ros2 rgb_cameras: true)
      registered_depth: true # depth_registered/<station>/image
      compressed: false      # spot_ros2's publish_compressed_images default
"""

from __future__ import annotations

from roqsim.context import SimContext

from .camera_common import join_topic
from .depth_camera import DepthCameraPlugin

#: The stations a Spot carries (Spot SDK ``docs/concepts/about_spot.md``: five stereo pairs).
STATIONS = ("frontleft", "frontright", "left", "right", "back")

#: spot_ros2's publisher QoS for every image topic: reliable, transient_local, keep the last 10.
SPOT_ROS2_IMAGE_QOS = {"reliability": "reliable", "durability": "transient_local", "depth": 10}


class SpotBodyCameraPlugin(DepthCameraPlugin):
    DEFAULT_CAMERA = "spot_fisheye"
    DEFAULT_DEPTH_CAMERA = "spot_depth"
    DEFAULT_FRAME_ID = "frontleft_fisheye"
    DEFAULT_RATE_HZ = 15.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_COLOR_ENCODING = "mono8"
    DEFAULT_DEPTH_ENCODING = "16UC1"
    #: The near plane is a modelling assumption; 4 m is the SDK's stated operating range (Spot SDK
    #: ``docs/concepts/about_spot.md``), where the Instructions for Use quote about 2 m.
    DEFAULT_CLIP_NEAR = 0.1
    DEFAULT_CLIP_FAR = 4.0
    IMAGE_LEAF = "image"
    COMPRESSED_TOPIC = "{image}/../compressed"
    ROS2_QOS = SPOT_ROS2_IMAGE_QOS

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        config = dict(config or {})
        config.setdefault("compressed", False)
        config.setdefault("registered_depth", True)
        super().__init__(config, name=name, entity=entity, label=label)
        self.station = str(self.config.get("station", ""))
        self.topic_prefix = join_topic("camera", self.station)
        self.depth_frame_id = self.config.get("depth_frame_id", self.station)

    def validate_config(self, config: dict) -> list[str]:
        errors = super().validate_config(config)
        if not config.get("station"):
            errors.append(f"'station' names the station, one of {', '.join(STATIONS)}")
        return errors

    def _configure_extra(self, ctx: SimContext, prefix: str) -> None:
        self._add_depth_endpoints(
            ctx,
            join_topic("depth", self.station, "image"),
            self.depth_frame_id,
            registered_topic=join_topic("depth_registered", self.station, "image"),
        )
