"""Sensor plugin: Intel RealSense D435(i) colour + depth + point cloud, via ``mujoco.Renderer``.

Topic/frame naming follows ``realsense-ros``'s conventions, so a world can point an unmodified
RealSense-based stack at it:

===============  ==========================================  =============================
output           topic                                       frame
===============  ==========================================  =============================
colour           ``<ns>/camera/color/image_raw``              ``camera_color_optical_frame``
colour info      ``<ns>/camera/color/camera_info``            ``camera_color_optical_frame``
depth            ``<ns>/camera/depth/image_rect_raw``         ``camera_depth_optical_frame``
depth info       ``<ns>/camera/depth/camera_info``            ``camera_depth_optical_frame``
point cloud      ``<ns>/camera/depth/color/points``           ``camera_depth_optical_frame``
IMU (D435i)      ``<ns>/camera/imu``                          ``camera_imu_optical_frame``
===============  ==========================================  =============================

The IMU is not this plugin's: the D435i's inertial module is a separate device inside the same
housing, so it is an ``imu`` component in ``d435.manifest.yaml`` (with the vendor's extrinsic) rather
than another stream rendered here. It arrives with ``spawn_sensor: {model: d435}`` and is switched off
per world with ``enabled: false``; the row is listed because a consumer looking for the device's
topics should find all of them in one table.

Depth and the cloud are both **opt-in** (``depth:``/``points:``), and ``points`` implies ``depth``.
The reason the cloud is not free: it reprojects every valid pixel, so a 640x480 frame is up to 307k
points per capture. A consumer that only wants an occupancy map often does better subscribing to the
depth image directly (MoveIt's ``DepthImageOctomapUpdater``); the cloud exists for the pipeline shape
the OM-X palm-harvesting benchmark reconstructs, which is D435 -> PCL cloud -> OctoMap.

Reference frames. The cloud is emitted in the **ROS optical convention** -- x right, y down, z along
the view direction -- because that is what ``realsense-ros`` publishes and what every depth-image
consumer assumes. A MuJoCo camera looks down its own ``-z`` with ``+y`` up, so the two conventions
differ by a fixed rotation; a world must publish the static transform from the mount body to
``camera_depth_optical_frame`` itself (this plugin publishes no TF, like every other sensor here).

Config (also inherits ``camera_common.CameraPlugin``'s own fields, undocumented here)::

    realsense_d435:
      depth: true         # publish the depth image (default: false)
      points: true        # publish a PointCloud2 -- implies depth (default: false)
      clip_near: 0.28     # m; the D435's minimum-Z. Outside [clip_near, clip_far] reads "no return"
      clip_far: 3.0       # m; the datasheet's usable range at default settings
      depth_frame_id: camera_depth_optical_frame
      depth_encoding: 32FC1  # or 16UC1 -- millimetres, 0 for invalid, as realsense-ros publishes it

References:

* https://github.com/IntelRealSense/realsense-ros -- topic layout, frame names, ``CameraInfo`` shape.
* https://www.intelrealsense.com/depth-camera-d435i/ -- D435i data sheet. Colour FOV 69.4 x 42.5 deg;
  DEPTH FOV 87 x 58 deg; min-Z ~0.28 m at 848x480; usable range ~0.3-3 m.

Note on FOV. One MuJoCo camera has one ``fovy``, while a real D435 images colour and depth through
different optics. A model that ships a ``d435_color`` camera therefore has to pick: the bundled
``d435`` model uses the colour FOV, and the OpenMANIPULATOR-X's eye-in-hand camera uses the *depth*
FOV (58 deg), because the depth path is the one its experiment consumes. Override ``fovy`` in plugin
config to choose per world.
"""

from __future__ import annotations

import numpy as np

from roqsim.context import Endpoint, SimContext

from .camera_common import join_topic
from .depth_camera import DepthCameraPlugin
from .payloads import PointCloud

#: realsense-ros publishes depth and the coloured cloud under `depth/`, and colour under `color/`,
#: so the two share no prefix -- the base class's single DEFAULT_TOPIC_PREFIX cannot express that.
DEPTH_PREFIX = "camera/depth"


class RealsenseD435Plugin(DepthCameraPlugin):
    DEFAULT_CAMERA = "d435_color"
    DEFAULT_FRAME_ID = "camera_color_optical_frame"
    DEFAULT_TOPIC_PREFIX = "camera/color"
    DEFAULT_RATE_HZ = 30.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_DEPTH_FRAME_ID = "camera_depth_optical_frame"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        cfg = dict(config or {})
        # The D435's own working range, not the OAK-D's 0.3-100 m: min-Z ~0.28 m, usable to ~3 m.
        # A wrong clip_far is not cosmetic here -- it is what decides whether the wall behind the
        # subject ends up in the occupancy map as an obstacle.
        cfg.setdefault("clip_near", 0.28)
        cfg.setdefault("clip_far", 3.0)
        super().__init__(cfg, name=name, entity=entity, label=label)
        self.points = bool(self.config.get("points", False))
        # `points` implies `depth`: the cloud IS the depth render, reprojected.
        self.depth = bool(self.config.get("depth", False)) or self.points
        self.depth_frame_id = self.config.get("depth_frame_id", self.DEFAULT_DEPTH_FRAME_ID)
        self._cloud: PointCloud | None = None
        self._points_ep: Endpoint | None = None
        self._uv: tuple[np.ndarray, np.ndarray] | None = None

    def _configure_extra(self, ctx: SimContext, prefix: str, ns: str) -> None:
        if not self.depth:
            return
        self._add_depth_endpoints(
            ctx,
            ns,
            self.topic_override("depth") or join_topic(DEPTH_PREFIX, "image_rect_raw"),
            self.depth_frame_id,
        )

        if not self.points:
            return
        self._points_ep = Endpoint(
            name="points",
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=lambda: self._cloud,
            rate_hz=self.rate_hz,
            lazy=True,  # as expensive to serialise as the colour frame; see camera_common's `image`
            backend={
                "ros2": {
                    "type": "sensor_msgs.msg.PointCloud2",
                    "topic": self.topic_override("points")
                    or join_topic(DEPTH_PREFIX, "color/points"),
                    "frame_id": self.depth_frame_id,
                }
            },
        )
        ctx.interface.add(self._points_ep)
        self._extra_outputs.append(self._points_ep)

    def _capture_extra(self, ctx: SimContext, renderer) -> None:
        if not self.depth:
            return
        super()._capture_extra(ctx, renderer)
        if self.points:
            self._cloud = PointCloud(points=self._reproject(self._depth))

    def _reset_extra(self, ctx: SimContext) -> None:
        super()._reset_extra(ctx)
        self._cloud = None

    def _reproject(self, depth: np.ndarray) -> np.ndarray:
        """Depth image -> (N, 3) float32 XYZ in the ROS optical frame (x right, y down, z forward).

        ``mujoco.Renderer``'s depth pass gives the distance along the view direction (a z-buffer in
        metres, not a radial range), which is exactly the ``z`` a pinhole reprojection wants. Rows go
        DOWN the image while the MuJoCo camera's ``+y`` points up, so image ``v`` already runs along
        the optical frame's ``+y`` -- no sign flip beyond the intrinsics' principal point.

        ``inf`` marks "no return" (see ``clip_near``/``clip_far``); those pixels are dropped rather
        than emitted at some sentinel range, so an occupancy map never carves free space out of a
        pixel the sensor could not see.
        """
        intr = self._intr
        h, w = depth.shape
        if self._uv is None or self._uv[0].shape != depth.shape:
            v, u = np.meshgrid(
                np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
            )
            self._uv = ((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy)
        ux, vy = self._uv
        valid = np.isfinite(depth)
        z = depth[valid]
        return np.stack((ux[valid] * z, vy[valid] * z, z), axis=1).astype(np.float32, copy=False)
