"""Shared depth-pass base for RGB-D sensor plugins: the render, the wire encoding, the endpoint.

:class:`~camera_common.CameraPlugin` owns the colour render; this adds the depth pass on top of it and
nothing device-specific, so an OAK-D, a RealSense and a Zivid are siblings here rather than one
subclassing another. A subclass sets its optics and its topic layout, and turns its depth
endpoints on through :meth:`DepthCameraPlugin._add_depth_endpoints`.

Config (in addition to ``camera_common.CameraPlugin``'s)::

    <plugin short name>:
      depth_camera: <DEFAULT_DEPTH_CAMERA>  # the MuJoCo camera depth is rendered from; default:
                              #   the device's own, else the colour `camera`
      depth_width: null       # override the depth camera's MJCF resolution (a separate depth
      depth_height: null      #   camera only; through the colour camera, `width`/`height` apply)
      clip_near: 0.3          # m; outside [clip_near, clip_far] a pixel reads "no return"
      clip_far: 100.0         # m
      depth_encoding: 32FC1   # or 16UC1 -- see below

**Which camera depth comes from.** Depth is rendered from the camera at the frame it is stamped in.
A device that images depth through its own optics -- a RealSense's stereo pair, whose depth frame is
not its colour frame -- has a second MuJoCo camera at its depth optical frame
(:data:`~camera_common.DEPTH_CAMERA_SUFFIX`), and a subclass names it as ``DEFAULT_DEPTH_CAMERA``:
depth, the depth ``camera_info`` and anything reprojected from them then come from that camera, at
its own FOV and resolution, while colour stays on ``camera``. A device whose driver aligns depth to
the colour image (an OAK-D's default) or whose depth and colour share one imager (a Zivid) renders
depth through ``camera`` itself, off the colour pass. A world overrides it with ``depth_camera:``,
and a camera that does not exist is refused at configure rather than replaced by the colour one.

**The two depth encodings, and why the choice exists.** ``self._depth`` is always float32 metres with
``inf`` for "no return": that is what a reprojection wants, and the point-cloud path consumes it
directly. But a real RealSense driver publishes ``16UC1`` -- **millimetres, with 0 for invalid** -- on
``depth/image_rect_raw``, so a stack (or a bag comparison) written against hardware sees a different
wire format than a ``32FC1`` sim run gives it. ``depth_encoding: 16UC1`` converts on the way out: the
device's own convention, half the bytes, and the precondition for ``compressedDepth``/RVL, which is a
16-bit codec.

``32FC1`` stays the default, because it is lossless in the unit the renderer produces. ``16UC1``
quantises to a millimetre and cannot represent a range beyond 65.535 m, so ``clip_far`` is validated
against that ceiling rather than silently saturating -- a clamp to 65535 would read as a surface
65.5 m away, which is a measurement, not an error.

**The compressed companion.** A ``16UC1`` camera also offers ``<depth topic>/compressedDepth``, the
transport a RealSense driver advertises for depth -- so ``compressed: false`` is the opt-out for both
streams, colour and depth. It is absent under ``32FC1`` because both codecs are 16-bit: that is the
format's constraint, not a policy, and asking for the topic anyway is an error rather than a silent
no-op. ``depth_codec`` picks between ``png`` (the default, and ``image_transport``'s own: 19 ms and
45 kB for a 1280x720 rendered frame) and ``rvl`` (42 ms, 370 kB, and what a driver configured for
speed emits -- the option when a bag has to match such a stream byte for byte). Both are lossless, so
the choice costs nothing but time and space. Its encoder drops returns past 10 m
(``image_transport``'s ``depth_max`` default, which we mirror so the bytes match a driver's), so a
camera that sees further must lower ``clip_far`` or switch the companion off rather than publish two
depth topics that disagree.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.rendering import FrameRenderer
from roqsim.types import CameraInfo, Image

from .camera_common import (
    CameraPlugin,
    Intrinsics,
    camera_info_of,
    intrinsics_from_model,
    sibling_topic,
)

#: uint16 millimetres saturate here, so this is the largest range `16UC1` can carry.
MAX_16UC1_RANGE_M = 65.535

#: `compressedDepth`'s encoder zeroes everything past this before compressing (image_transport's own
#: `depth_max` default). Stated here rather than imported from the bridge, for the reason
#: `camera_common`'s JPEG quality is: a sensor package must not depend on a transport backend -- the
#: two agree by both citing image_transport, not by sharing a symbol.
COMPRESSED_DEPTH_MAX_M = 10.0

#: The `compressedDepth` codecs, and the default. Both are lossless, so the choice is size and time:
#: on rendered depth (a z-buffer, no sensor noise, which a PNG row filter predicts almost exactly)
#: `png` measured 19 ms and 45 kB against `rvl`'s 42 ms and 370 kB at 1280x720. `rvl` is what a driver
#: configured for speed puts on the wire, so it is the option for byte-level parity with such a
#: stream -- its cost here is numpy against PNG's zlib in C, not the algorithm's.
DEPTH_CODECS = ("png", "rvl")
DEFAULT_DEPTH_CODEC = "png"


class DepthCameraPlugin(CameraPlugin):
    DEFAULT_DEPTH_ENCODING = "32FC1"
    DEPTH_ENCODINGS = ("32FC1", "16UC1")
    #: The camera depth is rendered from; ``None`` is the colour camera (see the module docstring).
    DEFAULT_DEPTH_CAMERA: str | None = None

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.depth_camera = (
            self.config.get("depth_camera") or self.DEFAULT_DEPTH_CAMERA or self.camera
        )
        self._depth_width_cfg = self.config.get("depth_width")
        self._depth_height_cfg = self.config.get("depth_height")
        self._depth_cam_id = -1
        self._depth_intr: Intrinsics | None = None
        #: The depth camera's own renderer, when it is not the colour camera.
        self._depth_frames: FrameRenderer | None = None
        self.clip_near = float(self.config.get("clip_near", 0.3))
        self.clip_far = float(self.config.get("clip_far", 100.0))
        self.depth_encoding = str(self.config.get("depth_encoding", self.DEFAULT_DEPTH_ENCODING))
        self.depth_codec = str(self.config.get("depth_codec", DEFAULT_DEPTH_CODEC))
        self._depth: np.ndarray | None = None
        #: The depth topic before the world's `topics:` rename, and its frame; set by
        #: `_add_depth_endpoints`, and empty on a device that publishes no depth.
        self._depth_topic = ""
        self._depth_frame_id = ""
        #: The encoded payload, cached for the frame in `_depth`; see `_depth_payload`.
        self._depth_wire: np.ndarray | None = None
        #: The "no return" mask, kept from the clip step so the conversion needs no `isfinite` pass.
        self._invalid: np.ndarray | None = None
        self._mm_scratch: np.ndarray | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = super().validate_config(config)
        for key in ("depth_width", "depth_height"):
            if config.get(key) is not None and int(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        depth_camera = config.get("depth_camera") or self.DEFAULT_DEPTH_CAMERA
        camera = config.get("camera", self.DEFAULT_CAMERA)
        if (depth_camera in (None, camera)) and (
            config.get("depth_width") is not None or config.get("depth_height") is not None
        ):
            errors.append(
                "'depth_width'/'depth_height' size a separate depth camera; this one renders depth "
                f"through the colour camera {camera!r}, which 'width'/'height' size"
            )
        if float(config.get("clip_near", 0.3)) < 0:
            errors.append("'clip_near' must be >= 0")
        if float(config.get("clip_far", 100.0)) <= float(config.get("clip_near", 0.3)):
            errors.append("'clip_far' must be > 'clip_near'")
        encoding = str(config.get("depth_encoding", self.depth_encoding))
        if encoding not in self.DEPTH_ENCODINGS:
            errors.append(
                f"'depth_encoding' must be one of {', '.join(self.DEPTH_ENCODINGS)}, "
                f"got {encoding!r}"
            )
        # Loudly at load time rather than per pixel at run time: a range this encoding cannot carry
        # would otherwise reach the wire as a wrong number, not as an error.
        elif (
            encoding == "16UC1" and float(config.get("clip_far", self.clip_far)) > MAX_16UC1_RANGE_M
        ):
            errors.append(
                f"'clip_far' must be <= {MAX_16UC1_RANGE_M} m with depth_encoding: 16UC1 "
                "(uint16 millimetres saturate there) -- lower it, or publish 32FC1"
            )
        codec = str(config.get("depth_codec", self.depth_codec))
        if codec not in DEPTH_CODECS:
            errors.append(f"'depth_codec' must be one of {', '.join(DEPTH_CODECS)}, got {codec!r}")
        compressed = bool(config.get("compressed", True))
        if encoding == "16UC1" and compressed:
            # compressedDepth's encoder zeroes returns past its depth_max, so a camera that sees
            # further would publish two depth topics that disagree beyond that distance -- with the
            # raw one right. Refuse the pair rather than ship the disagreement.
            if float(config.get("clip_far", self.clip_far)) > COMPRESSED_DEPTH_MAX_M:
                errors.append(
                    f"'clip_far' must be <= {COMPRESSED_DEPTH_MAX_M} m to offer the compressedDepth "
                    "topic (its encoder drops returns past that, so the raw and compressed streams "
                    "would disagree) -- lower it, or set 'compressed: false'"
                )
        elif (config.get("topics") or {}).get("depth_compressed"):
            errors.append(
                "topics['depth_compressed'] names a topic that will not exist: compressedDepth "
                "needs depth_encoding: 16UC1 (its codec is 16-bit) and 'compressed' left on"
            )
        return errors

    def _add_depth_endpoints(self, ctx: SimContext, topic: str, frame_id: str) -> None:
        """Turn on this camera's depth output(s), published on *topic* in *frame_id*.

        Every depth camera goes through here, so the payload an endpoint reads and the ``encoding``
        it carries cannot drift apart -- publishing metres as ``16UC1`` is a garbled image, not an
        error, at the far end. *topic* is the device's own layout (each device has its own, which is
        why it is not built here from a prefix); a world's ``topics: {depth: ...}`` renames it.
        """
        self._resolve_depth_camera(ctx)
        self._depth_topic = topic
        self._depth_frame_id = frame_id

    def _resolved_depth_topic(self) -> str:
        """Where ``depth`` is published: the world's rename, else the device's topic.

        The depth ``camera_info`` and ``compressedDepth`` topics are derived from it, so a world that
        hardwires the depth topic to match a driver gets the matching ones without naming them twice.
        """
        return self.topic_override("depth") or self._depth_topic

    # As expensive to serialise as the colour frame; see camera_common's `image`.
    @endpoint.out(
        name="depth",
        rate="rate_hz",
        lazy=True,
        when="_depth_topic",
        ros2=lambda self: {"topic": self._depth_topic, "frame_id": self._depth_frame_id},
    )
    def depth_image(self) -> Image | None:
        """The depth frame in the encoding this camera advertises; nothing before the first capture."""
        payload = self._depth_payload()
        return None if payload is None else Image(payload, self.depth_encoding)

    # A depth stream needs its OWN intrinsics: a consumer that rectifies or reprojects depth
    # subscribes to the info topic beside the depth image, and given only the colour stream's it
    # waits forever. `camera_info` is a sibling of its image in the same namespace (ROS's own
    # convention, and what realsense-ros, zivid-ros and a Gazebo rgbd_camera all publish), so the
    # topic is derived from the depth topic rather than spelled out per device.
    #
    # The payload is the depth camera's intrinsics -- the colour camera's only where depth is rendered
    # through it. Not lazy, and no render gate, for the same reasons the colour info is neither: it
    # needs no render and costs six floats.
    @endpoint.out(
        rate="rate_hz",
        when="_depth_topic",
        ros2=lambda self: {
            "topic": sibling_topic(self._resolved_depth_topic(), "camera_info"),
            "frame_id": self._depth_frame_id,
        },
    )
    def depth_camera_info(self) -> CameraInfo:
        """The pinhole intrinsics of the depth frame."""
        return camera_info_of(self._depth_intr)

    # `<depth topic>/compressedDepth`, image_transport's convention. Same payload as the raw
    # endpoint: one array, two wire formats, and the codec belongs to the bridge. Lazy: the encode is
    # paid only while something subscribes to THIS topic.
    @endpoint.out(
        rate="rate_hz",
        lazy=True,
        when=lambda self: (
            bool(self._depth_topic) and self.compressed and self.depth_encoding == "16UC1"
        ),
        ros2=lambda self: {
            "type": "sensor_msgs.msg.CompressedImage",
            "topic": f"{self._resolved_depth_topic()}/compressedDepth",
            "frame_id": self._depth_frame_id,
            "format": self.depth_codec,
        },
    )
    def depth_compressed(self) -> Image | None:
        """The depth frame, compressedDepth-encoded on the wire."""
        return self.depth_image()

    def _resolve_depth_camera(self, ctx: SimContext) -> None:
        """The depth camera's id and intrinsics: the colour camera's own when depth renders through it."""
        if self.depth_camera == self.camera:
            self._depth_cam_id, self._depth_intr = self._cam_id, self._intr
            return
        name = self._prefix + self.depth_camera
        self._depth_cam_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self._depth_cam_id < 0:
            raise RuntimeError(
                f"{type(self).__name__}: depth camera {name!r} not found. A model that renders "
                f"depth through its colour camera says so with 'depth_camera: {self.camera}'."
            )
        self._depth_intr = intrinsics_from_model(
            ctx.model,
            self._depth_cam_id,
            width=self._depth_width_cfg,
            height=self._depth_height_cfg,
            default_width=self.DEFAULT_WIDTH,
            default_height=self.DEFAULT_HEIGHT,
        )

    def _depth_payload(self) -> np.ndarray | None:
        """The depth image in the encoding this camera advertises.

        Cached for the current frame: with more than one depth topic subscribed the bridge reads the
        same frame once per endpoint, and the conversion must not be paid twice. `lazy=True` on those
        endpoints means an unsubscribed frame is never converted at all.
        """
        if self._depth is None or self.depth_encoding == "32FC1":
            return self._depth
        if self._depth_wire is None:
            self._depth_wire = self._to_millimetres(self._depth)
        return self._depth_wire

    def _to_millimetres(self, depth: np.ndarray) -> np.ndarray:
        """float32 metres (``inf`` = no return) -> uint16 millimetres (0 = no return)."""
        if self._mm_scratch is None or self._mm_scratch.shape != depth.shape:
            self._mm_scratch = np.empty(depth.shape, dtype=np.float32)
        mm = self._mm_scratch
        np.multiply(depth, 1000.0, out=mm)
        # Round rather than truncate: a cast alone biases every reading down by up to a millimetre.
        np.rint(mm, out=mm)
        # Floor the "no return" pixels BEFORE the cast -- inf to uint16 is undefined, and 0 is the
        # device's own marker for a pixel it could not see. Valid pixels all fit: `clip_far` is
        # validated against MAX_16UC1_RANGE_M.
        mm[self._invalid] = 0.0
        # A fresh array, not the scratch: the payload leaves the plugin, and the next capture would
        # rewrite a buffer a consumer still held (the colour path copies for the same reason).
        return mm.astype(np.uint16)

    def _capture_extra(self, ctx: SimContext, renderer) -> None:
        if self._depth_cam_id != self._cam_id:
            if self._depth_frames is None:
                intr = self._depth_intr
                self._depth_frames = FrameRenderer(
                    ctx.model, intr.width, intr.height, camera=self._depth_cam_id
                )
            renderer = self._depth_frames.raw
        renderer.enable_depth_rendering()
        renderer.update_scene(ctx.data, camera=self._depth_cam_id)
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()
        self._invalid = (depth < self.clip_near) | (depth > self.clip_far)
        depth[self._invalid] = np.inf
        self._depth = depth
        self._depth_wire = None  # a new frame invalidates the encoded copy of the last one

    def _reset_extra(self, ctx: SimContext) -> None:
        self._depth = None
        self._depth_wire = None

    def shutdown(self, ctx: SimContext) -> None:
        if self._depth_frames is not None:
            self._depth_frames.close()
            self._depth_frames = None
        super().shutdown(ctx)
