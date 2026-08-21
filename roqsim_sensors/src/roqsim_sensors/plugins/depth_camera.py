"""Shared depth-pass base for RGB-D sensor plugins: the render, the wire encoding, the endpoint.

:class:`~camera_common.CameraPlugin` owns the colour render; this adds the depth pass on top of it and
nothing device-specific, so an OAK-D, a RealSense and a Zivid are siblings here rather than one
subclassing another. A subclass sets its optics and its topic layout, and registers its depth
endpoint through :meth:`DepthCameraPlugin._add_depth_endpoints`.

Config (in addition to ``camera_common.CameraPlugin``'s)::

    <plugin short name>:
      clip_near: 0.3          # m; outside [clip_near, clip_far] a pixel reads "no return"
      clip_far: 100.0         # m
      depth_encoding: 32FC1   # or 16UC1 -- see below

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

**The compressed companion.** A ``16UC1`` camera also offers ``<depth topic>/compressedDepth`` (RVL,
lossless, roughly a fifth of the raw bytes), the transport a RealSense driver advertises for depth --
so ``compressed: false`` is the opt-out for both streams, colour and depth. It is absent under
``32FC1`` because the codec is 16-bit: that is the format's constraint, not a policy, and asking for
the topic anyway is an error rather than a silent no-op. Its encoder drops returns past 10 m
(``image_transport``'s ``depth_max`` default, which we mirror so the bytes match a driver's), so a
camera that sees further must lower ``clip_far`` or switch the companion off rather than publish two
depth topics that disagree.
"""

from __future__ import annotations

import numpy as np

from roqsim.context import Endpoint, SimContext

from .camera_common import CameraPlugin, sibling_topic

#: uint16 millimetres saturate here, so this is the largest range `16UC1` can carry.
MAX_16UC1_RANGE_M = 65.535

#: `compressedDepth`'s encoder zeroes everything past this before compressing (image_transport's own
#: `depth_max` default). Stated here rather than imported from the bridge, for the reason
#: `camera_common`'s JPEG quality is: a sensor package must not depend on a transport backend -- the
#: two agree by both citing image_transport, not by sharing a symbol.
COMPRESSED_DEPTH_MAX_M = 10.0

#: The codec `compressedDepth` uses for 16-bit depth, and the only one roqsim writes.
DEPTH_CODEC = "rvl"


class DepthCameraPlugin(CameraPlugin):
    DEFAULT_DEPTH_ENCODING = "32FC1"
    DEPTH_ENCODINGS = ("32FC1", "16UC1")

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.clip_near = float(self.config.get("clip_near", 0.3))
        self.clip_far = float(self.config.get("clip_far", 100.0))
        self.depth_encoding = str(self.config.get("depth_encoding", self.DEFAULT_DEPTH_ENCODING))
        self._depth: np.ndarray | None = None
        self._depth_ep: Endpoint | None = None
        self._depth_compressed_ep: Endpoint | None = None
        #: The encoded payload, cached for the frame in `_depth`; see `_depth_payload`.
        self._depth_wire: np.ndarray | None = None
        #: The "no return" mask, kept from the clip step so the conversion needs no `isfinite` pass.
        self._invalid: np.ndarray | None = None
        self._mm_scratch: np.ndarray | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = super().validate_config(config)
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

    def _add_depth_endpoints(self, ctx: SimContext, ns: str, topic: str, frame_id: str) -> Endpoint:
        """Register this camera's depth output(s) and return the raw image endpoint.

        Every depth camera registers through here, so the payload an endpoint reads and the
        ``encoding`` it advertises cannot drift apart -- publishing metres under a ``16UC1`` hint is
        a garbled image, not an error, at the far end. ``topic`` is the caller's already-resolved
        topic (each device has its own layout, which is why the endpoint is not built here from a
        prefix).
        """
        self._depth_ep = Endpoint(
            name="depth",
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=self._depth_payload,
            rate_hz=self.rate_hz,
            lazy=True,  # as expensive to serialise as the colour frame; see camera_common's `image`
            backend={
                "ros2": {
                    "type": "sensor_msgs.msg.Image",
                    "topic": topic,
                    "frame_id": frame_id,
                    "encoding": self.depth_encoding,
                }
            },
        )
        ctx.interface.add(self._depth_ep)
        # Gate the renderer on depth too: a consumer wanting only depth must still get frames.
        self._extra_outputs.append(self._depth_ep)

        # A depth stream needs its OWN intrinsics: a consumer that rectifies or reprojects depth
        # subscribes to the info topic beside the depth image, and given only the colour stream's it
        # waits forever. `camera_info` is a sibling of its image in the same namespace (ROS's own
        # convention, and what realsense-ros, zivid-ros and a Gazebo rgbd_camera all publish), so the
        # topic is derived from the resolved depth topic rather than spelled out per device -- a world
        # that hardwires the depth topic to match a driver gets the matching info topic with it.
        #
        # The payload is the colour intrinsics, because both streams come off ONE MuJoCo camera: real
        # hardware images depth through different optics with different intrinsics, and a plugin
        # rendering one camera cannot pretend otherwise. NOT in `_gate_endpoints`, and not lazy, for
        # the same reasons the colour info is neither: it needs no render and costs six floats.
        ctx.interface.add(
            Endpoint(
                name="depth_camera_info",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._intr,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.CameraInfo",
                        "topic": self.topic_override("depth_camera_info")
                        or sibling_topic(topic, "camera_info"),
                        "frame_id": frame_id,
                    }
                },
            )
        )
        if self.compressed and self.depth_encoding == "16UC1":
            # `<depth topic>/compressedDepth`, image_transport's convention, derived from the topic
            # resolved above -- so a world that hardwires the depth topic to match a driver gets the
            # matching compressed one without naming it twice. Same payload as the raw endpoint: one
            # array, two wire formats, and the codec belongs to the bridge.
            self._depth_compressed_ep = Endpoint(
                name="depth_compressed",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self._depth_payload,
                rate_hz=self.rate_hz,
                lazy=True,  # the encode is paid only while something subscribes to THIS topic
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.CompressedImage",
                        "topic": self.topic_override("depth_compressed")
                        or f"{topic}/compressedDepth",
                        "frame_id": frame_id,
                        "encoding": self.depth_encoding,
                        "format": DEPTH_CODEC,
                    }
                },
            )
            ctx.interface.add(self._depth_compressed_ep)
            self._extra_outputs.append(self._depth_compressed_ep)
        return self._depth_ep

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
        renderer.enable_depth_rendering()
        renderer.update_scene(ctx.data, camera=self._cam_id)
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()
        self._invalid = (depth < self.clip_near) | (depth > self.clip_far)
        depth[self._invalid] = np.inf
        self._depth = depth
        self._depth_wire = None  # a new frame invalidates the encoded copy of the last one
