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
"""

from __future__ import annotations

import numpy as np

from roqsim.context import Endpoint, SimContext

from .camera_common import CameraPlugin

#: uint16 millimetres saturate here, so this is the largest range `16UC1` can carry.
MAX_16UC1_RANGE_M = 65.535


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
