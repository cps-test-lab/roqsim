"""Shared MuJoCo-camera rendering base for RGB(-D) sensor plugins (``mujoco.Renderer``, GL).

Both ``oakd_camera`` and ``realsense_d435`` render from a named MuJoCo ``<camera>`` and publish an
``image`` + ``camera_info`` output endpoint the same way, so that plumbing lives here once:

* :func:`intrinsics_from_model` -- pinhole intrinsics from the MJCF camera's own ``resolution``/
  ``fovy`` (falling back to plugin config, then a per-sensor default), so the camera's optics are
  described once, in the model, not duplicated into plugin config.
* :class:`CameraPlugin` -- resolves the camera, owns a lazily-created ``mujoco.Renderer``, throttles
  capture to ``rate_hz``, skips rendering when the ``image`` endpoint reports no subscribers (see
  ``roqsim.context.Endpoint.has_subscribers``), and registers the two endpoints. Subclasses set
  class-level defaults and may override ``_configure_extra``/``_capture_extra`` to add more outputs
  (e.g. depth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin
from roqsim.rendering import FrameRenderer


@dataclass
class Intrinsics:
    """Pinhole camera intrinsics, shaped like a ``sensor_msgs/CameraInfo`` payload."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_model(
    model,
    cam_id: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fovy: float | None = None,
    default_width: int = 640,
    default_height: int = 480,
) -> Intrinsics:
    """Pinhole intrinsics for MuJoCo camera ``cam_id``.

    Resolution comes from the MJCF ``resolution="w h"`` attribute (MuJoCo's "unset" sentinel is
    ``(1, 1)``); ``fovy`` comes from the MJCF ``fovy`` attribute (always a concrete value, default
    45 deg). Either can be overridden by plugin config. Square pixels are assumed (``fx == fy``),
    matching our earlier in-house nav prototype's original OAK-D driver.
    """
    mw, mh = (int(v) for v in model.cam_resolution[cam_id])
    if width is None and height is None and mw > 1 and mh > 1:
        width, height = mw, mh
    width = int(width) if width else default_width
    height = int(height) if height else default_height
    fovy = float(fovy) if fovy is not None else float(model.cam_fovy[cam_id])
    f = height / (2.0 * math.tan(math.radians(fovy) / 2.0))
    return Intrinsics(width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)


def join_topic(*parts: str) -> str:
    return "/".join(p for p in parts if p)


class CameraPlugin(Plugin):
    """Base for a ``post_step`` RGB(-D) camera rendered from a named MuJoCo ``<camera>``.

    Config::

        <plugin short name>:
          robot: robot
          namespace: ""          # transport scope (default: inherited from spawn_robot's namespace)
          camera: <DEFAULT_CAMERA>
          width: null            # override the MJCF's resolution
          height: null
          fovy: null             # override the MJCF's fovy (degrees)
          rate_hz: <DEFAULT_RATE_HZ>
          frame_id: <DEFAULT_FRAME_ID>
          topics: {}             # optional: hardwire absolute topics, e.g.
                                 #   {image: /camera/color/image_raw, camera_info: /camera/color/camera_info}
                                 # (overrides namespace+default; see Plugin.topic_override)
    """

    parallel_safe = False  # owns a private mujoco.Renderer (not safe to share/parallelize)

    DEFAULT_CAMERA = "camera"
    DEFAULT_FRAME_ID = "camera_optical_frame"
    DEFAULT_TOPIC_PREFIX = ""
    DEFAULT_RATE_HZ = 30.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        # The mount entity: a spawn_sensor/spawn_robot wires ``robot: <name>``, while an eye-in-hand
        # camera injected via an arm's manifest is wired ``arm: <name>`` (spawn_arm's target key).
        # Accept either so a camera can sit on a fixed mount, a mobile base, or a manipulator flange.
        self.robot = self.config.get("robot") or self.config.get("arm") or "robot"
        self.camera = self.config.get("camera", self.DEFAULT_CAMERA)
        self.rate_hz = float(self.config.get("rate_hz", self.DEFAULT_RATE_HZ))
        self.frame_id = self.config.get("frame_id", self.DEFAULT_FRAME_ID)
        self._width_cfg = self.config.get("width")
        self._height_cfg = self.config.get("height")
        self._fovy_cfg = self.config.get("fovy")
        self._cam_id = -1
        self._intr: Intrinsics | None = None
        self._frames: FrameRenderer | None = None
        self._rgb: np.ndarray | None = None
        self._last_capture = float("-inf")
        self._image_ep: Endpoint | None = None
        # Output endpoints a subclass adds that are fed by the SAME render pass (depth, point cloud).
        # They gate the renderer alongside `image` -- see _gate_endpoints().
        self._extra_outputs: list[Endpoint] = []

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", self.DEFAULT_RATE_HZ)) <= 0:
            errors.append("'rate_hz' must be > 0")
        for key in ("width", "height"):
            if config.get(key) is not None and int(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model
        self._cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, prefix + self.camera)
        if self._cam_id < 0:
            raise RuntimeError(f"{type(self).__name__}: camera {prefix + self.camera!r} not found")
        self._intr = intrinsics_from_model(
            m,
            self._cam_id,
            width=self._width_cfg,
            height=self._height_cfg,
            fovy=self._fovy_cfg,
            default_width=self.DEFAULT_WIDTH,
            default_height=self.DEFAULT_HEIGHT,
        )

        self._image_ep = Endpoint(
            name="image",
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=lambda: self._rgb,
            rate_hz=self.rate_hz,
            backend={
                "ros2": {
                    "type": "sensor_msgs.msg.Image",
                    "topic": self.topic_override("image")
                    or join_topic(self.DEFAULT_TOPIC_PREFIX, "image_raw"),
                    "frame_id": self.frame_id,
                    "encoding": "rgb8",
                }
            },
        )
        ctx.interface.add(self._image_ep)
        ctx.interface.add(
            Endpoint(
                name="camera_info",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._intr,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.CameraInfo",
                        "topic": self.topic_override("camera_info")
                        or join_topic(self.DEFAULT_TOPIC_PREFIX, "camera_info"),
                        "frame_id": self.frame_id,
                    }
                },
            )
        )
        self._configure_extra(ctx, prefix, ns)

    def _configure_extra(self, ctx: SimContext, prefix: str, ns: str) -> None:
        """Hook for subclasses to register additional endpoints (e.g. depth)."""

    def _gate_endpoints(self) -> list[Endpoint]:
        """The output endpoints whose subscribers justify a render.

        Every one that CARRIES A RENDER PASS, not just the colour image: a subclass's depth or point
        cloud is produced by ``_capture_extra`` off the same render, so gating on the colour endpoint
        alone means a consumer that wants only depth gets an endless stream of nothing. That is not
        hypothetical -- MoveIt's octomap updater subscribes to the point cloud and never to the colour
        image, and it silently saw an empty world until this looked at both.

        ``camera_info`` is deliberately NOT here: it needs no render, so a lone info subscriber (an
        rviz panel, say) must not switch the renderer on.
        """
        return [ep for ep in (self._image_ep, *self._extra_outputs) if ep is not None]

    def _due(self, ctx: SimContext) -> bool:
        if ctx.sim_time - self._last_capture < 1.0 / self.rate_hz:
            return False
        gates = self._gate_endpoints()
        # `has_subscribers is None` = no introspection available (no bridge, or a backend that cannot
        # tell); then render, because the alternative is a sensor that never produces anything.
        return any(ep.has_subscribers is None or ep.has_subscribers() for ep in gates)

    def post_step(self, ctx: SimContext) -> None:
        if not self._due(ctx):
            return
        self._last_capture = ctx.sim_time
        if self._frames is None:
            self._frames = FrameRenderer(
                ctx.model, self._intr.width, self._intr.height, camera=self._cam_id
            )
        r = self._frames.raw
        r.disable_depth_rendering()
        self._rgb = self._frames.render(ctx.data).copy()
        self._capture_extra(ctx, r)

    def _capture_extra(self, ctx: SimContext, renderer: mujoco.Renderer) -> None:
        """Hook for subclasses to capture additional passes (e.g. depth) off the same renderer."""

    def shutdown(self, ctx: SimContext) -> None:
        if self._frames is not None:
            self._frames.close()
            self._frames = None
