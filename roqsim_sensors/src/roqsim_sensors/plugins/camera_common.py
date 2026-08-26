"""Shared MuJoCo-camera rendering base for RGB(-D) sensor plugins (``mujoco.Renderer``, GL).

Both ``oakd_camera`` and ``realsense_d435`` render from a named MuJoCo ``<camera>`` and publish an
``image`` + ``camera_info`` output endpoint the same way, so that plumbing lives here once:

* :func:`intrinsics_from_model` -- pinhole intrinsics from the MJCF camera's own ``resolution``/
  ``fovy`` (falling back to plugin config, then a per-sensor default), so the camera's optics are
  described once, in the model, not duplicated into plugin config.
* :class:`CameraPlugin` -- resolves the camera, owns a lazily-created ``mujoco.Renderer``, throttles
  capture to ``rate_hz``, skips rendering when no endpoint the render feeds reports a subscriber (see
  ``roqsim.context.Endpoint.has_subscribers``), and registers the endpoints. Subclasses set
  class-level defaults and may override ``_configure_extra``/``_capture_extra`` to add more outputs
  (e.g. depth).

Colour is published in both wire formats a real driver offers: a raw ``sensor_msgs/Image`` and a
``sensor_msgs/CompressedImage`` on ``<image topic>/compressed`` (``image_transport``'s convention).
Both read the *same* array -- the bridge's converter owns the codec, so nothing here imports one --
and both are ``lazy``, so an unsubscribed stream costs neither a render nor an encode. That is what
makes offering the second stream by default free: ``compressed: false`` opts out, ``jpeg_quality``
sets the quality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin
from roqsim.rendering import FrameRenderer

# image_transport's own default, and what real camera drivers ship with. Stated here rather than
# imported from the bridge: a sensor package must not depend on a transport backend (the endpoint
# names its ROS type as a string for the same reason), so the two defaults agree by both citing
# image_transport, not by sharing a symbol.
DEFAULT_JPEG_QUALITY = 95


@dataclass
class Intrinsics:
    """Pinhole camera intrinsics, shaped like a ``sensor_msgs/CameraInfo`` payload.

    ``d`` is the ``plumb_bob`` distortion vector the bridge publishes verbatim. It defaults to zeros
    because a MuJoCo render IS an ideal pinhole: a non-zero ``d`` here is a claim about the pixels,
    and is only true when :class:`CameraPlugin` was configured with ``distortion:`` and warped them
    to match. Never set it to a real camera's coefficients over an unwarped render -- that tells
    every consumer to undistort an image that was never distorted.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    d: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


def intrinsics_from_model(
    model,
    cam_id: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fovy: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    d: Sequence[float] | None = None,
    default_width: int = 640,
    default_height: int = 480,
) -> Intrinsics:
    """Pinhole intrinsics for MuJoCo camera ``cam_id``.

    Resolution comes from the MJCF ``resolution="w h"`` attribute (MuJoCo's "unset" sentinel is
    ``(1, 1)``) and can be overridden by plugin config.

    Two ways to get the rest, in precedence order:

    1. **The MJCF's own pinhole intrinsics** -- ``sensorsize`` + ``focalpixel``/``focal`` +
       ``principalpixel``/``principal``, which MuJoCo exposes as ``cam_sensorsize`` and
       ``cam_intrinsic`` and, crucially, RENDERS WITH. Used whenever ``sensorsize`` is set (its
       "unset" value is 0), because then the model describes a real lens: ``fx != fy`` and an
       off-centre principal point are both expressible, and the projection matrix already honours
       them.
    2. **``fovy``** -- the historical path, and still the default. One angle cannot express a
       non-central principal point, so ``cx, cy`` land at the image centre and ``fx == fy``.

    Plugin config overrides either, per field, and is applied last -- that is how a world states a
    measured calibration without editing a model other worlds share. Note the asymmetry this
    creates, and it is deliberate: ``fx``/``fy``/``cx``/``cy`` from config change the PUBLISHED
    intrinsics but not what MuJoCo rendered, so overriding them beyond what the model was built with
    makes ``camera_info`` disagree with the pixels. A world that wants a genuinely off-centre
    principal point must put it in the MJCF (path 1); config overrides are for stating what the
    model already renders.
    """
    mw, mh = (int(v) for v in model.cam_resolution[cam_id])
    if width is None and height is None and mw > 1 and mh > 1:
        width, height = mw, mh
    width = int(width) if width else default_width
    height = int(height) if height else default_height

    sw, sh = (float(v) for v in model.cam_sensorsize[cam_id])
    if fovy is None and sw > 0 and sh > 0:
        # cam_intrinsic is [focal_x_m, focal_y_m, principal_x_m, principal_y_m]; dividing by the
        # sensor size gives the fraction of the frame, so scaling by the pixel count converts to
        # pixels at whatever resolution this plugin renders -- the MJCF's own `resolution` need not
        # be the one in use. The `cy` sign flips because MJCF's principal +y points UP the image
        # while ROS's cy counts DOWN from the top row.
        fpx, fpy, ppx, ppy = (float(v) for v in model.cam_intrinsic[cam_id])
        m_fx, m_fy = fpx / sw * width, fpy / sh * height
        m_cx, m_cy = width / 2.0 + ppx / sw * width, height / 2.0 - ppy / sh * height
    else:
        fovy_deg = float(fovy) if fovy is not None else float(model.cam_fovy[cam_id])
        f = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
        m_fx = m_fy = f
        m_cx, m_cy = width / 2.0, height / 2.0

    return Intrinsics(
        width=width,
        height=height,
        fx=float(fx) if fx is not None else m_fx,
        fy=float(fy) if fy is not None else m_fy,
        cx=float(cx) if cx is not None else m_cx,
        cy=float(cy) if cy is not None else m_cy,
        d=tuple(float(v) for v in d) if d is not None else (0.0,) * 5,
    )


def _cv2_import_error() -> str | None:
    """``None`` when OpenCV imports, else the reason it does not.

    OpenCV is an optional extra here (a lidar-only world must not need it), so the distortion path
    checks for it rather than importing at module scope.
    """
    try:
        import cv2  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised by the missing-extra path
        return str(exc)
    return None


def _build_distortion_map(intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """The forward-distortion remap for ``intr``: ideal render -> what that lens would have imaged.

    THE DIRECTION IS THE WHOLE POINT, and it is the opposite of the familiar one. ``cv2.remap``
    samples the source at ``map[dst]``, so this must answer, for every pixel of the DISTORTED output,
    where that ray landed in the IDEAL render. That is :func:`cv2.undistortPoints` over the pixel
    grid -- *not* :func:`cv2.initUndistortRectifyMap`, which builds the inverse (it maps an
    undistorted destination back into a distorted source, because its job is rectifying a real
    image). Getting this backwards does not look wrong on a rendered frame; it silently doubles the
    error, which is why ``test_cameras.py`` checks it by undistorting the result rather than by eye.

    Built once, at configure: on a 1920x1080 frame this takes ~95 ms, against ~0.5 ms for the remap
    it saves per frame.
    """
    import cv2

    w, h = intr.width, intr.height
    k = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    m = cv2.undistortPoints(grid, k, np.asarray(intr.d, dtype=np.float64), P=k).reshape(h, w, 2)
    # .copy() because cv2.remap wants each map contiguous, and these are strided views of `m`.
    return m[..., 0].copy(), m[..., 1].copy()


def join_topic(*parts: str) -> str:
    return "/".join(p for p in parts if p)


def sibling_topic(topic: str, name: str) -> str:
    """The topic called ``name`` in ``topic``'s own namespace.

    ROS's convention for a ``camera_info`` beside its image, and the reason this is not
    :func:`join_topic` on the split parts: that drops empty parts, which turns an ABSOLUTE topic
    (``/cam/depth/image_raw``, as a world hardwires to match a driver) into a relative one that the
    bridge then puts under the robot's namespace.
    """
    namespace, slash, _ = topic.rpartition("/")
    return f"{namespace}{slash}{name}" if slash else name


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
          fx: null               # state the published focal length / principal point directly,
          fy: null               #   for a model whose MJCF already renders them (see
          cx: null               #   intrinsics_from_model); null means "take the model's"
          cy: null
          distortion: null       # [k1, k2, p1, p2, k3] -- WARPS the render and publishes these
                                 #   coefficients, so camera_info keeps describing its own pixels
          rate_hz: <DEFAULT_RATE_HZ>
          frame_id: <DEFAULT_FRAME_ID>
          compressed: true       # also publish <image topic>/compressed (CompressedImage)
          jpeg_quality: 95       # image_transport's default; only read when compressed is on
          topics: {}             # optional: hardwire absolute topics, e.g.
                                 #   {image: /camera/color/image_raw, camera_info: /camera/color/camera_info}
                                 # (overrides namespace+default; see Plugin.topic_override)
                                 # `image_compressed` follows `image` unless hardwired itself.
    """

    parallel_safe = False  # owns a private mujoco.Renderer (not safe to share/parallelize)

    DEFAULT_CAMERA = "camera"
    DEFAULT_FRAME_ID = "camera_optical_frame"
    DEFAULT_TOPIC_PREFIX = ""
    DEFAULT_RATE_HZ = 30.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        # The mount entity: whatever entry this camera is nested under -- a ``spawn_sensor`` on a
        # tripod, a ``spawn_robot``, or the ``spawn_arm`` an eye-in-hand camera arrives with from a
        # manifest. Ownership is where the entry sits, so a camera reads it rather than naming a
        # mount in its own config, and one attribute covers all three mounts.
        self.robot = self.entity
        self.camera = self.config.get("camera", self.DEFAULT_CAMERA)
        self.rate_hz = float(self.config.get("rate_hz", self.DEFAULT_RATE_HZ))
        self.frame_id = self.config.get("frame_id", self.DEFAULT_FRAME_ID)
        # A compressed companion to `image`, on by default: real camera drivers advertise both, and an
        # idle publisher costs nothing here because the endpoint is lazy (see configure()).
        self.compressed = bool(self.config.get("compressed", True))
        self.jpeg_quality = int(self.config.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
        self._width_cfg = self.config.get("width")
        self._height_cfg = self.config.get("height")
        self._fovy_cfg = self.config.get("fovy")
        self._intr_cfg = {k: self.config.get(k) for k in ("fx", "fy", "cx", "cy")}
        self._distortion_cfg = self.config.get("distortion")
        # The forward-distortion remap, built once in configure() -- see _build_distortion_map().
        self._dist_map: tuple[np.ndarray, np.ndarray] | None = None
        self._cam_id = -1
        self._intr: Intrinsics | None = None
        self._frames: FrameRenderer | None = None
        self._rgb: np.ndarray | None = None
        self._last_capture = float("-inf")
        self._image_ep: Endpoint | None = None
        self._compressed_ep: Endpoint | None = None
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
        if not 1 <= int(config.get("jpeg_quality", DEFAULT_JPEG_QUALITY)) <= 100:
            errors.append("'jpeg_quality' must be between 1 and 100")
        for key in ("fx", "fy"):
            if config.get(key) is not None and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        distortion = config.get("distortion")
        if distortion is not None:
            if not isinstance(distortion, (list, tuple)) or len(distortion) != 5:
                errors.append(
                    "'distortion' must be a list of 5 plumb_bob coefficients [k1,k2,p1,p2,k3]"
                )
            elif _cv2_import_error() is not None:
                # Loud, and at validate time rather than at the first frame: silently skipping the
                # warp would publish real coefficients over an unwarped render, which is the exact
                # inversion this feature exists to prevent.
                errors.append(
                    "'distortion' needs OpenCV, which is not installed "
                    f"({_cv2_import_error()}); install roqsim_sensors[distortion]"
                )
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
            d=self._distortion_cfg,
            default_width=self.DEFAULT_WIDTH,
            default_height=self.DEFAULT_HEIGHT,
            **{k: v for k, v in self._intr_cfg.items() if v is not None},
        )
        if self._distortion_cfg is not None:
            self._dist_map = _build_distortion_map(self._intr)

        # Resolved once: the compressed topic is derived from it, so a world that hardwires
        # `topics: {image: ...}` to match an external driver gets the matching `/compressed` for free.
        image_topic = self.topic_override("image") or join_topic(
            self.DEFAULT_TOPIC_PREFIX, "image_raw"
        )
        self._image_ep = Endpoint(
            name="image",
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=lambda: self._rgb,
            rate_hz=self.rate_hz,
            # A full raw frame is the most expensive thing this plugin can put on the wire (2.8 MB at
            # 1280x720), so it is not serialised while nothing is listening.
            lazy=True,
            backend={
                "ros2": {
                    "type": "sensor_msgs.msg.Image",
                    "topic": image_topic,
                    "frame_id": self.frame_id,
                    "encoding": "rgb8",
                }
            },
        )
        ctx.interface.add(self._image_ep)
        if self.compressed:
            # Same neutral payload as `image` -- one array, two wire formats. The bridge's converter
            # owns the codec, so this plugin never imports one, and `lazy` means the encode is paid
            # only while something subscribes to THIS topic (a raw-image consumer must not trigger it).
            self._compressed_ep = Endpoint(
                name="image_compressed",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._rgb,
                rate_hz=self.rate_hz,
                lazy=True,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.CompressedImage",
                        # `<image topic>/compressed` is image_transport's convention, which is what
                        # makes an unmodified driver-shaped consumer find it.
                        "topic": self.topic_override("image_compressed")
                        or join_topic(image_topic, "compressed"),
                        "frame_id": self.frame_id,
                        "encoding": "rgb8",
                        "format": "jpeg",
                        "quality": self.jpeg_quality,
                    }
                },
            )
            ctx.interface.add(self._compressed_ep)
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
        return [
            ep
            for ep in (self._image_ep, self._compressed_ep, *self._extra_outputs)
            if ep is not None
        ]

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
        if self._dist_map is not None:
            # Here, not in the endpoint's read(): `image` and `image_compressed` are two lazy views of
            # the SAME array, so warping downstream would either pay twice or warp twice. Depth is
            # deliberately untouched -- a real RealSense publishes an already-rectified depth stream
            # with D = 0, so distorting it would move the sim AWAY from the sensor.
            import cv2

            self._rgb = cv2.remap(
                self._rgb,
                self._dist_map[0],
                self._dist_map[1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        self._capture_extra(ctx, r)

    def _capture_extra(self, ctx: SimContext, renderer: mujoco.Renderer) -> None:
        """Hook for subclasses to capture additional passes (e.g. depth) off the same renderer."""

    def shutdown(self, ctx: SimContext) -> None:
        if self._frames is not None:
            self._frames.close()
            self._frames = None
