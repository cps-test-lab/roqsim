"""Per-sensor-type FOV adapters -- the only module that knows sensor specifics.

A small registry maps a sensor *type* to a function that builds a :class:`~.fov.SensorFov` from that
sensor's resolved parameters and world pose. This is the extension point for coverage: the sensor
*plugins* are never edited; a special or new sensor gets a new adapter registered here.

Adapters reuse each plugin's own resolution logic instead of re-hardcoding defaults (which would drift
from the plugins over time):

* the **camera** adapter reads the compiled MJCF via :func:`camera_common.intrinsics_from_model` -- the
  MJCF ``fovy``/``resolution`` *is* the source of truth -- when the camera exists in the world; for a
  hypothetical placement (not spawned) it builds the same :class:`Intrinsics` from the placement config.
* the **lidar/livox** adapters instantiate the plugin from the given config and read back its
  already-resolved attributes (``LidarPlugin(cfg).angle_min`` etc.), borrowing the plugin's default
  logic without importing its internals.

Two placement sources feed the adapters through :class:`PlacedSensor`:

* *in-world* -- the sensor is spawned; ``cam_id``/``site_id`` resolve pose (and camera intrinsics) from
  the model. Used by the ``sensor_coverage_probe`` plugin's ``sensors: auto`` discovery.
* *hypothetical* -- ``pos``/``rpy`` give the pose; FOV parameters come from config. Used by the CLI's
  placement search, which evaluates candidate mounts that are not (and need not be) spawned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..plugins.camera_common import Intrinsics, intrinsics_from_model
from ..plugins.lidar import LidarPlugin
from ..plugins.livox_mid360 import LivoxMid360Plugin
from .fov import FovKind, SensorFov

# Camera optics look along -z (MuJoCo convention). For a *hypothetical* placement we want rpy to read
# consistently with the lidars, where rpy=0 points the sensor along +x (world) with world-up as up.
# This base rotation maps the camera frame so that, at rpy=0, the optical axis (-z) points along +x
# and +y (up) points along +z_world; the placement's rpy is then applied on top (rot = R_rpy @ BASE).
_CAMERA_BASE = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


@dataclass
class PlacedSensor:
    """A sensor instance to evaluate: either spawned in the world or a hypothetical candidate."""

    sensor_type: str
    pos: np.ndarray | None = None  # (3,) world position -- hypothetical placement
    rpy: np.ndarray | None = (
        None  # (3,) roll/pitch/yaw [rad], fixed-axis XYZ -- hypothetical placement
    )
    cam_id: int = -1  # in-world MuJoCo camera id (pose + intrinsics from the model)
    site_id: int = -1  # in-world MuJoCo site id (pose from the model)
    config: dict = field(
        default_factory=dict
    )  # sensor config (FOV params + camera intrinsics overrides)
    label: str = ""


FovAdapter = Callable[[mujoco.MjModel, mujoco.MjData, PlacedSensor], SensorFov]

_REGISTRY: dict[str, FovAdapter] = {}


def register_adapter(*sensor_types: str):
    """Register ``fn`` as the FOV adapter for one or more sensor type names."""

    def _decorate(fn: FovAdapter) -> FovAdapter:
        for t in sensor_types:
            _REGISTRY[t] = fn
        return fn

    return _decorate


def registered_types() -> list[str]:
    return sorted(_REGISTRY)


def build_fov(model: mujoco.MjModel, data: mujoco.MjData, placed: PlacedSensor) -> SensorFov:
    """Build the :class:`SensorFov` for ``placed`` using the adapter registered for its type."""
    try:
        adapter = _REGISTRY[placed.sensor_type]
    except KeyError:
        raise KeyError(
            f"no coverage FOV adapter for sensor type {placed.sensor_type!r}; "
            f"known types: {registered_types()}. Register one in coverage/adapters.py."
        ) from None
    return adapter(model, data, placed)


def rpy_to_mat(rpy) -> np.ndarray:
    """world<-sensor rotation from roll/pitch/yaw (rad), fixed-axis XYZ (ROS/URDF), i.e. Rz@Ry@Rx."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _require_pose(placed: PlacedSensor) -> tuple[np.ndarray, np.ndarray]:
    if placed.pos is None or placed.rpy is None:
        raise ValueError(
            f"sensor {placed.label or placed.sensor_type!r} has no in-world id and no pos/rpy"
        )
    return np.asarray(placed.pos, dtype=np.float64).reshape(3), rpy_to_mat(placed.rpy)


def _own_body(model, *, cam_id: int = -1, site_id: int = -1) -> int:
    """The body a sensor is mounted on, for :attr:`SensorFov.body_exclude`; ``-1`` if unmounted.

    A *hypothetical* placement (pos/rpy, no in-world id) has no body and needs none: nothing of it
    exists to occlude. A spawned one always does, and its housing is in the way -- see
    :attr:`SensorFov.body_exclude`.
    """
    if cam_id >= 0:
        return int(model.cam_bodyid[cam_id])
    if site_id >= 0:
        return int(model.site_bodyid[site_id])
    return -1


# -- camera ------------------------------------------------------------------------------------------


@register_adapter(
    "oakd_camera", "realsense_d435", "realsense_d415", "realsense_d455", "zivid", "camera"
)
def camera_adapter(model, data, placed: PlacedSensor) -> SensorFov:
    cfg = placed.config
    if placed.cam_id >= 0:
        origin = np.array(data.cam_xpos[placed.cam_id], dtype=np.float64)
        rot = np.array(data.cam_xmat[placed.cam_id], dtype=np.float64).reshape(3, 3)
        intr = intrinsics_from_model(
            model,
            placed.cam_id,
            width=cfg.get("width"),
            height=cfg.get("height"),
            fovy=cfg.get("fovy"),
        )
    else:
        pos, r = _require_pose(placed)
        origin = pos
        rot = r @ _CAMERA_BASE
        width = int(cfg.get("width", 640))
        height = int(cfg.get("height", 480))
        fovy = float(cfg.get("fovy", 45.0))
        f = height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        intr = Intrinsics(width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)

    # Horizontal half-angle from the intrinsics; vertical from fovy. Stored for reference -- FRUSTUM
    # membership uses the intrinsics rectangle, not these bands, but they describe the sector honestly.
    h_half = float(np.arctan2(intr.width / 2.0, intr.fx))
    v_half = float(np.arctan2(intr.height / 2.0, intr.fy))
    # A camera has no physical far range: range_max is an explicit *detection-range assumption*. A
    # depth camera's clip_far is a clip, not a detection range -- keep range_is_physical False either way.
    near = float(cfg.get("near", cfg.get("clip_near", 0.05)))
    # A depth camera's clip_far is the honest detection-range assumption when far isn't given.
    far = float(cfg.get("far", cfg.get("range_max", cfg.get("clip_far", 10.0))))
    return SensorFov(
        kind=FovKind.FRUSTUM,
        origin=origin,
        rot=rot,
        h_fov=(-h_half, h_half),
        v_fov=(-v_half, v_half),
        range_min=near,
        range_max=far,
        range_is_physical=False,
        sensor_type=placed.sensor_type,
        intrinsics=intr,
        label=placed.label,
        body_exclude=_own_body(model, cam_id=placed.cam_id),
    )


# -- lidars ------------------------------------------------------------------------------------------


def _cone_pose(data, placed: PlacedSensor) -> tuple[np.ndarray, np.ndarray]:
    if placed.site_id >= 0:
        origin = np.array(data.site_xpos[placed.site_id], dtype=np.float64)
        rot = np.array(data.site_xmat[placed.site_id], dtype=np.float64).reshape(3, 3)
        return origin, rot
    return _require_pose(placed)


@register_adapter("lidar")
def lidar_adapter(model, data, placed: PlacedSensor) -> SensorFov:
    p = LidarPlugin(
        placed.config
    )  # __init__ only resolves config into attributes (no side effects)
    origin, rot = _cone_pose(data, placed)
    # A 2D lidar's FOV is a horizontal plane (zero vertical extent) -- that is honest, and points off
    # the plane are genuinely unseen. An analysis may widen it with a `v_fov` override (an assumption).
    v = placed.config.get("v_fov")
    v_fov = (float(v[0]), float(v[1])) if v else (0.0, 0.0)
    return SensorFov(
        kind=FovKind.CONE_BAND,
        origin=origin,
        rot=rot,
        h_fov=(p.angle_min, p.angle_max),
        v_fov=v_fov,
        range_min=p.range_min,
        range_max=p.range_max,
        range_is_physical=True,
        sensor_type=placed.sensor_type,
        label=placed.label,
        body_exclude=_own_body(model, site_id=placed.site_id),
    )


@register_adapter("livox_mid360")
def livox_adapter(model, data, placed: PlacedSensor) -> SensorFov:
    p = LivoxMid360Plugin(placed.config)
    origin, rot = _cone_pose(data, placed)
    return SensorFov(
        kind=FovKind.CONE_BAND,
        origin=origin,
        rot=rot,
        h_fov=(p.h_fov_min, p.h_fov_max),
        v_fov=(p.v_fov_min, p.v_fov_max),
        range_min=p.range_min,
        range_max=p.range_max,
        range_is_physical=True,
        sensor_type=placed.sensor_type,
        label=placed.label,
        body_exclude=_own_body(model, site_id=placed.site_id),
    )
