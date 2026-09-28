"""The one table from roqsim's neutral types to ROS messages, and how an endpoint binds to ROS.

Free of ROS imports: the converters fill and read messages they are handed, so the table, the
resolution of an endpoint's effective hints and ``roqsim plugins describe`` work without a sourced
ROS.

**The table** (:data:`TYPES`) maps each type of :mod:`roqsim.types`, and ``bool``/``float``/
``int``/``str``, to its ROS message and a converter each way:

=====================  =================================================================
``Twist``              ``geometry_msgs/Twist`` (``stamped: true``: ``TwistStamped``)
``Pose``               ``geometry_msgs/PoseStamped`` (``stamped: false``: ``Pose``)
``Odometry``           ``nav_msgs/Odometry``
``JointState``         ``sensor_msgs/JointState``
``JointPositions``     ``trajectory_msgs/JointTrajectory`` (one point; inbound, the last)
``Wrench``             ``geometry_msgs/WrenchStamped`` (``stamped: false``: ``Wrench``)
``Imu``                ``sensor_msgs/Imu``
``LaserScan``          ``sensor_msgs/LaserScan``
``Image``              ``sensor_msgs/Image`` (or ``type: sensor_msgs.msg.CompressedImage``)
``CameraInfo``         ``sensor_msgs/CameraInfo``
``PointCloud``         ``sensor_msgs/PointCloud2`` (x, y, z float32)
``bool`` ``float``     ``std_msgs/Bool``, ``std_msgs/Float64``,
``int`` ``str``        ``std_msgs/Int64``, ``std_msgs/String``
=====================  =================================================================

A package adds its own type once through the ``roqsim.ros2_types`` entry-point group
(:data:`ENTRY_POINT_GROUP`): the entry loads a :class:`RosType`, or an iterable of them.

**Resolving an endpoint** (:func:`resolve`) gives the hints the bridge binds with -- the message
``type`` (or ``service``), the ``topic`` (or service ``name``), the ``qos`` profile and the
producer's own hints -- and the converters. For a decorated endpoint:

* the payload type's row decides the message, and the producer's hints only deviate from it: a
  frame id, ``stamped``, ``emit_tf``, a ``static_tf``, a ``qos``, a ``field`` of a structure to
  publish alone (whose type then decides), or a ``type`` naming one of the row's messages;
* a payload type with no row is mapped **by field name** onto the message its ``type`` hint names:
  each dataclass field (or each parameter, for an endpoint taking plain parameters) must be a field
  of the message of a matching kind, nested dataclasses to nested messages, and the bridge refuses
  the binding naming every mismatch (:meth:`Binding.prepare`) rather than dropping one;
* with neither, the endpoint is not on ROS: :func:`describe` says so and the bridge logs it;
* a command without parameters is a ``std_srvs/Trigger`` service;
* the topic is the world's ``topics:`` name (``Endpoint.topic``), else the hint's, else the
  endpoint's name; the QoS is the world's ``qos:`` (``Endpoint.qos``), else the hint's, else
  ``default``.

A hand-built endpoint without a schema keeps its hint block as it is, with the per-message-type
converters and decoders of :mod:`roqsim_ros_bridge.registry`.
"""

from __future__ import annotations

import array
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np

from roqsim import types as T
from roqsim.endpoint import ValueType, qos_profile

from .frames import namespaced

#: Entry-point group through which a package maps its own types (:class:`RosType`).
ENTRY_POINT_GROUP = "roqsim.ros2_types"

Fill = Callable[[Any, Any, Any, dict], None]  # (msg, value, stamp, hints)
Decode = Callable[[Any], Any]  # msg -> value


@dataclass(frozen=True)
class Wire:
    """One ROS message a type travels as, and its converters."""

    msg: str
    fill: Fill
    decode: Decode


@dataclass(frozen=True)
class RosType:
    """A neutral type's ROS mapping: the messages it travels as, the first being the default.

    ``stamped`` names the pair a ``stamped`` hint chooses between, ``(unstamped, stamped)``.
    ``hints`` are defaults under every endpoint's own (a frame id the converter falls back on).
    """

    cls: type
    wires: tuple[Wire, ...]
    stamped: tuple[str, str] | None = None
    hints: Mapping[str, Any] = field(default_factory=dict)

    def wire(self, msg: str) -> Wire | None:
        return next((w for w in self.wires if w.msg == msg), None)


# -- helpers ---------------------------------------------------------------------------------------
def as_f32(values) -> array.array:
    """A ``float32[]`` field's value: ``array.array`` from the numpy buffer, accepted on every distro."""
    return array.array("f", np.ascontiguousarray(values, dtype=np.float32).tobytes())


def as_f64(values) -> array.array:
    return array.array("d", np.ascontiguousarray(values, dtype=np.float64).tobytes())


def frame(hints: dict, key: str, default: str) -> str:
    """A frame id from the hints, namespaced by the bridge (``frame_prefix``) unless global."""
    return namespaced(hints.get("frame_prefix", ""), hints.get(key, default))


def _header(msg, stamp, frame_id: str | None) -> None:
    msg.header.stamp = stamp
    if frame_id is not None:
        msg.header.frame_id = frame_id


def _set_xyz(vec, values) -> None:
    vec.x, vec.y, vec.z = (float(v) for v in values)


def _xyz(vec) -> np.ndarray:
    return np.array([vec.x, vec.y, vec.z], dtype=np.float64)


def _set_quat(q, wxyz) -> None:
    """ROS orders a quaternion (x, y, z, w); the neutral types, like MuJoCo, (w, x, y, z)."""
    w, x, y, z = (float(v) for v in wxyz)
    q.x, q.y, q.z, q.w = x, y, z, w


def _wxyz(q) -> np.ndarray:
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


# -- converters ------------------------------------------------------------------------------------
def _fill_twist(msg, v: T.Twist, stamp, hints) -> None:
    _set_xyz(msg.linear, (v.vx, v.vy, v.vz))
    _set_xyz(msg.angular, (v.wx, v.wy, v.wz))


def _decode_twist(msg) -> T.Twist:
    return T.Twist(
        msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z
    )


def _fill_twist_stamped(msg, v, stamp, hints) -> None:
    _header(msg, stamp, frame(hints, "frame_id", "base_link"))
    _fill_twist(msg.twist, v, stamp, hints)


def _pose_frame(v: T.Pose, hints) -> str:
    return namespaced(hints.get("frame_prefix", ""), v.frame_id or hints.get("frame_id", "world"))


def _fill_pose(msg, v: T.Pose, stamp, hints) -> None:
    _set_xyz(msg.position, v.position)
    _set_quat(msg.orientation, v.orientation)


def _decode_pose(msg) -> T.Pose:
    return T.Pose(_xyz(msg.position), _wxyz(msg.orientation))


def _fill_pose_stamped(msg, v: T.Pose, stamp, hints) -> None:
    _header(msg, stamp, _pose_frame(v, hints))
    _fill_pose(msg.pose, v, stamp, hints)


def _decode_pose_stamped(msg) -> T.Pose:
    pose = _decode_pose(msg.pose)
    pose.frame_id = msg.header.frame_id
    return pose


def _fill_odometry(msg, v: T.Odometry, stamp, hints) -> None:
    _header(msg, stamp, frame(hints, "frame_id", "odom"))
    msg.child_frame_id = frame(hints, "child_frame_id", "base_link")
    _set_xyz(msg.pose.pose.position, v.position)
    _set_quat(msg.pose.pose.orientation, v.orientation)
    _set_xyz(msg.twist.twist.linear, v.linear)
    _set_xyz(msg.twist.twist.angular, v.angular)


def _decode_odometry(msg) -> T.Odometry:
    return T.Odometry(
        _xyz(msg.pose.pose.position),
        _wxyz(msg.pose.pose.orientation),
        _xyz(msg.twist.twist.linear),
        _xyz(msg.twist.twist.angular),
    )


def _fill_joint_state(msg, v: T.JointState, stamp, hints) -> None:
    msg.header.stamp = stamp
    msg.name = list(v.names)
    msg.position = as_f64(v.positions)
    msg.velocity = as_f64(v.velocities)
    if v.efforts is not None:
        msg.effort = as_f64(v.efforts)


def _decode_joint_state(msg) -> T.JointState:
    return T.JointState(
        list(msg.name),
        np.array(msg.position, dtype=np.float64),
        np.array(msg.velocity, dtype=np.float64),
        np.array(msg.effort, dtype=np.float64) if len(msg.effort) else None,
    )


def _fill_joint_trajectory(msg, v: T.JointPositions, stamp, hints) -> None:
    msg.header.stamp = stamp
    msg.joint_names = list(v.names)
    from trajectory_msgs.msg import JointTrajectoryPoint

    point = JointTrajectoryPoint()
    point.positions = as_f64(v.positions)
    msg.points = [point]


def _decode_joint_trajectory(msg) -> T.JointPositions:
    """The last point: a streamed single-point trajectory's target (what moveit_servo sends)."""
    positions = list(msg.points[-1].positions) if msg.points else []
    return T.JointPositions(list(msg.joint_names), np.array(positions, dtype=np.float64))


def _fill_wrench(msg, v: T.Wrench, stamp, hints) -> None:
    _set_xyz(msg.force, v.force)
    _set_xyz(msg.torque, v.torque)


def _decode_wrench(msg) -> T.Wrench:
    return T.Wrench(_xyz(msg.force), _xyz(msg.torque))


def _fill_wrench_stamped(msg, v: T.Wrench, stamp, hints) -> None:
    _header(msg, stamp, frame(hints, "frame_id", "world"))
    _fill_wrench(msg.wrench, v, stamp, hints)


def _decode_wrench_stamped(msg) -> T.Wrench:
    return _decode_wrench(msg.wrench)


def _diag3(variance: float) -> list[float]:
    v = float(variance)
    return [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]


def _fill_imu(msg, v: T.Imu, stamp, hints) -> None:
    """REP 145: proper acceleration, and ``orientation_covariance[0] = -1`` for no attitude."""
    _header(msg, stamp, frame(hints, "frame_id", "imu_link"))
    if v.orientation_valid:
        _set_quat(msg.orientation, v.orientation)
        msg.orientation_covariance = _diag3(v.orientation_variance)
    else:
        cov = _diag3(0.0)
        cov[0] = -1.0
        msg.orientation_covariance = cov
    _set_xyz(msg.angular_velocity, v.angular_velocity)
    _set_xyz(msg.linear_acceleration, v.linear_acceleration)
    msg.angular_velocity_covariance = _diag3(v.angular_velocity_variance)
    msg.linear_acceleration_covariance = _diag3(v.linear_acceleration_variance)


def _decode_imu(msg) -> T.Imu:
    valid = msg.orientation_covariance[0] != -1.0
    return T.Imu(
        orientation=_wxyz(msg.orientation) if valid else np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=_xyz(msg.angular_velocity),
        linear_acceleration=_xyz(msg.linear_acceleration),
        orientation_valid=bool(valid),
        orientation_variance=float(msg.orientation_covariance[0]) if valid else 0.0,
        angular_velocity_variance=float(msg.angular_velocity_covariance[0]),
        linear_acceleration_variance=float(msg.linear_acceleration_covariance[0]),
    )


def _fill_scan(msg, v: T.LaserScan, stamp, hints) -> None:
    _header(msg, stamp, frame(hints, "frame_id", "lidar"))
    msg.angle_min = float(v.angle_min)
    msg.angle_max = float(v.angle_max)
    msg.angle_increment = float(v.angle_increment)
    msg.range_min = float(v.range_min)
    msg.range_max = float(v.range_max)
    msg.ranges = as_f32(v.ranges)


def _decode_scan(msg) -> T.LaserScan:
    return T.LaserScan(
        np.array(msg.ranges, dtype=np.float32),
        msg.angle_min,
        msg.angle_max,
        msg.angle_increment,
        msg.range_min,
        msg.range_max,
    )


#: Bytes per pixel, and the numpy layout, of the encodings an Image carries.
_ENCODINGS = {
    "rgb8": (3, np.uint8, 3),
    "mono8": (1, np.uint8, 1),
    "16UC1": (2, np.uint16, 1),
    "32FC1": (4, np.float32, 1),
}


def _fill_image(msg, v: T.Image, stamp, hints) -> None:
    from .registry import fill_image

    fill_image(msg, v.data, stamp, {**hints, "encoding": v.encoding})


def _decode_image(msg) -> T.Image:
    try:
        _, dtype, channels = _ENCODINGS[msg.encoding]
    except KeyError:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; expected one of {sorted(_ENCODINGS)}"
        ) from None
    data = np.frombuffer(bytes(msg.data), dtype=dtype)
    shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
    return T.Image(data.reshape(shape).copy(), msg.encoding)


def _fill_compressed_image(msg, v: T.Image, stamp, hints) -> None:
    from .registry import fill_compressed_image

    fill_compressed_image(msg, v.data, stamp, {**hints, "encoding": v.encoding})


def _decode_compressed_image(msg) -> T.Image:
    raise TypeError("a CompressedImage is published, not taken: decode it to an Image upstream")


def _fill_camera_info(msg, v: T.CameraInfo, stamp, hints) -> None:
    _header(msg, stamp, frame(hints, "frame_id", "camera_optical_frame"))
    msg.height, msg.width = int(v.height), int(v.width)
    msg.distortion_model = "plumb_bob"
    msg.d = [float(x) for x in v.d]
    msg.k = [v.fx, 0.0, v.cx, 0.0, v.fy, v.cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [v.fx, 0.0, v.cx, 0.0, 0.0, v.fy, v.cy, 0.0, 0.0, 0.0, 1.0, 0.0]


def _decode_camera_info(msg) -> T.CameraInfo:
    k = msg.k
    return T.CameraInfo(
        int(msg.width), int(msg.height), k[0], k[4], k[2], k[5], [float(x) for x in msg.d]
    )


def _fill_pointcloud(msg, v: T.PointCloud, stamp, hints) -> None:
    from .registry import fill_pointcloud

    fill_pointcloud(msg, v, stamp, {"frame_id": "livox_frame", **hints})


def _decode_pointcloud(msg) -> T.PointCloud:
    offsets = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"} <= set(offsets):
        raise ValueError(f"a PointCloud2 without x, y, z fields: {sorted(offsets)}")
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    cols = [raw[:, offsets[a] : offsets[a] + 4].copy().view(np.float32).ravel() for a in "xyz"]
    return T.PointCloud(np.stack(cols, axis=1))


def _scalar(cls: type, msg: str) -> RosType:
    def fill(m, v, stamp, hints):
        m.data = cls(v)

    def decode(m):
        return cls(m.data)

    return RosType(cls, (Wire(msg, fill, decode),))


#: The neutral types' ROS mapping (see the module docstring), keyed by class.
TYPES: dict[type, RosType] = {}


def register(rostype: RosType) -> None:
    """Add a type's ROS mapping; a class is mapped once."""
    if rostype.cls in TYPES and TYPES[rostype.cls] != rostype:
        raise ValueError(f"{rostype.cls.__qualname__} already has a ROS mapping")
    TYPES[rostype.cls] = rostype


for _rostype in (
    RosType(
        T.Twist,
        (
            Wire("geometry_msgs.msg.Twist", _fill_twist, _decode_twist),
            Wire(
                "geometry_msgs.msg.TwistStamped",
                _fill_twist_stamped,
                lambda m: _decode_twist(m.twist),
            ),
        ),
        stamped=("geometry_msgs.msg.Twist", "geometry_msgs.msg.TwistStamped"),
    ),
    RosType(
        T.Pose,
        (
            Wire("geometry_msgs.msg.PoseStamped", _fill_pose_stamped, _decode_pose_stamped),
            Wire("geometry_msgs.msg.Pose", _fill_pose, _decode_pose),
        ),
        stamped=("geometry_msgs.msg.Pose", "geometry_msgs.msg.PoseStamped"),
    ),
    RosType(
        T.Odometry,
        (Wire("nav_msgs.msg.Odometry", _fill_odometry, _decode_odometry),),
        hints={"frame_id": "odom", "child_frame_id": "base_link"},
    ),
    RosType(
        T.JointState, (Wire("sensor_msgs.msg.JointState", _fill_joint_state, _decode_joint_state),)
    ),
    RosType(
        T.JointPositions,
        (
            Wire(
                "trajectory_msgs.msg.JointTrajectory",
                _fill_joint_trajectory,
                _decode_joint_trajectory,
            ),
        ),
    ),
    RosType(
        T.Wrench,
        (
            Wire("geometry_msgs.msg.WrenchStamped", _fill_wrench_stamped, _decode_wrench_stamped),
            Wire("geometry_msgs.msg.Wrench", _fill_wrench, _decode_wrench),
        ),
        stamped=("geometry_msgs.msg.Wrench", "geometry_msgs.msg.WrenchStamped"),
    ),
    RosType(T.Imu, (Wire("sensor_msgs.msg.Imu", _fill_imu, _decode_imu),)),
    RosType(T.LaserScan, (Wire("sensor_msgs.msg.LaserScan", _fill_scan, _decode_scan),)),
    RosType(
        T.Image,
        (
            Wire("sensor_msgs.msg.Image", _fill_image, _decode_image),
            Wire(
                "sensor_msgs.msg.CompressedImage", _fill_compressed_image, _decode_compressed_image
            ),
        ),
    ),
    RosType(
        T.CameraInfo, (Wire("sensor_msgs.msg.CameraInfo", _fill_camera_info, _decode_camera_info),)
    ),
    RosType(
        T.PointCloud, (Wire("sensor_msgs.msg.PointCloud2", _fill_pointcloud, _decode_pointcloud),)
    ),
    _scalar(bool, "std_msgs.msg.Bool"),
    _scalar(float, "std_msgs.msg.Float64"),
    _scalar(int, "std_msgs.msg.Int64"),
    _scalar(str, "std_msgs.msg.String"),
):
    register(_rostype)


@functools.cache
def _load_entry_points() -> None:
    """Register every :class:`RosType` advertised in :data:`ENTRY_POINT_GROUP`, once.

    Loudly: a package that declares a mapping and fails to provide it would otherwise leave its
    endpoints silently off ROS.
    """
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        loaded = entry.load()
        items = [loaded] if isinstance(loaded, RosType) else list(loaded)
        for item in items:
            if not isinstance(item, RosType):
                raise TypeError(
                    f"{ENTRY_POINT_GROUP} entry {entry.name!r} ({entry.value}) must load a RosType "
                    f"or an iterable of them, got {type(item).__name__}"
                )
            register(item)


def lookup(cls: type | None) -> RosType | None:
    """The ROS mapping of *cls*, from the table or a package's entry point; ``None`` if it has none."""
    if cls is None:
        return None
    _load_entry_points()
    return TYPES.get(cls)


# -- mapping by field name -------------------------------------------------------------------------
_ROS_FLOATS = {"float", "double"}
_ROS_INTS = {
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "octet",
    "byte",
    "char",
}


def _element(ros_type: str) -> str | None:
    """The element type of a ROS sequence or array field, or ``None`` for a single value."""
    if ros_type.startswith("sequence<"):
        return ros_type[len("sequence<") : ros_type.rindex(">")].split(",")[0].strip()
    if ros_type.endswith("]"):
        return ros_type[: ros_type.index("[")]
    return None


def _scalar_fits(kind: str, ros_type: str) -> bool:
    if kind == "float":
        return ros_type in _ROS_FLOATS or ros_type in _ROS_INTS
    if kind == "int":
        return ros_type in _ROS_INTS
    if kind == "bool":
        return ros_type == "boolean"
    if kind == "str":
        return ros_type in ("string", "wstring") or ros_type.startswith("string<")
    return False


def _mismatches(fields, msg_cls, where: str) -> list[str]:
    """Each way *fields* (``Param``\\ s) do not fit *msg_cls*'s fields, by name."""
    ros_fields = msg_cls.get_fields_and_field_types()
    errors = []
    for f in fields:
        path = f"{where}.{f.name}"
        ros_type = ros_fields.get(f.name)
        vt = f.type
        if ros_type is None:
            errors.append(
                f"{path} has no field of that name in {msg_cls.__name__} ({', '.join(ros_fields)})"
            )
            continue
        element = _element(ros_type)
        if vt.kind in ("array", "list", "tuple"):
            if element is None:
                errors.append(f"{path} is a sequence but {msg_cls.__name__}.{f.name} is {ros_type}")
            elif vt.kind == "array" and not (
                element in _ROS_FLOATS or element in _ROS_INTS or element == "boolean"
            ):
                errors.append(
                    f"{path} is a numeric array but {msg_cls.__name__}.{f.name} is {ros_type}"
                )
            elif (
                vt.items
                and vt.items[0].kind in ("float", "int", "bool", "str")
                and not _scalar_fits(vt.items[0].kind, element)
            ):
                errors.append(
                    f"{path} holds {vt.items[0].name} but {msg_cls.__name__}.{f.name} is {ros_type}"
                )
        elif vt.kind == "struct":
            if "/" not in ros_type or element is not None:
                errors.append(f"{path} is {vt.name} but {msg_cls.__name__}.{f.name} is {ros_type}")
            else:
                errors += _mismatches(vt.fields, type(getattr(msg_cls(), f.name)), path)
        elif vt.kind in ("float", "int", "bool", "str"):
            if not _scalar_fits(vt.kind, ros_type):
                errors.append(f"{path} is {vt.name} but {msg_cls.__name__}.{f.name} is {ros_type}")
        else:
            errors.append(f"{path} is {vt.name}, which has no mapping onto {ros_type}")
    return errors


def _fill_by_name(msg, value, fields, stamp, hints) -> None:
    get = value.get if isinstance(value, Mapping) else lambda n: getattr(value, n)
    for f in fields:
        v = get(f.name)
        kind = f.type.kind
        if kind == "struct":
            _fill_by_name(getattr(msg, f.name), v, f.type.fields, stamp, hints)
        elif kind in ("array", "list", "tuple"):
            setattr(msg, f.name, np.asarray(v).tolist() if kind == "array" else list(v))
        elif kind == "float":
            setattr(msg, f.name, float(v))
        elif kind == "int":
            setattr(msg, f.name, int(v))
        elif kind == "bool":
            setattr(msg, f.name, bool(v))
        else:
            setattr(msg, f.name, v)
    if hasattr(msg, "header") and "header" not in {f.name for f in fields}:
        _header(msg, stamp, frame(hints, "frame_id", "") or None)


def _decode_by_name(msg, fields, cls) -> Any:
    values = {}
    for f in fields:
        v = getattr(msg, f.name)
        kind = f.type.kind
        if kind == "struct":
            values[f.name] = _decode_by_name(v, f.type.fields, f.type.cls)
        elif kind == "array":
            values[f.name] = np.asarray(list(v), dtype=f.type.dtype)
        elif kind in ("list", "tuple"):
            values[f.name] = list(v) if kind == "list" else tuple(v)
        else:
            values[f.name] = v
    return cls(**values) if cls is not None else values


# -- binding an endpoint -----------------------------------------------------------------------------
@dataclass
class Binding:
    """How the bridge carries one endpoint on ROS.

    ``hints`` are the effective hints, or ``None`` when the endpoint is not on ROS (``reason`` says
    why; ``required`` when the producer asked for ROS and it cannot be given, which the bridge
    refuses). ``fill`` publishes an ``out`` payload, ``decode`` turns an inbound message into what the
    endpoint's ``write`` takes. A mapping by field name is checked against the message class in
    :meth:`prepare` first.
    """

    hints: dict | None
    reason: str = ""
    required: bool = False
    fill: Fill | None = None
    decode: Decode | None = None
    by_name: ValueType | None = None
    _to_write: Callable[[Any], Any] | None = None
    _field: str | None = None

    def prepare(self, msg_cls) -> None:
        """Check a mapping by field name against *msg_cls* and build its converters; refuse, naming
        every field that does not fit. A no-op for a type with a row in the table."""
        if self.by_name is None:
            return
        errors = _mismatches(self.by_name.fields, msg_cls, self.by_name.name)
        if errors:
            raise ValueError(
                f"{self.hints['type']} cannot carry {self.by_name.name} by field name: "
                + "; ".join(errors)
            )
        fields, cls, field_name = self.by_name.fields, self.by_name.cls, self._field

        def fill(msg, value, stamp, hints):
            _fill_by_name(
                msg, getattr(value, field_name) if field_name else value, fields, stamp, hints
            )

        self.fill = fill
        to_write = self._to_write
        self.decode = lambda msg: to_write(_decode_by_name(msg, fields, cls))


_MISSING = object()


def _is_typed(ep) -> bool:
    return bool(ep.transport) or ep.params is not None or ep.payload_type is not None


def _field_type(vt: ValueType, name: str) -> ValueType:
    for f in vt.fields:
        if f.name == name:
            return f.type
    raise ValueError(
        f"ros2 hint field={name!r} is not a field of {vt.name} ({', '.join(f.name for f in vt.fields)})"
    )


def _writer(ep, carried: ValueType | None) -> Callable[[Any], Any]:
    """What ``write`` takes, from a decoded value: the one parameter, or the fields it names."""
    params = ep.params or ()
    if carried is not None and len(params) == 1 and params[0].type.cls is carried.cls:
        name = params[0].name
        return lambda value: {name: value}
    names = [p.name for p in params]
    if carried is None or carried.cls is None:
        return lambda value: {n: value[n] for n in names}
    return lambda value: {n: getattr(value, n) for n in names}


def _qos(ep, hints: dict) -> dict:
    return dict(ep.qos) if ep.qos is not None else qos_profile(hints.get("qos", "default"))


def resolve(ep) -> Binding | None:
    """How *ep* is carried on ROS, or ``None`` when it is not meant to be (no hint block for ROS and
    not a decorated endpoint, or a hint block of ``None``)."""
    raw = ep.backend.get("ros2", _MISSING)
    if raw is None:
        return None
    if raw is _MISSING:
        if not ep.transport:
            return None
        raw = {}
    hints = dict(raw)
    if not _is_typed(ep):
        return _legacy(ep, hints)
    if "service" in hints or "action" in hints:
        return _service(ep, hints)
    if ep.direction == "in" and ep.params == () and "type" not in hints:
        hints["service"] = "std_srvs.srv.Trigger"
        return _service(ep, hints)

    carried = ep.payload_type if ep.payload_type is not None else ep.result
    field_name = hints.get("field") if ep.direction == "out" else None
    if field_name is not None and carried is not None:
        carried = _field_type(carried, field_name)
    rostype = lookup(carried.cls if carried is not None else None)
    stamped = hints.pop("stamped", None)
    msg = hints.get("type")
    binding = Binding(hints=None, required=bool(raw))
    if rostype is not None and (msg is None or rostype.wire(msg) is not None):
        if msg is None:
            if stamped is not None:
                if rostype.stamped is None:
                    raise ValueError(
                        f"{ep.owner}/{ep.name}: ros2 hint stamped={stamped!r}, but {carried.name} "
                        f"travels as {', '.join(w.msg for w in rostype.wires)} only"
                    )
                msg = rostype.stamped[bool(stamped)]
            else:
                msg = rostype.wires[0].msg
        wire = rostype.wire(msg)
        hints = {**rostype.hints, **hints, "type": msg}
        if field_name is not None:
            binding.fill = lambda m, value, stamp, h, _f=wire.fill: _f(
                m, getattr(value, field_name), stamp, h
            )
        else:
            binding.fill = wire.fill
        to_write = _writer(ep, carried)
        binding.decode = lambda m, _d=wire.decode: to_write(_d(m))
    elif msg is not None:
        if stamped is not None:
            raise ValueError(
                f"{ep.owner}/{ep.name}: ros2 hint stamped= needs a type with a mapping; name the message in type= instead"
            )
        if carried is None:
            carried = ValueType("struct", "parameters of " + ep.name, fields=tuple(ep.params or ()))
        elif carried.kind != "struct":
            raise ValueError(
                f"{ep.owner}/{ep.name}: {carried.name} has no ROS mapping of its own, and only a "
                f"dataclass maps onto {msg} by field name"
            )
        binding.by_name = carried
        binding._field = field_name
        binding._to_write = _writer(ep, carried if carried.cls is not None else None)
    else:
        what = carried.name if carried is not None else "its parameters"
        binding.reason = (
            f"{what} has no ROS mapping: name a message in its ros2 hint (type=...) to map it by "
            f"field name, or register a converter in the {ENTRY_POINT_GROUP} entry-point group"
        )
        return binding
    hints["topic"] = ep.topic or hints.get("topic") or ep.name
    hints["qos"] = _qos(ep, hints)
    binding.hints = hints
    return binding


def _service(ep, hints: dict) -> Binding:
    hints["name"] = ep.topic or hints.get("name") or ep.name
    if ep.qos is not None or "qos" in hints:
        hints["qos"] = _qos(ep, hints)
    return Binding(hints=hints)


def _legacy(ep, hints: dict) -> Binding:
    """A hand-built endpoint's hint block, with the bridge's defaults and the registry's converters."""
    if "service" in hints or "action" in hints:
        hints.setdefault("name", ep.name)
        if ep.qos is not None or "qos" in hints:
            hints["qos"] = _qos(ep, hints)
        return Binding(hints=hints)
    hints.setdefault("topic", ep.name)
    hints["qos"] = _qos(ep, hints)
    msg = hints.get("type")

    def fill(m, payload, stamp, h):
        from .registry import get_converter

        get_converter(msg)(m, payload, stamp, h)

    def decode(m):
        from .params import payload_for
        from .registry import get_decoder

        return payload_for(ep, get_decoder(msg)(m))

    return Binding(hints=hints, fill=fill, decode=decode)


# -- describing ------------------------------------------------------------------------------------
_SHOWN = ("type", "service", "action", "topic", "name", "qos", "field")


def describe(ep) -> dict:
    """How ROS carries *ep*, for ``roqsim plugins describe`` (the ``roqsim.transports`` entry)."""
    try:
        binding = resolve(ep)
    except ValueError as exc:
        return {"on_ros": False, "error": str(exc)}
    if binding is None:
        return {"on_ros": False, "reason": "its ros2 hint is None"}
    if binding.hints is None:
        return {"on_ros": False, "reason": binding.reason}
    row = {"on_ros": True, **{k: binding.hints[k] for k in _SHOWN if k in binding.hints}}
    if binding.by_name is not None:
        row["mapping"] = "by field name"
    return row


def to_joint_tuple(payload) -> tuple:
    """A joint-state payload as ``(names, positions, velocities[, efforts])``, whichever form it has."""
    if isinstance(payload, T.JointState):
        base = (payload.names, payload.positions, payload.velocities)
        return base if payload.efforts is None else (*base, payload.efforts)
    return tuple(payload)


def odometry_pose(payload) -> tuple[np.ndarray, np.ndarray] | None:
    """``(position, (w, x, y, z))`` of an :class:`~roqsim.types.Odometry`, else ``None``."""
    if isinstance(payload, T.Odometry):
        return np.asarray(payload.position), np.asarray(payload.orientation)
    return None
