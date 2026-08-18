"""ROS 2 side of the bridge: resolve message types by string and fill them from neutral payloads.

The robot package declares each endpoint's ROS type as a *string* (e.g.
``"sensor_msgs.msg.LaserScan"``); :func:`resolve_type` turns that into the class via ``importlib``,
so the bridge has no hardcoded ``from sensor_msgs.msg import ...``. Field mapping lives here, keyed by
the type string:

  * ``CONVERTERS`` -- fill an *outbound* message in place from a neutral payload (``out`` endpoints).
  * ``DECODERS``   -- decode an *inbound* message to a neutral payload (``in`` endpoints).

Converters/decoders are duck-typed on the payload (``payload.ranges``, ``twist.linear.x``), never on a
specific robot package's dataclass, so the backend stays decoupled from any robot. Types without a
registered converter fall back to the reflective ``msg.data = payload`` path (std_msgs primitives);
a producer with a structured payload publishes one of its fields by naming it in the endpoint's
``field`` hint (see :func:`get_converter`), so a primitive type never needs a converter that knows
one producer's attribute names.
"""

from __future__ import annotations

import functools
import importlib
import math
from collections.abc import Callable
from typing import Any

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, Quaternion, TransformStamped

# type-string -> fill(msg, payload, stamp, hints) -> None   (outbound)
CONVERTERS: dict[str, Callable[[Any, Any, Time, dict], None]] = {}
# type-string -> decode(msg) -> neutral payload             (inbound)
DECODERS: dict[str, Callable[[Any], Any]] = {}


@functools.cache
def resolve_type(path: str):
    """Resolve ``"pkg.msg.Type"`` to the message class. Cached; import cost is paid once."""
    module_name, _, type_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"message type {path!r} must be fully qualified, e.g. 'pkg.msg.Type'")
    return getattr(importlib.import_module(module_name), type_name)


def converter(type_path: str):
    def register(fn):
        CONVERTERS[type_path] = fn
        return fn

    return register


def decoder(type_path: str):
    def register(fn):
        DECODERS[type_path] = fn
        return fn

    return register


def get_converter(type_path: str) -> Callable[[Any, Any, Time, dict], None]:
    """Registered converter for ``type_path``, or a reflective ``msg.data = payload`` fallback.

    A producer whose payload is a *structure* but whose ROS type is a single-field primitive names
    the field it publishes, with ``"field"`` in its ros2 backend hint::

        backend={"ros2": {"type": "std_msgs.msg.Bool", "field": "in_contact", "topic": "collision"}}

    That keeps the meaning of the field with the plugin that owns it, so no primitive type needs a
    converter here that would have to know one producer's attribute names (``contact_monitor``
    publishes a rich ``ContactReport`` and a plain ``Bool``; the same door serves a ``Float64`` or a
    ``String`` drawn from any other payload).
    """
    fn = CONVERTERS.get(type_path)
    if fn is not None:
        return fn

    def fill_reflective(msg, payload, stamp, hints):
        field = hints.get("field")
        if field is not None:
            try:
                payload = getattr(payload, field)
            except AttributeError:
                # Loudly: a mistyped field would otherwise publish the whole payload, which fails
                # one layer down as an opaque message-assignment error naming neither.
                raise TypeError(
                    f"backend hint field={field!r} for {type_path!r} is not an attribute of "
                    f"{type(payload).__name__}"
                ) from None
        if not hasattr(msg, "data"):
            raise TypeError(
                f"no converter registered for {type_path!r} and it has no 'data' field to fall back on"
            )
        msg.data = payload

    return fill_reflective


def get_decoder(type_path: str) -> Callable[[Any], Any]:
    """Registered decoder for ``type_path``, or a reflective ``msg.data`` fallback."""
    fn = DECODERS.get(type_path)
    if fn is not None:
        return fn
    return lambda msg: msg.data


# -- shared helpers ----------------------------------------------------------------------------
def to_time_msg(t: float) -> Time:
    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def yaw_to_quat(yaw: float) -> Quaternion:
    return Quaternion(z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


def namespaced(prefix: str, name: str) -> str:
    """Prefix a frame id with the bridge namespace so multi-robot TF trees stay unique."""
    return f"{prefix}/{name}" if prefix else name


def frame(hints: dict, key: str, default: str) -> str:
    """Frame id from a hint, prefixed by the bridge namespace so multi-robot TF trees stay unique."""
    return namespaced(hints.get("frame_prefix", ""), hints.get(key, default))


# -- converters (outbound: neutral payload -> ROS message, filled in place) --------------------
@converter("nav_msgs.msg.Odometry")
def fill_odom(msg, payload, stamp: Time, hints: dict) -> None:
    # Optional 7th element is the base height; legged robots report it so base_link sits at its true
    # z (a planar producer omits it -> z=0, unchanged). nav2 is 2D and ignores it.
    x, y, yaw, v, vy, w, *rest = payload
    z = float(rest[0]) if rest else 0.0
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "odom")
    msg.child_frame_id = frame(hints, "child_frame_id", "base_link")
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = z
    msg.pose.pose.orientation = yaw_to_quat(float(yaw))
    msg.twist.twist.linear.x = float(v)
    msg.twist.twist.linear.y = float(vy)
    msg.twist.twist.angular.z = float(w)


@converter("sensor_msgs.msg.LaserScan")
def fill_scan(msg, payload, stamp: Time, hints: dict) -> None:
    msg.header.stamp = stamp
    # Generic fallback only: every lidar producer sets frame_id (defaulting to its site). A
    # robot-specific name here would stamp one robot's scan in another's frame.
    msg.header.frame_id = frame(hints, "frame_id", "lidar")
    msg.angle_min = float(payload.angle_min)
    msg.angle_max = float(payload.angle_max)
    msg.angle_increment = float(payload.angle_increment)
    msg.range_min = float(payload.range_min)
    msg.range_max = float(payload.range_max)
    # float32[] slot accepts a matching-dtype numpy array directly -- one C-level copy, no per-ray
    # Python float boxing. inf == "no return", which LaserScan permits.
    msg.ranges = _as_f32(payload.ranges)


@converter("sensor_msgs.msg.PointCloud2")
def fill_pointcloud(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: a livox_mid360 PointCloud with .points, an (N, 3) float32 array of XYZ in the sensor
    # frame. Emitted as an unorganized cloud (height=1, width=N) of three FLOAT32 fields; the raw
    # (N, 3) buffer is already the wire layout (x,y,z contiguous per point), so one tobytes() copies it.
    import numpy as np
    from sensor_msgs.msg import PointField

    pts = np.ascontiguousarray(payload.points, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "livox_frame")
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * n
    msg.data = pts.tobytes()
    msg.is_dense = True


@converter("tf2_msgs.msg.TFMessage")
def fill_tf_message(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: [(child_frame_id, pos[3], quat_wxyz[4]), ...] -- one TransformStamped per entry, all
    # sharing the parent frame. Streams many world-frame bodies (e.g. a walker's 17 skeleton bones)
    # in one /tf message. Unlike make_tf (a single odom->base transform), this is the multi-transform
    # path; the child frames are the producer's body names, used verbatim so a viewer can bind by name.
    parent = frame(hints, "frame_id", "map")
    transforms = []
    for child_frame, pos, quat in payload:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child_frame
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w = float(quat[0])
        tf.transform.rotation.x = float(quat[1])
        tf.transform.rotation.y = float(quat[2])
        tf.transform.rotation.z = float(quat[3])
        transforms.append(tf)
    msg.transforms = transforms


@converter("vision_msgs.msg.Detection3DArray")
def fill_detection3d_array(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: [(class_id, pose7, size3, score), ...] where pose7 is (x, y, z, qw, qx, qy, qz) in the
    # producer's reporting frame. The standard message a 3D object pose estimator publishes, so a real
    # detector replaces the simulated producer without anything downstream changing.
    #
    # `bbox.center` and `results[0].pose.pose` carry the same pose deliberately: consumers in the wild
    # read one or the other and both are correct here. `id` repeats class_id because this is a
    # single-instance-per-class detector -- a real one with several parcels in view would need
    # instance ids, and that is the field they belong in.
    from vision_msgs.msg import (
        BoundingBox3D,
        Detection3D,
        ObjectHypothesis,
        ObjectHypothesisWithPose,
    )

    frame_id = frame(hints, "frame_id", "base_link")
    detections = []
    for class_id, pose, size, score in payload:
        det = Detection3D()
        det.header.stamp = stamp
        det.header.frame_id = frame_id
        det.id = str(class_id)

        pose_msg = Pose()
        pose_msg.position.x = float(pose[0])
        pose_msg.position.y = float(pose[1])
        pose_msg.position.z = float(pose[2])
        pose_msg.orientation.w = float(pose[3])
        pose_msg.orientation.x = float(pose[4])
        pose_msg.orientation.y = float(pose[5])
        pose_msg.orientation.z = float(pose[6])

        hypothesis = ObjectHypothesis()
        hypothesis.class_id = str(class_id)
        hypothesis.score = float(score)
        result = ObjectHypothesisWithPose()
        result.hypothesis = hypothesis
        result.pose.pose = pose_msg
        det.results = [result]

        bbox = BoundingBox3D()
        bbox.center = pose_msg
        bbox.size.x = float(size[0])
        bbox.size.y = float(size[1])
        bbox.size.z = float(size[2])
        det.bbox = bbox

        detections.append(det)
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.detections = detections


@converter("sensor_msgs.msg.JointState")
def fill_joint_state(msg, payload, stamp: Time, hints: dict) -> None:
    # (names, positions, velocities[, efforts]). Effort is optional so the wheel/locomotion producers
    # keep their 3-tuple, but a real driver does publish it -- ros2_control fills the effort interface,
    # and the G1's own /lowstate carries per-motor tau_est -- so an arm that can report it should.
    names, positions, velocities, *rest = payload
    msg.header.stamp = stamp
    msg.name = list(names)
    msg.position = _as_f64(positions)
    msg.velocity = _as_f64(velocities)
    if rest:
        msg.effort = _as_f64(rest[0])


@converter("control_msgs.msg.JointTrajectoryControllerState")
def fill_controller_state(msg, payload, stamp: Time, hints: dict) -> None:
    # (names, desired, actual, velocities). The third interface a ros2_control
    # JointTrajectoryController exposes, beside its action and command topic.
    #
    # Field names are the CURRENT ones: control_msgs renamed desired/actual to reference/feedback (the
    # old pair is deprecated and absent on Jazzy), so writing to `msg.desired` raises AttributeError at
    # publish time -- which only shows up with ROS actually running.
    names, desired, actual, velocities = payload
    msg.header.stamp = stamp
    msg.joint_names = list(names)
    msg.reference.positions = _as_f64(desired)
    msg.feedback.positions = _as_f64(actual)
    msg.feedback.velocities = _as_f64(velocities)
    msg.error.positions = _as_f64([a - c for a, c in zip(actual, desired, strict=True)])


# Bytes per pixel for the encodings roqsim_sensors' camera plugins use. One converter serves
# colour and depth alike (and any future mono/IR stream) via the `encoding` hint -- no per-stream
# message-filling code needed in a robot package.
_IMAGE_ENCODING_BYTES = {"rgb8": 3, "mono8": 1, "16UC1": 2, "32FC1": 4}


@converter("sensor_msgs.msg.Image")
def fill_image(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: a raw (H, W[, C]) numpy array (uint8 rgb8, or float32 32FC1 depth, ...).
    encoding = hints.get("encoding", "rgb8")
    height, width = payload.shape[:2]
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "camera_optical_frame")
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = width * _IMAGE_ENCODING_BYTES[encoding]
    msg.data = payload.tobytes()


@converter("sensor_msgs.msg.CameraInfo")
def fill_camera_info(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: a roqsim_sensors camera_common.Intrinsics (width, height, fx, fy, cx, cy).
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "camera_optical_frame")
    msg.height = payload.height
    msg.width = payload.width
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [payload.fx, 0.0, payload.cx, 0.0, payload.fy, payload.cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [
        payload.fx,
        0.0,
        payload.cx,
        0.0,
        0.0,
        payload.fy,
        payload.cy,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


# -- decoders (inbound: ROS message -> neutral payload) ----------------------------------------
@decoder("geometry_msgs.msg.Twist")
def decode_twist(msg) -> tuple[float, float, float]:
    return (msg.linear.x, msg.linear.y, msg.angular.z)


@decoder("geometry_msgs.msg.TwistStamped")
def decode_twist_stamped(msg) -> tuple[float, float, float]:
    return decode_twist(msg.twist)


@decoder("trajectory_msgs.msg.JointTrajectory")
def decode_joint_trajectory(msg) -> tuple[list[str], list[float]]:
    """Streamed joint command -> neutral ``(names, positions)`` for an arm's ``set_targets`` (same
    payload the FollowJointTrajectory action feeds). moveit_servo publishes a single-point trajectory
    each period; take the last point's positions (== the target). Empty points -> a no-op write."""
    if not msg.points:
        return (list(msg.joint_names), [])
    return (list(msg.joint_names), list(msg.points[-1].positions))


# -- transform (owned by the bridge, derived from an odom payload) -----------------------------
def make_tf(payload, stamp: Time, frame_id: str, child_frame_id: str) -> TransformStamped:
    # payload is the odom tuple (x, y, yaw, v, vy, w[, z]); the optional trailing z is the base
    # height, so a legged robot's base_link renders at its true elevation (planar -> z=0).
    x, y, yaw, *rest = payload
    z = float(rest[3]) if len(rest) >= 4 else 0.0
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
    tf.transform.translation.x = float(x)
    tf.transform.translation.y = float(y)
    tf.transform.translation.z = z
    tf.transform.rotation = yaw_to_quat(float(yaw))
    return tf


def make_static_tf(stamp: Time, parent_frame: str, child_frame: str, translation, rotation):
    """A fixed sensor-mount transform (parent -> child), from numbers a producer put on its endpoint.

    ``rotation`` is (w, x, y, z), matching the ``mujoco`` site quaternion the lidar plugin emits.
    Published once on the latched ``/tf_static`` by the bridge, so the stamp is not looked up.
    """
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent_frame
    tf.child_frame_id = child_frame
    tf.transform.translation.x = float(translation[0])
    tf.transform.translation.y = float(translation[1])
    tf.transform.translation.z = float(translation[2])
    tf.transform.rotation.w = float(rotation[0])
    tf.transform.rotation.x = float(rotation[1])
    tf.transform.rotation.y = float(rotation[2])
    tf.transform.rotation.z = float(rotation[3])
    return tf


# rclpy's float sequence fields (e.g. LaserScan.ranges, JointState.position) want an ``array.array``
# of the matching typecode or a list of Python floats. Humble's rclpy rejects a raw numpy array with
# an assertion ("each value of type 'float'"); Jazzy happens to accept it. Returning ``array.array``
# built from the numpy buffer is the fast path that works on both distros.
def _as_f32(arr):
    import array

    import numpy as np

    return array.array("f", np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def _as_f64(arr):
    import array

    import numpy as np

    return array.array("d", np.ascontiguousarray(arr, dtype=np.float64).tobytes())
