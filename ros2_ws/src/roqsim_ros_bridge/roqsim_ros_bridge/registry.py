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

from . import image_codec

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

    def decode_reflective(msg):
        # Loudly, and mirroring fill_reflective on the encoder side: an unregistered type with no
        # 'data' field would otherwise raise AttributeError inside the subscription callback, which
        # rclpy surfaces by killing the executor's spin thread -- every endpoint on the bridge then
        # goes silent at once, far from the type that caused it.
        if not hasattr(msg, "data"):
            raise TypeError(
                f"no decoder registered for {type_path!r} and it has no 'data' field to fall "
                f"back on"
            )
        return msg.data

    return decode_reflective


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
#: Keys of the 6-DOF odometry payload. A producer whose body genuinely pitches and rolls passes a
#: mapping with these instead of the planar tuple below -- see ``fill_odom``.
ODOM6_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz")


def is_odom6(payload) -> bool:
    """Is this the 6-DOF odometry payload rather than the planar tuple?"""
    return hasattr(payload, "keys") and "qw" in payload


@converter("nav_msgs.msg.Odometry")
def fill_odom(msg, payload, stamp: Time, hints: dict) -> None:
    """Odometry from either payload shape.

    TWO SHAPES, because two kinds of robot report honestly in different terms:

    * the planar tuple ``(x, y, yaw, v, vy, w[, z])`` -- a ground robot, with the optional trailing
      z the base height so a legged robot's base_link sits at its true elevation (planar -> z=0).
    * a mapping of ``ODOM6_KEYS`` -- a body that pitches and rolls, and for which the planar tuple
      is not merely lossy but WRONG: it would report tilt as zero and drop vertical speed entirely,
      so a drone flying on its side reads as level. Yaw-only orientation is a projection that a
      ground robot may make and an aircraft may not.
    """
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "odom")
    msg.child_frame_id = frame(hints, "child_frame_id", "base_link")

    if is_odom6(payload):
        msg.pose.pose.position.x = float(payload["x"])
        msg.pose.pose.position.y = float(payload["y"])
        msg.pose.pose.position.z = float(payload["z"])
        msg.pose.pose.orientation = Quaternion(
            x=float(payload["qx"]),
            y=float(payload["qy"]),
            z=float(payload["qz"]),
            w=float(payload["qw"]),
        )
        msg.twist.twist.linear.x = float(payload["vx"])
        msg.twist.twist.linear.y = float(payload["vy"])
        msg.twist.twist.linear.z = float(payload["vz"])
        msg.twist.twist.angular.x = float(payload.get("wx", 0.0))
        msg.twist.twist.angular.y = float(payload.get("wy", 0.0))
        msg.twist.twist.angular.z = float(payload.get("wz", 0.0))
        return

    x, y, yaw, v, vy, w, *rest = payload
    z = float(rest[0]) if rest else 0.0
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


@converter("sensor_msgs.msg.Imu")
def fill_imu(msg, payload, stamp: Time, hints: dict) -> None:
    """A strap-down IMU reading (``roqsim_sensors.plugins.imu.ImuReading``).

    Two details are REP 145 conventions rather than choices made here. The acceleration is proper
    acceleration -- gravity included -- which is what the producer reads out of MuJoCo and what every
    subscriber of this message assumes. And ``orientation_covariance[0] = -1`` is the standard marker
    for "this device does not report attitude"; a rate-only IMU must send that rather than an identity
    quaternion, which a consumer cannot tell from a level robot.

    The covariances are isotropic diagonals built from the producer's declared per-axis variance, so a
    filter weights the channel by the noise the world actually configured. A perfect sensor reports
    zeros, which is truthful: it is the producer's job to state a floor if its consumer needs one.
    """
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "imu_link")
    if getattr(payload, "orientation_valid", True):
        w, x, y, z = payload.orientation
        msg.orientation = Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
        msg.orientation_covariance = _diag3(payload.orientation_variance)
    else:
        # Leave the quaternion at its default and mark the channel absent.
        cov = _diag3(0.0)
        cov[0] = -1.0
        msg.orientation_covariance = cov
    wx, wy, wz = payload.angular_velocity
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (
        float(wx),
        float(wy),
        float(wz),
    )
    ax, ay, az = payload.linear_acceleration
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (
        float(ax),
        float(ay),
        float(az),
    )
    msg.angular_velocity_covariance = _diag3(payload.angular_velocity_variance)
    msg.linear_acceleration_covariance = _diag3(payload.linear_acceleration_variance)


def _diag3(variance: float) -> list:
    """A row-major 3x3 covariance with *variance* on the diagonal, as the nine floats ROS wants."""
    v = float(variance)
    return [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]


@converter("vision_msgs.msg.Detection2DArray")
def fill_detection2d_array(msg, payload, stamp: Time, hints: dict) -> None:
    """2D image-space detections from a mask.

    payload: ``[(class_id, class_name, instance_id, cx, cy, w, h), ...]`` in pixels, with the box
    inclusive of both edge rows (so a 1x1 instance has size 1, not 0). The standard message a 2D
    detector publishes, so a real one replaces the simulated producer without anything downstream
    changing.

    ``Detection2D.id`` carries the INSTANCE and the hypothesis carries the CLASS, which is the split
    the message intends and the reason both fields exist: two parcels in view are one class and two
    ids, and collapsing them (as a class-only producer must) makes a tracker associate them as one
    object. Score is 1.0 because these are ground-truth boxes -- a mask either covers a pixel or does
    not, and inventing a confidence would let a consumer threshold on a number that means nothing.
    """
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        ObjectHypothesis,
        ObjectHypothesisWithPose,
    )

    frame_id = frame(hints, "frame_id", "camera_optical_frame")
    detections = []
    # The numeric class is deliberately not on the wire: vision_msgs' class_id is a STRING, so the
    # declared name is what a consumer should read, and there is no field for the number that would
    # not be an abuse of one. It stays in the payload for an in-process reader, and a consumer that
    # needs to relate a box to the label image's pixel values reads the world's own `classes:` block
    # -- one mapping, in the place that defines it.
    for _class_id, class_name, instance_id, cx, cy, w, h in payload:
        det = Detection2D()
        det.header.stamp = stamp
        det.header.frame_id = frame_id
        det.id = str(instance_id)

        hypothesis = ObjectHypothesis()
        # The declared NAME, not the numeric id: vision_msgs' class_id is a string, and a consumer
        # reading "parcel" needs no copy of the world's class table to know what it saw. The number
        # is what the label image carries, and it is in the message too, as the instance's own row.
        hypothesis.class_id = str(class_name)
        hypothesis.score = 1.0
        result = ObjectHypothesisWithPose()
        result.hypothesis = hypothesis
        det.results = [result]

        bbox = BoundingBox2D()
        bbox.center.position.x = float(cx)
        bbox.center.position.y = float(cy)
        bbox.size_x = float(w)
        bbox.size_y = float(h)
        det.bbox = bbox

        detections.append(det)
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.detections = detections


@converter("sensor_msgs.msg.BatteryState")
def fill_battery_state(msg, payload, stamp: Time, hints: dict) -> None:
    """A robot's energy state (``roqsim.plugins.energy_monitor.EnergyReport``).

    The message a real platform publishes, so a stack that already watches a battery needs no change.
    Two of its conventions are load-bearing and easy to get wrong: an unknown value is ``NaN``, not
    zero (a zero voltage reads as a dead pack), and ``percentage`` is a FRACTION in [0, 1] rather
    than a percent. A producer with no configured capacity reports the charge fields as unknown
    instead of inventing a full battery.

    ``current`` is negative while discharging -- the sign REP 145's battery message uses for current
    leaving the pack -- so a consumer plotting it sees a drain rather than a charge.
    """
    from sensor_msgs.msg import BatteryState

    unknown = float("nan")
    known_charge = payload.charge_fraction >= 0.0
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "base_link")
    msg.voltage = float(payload.voltage) if payload.voltage > 0.0 else unknown
    msg.current = -float(payload.current_a) if payload.voltage > 0.0 else unknown
    msg.charge = (
        float(payload.capacity_wh * payload.charge_fraction / payload.voltage)
        if known_charge and payload.voltage > 0.0
        else unknown
    )
    msg.capacity = (
        float(payload.capacity_wh / payload.voltage)
        if payload.capacity_wh > 0.0 and payload.voltage > 0.0
        else unknown
    )
    msg.design_capacity = msg.capacity
    msg.percentage = float(payload.charge_fraction) if known_charge else unknown
    msg.power_supply_status = (
        BatteryState.POWER_SUPPLY_STATUS_NOT_CHARGING
        if payload.depleted
        else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
    )
    msg.power_supply_health = (
        BatteryState.POWER_SUPPLY_HEALTH_DEAD
        if payload.depleted
        else BatteryState.POWER_SUPPLY_HEALTH_GOOD
    )
    msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_UNKNOWN
    msg.present = True
    # The integral itself -- the number a paper actually quotes -- has no field here and is not
    # smuggled into one that means something else. A consumer that needs joules reads the endpoint's
    # payload in process, or the plugin's blackboard reader.


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
    # reference - feedback, the sign the message itself states ("essentially reference - feedback,
    # for a regular PID implementation") and the one ros2_control's own JointTrajectoryController
    # publishes. A consumer reads this as "how much further to go", so the inverse does not merely
    # look odd: an arm lagging behind its setpoint reports as leading it.
    msg.error.positions = _as_f64([d - a for d, a in zip(desired, actual, strict=True)])


# Bytes per pixel for the encodings roqsim_sensors' camera plugins use. One converter serves
# colour and depth alike (and any future mono/IR stream) via the `encoding` hint -- no per-stream
# message-filling code needed in a robot package.
_IMAGE_ENCODING_BYTES = {"rgb8": 3, "mono8": 1, "16UC1": 2, "32FC1": 4}


@converter("sensor_msgs.msg.Image")
def fill_image(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: a raw (H, W[, C]) numpy array (uint8 rgb8, or float32 32FC1 depth, ...).
    encoding = hints.get("encoding", "rgb8")
    try:
        stride = _IMAGE_ENCODING_BYTES[encoding]
    except KeyError:
        raise ValueError(
            f"unsupported image encoding {encoding!r}; expected one of "
            f"{sorted(_IMAGE_ENCODING_BYTES)}"
        ) from None
    # The encoding and the array's dtype are chosen independently -- by the producer's hint and by
    # its render code -- and a mismatch is not a visible failure: `step` would be right for the
    # encoding while `data` holds a different number of bytes, so the far end decodes a garbled or
    # truncated image and nothing says why. Depth is where this bites (float32 metres under a 16UC1
    # hint, or the reverse), so it fails on the first publish instead.
    per_pixel = payload.itemsize * (payload.shape[2] if payload.ndim > 2 else 1)
    if per_pixel != stride:
        raise ValueError(
            f"encoding {encoding!r} needs {stride} bytes per pixel, but the payload has "
            f"{per_pixel} (dtype {payload.dtype}, shape {payload.shape})"
        )
    height, width = payload.shape[:2]
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "camera_optical_frame")
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = width * stride
    msg.data = payload.tobytes()


# Compressed colour follows image_transport's convention for ``CompressedImage.format``:
# "<source encoding>; <codec> compressed <encoding of the decoded result>". The mapping is THEIRS, not
# ours -- consumers (cv_bridge, `image_transport republish`, rqt_image_view) read this string, so
# inventing our own spelling would break every one of them.
_COMPRESSED_WIRE = {"rgb8": "bgr8", "bgr8": "bgr8", "mono8": "mono8"}


@converter("sensor_msgs.msg.CompressedImage")
def fill_compressed_image(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: the SAME array `fill_image` publishes for this stream. One neutral payload, two wire
    # formats -- a producer offers both by declaring a second endpoint and nothing else, and needs no
    # codec of its own.
    #
    # Colour and depth share the message type and nothing else -- different codecs, framing and
    # format strings -- so this dispatches rather than growing one function with a branch. The
    # ENCODING decides, not the codec: `png` names a colour codec and a depth codec, and they are
    # different pipelines producing different bytes.
    encoding = hints.get("encoding", "rgb8")
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "camera_optical_frame")
    if encoding in _COMPRESSED_WIRE:
        _fill_compressed_colour(msg, payload, encoding, hints.get("format", "jpeg"), hints)
    else:
        _fill_compressed_depth(msg, payload, encoding, hints)


def _fill_compressed_colour(msg, payload, encoding: str, fmt: str, hints: dict) -> None:
    wire = _COMPRESSED_WIRE[encoding]
    msg.format = f"{encoding}; {fmt} compressed {wire}"
    msg.data = image_codec.encode(
        payload, fmt=fmt, quality=hints.get("quality", image_codec.DEFAULT_JPEG_QUALITY)
    )


def _fill_compressed_depth(msg, payload, encoding: str, hints: dict) -> None:
    if encoding != "16UC1":
        # Everything that is not a colour encoding lands here, so the message names both routes.
        # 32FC1 is the interesting case: the reference carries it through an inverse-depth
        # QUANTISATION into 16 bits before PNG -- lossy, a different pipeline, and not implemented --
        # so the answer is to publish depth as 16UC1 millimetres, which is what hardware does anyway.
        raise TypeError(
            f"cannot compress encoding {encoding!r}: colour compresses as one of "
            f"{sorted(_COMPRESSED_WIRE)}, and compressedDepth's codecs are 16-bit only. Publish the "
            "depth as 16UC1 (the device's own format), or leave it raw."
        )
    fmt = hints.get("format", "png")
    # image_transport's own spelling for this transport: "<source encoding>; compressedDepth <codec>".
    msg.format = f"{encoding}; compressedDepth {fmt}"
    msg.data = image_codec.encode_depth(
        payload, fmt=fmt, png_level=hints.get("png_level", image_codec.DEFAULT_PNG_LEVEL)
    )


@converter("sensor_msgs.msg.CameraInfo")
def fill_camera_info(msg, payload, stamp: Time, hints: dict) -> None:
    # payload: a roqsim_sensors camera_common.Intrinsics (width, height, fx, fy, cx, cy, d).
    msg.header.stamp = stamp
    msg.header.frame_id = frame(hints, "frame_id", "camera_optical_frame")
    msg.height = payload.height
    msg.width = payload.width
    msg.distortion_model = "plumb_bob"
    # Whatever the camera plugin says its pixels carry -- zeros for a plain MuJoCo render (an ideal
    # pinhole), the configured coefficients when that plugin warped the frame to match a real lens.
    # getattr, so an Intrinsics from an older roqsim_sensors still converts.
    msg.d = [float(v) for v in getattr(payload, "d", None) or [0.0] * 5]
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


@decoder("geometry_msgs.msg.PoseStamped")
def decode_pose_stamped(msg) -> tuple[float, float, float, float]:
    """Position setpoint -> neutral ``(x, y, z, yaw)``. Yaw is projected out of the quaternion: a
    setpoint names where to be and which way to face, and no consumer of this payload commands
    pitch or roll -- an airframe holds those to fly, it is not told them."""
    q = msg.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, yaw)


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
    # payload is either odom shape -- see fill_odom. Planar: (x, y, yaw, v, vy, w[, z]), the
    # optional trailing z being the base height so a legged robot's base_link renders at its true
    # elevation (planar -> z=0). 6-DOF: the ODOM6_KEYS mapping, whose full rotation must be carried
    # through rather than flattened to yaw, or TF would stand a banking drone upright.
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
    if is_odom6(payload):
        tf.transform.translation.x = float(payload["x"])
        tf.transform.translation.y = float(payload["y"])
        tf.transform.translation.z = float(payload["z"])
        tf.transform.rotation = Quaternion(
            x=float(payload["qx"]),
            y=float(payload["qy"]),
            z=float(payload["qz"]),
            w=float(payload["qw"]),
        )
        return tf
    x, y, yaw, *rest = payload
    z = float(rest[3]) if len(rest) >= 4 else 0.0
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
