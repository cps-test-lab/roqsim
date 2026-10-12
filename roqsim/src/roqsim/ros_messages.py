"""The ROS 2 sensor messages a camera or a scanner travels as, without ROS: their fields and their bytes.

Two consumers need the same answer to "what does this payload look like as a ROS message": the ROS 2
bridge, which fills ``rclpy`` messages and publishes them, and :mod:`roqsim.rebuild`, which writes the
same messages to an mcap file from a recording, where no ROS need be installed. So the answer lives
here once, as plain data:

* ``*_fields`` -- a payload's message fields (everything but the header) as a dict, with the
  encoding checks, the codecs (:mod:`roqsim.image_codec`) and the ``format`` strings
  ``image_transport`` defines. The bridge assigns them onto its message; :func:`encode` serialises
  them.
* :func:`encode` -- one message as the bytes a ROS 2 publisher puts on the wire: CDR, little-endian,
  with its 4-byte encapsulation header. A test with a sourced ROS checks each type against
  ``rclpy.serialization``.
* :data:`SCHEMAS` -- each type's ``ros2msg`` definition, dependencies included, as an mcap ``ros2``
  profile file carries it.

The types are the ones a rendered or ray-cast stream travels as: ``sensor_msgs/Image``,
``CompressedImage``, ``CameraInfo``, ``PointCloud2`` and ``LaserScan``. :func:`message_type` says which
one an endpoint is, the same way the bridge decides it: the ``type`` hint, else its payload type's
default.

Frame ids follow the bridge's rule (:func:`namespaced`): a frame is prefixed by its endpoint's
namespace, except the frames every robot shares.
"""

from __future__ import annotations

import struct

import numpy as np

from . import image_codec
from . import types as T

#: Frames every robot shares: the simulator's world and the map robots are localised in. A robot's
#: own frames, ``odom`` among them, are per robot and get its namespace.
GLOBAL_FRAMES = frozenset({"world", "map"})


def namespaced(prefix: str, name: str) -> str:
    """Prefix a robot's frame id with its namespace so multi-robot TF trees stay unique.

    A global frame is returned bare: ``<ns>/world`` is a frame nothing publishes.
    """
    return f"{prefix}/{name}" if prefix and name not in GLOBAL_FRAMES else name


def frame(hints: dict, key: str, default: str) -> str:
    """A frame id from the hints, prefixed by ``frame_prefix`` unless global."""
    return namespaced(hints.get("frame_prefix", ""), hints.get(key, default))


def stamp(t: float) -> tuple[int, int]:
    """Seconds as a ``builtin_interfaces/Time``'s ``(sec, nanosec)``."""
    sec = int(t)
    return sec, int(round((t - sec) * 1e9))


IMAGE = "sensor_msgs/msg/Image"
COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
POINT_CLOUD2 = "sensor_msgs/msg/PointCloud2"
LASER_SCAN = "sensor_msgs/msg/LaserScan"

#: The frame a message is stamped in when its endpoint names none.
DEFAULT_FRAME = {
    IMAGE: "camera_optical_frame",
    COMPRESSED_IMAGE: "camera_optical_frame",
    CAMERA_INFO: "camera_optical_frame",
    POINT_CLOUD2: "livox_frame",
    LASER_SCAN: "lidar",
}

#: The message a payload type travels as when its endpoint's hints name no ``type``.
DEFAULT_TYPE = {
    T.Image: IMAGE,
    T.CameraInfo: CAMERA_INFO,
    T.PointCloud: POINT_CLOUD2,
    T.LaserScan: LASER_SCAN,
}


def canonical(type_name: str) -> str:
    """``sensor_msgs.msg.Image`` (an endpoint hint's spelling) as ``sensor_msgs/msg/Image``."""
    return type_name.replace(".", "/")


def message_type(payload_cls: type | None, hints: dict) -> str | None:
    """The message an endpoint with *hints* and payload type *payload_cls* travels as, if this module
    knows it: the ``type`` hint, else the payload type's default. ``None`` otherwise."""
    named = hints.get("type")
    name = canonical(named) if named else DEFAULT_TYPE.get(payload_cls)
    return name if name in SCHEMAS else None


# -- fields -------------------------------------------------------------------------------------

#: Bytes per pixel of each encoding an ``Image`` may carry.
IMAGE_ENCODING_BYTES = {"rgb8": 3, "mono8": 1, "16UC1": 2, "32FC1": 4}


def image_fields(array: np.ndarray, encoding: str) -> dict:
    """A raw frame as ``sensor_msgs/Image`` fields."""
    try:
        stride = IMAGE_ENCODING_BYTES[encoding]
    except KeyError:
        raise ValueError(
            f"unsupported image encoding {encoding!r}; expected one of "
            f"{sorted(IMAGE_ENCODING_BYTES)}"
        ) from None
    # The encoding and the array's dtype are chosen independently -- by the producer's hint and by
    # its render code -- and a mismatch is not a visible failure: `step` would be right for the
    # encoding while `data` holds a different number of bytes, so the far end decodes a garbled or
    # truncated image and nothing says why. Depth is where this bites (float32 metres under a 16UC1
    # hint, or the reverse), so it fails on the first publish instead.
    per_pixel = array.itemsize * (array.shape[2] if array.ndim > 2 else 1)
    if per_pixel != stride:
        raise ValueError(
            f"encoding {encoding!r} needs {stride} bytes per pixel, but the payload has "
            f"{per_pixel} (dtype {array.dtype}, shape {array.shape})"
        )
    height, width = array.shape[:2]
    return {
        "height": int(height),
        "width": int(width),
        "encoding": encoding,
        "is_bigendian": 0,
        "step": int(width * stride),
        "data": np.ascontiguousarray(array).tobytes(),
    }


# Compressed colour follows image_transport's convention for ``CompressedImage.format``:
# "<source encoding>; <codec> compressed <encoding of the decoded result>". The mapping is THEIRS, not
# ours -- consumers (cv_bridge, `image_transport republish`, rqt_image_view) read this string, so
# inventing our own spelling would break every one of them.
COMPRESSED_WIRE = {"rgb8": "bgr8", "bgr8": "bgr8", "mono8": "mono8"}


def compressed_image_fields(array: np.ndarray, encoding: str, hints: dict) -> dict:
    """A frame as ``sensor_msgs/CompressedImage`` fields: ``format`` and ``data``.

    Colour and depth share the message type and nothing else -- different codecs, framing and
    format strings -- so this dispatches on the ENCODING, not the codec: ``png`` names a colour codec
    and a depth codec, and they are different pipelines producing different bytes. The hints carry
    ``format`` (the codec), ``quality`` (JPEG) and ``png_level`` (depth PNG).
    """
    if encoding in COMPRESSED_WIRE:
        fmt = hints.get("format", "jpeg")
        return {
            "format": f"{encoding}; {fmt} compressed {COMPRESSED_WIRE[encoding]}",
            "data": image_codec.encode(
                array, fmt=fmt, quality=hints.get("quality", image_codec.DEFAULT_JPEG_QUALITY)
            ),
        }
    if encoding != "16UC1":
        # Everything that is not a colour encoding lands here, so the message names both routes.
        # 32FC1 is the interesting case: the reference carries it through an inverse-depth
        # QUANTISATION into 16 bits before PNG -- lossy, a different pipeline, and not implemented --
        # so the answer is to publish depth as 16UC1 millimetres, which is what hardware does anyway.
        raise TypeError(
            f"cannot compress encoding {encoding!r}: colour compresses as one of "
            f"{sorted(COMPRESSED_WIRE)}, and compressedDepth's codecs are 16-bit only. Publish the "
            "depth as 16UC1 (the device's own format), or leave it raw."
        )
    fmt = hints.get("format", "png")
    # image_transport's own spelling for this transport: "<source encoding>; compressedDepth <codec>".
    return {
        "format": f"{encoding}; compressedDepth {fmt}",
        "data": image_codec.encode_depth(
            array, fmt=fmt, png_level=hints.get("png_level", image_codec.DEFAULT_PNG_LEVEL)
        ),
    }


#: The ``CompressedImage`` codecs that decode to the same pixels: PNG (colour or depth) and RVL
#: (depth). JPEG is the lossy one.
LOSSLESS_FORMATS = frozenset({"png", "rvl"})


def is_lossless_format(fmt: str) -> bool:
    """Whether a ``CompressedImage`` written with codec *fmt* decodes to the pixels it was given."""
    return fmt in LOSSLESS_FORMATS


def camera_info_fields(info) -> dict:
    """Pinhole intrinsics (``width``, ``height``, ``fx``, ``fy``, ``cx``, ``cy``, ``d``) as
    ``sensor_msgs/CameraInfo`` fields.

    ``d`` is whatever the camera says its pixels carry -- zeros for a plain MuJoCo render (an ideal
    pinhole), the configured coefficients when the camera warped the frame to match a real lens.
    """
    fx, fy, cx, cy = (float(v) for v in (info.fx, info.fy, info.cx, info.cy))
    return {
        "height": int(info.height),
        "width": int(info.width),
        "distortion_model": "plumb_bob",
        "d": [float(v) for v in getattr(info, "d", None) or [0.0] * 5],
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


#: ``sensor_msgs/PointField``'s datatype constant for a float32 field.
POINT_FIELD_FLOAT32 = 7


def pointcloud_fields(points) -> dict:
    """An ``(N, 3)`` array of XYZ in the sensor frame as ``sensor_msgs/PointCloud2`` fields.

    An unorganized cloud (``height = 1``, ``width = N``) of three FLOAT32 fields; the raw ``(N, 3)``
    buffer is already the wire layout (x, y, z contiguous per point), so one ``tobytes`` copies it.
    ``fields`` holds ``(name, offset, datatype, count)`` per field.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    return {
        "height": 1,
        "width": n,
        "fields": [(axis, 4 * i, POINT_FIELD_FLOAT32, 1) for i, axis in enumerate("xyz")],
        "is_bigendian": False,
        "point_step": 12,
        "row_step": 12 * n,
        "data": pts.tobytes(),
        "is_dense": True,
    }


def scan_fields(scan) -> dict:
    """A planar sweep as ``sensor_msgs/LaserScan`` fields; ``inf`` is "no return", as REP 117 has it.

    ``time_increment`` and ``scan_time`` are zero (every ray of a cast is taken at one instant) and
    there are no intensities.
    """
    return {
        "angle_min": float(scan.angle_min),
        "angle_max": float(scan.angle_max),
        "angle_increment": float(scan.angle_increment),
        "time_increment": 0.0,
        "scan_time": 0.0,
        "range_min": float(scan.range_min),
        "range_max": float(scan.range_max),
        "ranges": np.ascontiguousarray(scan.ranges, dtype=np.float32),
        "intensities": np.zeros(0, dtype=np.float32),
    }


def _pixels(payload, hints: dict) -> tuple[np.ndarray, str]:
    """An image payload's pixels and encoding: a typed :class:`roqsim.types.Image`, or a bare array
    with its encoding in the hints."""
    if isinstance(payload, T.Image):
        return np.asarray(payload.data), payload.encoding
    return np.asarray(payload), hints.get("encoding", "rgb8")


def fields(type_name: str, payload, hints: dict) -> dict:
    """*payload*'s fields as the message *type_name* (header aside)."""
    if type_name == IMAGE:
        return image_fields(*_pixels(payload, hints))
    if type_name == COMPRESSED_IMAGE:
        array, encoding = _pixels(payload, hints)
        return compressed_image_fields(array, encoding, hints)
    if type_name == CAMERA_INFO:
        return camera_info_fields(payload)
    if type_name == POINT_CLOUD2:
        return pointcloud_fields(payload.points)
    if type_name == LASER_SCAN:
        return scan_fields(payload)
    raise ValueError(f"no ROS-free encoding for {type_name!r}; this module knows {sorted(SCHEMAS)}")


# -- CDR ----------------------------------------------------------------------------------------


class _Cdr:
    """Little-endian plain CDR, as ROS 2's default middleware serialises a message.

    Every primitive is aligned to its own size, counted from the end of the 4-byte encapsulation
    header; a sequence is a ``uint32`` count then its elements, a string a ``uint32`` length that
    includes its terminating NUL.
    """

    HEADER = b"\x00\x01\x00\x00"  # CDR_LE, no options

    def __init__(self) -> None:
        self.buf = bytearray(self.HEADER)

    def _align(self, size: int) -> None:
        pad = -(len(self.buf) - 4) % size
        if pad:
            self.buf += b"\x00" * pad

    def u8(self, value: int) -> None:
        self.buf.append(int(value) & 0xFF)

    def u32(self, value: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", int(value))

    def i32(self, value: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", int(value))

    def f32(self, value: float) -> None:
        self._align(4)
        self.buf += struct.pack("<f", float(value))

    def string(self, value: str) -> None:
        raw = value.encode("utf-8")
        self.u32(len(raw) + 1)
        self.buf += raw + b"\x00"

    def octets(self, data: bytes) -> None:
        self.u32(len(data))
        self.buf += data

    def f32_seq(self, values: np.ndarray) -> None:
        self.u32(len(values))
        if len(values):
            self._align(4)
            self.buf += np.ascontiguousarray(values, dtype="<f4").tobytes()

    def f64_array(self, values, *, sequence: bool) -> None:
        if sequence:
            self.u32(len(values))
        if len(values):
            self._align(8)
            self.buf += np.asarray(values, dtype="<f8").tobytes()

    def header(self, t: float, frame_id: str) -> None:
        sec, nanosec = stamp(t)
        self.i32(sec)
        self.u32(nanosec)
        self.string(frame_id)


def _write_image(w: _Cdr, f: dict) -> None:
    w.u32(f["height"])
    w.u32(f["width"])
    w.string(f["encoding"])
    w.u8(f["is_bigendian"])
    w.u32(f["step"])
    w.octets(f["data"])


def _write_compressed_image(w: _Cdr, f: dict) -> None:
    w.string(f["format"])
    w.octets(f["data"])


def _write_camera_info(w: _Cdr, f: dict) -> None:
    w.u32(f["height"])
    w.u32(f["width"])
    w.string(f["distortion_model"])
    w.f64_array(f["d"], sequence=True)
    w.f64_array(f["k"], sequence=False)
    w.f64_array(f["r"], sequence=False)
    w.f64_array(f["p"], sequence=False)
    for _ in range(2):  # binning_x, binning_y
        w.u32(0)
    for _ in range(4):  # roi: x_offset, y_offset, height, width
        w.u32(0)
    w.u8(0)  # roi.do_rectify


def _write_pointcloud(w: _Cdr, f: dict) -> None:
    w.u32(f["height"])
    w.u32(f["width"])
    w.u32(len(f["fields"]))
    for name, offset, datatype, count in f["fields"]:
        w.string(name)
        w.u32(offset)
        w.u8(datatype)
        w.u32(count)
    w.u8(f["is_bigendian"])
    w.u32(f["point_step"])
    w.u32(f["row_step"])
    w.octets(f["data"])
    w.u8(f["is_dense"])


def _write_scan(w: _Cdr, f: dict) -> None:
    for key in (
        "angle_min",
        "angle_max",
        "angle_increment",
        "time_increment",
        "scan_time",
        "range_min",
        "range_max",
    ):
        w.f32(f[key])
    w.f32_seq(f["ranges"])
    w.f32_seq(f["intensities"])


_WRITERS = {
    IMAGE: _write_image,
    COMPRESSED_IMAGE: _write_compressed_image,
    CAMERA_INFO: _write_camera_info,
    POINT_CLOUD2: _write_pointcloud,
    LASER_SCAN: _write_scan,
}


def encode(type_name: str, payload, t: float, hints: dict) -> bytes:
    """*payload* as the CDR bytes of a *type_name* message stamped *t* (sim seconds).

    *hints* are the endpoint's: ``frame_id`` (default :data:`DEFAULT_FRAME`), ``frame_prefix`` (the
    namespace the frame takes), and the codec keys :func:`compressed_image_fields` reads.
    """
    if type_name not in _WRITERS:
        raise ValueError(
            f"no ROS-free encoding for {type_name!r}; this module knows {sorted(_WRITERS)}"
        )
    values = fields(type_name, payload, hints)
    w = _Cdr()
    w.header(t, frame(hints, "frame_id", DEFAULT_FRAME[type_name]))
    _WRITERS[type_name](w, values)
    return bytes(w.buf)


# -- ros2msg definitions --------------------------------------------------------------------------

_SEPARATOR = "=" * 80

_DEFINITIONS = {
    "builtin_interfaces/Time": "int32 sec\nuint32 nanosec\n",
    "std_msgs/Header": "builtin_interfaces/Time stamp\nstring frame_id\n",
    "sensor_msgs/RegionOfInterest": (
        "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify\n"
    ),
    "sensor_msgs/PointField": (
        "uint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\n"
        "uint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\n"
        "string name\nuint32 offset\nuint8 datatype\nuint32 count\n"
    ),
    IMAGE: (
        "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n"
        "uint8 is_bigendian\nuint32 step\nuint8[] data\n"
    ),
    COMPRESSED_IMAGE: "std_msgs/Header header\nstring format\nuint8[] data\n",
    CAMERA_INFO: (
        "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n"
        "float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\n"
        "uint32 binning_y\nsensor_msgs/RegionOfInterest roi\n"
    ),
    POINT_CLOUD2: (
        "std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\n"
        "bool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n"
    ),
    LASER_SCAN: (
        "std_msgs/Header header\nfloat32 angle_min\nfloat32 angle_max\nfloat32 angle_increment\n"
        "float32 time_increment\nfloat32 scan_time\nfloat32 range_min\nfloat32 range_max\n"
        "float32[] ranges\nfloat32[] intensities\n"
    ),
}

_DEPENDENCIES = {
    IMAGE: ("std_msgs/Header", "builtin_interfaces/Time"),
    COMPRESSED_IMAGE: ("std_msgs/Header", "builtin_interfaces/Time"),
    CAMERA_INFO: ("std_msgs/Header", "builtin_interfaces/Time", "sensor_msgs/RegionOfInterest"),
    POINT_CLOUD2: ("std_msgs/Header", "builtin_interfaces/Time", "sensor_msgs/PointField"),
    LASER_SCAN: ("std_msgs/Header", "builtin_interfaces/Time"),
}


def _schema(type_name: str) -> str:
    parts = [_DEFINITIONS[type_name]]
    for dep in _DEPENDENCIES[type_name]:
        parts.append(f"{_SEPARATOR}\nMSG: {dep}\n{_DEFINITIONS[dep]}")
    return "".join(parts)


#: Message type -> its ``ros2msg`` definition, dependencies appended as mcap's ``ros2`` profile wants.
SCHEMAS = {name: _schema(name) for name in _DEPENDENCIES}
