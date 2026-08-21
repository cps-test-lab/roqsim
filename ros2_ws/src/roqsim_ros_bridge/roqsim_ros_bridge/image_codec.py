"""Compress a rendered frame to bytes: pixels in, bytes out.

Colour goes out as JPEG/PNG (:func:`encode`) and depth as ``compressedDepth`` (:func:`encode_depth`),
whose codec lives in :mod:`roqsim_ros_bridge.rvl`; this module owns the framing that turns either into
the bytes a ``CompressedImage`` carries.

Deliberately free of ROS imports so it is testable without a sourced ROS, and so the ROS-facing
concerns stay in :mod:`roqsim_ros_bridge.registry` -- which encoding string maps to which wire format,
and what ``CompressedImage.format`` should say. This module knows only about arrays.

**Channel order: none is applied, and none is wanted.** JPEG stores YCbCr, and Pillow converts
RGB -> YCbCr with the same matrix OpenCV uses for BGR -> YCbCr, so a frame handed here as RGB decodes
correctly through ``cv2.imdecode`` (which yields BGR) without any swap. Swapping first -- the obvious
"but cv_bridge wants BGR" move -- produces visibly wrong colour and costs a full-frame copy. There is
a regression test for exactly this.

**``subsampling`` and ``optimize`` are pinned, and are not redundant defaults.** With these values
Pillow's JPEG output is byte-identical to ``cv2.imencode(".jpg", bgr, [IMWRITE_JPEG_QUALITY, q])``,
which is what ``image_transport`` -- and therefore real camera drivers -- put on the wire. That parity
is what lets a compressed stream from here be compared against one recorded from hardware at the same
quality setting; it is covered by a test, because leaving it to Pillow's defaults would make it a
property of the installed Pillow version rather than of this code.
"""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image

from . import rvl

#: Wire format -> Pillow writer name. The keys are what an endpoint's ``format`` hint may ask for.
_PILLOW_FORMAT = {"jpeg": "JPEG", "png": "PNG"}

#: ``image_transport``'s own default, and what camera drivers ship with.
DEFAULT_JPEG_QUALITY = 95


def encode(array: np.ndarray, *, fmt: str = "jpeg", quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Compress ``array`` to ``fmt`` bytes.

    ``array`` is an 8-bit frame: ``(H, W)`` grayscale or ``(H, W, 3)`` colour, in whatever channel
    order the caller's encoding declares (no reordering happens here -- see the module docstring).

    ``fmt="png"`` is lossless and supported for completeness, but it is not a live-stream option: a
    1280x720 frame measures ~335 ms to encode against ~2.9 ms for JPEG q95, and 1.26 MB against
    154 kB. Encoding runs on the physics thread, so PNG at video rate would dominate the step.
    """
    try:
        pillow_format = _PILLOW_FORMAT[fmt]
    except KeyError:
        raise ValueError(
            f"unsupported compressed format {fmt!r}; expected one of {sorted(_PILLOW_FORMAT)}"
        ) from None

    # Loudly, because the alternative is Pillow either raising something opaque about a buffer or
    # silently reinterpreting the bytes. A depth frame arriving here is the likely cause, and it is a
    # different wire format entirely rather than a quality setting -- see the registry's converter.
    if array.dtype != np.uint8:
        raise TypeError(
            f"compressed images need an 8-bit frame, got dtype {array.dtype}. A 16-bit or float "
            f"depth frame cannot be carried as CompressedImage."
        )
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError(f"expected (H, W) or (H, W, 3), got {array.shape}")
    if array.ndim not in (2, 3):
        raise ValueError(f"expected (H, W) or (H, W, 3), got {array.shape}")

    buf = io.BytesIO()
    if pillow_format == "JPEG":
        # Pinned for byte-parity with cv2/image_transport -- see the module docstring. Do not drop
        # these as "the defaults anyway": the parity, not the values, is the contract.
        Image.fromarray(array).save(
            buf, "JPEG", quality=int(quality), subsampling=2, optimize=False
        )
    else:
        Image.fromarray(array).save(buf, pillow_format)
    return buf.getvalue()


#: ``compressed_depth_image_transport``'s ``ConfigHeader``: an int32 format enum (``INV_DEPTH`` = 0,
#: which the reference writes for every stream) and two float32 quantisation parameters. Those are
#: read back only for a 32FC1 source, so zeros here -- the reference leaves them *uninitialised* on
#: the 16-bit path, which is why a byte comparison against it can only start after this header.
_CONFIG_HEADER = struct.pack("<iff", 0, 0.0, 0.0)

#: The reference encoder's own ``depth_max`` default. It zeroes everything beyond this *before*
#: compressing, so a stream from here matches a driver's for the same pixels; a camera whose range
#: reaches past it must not offer the compressed topic (the sensor plugin refuses that config).
DEFAULT_DEPTH_MAX_M = 10.0


def encode_depth(
    array: np.ndarray, *, fmt: str = "rvl", depth_max_m: float = DEFAULT_DEPTH_MAX_M
) -> bytes:
    """Frame a uint16 millimetre depth image as ``compressedDepth`` payload bytes.

    Layout, from ``compressed_depth_image_transport``'s ``codec.cpp``: the 12-byte ``ConfigHeader``,
    then ``uint32`` columns and rows, then the codec's own stream. ``msg.format`` (the registry's
    job) is what tells a subscriber which codec produced it.

    Only ``rvl`` is offered. The other ``compressedDepth`` codec is PNG over an inverse-depth
    *quantisation* of 32FC1 metres -- lossy, and a different pipeline; a caller that wants depth
    compressed publishes it as 16UC1 millimetres, which is what the hardware does anyway.
    """
    if fmt != "rvl":
        raise ValueError(f"unsupported compressed-depth format {fmt!r}; expected 'rvl'")
    if array.dtype != np.uint16:
        raise TypeError(
            f"compressedDepth needs 16-bit millimetre depth, got dtype {array.dtype}. A float32 "
            f"metre frame is carried as 32FC1 raw, or converted by the sensor (depth_encoding)."
        )
    if array.ndim != 2:
        raise ValueError(f"expected (H, W), got {array.shape}")
    rows, cols = array.shape
    limit = int(depth_max_m * 1000.0)
    if array.max(initial=0) > limit:
        # The reference's max-depth filter, mirrored so the bytes agree with a driver's. Reaching it
        # means the raw and compressed topics of one camera would disagree, which the sensor plugin
        # rejects at load time -- so this is a backstop, not a routine path.
        array = np.where(array > limit, np.uint16(0), array)
    return _CONFIG_HEADER + struct.pack("<II", cols, rows) + rvl.compress(array)
