"""Compress a rendered frame to bytes: pixels in, bytes out.

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

import numpy as np
from PIL import Image

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
