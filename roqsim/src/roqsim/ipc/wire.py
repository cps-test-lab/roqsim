"""The control protocol's messages: one JSON frame, then one raw frame per array.

A message is a list of ZeroMQ frames. The first is a UTF-8 JSON object; its ``value`` carries the
payload, in which every numpy array is replaced by ``{"__nd__": i, "dtype": ..., "shape": [...]}``
and travels as frame ``i + 1``, its bytes unchanged. A dataclass, a namedtuple or an object with
public attributes becomes an object of its fields; a tuple becomes a list; a numpy scalar its Python
value. Bytes travel as a frame too (``{"__bytes__": i}``).

:func:`encode` copies each array once, on the thread that calls it: a producer may overwrite the
buffer it returned in its next step, and the frames are sent later from another thread. The frames
are then handed to ZeroMQ with ``copy=False``, so that one copy is the only one.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from typing import Any

import numpy as np

_ND = "__nd__"
_BYTES = "__bytes__"


class Encoded:
    """A payload already turned into its JSON part and its frames (see :func:`encode`)."""

    __slots__ = ("frames", "obj")

    def __init__(self, obj: Any, frames: list):
        self.obj = obj
        self.frames = frames


def encode(value: Any) -> Encoded:
    """*value* as a JSON-ready structure plus the frames it refers to."""
    frames: list = []
    return Encoded(_enc(value, frames), frames)


def _enc(value: Any, frames: list) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, Encoded):  # encoded earlier, on another thread: renumber its frames
        offset = len(frames)
        frames.extend(value.frames)
        return _shift(value.obj, offset)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.ndarray):
        frames.append(np.array(value, order="C", copy=True))
        return {_ND: len(frames) - 1, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return _enc(value.item(), frames)
    if isinstance(value, bytes | bytearray | memoryview):
        frames.append(bytes(value))
        return {_BYTES: len(frames) - 1}
    if isinstance(value, enum.Enum):
        return _enc(value.value, frames)
    if isinstance(value, dict):
        return {str(k): _enc(v, frames) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _enc(getattr(value, f.name), frames) for f in dataclasses.fields(value)}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {k: _enc(v, frames) for k, v in zip(value._fields, value, strict=True)}
    if isinstance(value, list | tuple | set | frozenset):
        return [_enc(v, frames) for v in value]
    attrs = getattr(value, "__dict__", None)
    if attrs is not None:
        return {k: _enc(v, frames) for k, v in attrs.items() if not k.startswith("_")}
    return repr(value)


def _shift(obj: Any, offset: int) -> Any:
    if isinstance(obj, dict):
        if _ND in obj:
            return {**obj, _ND: obj[_ND] + offset}
        if _BYTES in obj:
            return {_BYTES: obj[_BYTES] + offset}
        return {k: _shift(v, offset) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shift(v, offset) for v in obj]
    return obj


def decode(obj: Any, frames: list) -> Any:
    """The inverse of :func:`encode`'s JSON part, with arrays read from *frames* (no copy)."""
    if isinstance(obj, dict):
        if _ND in obj:
            buf = frames[obj[_ND]]
            return np.frombuffer(buf, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        if _BYTES in obj:
            return bytes(frames[obj[_BYTES]])
        return {k: decode(v, frames) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v, frames) for v in obj]
    return obj


def pack(header: dict, value: Any = None) -> list:
    """A message: *header* with ``value`` set to *value*'s JSON part, then its frames."""
    enc = encode(value)
    return [json.dumps({**header, "value": enc.obj}).encode(), *enc.frames]


def unpack(frames: list) -> tuple[dict, Any]:
    """``(header, value)`` of a message; frames may be ``zmq.Frame`` or bytes."""
    raw = [f.buffer if hasattr(f, "buffer") else f for f in frames]
    header = json.loads(bytes(raw[0]))
    value = decode(header.pop("value", None), raw[1:])
    return header, value


def plain(value: Any, *, max_items: int = 0) -> Any:
    """*value* as JSON-ready Python for a person or a model to read.

    With *max_items*, an array larger than that is summarised (dtype, shape, min, max) rather
    than listed: a camera frame is a million numbers nobody reads.
    """
    if isinstance(value, np.ndarray):
        if max_items and value.size > max_items:
            summary = {"dtype": value.dtype.str, "shape": list(value.shape)}
            if value.size and np.issubdtype(value.dtype, np.number):
                summary.update(min=float(np.nanmin(value)), max=float(np.nanmax(value)))
            return {"array": summary}
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v, max_items=max_items) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v, max_items=max_items) for v in value]
    return value
