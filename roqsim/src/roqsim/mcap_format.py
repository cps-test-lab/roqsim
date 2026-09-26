"""The recording's on-disk shape: what the writer emits, and the framing loop that reads it back.

A run is recorded as **one mcap file** (https://mcap.dev) and nothing else. This module holds the
part of that which is pure format -- the channel and metadata names, the JSON schemas, the chunking
writer and the record framing -- and deliberately imports no MuJoCo, so that ``roqsim health`` can
follow a growing recording without importing the simulator it is checking.

**Why the reader is our own loop and not the library's.** The ``mcap`` package's reader wants the
summary section a finished file ends with, and refuses a file without one. A run killed with SIGKILL
has no summary: it has a header, every chunk closed before the kill, and possibly the front of one
more. That file is exactly the one somebody needs to read, so :func:`iter_records` walks the records
itself -- magic, then ``opcode, u64 length, body`` until the bytes run out -- and treats a partial
record at the tail as the end rather than as an error.

**Why chunks are closed by wall time as well as by size.** The library's writer closes a chunk when
it is full, which on a slow world can be minutes of samples held in memory and lost with the process.
:class:`ChunkedWriter` adds :meth:`ChunkedWriter.close_chunk`, and the recorder calls it at least once
per wall second, so a kill costs at most the open chunk.

The layout of every record here is the mcap specification's, spelled out in :mod:`struct` calls
rather than delegated to the library's record classes, so that a file written by any conforming
writer reads the same way and a change to the library's internals cannot change what this reads.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import zstandard
from mcap.opcode import Opcode
from mcap.writer import CompressionType, IndexType, Writer

#: The eight bytes an mcap file starts and ends with.
MAGIC = b"\x89MCAP0\r\n"

#: The profile the header names. A reader refuses any other: the channels below are this profile's.
PROFILE = "roqsim"

#: The recording's own format version, carried in the ``roqsim.recording`` metadata. Versions 1 and
#: 2 were numpy archives and have no reader here; a version above this one is refused by name.
FORMAT_VERSION = 3

#: The suffix a recording carries. What ``is_recording`` tests and what a bare ``--record out``
#: is completed with.
RECORDING_SUFFIX = ".mcap"

#: Channel topics. ``state`` is the MuJoCo state vector, the primary artifact; the three JSON channels
#: are decoded observables a reader without the model can use.
CHANNEL_STATE = "state"
CHANNEL_POSES = "poses"
CHANNEL_JOINTS = "joints"
CHANNEL_CLOCK = "clock"

#: Message encoding of the ``state`` channel: little-endian ``f64 t, f64 w, f32[state_size]`` and,
#: when the recording carries a camera track, ``f32[9]``. The sizes are in the recording metadata.
STATE_ENCODING = "roqsim.state"

#: Metadata record names. ``roqsim.recording`` is the provenance (written at the start and again at
#: close); ``roqsim.entities`` is the entity roster (at the start and whenever it changes). A reader
#: takes the **last** record of a name.
META_RECORDING = "roqsim.recording"
META_ENTITIES = "roqsim.entities"

#: The one key of every metadata record: its value is a JSON document.
META_KEY = "json"

#: A chunk closes when it holds this much, whatever the clock says.
CHUNK_BYTES = 256 * 1024

#: ... and at least this often in wall seconds, so a kill loses at most this much of a run.
CHUNK_SECONDS = 1.0

#: The width of one body's row in the ``poses`` channel: position, quaternion ``(x, y, z, w)``, and
#: the world-frame twist (linear, angular).
POSE_WIDTH = 13

_NUMBER = {"type": "number"}

#: JSON schema of a ``poses`` message: every recorded body at one sample.
POSES_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "roqsim.poses",
    "type": "object",
    "properties": {
        "t": {**_NUMBER, "description": "simulated seconds"},
        "w": {**_NUMBER, "description": "wall clock, seconds since the Unix epoch"},
        "bodies": {
            "type": "object",
            "description": "body name -> [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz], "
            "world frame; the twist from the solver, not from differencing",
            "additionalProperties": {
                "type": "array",
                "items": _NUMBER,
                "minItems": POSE_WIDTH,
                "maxItems": POSE_WIDTH,
            },
        },
    },
    "required": ["t", "w", "bodies"],
}

#: JSON schema of a ``joints`` message: every recorded hinge and slide joint at one sample.
JOINTS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "roqsim.joints",
    "type": "object",
    "properties": {
        "t": {**_NUMBER, "description": "simulated seconds"},
        "w": {**_NUMBER, "description": "wall clock, seconds since the Unix epoch"},
        "q": {
            "type": "object",
            "description": "joint name -> position, radians for a hinge and metres for a slide",
            "additionalProperties": _NUMBER,
        },
    },
    "required": ["t", "w", "q"],
}

#: JSON schema of a ``clock`` message: the pair a reader outside the process relates its own stamps to.
CLOCK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "roqsim.clock",
    "type": "object",
    "properties": {
        "wall_ts": {**_NUMBER, "description": "wall clock, seconds since the Unix epoch"},
        "sim_ts": {**_NUMBER, "description": "simulated seconds"},
    },
    "required": ["wall_ts", "sim_ts"],
}

#: Topic -> (schema name, schema document) for the JSON channels, in registration order.
JSON_CHANNELS = (
    (CHANNEL_POSES, "roqsim.poses", POSES_SCHEMA),
    (CHANNEL_JOINTS, "roqsim.joints", JOINTS_SCHEMA),
    (CHANNEL_CLOCK, "roqsim.clock", CLOCK_SCHEMA),
)

_RECORD_HEADER = struct.Struct("<BQ")
_CHUNK_HEAD = struct.Struct("<QQQI")
_MESSAGE_HEAD = struct.Struct("<HIQQ")


class RecordingError(RuntimeError):
    """A recording cannot be written or read (see the message)."""


def recording_path(path: str | Path) -> Path:
    """``path`` carrying :data:`RECORDING_SUFFIX`, so a bare ``--record out`` lands as ``out.mcap``.

    Settled up front rather than at close: what :meth:`roqsim.capture.StateRecorder.close` returns,
    what a replay is handed and what ``roqsim health`` finds have to be one name.
    """
    path = Path(path)
    return (
        path
        if path.suffix.lower() == RECORDING_SUFFIX
        else path.with_name(path.name + RECORDING_SUFFIX)
    )


def ns(seconds: float) -> int:
    """Seconds as the integer nanoseconds an mcap timestamp holds."""
    return int(round(seconds * 1e9))


def json_bytes(document) -> bytes:
    """A compact JSON encoding: no whitespace, since every sample carries one of these per channel."""
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


# ==================================================================================================
# Writing
# ==================================================================================================


class ChunkedWriter(Writer):
    """The library's writer with one addition: a chunk can be closed on demand.

    The base class closes a chunk only when it exceeds ``chunk_size``; everything since the last one
    sits in memory until then, and dies with the process. :meth:`close_chunk` finalises the open
    chunk and flushes the file, which is what lets a recorder promise that a kill loses at most a
    second of a run. It reaches the base class's chunk finalisation by its mangled name, because the
    library exposes no hook: the constructor checks that the name still exists, so a library release
    that renames it fails here, loudly, rather than silently turning the guarantee off.
    """

    _FINALIZE = "_Writer__finalize_chunk"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._stream = open(self.path, "wb")  # noqa: SIM115 - closed by finish()/abandon()
        try:
            super().__init__(
                self._stream,
                chunk_size=CHUNK_BYTES,
                compression=CompressionType.ZSTD,
                # No per-chunk message index: nothing here seeks by time within a chunk, and with
                # chunks closed every second the indexes would rival the messages in size.
                index_types=IndexType.CHUNK | IndexType.METADATA | IndexType.ATTACHMENT,
            )
        except Exception:
            self._stream.close()
            raise
        if not callable(getattr(self, self._FINALIZE, None)):
            self._stream.close()
            raise RecordingError(
                f"the installed mcap writer has no {self._FINALIZE!r}: chunks could not be closed "
                "by time, so a killed run would lose everything since its last full chunk. "
                "Pin mcap to a release that has it."
            )

    def close_chunk(self) -> None:
        """Finalise the open chunk (if it holds any message) and flush the file to the OS."""
        getattr(self, self._FINALIZE)()
        self._stream.flush()

    def flush(self) -> None:
        """Flush what is written so far -- the header and metadata before any chunk has closed."""
        self._stream.flush()

    def finish(self) -> None:
        """Write the summary section and close the file. The file is complete after this."""
        try:
            super().finish()
        finally:
            self._stream.close()

    def abandon(self) -> None:
        """Close the file without a summary -- what a writer that cannot finish leaves behind."""
        self._stream.close()

    def add_json_metadata(self, name: str, document) -> None:
        """A metadata record whose one value is ``document`` as JSON (see :data:`META_KEY`)."""
        self.add_metadata(name, {META_KEY: json.dumps(document)})


def register_channels(writer: Writer) -> dict[str, int]:
    """Register the four channels on ``writer``, in a fixed order, and return topic -> channel id."""
    channels = {CHANNEL_STATE: writer.register_channel(CHANNEL_STATE, STATE_ENCODING, 0)}
    for topic, schema_name, schema in JSON_CHANNELS:
        schema_id = writer.register_schema(schema_name, "jsonschema", json_bytes(schema))
        channels[topic] = writer.register_channel(topic, "json", schema_id)
    return channels


# ==================================================================================================
# Reading
# ==================================================================================================


@dataclass(frozen=True)
class Channel:
    id: int
    schema_id: int
    topic: str
    message_encoding: str


@dataclass(frozen=True)
class Schema:
    id: int
    name: str
    encoding: str
    data: bytes


@dataclass(frozen=True)
class Message:
    channel_id: int
    log_time: int
    publish_time: int
    data: bytes


@dataclass(frozen=True)
class ChunkInfo:
    """A chunk's header: enough to decide whether to decompress it."""

    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    compression: str


def frame(buf, pos: int = 0, stop: int | None = None) -> tuple[list[tuple[int, memoryview]], int]:
    """Every complete ``(opcode, body)`` record in ``buf[pos:stop]``, and where the last one ended.

    The one framing rule of the format: a record is one opcode byte, a little-endian ``u64`` body
    length, then the body. A record cut short by the end of the buffer is not returned, and the
    offset handed back stops *before* it -- so a reader following a growing file resumes there.
    """
    view = memoryview(buf)
    end = len(view) if stop is None else stop
    out: list[tuple[int, memoryview]] = []
    while pos + _RECORD_HEADER.size <= end:
        opcode, length = _RECORD_HEADER.unpack_from(view, pos)
        body_start = pos + _RECORD_HEADER.size
        if body_start + length > end:
            break
        out.append((opcode, view[body_start : body_start + length]))
        pos = body_start + length
    return out, pos


def _string(view: memoryview, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", view, pos)
    return bytes(view[pos + 4 : pos + 4 + n]).decode("utf-8"), pos + 4 + n


def _string_map(view: memoryview, pos: int) -> tuple[dict[str, str], int]:
    (n,) = struct.unpack_from("<I", view, pos)
    pos += 4
    end = pos + n
    out: dict[str, str] = {}
    while pos < end:
        key, pos = _string(view, pos)
        value, pos = _string(view, pos)
        out[key] = value
    return out, end


def parse_header(body: memoryview) -> tuple[str, str]:
    profile, pos = _string(body, 0)
    library, _ = _string(body, pos)
    return profile, library


def parse_schema(body: memoryview) -> Schema:
    (sid,) = struct.unpack_from("<H", body, 0)
    name, pos = _string(body, 2)
    encoding, pos = _string(body, pos)
    (n,) = struct.unpack_from("<I", body, pos)
    return Schema(sid, name, encoding, bytes(body[pos + 4 : pos + 4 + n]))


def parse_channel(body: memoryview) -> Channel:
    cid, schema_id = struct.unpack_from("<HH", body, 0)
    topic, pos = _string(body, 4)
    encoding, _ = _string(body, pos)
    return Channel(cid, schema_id, topic, encoding)


def parse_message(body: memoryview) -> Message:
    channel_id, _sequence, log_time, publish_time = _MESSAGE_HEAD.unpack_from(body, 0)
    return Message(channel_id, log_time, publish_time, bytes(body[_MESSAGE_HEAD.size :]))


def parse_metadata(body: memoryview) -> tuple[str, dict[str, str]]:
    name, pos = _string(body, 0)
    values, _ = _string_map(body, pos)
    return name, values


def parse_chunk_info(body: memoryview) -> ChunkInfo:
    """The chunk's header alone -- read before deciding whether the chunk is worth decompressing."""
    start, end, size, _crc = _CHUNK_HEAD.unpack_from(body, 0)
    compression, _ = _string(body, _CHUNK_HEAD.size)
    return ChunkInfo(start, end, size, compression)


def chunk_records(body: memoryview) -> list[tuple[int, memoryview]]:
    """Decompress a chunk record's body and frame the records inside it."""
    info = parse_chunk_info(body)
    _, pos = _string(body, _CHUNK_HEAD.size)
    (n,) = struct.unpack_from("<Q", body, pos)
    payload = body[pos + 8 : pos + 8 + n]
    if info.compression == "zstd":
        try:
            data = zstandard.ZstdDecompressor().decompress(
                bytes(payload), max_output_size=info.uncompressed_size
            )
        except zstandard.ZstdError as err:
            raise RecordingError(f"a chunk could not be decompressed ({err})") from err
    elif info.compression == "":
        data = bytes(payload)
    elif info.compression == "lz4":
        try:
            import lz4.frame  # type: ignore[import-not-found]
        except ImportError as err:
            raise RecordingError(
                "the recording's chunks are lz4-compressed and the lz4 package is not installed"
            ) from err
        data = lz4.frame.decompress(bytes(payload))
    else:
        raise RecordingError(f"a chunk uses the compression {info.compression!r}, which is unknown")
    records, _ = frame(data)
    return records


def iter_records(buf, pos: int = 0) -> Iterator[tuple[int, memoryview]]:
    """Every record of an mcap byte buffer in file order, chunks expanded, stopping at a torn tail.

    ``buf`` starts with the magic. The summary section a finished file ends with is walked too --
    its schema and channel records repeat the data section's, which is harmless -- and the walk
    stops at the footer. A file whose last record is incomplete simply ends early, which is what
    reading a killed run means.
    """
    view = memoryview(buf)
    if len(view) < len(MAGIC) or bytes(view[: len(MAGIC)]) != MAGIC:
        raise RecordingError("not an mcap file: the magic bytes are missing")
    records, _ = frame(view, pos or len(MAGIC))
    for opcode, body in records:
        if opcode == Opcode.CHUNK:
            yield from chunk_records(body)
        elif opcode == Opcode.FOOTER:
            return
        else:
            yield opcode, body


@dataclass
class Contents:
    """What one read of a recording found: the header, every metadata record and every message.

    ``metadata`` maps a name to the **last** record of that name, since the recorder writes some of
    them more than once. ``messages`` holds each channel's messages in file order.
    """

    profile: str = ""
    library: str = ""
    metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    schemas: dict[int, Schema] = field(default_factory=dict)
    channels: dict[int, Channel] = field(default_factory=dict)
    messages: dict[str, list[Message]] = field(default_factory=dict)
    finished: bool = False
    unknown_channel: int = 0

    def json_metadata(self, name: str):
        """The JSON document of the last metadata record called ``name``, or ``None``."""
        record = self.metadata.get(name)
        if record is None or META_KEY not in record:
            return None
        try:
            return json.loads(record[META_KEY])
        except ValueError as err:
            raise RecordingError(f"metadata {name!r} is not JSON ({err})") from err


def read_contents(buf) -> Contents:
    """Read a whole recording's bytes into :class:`Contents`."""
    out = Contents(finished=is_finished_bytes(buf))
    for opcode, body in iter_records(buf):
        if opcode == Opcode.MESSAGE:
            message = parse_message(body)
            channel = out.channels.get(message.channel_id)
            if channel is None:
                out.unknown_channel += 1
                continue
            out.messages.setdefault(channel.topic, []).append(message)
        elif opcode == Opcode.CHANNEL:
            channel = parse_channel(body)
            out.channels[channel.id] = channel
        elif opcode == Opcode.SCHEMA:
            schema = parse_schema(body)
            out.schemas[schema.id] = schema
        elif opcode == Opcode.METADATA:
            name, values = parse_metadata(body)
            out.metadata[name] = values
        elif opcode == Opcode.HEADER:
            out.profile, out.library = parse_header(body)
    return out


def read_file(path: str | Path) -> Contents:
    """Read a recording from disk. Works on a finished file and on a killed run's file alike."""
    return read_contents(Path(path).read_bytes())


def is_finished_bytes(buf) -> bool:
    """Whether the bytes end with a footer record and the closing magic: a summary was written.

    A recorder writes the summary from :meth:`ChunkedWriter.finish`, i.e. when a run ends on
    purpose; a killed run never reaches it. That is the one unambiguous end-of-run mark the file
    carries, which is what ``roqsim health --watch`` stops on.
    """
    view = memoryview(buf)
    tail = len(MAGIC) + _RECORD_HEADER.size + 20  # footer body: two u64 and a u32 crc
    if len(view) < len(MAGIC) + tail:
        return False
    if bytes(view[-len(MAGIC) :]) != MAGIC:
        return False
    opcode, length = _RECORD_HEADER.unpack_from(view, len(view) - tail)
    return opcode == Opcode.FOOTER and length == 20


def is_finished(path: str | Path) -> bool:
    """:func:`is_finished_bytes` on a file, reading only its tail."""
    path = Path(path)
    tail = 2 * len(MAGIC) + _RECORD_HEADER.size + 20
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if size < tail:
                return False
            handle.seek(size - tail)
            return is_finished_bytes(handle.read())
    except OSError:
        return False


def read_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """Every top-level metadata record of a recording, last of a name winning, reading nothing else.

    Metadata is written outside the chunks, so this seeks past every chunk body and reads only the
    record headers and the metadata bodies: the cost is the number of records, not the run's length.
    """
    out: dict[str, dict[str, str]] = {}
    with Path(path).open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise RecordingError(f"{path} is not an mcap file: the magic bytes are missing")
        while True:
            head = handle.read(_RECORD_HEADER.size)
            if len(head) < _RECORD_HEADER.size:
                return out
            opcode, length = _RECORD_HEADER.unpack(head)
            if opcode == Opcode.FOOTER:
                return out
            if opcode == Opcode.METADATA:
                body = handle.read(length)
                if len(body) < length:
                    return out
                name, values = parse_metadata(memoryview(body))
                out[name] = values
            else:
                handle.seek(length, 1)
