"""Rebuild a world's sensor streams from a recording, on demand: camera, depth, point cloud, scan.

    roqsim rebuild --state run.mcap --list                      # what can be rebuilt, as JSON
    roqsim rebuild --state run.mcap --sensor 'robot/oakd*' --at 12.5 13.0
    roqsim rebuild --state run.mcap --from 8 --to 20 --rate 5 --output sensors.mcap

A run recorded with ``roqsim sim --record`` keeps the full MuJoCo state per sample, so a stream that
is a function of that state -- a rendered frame, a ray cast -- need not be produced or stored while
the run happens. This produces it afterwards, **as the sensor itself would have**: each sample is
restored and the sensor plugin's own :meth:`~roqsim.plugin.Plugin.capture` runs on it, with its
intrinsics, resolution, frame ids, depth encoding, point-cloud layout and its noise, which is keyed on
the run's seed, episode and sim time (:meth:`roqsim.context.SimContext.rng_for`) and so is the noise
the live run drew. A sensor declares that it can be rebuilt (:attr:`roqsim.plugin.Plugin.rebuild_cost`);
nothing here names one.

**The interface is an iterator, and nothing is persisted.** :func:`open_streams` builds the recorded
world once; :meth:`Streams.frames` then yields the frames asked for and renders only those, each
moment restored directly from its sample -- random access, no replay from the start, one renderer per
sensor kept for the whole session, memory bounded by one frame per stream. A service holds one
:class:`Streams` open and answers each request from it::

    from roqsim.rebuild import open_streams

    with open_streams("run.mcap") as streams:
        chosen = streams.select(["robot/oakd_camera"])          # None: every rebuildable sensor
        for frame in streams.frames(chosen, at=[12.5, 13.0]):   # or start=, stop=, rate=
            frame.sim_time, frame.stream.topic, frame.payload   # the sensor's own payload
            frame.message                                       # its ROS 2 message, CDR bytes

A frame's ``payload`` and ``message`` hold until the iterator advances: the next capture may reuse
the sensor's buffers, as :meth:`roqsim.recording.Recording.range` re-poses one ``MjData``.

**Messages are what the ROS 2 bridge publishes**, encoded without ROS by :mod:`roqsim.ros_messages`,
which the bridge's own converters use: the same message type, fields, frame id and stamp (the sample's
sim time). A stream's topic and frame id are the ones the bridge gives it with its defaults -- the
endpoint's namespace, no bridge-wide ``namespace``, ``strip_namespace`` or ``frame_prefix``.

**Which moments.** ``at`` takes sim times, each served by the sample nearest it (never interpolated:
a blend of two states is a pose the simulation never had; the frame says which sample it is).
``start``/``stop`` take a range; its rate is the recording's rate divided by a whole number, since a
frame can only exist where a state was recorded. A ``rate`` above the recording's is refused with the
highest it supports, and one that does not divide it is refused with its two neighbours -- never
served by repeating samples. Without ``rate`` each sensor runs at the highest such rate not above its
own (a 10 Hz camera in a 25 fps recording: 8.33 Hz), which a recording at a multiple of the sensor
rate avoids.

**Which streams by default.** Every stream of every rebuildable sensor that has a ROS message, with
one choice made for size: where a stream is offered raw and compressed, the compressed one is taken
when its codec is lossless (``compressedDepth``, PNG colour) and the raw one otherwise (JPEG colour is
lossy, and a lossy default is not chosen silently). A selection by stream path takes exactly that
stream.

**What a recording cannot hold**, and a rebuild therefore cannot reproduce: a sensor fault written
while the run was in progress (a rebuild casts nominal), and a model change at run time that is not
presence (``model_override``'s values).

:func:`write_mcap` is one sink over the iterator, for a caller that wants the frames as a file: one
mcap, ``ros2`` profile, zstd-compressed chunks with the time and channel index, and the manifest
(:meth:`Streams.manifest`) as its ``roqsim.rebuild`` metadata.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from . import exit_status, logging_setup
from . import ros_messages as rm
from .endpoint import hints_for, topic_of
from .mcap_format import RecordingError, json_bytes, ns
from .recording import Recording, open_recording

log = logging.getLogger(__name__)

#: The version of the manifest's shape.
MANIFEST_VERSION = 1

#: The metadata record a written mcap carries its manifest under.
META_REBUILD = "roqsim.rebuild"


class RebuildError(RecordingError):
    """A rebuild cannot produce what was asked (see the message): a wrong request."""

    exit_status = exit_status.BAD_INPUT


@dataclass(frozen=True)
class Stream:
    """One stream a recording can rebuild: an output endpoint of a rebuildable sensor."""

    #: ``<sensor>/<endpoint>``, the path ``roqsim endpoints`` shows for it in a running simulation.
    path: str
    #: The producing sensor's path (its address, dots as slashes).
    sensor: str
    #: The endpoint's name.
    name: str
    #: The ROS topic the bridge publishes it on.
    topic: str
    #: The ROS message type, ``sensor_msgs/msg/Image`` etc.
    type: str
    #: The frame its messages are stamped in.
    frame_id: str
    #: The sensor's own rate.
    rate_hz: float
    #: What producing it costs (:meth:`roqsim.plugin.Plugin.output_cost`), ``None`` when cheap.
    cost: str | None
    #: The endpoint's ROS hints with ``frame_prefix`` set: what :func:`roqsim.ros_messages.encode` reads.
    hints: dict = field(repr=False, compare=False, hash=False)

    def describe(self) -> dict:
        out = {
            "path": self.path,
            "sensor": self.sensor,
            "topic": self.topic,
            "type": self.type,
            "frame_id": self.frame_id,
            "rate_hz": self.rate_hz,
            "cost": self.cost,
        }
        if self.type == rm.COMPRESSED_IMAGE:
            out["format"] = self.hints.get("format")
        return out


@dataclass
class Frame:
    """One stream's output at one restored sample."""

    stream: Stream
    sim_time: float
    sample_index: int
    #: The sensor's own payload (:mod:`roqsim.types`), as its endpoint returns it.
    payload: object
    _message: bytes | None = field(default=None, repr=False)

    @property
    def message(self) -> bytes:
        """The ROS 2 message the bridge would publish, CDR-serialised. Encoded on first access."""
        if self._message is None:
            self._message = rm.encode(
                self.stream.type, self.payload, self.sim_time, self.stream.hints
            )
        return self._message


def _path(producer: str, name: str) -> str:
    return f"{producer.replace('.', '/')}/{name}" if producer else name


def _ros_topic(ep, topic: str) -> str:
    """The topic as the bridge resolves it with its defaults: an absolute one verbatim, a relative one
    under the endpoint's namespace."""
    if topic.startswith("/"):
        return topic
    return "/" + "/".join(p for p in (ep.namespace.strip("/"), topic) if p)


def rebuildable_streams(ctx, plugins) -> tuple[list[Stream], dict[str, object]]:
    """Every stream the world in *ctx*, built with *plugins*, can rebuild, and ``{sensor path:
    plugin}``.

    A stream is an ``out`` endpoint of a plugin that declares
    :attr:`~roqsim.plugin.Plugin.rebuild_cost`, carried on ROS as a message
    :mod:`roqsim.ros_messages` encodes. Endpoints of such a plugin that are not (a detection list, a
    fault report, a static transform) are not streams.
    """
    plugins = {p.address: p for p in plugins if p.rebuild_cost is not None}
    streams: list[Stream] = []
    for ep in ctx.interface.all():
        plugin = plugins.get(ep.producer)
        if plugin is None or ep.direction != "out" or ep.read is None:
            continue
        hints = hints_for(ep, "ros2")
        if hints is None or hints.get("static") or hints.get("field"):
            continue
        payload_cls = getattr(ep.payload_type, "cls", None)
        msg_type = rm.message_type(payload_cls, hints)
        topic = topic_of(ep, "ros2")
        if msg_type is None or topic is None:
            continue
        hints = {**hints, "frame_prefix": ep.namespace.strip("/")}
        streams.append(
            Stream(
                path=_path(ep.producer, ep.name),
                sensor=ep.producer.replace(".", "/"),
                name=ep.name,
                topic=_ros_topic(ep, topic),
                type=msg_type,
                frame_id=rm.frame(hints, "frame_id", rm.DEFAULT_FRAME[msg_type]),
                rate_hz=float(ep.rate_hz),
                cost=plugin.output_cost(ep),
                hints=hints,
            )
        )
    return streams, {a.replace(".", "/"): p for a, p in plugins.items()}


def default_streams(streams: list[Stream]) -> list[Stream]:
    """*streams* without the redundant wire format of each raw/compressed pair (module docstring)."""
    drop: set[str] = set()
    for s in streams:
        if s.type != rm.COMPRESSED_IMAGE:
            continue
        parent = s.topic.rpartition("/")[0]
        raw = next(
            (
                r
                for r in streams
                if r.sensor == s.sensor and r.type == rm.IMAGE and r.topic == parent
            ),
            None,
        )
        if rm.is_lossless_format(s.hints.get("format", "jpeg")):
            if raw is not None:
                drop.add(raw.path)
        else:
            drop.add(s.path)
    return [s for s in streams if s.path not in drop]


class Streams:
    """A recording opened for rebuilding its sensor streams; see the module docstring."""

    def __init__(self, rec: Recording, target: str | None = None) -> None:
        self.recording = rec
        if rec.meta.get("seed") is None:
            raise RebuildError(
                f"{rec.path} records no seed, so a rebuilt sensor's noise would not be the noise "
                "the run drew. Re-record it with a seed."
            )
        t0 = time.perf_counter()
        _model, self._ctx = rec.build(target)
        self.build_seconds = time.perf_counter() - t0
        self.streams, self._plugins = rebuildable_streams(self._ctx, self._ctx.engine.plugins)

    def close(self) -> None:
        self.recording.close()

    def __enter__(self) -> Streams:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- selection --------------------------------------------------------------------------------

    def select(self, patterns: Iterable[str] | None = None) -> list[Stream]:
        """The streams *patterns* name: a sensor path takes its default streams, a stream path that
        stream; globs allowed. ``None``: the default streams of every rebuildable sensor. A pattern
        that matches nothing is an error naming what exists."""
        if patterns is None:
            return default_streams(self.streams)
        chosen: list[Stream] = []
        for pattern in patterns:
            sensors = [s for s in self._plugins if fnmatch.fnmatchcase(s, pattern)]
            hits = [s for s in self.streams if fnmatch.fnmatchcase(s.path, pattern)]
            if sensors:
                hits += default_streams([s for s in self.streams if s.sensor in sensors])
            if not hits:
                have = ", ".join(
                    sorted({s.sensor for s in self.streams} | {s.path for s in self.streams})
                )
                raise RebuildError(
                    f"--sensor {pattern!r} names no rebuildable sensor or stream of this world. "
                    f"It has: {have or 'none'}."
                )
            chosen += [h for h in hits if h not in chosen]
        return chosen

    # -- moments ----------------------------------------------------------------------------------

    @property
    def fps(self) -> Fraction:
        return self.recording.fps

    def step_for(self, rate: float | None, sensor_rate: float) -> int:
        """Samples between two frames: for *rate*, which must divide the recording's rate, else the
        highest rate on the recording's grid that is not above *sensor_rate*."""
        fps = float(self.fps)
        if rate is None:
            return max(1, math.ceil(fps / sensor_rate - 1e-9)) if sensor_rate > 0 else 1
        if rate <= 0:
            raise RebuildError(f"--rate must be > 0, got {rate:g}")
        if rate > fps * (1 + 1e-9):
            raise RebuildError(
                f"--rate {rate:g} Hz is above this recording's {fps:g} fps: a frame exists only "
                f"where a state was recorded, so {fps:g} Hz is the most it can rebuild."
            )
        k = fps / rate
        if abs(k - round(k)) > 1e-6:
            lo, hi = math.floor(k), math.ceil(k)
            raise RebuildError(
                f"--rate {rate:g} Hz does not divide this recording's {fps:g} fps; the nearest it "
                f"holds are {fps / hi:g} and {fps / lo:g} Hz."
            )
        return int(round(k))

    def schedule(
        self,
        chosen: list[Stream],
        *,
        at: Iterable[float] | None = None,
        start: float | None = None,
        stop: float | None = None,
        rate: float | None = None,
    ) -> list[tuple[int, list[str]]]:
        """``[(sample index, [sensor path, ...])]`` in order: which sensors capture at which sample."""
        sensors = list(dict.fromkeys(s.sensor for s in chosen))
        if not sensors:
            raise RebuildError("no stream selected, so there is nothing to rebuild")
        rec = self.recording
        if at is not None:
            if start is not None or stop is not None or rate is not None:
                raise RebuildError("--at names moments; it takes no --from/--to/--rate")
            indices = list(dict.fromkeys(rec.index_at(float(t)) for t in at))
            return [(i, sensors) for i in indices]
        lo = 0 if start is None else rec.index_at(start)
        hi = len(rec) - 1 if stop is None else rec.index_at(stop)
        if hi < lo:
            lo, hi = hi, lo
        steps = {
            s: self.step_for(rate, max(x.rate_hz for x in chosen if x.sensor == s)) for s in sensors
        }
        plan = []
        for index in range(lo, hi + 1):
            due = [s for s in sensors if (index - lo) % steps[s] == 0]
            if due:
                plan.append((index, due))
        return plan

    # -- frames -----------------------------------------------------------------------------------

    def frames(
        self,
        chosen: list[Stream] | None = None,
        *,
        at: Iterable[float] | None = None,
        start: float | None = None,
        stop: float | None = None,
        rate: float | None = None,
    ) -> Iterator[Frame]:
        """Yield the frames of *chosen* (default :meth:`select`'s) at the moments asked, in time order.

        Each moment restores its sample and runs each due sensor's ``capture`` once; its streams
        follow. A sensor that raises, or produces nothing, stops the iteration with the time and the
        sensor named -- a previous frame is never handed out as this one.
        """
        chosen = self.select() if chosen is None else chosen
        plan = self.schedule(chosen, at=at, start=start, stop=stop, rate=rate)
        by_sensor: dict[str, list[Stream]] = {}
        for s in chosen:
            by_sensor.setdefault(s.sensor, []).append(s)
        endpoints = {
            _path(ep.producer, ep.name): ep
            for ep in self._ctx.interface.all()
            if ep.direction == "out" and ep.producer
        }
        for index, sensors in plan:
            sample = self.recording.sample(index)
            for sensor in sensors:
                try:
                    self._plugins[sensor].capture(self._ctx)
                except Exception as err:
                    raise RebuildError(
                        f"{sensor} could not capture at t={sample.sim_time:.6f} s "
                        f"(sample {index}): {err}"
                    ) from err
                for stream in by_sensor[sensor]:
                    payload = endpoints[stream.path].read()
                    if payload is None:
                        raise RebuildError(
                            f"{stream.path} produced nothing at t={sample.sim_time:.6f} s "
                            f"(sample {index}) after its sensor captured."
                        )
                    yield Frame(stream, sample.sim_time, index, payload)

    # -- metadata ---------------------------------------------------------------------------------

    def manifest(
        self,
        chosen: list[Stream],
        *,
        at: Iterable[float] | None = None,
        start: float | None = None,
        stop: float | None = None,
        rate: float | None = None,
    ) -> dict:
        """What :meth:`frames` would produce for these arguments, without rendering anything: the
        source recording, and per stream its topic, type, frame, rate, frame count and time span."""
        at = None if at is None else list(at)
        plan = self.schedule(chosen, at=at, start=start, stop=stop, rate=rate)
        times = self.recording.times
        out_streams = []
        for s in chosen:
            idx = [i for i, sensors in plan if s.sensor in sensors]
            entry = s.describe()
            entry["frames"] = len(idx)
            entry["span"] = [float(times[idx[0]]), float(times[idx[-1]])] if idx else None
            if at is None and len(idx) > 1:
                entry["rebuilt_rate_hz"] = float(self.fps) / (idx[1] - idx[0])
            out_streams.append(entry)
        rec = self.recording
        return {
            "manifest_version": MANIFEST_VERSION,
            "source": {
                "path": str(rec.path),
                "sha256": _sha256(rec.path),
                "world": rec.world,
                "seed": rec.meta.get("seed"),
                "episode": rec.meta.get("episode") or 0,
                "fps": float(self.fps),
                "samples": len(rec),
                "span": list(rec.span),
                "packages": rec.meta.get("packages"),
            },
            "naming": "ROS 2 bridge defaults: each endpoint's namespace, no bridge-wide namespace",
            "message_encoding": "cdr",
            "frames": sum(e["frames"] for e in out_streams),
            "streams": out_streams,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_streams(state: str | Path, target: str | None = None) -> Streams:
    """Open the recording at *state* and rebuild its world (or *target*, as
    :meth:`roqsim.recording.Recording.build` takes it), ready to yield frames."""
    rec = open_recording(state)
    try:
        return Streams(rec, target)
    except BaseException:
        rec.close()
        raise


# -- the mcap sink ------------------------------------------------------------------------------------

#: Chunk size of a written file: large, since nothing is lost to a kill of an offline writer, and a
#: larger chunk compresses better.
CHUNK_BYTES = 4 * 1024 * 1024


def write_mcap(frames: Iterable[Frame], out: str | Path, manifest: dict | None = None) -> dict:
    """Write *frames* to one mcap file at *out*: their messages on one channel per stream, the
    manifest as ``roqsim.rebuild`` metadata. Returns ``{stream path: frames written}``."""
    from mcap.writer import CompressionType, Writer

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(out, "wb") as fh:
        writer = Writer(fh, chunk_size=CHUNK_BYTES, compression=CompressionType.ZSTD)
        writer.start(profile="ros2", library="roqsim")
        schemas: dict[str, int] = {}
        channels: dict[str, int] = {}
        for frame in frames:
            s = frame.stream
            if s.path not in channels:
                if s.type not in schemas:
                    schemas[s.type] = writer.register_schema(
                        s.type, "ros2msg", rm.SCHEMAS[s.type].encode()
                    )
                channels[s.path] = writer.register_channel(
                    s.topic,
                    "cdr",
                    schemas[s.type],
                    metadata={"roqsim.stream": s.path, "frame_id": s.frame_id},
                )
            stamp = ns(frame.sim_time)
            writer.add_message(channels[s.path], stamp, frame.message, stamp)
            counts[s.path] = counts.get(s.path, 0) + 1
        if manifest is not None:
            writer.add_metadata(META_REBUILD, {"json": json_bytes(manifest).decode()})
        writer.finish()
    return counts


# -- the command --------------------------------------------------------------------------------------


def run_rebuild(
    state,
    target: str | None = None,
    *,
    sensors: list[str] | None = None,
    at: list[float] | None = None,
    start: float | None = None,
    stop: float | None = None,
    rate: float | None = None,
    output: str | Path | None = None,
    list_only: bool = False,
) -> dict:
    """What ``roqsim rebuild`` does; returns the JSON record it prints."""
    with open_streams(state, target) as streams:
        if list_only and not sensors:
            chosen = streams.streams
        else:
            chosen = streams.select(sensors)
        manifest = streams.manifest(chosen, at=at, start=start, stop=stop, rate=rate)
        if list_only:
            manifest["default"] = [s.path for s in default_streams(streams.streams)]
            return manifest
        t0 = time.perf_counter()
        first: list[float] = []

        def timed(frames):
            for frame in frames:
                if not first:
                    first.append(time.perf_counter() - t0)
                yield frame

        frames = timed(streams.frames(chosen, at=at, start=start, stop=stop, rate=rate))
        if output is not None:
            counts = write_mcap(frames, output, manifest)
        else:
            counts = {}
            for frame in frames:
                frame.message  # noqa: B018 - encoding is part of what is timed
                counts[frame.stream.path] = counts.get(frame.stream.path, 0) + 1
        seconds = time.perf_counter() - t0
        total = sum(counts.values())
        steady = seconds - first[0] if first else 0.0
        return {
            **manifest,
            "output": None if output is None else str(Path(output).resolve()),
            "written": counts,
            "timing": {
                "build_s": round(streams.build_seconds, 3),
                "first_frame_s": round(first[0], 4) if first else None,
                "frames_s": round(seconds, 3),
                # After the first frame, which pays for the renderers' GL contexts.
                "steady_frames_per_s": round((total - 1) / steady, 1) if steady > 0 else None,
            },
        }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim rebuild",
        description=__doc__.split("\n")[0],
        epilog=exit_status.epilog(
            exit_status.BAD_INPUT,
            exit_status.RECORDING,
            note="The JSON record goes to stdout; frames go to --output, if given.",
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="the world, if the recording's own name cannot be resolved (normally omitted)",
    )
    parser.add_argument(
        "--state", required=True, metavar="PATH", help="a `roqsim sim --record` recording"
    )
    parser.add_argument(
        "--sensor",
        action="extend",
        nargs="+",
        metavar="PATH",
        help="a sensor (its default streams) or one stream, by path; globs allowed (default: "
        "every rebuildable sensor; see --list)",
    )
    parser.add_argument(
        "--at", action="extend", nargs="+", type=float, metavar="T", help="moments (s of sim time)"
    )
    parser.add_argument("--from", dest="start", type=float, metavar="T", help="range start (s)")
    parser.add_argument("--to", dest="stop", type=float, metavar="T", help="range end (s)")
    parser.add_argument(
        "--rate",
        type=float,
        metavar="HZ",
        help="frames per second over a range: the recording's rate divided by a whole number "
        "(default: each sensor's own rate, on that grid)",
    )
    parser.add_argument(
        "--output", metavar="PATH", help="write the frames to this mcap file (default: none kept)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the streams and what would be rebuilt, rendering nothing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging_setup.configure(verbose=args.verbose, stream=sys.stderr)
    try:
        record = run_rebuild(
            args.state,
            args.target,
            sensors=args.sensor,
            at=args.at,
            start=args.start,
            stop=args.stop,
            rate=args.rate,
            output=args.output,
            list_only=args.list,
        )
    except RecordingError as err:
        return exit_status.fail("roqsim rebuild", err)
    print(json.dumps(record))
    return exit_status.OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
