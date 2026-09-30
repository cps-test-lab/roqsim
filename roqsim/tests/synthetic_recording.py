"""Write a recording by hand, for tests that need to state the motion they are testing.

A test of the onset detector wants "falls for one second, then drives", and a test of the health
checks wants "an hour at realtime, then the clock wedges" -- neither is something a simulated world
can be tuned to produce on demand. So these write the file directly through the same writer the
recorder uses (:class:`roqsim.mcap_format.ChunkedWriter`), with whatever rows the test dictates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from roqsim.mcap_format import (
    CHANNEL_CLOCK,
    CHANNEL_JOINTS,
    CHANNEL_POSES,
    CHANNEL_STATE,
    FORMAT_VERSION,
    META_ENTITIES,
    META_RECORDING,
    POSE_WIDTH,
    PROFILE,
    ChunkedWriter,
    json_bytes,
    ns,
    register_channels,
)


def write_state_recording(
    path: Path, meta: dict, samples: np.ndarray, *, finish: bool = True
) -> Path:
    """A recording holding ``samples`` (a ``record_dtype`` array) on its ``state`` channel.

    ``meta`` is completed with the format version. The JSON channels carry one message per sample
    too, so a reader that wants them finds the shape the recorder writes: a ``clock`` pair, and
    empty ``poses``/``joints`` documents.
    """
    path = Path(path)
    meta = {"format_version": FORMAT_VERSION, **meta}
    writer = ChunkedWriter(path)
    writer.start(PROFILE, library="synthetic")
    writer.add_json_metadata(META_RECORDING, meta)
    channels = register_channels(writer)
    epoch0 = float(meta.get("wall_start_epoch") or 1.7e9)
    for row in samples:
        t, w = float(row["t"]), float(row["w"])
        log_time, publish_time = ns(t), ns(epoch0 + w)
        writer.add_message(channels[CHANNEL_STATE], log_time, row.tobytes(), publish_time)
        writer.add_message(
            channels[CHANNEL_POSES],
            log_time,
            json_bytes({"t": t, "w": epoch0 + w, "bodies": {}}),
            publish_time,
        )
        writer.add_message(
            channels[CHANNEL_JOINTS],
            log_time,
            json_bytes({"t": t, "w": epoch0 + w, "q": {}}),
            publish_time,
        )
        writer.add_message(
            channels[CHANNEL_CLOCK],
            log_time,
            json_bytes({"wall_ts": epoch0 + w, "sim_ts": t}),
            publish_time,
        )
    if finish:
        writer.finish()
    else:
        writer.close_chunk()
        writer.abandon()
    return path


class RowWriter:
    """Write ``clock`` and ``poses`` messages one sample at a time, closing chunks on request.

    What the health tests drive: a clock row is one message, a sample's poses are one message
    holding every body, and ``chunk()`` is the recorder's once-a-second chunk close.
    """

    def __init__(self, path: Path, *, roster: list[dict] | None = None, meta: dict | None = None):
        self.path = Path(path)
        self.writer = ChunkedWriter(self.path)
        self.writer.start(PROFILE, library="synthetic")
        self.writer.add_json_metadata(
            META_RECORDING, {"format_version": FORMAT_VERSION, **(meta or {})}
        )
        if roster is not None:
            self.writer.add_json_metadata(META_ENTITIES, {"entities": roster})
        self.channels = register_channels(self.writer)
        self.uncompressed = 0

    def roster(self, entities: list[dict]) -> None:
        self.writer.add_json_metadata(META_ENTITIES, {"entities": entities})

    def clock(self, wall: float, sim: float) -> None:
        data = json_bytes({"wall_ts": wall, "sim_ts": sim})
        self.uncompressed += len(data)
        self.writer.add_message(self.channels[CHANNEL_CLOCK], ns(sim), data, ns(wall))

    def poses(
        self, sim: float, bodies: dict[str, tuple[float, float, float]], wall: float | None = None
    ) -> None:
        wall = sim if wall is None else wall
        rows = {name: [*pos, 0.0, 0.0, 0.0, 1.0, *([0.0] * 6)] for name, pos in bodies.items()}
        assert all(len(v) == POSE_WIDTH for v in rows.values())
        data = json_bytes({"t": sim, "w": wall, "bodies": rows})
        self.uncompressed += len(data)
        self.writer.add_message(self.channels[CHANNEL_POSES], ns(sim), data, ns(wall))

    def raw(self, topic: str, data: bytes, sim: float = 0.0) -> None:
        """A message as given -- for a test that wants an unreadable one in the file."""
        self.writer.add_message(self.channels[topic], ns(sim), data, ns(sim))

    def chunk(self) -> None:
        self.writer.close_chunk()

    def finish(self) -> Path:
        self.writer.finish()
        return self.path

    def abandon(self) -> Path:
        """Close the open chunk and leave the file without a summary: a killed run."""
        self.writer.close_chunk()
        self.writer.abandon()
        return self.path
