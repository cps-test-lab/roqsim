"""The timeline a replay scrubs: where a time lands, and how playing advances it.

Pure: numpy and nothing else. The window that draws a slider and the driver that ticks it both work
through this, so "which sample is showing" has one answer and one rounding rule.

Two things here differ from :class:`roqsim.recording.Recording`, both deliberately:

* **Seeking clamps rather than raises.** ``Recording.index_at`` refuses a time outside the recording,
  which is right for ``--at``: a render of a moment that is not in the file is a wrong answer. A
  slider dragged to its end, or a play that runs past the last sample, is not a wrong answer -- it is
  the end of the recording -- so it lands on the last sample.
* **Playing accumulates sim seconds, not samples.** :meth:`Timeline.advance` moves a cursor by
  ``elapsed x speed`` and re-derives the index from it, so a tick that took too long skips the frames
  it missed instead of queueing them. A recording with uneven spacing plays at the right speed for the
  same reason: the cursor is in the recording's own time axis.

Ties resolve to the earlier sample, matching ``Recording.index_at``, so the index this reports and the
index a render of the same time produces are the same sample.
"""

from __future__ import annotations

import numpy as np


class Timeline:
    """The sample axis of one recording, plus a cursor: seek it, step it, play it."""

    def __init__(self, times, fps: float) -> None:
        self.times = np.asarray(times, dtype=float)
        if self.times.size == 0:
            raise ValueError("a timeline needs at least one sample")
        self.fps = float(fps)
        self._index = 0
        self._cursor = float(self.times[0])

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def index(self) -> int:
        """Which sample is showing."""
        return self._index

    @property
    def time(self) -> float:
        """That sample's own sim time -- never the time that was asked for."""
        return float(self.times[self._index])

    @property
    def span(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    @property
    def at_end(self) -> bool:
        return self._index >= len(self) - 1

    # -- moving --------------------------------------------------------------------------------

    def seek_index(self, index: int) -> int:
        """Show sample ``index``, clamped into the recording. Returns where it landed."""
        self._index = int(min(max(int(index), 0), len(self) - 1))
        self._cursor = self.time
        return self._index

    def seek_time(self, when: float) -> int:
        """Show the sample nearest ``when``, clamped into the recording. Returns where it landed.

        The cursor keeps the *requested* time within the recording's span, so playing on from a seek
        continues from where the drag left off rather than from the sample's own timestamp.
        """
        low, high = self.span
        self._cursor = min(max(float(when), low), high)
        self._index = self._nearest(self._cursor)
        return self._index

    def step(self, count: int) -> int:
        """Move ``count`` samples (negative goes back), clamped. Returns where it landed."""
        return self.seek_index(self._index + int(count))

    def advance(self, elapsed: float, speed: float = 1.0) -> tuple[int, bool]:
        """Play for ``elapsed`` real seconds at ``speed``. Returns ``(index, hit_end)``.

        ``hit_end`` says the cursor reached a boundary on this call, which is what a caller pauses or
        loops on; it is reported once, when the cursor arrives, rather than on every later tick.
        """
        low, high = self.span
        before = self._cursor
        self._cursor += float(elapsed) * float(speed)
        hit_end = (self._cursor >= high > before) or (self._cursor <= low < before)
        self._cursor = min(max(self._cursor, low), high)
        self._index = self._nearest(self._cursor)
        return self._index, hit_end

    def _nearest(self, when: float) -> int:
        """The sample nearest ``when``, ties to the earlier one -- ``Recording.index_at``'s rule."""
        pos = int(np.searchsorted(self.times, when, side="left"))
        if pos <= 0:
            return 0
        if pos >= len(self):
            return len(self) - 1
        before, after = self.times[pos - 1], self.times[pos]
        return pos - 1 if (when - before) <= (after - when) else pos

    # -- saying where it is --------------------------------------------------------------------

    def label(self) -> str:
        """``"12.480 s   #312 / 1044"`` -- what the transport window and the overlay both show."""
        return f"{format_time(self.time)} s   #{self._index} / {len(self) - 1}"

    def bar(self, width: int = 20) -> str:
        """An ASCII progress bar of ``width`` cells, for the overlay, which has no widgets."""
        width = max(int(width), 1)
        filled = 0 if len(self) == 1 else round(self._index / (len(self) - 1) * width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"


def parse_time(text: str) -> float:
    """Read a time a person typed: ``"12.48"``, ``"12.48s"``, ``"1:05.2"``.

    Minutes are accepted because a long run is read off a bar in minutes, and the ``s`` because it is
    what :func:`format_time`'s own label shows. Anything else raises ``ValueError`` naming what was
    typed -- a time box that silently became 0.0 would seek to the start of the run.
    """
    raw = str(text).strip().rstrip("s").strip()
    if not raw:
        raise ValueError("no time given")
    minutes, sep, seconds = raw.partition(":")
    try:
        if sep:
            return float(minutes) * 60.0 + float(seconds)
        return float(raw)
    except ValueError:
        raise ValueError(f"{text!r} is not a time (try 12.48, 12.48s or 1:05.2)") from None


def format_time(seconds: float) -> str:
    """A time as the boxes and labels write it: fixed to milliseconds, and parseable back."""
    return f"{float(seconds):.3f}"
