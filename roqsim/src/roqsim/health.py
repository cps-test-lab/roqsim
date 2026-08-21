# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Check a run for the three faults a simulation can have while still looking alive.

    roqsim health <run-dir> --robot base_link           # what the run looks like right now
    roqsim health <run-dir> --robot base_link --watch   # keep checking until something is wrong
    roqsim health --clock run.clock_map.csv --poses sim_poses.csv --watch

Three checks, and deliberately no more, because each is a fault that raises **no error anywhere**: a
simulation that never starts stepping, one that steps far slower than realtime, and a robot that
stands still for a simulated minute. A run with any of them looks healthy to every other signal --
the process is up, the log is quiet, the exit status is 0 -- while producing nothing worth
analysing. Everything else a run can get wrong already reports itself.

**A separate process, on purpose.** These checks are not a plugin and hold nothing inside the
simulator: a health check that ran in the simulated process would share the process's failure modes,
and one that ran over a transport bridge could not diagnose a broken bridge. This reads files, so it
can say something true about a run that is wedged, and about one that is already dead.

**It reads what the recorder already streams**, and adds nothing to a run:

===========================  ===================================  ==========================
file                         columns used                         serves
===========================  ===================================  ==========================
``*.clock_map.csv``          ``wall_ts`` (epoch), ``sim_ts``      checks 2 and 3
``sim_poses.csv``            ``timestamp``, ``frame``, position   check 1
===========================  ===================================  ==========================

Both are flushed per sample by :mod:`roqsim.capture`, which writes the clock record explicitly "for
readers outside the process". So the channel already existed; this is a reader for it, not a new
mechanism. Nothing here imports the model, rebuilds the world, or asks MuJoCo anything -- the two
files carry *decoded* observables, which is exactly why they and not the ``.npz`` are the source.

**The precondition, and it is a real one.** Those files appear only when a run is recording
(``ROQSIM_RECORD``, plus ``ROQSIM_SIM_POSES`` for poses), and writing them is best-effort even
then. Without them there is nothing to check, so this exits 2 saying exactly that, rather than
reporting health it never observed.

**A stall is read from silence, so the two modes ask different questions.** Both files are sampled
on *simulated*-time boundaries, so a frozen simulation writes no rows at all and the only evidence
for "it stopped" is that nothing arrives. Silence is therefore counted against a run **only while
there is reason to think it is still alive**:

* ``--watch`` is that reason -- it is watching a run it expects to continue -- so wall time keeps
  advancing while the record does not, and a wedge is caught. It stops without complaint when the
  recording's ``.npz`` appears, because that file is written by ``close()`` and so means the run
  ended on purpose rather than stopped dead.
* a one-shot check has no such premise: it is handed a directory and asked what is in it. It judges
  the span the record actually covers, because what happened after the last row is not something the
  file can say. Otherwise every finished run would be reported as a stall a minute after it ended.

Either way a message states what was *observed* -- how long since the last row, and where -- and
never asserts a cause.

Alongside the findings, ``--json`` reports a ``state`` block: the last pose of every root body and
the clock, with the sim-to-wall rate. It is here rather than in a command of its own because both
answers come from the same two records in the same read, and a caller asking "is anything wrong"
almost always wants "and where is it" next. A *latest-value* answer, not an interpolation to the
instant of the call -- these records are sampled, and the newest sample is what they honestly hold.

Deliberately not in it: the scenario's behaviour tree. That lives in a different record, needs a fold
over the whole file rather than a tail read, and belongs to whoever owns it -- while this command has
to stay cheap enough to poll.

Exit status: ``0`` nothing wrong (warnings are still printed), ``5`` an error-level finding, ``2``
the checks could not run at all. Exiting on a finding is how a backgrounded invocation reports one:
its output is invisible until it exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_FINDING = 5

#: Where the recorder puts each record. Duplicated from :mod:`roqsim.capture` rather than imported:
#: that module imports MuJoCo, and nothing in this one needs it. ``test_health.py`` asserts the two
#: agree, so the duplication cannot drift -- which is the only thing wrong with duplicating it.
CLOCK_MAP_SUFFIX = ".clock_map.csv"
SIM_POSE_FILENAME = "sim_poses.csv"
ENTITIES_FILENAME = "entities.json"

#: Check 1: a robot that moves less than this in :data:`MOTION_WINDOW_S` of sim time is standing still.
MOTION_M = 0.01
MOTION_WINDOW_S = 60.0

#: Check 2: how long sim time may take to start advancing.
START_TIMEOUT_S = 60.0

#: Check 3: the slowest a run may advance sim time and still be worth waiting for -- 5 s of sim per
#: 60 s of wall, i.e. 0.083x realtime. Far below any usable run, so this fires on a wedge and not on
#: a merely expensive world.
MIN_SIM_ADVANCE_S = 5.0
RATE_WINDOW_S = 60.0

#: Seconds between polls in ``--watch``. The checks answer questions measured in minutes, so this
#: only decides how promptly a finding is reported, never whether it is found.
POLL_S = 2.0

#: A sim_ts smaller than the one before it by more than this is a reset, not a rewind. The columns
#: are written at six decimals, so anything above rounding is a real step backwards.
SERIES_EPS = 1e-6

WARN = "warn"
ERROR = "error"


@dataclass(frozen=True)
class Finding:
    """One thing observed to be wrong. ``detail`` states the observation, never a diagnosis."""

    level: str  # WARN | ERROR
    check: str  # stable slug, so a caller can match on it
    detail: str

    def line(self) -> str:
        return f"{'ERROR' if self.level == ERROR else 'warn ':5s} {self.check:16s} {self.detail}"


@dataclass(frozen=True)
class ClockRow:
    wall_ts: float  # epoch seconds, as the recorder writes them
    sim_ts: float


@dataclass(frozen=True)
class PoseRow:
    """One body at one sample.

    Check 1 needs only ``pos``, so the rest carry defaults and existing callers are
    unaffected -- they are here because the record already holds them and a caller asking
    "where is everything" wants the pose it was written with, not a truncation of it. The
    twist comes from the solver rather than from differencing positions, which is the point
    of the record: a difference is only as good as the interval it is divided by.
    """

    sim_ts: float
    frame: str
    pos: tuple[float, float, float]
    wall_ts: float | None = None
    quat: tuple[float, float, float, float] | None = None  # x, y, z, w -- as the record orders it
    twist_linear: tuple[float, float, float] | None = None
    twist_angular: tuple[float, float, float] | None = None


# -- reading the records --------------------------------------------------------------------------


class Tail:
    """Follow a CSV another process is appending to, yielding only complete rows.

    Two things this has to survive, both of them normal in a run rather than exotic:

    **A partial trailing line.** The writer flushes per row, but a reader can still arrive between
    the write and the newline, and half a row parsed as a whole one would put a truncated float into
    a check.

    **The file being re-created underneath us.** Both records are best-effort by design: on an
    ``OSError`` the writer drops its handle, and the next sample reopens the path in mode ``"w"`` --
    truncating it and writing a fresh header (``capture.py:487``, ``capture.py:524``). A reader
    following a byte offset would then sit past EOF forever, read nothing, and let a check conclude
    that the run had gone silent. That is a false ERROR manufactured by a transient disk hiccup, so
    it is handled here rather than left to the checks: a file shorter than our position is a new
    file, and it is read from the beginning.

    Read in binary because the position has to be a byte offset that can be compared with a size;
    a text handle's ``tell()`` is documented as an opaque cookie. Lines are decoded one at a time,
    so a multi-byte character split across two reads cannot corrupt one.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.malformed = 0
        self.restarts = 0
        self._handle = None
        self._buffer = b""
        self._fields: list[str] | None = None

    def _restart(self) -> None:
        self.close()
        self._buffer = b""
        self._fields = None

    def rows(self) -> list[dict[str, str]]:
        """Every complete row appended since the last call. Empty while the file does not exist."""
        if self._handle is not None:
            try:
                truncated = self.path.stat().st_size < self._handle.tell()
            except OSError:  # unlinked mid-run: let the reopen below decide
                truncated = True
            if truncated:
                self.restarts += 1
                self._restart()
        if self._handle is None:
            if not self.path.exists():
                return []
            self._handle = self.path.open("rb")
        self._buffer += self._handle.read()
        *complete, self._buffer = self._buffer.split(b"\n")
        out: list[dict[str, str]] = []
        for raw in complete:
            try:
                line = raw.decode("utf-8").strip("\r")
            except UnicodeDecodeError:
                self.malformed += 1
                continue
            if not line:
                continue
            values = line.split(",")
            if self._fields is None:
                self._fields = values
                continue
            if len(values) != len(self._fields):
                # A row we cannot trust. Skipped, never fatal: a diagnostic that dies on its own
                # input is worse than one that reports on the rest.
                self.malformed += 1
                continue
            # strict: the length check above already guarantees they pair up.
            out.append(dict(zip(self._fields, values, strict=True)))
        return out

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def find_clock_record(run_dir: Path) -> Path | None:
    """The run's clock record. Named after the recording, so it is found by suffix, newest first.

    Looks in *run_dir* first, then one and two levels below it, stopping at the first depth that
    holds one. A caller does not always know which run is current: a supervisor may hold several
    runs under one output root and be able to name the root but not the run inside it.
    Newest-by-mtime is exactly the right answer there -- the record still being written is the run
    still going -- so searching saves the caller from guessing, and guessing wrong means reporting
    on a run that already ended.

    Two levels is the bound because that is the depth of the layout (``<root>/<config>/<run>``),
    not an arbitrary limit: a full walk over a campaign's output would cost far more than the
    question is worth. Shallowest-first so a caller that *did* name the run exactly is answered
    from it rather than from a sibling that happens to be newer.
    """
    for pattern in ("*", "*/*", "*/*/*"):
        found = list(run_dir.glob(pattern + CLOCK_MAP_SUFFIX))
        if found:
            return max(found, key=lambda p: p.stat().st_mtime)
    return None


class FileSource:
    """The two streamed records, read as they grow. The only source there is today.

    Kept behind this small surface -- ``clock()`` and ``poses()`` returning plain rows -- because the
    checks below are written against the rows and not against the files. A source that answered from
    somewhere else (an endpoint, for a checker that has to run off-node) would change nothing else.
    """

    def __init__(self, clock_path: Path | None, poses_path: Path | None) -> None:
        self.clock_path = clock_path
        self.poses_path = poses_path
        self._clock = Tail(clock_path) if clock_path else None
        self._poses = Tail(poses_path) if poses_path else None

    def clock(self) -> list[ClockRow]:
        if self._clock is None:
            return []
        out = []
        for row in self._clock.rows():
            try:
                out.append(ClockRow(float(row["wall_ts"]), float(row["sim_ts"])))
            except (KeyError, ValueError):
                self._clock.malformed += 1
        return out

    def poses(self) -> list[PoseRow]:
        if self._poses is None:
            return []
        out = []
        for row in self._poses.rows():
            try:
                out.append(
                    PoseRow(
                        float(row["timestamp"]),
                        row["frame"],
                        _triple(row, "position"),
                        wall_ts=float(row["wall_time"]),
                        quat=(
                            float(row["orientation.x"]),
                            float(row["orientation.y"]),
                            float(row["orientation.z"]),
                            float(row["orientation.w"]),
                        ),
                        twist_linear=_triple(row, "twist.linear"),
                        twist_angular=_triple(row, "twist.angular"),
                    )
                )
            except (KeyError, ValueError):
                self._poses.malformed += 1
        return out

    def tails(self) -> list[tuple[str, Tail]]:
        """The open readers, labelled -- so a report can say what happened to each record."""
        return [(n, t) for n, t in (("clock", self._clock), ("pose", self._poses)) if t is not None]

    def close(self) -> None:
        for _, tail in self.tails():
            tail.close()


def _triple(row: dict, prefix: str) -> tuple[float, float, float]:
    """``position`` -> ``(position.x, position.y, position.z)`` as floats."""
    return (float(row[f"{prefix}.x"]), float(row[f"{prefix}.y"]), float(row[f"{prefix}.z"]))


class SeriesSplitter:
    """Cut a row stream where sim time goes backwards, because that is a reset and not a rewind.

    ``StateRecorder.on_reset`` restarts the sample schedule and keeps writing to the **same** file,
    while the wall column deliberately keeps climbing ("real time did not restart"). A campaign
    resets between repetitions, so this is the common case and not an edge one.

    Measuring across that boundary is how a healthy run gets failed: check 3 differences the first
    and last sim stamp in its window, and a window spanning a reset sees a full minute of progress
    as zero or negative advance. So the boundary restarts the checks rather than being averaged
    through. The carry-over matters -- a reset usually falls *between* two polls, not inside one
    batch -- which is why the last stamp is remembered here rather than recomputed per call.

    Returns one list per series: the first continues whatever came before, and every later one
    begins at a boundary.
    """

    def __init__(self) -> None:
        self._last: float | None = None

    def split(self, rows: list) -> list[list]:
        series: list[list] = [[]]
        for row in rows:
            if self._last is not None and row.sim_ts < self._last - SERIES_EPS:
                series.append([])
            series[-1].append(row)
            self._last = row.sim_ts
        return series


# -- the checks -----------------------------------------------------------------------------------
#
# Each is fed rows and asked for findings, and each reports a given finding once: an agent told the
# same thing on every poll learns to ignore the channel. They hold their own thresholds because the
# thresholds are the check -- there is no knob for them anywhere else, on purpose.


class SimTimeStarts:
    """Check 2: sim time begins advancing within ``timeout``, measured from the first record.

    From the first record rather than from this process's start, so that pointing the checker at a
    run already in progress does not report a fault that happened before it was looking -- and so a
    one-shot run over a finished file judges the span the file actually covers.
    """

    slug = "sim-time-start"

    def __init__(self, timeout: float = START_TIMEOUT_S) -> None:
        self.timeout = timeout
        self.started = False
        self.rows = 0
        self._first: ClockRow | None = None
        self._latest: ClockRow | None = None
        self._reported = False

    def on_new_series(self) -> None:
        """A reset proves the loop is running, so the question this check asks is already answered."""
        self.started = True

    def update(self, rows: list[ClockRow]) -> None:
        for row in rows:
            self.rows += 1
            if self._first is None:
                self._first = row
            elif row.sim_ts > self._first.sim_ts:
                self.started = True
            self._latest = row

    def findings(self, now: float, origin: float) -> list[Finding]:
        if self.started or self._reported:
            return []
        # `origin` is this process's start; the first record wins when it is older, because a run
        # that was already stepping before we looked has already answered this question.
        since = self._first.wall_ts if self._first else origin
        waited = now - since
        if waited < self.timeout:
            return []
        self._reported = True
        where = "no clock record rows yet" if not self.rows else f"{self.rows} rows, sim time flat"
        return [
            Finding(
                ERROR,
                self.slug,
                f"sim time has not advanced in {waited:.0f} s ({where}); "
                "the run may not have reached its first step",
            )
        ]


class SimTimeRate:
    """Check 3: sim time advances at least ``min_advance`` per ``window`` of wall time.

    Wall time comes from this process's clock rather than from the newest row, which is what makes a
    silent record count against the run: if rows stop arriving, the measured advance stops while the
    window keeps sliding, and the rate falls. Judged only once a full window of history exists, so a
    run is never failed for the first minute of its life.
    """

    slug = "sim-time-rate"

    def __init__(
        self, min_advance: float = MIN_SIM_ADVANCE_S, window: float = RATE_WINDOW_S
    ) -> None:
        self.min_advance = min_advance
        self.window = window
        self._rows: deque[ClockRow] = deque()
        self._reported = False

    def on_new_series(self) -> None:
        """Drop the window: an advance measured across a reset is not an advance."""
        self._rows.clear()

    def update(self, rows: list[ClockRow]) -> None:
        self._rows.extend(rows)

    def findings(self, now: float, origin: float) -> list[Finding]:
        if self._reported or not self._rows:
            return []
        # Keep one row from before the window as the anchor: the advance is measured across the
        # whole window, so dropping every older row would shorten it to the newest arrivals.
        cutoff = now - self.window
        while len(self._rows) >= 2 and self._rows[1].wall_ts <= cutoff:
            self._rows.popleft()
        anchor, latest = self._rows[0], self._rows[-1]
        span = now - anchor.wall_ts
        if span < self.window:
            return []
        advance = latest.sim_ts - anchor.sim_ts
        if advance / span >= self.min_advance / self.window:
            return []
        self._reported = True
        quiet = now - latest.wall_ts
        return [
            Finding(
                ERROR,
                self.slug,
                f"sim time advanced {advance:.2f} s over {span:.0f} s of wall time "
                f"({advance / span:.3f}x realtime, floor {self.min_advance / self.window:.3f}x); "
                f"last row {quiet:.0f} s ago",
            )
        ]


class RobotMoves:
    """Check 1: every watched robot moves at least ``distance`` per ``window`` of sim time.

    A warning, never an error. A robot standing still for a simulated minute is often correct --
    waiting on a pedestrian, a perception-only run, a manipulator-only phase -- and a channel that
    interrupts healthy runs is one nobody reads.

    Measured in *sim* time, so a slow run is not also reported as a stuck one; check 3 owns that.
    """

    slug = "robot-motion"

    def __init__(
        self, robots: list[str], distance: float = MOTION_M, window: float = MOTION_WINDOW_S
    ) -> None:
        self.robots = set(robots)
        self.distance = distance
        self.window = window
        self.seen: set[str] = set()
        self._anchor: dict[str, tuple[float, tuple[float, float, float]]] = {}
        self._reported: set[str] = set()
        self._pending: list[Finding] = []
        self._frames: set[str] = (
            set()
        )  # every frame the record offers, for the "no such robot" report
        self._span: tuple[float, float] | None = None
        self._samples = 0  # distinct sample stamps seen; every sample writes every root body

    def on_new_series(self) -> None:
        """Re-anchor every robot. A reset either teleports it home (a delta that would mask a real
        stall) or restores the pose it already had (no delta, and a stall reported that never
        happened) -- both are artefacts of the boundary rather than motion."""
        self._anchor.clear()

    def update(self, rows: list[PoseRow]) -> None:
        # One row per root body per sample, so a frame is revisited once per sample. Each row is a
        # coherent snapshot of `sim - dt` carrying the label `sim` (capture.py states the
        # convention): a one-step lag that cancels in any difference, and this check differences
        # over a minute. Do not "correct" it -- it is what makes this table and the TF one describe
        # the same instant.
        for row in rows:
            self._frames.add(row.frame)
            if self._span is None:
                self._span, self._samples = (row.sim_ts, row.sim_ts), 1
            else:
                if row.sim_ts != self._span[1]:
                    self._samples += 1
                self._span = (self._span[0], row.sim_ts)
            if row.frame not in self.robots:
                continue
            self.seen.add(row.frame)
            anchor = self._anchor.get(row.frame)
            if anchor is None:
                self._anchor[row.frame] = (row.sim_ts, row.pos)
                continue
            since, origin = anchor
            if math.dist(row.pos, origin) >= self.distance:
                self._anchor[row.frame] = (row.sim_ts, row.pos)
            elif row.sim_ts - since >= self.window and row.frame not in self._reported:
                self._reported.add(row.frame)
                self._pending.append(
                    Finding(
                        WARN,
                        self.slug,
                        f"robot {row.frame!r} moved under {self.distance * 100:.0f} cm in "
                        f"{row.sim_ts - since:.0f} s of sim time",
                    )
                )

    def findings(self, now: float, origin: float) -> list[Finding]:
        out, self._pending = self._pending, []
        out.extend(self._missing())
        return out

    def inconclusive(self) -> str | None:
        """Why check 1 has no verdict, when it has none. ``None`` when it has one.

        A record shorter than the motion window cannot produce a stall, so reporting nothing wrong
        would overstate what was checked -- the same reason a robot nobody named is reported as
        skipped rather than passed.

        Deliberately phrased for **both** of the situations that produce it, because the caller
        routes it and they are not the same news: a closed record that never covered the window has
        skipped the check for good, while one still being written is simply early and will have a
        verdict shortly. Saying "no verdict was possible" served the first and misled the second --
        four seconds into a healthy run it reads as a gap in the checking rather than as the clock.
        """
        if not self.robots or self._span is None:
            return None
        span = self._span[1] - self._span[0]
        if span >= self.window:
            return None
        return (
            f"check 1 (robot-motion): the record covers {span:.0f} s of sim time, less than the "
            f"{self.window:.0f} s a robot must be still for"
        )

    def _missing(self) -> list[Finding]:
        """Robots named on the command line that the record never mentions.

        A name that matches nothing must never read as a pass. `roqsim state` draws the same line
        for the same reason -- "an unmatched selector is an error, never an empty column" -- and it
        is sharper here, because the answer this tool returns is a verdict: check 1 would report
        nothing wrong about a robot it never once looked at.

        Held only until a second sample has been seen, not until a full motion window: every sample
        writes every root body, so two of them establish the whole roster. Waiting the window would
        mean a mistyped name went unreported on every run shorter than a simulated minute -- which
        is the run most likely to be a quick check with a mistyped name in it.
        """
        if self._samples < 2:
            return []
        out = []
        for name in sorted(self.robots - self.seen - self._reported):
            self._reported.add(name)
            offered = ", ".join(sorted(self._frames)[:8]) or "none"
            out.append(
                Finding(
                    WARN,
                    self.slug,
                    f"robot {name!r} never appears in the pose record, so check 1 did not run for "
                    f"it; the record offers: {offered}" + (" ..." if len(self._frames) > 8 else ""),
                )
            )
        return out


# -- running them ---------------------------------------------------------------------------------


@dataclass
class Report:
    """What one run of the checks concluded."""

    findings: list[Finding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Where everything was at the newest sample read -- see :meth:`Monitor.state`. Reported
    #: alongside the findings because a caller asking "is anything wrong" almost always wants
    #: "and where is it" next, and both come from the records already open. Empty when the
    #: records held nothing to report.
    state: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return any(f.level == ERROR for f in self.findings)

    def as_json(self) -> str:
        return json.dumps(
            {
                "findings": [
                    {"level": f.level, "check": f.check, "detail": f.detail} for f in self.findings
                ],
                "skipped": self.skipped,
                "notes": self.notes,
                "state": self.state,
                "exit": EXIT_FINDING if self.failed else EXIT_OK,
            },
            indent=2,
        )


class Monitor:
    """The three checks over one source, polled until something is wrong or the caller stops."""

    def __init__(self, source: FileSource, robots: list[str], origin: float,
                 no_robots: str = "") -> None:
        self.source = source
        self.origin = origin
        self.report = Report()
        self.clock_checks = [SimTimeStarts(), SimTimeRate()]
        self.motion = RobotMoves(robots)
        self.robots = robots
        self._clock_series = SeriesSplitter()
        self._pose_series = SeriesSplitter()
        self._latest_row: float | None = None
        self.pose_rows = 0
        #: First and last clock row of the *current* series, for the sim-to-wall rate. Bounded
        #: to one series because a reset restarts sim time while wall time keeps climbing, so a
        #: window spanning one would report a healthy run as making no progress.
        self._clock_window: tuple[ClockRow, ClockRow] | None = None
        #: Newest row per body. Cleared at a reset: a re-posed world must not be described with
        #: positions from before it, which would be a stale answer presented as a current one.
        self._latest_pose: dict[str, PoseRow] = {}
        if not robots:
            # Why, not just that: "nothing to watch" is a different situation depending on whether
            # the roster was missing, held no robots, or was overridden with nothing -- and a
            # skipped check whose reason is unstated is one nobody can act on.
            self.report.skipped.append(
                "check 1 (robot-motion): nothing to watch. "
                + (no_robots or f"{SIM_POSE_FILENAME} names root bodies; it does not say which "
                                f"are robots, and no {ENTITIES_FILENAME} said either.")
            )

    @property
    def latest_wall(self) -> float:
        """Wall stamp of the newest clock row, or this process's start before any have arrived.

        What a one-shot check uses for "now" -- see the mode note in the module docstring.
        """
        return self._latest_row if self._latest_row is not None else self.origin

    def ingest(self) -> None:
        """Read whatever has arrived and feed it to the checks.

        Separate from :meth:`evaluate` because a one-shot check judges the record against its own
        newest stamp, which is not known until the rows have been read.
        """
        clock_rows = self.source.clock()
        if clock_rows:
            # The newest row, not the newest we have ever computed: a one-shot check judges the
            # record's own span, and every row of a finished run is in the past. Taking a max
            # against this process's start would quietly put the wall clock back in.
            self._latest_row = clock_rows[-1].wall_ts
        pose_rows = self.source.poses()
        self.pose_rows += len(pose_rows)
        # Split once and share: the splitters are stateful, so calling them twice would consume
        # the boundary and leave the second caller measuring across a reset.
        clock_series = self._clock_series.split(clock_rows)
        pose_series = self._pose_series.split(pose_rows)
        self._track_state(clock_series, pose_series)
        pairs = ((self.clock_checks, clock_series), ([self.motion], pose_series))
        for checks, series in pairs:
            for check in checks:
                self._guard(check, lambda c=check, s=series: self._feed(c, s))

    def _track_state(self, clock_series: list[list], pose_series: list[list]) -> None:
        """Remember the newest clock window and the newest row per body, for :meth:`state`.

        Reads the same already-split series the checks are fed, so the reset boundary is
        honoured identically -- and cheaply: this is a couple of assignments per row, on records
        that are being parsed anyway.
        """
        for index, rows in enumerate(clock_series):
            if index:  # a reset: the rate is only meaningful within one series
                self._clock_window = None
            for row in rows:
                first = self._clock_window[0] if self._clock_window else row
                self._clock_window = (first, row)
        for index, rows in enumerate(pose_series):
            if index:
                self._latest_pose.clear()
            for row in rows:
                self._latest_pose[row.frame] = row

    def state(self) -> dict:
        """Where everything was at the newest sample, and how fast sim time is running.

        A *latest-value* answer, which is what a caller asking "where is everything now" wants
        and all these records can honestly give: they are sampled, so this is the most recent
        sample and not an interpolation to the instant of the call.

        ``rate`` is sim seconds per wall second over the current series -- the same quantity
        check 3 judges, reported rather than judged, so a caller can see 0.05x without having to
        infer it from a finding. Absent when the window is too short to divide.

        ``kind`` is added by the caller from the recorder's roster when there is one (see
        :func:`read_roster`), and left off otherwise. Never guessed here: the pose record names root
        bodies without saying which are robots, and inferring the distinction from a name would be a
        guess presented as a fact.
        """
        out: dict = {}
        if self._clock_window is not None:
            first, last = self._clock_window
            out["sim_ts"] = round(last.sim_ts, 6)
            out["wall_ts"] = round(last.wall_ts, 6)
            wall_span = last.wall_ts - first.wall_ts
            if wall_span > 0:
                out["rate"] = round((last.sim_ts - first.sim_ts) / wall_span, 4)
        if self._latest_pose:
            out["entities"] = [
                {
                    "name": name,
                    "sim_ts": round(row.sim_ts, 6),
                    "position": [round(v, 6) for v in row.pos],
                    **({"orientation": [round(v, 6) for v in row.quat]} if row.quat else {}),
                    **({"twist_linear": [round(v, 6) for v in row.twist_linear]}
                       if row.twist_linear else {}),
                    **({"twist_angular": [round(v, 6) for v in row.twist_angular]}
                       if row.twist_angular else {}),
                }
                for name, row in sorted(self._latest_pose.items())
            ]
        return out

    @staticmethod
    def _feed(check, series: list[list]) -> None:
        for index, rows in enumerate(series):
            if index:  # every series after the first begins at a reset
                check.on_new_series()
            check.update(rows)

    def evaluate(self, now: float) -> list[Finding]:
        """Ask every check what it makes of what it has been given."""
        found: list[Finding] = []
        for check in (*self.clock_checks, self.motion):
            self._guard(check, lambda c=check: found.extend(c.findings(now, self.origin)), found)
        self.report.findings.extend(found)
        return found

    def _guard(self, check, work, sink: list[Finding] | None = None) -> None:
        """Run one check's work; a check that raises reports itself and the others carry on.

        Loud in the channel the checks own rather than a silent pass: the run is still unchecked on
        that axis and the caller has to know it. A diagnostic must not be able to fail a run, and it
        must not be able to fake one either.
        """
        try:
            work()
        except Exception as err:  # noqa: BLE001 -- one broken check must not take the others down
            log.debug("check %s raised", check.slug, exc_info=True)
            finding = Finding(WARN, check.slug, f"check errored and was abandoned: {err}")
            (sink if sink is not None else self.report.findings).append(finding)

    def poll(self, now: float | None = None) -> list[Finding]:
        """Read what has arrived and judge it. ``now`` defaults to the wall clock."""
        self.ingest()
        return self.evaluate(time.time() if now is None else now)


def recording_of(clock_path: Path) -> Path:
    """The ``.npz`` the clock record belongs to. Its existence means the recorder ran ``close()``.

    Which is the one unambiguous end-of-run signal these files carry: the archive is written once, at
    close, so it appears when a run finished on purpose and never when one was killed or is still
    going. That is exactly the distinction trailing silence cannot make on its own.
    """
    return Path(str(clock_path).removesuffix(CLOCK_MAP_SUFFIX) + ".npz")


def resolve_paths(args) -> tuple[Path | None, Path | None, str | None]:
    """Locate the two records. Returns ``(clock, poses, error)``; ``error`` is fatal when set."""
    if args.clock or args.poses:
        clock = Path(args.clock) if args.clock else None
        poses = Path(args.poses) if args.poses else None
        if clock and not clock.exists():
            return None, None, f"no clock record at {clock}"
        return clock, poses, None
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        return None, None, f"{run_dir} is not a directory"
    # The pose record is NOT tested for existence here: it is created at the first sample, which
    # can be after the clock record, and a watcher that started early would otherwise decide for
    # the whole run that poses were never recorded. Tail tolerates a path that is not there yet, so
    # whether poses arrived is answered at the end, from what actually turned up.
    #
    # Taken from beside the clock record rather than from *run_dir*: the two records are written
    # together, and ``find_clock_record`` may have found the run one level down. Deriving it from
    # the argument instead would read the clock of one run and the poses of nothing.
    clock = find_clock_record(run_dir)
    return clock, (clock.parent if clock else run_dir) / SIM_POSE_FILENAME, None


def read_roster(directory: Path) -> tuple[dict, str | None]:
    """``{frame: (entity name, kind, present)}`` from the recorder's roster, or why there is none.

    The roster is what turns check 1 from a per-world configuration job into something that runs
    unattended: the pose record names root bodies, and only the entity registry knows which of them
    is a robot. See ``capture.ENTITIES_FILENAME``.

    Keyed by the *body* name, because that is what the pose record's ``frame`` column holds; an
    entity with no body of its own is keyed by its own name, which is what a driver that registers
    one without a body would then report. Absence is a reason and never an exception: a run that
    predates the roster, or one whose recorder could not write it, is a run this still checks what
    it can.
    """
    path = directory / ENTITIES_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, f"no {ENTITIES_FILENAME} beside the pose record"
    except (OSError, ValueError) as err:
        return {}, f"{path} could not be read ({err})"
    out = {}
    for entry in document.get("entities") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        frame = entry.get("body") or entry["name"]
        out[str(frame)] = (str(entry["name"]), str(entry.get("kind") or ""),
                           bool(entry.get("present", True)))
    if not out:
        return {}, f"{path} names no entities"
    return out, None


def robots_in(roster: dict) -> list[str]:
    """The frames check 1 should watch: the robots that are currently there.

    An **absent** entity is excluded deliberately. Its body stays in the compiled model, so the
    recorder still writes a row for it every sample -- and a robot the trial has not brought in yet
    is standing still entirely correctly. Watching it would make check 1 fire on a world doing
    exactly what it was told to.
    """
    return sorted(frame for frame, (_name, kind, present) in roster.items()
                  if kind == "robot" and present)


def _await_clock(run_dir: Path, deadline: float, poll: float) -> Path | None:
    """Wait for a clock record to appear, for a run started at about the same time as this check."""
    while True:
        found = find_clock_record(run_dir)
        if found is not None or time.time() >= deadline:
            return found
        time.sleep(poll)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roqsim health", description=__doc__.split("\n")[0])
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=".",
        help="a run's output directory (holds the clock record and sim_poses.csv)",
    )
    parser.add_argument("--clock", metavar="PATH", help="the clock record, if not under run_dir")
    parser.add_argument("--poses", metavar="PATH", help="sim_poses.csv, if not under run_dir")
    parser.add_argument(
        "--robot",
        action="extend",
        nargs="+",
        default=[],
        metavar="FRAME",
        help="a body name from sim_poses.csv to watch for motion; check 1 is skipped without one",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="follow a live run until something is wrong; without it, judge the record as it stands",
    )
    parser.add_argument(
        "--for",
        dest="duration",
        type=float,
        default=0.0,
        metavar="S",
        help="with --watch, stop after this many seconds (default: until a finding)",
    )
    parser.add_argument(
        "--poll", type=float, default=POLL_S, metavar="S", help="seconds between polls with --watch"
    )
    parser.add_argument("--json", action="store_true", help="report as JSON rather than as text")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    from . import logging_setup

    logging_setup.configure(verbose=args.verbose)

    origin = time.time()
    clock_path, poses_path, error = resolve_paths(args)
    if error is None and clock_path is None and args.watch:
        # A checker started alongside its run legitimately arrives first; give the record the same
        # grace the start check itself allows before concluding anything.
        clock_path = _await_clock(Path(args.run_dir), origin + START_TIMEOUT_S, args.poll)
    if error is None and clock_path is None:
        # What was observed, not why. The clock record is absent when a run is not recording
        # (ROQSIM_RECORD unset), when it has not taken its first sample yet, and also when
        # recording is working fine but the record itself could not be written -- capture.py makes
        # that path best-effort on purpose ("a recording that cannot write its clock record is
        # still a recording"). Naming one of the three as the cause would be a guess.
        error = (
            f"no readable clock record (*{CLOCK_MAP_SUFFIX}) under {args.run_dir}. These checks "
            "read what the recorder streams, so there is nothing to check here"
        )
    if error:
        print(f"roqsim health: {error}", file=sys.stderr)
        return EXIT_BAD_ARGS

    source = FileSource(clock_path, poses_path)
    # The roster answers "which of these bodies is a robot" so nobody has to pass the names per
    # world -- which is what makes check 1 safe to run unattended, from a command that is the same
    # for every campaign. ``--robot`` stays, as an override for a run with no roster and for
    # watching something the registry does not call a robot.
    roster, roster_error = read_roster(poses_path.parent if poses_path else Path(args.run_dir))
    robots, no_robots = args.robot, ""
    if not robots:
        robots = robots_in(roster)
        if not robots:
            no_robots = (roster_error or f"{ENTITIES_FILENAME} lists no robot that is present") \
                + ", and no --robot was given"
    monitor = Monitor(source, robots, origin, no_robots=no_robots)
    recording = recording_of(clock_path)
    deadline = origin + args.duration if (args.watch and args.duration > 0) else None
    try:
        while True:
            # Live, "now" is the wall clock, so a record that stops arriving counts against the run.
            # One-shot, it is the newest row: the file cannot say what happened after it, and
            # assuming the worst would report every finished run as a stall.
            monitor.ingest()
            found = monitor.evaluate(time.time() if args.watch else monitor.latest_wall)
            if not args.json:
                for finding in found:
                    print(finding.line(), flush=True)
            if not args.watch or any(f.level == ERROR for f in found):
                break
            if recording.exists():
                monitor.report.notes.append(f"run ended: {recording.name} was written")
                # A last pass over the completed record, on its own terms rather than against a
                # clock that has kept running since it closed.
                monitor.ingest()
                monitor.evaluate(monitor.latest_wall)
                break
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        monitor.report.notes.append("interrupted")
    finally:
        source.close()

    report = monitor.report
    report.state = monitor.state()
    # ``kind`` comes from the roster and from nowhere else. The pose record cannot carry it -- a
    # body name does not say whether it is a robot, a prop or a walker -- and it is what lets a
    # reader of this document tell "the robot has not moved" from "the furniture has not moved"
    # without knowing the world.
    for entity in report.state.get("entities") or []:
        known = roster.get(entity.get("name"))
        if known:
            entity["kind"] = known[1]
    report.notes.append(f"clock record: {clock_path}")
    if monitor.pose_rows:
        report.notes.append(f"pose record: {poses_path}")
    elif robots:
        # Answered from what arrived rather than from a path test at startup, because the pose
        # record is created at the first sample and a watcher can be running before then.
        report.skipped.append(
            f"check 1 (robot-motion): no rows in {SIM_POSE_FILENAME}, so this run did not record "
            "poses (ROQSIM_SIM_POSES)."
        )
    short = monitor.motion.inconclusive()
    if short:
        # A skip only once the record is CLOSED and the window never arrived. While it is still
        # being written the check is armed and merely early, which is a note: a reader -- or an
        # agent -- handed it as a skip treats an ordinary young run as one whose motion nobody is
        # checking, and goes looking for the reason.
        if recording.exists():
            report.skipped.append(short + ", so no verdict was possible")
        else:
            report.notes.append(short + "; still accumulating")
    for label, tail in source.tails():
        if tail.restarts:
            # Worth reporting: the writer re-created the file, so the series the checks measured is
            # not the whole run.
            report.notes.append(f"{label} record was re-created {tail.restarts}x during the watch")
        if tail.malformed:
            report.notes.append(f"{label} record had {tail.malformed} unreadable rows")
    if args.json:
        print(report.as_json())
    else:
        for note in report.skipped:
            print(f"skip  {note}")
        if report.state:
            # One line, because the text mode is for a person watching and the detail belongs in
            # --json. Enough to see a wedged clock without asking a second question.
            rate = report.state.get("rate")
            print(f"state sim {report.state.get('sim_ts', '?')}s"
                  + (f" at {rate}x" if rate is not None else "")
                  + f", {len(report.state.get('entities', []))} bodies")
        if not report.findings:
            print("ok    nothing wrong observed")
    return EXIT_FINDING if report.failed else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
