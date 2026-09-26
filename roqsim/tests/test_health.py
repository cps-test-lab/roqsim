# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""`roqsim health` reads the recorder's own file; these pin what it assumes about it.

The checks are driven from synthetic recordings rather than from a simulation: they are pure
functions of two message streams, and a test that had to run a world would be slow, flaky, and
unable to produce the cases that matter (a wedged sim, a truncated record, a reset mid-window).

Two tests here are of a different kind, and are the reason this file matters more than its size
suggests. `roqsim health` reads a file another process is appending to, and deliberately makes no
change to the simulation runtime to guarantee that it can -- so the guarantees it depends on live in
`capture.py` and are checked from the outside, here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from synthetic_recording import RowWriter

from roqsim import health
from roqsim.mcap_format import CHANNEL_POSES, MAGIC


def levels(findings, slug):
    return [f.level for f in findings if f.check == slug]


def write_run(
    tmp_path: Path,
    clock_rows: list[tuple[float, float]],
    pose_rows: list[tuple[float, str, float]] | None = None,
    *,
    roster: list[dict] | None = None,
    finish: bool = False,
    chunk_every: int = 25,
    name: str = "run.mcap",
) -> Path:
    """A recording of ``(wall, sim)`` clock rows and ``(sim, frame, x)`` pose rows.

    Pose rows with one sim stamp become one ``poses`` message, as the recorder writes them. A chunk
    is closed every ``chunk_every`` samples -- the recorder's once-a-second close, at a cadence a
    test can count. ``finish`` writes the summary, which is what a run that ended on purpose has.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    writer = RowWriter(tmp_path / name, roster=roster)
    poses: dict[float, dict] = {}
    for sim, frame, x in pose_rows or []:
        poses.setdefault(sim, {})[frame] = (x, 0.0, 0.0)
    pose_stamps = sorted(poses)
    pose_index = 0
    for index, (wall, sim) in enumerate(clock_rows):
        writer.clock(wall, sim)
        # Poses at or before this clock stamp ride in the same sample; the two channels are written
        # together by the recorder, so a test's pose rows land beside their clock row.
        while pose_index < len(pose_stamps) and (
            pose_stamps[pose_index] <= sim or index == len(clock_rows) - 1
        ):
            stamp = pose_stamps[pose_index]
            writer.poses(stamp, poses[stamp], wall=wall)
            pose_index += 1
        if (index + 1) % chunk_every == 0:
            writer.chunk()
    while pose_index < len(pose_stamps):
        stamp = pose_stamps[pose_index]
        writer.poses(stamp, poses[stamp])
        pose_index += 1
    return writer.finish() if finish else writer.abandon()


def clock_rows(now: float, seconds: int, *, rate: float = 1.0) -> list[tuple[float, float]]:
    """``seconds`` of clock rows ending at ``now``, sim advancing ``rate`` per wall second."""
    return [(now - seconds + t, t * rate) for t in range(seconds)]


# -- what this tool assumes about capture.py -------------------------------------------------------


def test_the_recording_is_readable_while_the_recorder_is_still_open(tmp_path, monkeypatch):
    """The whole design rests on the file being tailable mid-run, which is a property of the
    writer closing a chunk at least once per wall second -- documented as a guarantee in
    `mcap_format.CHUNK_SECONDS`, and asserted from outside here because this tool deliberately
    changes nothing in the simulation runtime.
    """
    mujoco = pytest.importorskip("mujoco")
    from roqsim import capture
    from roqsim.capture import StateRecorder, snap_fps

    monkeypatch.setattr(capture, "CHUNK_SECONDS", 0.0)  # every sample closes its chunk
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='thing'><freejoint/>"
        "<geom size='0.1'/></body></worldbody></mujoco>"
    )

    class Ctx:
        pass

    ctx = Ctx()
    ctx.model = model
    ctx.data = mujoco.MjData(model)
    ctx.sim_time = 0.0
    ctx.seed = 0

    rate = snap_fps(1 / model.opt.timestep, model.opt.timestep)  # every step
    recorder = StateRecorder(ctx, tmp_path / "run.mcap", rate)
    for _ in range(5):
        mujoco.mj_step(model, ctx.data)
        ctx.sim_time = float(ctx.data.time)
        recorder.sample(ctx)

    # The recorder is still open and close() has NOT been called -- that is the point.
    tail = health.Tail(tmp_path / "run.mcap")
    clock, poses = tail.rows()
    assert len(clock) == 5, "closed chunks are on disk before the run ends"
    assert {p.frame for p in poses} == {"thing"}
    assert not tail.finished and not health.finished(tmp_path / "run.mcap"), (
        "the summary is written at close(); its absence is why a watcher keeps watching"
    )
    recorder.close()
    assert health.finished(tmp_path / "run.mcap")


# -- the tailer ------------------------------------------------------------------------------------


def test_tail_holds_back_a_partial_record(tmp_path):
    """A chunk the writer is halfway through must not be parsed as a whole one."""
    path = write_run(tmp_path, [(100.0, 1.0), (100.5, 1.5)], chunk_every=1)
    whole = path.read_bytes()
    tail = health.Tail(path)
    clock, _ = tail.rows()
    assert [r.sim_ts for r in clock] == [1.0, 1.5]

    # Append the front of the next chunk record, as a writer mid-flush would leave it.
    more = write_run(tmp_path, [(101.0, 2.0)], chunk_every=1, name="more.mcap").read_bytes()
    extra = more[
        len(MAGIC) :
    ]  # the records after the header -- not a valid continuation, but a torn one
    path.write_bytes(whole + extra[: len(extra) // 2])
    assert tail.rows() == ([], []), "a torn record is not a record"


def test_tail_recovers_when_the_file_is_recreated(tmp_path):
    """A run restarted at the same path writes a new file. A tailer that kept its offset would
    then read nothing for the rest of the run and a check would call that silence."""
    path = write_run(tmp_path, [(100.0, 1.0), (101.0, 2.0)], chunk_every=1)
    tail = health.Tail(path)
    assert len(tail.rows()[0]) == 2
    write_run(tmp_path, [(200.0, 0.5)], chunk_every=1)  # shorter: a new file at the same path
    clock, _ = tail.rows()
    assert [r.sim_ts for r in clock] == [0.5], "must re-read from the start, not sit at EOF"
    assert tail.restarts == 1


def test_tail_skips_a_message_it_cannot_trust(tmp_path):
    writer = RowWriter(tmp_path / "run.mcap")
    writer.clock(100.0, 1.0)
    writer.raw(CHANNEL_POSES, b"{not json", sim=1.0)
    writer.poses(2.0, {"base": (0.0, 0.0, 0.0)})
    writer.chunk()
    writer.abandon()
    tail = health.Tail(tmp_path / "run.mcap")
    clock, poses = tail.rows()
    assert [r.sim_ts for r in clock] == [1.0]
    assert [p.sim_ts for p in poses] == [2.0]
    assert tail.malformed == 1


def test_tail_waits_for_a_file_that_is_not_there_yet(tmp_path):
    tail = health.Tail(tmp_path / "run.mcap")
    assert tail.rows() == ([], [])


# -- the tailer enters a long record at its window ------------------------------------------------
#
# A supervisor runs `roqsim health` as a NEW PROCESS on every poll, inside the simulator's own
# container and memory budget. A reader that decompressed the whole record on every poll would have
# a transient footprint that grew for as long as the run did -- until, several minutes into a run
# under a calibrated limit, the poll's own allocation was what put the simulator over it. The checks
# judge the newest minute, so that is all a reader may read.


def test_a_windowed_tail_reads_the_window_and_skips_the_rest(tmp_path):
    path = write_run(tmp_path, clock_rows(11000.0, 10000), chunk_every=10)
    tail = health.Tail(path, clock_window=70.0, pose_window=70.0)
    clock, _ = tail.rows()
    stamps = [r.wall_ts for r in clock]
    assert stamps[-1] == 10999.0, "the newest row is read"
    assert stamps[0] <= 10999.0 - 70.0, "the whole window is covered"
    assert stamps[0] > 10999.0 - 70.0 - 20, "and at most a chunk more"
    assert tail.skipped > 0.9 * path.stat().st_size, "the rest of the file was never read"


def test_a_windowed_tail_still_follows_what_arrives(tmp_path):
    path = write_run(tmp_path, clock_rows(2000.0, 1000), chunk_every=10)
    tail = health.Tail(path, clock_window=70.0, pose_window=70.0)
    assert tail.rows()[0]
    whole = path.read_bytes()
    more = write_run(
        tmp_path, [(2000.0, 1000.0), (2001.0, 1001.0)], chunk_every=2, name="more.mcap"
    )
    path.write_bytes(whole + more.read_bytes()[len(MAGIC) :])
    assert [r.sim_ts for r in tail.rows()[0]] == [1000.0, 1001.0]


def test_a_windowed_tail_reads_a_short_record_whole(tmp_path):
    path = write_run(tmp_path, clock_rows(1030.0, 30), chunk_every=10)
    tail = health.Tail(path, clock_window=70.0, pose_window=70.0)
    assert len(tail.rows()[0]) == 30
    assert tail.skipped == 0


def test_a_windowed_tail_enters_a_reset_record_no_earlier_than_the_reset(tmp_path):
    """The channels' sim time goes back to zero at a world reset. Scanning backwards from a young
    series, the chunks before the reset are OLDER in the file but LATER in sim time, so the window's
    arithmetic would never find a chunk old enough and the scan would read the whole run. The
    checks re-anchor at a reset, so nothing before it changes a verdict: stop there."""
    before = [(1000.0 + t, float(t)) for t in range(5000)]  # long; all before the reset
    after = [(6000.0 + t, float(t)) for t in range(20)]  # young: shorter than any window
    path = write_run(tmp_path, before + after, chunk_every=10)
    tail = health.Tail(path, clock_window=70.0, pose_window=70.0)
    clock, _ = tail.rows()
    assert [r.sim_ts for r in clock][-20:] == [float(t) for t in range(20)]
    assert len(clock) == 20, "nothing from before the reset"
    assert tail.skipped > 0


def test_a_one_shot_check_on_a_long_run_costs_the_window_and_not_the_run(tmp_path, capsys):
    """The property the tailer's window exists for, asserted end to end through the CLI.

    A run an hour long at 25 Hz, one body: tens of megabytes of messages, of which the checks need
    the last minute. Peak allocation while judging it must not scale with the record -- the bound
    here is loose against the window and far below what decompressing the whole record costs.
    """
    import tracemalloc

    now = time.time()
    hz, seconds = 25, 3600
    writer = RowWriter(tmp_path / "run.mcap")
    for i in range(seconds * hz):
        sim = i / hz
        writer.clock(now - seconds + sim, sim)
        writer.poses(sim, {"base": (sim * 0.5, 0.0, 0.0)}, wall=now - seconds + sim)
        if i % hz == hz - 1:
            writer.chunk()
    writer.abandon()
    uncompressed = writer.uncompressed
    assert uncompressed > 8 * 1024 * 1024, (
        "the record has to be large for the bound to mean anything"
    )
    tracemalloc.start()
    try:
        code = health.main([str(tmp_path), "--robot", "base", "--json"])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert code == health.EXIT_OK
    assert peak < uncompressed / 4, f"peak {peak} bytes against {uncompressed} bytes of messages"
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"]["sim_ts"] == pytest.approx(seconds - 1 / hz)
    assert any("were not read" in note for note in payload["notes"]), "and the report says so"


def test_a_long_run_whose_robot_stopped_is_still_caught_from_the_window(tmp_path, capsys):
    """A robot that moved for an hour and has stood still for the last two minutes: the finding
    comes from the newest minute alone, which is exactly what the window keeps."""
    now = time.time()
    moving = [(float(t), "base", t * 0.5) for t in range(0, 3600)]
    parked = [(float(t), "base", 1800.0) for t in range(3600, 3720)]
    write_run(tmp_path, clock_rows(now, 3720), moving + parked)
    assert health.main([str(tmp_path), "--robot", "base"]) == health.EXIT_OK
    out = capsys.readouterr().out
    assert "robot-motion" in out and "moved under 1 cm" in out


def test_a_long_run_whose_clock_wedged_is_still_caught_from_the_window(tmp_path, capsys):
    """An hour at realtime, then sim time flat for two minutes while wall time went on."""
    now = time.time()
    running = [(now - 3720 + t, float(t)) for t in range(0, 3600)]
    wedged = [(now - 3720 + t, 3600.0) for t in range(3600, 3720)]
    write_run(tmp_path, running + wedged)
    assert health.main([str(tmp_path)]) == health.EXIT_FINDING
    assert "sim-time-rate" in capsys.readouterr().out


# -- check 2: sim time starts ------------------------------------------------------------------


def test_start_check_passes_once_sim_time_advances():
    check = health.SimTimeStarts(timeout=60.0)
    check.update([health.ClockRow(100.0, 0.0), health.ClockRow(100.1, 0.02)])
    assert check.findings(now=1000.0, origin=100.0) == []


def test_start_check_fires_when_sim_time_stays_flat():
    """Rows arrive, so the recorder is alive -- but sim time never moves off its first value."""
    check = health.SimTimeStarts(timeout=60.0)
    check.update([health.ClockRow(100.0, 0.0), health.ClockRow(150.0, 0.0)])
    assert check.findings(now=100.0 + 59.0, origin=100.0) == [], "silent inside the window"
    findings = check.findings(now=100.0 + 61.0, origin=100.0)
    assert [f.level for f in findings] == [health.ERROR]
    assert "sim time flat" in findings[0].detail


def test_start_check_is_measured_from_the_first_record_not_from_our_start():
    """Pointing the checker at a run already in progress must not report a fault from before it
    was looking -- and a one-shot pass over a finished file must judge the span the file covers."""
    check = health.SimTimeStarts(timeout=60.0)
    check.update([health.ClockRow(1000.0, 0.0)])
    # Our process started long ago; the record is recent, so the record wins.
    assert check.findings(now=1030.0, origin=1.0) == []


def test_start_check_reports_once():
    check = health.SimTimeStarts(timeout=10.0)
    check.update([health.ClockRow(100.0, 0.0)])
    first = check.findings(now=200.0, origin=100.0)
    assert [f.level for f in first] == [health.ERROR]
    assert check.findings(now=300.0, origin=100.0) == [], "an agent told twice learns to ignore it"


def test_start_check_is_silent_when_no_record_has_appeared_yet():
    """Nothing observed is not the same as something wrong -- until the timeout."""
    check = health.SimTimeStarts(timeout=60.0)
    assert check.findings(now=100.0 + 30.0, origin=100.0) == []


# -- check 3: sim time rate --------------------------------------------------------------------


def test_rate_check_passes_a_realtime_run():
    check = health.SimTimeRate(min_advance=5.0, window=60.0)
    check.update([health.ClockRow(100.0 + t, float(t)) for t in range(0, 121)])
    assert check.findings(now=220.0, origin=100.0) == []


def test_rate_check_waits_for_a_full_window():
    check = health.SimTimeRate(min_advance=5.0, window=60.0)
    check.update([health.ClockRow(100.0, 0.0), health.ClockRow(110.0, 0.01)])
    assert check.findings(now=130.0, origin=100.0) == [], "never fail a run for its first minute"


def test_rate_check_fires_on_a_wedged_run():
    """Rows stop arriving; the window keeps sliding. Silence is what makes the rate fall."""
    check = health.SimTimeRate(min_advance=5.0, window=60.0)
    check.update([health.ClockRow(100.0 + t, t * 0.001) for t in range(0, 10)])
    findings = check.findings(now=100.0 + 90.0, origin=100.0)
    assert [f.level for f in findings] == [health.ERROR]
    assert "last message" in findings[0].detail


def test_rate_check_is_not_fooled_by_a_reset(tmp_path):
    """A reset restarts sim time in the same file. Differencing across it reads a healthy minute
    as zero advance, which would fail every campaign that repeats a configuration."""
    check = health.SimTimeRate(min_advance=5.0, window=60.0)
    splitter = health.SeriesSplitter()
    rows = [health.ClockRow(100.0 + t, float(t)) for t in range(0, 60)]
    rows += [health.ClockRow(160.0 + t, float(t)) for t in range(0, 60)]  # reset: sim time restarts
    for index, series in enumerate(splitter.split(rows)):
        if index:
            check.on_new_series()
        check.update(series)
    assert check.findings(now=220.0, origin=100.0) == []


# -- check 1: robot motion -----------------------------------------------------------------------


def test_motion_check_passes_a_moving_robot():
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(float(t), "base", (t * 0.5, 0.0, 0.0)) for t in range(0, 120)])
    assert check.findings(now=0.0, origin=0.0) == []


def test_motion_check_warns_on_a_parked_robot():
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(float(t), "base", (0.0, 0.0, 0.0)) for t in range(0, 120)])
    findings = check.findings(now=0.0, origin=0.0)
    assert [f.level for f in findings] == [health.WARN], "standing still is often correct"
    assert "base" in findings[0].detail


def test_motion_check_ignores_a_frame_nobody_asked_about():
    """A parked crate is not a stalled robot -- the channel holds every recorded body, not just robots."""
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    rows = []
    for t in range(0, 120):
        rows.append(health.PoseRow(float(t), "base", (t * 0.5, 0.0, 0.0)))
        rows.append(health.PoseRow(float(t), "a_crate", (0.0, 0.0, 0.0)))
    check.update(rows)
    assert check.findings(now=0.0, origin=0.0) == []


def test_motion_check_reports_a_robot_that_never_appears():
    """A --robot that matches nothing must not read as a pass: check 1 would be reporting nothing
    wrong about a robot it never once looked at. `roqsim state` draws the same line."""
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(float(t), "a_crate", (0.0, 0.0, 0.0)) for t in range(0, 120)])
    findings = check.findings(now=0.0, origin=0.0)
    assert [f.level for f in findings] == [health.WARN]
    assert "never appears" in findings[0].detail
    assert "a_crate" in findings[0].detail, "say what the channel does offer"


def test_motion_check_waits_for_a_second_sample_before_calling_a_robot_absent():
    """A frame not written yet is not an absent one -- but two samples settle it.

    Every sample writes every recorded body, so the roster is known after two of them. Waiting a
    full motion window instead would leave a mistyped --robot unreported on any run shorter than a
    simulated minute, which is exactly the quick run a name gets mistyped in.
    """
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(0.0, "a_crate", (0.0, 0.0, 0.0))])
    assert check.findings(now=0.0, origin=0.0) == [], "one sample is not a roster"
    check.update([health.PoseRow(1.0, "a_crate", (0.0, 0.0, 0.0))])
    assert [f.level for f in check.findings(now=0.0, origin=0.0)] == [health.WARN]


def test_a_run_shorter_than_the_window_says_check_1_was_inconclusive():
    """Reporting nothing wrong would overstate what was checked.

    The message states the FACT and not what it means: whether a short record is a skip or merely
    an early one is the caller's to decide -- see the two CLI tests below -- and the same sentence
    has to serve both."""
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(float(t), "base", (0.0, 0.0, 0.0)) for t in range(0, 20)])
    assert check.findings(now=0.0, origin=0.0) == [], "20 s cannot prove a 60 s stall"
    assert "19 s of sim time, less than the 60 s" in check.inconclusive()
    assert "no verdict" not in check.inconclusive(), (
        "that conclusion belongs to a closed record, and this method cannot tell"
    )


def _short_run(tmp_path, *, finish: bool):
    """A run whose record is well under the motion window, with the robot named."""
    now = time.time()
    return write_run(
        tmp_path,
        clock_rows(now, 20),
        [(float(t), "base", 0.0) for t in range(0, 20)],
        finish=finish,
    )


def test_a_short_record_that_is_still_growing_is_a_note_and_not_a_skip(tmp_path, capsys):
    """An early run has not skipped anything -- it will have a verdict shortly. Reported as a
    *skip* it says nobody is checking the robot's motion, which sends a reader (or an agent
    reading this document) looking for the reason four seconds into a healthy run."""
    _short_run(tmp_path, finish=False)
    assert health.main([str(tmp_path), "--robot", "base", "--json"]) == health.EXIT_OK
    report = json.loads(capsys.readouterr().out)

    assert report["skipped"] == [], "an early run has skipped nothing"
    assert any("less than the 60 s" in note for note in report["notes"])
    assert any("still accumulating" in note for note in report["notes"])


def test_a_short_record_that_has_CLOSED_is_a_skip(tmp_path, capsys):
    """Once the recorder has written its summary the record is final, so a window that never
    arrived never will: check 1 is genuinely skipped and saying so is the honest answer."""
    _short_run(tmp_path, finish=True)  # the one unambiguous end-of-run marker (see `finished`)

    assert health.main([str(tmp_path), "--robot", "base", "--json"]) == health.EXIT_OK
    report = json.loads(capsys.readouterr().out)

    assert any("no verdict was possible" in skip for skip in report["skipped"])


def test_a_long_enough_run_reaches_a_verdict():
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(float(t), "base", (t * 0.5, 0.0, 0.0)) for t in range(0, 120)])
    assert check.inconclusive() is None


def test_motion_check_re_anchors_across_a_reset():
    """A reset teleports the robot home. That jump is not motion, and the pose it is restored to is
    not a stall -- both are artefacts of the boundary."""
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    splitter = health.SeriesSplitter()
    rows = [health.PoseRow(float(t), "base", (5.0, 0.0, 0.0)) for t in range(0, 40)]
    rows += [health.PoseRow(float(t), "base", (0.0, 0.0, 0.0)) for t in range(0, 40)]
    for index, series in enumerate(splitter.split(rows)):
        if index:
            check.on_new_series()
        check.update(series)
    assert check.findings(now=0.0, origin=0.0) == [], "neither half is a stalled minute"


# -- the splitter --------------------------------------------------------------------------------


def test_splitter_finds_a_boundary_between_polls():
    """A reset usually falls between two polls, not inside one batch."""
    splitter = health.SeriesSplitter()
    assert len(splitter.split([health.ClockRow(100.0, 10.0)])) == 1
    assert len(splitter.split([health.ClockRow(101.0, 0.1)])) == 2


def test_splitter_leaves_a_monotonic_stream_alone():
    splitter = health.SeriesSplitter()
    rows = [health.ClockRow(100.0 + t, float(t)) for t in range(5)]
    assert len(splitter.split(rows)) == 1


# -- end to end ------------------------------------------------------------------------------------


def test_cli_reports_a_healthy_run(tmp_path, capsys):
    now = time.time()
    write_run(tmp_path, clock_rows(now, 120), [(float(t), "base", t * 0.5) for t in range(0, 120)])
    assert health.main([str(tmp_path), "--robot", "base"]) == health.EXIT_OK
    assert "nothing wrong observed" in capsys.readouterr().out


def test_cli_exits_5_on_a_wedged_run(tmp_path, capsys):
    """--watch is the mode that expects the run to continue, so silence counts against it."""
    now = time.time()
    write_run(tmp_path, [(now - 300 + t, t * 0.001) for t in range(0, 10)])
    assert health.main([str(tmp_path), "--watch"]) == health.EXIT_FINDING
    assert "sim-time-rate" in capsys.readouterr().out


def test_cli_one_shot_does_not_call_a_finished_run_stalled(tmp_path, capsys):
    """The same record, judged without the premise that the run is still going.

    A finished run's rows are all in the past, and they stop. Counting that gap against the run
    would report every completed campaign job as a stall a minute after it ended -- while saying
    nothing the file supports, since what happened after the last row is not in it.
    """
    now = time.time()
    write_run(tmp_path, [(now - 300 + t, float(t)) for t in range(0, 120)])
    assert health.main([str(tmp_path)]) == health.EXIT_OK
    assert "nothing wrong observed" in capsys.readouterr().out


def test_cli_still_fails_a_run_that_was_slow_while_it_ran(tmp_path, capsys):
    """One-shot judges the recorded span -- so a run that crawled is caught from the record alone."""
    now = time.time()
    write_run(tmp_path, [(now - 300 + t, t * 0.01) for t in range(0, 200)])
    assert health.main([str(tmp_path)]) == health.EXIT_FINDING
    assert "realtime" in capsys.readouterr().out


def test_watch_stops_without_complaint_when_the_recording_closes(tmp_path, capsys):
    """The summary is written by close(), so its presence means the run ended rather than stopped."""
    now = time.time()
    write_run(
        tmp_path, [(now - 30 + t * 0.25, float(t) * 0.25) for t in range(0, 120)], finish=True
    )
    assert health.main([str(tmp_path), "--watch", "--poll", "0.01"]) == health.EXIT_OK
    assert "nothing wrong observed" in capsys.readouterr().out


def test_cli_exits_2_without_a_recording(tmp_path, capsys):
    assert health.main([str(tmp_path)]) == health.EXIT_BAD_ARGS
    err = capsys.readouterr().err
    assert "no recording" in err
    # It must not name a cause it did not observe: the file appears at the first sample, so its
    # absence does not prove recording was off.
    assert "is not recording" not in err


def test_cli_takes_a_recording_path(tmp_path, capsys):
    now = time.time()
    path = write_run(tmp_path / "elsewhere", clock_rows(now, 120), name="take.mcap")
    assert health.main(["--recording", str(path)]) == health.EXIT_OK
    assert health.main(["--recording", str(tmp_path / "absent.mcap")]) == health.EXIT_BAD_ARGS


def test_cli_skips_check_1_when_no_robot_is_named(tmp_path, capsys):
    now = time.time()
    write_run(tmp_path, clock_rows(now, 120))
    assert health.main([str(tmp_path)]) == health.EXIT_OK
    out = capsys.readouterr().out
    assert "check 1" in out and "skip" in out, "a skipped check must say so, never pass quietly"


def test_cli_json_carries_the_findings(tmp_path, capsys):
    now = time.time()
    write_run(tmp_path, [(now - 300 + t, t * 0.001) for t in range(0, 10)])
    assert health.main([str(tmp_path), "--watch", "--json"]) == health.EXIT_FINDING
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit"] == health.EXIT_FINDING
    assert payload["findings"][0]["check"] == "sim-time-rate"
    assert set(payload) == {"findings", "skipped", "notes", "state", "exit"}


# -- the state block: where everything is, alongside what is wrong ---------------------------------


def test_json_reports_the_last_pose_and_clock(tmp_path, capsys):
    """A caller asking "is anything wrong" almost always wants "and where is it" next, and both
    come from channels already open -- so one read answers both rather than two."""
    now = time.time()
    write_run(
        tmp_path,
        clock_rows(now, 120),
        # One message per sample holding every recorded body, as the recorder writes it.
        [
            row
            for t in range(0, 120)
            for row in ((float(t), "base", t * 0.5), (float(t), "crate", 3.0))
        ],
    )
    assert health.main([str(tmp_path), "--robot", "base", "--json"]) == health.EXIT_OK
    state = json.loads(capsys.readouterr().out)["state"]
    assert state["sim_ts"] == 119.0
    assert state["rate"] == 1.0  # one sim second per wall second
    names = [e["name"] for e in state["entities"]]
    assert names == ["base", "crate"]  # every recorded body, not just the watched one
    base = state["entities"][0]
    assert base["position"] == [59.5, 0.0, 0.0]  # the *last* sample, not the first
    assert base["orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert "twist_linear" in base


def test_state_rate_is_measured_within_one_series(tmp_path, capsys):
    """The reset hazard again, this time for the reported rate rather than the verdict: a window
    spanning a reset sees a healthy minute as no progress, and reporting 0.0x would be a lie about
    a fine run."""
    now = time.time()
    rows = [(now - 120 + t, float(t)) for t in range(0, 60)]
    rows += [(now - 60 + t, float(t)) for t in range(0, 60)]  # sim time restarts
    write_run(tmp_path, rows)
    health.main([str(tmp_path), "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert state["sim_ts"] == 59.0  # the new series, not the old one
    assert state["rate"] == 1.0


def test_state_drops_bodies_from_before_a_reset(tmp_path, capsys):
    """A re-posed world must not be described with positions from before it: a stale answer
    presented as a current one is worse than no answer."""
    now = time.time()
    writer = RowWriter(tmp_path / "run.mcap")
    for t in range(0, 30):
        writer.clock(now - 60 + t, float(t))
        writer.poses(float(t), {"gone": (1.0, 0.0, 0.0)}, wall=now - 60 + t)
    for t in range(0, 30):  # sim time restarts: a reset
        writer.clock(now - 30 + t, float(t))
        writer.poses(float(t), {"here": (2.0, 0.0, 0.0)}, wall=now - 30 + t)
    writer.abandon()
    health.main([str(tmp_path), "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert [e["name"] for e in state["entities"]] == ["here"]


def test_state_is_absent_when_there_is_nothing_to_report(tmp_path, capsys):
    """Empty rather than zeros: "no messages" and "everything at the origin" are different answers
    and must not render the same."""
    write_run(tmp_path, [])
    health.main([str(tmp_path), "--json"])
    assert json.loads(capsys.readouterr().out)["state"] == {}


def test_state_names_no_kind(tmp_path, capsys):
    """The poses channel names bodies without saying which are robots. Inventing the
    distinction here would be a guess presented as a fact -- it belongs to whoever holds the
    entity registry."""
    now = time.time()
    write_run(tmp_path, clock_rows(now, 10), [(float(t), "base", 0.0) for t in range(0, 10)])
    health.main([str(tmp_path), "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert "kind" not in state["entities"][0]


def test_a_run_is_found_one_level_below_the_directory_given(tmp_path, capsys):
    """A supervisor can name an output root without knowing which run inside it is current --
    on a packed job it cannot know. Newest-by-mtime is the right answer there."""
    now = time.time()
    old_run, new_run = tmp_path / "cfgA" / "0", tmp_path / "cfgA" / "1"
    write_run(old_run, clock_rows(now - 600, 60), [(float(t), "base", 0.0) for t in range(0, 60)])
    write_run(new_run, clock_rows(now, 60), [(float(t), "base", t * 0.5) for t in range(0, 60)])
    os.utime(new_run / "run.mcap", (now, now))
    os.utime(old_run / "run.mcap", (now - 600, now - 600))

    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    state = json.loads(capsys.readouterr().out)["state"]
    # The newest run, and its own poses -- 29.5 is the last sample of the moving robot.
    assert state["entities"][0]["position"] == [29.5, 0.0, 0.0]


def test_a_re_creation_is_reported_rather_than_hidden(tmp_path, capsys):
    """The gap it leaves is real -- messages were lost -- so the report says so rather than
    presenting a partial series as a whole one. Driven through the CLI, because the note is only
    useful if it survives to the document a caller reads."""
    now = time.time()
    write_run(tmp_path, clock_rows(now, 120))
    real_sleep = time.sleep

    def shrink_between_polls(seconds):
        # A run restarted at the same path: a fresh, shorter file. Done between two of the
        # watcher's reads, which is exactly when a run does it.
        write_run(tmp_path, [(now, 120.0)])
        real_sleep(seconds)

    health.time.sleep = shrink_between_polls
    try:
        assert (
            health.main([str(tmp_path), "--watch", "--poll", "0.01", "--for", "0.05", "--json"])
            == health.EXIT_OK
        )
    finally:
        health.time.sleep = real_sleep
    notes = " ".join(json.loads(capsys.readouterr().out)["notes"])
    assert "re-created" in notes, "a series that is not the whole run must say so"


# -- the roster: which of those bodies is a robot ---------------------------------------------------


def test_check_1_watches_the_robots_the_roster_names(tmp_path, capsys):
    """The point of the roster: the same command works on every world, with no --robot to forget.

    A static prop in the same record must not be watched -- watching every recorded body would fire
    check 1 on the furniture, which is the reason the roster exists rather than a list of names.
    """
    now = time.time()
    write_run(
        tmp_path,
        clock_rows(now, 120),
        [row for t in range(0, 120) for row in ((float(t), "base", 0.0), (float(t), "shelf", 3.0))],
        roster=[
            {"name": "robot", "kind": "robot", "body": "base"},
            {"name": "shelf", "kind": "object", "body": "shelf"},
        ],
    )
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK  # check 1 warns, never errors
    payload = json.loads(capsys.readouterr().out)
    motion = [f for f in payload["findings"] if f["check"] == "robot-motion"]
    assert len(motion) == 1, "the standing robot is a finding; the standing shelf is not"
    assert "'base'" in motion[0]["detail"]
    assert not payload["skipped"], "check 1 ran, so nothing should be reported as skipped"


def test_an_absent_robot_is_not_watched(tmp_path, capsys):
    """Its body is still in the model, so the recorder still writes rows for it -- and a robot the
    trial has not brought in yet is standing still entirely correctly."""
    now = time.time()
    write_run(
        tmp_path,
        clock_rows(now, 120),
        [(float(t), "base", 0.0) for t in range(0, 120)],
        roster=[{"name": "robot", "kind": "robot", "body": "base", "present": False}],
    )
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert not [f for f in payload["findings"] if f["check"] == "robot-motion"]
    assert any("check 1" in s for s in payload["skipped"]), "nothing watched must say so"


def test_robot_overrides_the_roster(tmp_path, capsys):
    """--robot stays useful: a run with no roster, and watching something not called a robot."""
    now = time.time()
    write_run(
        tmp_path,
        clock_rows(now, 120),
        [(float(t), "shelf", 3.0) for t in range(0, 120)],
        roster=[{"name": "shelf", "kind": "object", "body": "shelf"}],
    )
    assert health.main([str(tmp_path), "--robot", "shelf", "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [f for f in payload["findings"] if f["check"] == "robot-motion"]


def test_a_missing_roster_says_so_rather_than_naming_a_flag(tmp_path, capsys):
    """The skip has to be actionable: which of the two reasons applies decides what to do."""
    now = time.time()
    write_run(tmp_path, clock_rows(now, 120), [(float(t), "base", 0.0) for t in range(0, 120)])
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    skipped = " ".join(payload["skipped"])
    assert "roqsim.entities" in skipped and "check 1" in skipped


def test_the_state_block_carries_kind_from_the_roster(tmp_path, capsys):
    """What the poses channel cannot say: whether the thing that has not moved is a robot."""
    now = time.time()
    write_run(
        tmp_path,
        clock_rows(now, 120),
        [
            row
            for t in range(0, 120)
            for row in ((float(t), "base", t * 0.5), (float(t), "shelf", 3.0))
        ],
        roster=[
            {"name": "robot", "kind": "robot", "body": "base"},
            {"name": "shelf", "kind": "object", "body": "shelf"},
        ],
    )
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    kinds = {
        e["name"]: e.get("kind") for e in json.loads(capsys.readouterr().out)["state"]["entities"]
    }
    assert kinds == {"base": "robot", "shelf": "object"}


def test_the_last_roster_wins(tmp_path):
    """The recorder writes the roster again whenever it changes; a reader takes the newest."""
    now = time.time()
    writer = RowWriter(
        tmp_path / "run.mcap", roster=[{"name": "robot", "kind": "robot", "body": "base"}]
    )
    for t in range(0, 10):
        writer.clock(now - 20 + t, float(t))
    writer.chunk()
    writer.roster(
        [
            {"name": "robot", "kind": "robot", "body": "base"},
            {"name": "obstacle", "kind": "object", "body": "box"},
        ]
    )
    for t in range(10, 20):
        writer.clock(now - 20 + t, float(t))
    writer.abandon()
    roster, error = health.read_roster(tmp_path / "run.mcap")
    assert error is None
    assert set(roster) == {"base", "box"}
    tail = health.Tail(tmp_path / "run.mcap")
    tail.rows()
    assert [e["name"] for e in tail.roster] == ["robot", "obstacle"]


def test_a_malformed_roster_is_a_reason_and_not_a_crash(tmp_path, capsys):
    now = time.time()
    writer = RowWriter(tmp_path / "run.mcap")
    writer.writer.add_metadata("roqsim.entities", {"json": "{not json"})
    for wall, sim in clock_rows(now, 120):
        writer.clock(wall, sim)
    writer.abandon()
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert any("check 1" in s for s in payload["skipped"])


def test_the_recorder_writes_the_roster_and_follows_a_change(tmp_path, monkeypatch):
    """Asserted against the writer, since the roster only helps if it is there and stays true.

    A roster written once at the first sample would describe the world the trial started in; a run
    that spawns an obstacle mid-trial is the normal case, not the exotic one.
    """
    mujoco = pytest.importorskip("mujoco")
    from roqsim import capture
    from roqsim.capture import StateRecorder, snap_fps
    from roqsim.context import Entity, EntityRegistry

    monkeypatch.setattr(capture, "CHUNK_SECONDS", 0.0)
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='base'><freejoint/><geom size='0.1'/></body>"
        "<body name='box' pos='2 0 0'><freejoint/><geom size='0.1'/></body></worldbody></mujoco>"
    )

    class Ctx:
        pass

    ctx = Ctx()
    ctx.model = model
    ctx.data = mujoco.MjData(model)
    ctx.sim_time = 0.0
    ctx.seed = 0
    ctx.entities = EntityRegistry()
    ctx.entities.add(Entity(name="robot", kind="robot", body="base"))

    rate = snap_fps(1 / model.opt.timestep, model.opt.timestep)
    recorder = StateRecorder(ctx, tmp_path / "run.mcap", rate)

    def step():
        mujoco.mj_step(model, ctx.data)
        ctx.sim_time = float(ctx.data.time)
        recorder.sample(ctx)

    step()
    roster, _ = health.read_roster(tmp_path / "run.mcap")
    assert roster == {"base": ("robot", "robot", True)}

    ctx.entities.add(Entity(name="obstacle", kind="object", body="box"))
    step()
    roster, _ = health.read_roster(tmp_path / "run.mcap")
    assert set(roster) == {"base", "box"}, "a spawn mid-run must show up in the roster"
    assert health.robots_in(roster) == ["base"]


def test_the_recorder_needs_no_registry(tmp_path, monkeypatch):
    """A driver that keeps no registry still records; the roster is simply absent, which the
    reader reports as a reason. Nothing about the recording depends on it."""
    mujoco = pytest.importorskip("mujoco")
    from roqsim import capture
    from roqsim.capture import StateRecorder, snap_fps

    monkeypatch.setattr(capture, "CHUNK_SECONDS", 0.0)
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='base'><freejoint/>"
        "<geom size='0.1'/></body></worldbody></mujoco>"
    )

    class Ctx:
        pass

    ctx = Ctx()
    ctx.model = model
    ctx.data = mujoco.MjData(model)
    ctx.sim_time = 0.0
    ctx.seed = 0

    recorder = StateRecorder(ctx, tmp_path / "run.mcap", rate=snap_fps(10.0, model.opt.timestep))
    mujoco.mj_step(model, ctx.data)
    ctx.sim_time = float(ctx.data.time)
    recorder.sample(ctx)
    _, poses = health.Tail(tmp_path / "run.mcap").rows()
    assert {p.frame for p in poses} == {"base"}
    roster, error = health.read_roster(tmp_path / "run.mcap")
    assert roster == {} and "roqsim.entities" in error
