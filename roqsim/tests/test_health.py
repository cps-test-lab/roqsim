# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""`roqsim health` reads the recorder's streamed records; these pin what it assumes about them.

The checks are driven from synthetic CSVs rather than from a simulation: they are pure functions of
two row streams, and a test that had to run a world would be slow, flaky, and unable to produce the
cases that matter (a wedged sim, a truncated record, a reset mid-window).

Two tests here are of a different kind, and are the reason this file matters more than its size
suggests. `roqsim health` reads files another process is appending to, and deliberately makes no
change to the simulation runtime to guarantee that it can -- so the guarantees it depends on live in
`capture.py` and are checked from the outside, here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from roqsim import health

CLOCK_HEADER = "wall_ts,sim_ts\n"
POSE_HEADER = ",".join(
    [
        "timestamp",
        "wall_time",
        "frame",
        "position.x",
        "position.y",
        "position.z",
        "orientation.x",
        "orientation.y",
        "orientation.z",
        "orientation.w",
        "twist.linear.x",
        "twist.linear.y",
        "twist.linear.z",
        "twist.angular.x",
        "twist.angular.y",
        "twist.angular.z",
    ]
) + "\n"


def clock_line(wall: float, sim: float) -> str:
    return f"{wall:.6f},{sim:.6f}\n"


def pose_line(sim: float, frame: str, x: float, y: float = 0.0, z: float = 0.0) -> str:
    return (
        f"{sim:.6f},{sim:.6f},{frame},{x:.6f},{y:.6f},{z:.6f},"
        "0.000000,0.000000,0.000000,1.000000,"
        "0.000000,0.000000,0.000000,0.000000,0.000000,0.000000\n"
    )


def levels(findings, slug):
    return [f.level for f in findings if f.check == slug]


# -- what this tool assumes about capture.py -------------------------------------------------------


def test_the_record_names_match_the_writer():
    """health.py duplicates the two filenames so it need not import MuJoCo. Duplication is only
    safe while something notices it drifting, and this is that something."""
    from roqsim import capture

    assert health.CLOCK_MAP_SUFFIX == capture.CLOCK_MAP_SUFFIX
    assert health.SIM_POSE_FILENAME == capture.SIM_POSE_FILENAME
    # The column names the checks index by must be the ones the writer emits, in any order.
    assert set(capture.CLOCK_MAP_FIELDS) >= {"wall_ts", "sim_ts"}
    assert set(capture.SIM_POSE_FIELDS) >= {"timestamp", "frame", "position.x", "position.y",
                                            "position.z"}


def test_the_records_are_readable_while_the_recorder_is_still_open(tmp_path):
    """The whole design rests on these files being tailable mid-run, which is a property of the
    writer's `buffering=1` plus its per-row flush -- documented nowhere as a guarantee.

    Asserted from outside rather than fixed in code, because this tool deliberately changes nothing
    in the simulation runtime. If someone buffers those writes for throughput, this fails here
    instead of silently turning `roqsim health` into a tool that reports nothing.
    """
    mujoco = pytest.importorskip("mujoco")
    from roqsim.capture import StateRecorder, snap_fps

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
    recorder = StateRecorder(ctx, tmp_path / "run.npz", rate, sim_poses=True)
    for _ in range(5):
        mujoco.mj_step(model, ctx.data)
        ctx.sim_time = float(ctx.data.time)
        recorder.sample(ctx)

    # The recorder is still open and close() has NOT been called -- that is the point.
    clock = tmp_path / "run.clock_map.csv"
    poses = tmp_path / "sim_poses.csv"
    assert clock.exists() and poses.exists(), "records must appear before the run ends"
    assert len(clock.read_text().splitlines()) >= 6, "rows must be flushed, not buffered to close()"
    assert "thing" in poses.read_text()
    assert not (tmp_path / "run.npz").exists(), "the npz is written at close(); that is why we tail"
    # The samples stream to disk too, but *buffered* -- so it is not tailable and health must not
    # learn to read it. Its presence without an .npz beside it means a run that never closed.
    assert (tmp_path / "run.npz.part").exists()


# -- the tailer ------------------------------------------------------------------------------------


def test_tail_holds_back_a_partial_row(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(CLOCK_HEADER)
    tail = health.Tail(path)
    assert tail.rows() == []
    with path.open("a") as handle:
        handle.write(clock_line(100.0, 1.0))
        handle.write("100.5,1.5")  # no newline yet
        handle.flush()
    assert [r["sim_ts"] for r in tail.rows()] == ["1.000000"]
    with path.open("a") as handle:
        handle.write("\n")
    assert [r["sim_ts"] for r in tail.rows()] == ["1.5"]


def test_tail_recovers_when_the_writer_recreates_the_file(tmp_path):
    """capture.py drops its handle on OSError and reopens in mode 'w'. A tailer that kept its
    offset would then read nothing for the rest of the run and a check would call that silence."""
    path = tmp_path / "c.csv"
    path.write_text(CLOCK_HEADER + clock_line(100.0, 1.0) + clock_line(101.0, 2.0))
    tail = health.Tail(path)
    assert len(tail.rows()) == 2
    path.write_text(CLOCK_HEADER + clock_line(200.0, 0.5))  # the writer's mode="w" reopen
    rows = tail.rows()
    assert [r["sim_ts"] for r in rows] == ["0.500000"], "must re-read from the start, not sit at EOF"
    assert tail.restarts == 1


def test_tail_skips_a_row_it_cannot_trust(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(CLOCK_HEADER + "100.0\n" + clock_line(101.0, 2.0))
    tail = health.Tail(path)
    assert [r["sim_ts"] for r in tail.rows()] == ["2.000000"]
    assert tail.malformed == 1


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
    assert "last row" in findings[0].detail


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
    """A parked crate is not a stalled robot -- the record holds every root body, not just robots."""
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
    assert "a_crate" in findings[0].detail, "say what the record does offer"


def test_motion_check_waits_for_a_second_sample_before_calling_a_robot_absent():
    """A frame not written yet is not an absent one -- but two samples settle it.

    Every sample writes every root body, so the roster is known after two of them. Waiting a full
    motion window instead would leave a mistyped --robot unreported on any run shorter than a
    simulated minute, which is exactly the quick run a name gets mistyped in.
    """
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(0.0, "a_crate", (0.0, 0.0, 0.0))])
    assert check.findings(now=0.0, origin=0.0) == [], "one sample is not a roster"
    check.update([health.PoseRow(1.0, "a_crate", (0.0, 0.0, 0.0))])
    assert [f.level for f in check.findings(now=0.0, origin=0.0)] == [health.WARN]


def test_a_run_shorter_than_the_window_says_check_1_was_inconclusive():
    """Reporting nothing wrong would overstate what was checked."""
    check = health.RobotMoves(["base"], distance=0.01, window=60.0)
    check.update([health.PoseRow(float(t), "base", (0.0, 0.0, 0.0)) for t in range(0, 20)])
    assert check.findings(now=0.0, origin=0.0) == [], "20 s cannot prove a 60 s stall"
    assert "no verdict was possible" in check.inconclusive()


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


def write_run(tmp_path: Path, clock_rows: list[str], pose_rows: list[str] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run.clock_map.csv").write_text(CLOCK_HEADER + "".join(clock_rows))
    if pose_rows is not None:
        (tmp_path / "sim_poses.csv").write_text(POSE_HEADER + "".join(pose_rows))
    return tmp_path


def test_cli_reports_a_healthy_run(tmp_path, capsys):
    now = time.time()
    write_run(
        tmp_path,
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        [pose_line(float(t), "base", t * 0.5) for t in range(0, 120)],
    )
    assert health.main([str(tmp_path), "--robot", "base"]) == health.EXIT_OK
    assert "nothing wrong observed" in capsys.readouterr().out


def test_cli_exits_5_on_a_wedged_run(tmp_path, capsys):
    """--watch is the mode that expects the run to continue, so silence counts against it."""
    now = time.time()
    write_run(tmp_path, [clock_line(now - 300 + t, t * 0.001) for t in range(0, 10)])
    assert health.main([str(tmp_path), "--watch"]) == health.EXIT_FINDING
    assert "sim-time-rate" in capsys.readouterr().out


def test_cli_one_shot_does_not_call_a_finished_run_stalled(tmp_path, capsys):
    """The same record, judged without the premise that the run is still going.

    A finished run's rows are all in the past, and they stop. Counting that gap against the run
    would report every completed campaign job as a stall a minute after it ended -- while saying
    nothing the file supports, since what happened after the last row is not in it.
    """
    now = time.time()
    write_run(tmp_path, [clock_line(now - 300 + t, float(t)) for t in range(0, 120)])
    assert health.main([str(tmp_path)]) == health.EXIT_OK
    assert "nothing wrong observed" in capsys.readouterr().out


def test_cli_still_fails_a_run_that_was_slow_while_it_ran(tmp_path, capsys):
    """One-shot judges the recorded span -- so a run that crawled is caught from the record alone."""
    now = time.time()
    write_run(tmp_path, [clock_line(now - 300 + t, t * 0.01) for t in range(0, 200)])
    assert health.main([str(tmp_path)]) == health.EXIT_FINDING
    assert "realtime" in capsys.readouterr().out


def test_watch_stops_without_complaint_when_the_recording_closes(tmp_path, capsys):
    """The .npz is written by close(), so its presence means the run ended rather than stopped."""
    now = time.time()
    write_run(tmp_path, [clock_line(now - 30 + t * 0.25, float(t) * 0.25) for t in range(0, 120)])
    (tmp_path / "run.npz").write_bytes(b"not a real archive, only its existence is read")
    assert health.main([str(tmp_path), "--watch", "--poll", "0.01"]) == health.EXIT_OK
    assert "nothing wrong observed" in capsys.readouterr().out


def test_cli_exits_2_without_a_clock_record(tmp_path, capsys):
    assert health.main([str(tmp_path)]) == health.EXIT_BAD_ARGS
    err = capsys.readouterr().err
    assert "no readable clock record" in err
    # It must not name a cause it did not observe: the record is best-effort, so its absence does
    # not prove recording was off.
    assert "is not recording" not in err


def test_cli_skips_check_1_when_no_robot_is_named(tmp_path, capsys):
    now = time.time()
    write_run(tmp_path, [clock_line(now - 120 + t, float(t)) for t in range(0, 120)])
    assert health.main([str(tmp_path)]) == health.EXIT_OK
    out = capsys.readouterr().out
    assert "check 1" in out and "skip" in out, "a skipped check must say so, never pass quietly"


def test_cli_json_carries_the_findings(tmp_path, capsys):
    import json

    now = time.time()
    write_run(tmp_path, [clock_line(now - 300 + t, t * 0.001) for t in range(0, 10)])
    assert health.main([str(tmp_path), "--watch", "--json"]) == health.EXIT_FINDING
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit"] == health.EXIT_FINDING
    assert payload["findings"][0]["check"] == "sim-time-rate"


# -- the state block: where everything is, alongside what is wrong ---------------------------------


def test_json_reports_the_last_pose_and_clock(tmp_path, capsys):
    """A caller asking "is anything wrong" almost always wants "and where is it" next, and both
    come from records already open -- so one read answers both rather than two."""
    now = time.time()
    write_run(
        tmp_path,
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        # Interleaved, as the recorder writes it: one row per root body per sample.
        [line for t in range(0, 120)
         for line in (pose_line(float(t), "base", t * 0.5),
                      pose_line(float(t), "crate", 3.0))],
    )
    assert health.main([str(tmp_path), "--robot", "base", "--json"]) == health.EXIT_OK
    state = json.loads(capsys.readouterr().out)["state"]
    assert state["sim_ts"] == 119.0
    assert state["rate"] == 1.0                      # one sim second per wall second
    names = [e["name"] for e in state["entities"]]
    assert names == ["base", "crate"]                # every root body, not just the watched one
    base = state["entities"][0]
    assert base["position"] == [59.5, 0.0, 0.0]      # the *last* sample, not the first
    assert base["orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert "twist_linear" in base


def test_state_rate_is_measured_within_one_series(tmp_path, capsys):
    """The reset hazard again, this time for the reported rate rather than the verdict: a window
    spanning a reset sees a healthy minute as no progress, and reporting 0.0x would be a lie about
    a fine run."""
    now = time.time()
    rows = [clock_line(now - 120 + t, float(t)) for t in range(0, 60)]
    rows += [clock_line(now - 60 + t, float(t)) for t in range(0, 60)]   # sim time restarts
    write_run(tmp_path, rows)
    health.main([str(tmp_path), "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert state["sim_ts"] == 59.0                   # the new series, not the old one
    assert state["rate"] == 1.0


def test_state_drops_bodies_from_before_a_reset(tmp_path, capsys):
    """A re-posed world must not be described with positions from before it: a stale answer
    presented as a current one is worse than no answer."""
    now = time.time()
    poses = [pose_line(float(t), "gone", 1.0) for t in range(0, 30)]
    poses += [pose_line(float(t), "here", 2.0) for t in range(0, 30)]   # sim time restarts: a reset
    write_run(tmp_path, [clock_line(now - 60 + t, float(t)) for t in range(0, 60)], poses)
    health.main([str(tmp_path), "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert [e["name"] for e in state["entities"]] == ["here"]


def test_state_is_absent_when_there_is_nothing_to_report(tmp_path, capsys):
    """Empty rather than zeros: "no records" and "everything at the origin" are different answers
    and must not render the same."""
    write_run(tmp_path, [])
    health.main([str(tmp_path), "--json"])
    assert json.loads(capsys.readouterr().out)["state"] == {}


def test_state_names_no_kind(tmp_path, capsys):
    """The pose record names root bodies without saying which are robots. Inventing the
    distinction here would be a guess presented as a fact -- it belongs to whoever holds the
    entity registry."""
    now = time.time()
    write_run(tmp_path, [clock_line(now - 10 + t, float(t)) for t in range(0, 10)],
              [pose_line(float(t), "base", 0.0) for t in range(0, 10)])
    health.main([str(tmp_path), "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert "kind" not in state["entities"][0]


def test_a_run_is_found_one_level_below_the_directory_given(tmp_path, capsys):
    """A supervisor can name an output root without knowing which run inside it is current --
    on a packed job it cannot know. Newest-by-mtime is the right answer there, and the pose
    record has to come from beside the clock record rather than from the argument, or the reply
    mixes one run's clock with nothing's poses."""
    now = time.time()
    old_run, new_run = tmp_path / "cfgA" / "0", tmp_path / "cfgA" / "1"
    write_run(old_run, [clock_line(now - 600 + t, float(t)) for t in range(0, 60)],
              [pose_line(float(t), "base", 0.0) for t in range(0, 60)])
    write_run(new_run, [clock_line(now - 60 + t, float(t)) for t in range(0, 60)],
              [pose_line(float(t), "base", t * 0.5) for t in range(0, 60)])
    os.utime(new_run / "run.clock_map.csv", (now, now))
    os.utime(old_run / "run.clock_map.csv", (now - 600, now - 600))

    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    state = json.loads(capsys.readouterr().out)["state"]
    # The newest run, and its own poses -- 29.5 is the last sample of the moving robot.
    assert state["entities"][0]["position"] == [29.5, 0.0, 0.0]



def test_tail_reads_a_re_created_file_from_the_start(tmp_path):
    """The recorder re-creates a record after an `OSError` -- mode "w", fresh header -- so a reader
    following a byte offset would sit past EOF forever and let a check conclude the run had gone
    silent. That is a false ERROR manufactured by a transient disk hiccup, which is why the tailer
    handles it and not the checks.

    Asserted rather than trusted: this is one of two hazards whose handling exists only as a method
    name, and a name is not a guarantee.
    """
    path = tmp_path / "c.csv"
    path.write_text(CLOCK_HEADER + "".join(clock_line(t, float(t)) for t in range(1, 5)))
    tail = health.Tail(path)
    assert len(tail.rows()) == 4

    # Shorter than where the reader stands, with its own header: a new file at the same path. Short
    # is the documented trigger, and it is the real shape of one -- the writer reopens in mode "w"
    # and starts again from a header, so a re-created record is far shorter than a mid-run position.
    path.write_text(CLOCK_HEADER + clock_line(9.0, 9.0))
    rows = tail.rows()

    assert [r["sim_ts"] for r in rows] == ["9.000000"], "a re-created record must be re-read"
    assert tail.restarts == 1, "and the reader must say the series it measured is not the whole run"
    assert "sim_ts" in rows[0], "the fresh header must be consumed as a header, not as a row"


def test_a_re_creation_is_reported_rather_than_hidden(tmp_path, capsys):
    """The gap it leaves is real -- rows were lost -- so the report says so rather than presenting a
    partial series as a whole one. Driven through the CLI, because the note is only useful if it
    survives to the document a caller reads."""
    now = time.time()
    write_run(tmp_path, [clock_line(now - 120 + t, float(t)) for t in range(0, 120)])
    clock = tmp_path / "run.clock_map.csv"
    real_sleep = time.sleep

    def shrink_between_polls(seconds):
        # The writer's own recovery path: reopen in mode "w", fresh header. Done between two of the
        # watcher's reads, which is exactly when a run does it.
        clock.write_text(CLOCK_HEADER + clock_line(now, 120.0))
        real_sleep(seconds)

    health.time.sleep = shrink_between_polls
    try:
        assert health.main([str(tmp_path), "--watch", "--poll", "0.01", "--for", "0.05",
                            "--json"]) == health.EXIT_OK
    finally:
        health.time.sleep = real_sleep
    notes = " ".join(json.loads(capsys.readouterr().out)["notes"])
    assert "re-created" in notes, "a series that is not the whole run must say so"

# -- the roster: which of those bodies is a robot ---------------------------------------------------


def write_roster(run_dir: Path, entities: list[dict]) -> Path:
    path = run_dir / health.ENTITIES_FILENAME
    path.write_text(json.dumps({"entities": entities}) + "\n")
    return path


def test_the_roster_name_matches_the_writer():
    """Duplicated like the other two filenames, and pinned for the same reason."""
    from roqsim import capture

    assert health.ENTITIES_FILENAME == capture.ENTITIES_FILENAME


def test_check_1_watches_the_robots_the_roster_names(tmp_path, capsys):
    """The point of the roster: the same command works on every world, with no --robot to forget.

    A static prop in the same record must not be watched -- watching every root body would fire
    check 1 on the furniture, which is the reason this file exists rather than a list of names.
    """
    now = time.time()
    write_run(
        tmp_path,
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        [line for t in range(0, 120)
         for line in (pose_line(float(t), "base", 0.0), pose_line(float(t), "shelf", 3.0))],
    )
    write_roster(tmp_path, [{"name": "robot", "kind": "robot", "body": "base"},
                            {"name": "shelf", "kind": "object", "body": "shelf"}])
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
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        [pose_line(float(t), "base", 0.0) for t in range(0, 120)],
    )
    write_roster(tmp_path, [{"name": "robot", "kind": "robot", "body": "base", "present": False}])
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert not [f for f in payload["findings"] if f["check"] == "robot-motion"]
    assert any("check 1" in s for s in payload["skipped"]), "nothing watched must say so"


def test_robot_overrides_the_roster(tmp_path, capsys):
    """--robot stays useful: a run with no roster, and watching something not called a robot."""
    now = time.time()
    write_run(
        tmp_path,
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        [pose_line(float(t), "shelf", 3.0) for t in range(0, 120)],
    )
    write_roster(tmp_path, [{"name": "shelf", "kind": "object", "body": "shelf"}])
    assert health.main([str(tmp_path), "--robot", "shelf", "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [f for f in payload["findings"] if f["check"] == "robot-motion"]


def test_a_missing_roster_says_so_rather_than_naming_a_flag(tmp_path, capsys):
    """The skip has to be actionable: which of the two reasons applies decides what to do."""
    now = time.time()
    write_run(
        tmp_path,
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        [pose_line(float(t), "base", 0.0) for t in range(0, 120)],
    )
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    skipped = " ".join(payload["skipped"])
    assert health.ENTITIES_FILENAME in skipped and "check 1" in skipped


def test_the_state_block_carries_kind_from_the_roster(tmp_path, capsys):
    """What the pose record cannot say: whether the thing that has not moved is a robot."""
    now = time.time()
    write_run(
        tmp_path,
        [clock_line(now - 120 + t, float(t)) for t in range(0, 120)],
        [line for t in range(0, 120)
         for line in (pose_line(float(t), "base", t * 0.5), pose_line(float(t), "shelf", 3.0))],
    )
    write_roster(tmp_path, [{"name": "robot", "kind": "robot", "body": "base"},
                            {"name": "shelf", "kind": "object", "body": "shelf"}])
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    kinds = {e["name"]: e.get("kind")
             for e in json.loads(capsys.readouterr().out)["state"]["entities"]}
    assert kinds == {"base": "robot", "shelf": "object"}


def test_a_malformed_roster_is_a_reason_and_not_a_crash(tmp_path, capsys):
    now = time.time()
    write_run(tmp_path, [clock_line(now - 120 + t, float(t)) for t in range(0, 120)])
    (tmp_path / health.ENTITIES_FILENAME).write_text("{not json")
    assert health.main([str(tmp_path), "--json"]) == health.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert any("check 1" in s for s in payload["skipped"])


def test_the_recorder_writes_the_roster_and_follows_a_change(tmp_path):
    """Asserted against the writer, since the file only helps if it is there and stays true.

    A roster written once at the first sample would describe the world the trial started in; a run
    that spawns an obstacle mid-trial is the normal case, not the exotic one.
    """
    mujoco = pytest.importorskip("mujoco")
    from roqsim.capture import ENTITIES_FILENAME, StateRecorder, snap_fps
    from roqsim.context import Entity, EntityRegistry

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
    recorder = StateRecorder(ctx, tmp_path / "run.npz", rate, sim_poses=True)

    def step():
        mujoco.mj_step(model, ctx.data)
        ctx.sim_time = float(ctx.data.time)
        recorder.sample(ctx)

    step()
    roster = json.loads((tmp_path / ENTITIES_FILENAME).read_text())["entities"]
    assert [(e["name"], e["kind"], e["body"]) for e in roster] == [("robot", "robot", "base")]

    ctx.entities.add(Entity(name="obstacle", kind="object", body="box"))
    step()
    names = [e["name"] for e in json.loads((tmp_path / ENTITIES_FILENAME).read_text())["entities"]]
    assert names == ["robot", "obstacle"], "a spawn mid-run must show up in the roster"


def test_the_recorder_needs_no_registry(tmp_path):
    """A driver that keeps no registry still records; the roster is simply absent, which the
    reader reports as a reason. Nothing about the recording depends on it."""
    mujoco = pytest.importorskip("mujoco")
    from roqsim.capture import ENTITIES_FILENAME, StateRecorder, snap_fps

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

    recorder = StateRecorder(ctx, tmp_path / "run.npz", rate=snap_fps(10.0, model.opt.timestep),
                             sim_poses=True)
    mujoco.mj_step(model, ctx.data)
    ctx.sim_time = float(ctx.data.time)
    recorder.sample(ctx)
    assert (tmp_path / "sim_poses.csv").exists()
    assert not (tmp_path / ENTITIES_FILENAME).exists()
