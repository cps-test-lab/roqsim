"""What a recording must preserve, and what it must refuse.

The first test in this file is the important one. ``mjSTATE_FULLPHYSICS`` -- the obvious choice --
silently drops ``ctrl`` and the mocap fields, so a recording
made with it replays every pedestrian and moving prop frozen at its compile-time pose and every door
driven toward 0. A fidelity test on a *static* world passes the whole time that is happening. So the
world here has a mocap body and a
nonzero ``ctrl``, and the assertion is field by field.

The second group pins the file: one mcap with four channels and two metadata records, chunked so
that a killed run keeps every chunk closed before the kill, and a chunk closed at least once per
wall second so that "before the kill" is at most a second ago.
"""

from __future__ import annotations

import json
import logging
import time

import mujoco
import numpy as np
import pytest
from synthetic_recording import write_state_recording

from roqsim import capture
from roqsim.capture import (
    STATE_FIELDS,
    STATE_SPEC,
    CaptureRate,
    StateRecorder,
    camera_from_row,
    record_dtype,
    select_tracks,
    snap_fps,
)
from roqsim.context import Entity, EntityRegistry
from roqsim.mcap_format import (
    CHANNEL_CLOCK,
    CHANNEL_JOINTS,
    CHANNEL_POSES,
    CHANNEL_STATE,
    FORMAT_VERSION,
    META_ENTITIES,
    META_RECORDING,
    PROFILE,
    ChunkedWriter,
    is_finished,
)
from roqsim.recording import RecordingError, open_recording

# A mocap body (what `walker` and `moving_box` drive), an actuated hinge (what `door` drives through
# data.ctrl), and a sensor -- the three things FULLPHYSICS gets wrong or right.
_MOVING_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body name="mo" mocap="true" pos="1 0 .5"><geom type="box" size=".2 .2 .2"/></body>
    <body name="arm" pos="0 0 .2">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".05" fromto="0 0 0 .5 0 0"/>
      <site name="tip" pos=".5 0 0"/>
    </body>
  </worldbody>
  <actuator><position joint="j" kp="20"/></actuator>
  <sensor><jointpos joint="j"/><framepos objtype="site" objname="tip"/></sensor>
</mujoco>
"""


@pytest.fixture
def moving():
    """A stepped world with a moved mocap body and a nonzero ctrl -- state worth round-tripping."""
    model = mujoco.MjModel.from_xml_string(_MOVING_XML)
    data = mujoco.MjData(model)
    data.ctrl[0] = 0.7
    data.mocap_pos[0] = [2.0, 0.3, 0.9]
    data.mocap_quat[0] = [0.9239, 0.0, 0.0, 0.3827]  # a 45 deg yaw, so quat is not identity
    for _ in range(300):
        mujoco.mj_step(model, data)
    return model, data


# -- the state spec: the bug this file exists to prevent -------------------------------------------


def test_the_recorded_spec_round_trips_mocap_ctrl_and_poses(moving):
    """Field by field, on a world that actually moves things by mocap and by ctrl."""
    model, data = moving
    size = mujoco.mj_stateSize(model, STATE_SPEC)
    buf = np.zeros(size)
    mujoco.mj_getState(model, data, buf, STATE_SPEC)

    restored = mujoco.MjData(model)
    mujoco.mj_setState(model, restored, buf, STATE_SPEC)
    mujoco.mj_forward(model, restored)

    for field in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xpos", "sensordata"):
        assert np.allclose(getattr(restored, field), getattr(data, field)), f"{field} was lost"


def test_fullphysics_would_lose_mocap_and_ctrl(moving):
    """The negative control: proves the test above is testing something real.

    If MuJoCo ever widens FULLPHYSICS to include these, this test fails and the comment explaining why
    the spec is composed by hand can be revisited -- which is the point of asserting it.
    """
    model, data = moving
    fp = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    buf = np.zeros(mujoco.mj_stateSize(model, fp))
    mujoco.mj_getState(model, data, buf, fp)
    restored = mujoco.MjData(model)
    mujoco.mj_setState(model, restored, buf, fp)
    mujoco.mj_forward(model, restored)
    assert not np.allclose(restored.mocap_pos, data.mocap_pos)
    assert not np.allclose(restored.ctrl, data.ctrl)


def test_the_spec_covers_every_mujoco_keyframe_field():
    """The spec is MuJoCo's own notion of a saved state, so it must cover every ``key_*`` field.

    Pins the choice to MuJoCo rather than to our judgement: a release that extends its own keyframe
    definition fails here instead of silently narrowing every recording written afterwards.
    """
    model = mujoco.MjModel.from_xml_string(_MOVING_XML)
    keyframe_fields = {
        name[len("key_") :]
        for name in dir(model)
        if name.startswith("key_") and not name.endswith(("adr", "num"))
    }
    covered = set(STATE_FIELDS) | {"mpos": "mocap_pos", "mquat": "mocap_quat"}.keys()
    # MuJoCo spells the mocap keyframe fields mpos/mquat; the state spec spells them mocap_pos/quat.
    aliases = {"mpos": "mocap_pos", "mquat": "mocap_quat"}
    missing = {
        f for f in keyframe_fields if aliases.get(f, f) not in STATE_FIELDS and f not in covered
    }
    assert not missing, f"MuJoCo keyframe fields not in the recorded spec: {sorted(missing)}"


def test_the_spec_is_not_a_preset():
    """Neither FULLPHYSICS nor INTEGRATION: one drops fields, the other adds 6*nbody of zeros."""
    assert STATE_SPEC != int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    assert STATE_SPEC != int(mujoco.mjtState.mjSTATE_INTEGRATION)
    assert not STATE_SPEC & int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)


# -- recording and reading back --------------------------------------------------------------------


class _Ctx:
    """The three members StateRecorder touches, so a test needs no Engine."""

    def __init__(self, model, data):
        self.model, self.data = model, data

    @property
    def sim_time(self) -> float:
        return float(self.data.time)


def _record(tmp_path, model, data, *, fps=25, steps=600, camera=False, world="w.yaml"):
    ctx = _Ctx(model, data)
    rec = StateRecorder(
        ctx, tmp_path / "run.mcap", snap_fps(fps, model.opt.timestep), world=world, camera=camera
    )
    for _ in range(steps):
        mujoco.mj_step(model, data)
        rec.sample(ctx, cam=_a_camera() if camera else None)
    return rec, rec.close()


def _a_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.0, 2.0, 3.0]
    cam.distance, cam.azimuth, cam.elevation = 7.5, 123.0, -34.0
    return cam


def test_a_recording_is_a_standard_mcap_file(tmp_path, moving):
    """Readable by the mcap library, with no roqsim at all -- the reason the container is mcap.

    The four channels and the two metadata records are the file's contract; the library's own
    reader is the witness that the file is what the format says it is.
    """
    from mcap.reader import make_reader

    model, data = moving
    _record(tmp_path, model, data)
    with (tmp_path / "run.mcap").open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        assert reader.get_header().profile == PROFILE
        assert {c.topic for c in summary.channels.values()} == {
            CHANNEL_STATE,
            CHANNEL_POSES,
            CHANNEL_JOINTS,
            CHANNEL_CLOCK,
        }
        assert {s.name for s in summary.schemas.values()} == {
            "roqsim.poses",
            "roqsim.joints",
            "roqsim.clock",
        }
        assert [m.name for m in summary.metadata_indexes] == [META_RECORDING, META_ENTITIES][:1] + [
            META_RECORDING
        ], "the provenance is written at the start and again at close; no roster without a registry"
        counts = summary.statistics.channel_message_counts
        assert len(set(counts.values())) == 1, "one message per channel per sample"
        assert all(i.compression == "zstd" for i in summary.chunk_indexes)


def test_the_provenance_metadata_is_json_and_the_last_one_wins(tmp_path, moving):
    from mcap.reader import make_reader

    model, data = moving
    rec, _ = _record(tmp_path, model, data)
    with (tmp_path / "run.mcap").open("rb") as handle:
        records = [m for m in make_reader(handle).iter_metadata() if m.name == META_RECORDING]
    first, last = (json.loads(r.metadata["json"]) for r in (records[0], records[-1]))
    assert first["state_spec"] == STATE_SPEC
    assert "samples" not in first and last["samples"] == rec.frames
    assert open_recording(tmp_path / "run.mcap").meta["samples"] == rec.frames


def test_samples_are_one_structured_array(tmp_path, moving):
    """Not parallel states/times arrays: one record, so time and state cannot desynchronise."""
    model, data = moving
    _record(tmp_path, model, data)
    samples = open_recording(tmp_path / "run.mcap").samples
    assert samples.dtype.names == ("t", "w", "s")
    assert samples["t"].dtype == np.float64  # times stay f8: f32 degrades with magnitude
    assert samples["w"].dtype == np.float64  # ... and so does the wall clock, for the same reason
    assert samples["s"].dtype == np.float32  # states are f32: micrometre precision, half the size


# -- the two clocks --------------------------------------------------------------------------------


def test_every_sample_carries_both_clocks(tmp_path, moving):
    """Sim time and wall time are independent measurements, so a recording must hold both.

    Neither is derivable from the other: the ratio between them is the run's real-time factor, which is
    the thing that varies.
    """
    model, data = moving
    _record(tmp_path, model, data, steps=400)
    samples = open_recording(tmp_path / "run.mcap").samples
    sim, wall = samples["t"], samples["w"]

    assert np.all(np.diff(sim) > 0), "sim time advances every sample"
    assert np.all(np.diff(wall) >= 0), "a monotonic clock never goes backwards"
    # 400 steps of a two-body world take milliseconds of wall time but 0.8 s of sim time, so this
    # would fail if wall were quietly a copy of sim.
    assert wall[-1] != pytest.approx(sim[-1], abs=1e-3)


def test_the_wall_clock_starts_at_zero_and_is_not_a_timestamp(tmp_path, moving):
    """Elapsed seconds from the recorder's start -- 1.7e9 would mean somebody wrote the epoch in."""
    model, data = moving
    _record(tmp_path, model, data, steps=200)
    wall = open_recording(tmp_path / "run.mcap").samples["w"]
    assert 0.0 <= wall[0] < 1.0, "the first sample is a few ms in, not a Unix timestamp"
    assert wall[-1] < 60.0, "200 steps of a toy world cannot take a minute"


def test_the_wall_clock_origin_is_named_in_the_provenance(tmp_path, moving):
    """A reader without our source must be able to learn what ``w``'s zero is."""
    model, data = moving
    _record(tmp_path, model, data)
    meta = open_recording(tmp_path / "run.mcap").meta
    assert "perf_counter" in meta["wall_clock_origin"]
    assert meta["wall_start_epoch"] == pytest.approx(time.time(), abs=120)


def test_the_json_channels_carry_the_epoch_and_the_state_the_elapsed_clock(tmp_path, moving):
    """Two spellings of one wall clock, each for its reader: a process outside relates the epoch to
    its own stamps; the state's elapsed column keeps nanosecond resolution and monotonicity."""
    model, data = moving
    _record(tmp_path, model, data, steps=200)
    rec = open_recording(tmp_path / "run.mcap")
    clock = rec.clock
    assert clock[0]["wall_ts"] == pytest.approx(
        rec.meta["wall_start_epoch"] + rec.wall_times[0], abs=1e-3
    )
    assert clock[-1]["sim_ts"] == pytest.approx(rec.times[-1])
    assert rec.poses(0)["w"] == pytest.approx(clock[0]["wall_ts"], abs=1e-5)


def test_a_restored_sample_reports_its_wall_time(tmp_path, moving):
    """The reader side of the column: it survives to :class:`Sample`, not just to the file."""
    model, data = moving
    _record(tmp_path, model, data, steps=400)
    rec = open_recording(tmp_path / "run.mcap")
    rec._model, rec._ctx, rec._data = model, None, mujoco.MjData(model)
    rec._buf = np.empty(int(rec.meta["state_size"]))
    rec.build = lambda *a, **k: (model, None)  # the world file is not on disk in this test

    wall = rec.wall_times
    sample = rec.at()
    assert sample.wall_time == pytest.approx(float(wall[-1]))
    assert rec.real_time_factor == pytest.approx((rec.span[1] - rec.span[0]) / (wall[-1] - wall[0]))
    assert rec.describe()["real_time_factor"] == rec.real_time_factor
    json.dumps(rec.describe())  # --check must stay JSON-safe with the new fields


def test_a_pause_shows_up_in_the_wall_clock_but_not_in_sim_time(tmp_path, moving):
    """What the column is *for*: real time the physics did not account for.

    A stall (a paused viewer, a slow sensor, a reset that rebuilt the world) is invisible in ``t`` by
    construction. Simulated here with a real sleep, because a mocked clock would test the mock.
    """
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, model.opt.timestep), world="w")
    for step in range(200):
        mujoco.mj_step(model, data)
        if step == 100:
            time.sleep(0.05)  # the stall
        rec.sample(ctx)
    rec.close()

    samples = open_recording(tmp_path / "run.mcap").samples
    sim_gaps, wall_gaps = np.diff(samples["t"]), np.diff(samples["w"])
    assert sim_gaps.max() - sim_gaps.min() < 1e-9, "sim time is on the capture grid throughout"
    assert wall_gaps.max() > 0.04, "the stall is visible in wall time"


def test_a_reset_restarts_the_schedule_but_not_the_wall_clock(tmp_path, moving):
    """Real time did not restart, and a reset's own cost is exactly what somebody would look for."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(200):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    before = rec._last_w
    rec.on_reset()
    for _ in range(200):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    assert rec._last_w > before, "the wall clock kept running across the reset"


def test_a_state_survives_the_float32_round_trip(tmp_path, moving):
    """The fidelity claim, asserted rather than argued: geom poses to well under a millimetre."""
    model, data = moving
    _record(tmp_path, model, data, steps=400)
    before = data.geom_xpos.copy()
    rec = open_recording(tmp_path / "run.mcap")
    monkey = (
        rec.build
    )  # the world is rebuilt from provenance; here we inject the model we already have
    rec._model, rec._ctx, rec._data = model, None, mujoco.MjData(model)
    rec._buf = np.empty(int(rec.meta["state_size"]))
    rec.build = lambda *a, **k: (model, None)
    try:
        sample = rec.at()  # the last sample, which is the state we just stepped to
    finally:
        rec.build = monkey
    assert np.allclose(sample.data.geom_xpos, before, atol=5e-5)


def test_nothing_sampled_writes_nothing_and_says_so(tmp_path, moving, caplog):
    """An empty file would pass every downstream existence check, so it must not be written."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, 0.002), world="w")
    with caplog.at_level(logging.WARNING):
        assert rec.close() is None
    assert not (tmp_path / "run.mcap").exists()
    assert list(tmp_path.iterdir()) == [], "the file opens on the first sample, not before"
    assert "no samples" in caplog.text


def test_close_is_idempotent(tmp_path, moving):
    """Every call site is a finally, and a driver may unwind more than once."""
    model, data = moving
    rec, first = _record(tmp_path, model, data)
    assert first is not None
    assert rec.close() is None


def test_a_run_leaves_exactly_one_file(tmp_path, moving):
    """The recording is the run's whole record: no sidecar, no stream, no roster file.

    A campaign lists every file under a run directory as one of its outputs, so anything beside the
    recording is an artifact published.
    """
    model, data = moving
    rec, path = _record(tmp_path, model, data)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.mcap"]
    assert rec.frames > 0
    assert is_finished(path)


# -- the file while the run is still going --------------------------------------------------------


def test_the_file_is_readable_while_the_run_is_still_going(tmp_path, moving, monkeypatch):
    """The point of chunking: what has been closed is on disk and opens, summary or no summary."""
    monkeypatch.setattr(capture, "CHUNK_SECONDS", 0.0)  # close a chunk at every sample
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    live = open_recording(tmp_path / "run.mcap")
    assert len(live) == rec.frames, "every closed chunk is readable before close()"
    assert not live.finished and not is_finished(tmp_path / "run.mcap")
    rec.close()
    assert open_recording(tmp_path / "run.mcap").finished


def test_a_chunk_closes_within_a_second_of_wall_time(tmp_path, moving):
    """What bounds the loss on a kill. Pinned against the clock, not the mechanism: a library
    change that removed the chunk-closing hook would fail here rather than the guarantee."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(100):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    with pytest.raises(RecordingError, match="no samples"):
        open_recording(tmp_path / "run.mcap")  # under a second in: everything is in the open chunk
    time.sleep(1.05)
    for _ in range(20):  # the next sample is due; taking it closes the chunk
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    assert len(open_recording(tmp_path / "run.mcap")) == rec.frames
    rec.close()


def test_a_killed_writer_leaves_every_closed_chunk_readable(tmp_path, moving, monkeypatch):
    """Drop the recorder without close(): what is on disk is what a SIGKILL leaves."""
    monkeypatch.setattr(capture, "CHUNK_SECONDS", 0.0)
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    frames = rec.frames
    del rec
    killed = open_recording(tmp_path / "run.mcap")
    assert len(killed) == frames
    assert not killed.finished
    assert killed.poses(frames - 1)["bodies"].keys() == {"mo", "arm"}


def test_a_file_cut_mid_chunk_opens_up_to_the_last_whole_chunk(tmp_path, moving, monkeypatch):
    monkeypatch.setattr(capture, "CHUNK_SECONDS", 0.0)
    model, data = moving
    _record(tmp_path, model, data, steps=600)
    whole = (tmp_path / "run.mcap").read_bytes()
    cut = tmp_path / "cut.mcap"
    cut.write_bytes(whole[: len(whole) // 2])
    partial = open_recording(cut)
    assert 0 < len(partial) < 30
    assert not partial.finished
    assert np.array_equal(
        partial.times, open_recording(tmp_path / "run.mcap").times[: len(partial)]
    )


def test_the_chunk_closing_hook_is_pinned(tmp_path):
    """The writer reaches the library's chunk finalisation by name; losing it must fail loudly."""
    from mcap.writer import Writer

    assert callable(getattr(Writer, ChunkedWriter._FINALIZE, None))

    class Hookless(ChunkedWriter):
        _FINALIZE = "_Writer__no_such_hook"

    with pytest.raises(RecordingError, match="chunks could not be closed"):
        Hookless(tmp_path / "x.mcap")
    assert not (tmp_path / "x.mcap").read_bytes(), "nothing was written on the refusal"


def test_replay_before_close_sees_the_samples_taken_so_far(tmp_path, moving):
    """A driver that still holds the world reads the samples back off the file, open chunk included."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    assert len(list(rec.replay(ctx))) == rec.frames
    rec.close()


def test_replay_after_close_still_works(tmp_path, moving):
    model, data = moving
    ctx = _Ctx(model, data)
    rec, path = _record(tmp_path, model, data)
    assert path is not None
    replayed = [t for t, _ in rec.replay(ctx)]
    assert len(replayed) == rec.frames
    assert replayed == sorted(replayed)


def test_a_recording_path_without_the_suffix_is_still_found_where_it_says(tmp_path, moving):
    """``--record out`` lands as ``out.mcap``, and close() returns the path that exists."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "out", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    written = rec.close()
    assert written == tmp_path / "out.mcap"
    assert written.exists()
    assert open_recording(written).meta["world"] == "w"


def test_the_sample_rate_is_in_simulated_seconds(tmp_path, moving):
    """1000 steps at dt=0.002 is 2 s of sim time, so 25 fps must give about 50 samples."""
    model, data = moving
    rec, _ = _record(tmp_path, model, data, fps=25, steps=1000)
    assert rec.frames == pytest.approx(50, abs=2)


def test_a_slower_rate_takes_proportionally_fewer_samples(tmp_path, moving):
    model, data = moving
    rec, _ = _record(tmp_path, model, data, fps=10, steps=1000)
    assert rec.frames == pytest.approx(20, abs=2)


# -- the camera track ------------------------------------------------------------------------------


def test_a_camera_track_round_trips(tmp_path, moving):
    """So a render can reproduce what the person was looking at, drags and arrow-key flight included."""
    model, data = moving
    _record(tmp_path, model, data, camera=True, steps=200)
    rec = open_recording(tmp_path / "run.mcap")
    assert rec.has_camera and "cam" in rec.samples.dtype.names
    cam = camera_from_row(rec.samples["cam"][0])
    assert list(cam.lookat) == pytest.approx([1.0, 2.0, 3.0])
    assert (cam.distance, cam.azimuth, cam.elevation) == pytest.approx((7.5, 123.0, -34.0))


def test_a_headless_recording_has_no_camera_track(tmp_path, moving):
    model, data = moving
    _record(tmp_path, model, data, camera=False, steps=200)
    rec = open_recording(tmp_path / "run.mcap")
    # Both clocks are always there; only the camera track is conditional.
    assert rec.samples.dtype.names == ("t", "w", "s")
    assert rec.meta["camera_track"] is False and rec.meta["camera"] is False


# -- provenance ------------------------------------------------------------------------------------


def test_provenance_names_the_world_resolvably_and_the_versions(tmp_path, moving):
    """A path alone is useless to another process; the ref plus versions is what lets it rebuild."""
    model, data = moving
    _record(tmp_path, model, data, world="roqsim_scenes:depot")
    meta = open_recording(tmp_path / "run.mcap").meta
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["world"] == "roqsim_scenes:depot"
    assert {"roqsim", "mujoco", "numpy"} <= set(meta["packages"])
    assert meta["state_fields"] == list(STATE_FIELDS)
    assert meta["state_size"] == mujoco.mj_stateSize(model, STATE_SPEC)
    assert meta["capture_fps"] == [25, 1]
    assert meta["model"]["nmocap"] == 1 and meta["model"]["nu"] == 1
    assert meta["tracks"] == {"bodies": ["mo", "arm"], "joints": ["j"]}


def test_numpy_version_is_recorded(tmp_path, moving):
    """Because rng.choice(replace=False) consumption is implementation-dependent, so exact *noise*
    replay is pinned to a numpy version even though the physics is not."""
    model, data = moving
    _record(tmp_path, model, data)
    assert open_recording(tmp_path / "run.mcap").meta["packages"]["numpy"] == np.__version__


# -- refusals --------------------------------------------------------------------------------------


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(RecordingError, match="no such recording"):
        open_recording(tmp_path / "absent.mcap")


def test_a_numpy_archive_is_refused_by_name(tmp_path):
    """The formats before this one; there is no reader for them, and the refusal says which this is."""
    old = tmp_path / "run.npz"
    np.savez(old, meta=np.array("{}"), samples=np.zeros(3))
    with pytest.raises(RecordingError) as err:
        open_recording(old)
    assert "numpy archive" in str(err.value) and "format 1 or 2" in str(err.value)
    assert f"format {FORMAT_VERSION}" in str(err.value)

    renamed = tmp_path / "run.mcap"  # the same bytes under the new suffix are still refused
    renamed.write_bytes(old.read_bytes())
    with pytest.raises(RecordingError, match="zip archive"):
        open_recording(renamed)


def test_something_that_is_not_mcap_is_refused(tmp_path):
    bad = tmp_path / "x.mcap"
    bad.write_bytes(b"not a recording at all")
    with pytest.raises(RecordingError, match="not an mcap file"):
        open_recording(bad)


def test_a_newer_format_version_is_refused(tmp_path, moving):
    """Written to a contract this code has not seen: refused by name, not read by overlap."""
    model, data = moving
    size = mujoco.mj_stateSize(model, STATE_SPEC)
    samples = np.zeros(3, record_dtype(size, False))
    samples["t"] = [0.0, 0.04, 0.08]
    meta = {
        "format_version": FORMAT_VERSION + 1,
        "state_size": size,
        "capture_fps": [25, 1],
        "camera_track": False,
        "state_spec": STATE_SPEC,
    }
    write_state_recording(tmp_path / "next.mcap", meta, samples)
    with pytest.raises(
        RecordingError, match=f"v{FORMAT_VERSION + 1}; this roqsim reads v{FORMAT_VERSION}"
    ):
        open_recording(tmp_path / "next.mcap")


def test_a_layout_that_disagrees_with_the_provenance_is_refused(tmp_path):
    """The file and its declared layout must agree, or a reader silently misreads columns."""
    samples = np.zeros(3, record_dtype(4, False))
    meta = {
        "state_size": 99,
        "camera_track": False,
        "capture_fps": [25, 1],
        "state_spec": STATE_SPEC,
    }
    write_state_recording(tmp_path / "lying.mcap", meta, samples)
    with pytest.raises(RecordingError, match="disagree"):
        open_recording(tmp_path / "lying.mcap")


def test_a_foreign_mcap_is_refused(tmp_path):
    writer = ChunkedWriter(tmp_path / "other.mcap")
    writer.start("ros2", library="x")
    writer.finish()
    with pytest.raises(RecordingError, match="profile 'ros2'"):
        open_recording(tmp_path / "other.mcap")


def test_a_fullphysics_recording_is_refused_not_rendered(tmp_path, moving):
    """The whole point: a FULLPHYSICS recording must fail loudly, not replay with frozen pedestrians."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "old.mcap", snap_fps(25, 0.002), world="w")
    rec._provenance["state_spec"] = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    for _ in range(200):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    rec.close()
    opened = open_recording(tmp_path / "old.mcap")
    with pytest.raises(RecordingError) as err:
        opened._check_size(model, "w")
    message = str(err.value)
    assert "8223" in message, "the offending spec must be named"
    assert "frozen at its compile-time pose" in message, "and what it would have done"
    # Must NOT blame the world: the model dimensions are identical, only the format differs.
    assert "does not match this recording" not in message


# -- selecting a moment ----------------------------------------------------------------------------


def _fake_recording(times, tmp_path):
    from roqsim.recording import Recording

    samples = np.zeros(len(times), record_dtype(1, False))
    samples["t"] = times
    meta = {
        "format_version": FORMAT_VERSION,
        "state_size": 1,
        "capture_fps": [25, 1],
        "camera_track": False,
        "state_spec": STATE_SPEC,
    }
    return Recording(tmp_path / "x.mcap", meta, samples)


def test_at_picks_the_nearer_sample_not_the_preceding_one(tmp_path):
    rec = _fake_recording([0.0, 0.04, 0.08, 0.12], tmp_path)
    assert rec.index_at(0.071) == 2  # nearer 0.08 than 0.04
    assert rec.index_at(0.05) == 1  # nearer 0.04 than 0.08


def test_a_tie_resolves_to_the_earlier_sample(tmp_path):
    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    assert rec.index_at(0.02) == 0


def test_an_exact_hit_is_exact(tmp_path):
    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    assert rec.index_at(0.04) == 1


def test_out_of_range_refuses_and_names_the_span(tmp_path):
    """Clamping would make a wrong answer look right."""
    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    with pytest.raises(RecordingError) as err:
        rec.index_at(999.0)
    assert "0.000..0.080" in str(err.value)
    with pytest.raises(RecordingError):
        rec.index_at(-5.0)


def test_just_outside_the_span_is_tolerated_within_one_period(tmp_path):
    """A caller asking for the end of a run should not trip over a rounding of the last timestamp."""
    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    assert rec.index_at(0.081) == 2


def test_the_at_record_reports_which_sample_was_used(tmp_path):
    """A caller must see it landed 12 ms early rather than assume it got what it asked for."""
    from roqsim.recording import Sample

    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    data = mujoco.MjData(mujoco.MjModel.from_xml_string(_MOVING_XML))
    sample = Sample(0.08, 2, data, None, 1.25)
    record = rec.at_record(0.071, sample)
    assert record == {
        "sim_time": 0.08,
        "wall_time": 1.25,  # when in the run's real elapsed time this moment happened
        "sample_index": 2,
        "requested_at": 0.071,
        "at_error": 0.009,
    }


def test_the_at_record_of_a_defaulted_request_has_no_error(tmp_path):
    from roqsim.recording import Sample

    rec = _fake_recording([0.0, 0.04], tmp_path)
    sample = Sample(0.04, 1, mujoco.MjData(mujoco.MjModel.from_xml_string(_MOVING_XML)))
    record = rec.at_record(None, sample)
    assert record["requested_at"] is None and record["at_error"] is None


def test_describe_is_json_safe(tmp_path):
    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    json.dumps(rec.describe())  # must not raise


def test_span_and_len(tmp_path):
    rec = _fake_recording([0.0, 0.04, 0.08], tmp_path)
    assert len(rec) == 3
    assert rec.span == (0.0, pytest.approx(0.08))


def test_a_rate_is_read_back_as_the_exact_rational(tmp_path):
    from roqsim.recording import Recording

    samples = np.zeros(2, record_dtype(1, False))
    meta = {
        "format_version": FORMAT_VERSION,
        "state_size": 1,
        "capture_fps": [500, 17],
        "camera_track": False,
        "state_spec": STATE_SPEC,
    }
    rec = Recording(tmp_path / "x.mcap", meta, samples)
    assert rec.fps == CaptureRate(snap_fps(30, 0.002).fps, 17, snap_fps(30, 0.002).fps, 0).fps


# -- the poses and joints channels -----------------------------------------------------------------

# A free base carrying a hinged link, a tool welded to that link, and one unnamed body: the three
# kinds of thing below a robot's root that a success rule may read, plus the one the channel cannot
# name. Prefixed the way a spawn plugin prefixes a model's bodies.
_ARM_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="crate" pos="2 0 .1"><geom type="box" size=".1 .1 .1"/></body>
    <body name="r/base" pos="0 0 .5">
      <freejoint name="r/root"/>
      <geom type="box" size=".1 .1 .1"/>
      <body name="r/link" pos="0 0 .1">
        <joint name="r/shoulder" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0"/>
        <body name="r/tool" pos=".3 0 0">
          <joint name="r/slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".02"/>
        </body>
        <body pos="0 0 .05"><geom type="sphere" size=".01"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _arm():
    model = mujoco.MjModel.from_xml_string(_ARM_XML)
    ctx = _Ctx(model, mujoco.MjData(model))
    ctx.entities = EntityRegistry()
    ctx.entities.add(Entity(name="robot", kind="robot", body="r/base", meta={"prefix": "r/"}))
    ctx.entities.add(Entity(name="crate", kind="object", body="crate"))
    return model, ctx


def test_the_poses_channel_carries_every_named_body_not_only_roots(tmp_path, caplog):
    model, ctx = _arm()
    rec = StateRecorder(
        ctx, tmp_path / "run.mcap", snap_fps(1 / model.opt.timestep, model.opt.timestep)
    )
    with caplog.at_level(logging.INFO):
        for _ in range(20):
            mujoco.mj_step(model, ctx.data)
            rec.sample(ctx)
    rec.close()
    opened = open_recording(tmp_path / "run.mcap")
    last = opened.poses(len(opened) - 1)
    assert last["bodies"].keys() == {"crate", "r/base", "r/link", "r/tool"}, "every named body"
    assert last["t"] == pytest.approx(opened.times[-1])
    # xpos after the last step is the pose the last row was taken from (capture.py's one-step note).
    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r/tool")
    assert last["bodies"]["r/tool"][:3] == pytest.approx(ctx.data.xpos[tool], abs=1e-5)
    # Quaternion (x, y, z, w) and a 13-wide row: position, orientation, linear and angular twist.
    assert len(last["bodies"]["crate"]) == 13
    assert last["bodies"]["crate"][3:7] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    # The unnamed body is reported, with its parent, rather than silently absent.
    assert "1 unnamed bodies have no entry (under r/link)" in caplog.text


def test_the_joints_channel_carries_every_scalar_joint(tmp_path):
    model, ctx = _arm()
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(50, model.opt.timestep))
    for _ in range(40):
        mujoco.mj_step(model, ctx.data)
        rec.sample(ctx)
    rec.close()
    opened = open_recording(tmp_path / "run.mcap")
    q = opened.joints(len(opened) - 1)["q"]
    assert q.keys() == {"r/shoulder", "r/slide"}, "hinge and slide joints; not the free joint"
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r/shoulder")]
    assert q["r/shoulder"] == pytest.approx(ctx.data.qpos[adr], abs=1e-5)


def test_the_roster_is_written_and_the_last_one_wins(tmp_path):
    """A roster written once at the first sample would describe the world the trial started in; a
    run that spawns an obstacle mid-trial is the normal case, not the exotic one."""
    model, ctx = _arm()
    rec = StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(50, model.opt.timestep))
    for _ in range(20):
        mujoco.mj_step(model, ctx.data)
        rec.sample(ctx)
    ctx.entities.add(Entity(name="late", kind="object", body="crate"))
    for _ in range(20):
        mujoco.mj_step(model, ctx.data)
        rec.sample(ctx)
    rec.close()
    names = [e["name"] for e in open_recording(tmp_path / "run.mcap").entities]
    assert names == ["robot", "crate", "late"]


def test_a_recorder_without_a_registry_records_no_roster(tmp_path, moving):
    model, data = moving
    _record(tmp_path, model, data, steps=100)
    assert open_recording(tmp_path / "run.mcap").entities is None


# -- narrowing the decoded channels ---------------------------------------------------------------


def test_every_body_and_joint_is_recorded_by_default():
    model, ctx = _arm()
    bodies, joints, skipped = select_tracks(model, ctx.entities)
    assert [n for _, n in bodies] == ["crate", "r/base", "r/link", "r/tool"]
    assert [n for _, n, _ in joints] == ["r/shoulder", "r/slide"]
    assert skipped == ["r/link"]


@pytest.mark.parametrize(
    ("tracks", "exclude", "bodies", "joints"),
    [
        ("robot/**", None, ["r/base", "r/link", "r/tool"], ["r/shoulder", "r/slide"]),
        ("robot/*", None, ["r/base", "r/link", "r/tool"], ["r/shoulder", "r/slide"]),
        ("robot/base", None, ["r/base"], []),
        ("robot/shoulder", None, [], ["r/shoulder"]),
        ("crate", None, ["crate"], []),
        ("r/tool", None, ["r/tool"], []),  # a bare name is the compiled model's own
        ("crate/**,robot/tool", None, ["crate", "r/tool"], []),
        (None, "robot/**", ["crate"], []),
        ("robot/**", "robot/slide", ["r/base", "r/link", "r/tool"], ["r/shoulder"]),
        ("robot/tool", "robot/tool", [], []),  # an exclude wins
    ],
)
def test_tracks_and_exclude_narrow_the_channels(tracks, exclude, bodies, joints):
    """The pattern grammar: ``<entity>/<local>`` with ``**`` for all and ``*`` for one segment,
    the entity's spawn prefix removed from the local name, or a bare compiled name."""
    model, ctx = _arm()
    got_bodies, got_joints, _ = select_tracks(model, ctx.entities, tracks, exclude)
    assert [n for _, n in got_bodies] == bodies
    assert [n for _, n, _ in got_joints] == joints


def test_a_pattern_that_matches_nothing_refuses_naming_what_exists(tmp_path):
    """A typo must fail the run's start, not record a run that lacks the track it is judged on."""
    model, ctx = _arm()
    with pytest.raises(RecordingError) as err:
        StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, 0.002), tracks="robot/gripper")
    message = str(err.value)
    assert "'robot/gripper'" in message
    assert "robot (prefix 'r/')" in message and "crate" in message
    assert "r/base" in message and "r/shoulder" in message
    with pytest.raises(RecordingError, match="'nope'"):
        StateRecorder(ctx, tmp_path / "run.mcap", snap_fps(25, 0.002), exclude="nope")
    assert not (tmp_path / "run.mcap").exists()


def test_a_narrowed_recording_keeps_the_state_whole(tmp_path):
    model, ctx = _arm()
    rec = StateRecorder(
        ctx, tmp_path / "run.mcap", snap_fps(50, model.opt.timestep), tracks="crate"
    )
    for _ in range(40):
        mujoco.mj_step(model, ctx.data)
        rec.sample(ctx)
    rec.close()
    opened = open_recording(tmp_path / "run.mcap")
    assert opened.poses(0)["bodies"].keys() == {"crate"}
    assert opened.joints(0)["q"] == {}
    assert opened.meta["tracks"] == {"bodies": ["crate"], "joints": []}
    assert opened.samples["s"].shape[1] == mujoco.mj_stateSize(model, STATE_SPEC)
    assert len(opened.clock) == len(opened)


# -- decimation: fewer samples, the same states ----------------------------------------------------


def test_decimating_keeps_original_rows_and_divides_the_rate_exactly(tmp_path, moving):
    """The invariant ``roqsim render`` rests on: every frame is a state the simulation actually had.

    So decimation must *drop* rows, never resample or interpolate them -- and the declared rate has to
    follow exactly, which works because ``capture_fps`` is a ``[numerator, denominator]`` pair: 250 Hz
    by 8 is ``[250, 8]``, i.e. 31.25 fps rather than a rounded 31.
    """
    from fractions import Fraction

    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=250, samples=80)
    out = decimated(rec, 8, tmp_path / "thin.mcap")
    thin = open_recording(out)

    assert thin.fps == Fraction(250, 8)
    assert len(thin) == len(range(0, len(rec), 8))
    assert thin.samples.dtype == rec.samples.dtype
    for i in range(len(thin)):
        assert np.array_equal(thin.samples["s"][i], rec.samples["s"][i * 8])
        assert thin.samples["t"][i] == rec.samples["t"][i * 8]
        assert thin.clock[i] == rec.clock[i * 8]
    assert thin.finished


def test_decimating_by_one_is_a_copy(tmp_path, moving):
    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=50, samples=20)
    same = open_recording(decimated(rec, 1, tmp_path / "same.mcap"))

    assert same.fps == rec.fps
    assert len(same) == len(rec)
    assert np.array_equal(same.samples["s"], rec.samples["s"])


def test_decimating_away_the_span_is_refused(tmp_path, moving):
    """Two samples is the minimum that still has a span; one is a still, not a recording."""
    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=25, samples=6)

    with pytest.raises(RecordingError, match="at least two"):
        decimated(rec, 100, tmp_path / "gone.mcap")


def test_a_decimate_factor_below_one_is_refused(tmp_path, moving):
    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=25, samples=10)

    with pytest.raises(RecordingError, match="1 or more"):
        decimated(rec, 0, tmp_path / "no.mcap")


def _recorded(tmp_path, model, data, *, fps: int, samples: int):
    """A recording of ``model`` stepped ``samples`` times, at a declared ``fps``."""
    size = mujoco.mj_stateSize(model, STATE_SPEC)
    rows = np.zeros(samples, dtype=record_dtype(size, False))
    buf = np.zeros(size)
    for i in range(samples):
        mujoco.mj_step(model, data)
        mujoco.mj_getState(model, data, buf, STATE_SPEC)
        rows["t"][i] = data.time
        rows["w"][i] = i / fps
        rows["s"][i] = buf
    meta = {
        "state_size": size,
        "capture_fps": [fps, 1],
        "camera_track": False,
        "state_spec": STATE_SPEC,
        "world": "synthetic.yaml",
        "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)},
    }
    path = write_state_recording(tmp_path / f"src-{fps}-{samples}.mcap", meta, rows)
    return open_recording(path)
