"""What a recording must preserve, and what it must refuse.

The first test in this file is the important one. ``mjSTATE_FULLPHYSICS`` -- the obvious choice --
silently drops ``ctrl`` and the mocap fields, so a recording
made with it replays every pedestrian and moving prop frozen at its compile-time pose and every door
driven toward 0. A fidelity test on a *static* world passes the whole time that is happening. So the
world here has a mocap body and a
nonzero ``ctrl``, and the assertion is field by field.
"""

from __future__ import annotations

import json
import logging

import mujoco
import numpy as np
import pytest

from roqsim.capture import (
    STATE_FIELDS,
    STATE_SPEC,
    STREAM_SUFFIX,
    CaptureRate,
    StateRecorder,
    camera_from_row,
    record_dtype,
    snap_fps,
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
        ctx, tmp_path / "run.npz", snap_fps(fps, model.opt.timestep), world=world, camera=camera
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


def test_a_recording_opens_with_bare_numpy(tmp_path, moving):
    """Readable by anyone with numpy and no roqsim at all -- the reason the container is an .npz."""
    model, data = moving
    _record(tmp_path, model, data)
    archive = np.load(tmp_path / "run.npz", allow_pickle=False)
    assert set(archive.files) == {"meta", "samples"}
    meta = json.loads(str(archive["meta"]))
    assert meta["state_spec"] == STATE_SPEC
    assert len(archive["samples"]) > 0


def test_samples_are_one_structured_array(tmp_path, moving):
    """Not parallel states/times arrays: one record, so time and state cannot desynchronise."""
    model, data = moving
    _record(tmp_path, model, data)
    samples = np.load(tmp_path / "run.npz")["samples"]
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
    samples = np.load(tmp_path / "run.npz")["samples"]
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
    wall = np.load(tmp_path / "run.npz")["samples"]["w"]
    assert 0.0 <= wall[0] < 1.0, "the first sample is a few ms in, not a Unix timestamp"
    assert wall[-1] < 60.0, "200 steps of a toy world cannot take a minute"


def test_the_wall_clock_origin_is_named_in_the_provenance(tmp_path, moving):
    """A bare-numpy reader must be able to learn what ``w``'s zero is without reading our source."""
    model, data = moving
    _record(tmp_path, model, data)
    meta = json.loads(str(np.load(tmp_path / "run.npz")["meta"]))
    assert "perf_counter" in meta["wall_clock_origin"]


def test_a_restored_sample_reports_its_wall_time(tmp_path, moving):
    """The reader side of the column: it survives to :class:`Sample`, not just to the file."""
    model, data = moving
    _record(tmp_path, model, data, steps=400)
    rec = open_recording(tmp_path / "run.npz")
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
    import time

    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world="w")
    for step in range(200):
        mujoco.mj_step(model, data)
        if step == 100:
            time.sleep(0.05)  # the stall
        rec.sample(ctx)
    rec.close()

    samples = np.load(tmp_path / "run.npz")["samples"]
    sim_gaps, wall_gaps = np.diff(samples["t"]), np.diff(samples["w"])
    assert sim_gaps.max() - sim_gaps.min() < 1e-9, "sim time is on the capture grid throughout"
    assert wall_gaps.max() > 0.04, "the stall is visible in wall time"


def test_a_reset_restarts_the_schedule_but_not_the_wall_clock(tmp_path, moving):
    """Real time did not restart, and a reset's own cost is exactly what somebody would look for."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(200):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    before = rec._last_w
    rec.on_reset()
    for _ in range(200):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    assert rec._last_w > before, "the wall clock kept running across the reset"


def test_the_archive_is_deflated(tmp_path, moving):
    """Compressed, and a regression to a stored archive must fail a test rather than a review.

    Not for the float mantissas -- those are incompressible noise in the low bits, and float32 has
    already dropped the worst of them -- but because a state vector *repeats*: most of a world stands
    still, so most of each record is the previous record. Measured on real recordings from this
    substrate that is 70% of raw for a bare mobile robot and 11% for a world of pedestrians, and it is
    paid for once at close rather than inside the loop.
    """
    import zipfile

    model, data = moving
    _record(tmp_path, model, data)
    with zipfile.ZipFile(tmp_path / "run.npz") as zf:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())


def test_a_state_survives_the_float32_round_trip(tmp_path, moving):
    """The fidelity claim, asserted rather than argued: geom poses to well under a millimetre."""
    model, data = moving
    _record(tmp_path, model, data, steps=400)
    before = data.geom_xpos.copy()
    rec = open_recording(tmp_path / "run.npz")
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
    rec = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, 0.002), world="w")
    with caplog.at_level(logging.WARNING):
        assert rec.close() is None
    assert not (tmp_path / "run.npz").exists()
    assert list(tmp_path.iterdir()) == [], "the sample stream opens on the first sample, not before"
    assert "no samples" in caplog.text


def test_close_is_idempotent(tmp_path, moving):
    """Every call site is a finally, and a driver may unwind more than once."""
    model, data = moving
    rec, first = _record(tmp_path, model, data)
    assert first is not None
    assert rec.close() is None


def test_the_sample_stream_is_packed_away_at_close(tmp_path, moving):
    """The samples live on disk during the run, so close() must leave no trace of that.

    A campaign lists every file under a run directory as one of its outputs, so a temporary left
    behind is a temporary published.
    """
    model, data = moving
    rec, path = _record(tmp_path, model, data)
    stream = tmp_path / ("run.npz" + STREAM_SUFFIX)
    assert not stream.exists(), "the stream is unlinked once it is packed"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.clock_map.csv", "run.npz"]
    assert rec.frames > 0, "the sample count survives the stream it was counting"


def test_a_stream_left_by_an_earlier_run_is_not_adopted(tmp_path, moving):
    """A killed run leaves its stream at exactly the path the next run's recorder would use.

    So a run that ends before its first sample must not pack the *previous* run's samples into its
    own archive, under its own provenance -- which is what keying the pack off the file's existence
    rather than off having written it would do.
    """
    model, data = moving
    _record(tmp_path, model, data)  # a complete earlier run
    stale = tmp_path / ("run.npz" + STREAM_SUFFIX)
    stale.write_bytes(b"\x00" * (record_dtype(mujoco.mj_stateSize(model, STATE_SPEC), False).itemsize * 3))
    (tmp_path / "run.npz").unlink()

    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world="w")
    assert rec.close() is None, "no samples of its own means no archive"
    assert not (tmp_path / "run.npz").exists()
    assert stale.exists(), "somebody else's stream is left where it was, not consumed"


def test_the_samples_are_on_disk_while_the_run_is_still_going(tmp_path, moving):
    """The point of the whole arrangement: the run's memory does not grow with its length."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world="w")
    stream = tmp_path / ("run.npz" + STREAM_SUFFIX)
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    assert stream.exists(), "samples are written as they are taken"
    assert not (tmp_path / "run.npz").exists(), "the archive still appears only at close()"
    rec.close()


def test_replay_before_close_sees_the_samples_taken_so_far(tmp_path, moving):
    """The scenario adapter derives its run capture *before* closing, off the buffered stream.

    So the stream has to be flushed and mapped on demand, not only once it is finished -- otherwise a
    campaign's capture would silently hold whatever happened to have left the write buffer.
    """
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    assert len(list(rec.replay(ctx))) == rec.frames
    rec.close()


def test_replay_after_close_still_works(tmp_path, moving):
    """``roqsim sim`` closes first and derives afterwards, which is why the mapping outlives the file."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec, path = _record(tmp_path, model, data)
    assert path is not None
    replayed = [t for t, _ in rec.replay(ctx)]
    assert len(replayed) == rec.frames
    assert replayed == sorted(replayed)


def test_a_recording_path_without_the_suffix_is_still_found_where_it_says(tmp_path, moving):
    """np.savez appends .npz behind our backs; close() must not return a missing path."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "out", snap_fps(25, model.opt.timestep), world="w")
    for _ in range(600):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    written = rec.close()
    assert written == tmp_path / "out.npz"
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
    samples = np.load(tmp_path / "run.npz")["samples"]
    assert "cam" in samples.dtype.names
    cam = camera_from_row(samples["cam"][0])
    assert list(cam.lookat) == pytest.approx([1.0, 2.0, 3.0])
    assert (cam.distance, cam.azimuth, cam.elevation) == pytest.approx((7.5, 123.0, -34.0))


def test_a_headless_recording_has_no_camera_track(tmp_path, moving):
    model, data = moving
    _record(tmp_path, model, data, camera=False, steps=200)
    samples = np.load(tmp_path / "run.npz")["samples"]
    # Both clocks are always there; only the camera track is conditional.
    assert samples.dtype.names == ("t", "w", "s")
    assert json.loads(str(np.load(tmp_path / "run.npz")["meta"]))["camera_track"] is False


# -- provenance ------------------------------------------------------------------------------------


def test_provenance_names_the_world_resolvably_and_the_versions(tmp_path, moving):
    """A path alone is useless to another process; the ref plus versions is what lets it rebuild."""
    model, data = moving
    _record(tmp_path, model, data, world="roqsim_scenes:depot")
    meta = json.loads(str(np.load(tmp_path / "run.npz")["meta"]))
    assert meta["world"] == "roqsim_scenes:depot"
    assert {"roqsim", "mujoco", "numpy"} <= set(meta["packages"])
    assert meta["state_fields"] == list(STATE_FIELDS)
    assert meta["capture_fps"] == [25, 1]
    assert meta["model"]["nmocap"] == 1 and meta["model"]["nu"] == 1


def test_numpy_version_is_recorded(tmp_path, moving):
    """Because rng.choice(replace=False) consumption is implementation-dependent, so exact *noise*
    replay is pinned to a numpy version even though the physics is not."""
    model, data = moving
    _record(tmp_path, model, data)
    meta = json.loads(str(np.load(tmp_path / "run.npz")["meta"]))
    assert meta["packages"]["numpy"] == np.__version__


# -- refusals --------------------------------------------------------------------------------------


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(RecordingError, match="no such recording"):
        open_recording(tmp_path / "absent.npz")


def test_a_non_npz_is_refused_with_the_sigkill_hint(tmp_path):
    """The likely cause of an unreadable archive is a hard kill, so say that."""
    bad = tmp_path / "truncated.npz"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(RecordingError) as err:
        open_recording(bad)
    assert "SIGKILL" in str(err.value)


def test_a_truncated_recording_is_unreadable_and_says_why(tmp_path, moving):
    """The accepted cost of a standard container, asserted so nobody assumes otherwise."""
    model, data = moving
    _record(tmp_path, model, data)
    path = tmp_path / "run.npz"
    whole = path.read_bytes()
    path.write_bytes(whole[: int(len(whole) * 0.7)])
    with pytest.raises(RecordingError, match="not a readable recording"):
        open_recording(path)


def test_missing_members_are_named(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(RecordingError) as err:
        open_recording(path)
    assert "meta" in str(err.value) and "samples" in str(err.value)


def test_a_dtype_that_disagrees_with_the_provenance_is_refused(tmp_path):
    """The file and its declared layout must agree, or a reader silently misreads columns."""
    path = tmp_path / "lying.npz"
    meta = {
        "format_version": 1,
        "state_size": 99,
        "camera_track": False,
        "capture_fps": [25, 1],
        "state_spec": STATE_SPEC,
    }
    np.savez(path, meta=np.array(json.dumps(meta)), samples=np.zeros(3, record_dtype(4, False)))
    with pytest.raises(RecordingError, match="disagree"):
        open_recording(path)


def test_a_fullphysics_recording_is_refused_not_rendered(tmp_path, moving):
    """The whole point: a FULLPHYSICS recording must fail loudly, not replay with frozen pedestrians."""
    model, data = moving
    ctx = _Ctx(model, data)
    rec = StateRecorder(ctx, tmp_path / "old.npz", snap_fps(25, 0.002), world="w")
    rec._provenance["state_spec"] = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    for _ in range(200):
        mujoco.mj_step(model, data)
        rec.sample(ctx)
    rec.close()
    opened = open_recording(tmp_path / "old.npz")
    with pytest.raises(RecordingError) as err:
        opened._check_size(model, "w")
    message = str(err.value)
    assert "8223" in message, "the offending spec must be named"
    assert "frozen at its compile-time pose" in message, "and what it would have done"
    # Must NOT blame the world: the model dimensions are identical, only the format differs.
    assert "does not match this recording" not in message


# -- selecting a moment ----------------------------------------------------------------------------


class _Fake(list):
    """A recording stub with known sample times, so --at can be tested without a world."""


def _fake_recording(times, tmp_path):
    from roqsim.recording import Recording

    samples = np.zeros(len(times), record_dtype(1, False))
    samples["t"] = times
    meta = {
        "state_size": 1,
        "capture_fps": [25, 1],
        "camera_track": False,
        "state_spec": STATE_SPEC,
    }
    return Recording(tmp_path / "x.npz", meta, samples)


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
        "state_size": 1,
        "capture_fps": [500, 17],
        "camera_track": False,
        "state_spec": STATE_SPEC,
    }
    rec = Recording(tmp_path / "x.npz", meta, samples)
    assert rec.fps == CaptureRate(snap_fps(30, 0.002).fps, 17, snap_fps(30, 0.002).fps, 0).fps


# -- the pose record --------------------------------------------------------------------------------

# A free base carrying a hinged link, a tool welded to that link, and one unnamed body: the three
# kinds of thing below a robot's root that a success rule may read, plus the one the record cannot
# name.
_ARM_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 .5">
      <freejoint/>
      <geom type="box" size=".1 .1 .1"/>
      <body name="link" pos="0 0 .1">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0"/>
        <body name="tool" pos=".3 0 0"><geom type="sphere" size=".02"/></body>
        <body pos="0 0 .05"><geom type="sphere" size=".01"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_the_pose_record_carries_every_named_body_not_only_roots(tmp_path, caplog):
    model = mujoco.MjModel.from_xml_string(_ARM_XML)
    data = mujoco.MjData(model)
    ctx = _Ctx(model, data)
    rec = StateRecorder(
        ctx, tmp_path / "run.npz", snap_fps(1 / model.opt.timestep, model.opt.timestep),
        sim_poses=True,
    )
    with caplog.at_level(logging.INFO):
        for _ in range(20):
            mujoco.mj_step(model, data)
            rec.sample(ctx)

    rows = [line.split(",") for line in (tmp_path / "sim_poses.csv").read_text().splitlines()[1:]]
    assert {r[2] for r in rows} == {"base", "link", "tool"}, "every named body, and only those"
    last_tool = [r for r in rows if r[2] == "tool"][-1]
    # xpos after the last step is the pose the last row was taken from (capture.py's one-step note).
    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    assert [float(v) for v in last_tool[3:6]] == pytest.approx(data.xpos[tool], abs=1e-5)
    # The unnamed body is reported, with its parent, rather than silently absent.
    assert "1 unnamed bodies have no row (under link)" in caplog.text
    rec.close()


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
    out = decimated(rec, 8, tmp_path / "thin.npz")
    thin = open_recording(out)

    assert thin.fps == Fraction(250, 8)
    assert len(thin) == len(range(0, len(rec), 8))
    assert thin.samples.dtype == rec.samples.dtype
    for i in range(len(thin)):
        assert np.array_equal(thin.samples["s"][i], rec.samples["s"][i * 8])
        assert thin.samples["t"][i] == rec.samples["t"][i * 8]


def test_decimating_by_one_is_a_copy(tmp_path, moving):
    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=50, samples=20)
    same = open_recording(decimated(rec, 1, tmp_path / "same.npz"))

    assert same.fps == rec.fps
    assert len(same) == len(rec)
    assert np.array_equal(same.samples["s"], rec.samples["s"])


def test_decimating_away_the_span_is_refused(tmp_path, moving):
    """Two samples is the minimum that still has a span; one is a still, not a recording."""
    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=25, samples=6)

    with pytest.raises(RecordingError, match="at least two"):
        decimated(rec, 100, tmp_path / "gone.npz")


def test_a_decimate_factor_below_one_is_refused(tmp_path, moving):
    from roqsim.capture import decimated

    model, data = moving
    rec = _recorded(tmp_path, model, data, fps=25, samples=10)

    with pytest.raises(RecordingError, match="1 or more"):
        decimated(rec, 0, tmp_path / "no.npz")


def _recorded(tmp_path, model, data, *, fps: int, samples: int):
    """A recording of ``model`` stepped ``samples`` times, at a declared ``fps``."""
    import json

    from roqsim.capture import STATE_SPEC, record_dtype

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
        "format_version": 2,
        "state_size": size,
        "capture_fps": [fps, 1],
        "world": "synthetic.yaml",
        "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)},
    }
    path = tmp_path / f"src-{fps}-{samples}.npz"
    np.savez(path, meta=np.array(json.dumps(meta)), samples=rows)
    return open_recording(path)


# A flex's vertex bodies are one row each per sample, and no success rule reads one by name: the pose
# record leaves them out, keeps the body the flex hangs from, and says so once per flex.
_FLEX_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="table" pos="0 0 .5">
      <geom type="box" size=".1 .1 .01"/>
      <flexcomp name="pad" type="grid" count="3 3 1" spacing=".05 .05 .05" dim="2" radius=".001"
                pos="0 0 .05">
        <edge equality="true"/><pin id="0 1 2"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
"""


def test_the_pose_record_leaves_a_flexs_own_bodies_out(tmp_path, caplog):
    model = mujoco.MjModel.from_xml_string(_FLEX_XML)
    data = mujoco.MjData(model)
    ctx = _Ctx(model, data)
    rate = snap_fps(1 / model.opt.timestep, model.opt.timestep)
    rec = StateRecorder(ctx, tmp_path / "run.npz", rate, sim_poses=True)
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            mujoco.mj_step(model, data)
            rec.sample(ctx)
    rec.close()

    rows = [line.split(",") for line in (tmp_path / "sim_poses.csv").read_text().splitlines()[1:]]
    # The pinned vertices sit on `table`, which stays; the six free vertices' bodies (pad_3..pad_8)
    # are the flex's own.
    assert {r[2] for r in rows} == {"table"}
    assert "omits the 6 bodies of flex 'pad'" in caplog.text
    assert "0 unnamed bodies have no row" in caplog.text
