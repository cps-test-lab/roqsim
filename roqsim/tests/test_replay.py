"""Replaying a recording in the viewer: what the keys do, and what a live run must not notice.

The last test in this file is the important one. A replay adds a window, a tkinter transport and a
set of keys to ``roqsim sim``, and none of that may reach a run that is simulating a world -- the
command it has always been has to keep behaving exactly as it did.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys

import mujoco
import pytest

from roqsim import keys as keybind
from roqsim import render
from roqsim.capture import StateRecorder, snap_fps
from roqsim.recording import open_recording
from roqsim.replay import PLAY, SCRUB, SHOT, Replay, ReplayKeys, is_recording
from roqsim.shots import read_shots

_WORLD = """\
sim:
  pacing: asap
components:
- dummy: {}
"""


class _Ctx:
    """The three members StateRecorder touches, so a test needs no Engine."""

    def __init__(self, model, data):
        self.model, self.data = model, data

    @property
    def sim_time(self) -> float:
        return float(self.data.time)


class _Handle:
    """The four members a replay touches on a viewer handle, without a window."""

    def __init__(self):
        self.cam = mujoco.MjvCamera()
        self.syncs = 0

    def sync(self):
        self.syncs += 1

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self):
        return True

    def set_texts(self, _texts):
        pass


@pytest.fixture
def rec(tmp_path):
    world = tmp_path / "w.yaml"
    world.write_text(_WORLD, encoding="utf-8")
    model, data, _ctx, _view, _cam = render.build_target(str(world), None)
    ctx = _Ctx(model, data)
    recorder = StateRecorder(
        ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world=str(world)
    )
    for _ in range(600):
        mujoco.mj_step(model, data)
        recorder.sample(ctx)
    recorder.close()
    opened = open_recording(tmp_path / "run.npz")
    opened.build()
    return opened


@pytest.fixture
def replay(rec, tmp_path):
    return Replay(rec, _Handle(), state=tmp_path / "run.npz", shots=tmp_path / "shots.yaml")


# -- which target replays --------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["run.npz", "runs/x.NPZ", "/abs/path/run.npz"])
def test_a_recording_replays(target):
    assert is_recording(target)


@pytest.mark.parametrize("target", ["world.yaml", "scene.xml", "roqsim_assets:table", "bunny.obj"])
def test_everything_else_still_simulates(target):
    assert not is_recording(target)


# -- the keys ---------------------------------------------------------------------------------------


def test_a_replay_takes_only_keys_a_live_run_leaves_free():
    """Merged against the whole catalogue: a clash raises rather than binding one key to two things."""
    assert keybind.merge(ReplayKeys, *keybind.CATALOGUE) is not None


def test_the_camera_keeps_its_keys():
    """Flight, the mode switch and the F1 list are how a shot gets framed, so a replay takes none."""
    replay_codes = {code for binding in ReplayKeys.key_bindings for code in binding.codes}
    camera_codes = {code for binding in keybind.WALK for code in binding.codes}
    assert not replay_codes & camera_codes
    assert keybind.KEY_F10 not in replay_codes
    assert keybind.KEY_F1 not in replay_codes


def test_scrubbing_accumulates_and_is_taken_once():
    handler = ReplayKeys()
    forward, back = SCRUB.keys[1].code, SCRUB.keys[0].code
    handler.key_callback(forward)
    handler.key_callback(forward)
    handler.key_callback(back)
    assert handler.take_scrub() == 1
    assert handler.take_scrub() == 0


def test_play_and_shot_ignore_an_auto_repeat():
    """A held key delivers several presses; a toggle that acted on each would flicker."""
    handler = ReplayKeys()
    for _ in range(5):
        handler.key_callback(PLAY.keys[0].code)
        handler.key_callback(SHOT.keys[0].code)
    assert handler.take_play() is True
    assert handler.take_shot() is True
    assert handler.take_play() is False


# -- moving through the recording -------------------------------------------------------------------


def test_it_opens_on_the_first_sample(replay):
    assert replay.timeline.index == 0
    assert not replay.playing


def test_what_is_shown_is_the_sample_it_claims(replay):
    """The restored state is the recording's, so the window shows what happened rather than a re-run."""
    replay.seek_time(0.4)
    sample = replay.sample
    assert sample.index == replay.timeline.index
    assert sample.sim_time == pytest.approx(replay.timeline.time)


def test_playing_advances_and_stops_at_the_end(replay):
    replay.playing = True
    replay.tick(0.2)
    assert replay.timeline.index > 0
    replay.tick(1000.0)
    assert replay.timeline.at_end
    assert not replay.playing


def test_scrubbing_stops_playback(rec, tmp_path):
    """A scrub is a person taking over; carrying on playing would move the frame out from under them."""
    handler = ReplayKeys()
    replay = Replay(
        rec, _Handle(), state=tmp_path / "run.npz", shots=tmp_path / "shots.yaml", keys=handler
    )
    replay.playing = True
    handler.key_callback(SCRUB.keys[1].code)
    replay.tick(0.0)
    assert not replay.playing
    assert replay.timeline.index == 1


# -- writing shots ----------------------------------------------------------------------------------


def test_a_shot_lands_in_the_file_and_names_the_moment(replay, tmp_path):
    replay.seek_time(0.4)
    doc = replay.add_shot("a moment")
    assert doc["at"] == pytest.approx(replay.timeline.time)
    assert [d["id"] for d in read_shots(tmp_path / "shots.yaml")] == [doc["id"]]


def test_every_shot_keeps_its_own_id(replay):
    first = replay.add_shot("same")
    replay.seek_time(0.4)
    second = replay.add_shot("same")
    assert first["id"] != second["id"]
    assert replay.shot_ids == (first["id"], second["id"])


def test_a_shot_taken_while_following_the_recorded_camera_states_no_view(replay):
    replay.follow_recorded = True
    assert "view" not in replay.add_shot()


def test_a_shot_of_a_flown_camera_states_where_it_was(replay):
    replay.follow_recorded = False
    replay.handle.cam.distance = 7.25
    assert replay.add_shot()["view"]["distance"] == pytest.approx(7.25)


# -- a camera take ------------------------------------------------------------------------------------


def test_shift_f9_is_a_take_and_f9_a_shot(monkeypatch):
    """One key, two edges: Shift decides which. Both are debounced against auto-repeat."""
    handler = ReplayKeys()
    monkeypatch.setattr(handler, "_shift_held", lambda: True)
    handler.key_callback(SHOT.keys[0].code)
    assert handler.take_take() is True and handler.take_shot() is False
    monkeypatch.setattr(handler, "_shift_held", lambda: False)
    handler._last_shot = 0.0
    handler.key_callback(SHOT.keys[0].code)
    assert handler.take_shot() is True and handler.take_take() is False


def test_a_take_records_the_flown_camera_while_playing(replay, tmp_path):
    """Play through a take while moving the camera; the clip that lands reproduces the flight."""
    from roqsim.camera_path import CameraPath
    from roqsim.shots import render_args

    replay.follow_recorded = False
    replay.seek_time(0.2)
    assert replay.toggle_take() is None and replay.take == {}
    replay.playing = True
    for i in range(10):
        replay.handle.cam.azimuth = 90.0 + 5.0 * i
        replay.tick(0.08)  # 0.08 s of sim time per tick at 1x
    assert len(replay.take) == 10
    doc = replay.toggle_take("flight")
    assert replay.take is None
    assert doc["camera_source"] == "take" and doc["label"] == "flight"
    assert doc["from"] == pytest.approx(0.28, abs=0.05)
    assert doc["to"] > doc["from"]
    written = tmp_path / doc["camera_path"]
    assert written.exists() and written.name == f"{doc['id']}.camera.yaml"
    path = CameraPath.from_arg(str(written))
    assert path.keys == {"lookat", "distance", "azimuth", "elevation"}
    assert path.at(doc["to"])["azimuth"] == pytest.approx(135.0)
    assert path.at(doc["from"])["azimuth"] == pytest.approx(90.0)
    assert [d["id"] for d in read_shots(tmp_path / "shots.yaml")] == [doc["id"]]
    args = render_args(doc)
    assert args[args.index("--camera-path") + 1] == doc["camera_path"]
    assert "--from" in args and "--to" in args and "--view" not in args


def test_a_take_that_never_played_writes_nothing(replay, tmp_path, caplog):
    replay.toggle_take()
    replay.tick(0.05)  # paused: nothing recorded
    assert replay.toggle_take() is None
    assert not (tmp_path / "shots.yaml").exists()
    assert "never playing" in caplog.text


def test_scrubbing_back_during_a_take_overwrites_rather_than_doubles(replay):
    replay.follow_recorded = False
    replay.seek_time(0.2)
    replay.toggle_take()
    replay.playing = True
    for _ in range(5):
        replay.tick(0.08)
    replay.seek_time(0.2)
    replay.playing = True
    for _ in range(5):
        replay.tick(0.08)
    assert len(replay.take) == 5


# -- what a live run must not notice ------------------------------------------------------------------


def test_a_live_run_loads_no_gui_toolkit():
    """``roqsim sim <world>`` is the command it has always been: a replay's window stays unimported
    until a recording asks for one."""
    probe = (
        "import sys; from roqsim import runner; "
        "assert 'tkinter' not in sys.modules, sorted(sys.modules)[:0] or 'tkinter imported'; "
        "assert 'roqsim.transport_window' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_live_only_options_are_refused_by_name(tmp_path, rec, capsys):
    """Ignoring them is how a wrong command line survives in a checked-in script."""
    from roqsim import runner

    with pytest.raises(SystemExit):
        runner.main([str(tmp_path / "run.npz"), "--record", str(tmp_path / "out.npz")])
    assert "--record" in capsys.readouterr().err


# -- a world named for the rebuild ---------------------------------------------------------------------


def test_a_recording_whose_world_moved_rebuilds_from_the_world_named(rec, tmp_path):
    """The case: a world that loaded files from beside itself, replayed where they are not."""
    moved = tmp_path / "elsewhere" / "w.yaml"
    moved.parent.mkdir()
    moved.write_text(_WORLD, encoding="utf-8")
    reopened = open_recording(tmp_path / "run.npz")
    model, _ctx = reopened.build(str(moved))
    assert model.nbody == rec._model.nbody


def test_a_shot_taken_in_such_a_replay_names_the_world_for_its_render(rec, tmp_path):
    from roqsim.shots import render_args

    replay = Replay(
        rec,
        _Handle(),
        state=tmp_path / "run.npz",
        shots=tmp_path / "shots.yaml",
        world=tmp_path / "elsewhere" / "w.yaml",
    )
    doc = replay.add_shot("moved")
    assert doc["world_target"] == str(tmp_path / "elsewhere" / "w.yaml")
    assert render_args(doc)[0] == doc["world_target"] and render_args(doc)[1] == "--state"
    plain = Replay(
        rec, _Handle(), state=tmp_path / "run.npz", shots=tmp_path / "shots2.yaml"
    ).add_shot("own")
    assert "world_target" not in plain and render_args(plain)[0] == "--state"
