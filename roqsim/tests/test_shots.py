"""What a shot promises, and what ``roqsim render`` does with it.

The shot file is the only thing between picking a moment in a window and drawing it hours later in a
figure, so the tests that matter here are the ones that hold the two ends together: the ``--at`` a
shot writes must select the sample it claims, and the camera it writes must survive the command line
unchanged. Those are checked against ``roqsim render``'s own parser rather than a copy of its
grammar -- a second reading of ``--view`` is exactly the drift the frozen key set exists to prevent.
"""

from __future__ import annotations

import mujoco
import pytest
import yaml

from roqsim import render
from roqsim.capture import StateRecorder, snap_fps
from roqsim.recording import open_recording
from roqsim.shots import append_shot, read_shots, render_args, shot_document, shot_id

_WORLD = """\
sim:
{view}
components:
- dummy: {{}}
"""


class _Ctx:
    """The three members StateRecorder touches, so a test needs no Engine."""

    def __init__(self, model, data):
        self.model, self.data = model, data

    @property
    def sim_time(self) -> float:
        return float(self.data.time)


def _recording(tmp_path, view: str = ""):
    """A real recording of a core-only world, rebuilt and ready to hand out samples."""
    world = tmp_path / "w.yaml"
    world.write_text(_WORLD.format(view=view or "  pacing: asap"), encoding="utf-8")
    model, data, _ctx, _view, _cam = render.build_target(str(world), None)
    ctx = _Ctx(model, data)
    recorder = StateRecorder(
        ctx, tmp_path / "run.npz", snap_fps(25, model.opt.timestep), world=str(world)
    )
    for _ in range(600):
        mujoco.mj_step(model, data)
        recorder.sample(ctx)
    recorder.close()
    rec = open_recording(tmp_path / "run.npz")
    rec.build()
    return rec


def _camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.25, -0.5, 0.4]
    cam.distance, cam.azimuth, cam.elevation = 6.437, -37.5, -18.0
    return cam


@pytest.fixture
def rec(tmp_path):
    return _recording(tmp_path)


@pytest.fixture
def doc(rec, tmp_path):
    return shot_document(
        rec, rec.at(0.4), _camera(), state=str(tmp_path / "run.npz"), label="a moment"
    )


# -- what a shot says ------------------------------------------------------------------------------


def test_a_shot_names_the_sample_it_landed_on(rec, doc):
    """The sample's own time, not the time that was asked for: writing the request back would make
    a shot that re-renders a *different* sample every time the capture rate changes."""
    sample = rec.at(0.4)
    assert doc["at"] == pytest.approx(sample.sim_time)
    assert doc["sample_index"] == sample.index


def test_the_id_is_filename_safe_and_says_where_it_came_from(doc):
    assert doc["id"] == f"a_moment-{doc['sample_index']:04d}"


def test_two_shots_of_one_moment_get_their_own_ids(rec, tmp_path):
    first = shot_document(rec, rec.at(0.4), _camera(), state="run.npz", label="x")
    second = shot_document(
        rec, rec.at(0.4), _camera(), state="run.npz", label="x", taken=(first["id"],)
    )
    assert second["id"] != first["id"]


def test_the_world_is_recorded_as_identity(rec, doc):
    """Named so a reader knows which world this is, and never passed to the render: a recording
    rebuilds from its own resolved tree only while no target is given."""
    assert doc["world"] == rec.meta["world"]
    assert doc["world"] not in render_args(doc)


# -- what the render command line does with it -----------------------------------------------------


def test_the_at_it_writes_selects_the_sample_it_claims(rec, doc):
    """The round trip that matters: the text on the command line reproduces the sample."""
    args = render_args(doc)
    at = float(args[args.index("--at") + 1])
    assert rec.index_at(at) == doc["sample_index"]


def test_the_camera_survives_the_command_line(doc):
    """Parsed by ``roqsim render``'s own ``--view`` reader, so the two cannot drift."""
    args = render_args(doc)
    tokens = args[args.index("--view") + 1 : args.index("--size")]
    assert render.view_overrides(tokens)["sim"]["view"] == doc["view"]


def test_a_shot_taken_through_the_recorded_camera_states_no_view(rec):
    """No ``--view`` at all is what leaves the render following the camera the run was watched
    through; a view echoed back would override the very thing it was describing."""
    doc = shot_document(rec, rec.at(0.4), None, state="run.npz", label="recorded")
    assert doc["camera_source"] == "recorded"
    assert "view" not in doc
    assert "--view" not in render_args(doc)


def test_a_tracked_world_is_untracked_explicitly(tmp_path):
    """``--view`` merges over the world's own ``sim.view``, and any ``track`` target keeps the
    camera tracking with ``lookat`` ignored -- so a hand-framed shot has to turn both keys off."""
    rec = _recording(tmp_path, view="  view: {track: DummyPlugin, distance: 3.0}")
    doc = shot_document(rec, rec.at(0.4), _camera(), state="run.npz", label="framed")
    assert doc["view"]["track"] is None
    assert doc["view"]["follow_heading"] is False
    args = render_args(doc)
    tokens = args[args.index("--view") + 1 : args.index("--size")]
    assert "track=null" in tokens and "follow_heading=false" in tokens
    parsed = render.view_overrides(tokens)["sim"]["view"]
    assert parsed["track"] is None and parsed["follow_heading"] is False


def test_an_untracked_world_says_nothing_about_tracking(doc):
    assert "track" not in doc["view"]
    assert "follow_heading" not in doc["view"]


def test_no_ceiling_is_emitted_only_when_it_is_asked_for(rec):
    plain = shot_document(rec, rec.at(0.4), _camera(), state="run.npz")
    opened = shot_document(rec, rec.at(0.4), _camera(), state="run.npz", no_ceiling=True)
    assert "--no-ceiling" not in render_args(plain)
    assert "--no-ceiling" in render_args(opened)


def test_size_and_output_can_be_overridden_without_rewriting_the_shot(doc):
    """How one shot list renders a second time at a second resolution."""
    args = render_args(doc, size="3840x2160", out="big.png")
    assert args[args.index("--size") + 1] == "3840x2160"
    assert args[args.index("--out") + 1] == "big.png"


# -- the file --------------------------------------------------------------------------------------


def test_appending_keeps_every_document(tmp_path, rec):
    path = tmp_path / "shots.yaml"
    ids = []
    for index, when in enumerate((0.2, 0.4, 0.6)):
        doc = shot_document(
            rec, rec.at(when), _camera(), state="run.npz", label=f"shot {index}", taken=tuple(ids)
        )
        ids.append(doc["id"])
        assert append_shot(path, doc) == index + 1
    assert [d["id"] for d in read_shots(path)] == ids


def test_a_file_that_is_not_there_yet_holds_no_shots(tmp_path):
    assert read_shots(tmp_path / "nothing.yaml") == []


def test_a_document_from_another_schema_is_refused(tmp_path):
    """Read with the keys that happen to overlap, it would render a plausible-looking wrong moment."""
    path = tmp_path / "shots.yaml"
    path.write_text("---\n" + yaml.safe_dump({"schema": 99, "id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        read_shots(path)


def test_an_id_avoids_the_ones_already_in_the_file():
    assert shot_id("x", 3, taken=()) == "x-0003"
    assert shot_id("x", 3, taken=("x-0003",)) == "x-0003-2"
    assert shot_id("", 3, "run", taken=()) == "run-0003"


def test_a_view_can_name_the_body_a_camera_follows():
    """``track`` takes a body name, not a number -- the one view key whose value is not numeric."""
    from roqsim.shots import render_args

    doc = {
        "schema": 1,
        "id": "tracked",
        "state": "run.npz",
        "at": 3.0,
        "size": "960x540",
        "png": "out.png",
        "view": {"track": "base_link", "follow_heading": False, "distance": 9.0},
    }
    args = render_args(doc)
    view = args[args.index("--view") + 1 : args.index("--size")]
    assert view == ["track=base_link", "follow_heading=false", "distance=9.0"]


def test_a_camera_path_and_overlays_survive_the_command_line():
    """Emitted as one JSON token each, read back by the same parsers `roqsim render` uses."""
    import json

    from roqsim.camera_path import CameraPath
    from roqsim.overlays import parse_spec
    from roqsim.shots import render_args

    doc = {
        "schema": 1,
        "id": "clip",
        "state": "run.npz",
        "from": "onset",
        "to": "onset+10",
        "size": "960x540",
        "video": "clip.mp4",
        "camera_path": {"ease": "smoothstep", "keyframes": [{"t": "onset", "azimuth": 180}]},
        "overlays": ["clock", {"costmap": {"anchor": "top-right", "width": 0.3}}],
    }
    args = render_args(doc)
    path = CameraPath.from_arg(args[args.index("--camera-path") + 1])
    assert path.ease == "smoothstep" and path.moments == {"onset"}
    specs = [args[i + 1] for i, a in enumerate(args) if a == "--overlay"]
    assert [parse_spec(s) for s in specs] == [
        ("clock", {}),
        ("costmap", {"anchor": "top-right", "width": 0.3}),
    ]
    assert args[args.index("--to") + 1] == "onset+10"
    assert json.loads(specs[1]) == doc["overlays"][1]


def test_a_path_file_is_passed_as_given():
    from roqsim.shots import render_args

    doc = {
        "schema": 1,
        "id": "c",
        "state": "run.npz",
        "from": 0,
        "size": "960x540",
        "video": "c.mp4",
        "camera_path": "c.camera.yaml",
    }
    args = render_args(doc)
    assert args[args.index("--camera-path") + 1] == "c.camera.yaml"


def test_a_fixed_camera_is_one_or_the_other():
    from roqsim.shots import render_args

    doc = {
        "schema": 1,
        "id": "c",
        "state": "run.npz",
        "at": 1.0,
        "size": "960x540",
        "png": "c.png",
        "camera": "overhead",
    }
    assert "--camera" in render_args(doc)
    with pytest.raises(ValueError, match="one or the other"):
        render_args({**doc, "view": {"azimuth": 1}})
