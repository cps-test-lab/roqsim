"""``roqsim render``'s contracts: the target grammar, the camera vocabulary, and its output record.

GL-free by design -- every test here stops before a frame is rasterised, so the suite runs on a machine
with no display and no GPU. The one test that does render is opt-in at the bottom and skips itself when
there is no usable offscreen backend.
"""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np
import pytest

from roqsim import render
from roqsim.config import _VIEW_KEYS, load_config_from_dict
from roqsim.plugin import PluginError

# -- --size --------------------------------------------------------------------------------------


def test_size_parsed():
    assert render.parse_size("960x540") == (960, 540)


def test_size_is_evened_not_refused():
    """yuv420p needs even dimensions, and a one-pixel nudge is not what the caller was expressing."""
    assert render.parse_size("641x361") == (642, 362)


@pytest.mark.parametrize("bad", ["nope", "960", "axb", "960x", ""])
def test_size_garbage_refused(bad):
    with pytest.raises(render.RenderError, match="expected WxH"):
        render.parse_size(bad)


def test_size_too_small_refused():
    with pytest.raises(render.RenderError, match="at least 16"):
        render.parse_size("8x8")


# -- --view: sugar over the world's own sim.view --------------------------------------------------


def test_view_lowers_to_sim_view_overrides():
    """--view must go through the same override path --set uses, so sim.view has one validator."""
    assert render.view_overrides(["azimuth=90", "distance=30"]) == {
        "sim": {"view": {"azimuth": 90, "distance": 30}}
    }


def test_view_is_partial():
    """Only the given keys appear, so the rest keep whatever the world stated."""
    assert render.view_overrides(["elevation=-85"])["sim"]["view"] == {"elevation": -85}


def test_view_parses_types_like_set_does():
    got = render.view_overrides(["lookat=[1, 2, 0]", "follow_heading=true"])["sim"]["view"]
    assert got == {"lookat": [1, 2, 0], "follow_heading": True}


def test_view_accepts_a_comma_separated_lookat():
    """`lookat=1,2,0` is how a three-vector is written on a command line, and it must reach sim.view
    as three numbers. YAML reads a bare comma list as one scalar string, which the camera then
    iterated character by character and died on `float('.')`."""
    got = render.view_overrides(["lookat=2.5,1.0,0"])["sim"]["view"]
    assert got == {"lookat": [2.5, 1.0, 0]}


def test_view_accepts_a_space_separated_lookat():
    """MJCF spells a three-vector `-3.2 -1.3 1.9`, so a caller quoting one -- or pasting a pose back out
    of a render's own camera record -- must not be told it is not three numbers."""
    got = render.view_overrides(["lookat=-3.2 -1.3 1.9"])["sim"]["view"]
    assert got == {"lookat": [-3.2, -1.3, 1.9]}


def test_view_rejoins_a_lookat_the_shell_split():
    """Unquoted, `--view lookat=-3.2 -1.3 1.9` arrives as three tokens; they are one value."""
    got = render.view_overrides(["lookat=-3.2", "-1.3", "1.9", "distance=2"])["sim"]["view"]
    assert got == {"lookat": [-3.2, -1.3, 1.9], "distance": 2}


def test_view_leaves_a_malformed_lookat_to_the_one_validator():
    """Not parseable as numbers -> untouched, so sim.view's validator rejects it quoting what was typed."""
    assert render.view_overrides(["lookat=a b c"])["sim"]["view"] == {"lookat": "a b c"}
    with pytest.raises(PluginError, match="sim.view.lookat: expected 3 numbers"):
        load_config_from_dict({"sim": {"view": {"lookat": "a b c"}}, "plugins": []})


def test_view_unknown_key_names_the_frozen_set():
    with pytest.raises(render.RenderError) as err:
        render.view_overrides(["nonsense=1"])
    message = str(err.value)
    assert "nonsense" in message
    # The message must list the same vocabulary a world YAML is held to, or the two drift.
    for key in _VIEW_KEYS:
        assert key in message


def test_view_without_value_refused_and_names_the_likely_cause():
    """A greedy `--view` can swallow the positional target, so say that rather than only the grammar."""
    with pytest.raises(render.RenderError) as err:
        render.view_overrides(["world.yaml"])
    assert "KEY=VALUE" in str(err.value) and "put it before the flag" in str(err.value)


def test_several_view_keys_share_one_flag():
    """`--view azimuth=90 distance=30` is how a camera is naturally written; nargs="+" allows it."""
    argv = ["w.yaml", "--view", "azimuth=90", "distance=30"]
    parsed = _parse(argv)
    assert parsed.view == ["azimuth=90", "distance=30"]


def test_repeated_view_flags_accumulate():
    parsed = _parse(["w.yaml", "--view", "azimuth=90", "--view", "distance=30"])
    assert parsed.view == ["azimuth=90", "distance=30"]


def _parse(argv):
    """Drive main()'s parser without running a render, to test the flag grammar itself."""
    captured = {}

    class _Args:
        pass

    import argparse as _ap

    real = _ap.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        captured["ns"] = ns
        raise _Stop

    _ap.ArgumentParser.parse_args = spy
    try:
        with pytest.raises(_Stop):
            render.main(argv)
    finally:
        _ap.ArgumentParser.parse_args = real
    return captured["ns"]


def test_view_empty_is_no_override():
    assert render.view_overrides(None) == {} and render.view_overrides([]) == {}


def test_unknown_view_key_reaches_the_world_validator_too(tmp_path):
    """Belt and braces: even if the CLI check were bypassed, load_config still refuses the key."""
    world = tmp_path / "w.yaml"
    world.write_text("sim: {view: {nonsense: 1}}\nplugins: []\n")
    with pytest.raises(PluginError, match="sim.view: unknown key"):
        from roqsim.config import load_config

        load_config(str(world))


# -- the target grammar --------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["p.obj", "P.STL", "a/b/c.glb"])
def test_meshes_take_the_mesh_branch(target):
    assert render.is_mesh(target)


@pytest.mark.parametrize(
    "target", ["w.yaml", "w.yml", "s.xml", "roqsim_assets:industrial_table", "table"]
)
def test_non_meshes_do_not(target):
    assert not render.is_mesh(target)


def test_render_dispatches_through_the_runners_grammar(monkeypatch, tmp_path):
    """A non-mesh target must go through config_for_input, so `roqsim render` and `roqsim sim` agree."""
    seen = {}

    def fake_config_for_input(target, overrides):
        seen["target"], seen["overrides"] = target, overrides
        raise _Stop

    monkeypatch.setattr("roqsim.runner.config_for_input", fake_config_for_input)
    with pytest.raises(_Stop):
        render.build_target("roqsim_assets:industrial_table", {"sim": {"view": {"azimuth": 1}}})
    assert seen["target"] == "roqsim_assets:industrial_table"
    assert seen["overrides"] == {"sim": {"view": {"azimuth": 1}}}


class _Stop(Exception):
    """Sentinel: proves the call was made without needing a world to compile."""


# -- output type comes from the extension --------------------------------------------------------


def test_video_extension_without_state_is_refused(tmp_path):
    with pytest.raises(render.RenderError, match="needs a recording"):
        render.render_target("whatever.yaml", tmp_path / "out.webm")


def test_unknown_extension_is_refused(tmp_path):
    with pytest.raises(render.RenderError, match="unknown extension"):
        render.render_target("whatever.yaml", tmp_path / "out.tiff")


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root ignores the permission bits, so an unwritable directory cannot be constructed "
    "(the ros CI job runs as root in a container; the check itself is exercised in the core job)",
)
def test_unwritable_output_directory_is_refused(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(render.RenderError, match="not writable"):
            render.render_target("whatever.yaml", locked / "out.png")
    finally:
        locked.chmod(0o700)


def test_headless_without_a_backend_names_both_options(tmp_path, monkeypatch):
    """The verdict comes from the backend mujoco BOUND, not from DISPLAY.

    ``DISPLAY`` is deliberately set here, and must not rescue the render. The old spelling was
    ``not os.environ.get("MUJOCO_GL") and not has_display()``, which a container defeats twice
    over: the RoboVAST base image exports ``DISPLAY=:0`` with no X server behind it, and
    ``MUJOCO_GL`` can be set long after ``import mujoco`` already bound something else. Both let a
    doomed render past this guard and into ``mujoco.FatalError: gladLoadGL error``.
    """
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("roqsim.rendering.bound_gl_backend", lambda: "glfw")
    with pytest.raises(render.RenderError) as err:
        render.render_target("whatever.yaml", tmp_path / "out.png")
    assert "egl" in str(err.value) and "osmesa" in str(err.value)


def test_check_creates_nothing(tmp_path, monkeypatch):
    """A dry run must not create the directory it was only asked to validate."""
    monkeypatch.setenv("MUJOCO_GL", "egl")
    target = tmp_path / "absent" / "out.png"
    with pytest.raises(_Stop):
        monkeypatch.setattr(render, "build_target", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
        render.render_target("whatever.yaml", target, check=True)
    assert not target.parent.exists()


# -- --no-ceiling ---------------------------------------------------------------------------------


def _cfg(text: str, tmp_path):
    from roqsim.config import load_config

    world = tmp_path / "w.yaml"
    world.write_text(text)
    return load_config(str(world))


def test_no_ceiling_opens_the_roof(tmp_path):
    """`keep` means *keep the ceiling*, so opening the roof sets it False.

    Not the reserved `enabled:` sibling: this plugin works by REMOVING geometry, so turning the
    component off would leave the ceiling standing -- the opposite of what the flag asks for.
    """
    cfg = _cfg("components:\n  - ceiling: {above_z: 2.6, keep: true}\n", tmp_path)
    assert render._disable_ceiling(cfg) is True
    assert cfg.plugins[0].config["keep"] is False
    assert cfg.plugins[0].enabled is True


def test_no_ceiling_is_a_noop_without_a_ceiling_plugin(tmp_path):
    """A ceiling-less world already satisfies --no-ceiling; failing there would be unhelpful.

    Applying it as a --set override *would* fail: an override matching no component is refused, and
    the message lists what the document has -- several hundred names in a populated scene. That is
    right for --set (where a typo is otherwise silent) and wrong for this flag.
    """
    cfg = _cfg("plugins:\n  - dummy: {}\n", tmp_path)
    assert render._disable_ceiling(cfg) is False
    assert cfg.plugins[0].config == {}


def test_no_ceiling_matches_a_named_instance(tmp_path):
    cfg = _cfg("plugins:\n  - ceiling: {above_z: 2.0}\n    name: ceiling\n", tmp_path)
    assert render._disable_ceiling(cfg) is True


# -- the home keyframe, shared with make thumbnails ----------------------------------------------

_TWO_POSE_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body name="arm" pos="0 0 .5">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".05" fromto="0 0 0 .5 0 0"/>
    </body>
  </worldbody>
  <keyframe><key name="home" qpos="1.2"/></keyframe>
</mujoco>
"""


def test_home_keyframe_wins_over_qpos0():
    """A model that declares `home` must render in that pose, not qpos0.

    This is why `roqsim render` and `make thumbnails` share the helper: for an articulated robot the two
    poses look nothing alike (the TIAGo Pro's arms stick straight out at qpos0).
    """
    model = mujoco.MjModel.from_xml_string(_TWO_POSE_XML)
    data = mujoco.MjData(model)
    render.reset_to_home(model, data)
    assert data.qpos[0] == pytest.approx(1.2)


def test_no_home_keyframe_falls_back_to_qpos0():
    model = mujoco.MjModel.from_xml_string(_TWO_POSE_XML.replace('name="home"', 'name="other"'))
    data = mujoco.MjData(model)
    render.reset_to_home(model, data)
    assert data.qpos[0] == pytest.approx(0.0)


def test_reset_to_home_leaves_kinematics_current():
    """Callers frame a camera off geom world poses straight after, so mj_forward must have run."""
    model = mujoco.MjModel.from_xml_string(_TWO_POSE_XML)
    data = mujoco.MjData(model)
    render.reset_to_home(model, data)
    assert not np.allclose(data.geom_xpos[1], 0.0)


# -- the JSON record -----------------------------------------------------------------------------


def test_camera_record_uses_the_view_vocabulary():
    """The reported camera must be re-enterable as --view, or the loop back into a scripted render breaks."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.0, 2.0, 3.0]
    cam.distance, cam.azimuth, cam.elevation = 12.5, 90.0, -45.0
    record = render._camera_record(cam)
    assert set(record) <= set(_VIEW_KEYS)
    assert record == {
        "lookat": [1.0, 2.0, 3.0],
        "distance": 12.5,
        "azimuth": 90.0,
        "elevation": -45.0,
    }


def test_camera_record_of_a_fixed_camera():
    assert render._camera_record("head_cam") == {"fixed": "head_cam"}


def test_cli_prints_exactly_one_json_line(monkeypatch, capsys):
    """Stdout is a machine contract: one parseable line, with logs on stderr."""
    monkeypatch.setattr(
        render, "render_target", lambda *a, **k: {"path": "/tmp/x.png", "rendered": True}
    )
    assert render.main(["w.yaml", "--out", "/tmp/x.png"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert json.loads(out[0])["path"] == "/tmp/x.png"


def test_cli_reports_errors_on_stderr_with_a_nonzero_code(monkeypatch, capsys):
    def boom(*a, **k):
        raise render.RenderError("nope")

    monkeypatch.setattr(render, "render_target", boom)
    assert render.main(["w.yaml"]) != 0
    captured = capsys.readouterr()
    assert captured.out == "" and "nope" in captured.err


def test_cli_refuses_an_output_file_as_the_target(capsys):
    with pytest.raises(SystemExit):
        render.main(["shot.png"])
    assert "looks like an output file" in capsys.readouterr().err


def test_show_is_off_by_default(monkeypatch):
    """`roqsim render` must stay headless and machine-callable: the MCP tool shells out to it."""
    monkeypatch.setattr(render, "render_target", lambda *a, **k: {"path": "/x", "rendered": True})
    calls = []
    monkeypatch.setattr(render, "_open_in_viewer", lambda p: calls.append(p))
    render.main(["w.yaml"])
    assert calls == []
    render.main(["w.yaml", "--show"])
    assert len(calls) == 1


# -- one opt-in end-to-end render ----------------------------------------------------------------

_SMOKE_XML = """
<mujoco>
  <worldbody>
    <light pos="1 -1 2" dir="-1 1 -2"/>
    <geom type="plane" size="3 3 .1" rgba=".4 .4 .5 1"/>
    <body pos="0 0 .3"><geom type="box" size=".2 .2 .3" rgba=".8 .3 .2 1"/></body>
  </worldbody>
</mujoco>
"""


def test_renders_a_real_frame(tmp_path, monkeypatch):
    """Deliberately a two-geom .xml, not the `dummy` world: an empty model gives a degenerate camera."""
    monkeypatch.setenv("MUJOCO_GL", __import__("os").environ.get("MUJOCO_GL", "egl"))
    scene = tmp_path / "scene.xml"
    scene.write_text(_SMOKE_XML)
    out = tmp_path / "shot.png"
    try:
        record = render.render_target(str(scene), out, size="160x120")
    except render.RenderError as err:
        pytest.skip(f"no usable offscreen GL here: {err}")
    assert record["rendered"] and (record["width"], record["height"]) == (160, 120)
    from PIL import Image

    frame = np.asarray(Image.open(out).convert("RGB"))
    assert frame.shape == (120, 160, 3)
    assert frame.std() > 10, "rendered frame is uniform -- nothing was drawn"


# -- rendering from a recording -------------------------------------------------------------------


@pytest.fixture
def a_recording(tmp_path, monkeypatch):
    """A real recording of a two-geom world, so the render path can be tested end to end."""
    monkeypatch.setenv("MUJOCO_GL", __import__("os").environ.get("MUJOCO_GL", "egl"))
    scene = tmp_path / "scene.xml"
    scene.write_text(_SMOKE_XML.replace("<worldbody>", '<option timestep="0.002"/>\n  <worldbody>'))
    from roqsim.runner import run

    out = tmp_path / "run.npz"
    run(str(scene), headless=True, pacing="asap", seconds=2.0, record=str(out), capture_fps=25)
    return str(scene), out


def test_video_needs_a_recording(tmp_path):
    with pytest.raises(render.RenderError, match="needs a recording"):
        render.render_target("w.yaml", tmp_path / "x.webm")


def test_at_needs_a_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(render.RenderError, match="need --state"):
        render.render_target("w.yaml", tmp_path / "x.png", at=1.0)


def test_at_and_a_range_are_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(render.RenderError, match="one or the other"):
        render.render_target(None, tmp_path / "x.png", state="r.npz", at=1.0, start=0.0)


def test_camera_cannot_combine_with_view_or_focus(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    for extra in ({"view": ["azimuth=1"]}, {"focus": ["robot"]}):
        with pytest.raises(render.RenderError, match="cannot be combined"):
            render.render_target("w.yaml", tmp_path / "x.png", camera="cam", **extra)


def test_nothing_to_render_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(render.RenderError, match="nothing to render"):
        render.render_target(None, tmp_path / "x.png")


def test_the_world_target_is_optional_with_state(a_recording, tmp_path):
    """The provenance names the world, so a caller need not repeat what the file already knows."""
    _scene, npz = a_recording
    record = render.render_target(None, tmp_path / "o.png", size="120x90", state=npz)
    assert record["rendered"] and record["sample_index"] >= 0


def test_at_reports_which_sample_it_landed_on(a_recording, tmp_path):
    _scene, npz = a_recording
    record = render.render_target(None, tmp_path / "o.png", size="120x90", state=npz, at=1.0)
    assert record["requested_at"] == 1.0
    assert abs(record["at_error"]) <= 0.02 + 1e-9  # within one 25 fps period
    assert record["sim_time"] == pytest.approx(1.0, abs=0.02)


def test_state_without_at_renders_the_last_sample(a_recording, tmp_path):
    _scene, npz = a_recording
    record = render.render_target(None, tmp_path / "o.png", size="120x90", state=npz)
    assert record["requested_at"] is None and record["at_error"] is None
    from roqsim.recording import open_recording

    assert record["sample_index"] == len(open_recording(npz)) - 1


def test_out_of_range_at_is_refused(a_recording, tmp_path):
    from roqsim.recording import RecordingError

    _scene, npz = a_recording
    with pytest.raises(RecordingError, match="outside this recording"):
        render.render_target(None, tmp_path / "o.png", size="120x90", state=npz, at=999.0)


def test_a_replay_uses_the_worlds_own_camera(a_recording, tmp_path):
    """Rendering world X and replaying a recording of world X must look the same.

    Without this the replay auto-frames, and the same world looks different depending on how it was
    reached -- which is exactly the kind of inconsistency nobody would think to check for.
    """
    scene, npz = a_recording
    live = render.render_target(scene, tmp_path / "a.png", size="120x90")
    replay = render.render_target(None, tmp_path / "b.png", size="120x90", state=npz)
    assert live["camera"] == replay["camera"]


def test_a_video_has_one_frame_per_sample(a_recording, tmp_path):
    """The invariant that makes --fps safe: it changes the declared rate, never which samples are used."""
    from roqsim.recording import open_recording

    _scene, npz = a_recording
    samples = len(open_recording(npz))
    for kwargs in ({}, {"speed": 4.0}, {"fps": "100"}):
        out = tmp_path / f"v{len(kwargs)}{kwargs.get('fps', '')}.webm"
        record = render.render_target(None, out, size="96x64", state=npz, **kwargs)
        assert record["frames"] == samples, f"{kwargs} changed the frame count"


def test_speed_is_reported(a_recording, tmp_path):
    _scene, npz = a_recording
    record = render.render_target(None, tmp_path / "v.webm", size="96x64", state=npz, speed=4.0)
    assert record["speed"] == 4.0
    assert record["declared_fps"] == pytest.approx(100.0)


def test_the_encoder_gets_an_exact_rational(monkeypatch, tmp_path):
    """A 30 fps request on a 0.002 s world is really 500/17; -r 29.41 would drift."""
    seen = {}

    def fake_encode(frames, out, fps, size):
        seen["fps"] = fps
        for _ in frames:
            pass

    monkeypatch.setattr(render, "encode_frames", fake_encode)

    class _Rec:
        from fractions import Fraction as _F

        fps = _F(500, 17)

        def describe(self):
            return {}

        def range(self, a, b):
            return iter(())

    render._render_video(
        _Rec(), None, None, tmp_path / "x.webm", (64, 64), None, None, None, None, None, None, None
    )
    assert seen["fps"] == "500/17"


def test_an_unknown_fixed_camera_lists_what_the_world_has():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="head"/><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    with pytest.raises(render.RenderError) as err:
        render._fixed_camera(model, "nope")
    assert "head" in str(err.value)


def test_a_known_fixed_camera_resolves():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="head"/><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    assert render._fixed_camera(model, "head") == "head"


# -- progress, and telling the caller which moment they got ----------------------------------------


def test_progress_stays_sparse_in_a_log(monkeypatch, caplog):
    """A log wants a handful of durable lines, not one per frame."""
    import logging as logging_mod

    monkeypatch.setattr(render.sys.stderr, "isatty", lambda: False, raising=False)
    progress = render._Progress(200, "rendering x.webm", render.log)
    with caplog.at_level(logging_mod.INFO):
        for _ in range(200):
            progress.tick()
        progress.done()
    lines = [r for r in caplog.records if "frames" in r.getMessage()]
    assert 1 <= len(lines) <= 5, f"expected a handful of lines for 200 frames, got {len(lines)}"


def test_progress_rewrites_one_line_on_a_tty(monkeypatch):
    """A person watching wants it to move, without filling the scrollback."""
    written = []

    class _FakeErr:
        @staticmethod
        def isatty():
            return True

        @staticmethod
        def write(text):
            written.append(text)

        @staticmethod
        def flush():
            pass

    monkeypatch.setattr(render.sys, "stderr", _FakeErr)
    progress = render._Progress(10, "rendering x.webm", render.log)
    for _ in range(10):
        progress.tick()
    progress.done()
    assert all(t.startswith("\r") for t in written), "every update must rewrite the same line"
    assert len(written) == 11  # ten ticks plus the clear
    assert written[-1].strip() == "", (
        "the line must be cleared so it cannot collide with later output"
    )


def test_progress_survives_an_unknown_total(monkeypatch, caplog):
    monkeypatch.setattr(render.sys.stderr, "isatty", lambda: False, raising=False)
    progress = render._Progress(None, "x", render.log)
    progress.tick()
    progress.done()  # must not raise


def test_frames_in_range_counts_the_samples(a_recording):
    from roqsim.recording import open_recording

    _scene, npz = a_recording
    rec = open_recording(npz)
    assert render._frames_in_range(rec, None, None) == len(rec)
    assert render._frames_in_range(rec, 0.0, 0.5) < len(rec)


def test_the_last_sample_default_is_announced(a_recording, tmp_path, caplog):
    """ "The last sample" is a choice the caller did not make, so it must not be silent."""
    import logging as logging_mod

    _scene, npz = a_recording
    with caplog.at_level(logging_mod.INFO):
        render.render_target(None, tmp_path / "o.png", size="64x48", state=npz)
    assert "LAST" in caplog.text and "--at" in caplog.text


def test_an_explicit_at_is_not_announced(a_recording, tmp_path, caplog):
    import logging as logging_mod

    _scene, npz = a_recording
    with caplog.at_level(logging_mod.INFO):
        render.render_target(None, tmp_path / "o.png", size="64x48", state=npz, at=0.5)
    assert "LAST" not in caplog.text


# -- the preview light -----------------------------------------------------------------------------
#
# GL-free like the rest of this file: the fix is a change to the light's pose, so it is asserted on the
# pose rather than on pixels. What it prevents (shadow acne combing a thin visual shell) is documented
# on `render.tilt_preview_light`.

_LIGHTS = """
<mujoco>
  <worldbody>
    <light name="ceiling" pos="0 0 2.5" dir="0 0 -1"/>
    <light name="aimed" pos="2 -2 3" dir="-0.4082 0.4082 -0.8165"/>
    <geom type="plane" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""


def _lit():
    model = mujoco.MjModel.from_xml_string(_LIGHTS)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_a_straight_down_preview_light_is_tilted_off_vertical():
    model, data = _lit()
    assert render.tilt_preview_light(model, data) == 1
    i = model.light("ceiling").id
    direction = np.asarray(model.light_dir[i])
    assert -direction[2] < 0.99  # no longer parallel to a robot's vertical walls
    elevation = np.degrees(np.arcsin(-direction[2] / np.linalg.norm(direction)))
    assert 30.0 < elevation < 60.0
    assert model.light_pos[i][2] == pytest.approx(2.5)  # same height, moved sideways


def test_a_light_the_world_aimed_is_left_alone():
    model, data = _lit()
    before = np.array(model.light_dir[model.light("aimed").id])
    render.tilt_preview_light(model, data)
    assert np.allclose(model.light_dir[model.light("aimed").id], before)


def test_the_tilt_reaches_the_renderer():
    """The renderer reads ``data.light_xdir``, so a model-only edit would be invisible."""
    model, data = _lit()
    render.tilt_preview_light(model, data)
    i = model.light("ceiling").id
    assert np.allclose(data.light_xdir[i], model.light_dir[i], atol=1e-6)


def test_tilting_is_idempotent():
    model, data = _lit()
    render.tilt_preview_light(model, data)
    after_once = np.array(model.light_dir[model.light("ceiling").id])
    assert render.tilt_preview_light(model, data) == 0
    assert np.allclose(model.light_dir[model.light("ceiling").id], after_once)
