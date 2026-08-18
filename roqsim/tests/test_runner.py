"""The runner's CLI surface: how flags map onto run()'s arguments.

These stop at the boundary -- run() is stubbed, so no world is loaded and no window is opened. That
the panel switches then reach ``launch_passive`` is :mod:`test_viewer`'s job (``ui_kwargs``), and
that ``manual_control`` actually stops a controller writing ``ctrl`` is the robot packages' (e.g.
``roqsim_manipulation``'s ``test_arm_scene``).
"""

from __future__ import annotations

import logging

import pytest

from roqsim import runner
from roqsim.plugin import PluginError


@pytest.fixture
def cli(monkeypatch):
    """Run the CLI far enough to capture the kwargs it hands to run()."""

    def invoke(*argv):
        captured = {}
        monkeypatch.setattr(runner, "run", lambda target, **kw: captured.update(kw))
        assert runner.main(["unused.yaml", *argv]) == 0
        return captured

    return invoke


# -- panel flags ---------------------------------------------------------------------------------


def test_panels_hidden_by_default(cli):
    args = cli()
    assert (args["left_ui"], args["right_ui"]) == (False, False)


def test_panel_flags(cli):
    assert cli("--right-ui")["right_ui"] is True
    assert cli("--left-ui")["left_ui"] is True


def test_panels_are_not_world_config(cli):
    """The panels are a run-level switch: they must never be written into the world."""
    assert cli("--left-ui", "--right-ui")["overrides"] == {}


# -- manual control ------------------------------------------------------------------------------


def test_manual_control_defaults_off(cli):
    assert cli()["manual_control"] is False


def test_manual_control_opens_the_slider_panel(cli):
    """The sliders live in the right panel, so the flag is useless without it -- imply it."""
    args = cli("--manual-control")
    assert args["manual_control"] is True
    assert args["right_ui"] is True


def test_manual_control_headless_is_refused():
    """No window means no sliders: nobody would drive the robot. Fail loudly instead."""
    with pytest.raises(SystemExit):
        runner.main(["unused.yaml", "--manual-control", "--headless"])


# -- transport (--ros / --no-communication) ------------------------------------------------------


def test_transport_is_off_on_both_sides_by_default(cli):
    """A world is run exactly as authored: nothing appended, nothing stripped."""
    args = cli()
    assert args["transport"] is None
    assert args["no_transport"] is False


def test_no_communication_reaches_run(cli):
    assert cli("--no-communication")["no_transport"] is True


def test_no_communication_contradicting_ros_is_refused():
    """Appending a bridge and stripping one are opposite intents; the order must not decide it."""
    for opposite in (["--ros"], ["--sim-control"], ["--tf-namespace", "robot"]):
        with pytest.raises(SystemExit):
            runner.main(["unused.yaml", "--no-communication", *opposite])


def _warnings_from(dropped: list[str]) -> str:
    """What ``_warn_no_transport`` logs, captured off the logger itself.

    Not ``caplog``: under a ROS-sourced pytest something in the ament plugin set takes the records
    before pytest's capturing handler sees them, so ``caplog.text`` comes back empty while the line
    is plainly on stderr (it costs seven other tests in this suite the same way). Attaching a handler
    directly is immune to that, and is the more direct assertion anyway.
    """
    log = logging.getLogger(runner.__name__)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    log.addHandler(handler)
    try:
        runner._warn_no_transport(dropped)
    finally:
        log.removeHandler(handler)
    return "\n".join(r.getMessage() for r in records if r.levelno >= logging.WARNING)


def test_no_communication_warns_that_the_run_is_mute():
    """The flag's whole risk is that it would otherwise be silent about being silent."""
    text = _warnings_from(["ros2_bridge", "sim_interfaces"])
    assert "ros2_bridge, sim_interfaces" in text
    assert "COMMUNICATES WITH NOTHING" in text


def test_no_communication_on_an_offline_world_says_it_changed_nothing():
    assert "nothing was dropped" in _warnings_from([])


# -- positional dispatch (_config_for_input) -----------------------------------------------------


def test_xml_runs_as_the_world():
    """A .xml is a baked scene: it becomes sim.world (as an absolute path), no plugins added."""
    cfg = runner._config_for_input("scene.xml", None)
    assert cfg.sim["world"].endswith("/scene.xml") and cfg.plugins == []


def test_model_ref_is_shown_via_spawn_model():
    """A bare reference is welded into the default empty room by spawn_model."""
    cfg = runner._config_for_input("roqsim_assets:industrial_table", None)
    assert [p.ref for p in cfg.plugins] == ["spawn_model"]
    assert cfg.plugins[0].config["model"] == "roqsim_assets:industrial_table"


def test_raw_mesh_is_refused():
    """roqsim runs worlds/scenes/models, not geometry -- point at finalize instead."""
    with pytest.raises(PluginError, match="raw meshes"):
        runner._config_for_input("prop.obj", None)


def test_overrides_reach_a_synthesized_world():
    cfg = runner._config_for_input("roqsim_assets:industrial_table", {"sim": {"pacing": "asap"}})
    assert cfg.sim.get("pacing") == "asap"


# -- recording (a run-level switch, like the panels) ----------------------------------------------


def test_recording_off_by_default(cli):
    assert cli()["record"] is None


def test_record_requires_a_path(monkeypatch):
    """A recording is an artifact you name again later, and a default would overwrite the last run's."""
    monkeypatch.setattr(runner, "run", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        runner.main(["w.yaml", "--record"])


def test_f9_still_works_with_no_record_flag(cli):
    """The default path is still needed: F9 must start a take in a windowed run given no flags at all."""
    from roqsim.runner import _DEFAULT_RECORD

    assert cli()["record"] is None
    assert _DEFAULT_RECORD.endswith(".npz")


def test_record_takes_a_path(cli):
    assert cli("--record", "/tmp/x.npz")["record"] == "/tmp/x.npz"


def test_capture_fps_defaults_and_passes_through(cli):
    from roqsim.capture import DEFAULT_FPS

    assert cli()["capture_fps"] == DEFAULT_FPS
    assert cli("--capture-fps", "500/17")["capture_fps"] == "500/17"


def test_recording_is_not_world_config(cli):
    """The twin of test_panels_are_not_world_config: capture describes a session, not an experiment."""
    assert cli("--record", "/tmp/x.npz", "--capture-fps", "10")["overrides"] == {}


@pytest.mark.parametrize("world", ["world.yaml", "scene.xml", "w.yml"])
def test_record_refuses_to_eat_the_world(monkeypatch, world):
    """`roqsim sim --record world.yaml` -- argparse reports a *missing target*, which is useless.

    The check must run before parsing: by the time argparse has assigned world.yaml to --record there
    is no positional left, so a post-parse guard can never fire for this shape.
    """
    monkeypatch.setattr(runner, "run", lambda *a, **k: None)
    with pytest.raises(SystemExit) as err:
        runner.main(["--record", world])
    assert "takes an output path" in str(err.value)
    assert "Put the world first" in str(err.value)


def test_record_after_the_target_is_fine(cli):
    """The normal spelling must not be caught by that guard."""
    assert cli("--record", "/tmp/out.npz")["record"] == "/tmp/out.npz"


def test_video_off_by_default(cli):
    assert cli()["video"] is None


def test_video_bare_flag_uses_a_default_path(cli):
    from roqsim.runner import _DEFAULT_VIDEO

    assert cli("--video")["video"] == _DEFAULT_VIDEO


def test_video_has_no_fps_or_speed_flag():
    """Deliberate: the file is encoded at --capture-fps, so 1 s of file is 1 s of sim time.

    A run's video is a *record* — one that plays at an arbitrary multiple is not comparable with another
    run's. Other speeds are a presentation choice and belong to `roqsim render --state --speed`.
    """
    import argparse

    from roqsim import runner

    parser_flags = set()

    real = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kwargs):
        parser_flags.update(a for a in args if isinstance(a, str) and a.startswith("--"))
        return real(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = spy
    try:
        with pytest.raises(SystemExit):
            runner.main(["--help"])
    finally:
        argparse.ArgumentParser.add_argument = real
    assert "--video" in parser_flags
    assert "--video-fps" not in parser_flags and "--video-speed" not in parser_flags


@pytest.mark.parametrize("world", ["world.yaml", "scene.xml"])
def test_video_refuses_to_eat_the_world(monkeypatch, world):
    monkeypatch.setattr(runner, "run", lambda *a, **k: None)
    with pytest.raises(SystemExit) as err:
        runner.main(["--video", world])
    assert "takes an output path" in str(err.value)


def test_video_implies_a_recording_beside_it(monkeypatch, tmp_path):
    """--video needs a recording; putting it beside the video avoids a second convention."""
    seen = {}

    class _Rec:
        def __init__(self, ctx, path, rate, **kw):
            seen["path"] = str(path)

        def sample(self, *a, **k):
            return False

        def close(self):
            return None

    monkeypatch.setattr("roqsim.runner.StateRecorder", _Rec)
    monkeypatch.setattr("roqsim.runner._run_headless", lambda *a, **k: None)
    scene = tmp_path / "s.xml"
    scene.write_text("<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
    runner.run(str(scene), headless=True, max_steps=1, video=str(tmp_path / "out.webm"))
    assert seen["path"] == str(tmp_path / "out.npz")


# -- graceful stop -------------------------------------------------------------------------------
#
# A supervised run ends on SIGTERM, not Ctrl+C: a container teardown, `docker stop`, a Kubernetes
# eviction and a campaign timeout all send it. Its *default* action kills the process outright, so no
# `finally` runs and the recording and the run capture are both lost -- which is how one campaign
# finished 1/1 clean and produced no `run.npz` at all. These pin that both signals now flip run-control
# instead, and that the driver leaves the process's handlers as it found them.


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGINT"])
def test_a_stop_signal_asks_the_loop_to_quit_instead_of_raising(signame, caplog):
    import logging
    import os
    import signal

    from roqsim import control as ctl

    caplog.set_level(logging.INFO)
    control = ctl.RunControl()
    with runner._graceful_stop(control, logging.getLogger("test")):
        os.kill(os.getpid(), getattr(signal, signame))
        assert control.state == ctl.QUITTING, f"{signame} did not reach run-control"
    assert "shutting down" in caplog.text


def test_the_previous_handlers_are_restored():
    """The driver is embeddable, so it must not leave its handlers installed on the way out."""
    import logging
    import signal

    from roqsim import control as ctl

    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with runner._graceful_stop(ctl.RunControl(), logging.getLogger("test")):
        assert signal.getsignal(signal.SIGTERM) not in before.values()
    assert {s: signal.getsignal(s) for s in before} == before


def test_off_the_main_thread_it_is_a_no_op():
    """signal.signal only works on the main thread; an embedded driver must still run there."""
    import logging
    import threading

    from roqsim import control as ctl

    ran = []

    def body():
        with runner._graceful_stop(ctl.RunControl(), logging.getLogger("test")):
            ran.append(True)

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    assert ran == [True], "the context manager must yield off the main thread, not raise"


def test_a_c_installed_handler_does_not_break_the_teardown():
    """``getsignal`` returns None for a handler installed from C -- a process embedding rclpy and
    MuJoCo may have one. Passing None back to ``signal.signal`` raises TypeError, and this restore
    runs in the driver's teardown, where raising would mask the run's outcome and lose the recording."""
    import logging
    import signal

    from roqsim import control as ctl

    real = signal.getsignal(signal.SIGTERM)
    try:
        with runner._graceful_stop(ctl.RunControl(), logging.getLogger("test")) as _:
            # Simulate what the C-installed case looks like on the way out.
            runner._restore(signal.SIGTERM, None)  # must not raise
    finally:
        signal.signal(signal.SIGTERM, real)


def test_int_and_term_together_are_one_stop_not_an_escalation():
    """A supervisor delivers both for one stop (ros2 launch forwards SIGINT *and* SIGTERM).

    On a shared hit counter the second of the pair took the "signal again to force" branch and raised
    KeyboardInterrupt -- observed landing inside the capture export, where `except Exception` cannot
    catch a BaseException, so it escaped and cost the run its capture. Two different signals are one
    request; only a repeat of the *same* one is somebody insisting.
    """
    import logging
    import os
    import signal

    from roqsim import control as ctl

    control = ctl.RunControl()
    with runner._graceful_stop(control, logging.getLogger("test")):
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)  # must NOT raise
        assert control.state == ctl.QUITTING


def test_the_same_signal_twice_still_forces():
    """The interactive affordance survives: a second Ctrl+C means the user is done waiting."""
    import logging
    import os
    import signal

    from roqsim import control as ctl

    with pytest.raises(KeyboardInterrupt):
        with runner._graceful_stop(ctl.RunControl(), logging.getLogger("test")):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)


def test_the_teardown_runs_inside_the_protected_window(monkeypatch, tmp_path):
    """A SIGTERM during the flush must not kill the process: the handlers outlive the run loop.

    This is the ordering, not the signal: if `run()` restored the defaults before calling
    `recorder.close()`, a TERM arriving mid-write -- which is when a supervisor sends it -- would end the
    process with a partial file. Asserted by sending one from inside close() itself.
    """
    import os
    import signal

    closed = []

    class _Rec:
        frames = 0

        def __init__(self, ctx, path, rate, **kw):
            pass

        def sample(self, *a, **k):
            return False

        def close(self):
            os.kill(os.getpid(), signal.SIGTERM)  # would be fatal outside the window
            closed.append(True)
            return None

    monkeypatch.setattr("roqsim.runner.StateRecorder", _Rec)
    monkeypatch.setattr("roqsim.runner._run_headless", lambda *a, **k: None)
    scene = tmp_path / "s.xml"
    scene.write_text("<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
    runner.run(str(scene), headless=True, max_steps=1, record=str(tmp_path / "r.npz"))
    assert closed == [True], "close() did not complete -- the flush was outside the signal window"
