"""Standalone driver: owns the loop, pacing, and the optional interactive viewer.

    roqsim sim world.yaml                   # windowed, real-time
    roqsim sim scene.xml                    # a baked MJCF scene, run as the world
    roqsim sim roqsim_assets:industrial_table  # show one model on the floor of an empty room
    roqsim sim world.yaml --headless --pacing asap --steps 1000 --profile
    roqsim sim world.yaml --right-ui        # open MuJoCo's control panel
    roqsim sim world.yaml --manual-control  # ... and drive the robot with its sliders
    roqsim sim world.yaml --set plugins.floorplan.size=4.0   # override world values
    roqsim sim world.yaml --record run.npz                  # record state for later rendering
    roqsim sim world.yaml --video run.webm                  # ... and render it when the run ends

The positional argument is the thing to run, dispatched by shape (see :func:`config_for_input`):
a ``.yaml`` world, a ``.xml`` MJCF scene, or a model reference (``<pkg>:<name>``, a bundled model
name, or a path) shown on its own in an empty room.

This is one of two drivers (the other is :mod:`roqsim.scenario_adapter`); both wrap the same
:class:`Engine`.

Recording is the driver's own: ``--record`` samples MuJoCo state into a ``.npz`` (see
:class:`roqsim.capture.StateRecorder`), and every image is rendered *afterwards* from that file by
``roqsim render``. Capture is a session concern, not an experiment one — the same footing as the panel
switches below — so it is a run-level flag and deliberately not world-YAML config. A render costs 41
physics steps and a state sample a fiftieth of one, which is why nothing is drawn while the loop runs.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path

from . import control as ctl
from . import logging_setup
from .capture import (
    DEFAULT_FPS,
    CaptureError,
    RecordToggle,
    StateRecorder,
    TakeRecorder,
    env_flag,
    parse_fps,
    snap_fps,
)
from .clock import Pacer
from .config import (
    apply_overrides,
    deep_merge,
    drop_transport,
    load_config,
    load_config_from_dict,
    overrides_from_dotlist,
    overrides_from_files,
    with_transport,
)
from .engine import Engine
from .gl import select_offscreen_gl
from .models import ModelError
from .plugin import PluginError
from .splash import clear_loading_overlay, show_loading_overlay
from .view_save import SaveViewKey, save_current_view
from .viewer import (
    GL_HELP,
    DisplayError,
    ViewError,
    apply_walk,
    close_viewer,
    has_display,
    launch_viewer,
    prepare_viewer_gl,
    setup_camera,
)
from .window_title import retitle_window_async
from .world import resolve_world_yaml_ref

log = logging.getLogger(__name__)

#: Mesh extensions we deliberately refuse: roqsim runs worlds/scenes/models, not raw geometry.
#: Finalizing a mesh into a model (roqsim_assets/tools/finalize_mujoco.py) is a separate step.
_MESH_EXT = (".obj", ".stl", ".glb", ".gltf", ".fbx", ".dae", ".ply")

# Backward-compatible alias: the message text now lives in roqsim.viewer.
_GL_HELP = GL_HELP

#: Interactive-viewer render rate (Hz), decoupled from the physics step rate. High-rate worlds step
#: at e.g. 500 Hz; syncing the viewer every step makes the scene update dominate the loop and run
#: below real-time. Rendering at a fixed display rate keeps physics real-time and the view smooth.
_RENDER_HZ = 60.0

#: Where ``--record`` writes when given no path.
_DEFAULT_RECORD = "run.npz"

#: Where ``--video`` writes when given no path. Its recording lands beside it as ``run.npz``.
_DEFAULT_VIDEO = "run.webm"

#: Environment anchors for a relative session output path, most specific first. A campaign runner sets
#: these to the directories it collects results from, so a relative ``ROQSIM_RECORD`` lands beside
#: the run's other artifacts rather than wherever the launch happened to leave the working directory.
#:
#: The order is the whole point, and getting it wrong is not visibly wrong: ``SCENARIO_OUTPUT_DIR`` is
#: the *campaign* root (every run's results live in subdirectories of it), so anchoring there writes
#: one shared path that each run of a sweep overwrites in turn. It is deliberately not in this list --
#: a per-run artifact has no business at the campaign root, and falling back to the working directory
#: is at least obviously local rather than quietly shared.
_OUTPUT_DIR_VARS = ("RUN_OUTPUT_DIR", "OUTPUT_DIR")


def _session_path(value: str) -> Path:
    """Resolve a session output path from the environment: absolute wins, relative gets anchored.

    Mirrors what the scenario adapter does with the ``output_dir`` scenario-execution hands it. The
    adapter is told; a launch-file run has to ask, because nothing passes it down.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    for var in _OUTPUT_DIR_VARS:
        base = os.environ.get(var)
        if base:
            return Path(base) / path
    return path


@contextlib.contextmanager
def _graceful_stop(control: ctl.RunControl, logger: logging.Logger):
    """Turn the first stop signal into a clean QUITTING request instead of a raised exception.

    A KeyboardInterrupt unwinding through ``launch_passive``'s live native GL render thread
    segfaults; flipping run-control lets the loop exit and the viewer tear down in order. A
    second signal restores default handling and forces the interrupt. No-op off the main thread.

    SIGTERM is handled alongside SIGINT because that is how a *supervised* run ends: a container
    teardown, ``docker stop``, a Kubernetes eviction and a campaign timeout all send TERM, whose
    default action kills the process outright — no ``finally``, so the recording and the capture
    are both lost. Under a campaign that is the common exit, not the exceptional one; only SIGKILL
    can still take the artifacts with it.

    Escalation is counted **per signal**, which matters as soon as TERM is handled: a supervisor
    routinely delivers INT *and* TERM for one stop (ros2 launch forwards both), and on a shared counter
    the second of the pair reads as "the user is insisting" and raises ``KeyboardInterrupt`` — observed
    landing inside the capture export, whose ``except Exception`` cannot catch a ``BaseException``. Two
    different signals are one request; only a repeat of the *same* one is somebody insisting.
    """
    prev: dict[int, object] = {}
    hits: dict[int, int] = {}

    def handler(signum, frame):
        hits[signum] = hits.get(signum, 0) + 1
        name = "Ctrl+C" if signum == signal.SIGINT else signal.Signals(signum).name
        if hits[signum] == 1:
            logger.info("%s: shutting down (%s again to force)", name, name)
            control.set_state(ctl.QUITTING)
        else:
            _restore(signum, prev[signum])
            raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev[sig] = signal.signal(sig, handler)
        except ValueError:
            # signal.signal only works on the main thread; leave the caller's handling in place.
            break
    try:
        yield
    finally:
        for sig, previous in prev.items():
            _restore(sig, previous)


def _restore(signum: int, previous) -> None:
    """Put a signal's previous handler back, tolerating one that Python cannot reinstall.

    ``getsignal``/``signal`` return ``None`` for a handler installed from C — which a process
    embedding rclpy and MuJoCo may well have — and passing that back raises ``TypeError``. This runs
    in the driver's teardown, so raising here would mask the run's own outcome and cost the recording
    that the rest of this function exists to save. Leaving our handler installed is the lesser harm,
    and the process is on its way out in every case that matters.
    """
    if previous is None:
        return
    signal.signal(signum, previous)


@contextlib.contextmanager
def _deaf_to_stop_signals(logger: logging.Logger):
    """Ignore INT/TERM while a run's artifacts are being written.

    The escalation counter in :func:`_graceful_stop` is per signal, so one supervisor's INT-then-TERM
    pair reads as a single request. That is not enough when a run has **two** supervisors, which is
    the normal case for a campaign whose simulator is a launch subprocess: ``ros2 launch`` forwards
    the stop to its children *and* the scenario's own cleanup signals the process group, so the
    simulator receives the same signal twice and the second one is, by every available measure,
    somebody insisting.

    Observed cost: the resulting ``KeyboardInterrupt`` landed in ``mujoco.mj_forward`` inside the
    capture export, which is guarded by ``except Exception`` and so cannot catch a ``BaseException``.
    It unwound the whole teardown -- taking not just the viewer artifact but ``engine.shutdown()``,
    and with it the CSV a scoring plugin writes there. The run reported success and left no verdict.

    So the teardown is deaf rather than merely defensive: it is short, bounded, and the last thing
    that happens before the process exits, and everything a run is judged by is written inside it.
    A supervisor that wants this process gone regardless still has SIGKILL, which no handler sees.
    """
    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
        except ValueError:
            # Off the main thread there is nothing to install; the caller's handling stands.
            break
    try:
        yield
    finally:
        for sig, prev in previous.items():
            _restore(sig, prev)
        if previous:
            logger.debug("teardown complete; stop signals handled normally again")


def _tick(engine: Engine, pacer: Pacer) -> bool:
    """Honor run-control; return True if a physics step was taken this iteration."""
    c = engine.ctx.control
    if c.take_reset():
        engine.reset()
        pacer.reset()
    if c.state == ctl.QUITTING:
        return False
    if c.should_step():
        pacer.wait()
        engine.step()
        return True
    time.sleep(0.005)  # idle while paused
    return None  # paused, not quitting


def _run_headless(
    engine: Engine, pacer: Pacer, max_steps: int | None, recorder: StateRecorder | None = None
) -> None:
    step = 0
    while max_steps is None or step < max_steps:
        if engine.ctx.stop_requested:
            break
        took = _tick(engine, pacer)
        if took is False:
            break
        if took:
            step += 1
            if recorder is not None:
                recorder.sample(engine.ctx)


def open_loading_viewer(*, left_ui: bool = False, right_ui: bool = False, key_callback=None):
    """Open the viewer on an empty placeholder and draw the splash overlay immediately.

    An empty model compiles and opens in ~1 ms (no meshes/textures to build), so the window and the
    splash come up as fast as MuJoCo can create the GL context -- *before* the real world compiles,
    so the splash covers the whole load. The real world is swapped into this same window later by
    :func:`adopt_world`; the splash (full-bleed navy art) stays up until then.

    Returns ``(handle, open_seconds)``, or ``(None, 0.0)`` if the window could not be opened -- the
    caller then falls back to opening on the real model, where the GL error is reported properly.
    """
    import mujoco

    started = time.perf_counter()
    try:
        model = mujoco.MjSpec().compile()  # empty world (worldbody only); compiles in ~1 ms
        data = mujoco.MjData(model)
        handle = launch_viewer(
            model, data, left_ui=left_ui, right_ui=right_ui, key_callback=key_callback
        )
    except Exception as err:  # noqa: BLE001 — no pre-window is fine; the real launch reports GL errors
        log.debug("loading window skipped: %s", err)
        return None, 0.0
    show_loading_overlay(handle)
    return handle, time.perf_counter() - started


def adopt_world(viewer, engine: Engine, *, name: str | None = None) -> float:
    """Load the compiled world into the already-open loading window. Returns the swap duration (s).

    ``sim.load`` blocks until the world is renderable (mesh/texture upload done), so the splash stays
    up and correct throughout -- the caller clears it exactly once, after framing the camera.
    """
    import mujoco

    sim = viewer._get_sim()  # noqa: SLF001 — the only route to MuJoCo's in-place model load
    if sim is None:
        raise DisplayError("the viewer window was closed before the world finished loading")
    started = time.perf_counter()
    sim.load(engine.ctx.model, engine.ctx.data, "")
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    retitle_window_async(engine.ctx.model, name=name)
    return time.perf_counter() - started


def _run_windowed(
    engine: Engine,
    pacer: Pacer,
    max_steps: int | None,
    view: dict | None = None,
    *,
    name: str | None = None,
    viewer=None,
    profile: bool = False,
    left_ui: bool = False,
    right_ui: bool = False,
    preview: bool = False,
    recorder=None,
    toggle: RecordToggle | None = None,
    saver: SaveViewKey | None = None,
    world_yaml: Path | None = None,
) -> None:
    # ``loading`` is True when the window is already up showing the splash (from open_loading_viewer):
    # the world is swapped in below and the splash is cleared once, after the camera is framed. The
    # fallback (no pre-opened window) opens on the real model with no overlay -- the model is already
    # loaded by then, so an overlay could only flash.
    loading = viewer is not None
    if not loading:
        # glad/GLFW/EGL failures surface here as native errors; translate them into one actionable
        # message instead of a deep traceback.
        try:
            viewer = launch_viewer(
                engine.ctx.model,
                engine.ctx.data,
                left_ui=left_ui,
                right_ui=right_ui,
                key_callback=_key_callback(toggle, saver),
            )
        except Exception as err:  # noqa: BLE001 — any GL init failure maps to the same guidance
            raise DisplayError(GL_HELP.format(err=err)) from err

    # Not ``with viewer:`` -- ``Handle.__exit__`` only *requests* the window's exit, which leaves the
    # teardown racing the process exit (see close_viewer). Everything that can raise with the window
    # already open (the world swap-in, camera setup) belongs inside, so it is always closed in order.
    try:
        if loading:
            swap_s = adopt_world(viewer, engine, name=name)
            log.debug("loading window: world swap-in took %.3fs", swap_s)
            if profile:
                print(f"[loading] world swap-in: {swap_s * 1e3:.0f} ms", file=sys.stderr)
        else:
            retitle_window_async(engine.ctx.model, name=name)
        camera = setup_camera(viewer, view, engine.ctx, preview=preview)
        if loading:
            # Scene loaded and camera framed under the splash: reveal it in one clean cut.
            clear_loading_overlay(viewer)
        step = 0
        render_period = 1.0 / _RENDER_HZ
        last_render = 0.0
        while viewer.is_running() and (max_steps is None or step < max_steps):
            if engine.ctx.stop_requested:
                break
            took = _tick(engine, pacer)
            if took is False:
                break
            # F9 and F8 are acted on here, on the physics thread: the UI-thread callbacks only set a
            # flag. That is also what makes the F8 dialog safe to open -- it blocks this loop, so the
            # pose being saved cannot move while the question is on screen.
            if toggle is not None and toggle.take_pending():
                recorder.toggle()
                retitle_window_async(engine.ctx.model, name=_rec_name(name, recorder.recording))
            if saver is not None and saver.take_pending():
                save_current_view(
                    _live_camera(viewer),
                    world_yaml,
                    view,
                    azimuth_offset=camera.azimuth_offset if camera is not None else None,
                    logger=engine.logger or log,
                )
            if took:
                step += 1
                if recorder is not None:
                    # Gated on sim time inside sample(); the live viewer camera goes in alongside the
                    # state so a render can reproduce what the person was actually looking at.
                    recorder.sample(engine.ctx, cam=_live_camera(viewer))
            # Render on a fixed wall-clock cadence, not every physics step: at 500 Hz physics a
            # per-step sync dominates the loop and drags it below real-time. Sync even when paused
            # (took is None) so the window stays responsive for camera drag.
            now = time.perf_counter()
            if now - last_render >= render_period:
                if camera is not None:
                    camera.update(viewer)
                apply_walk(viewer)  # arrow-key camera travel, integrated per rendered frame
                viewer.sync()
                last_render = now
    finally:
        close_viewer(viewer)


def _hotkeys() -> tuple[RecordToggle, SaveViewKey]:
    """roqsim's own viewer hotkeys, chained: F9 starts/stops a recording take, F8 saves the camera view.

    ``launch_passive`` accepts exactly one key callback, so each handler takes a ``chain`` it forwards
    the raw keycode to before looking at it. Built here, together, because the order is a property of
    the set rather than of any one handler; :class:`roqsim.viewer.WalkKeys` then chains onto the
    outermost of them (:func:`_key_callback`).
    """
    toggle = RecordToggle()
    return toggle, SaveViewKey(chain=toggle.key_callback)


def _key_callback(toggle: RecordToggle | None, saver: SaveViewKey | None):
    """The outermost handler of the chain built by :func:`_hotkeys` -- or ``None`` for a mute window."""
    if saver is not None:
        return saver.key_callback  # forwards into the toggle it was built with
    return toggle.key_callback if toggle is not None else None


def _rec_name(name: str | None, recording: bool) -> str | None:
    """Append a recording marker to the window title, so the state is visible where the person looks.

    A toggle you cannot see is a toggle you will get wrong -- and the window title is the one piece of
    chrome roqsim already owns (:mod:`roqsim.window_title` retitles through ctypes libX11).
    """
    base = name or ""
    return f"{base} [REC]" if recording else base or None


def _live_camera(viewer):
    """A snapshot of the window's current camera, or ``None`` if there is no window.

    ``mujoco.viewer`` hands ``handle.cam`` to its C++ ``Simulate`` by reference -- the same fact
    :func:`roqsim.viewer.apply_view` and :meth:`roqsim.viewer.WalkKeys.apply` already rely on -- so reading it
    is reading what the person is looking at, mouse drags and arrow-key flight included. Read under the
    viewer's lock because MuJoCo's render thread reads the same struct while rasterising; the copy is a
    handful of field reads and never a render.
    """
    if viewer is None:
        return None
    try:
        with viewer.lock():
            return _copy_camera(viewer.cam)
    except Exception:  # noqa: BLE001 - a closing window must not take the recording down
        return None


def _copy_camera(cam):
    import mujoco

    out = mujoco.MjvCamera()
    out.type, out.fixedcamid, out.trackbodyid = cam.type, cam.fixedcamid, cam.trackbodyid
    out.lookat[:] = cam.lookat
    out.distance, out.azimuth, out.elevation = cam.distance, cam.azimuth, cam.elevation
    return out


def is_model_ref(target: str) -> bool:
    """True when ``target`` names a single model/robot shown by itself (not a world YAML or scene).

    This is exactly the branch :func:`config_for_input` welds into the empty room, so the windowed
    viewers can ask "should I frame on the one model?" without re-deriving the dispatch. A raw mesh
    is not a model ref -- it is refused by :func:`config_for_input` before it ever reaches a viewer.
    Kept in step with :func:`config_for_input`: a ``<pkg>:<world>`` ref that resolves to a registered
    world is a world, not a model (else a viewless world would be framed as if it were one prop).
    """
    if Path(target).suffix.lower() in {".yaml", ".yml", ".xml", *_MESH_EXT}:
        return False
    if ":" in target and not Path(target).exists():
        try:
            if resolve_world_yaml_ref(target) is not None:
                return False  # names a world -> not a model ref
        except FileNotFoundError:
            pass  # provider exists but has no such world -> a model ref
    return True


def world_yaml_path(target: str) -> Path | None:
    """The world YAML file ``target`` is *defined by*, or ``None`` when it has none.

    The inverse question to :func:`config_for_input`: not "what does this run?" but "which file did
    this run come out of?" -- what a caller wanting to write back into the world needs (the viewer's
    F8 view save, :mod:`roqsim.view_save`). An MJCF scene and a model reference genuinely have no such
    file, and ``None`` says so rather than nominating a plausible one.

    Kept in step with :func:`config_for_input` and :func:`is_model_ref`, which make the same dispatch:
    a ``<pkg>:<world>`` ref that resolves to a registered world is a world, anything else is not. The
    *leaf* YAML is the answer even when it ``extends`` another -- that is the file whose ``sim`` block
    wins, so it is the one a write must land in.
    """
    if Path(target).suffix.lower() in (".yaml", ".yml"):
        path = Path(target).resolve()
        return path if path.is_file() else None
    if ":" in target and not Path(target).exists():
        try:
            resolved = resolve_world_yaml_ref(target)
        except FileNotFoundError:
            return None  # a worlds provider with no such world -> a model ref, not a world
        if resolved is not None:
            return Path(resolved).resolve()
    return None


def config_for_input(target: str, overrides: dict | None = None, transport: dict | None = None):
    """Build a :class:`SimConfig` from whatever the positional argument names, dispatched by shape.

    - ``*.yaml`` / ``*.yml`` — a world config (the full plugin/pacing/view surface).
    - ``*.xml`` — a baked MJCF scene, run as ``sim.world``.
    - a ``<pkg>:<world>`` ref that names a registered ``roqsim.worlds`` world — that world YAML
      (so ``roqsim_scenes:depot`` runs the world, not a model named ``depot``).
    - anything else — a model reference (``<pkg>:<name>``, a bundled model name, or a file path) shown
      by itself: welded into the default ``empty_room`` via the ``spawn_model`` plugin, so it stands on
      a lit floor with walls to judge its placement against (see :func:`is_model_ref`).

    Raw meshes are refused with an actionable message — they must be finalized into a model first.
    """
    suffix = Path(target).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_config(target, overrides, transport)
    if suffix in _MESH_EXT:
        raise PluginError(
            f"{target}: roqsim runs worlds, scenes and models, not raw meshes. Finalize it with "
            f"roqsim_assets/tools/finalize_mujoco.py and pass the resulting <name>.xml (or the model)."
        )
    if suffix == ".xml":
        raw = {"sim": {"world": str(Path(target).resolve())}}
        base_dir = Path(target).resolve().parent
    else:
        # A ``<pkg>:<name>`` ref names a world YAML before it names a model: resolve it against the
        # registered ``roqsim.worlds`` providers first, so a world runs as a world. ``None`` (the
        # left side is no worlds provider) and ``FileNotFoundError`` (it is, but has no such world)
        # both mean "not a world" -> fall through to the model-reference branch.
        if ":" in target and not Path(target).exists():
            try:
                world_yaml = resolve_world_yaml_ref(target)
            except FileNotFoundError:
                world_yaml = None
            if world_yaml is not None:
                return load_config(world_yaml, overrides, transport)
        # A model reference: let spawn_model resolve it and place it in the empty room.
        raw = {"plugins": [{"spawn_model": {"model": target}}]}
        base_dir = Path.cwd()
    if overrides:
        raw = apply_overrides(raw, overrides)
    if transport:
        raw = with_transport(raw, **transport)
    return load_config_from_dict(raw, base_dir=base_dir)


# Back-compat alias: this was ``_config_for_input`` before it became a public reuse point.
_config_for_input = config_for_input


def _warn_no_transport(dropped: list[str]) -> None:
    """Say what ``--no-communication`` cost, in terms of what the run no longer does.

    Naming the dropped plugins is not enough: someone reaching for the flag wants a window to open,
    and a line reading "dropped ros2_bridge" reads like bookkeeping rather than like the run having
    gone silent. So the warning is about the consequence -- nothing published, nothing received --
    and it fires on every such run, because that is the one fact separating this from a normal one.
    """
    if not dropped:
        log.warning(
            "--no-communication: this world declares no transport plugin, so nothing was dropped. "
            "The run was already mute; the flag changed nothing."
        )
        return
    log.warning(
        "--no-communication: dropped %s -- THIS RUN COMMUNICATES WITH NOTHING. Nothing is published "
        "(no clock, no TF, no sensor topics, no odometry) and no command is received (no cmd_vel, "
        "no goal, no service). Anything outside this process -- a nav2 stack, a scenario, a "
        "recording node -- sees a simulator that was never started, and a robot it would have "
        "driven stands still. Use it to look at the world, not to run its experiment.",
        ", ".join(dropped),
    )


def run(
    target: str,
    *,
    headless: bool = False,
    pacing=None,
    max_steps: int | None = None,
    seconds: float | None = None,
    profile: bool = False,
    left_ui: bool = False,
    right_ui: bool = False,
    manual_control: bool = False,
    overrides: dict | None = None,
    record: str | None = None,
    capture_fps=DEFAULT_FPS,
    video: str | None = None,
    video_size: str = "960x540",
    seed: int | None = None,
    transport: dict | None = None,
    no_transport: bool = False,
    logger: logging.Logger | None = None,
) -> Engine:
    """Load ``target``, run the loop, and return the (shut-down) engine. Programmatic entry point.

    ``target`` is a world YAML, an MJCF scene, or a model reference — see :func:`config_for_input`.
    ``overrides`` is a nested dict deep-merged into the resulting world before it is built (see
    :func:`roqsim.config.apply_overrides`).

    ``left_ui`` / ``right_ui`` show the viewer's Simulate side panels, and ``manual_control`` hands
    ``data.ctrl`` to the right panel's sliders instead of the world's controller plugins (see
    :attr:`roqsim.context.SimContext.manual_control`). All three are windowed-only run-level
    switches, deliberately not expressible in the world YAML.

    ``record`` writes a state recording to that path, sampled ``capture_fps`` times per *simulated*
    second (snapped onto the world's physics grid -- see :func:`roqsim.capture.snap_fps`). It is on the
    same footing as those switches: a session concern, not part of the experiment. Render it afterwards
    with ``roqsim render --state``.

    ``video`` implies ``record`` and renders that recording **after the loop has finished**, so the
    video exists when the process exits without a single frame having been drawn while physics was
    stepping. There is deliberately no fps or speed argument here: the file is encoded at
    ``capture_fps``, which makes one second of file exactly one second of sim time whatever pacing the
    run used -- a *record*, comparable with another run's. Deliberate other speeds are a presentation
    choice, and belong to ``roqsim render --state --speed``.

    ``no_transport`` is the opposite of ``transport``: it strips the bridge a world declares (see
    :func:`roqsim.config.drop_transport`) so a ``*_ros`` world can be *watched* without its middleware
    installed. The run is then mute -- it publishes and receives nothing -- which the loop says out
    loud, because a mute simulation is a different experiment rather than a quieter one.
    """
    # Parse the world before any GL: a bad target (missing YAML, unknown model ref, a schema error) is
    # a millisecond away and must fail as a plain message, not behind a window that then has to be torn
    # down again. The splash exists to cover the *world compile*, which is what follows.
    cfg = config_for_input(target, overrides, transport)
    if no_transport:
        _warn_no_transport(drop_transport(cfg))

    # Open the viewer on an empty placeholder first, so the roqsim logo is on screen while the
    # (slow) world compiles; the compiled world is swapped into this same window below. Windowed
    # runs only; if it can't open we get None and fall back to opening on the real model.
    # Built before the window, because the window opens on an empty placeholder *before* the world
    # compiles -- so the key callback has to exist first. Both are pointed at their targets below: the
    # toggle at the recorder, the saver at the world YAML this run came out of.
    toggle, saver = _hotkeys() if not headless else (None, None)

    loading_view, open_s = (None, 0.0)
    if not headless and has_display():
        loading_view, open_s = open_loading_viewer(
            left_ui=left_ui,
            right_ui=right_ui,
            key_callback=_key_callback(toggle, saver),
        )
        if open_s:
            log.debug("loading window: opened in %.3fs", open_s)
            if profile:
                print(f"[loading] placeholder window open: {open_s * 1e3:.0f} ms", file=sys.stderr)

    try:
        engine = Engine(cfg, logger=logger, profile=profile)
        # Before setup(): configure() may read it, and pre_step certainly does.
        engine.ctx.manual_control = manual_control
        # Drawn rather than defaulted to 0, so an unseeded run is still varied -- but *reported* and
        # recorded, so it can be repeated. Before this the sensors read a ctx.rng nothing ever set, so a
        # noisy run could not be reproduced at all.
        engine.ctx.seed = _resolve_seed(seed, logger or log, config_seed=getattr(cfg, "seed", None))
        engine.setup()
    except BaseException:
        # The compile failed while the loading window is up -- close it so it doesn't dangle.
        if loading_view is not None:
            close_viewer(loading_view)
        raise
    if profile:
        print(engine.format_load_report(), file=sys.stderr)
    engine.reset()

    dt = engine.dt
    pacer = Pacer.from_config(pacing if pacing is not None else cfg.pacing, dt)
    pacer.reset()

    if seconds is not None:
        steps_from_seconds = int(round(seconds / dt))
        max_steps = steps_from_seconds if max_steps is None else min(max_steps, steps_from_seconds)

    if video and not record:
        # A recording is how a video is made, so --video needs one; put it beside the video by default
        # rather than inventing a second convention for where it lives.
        record = str(Path(video).with_suffix(".npz"))

    # Session defaults from the environment, for a run nobody launched by hand. A campaign starts this
    # world through a ROS launch file (roqsim_ros_bridge.run_bridge -> here), so there is no command line
    # to add --record to without editing a launch file that two backends share. Recording is a session
    # concern -- the same footing as `sim.headless`, which the world YAML rejects on purpose -- so the
    # environment is the right channel, and it is the one the scenario adapter already uses.
    # An explicit flag always wins.
    if record is None:
        record = os.environ.get("ROQSIM_RECORD") or None
        if record:
            record = str(_session_path(record))
    if capture_fps == DEFAULT_FPS and os.environ.get("ROQSIM_CAPTURE_FPS"):
        capture_fps = parse_fps(os.environ["ROQSIM_CAPTURE_FPS"])

    # The rate is checked against the *compiled* timestep, which is the only authority: a world can
    # inherit it from a baked MJCF rather than declaring it in `sim:`.
    rate = snap_fps(capture_fps, dt)
    if record or not headless:
        rate.report(engine.logger or log)

    # NB: ``toggle``/``saver`` are the ones built above and already wired into the window's key
    # callback -- do not rebind them here. Re-initialising ``toggle = None`` at this point is what
    # silently killed F9: the window went on setting the flag on an object the loop no longer held.
    recorder = None
    if headless:
        if record:
            recorder = StateRecorder(
                engine.ctx,
                record,
                rate,
                world=target,
                overrides=overrides,
                camera=False,
                sim_poses=env_flag("ROQSIM_SIM_POSES"),
                logger=engine.logger or log,
            )
    else:
        # Windowed: always build the take recorder, even without --record, so F9 works in any windowed
        # run. It costs nothing until a take starts -- recording is a memcpy.
        recorder = TakeRecorder(
            engine.ctx,
            record or _DEFAULT_RECORD,
            rate,
            world=target,
            overrides=overrides,
            camera=True,
            logger=engine.logger or log,
        )
        if record:
            recorder.start()

    # The handlers stay installed across the teardown, not just the loop: the flush is the part that must
    # survive a signal, and restoring the defaults first would mean a SIGTERM arriving during
    # recorder.close() -- which is exactly when a supervisor sends it -- killing the process mid-write.
    with _graceful_stop(engine.ctx.control, engine.logger):
        try:
            if headless:
                _run_headless(engine, pacer, max_steps, recorder)
            else:
                _run_windowed(
                    engine,
                    pacer,
                    max_steps,
                    cfg.view,
                    name=cfg.name,
                    viewer=loading_view,
                    profile=profile,
                    left_ui=left_ui,
                    right_ui=right_ui,
                    preview=is_model_ref(target),
                    recorder=recorder,
                    toggle=toggle,
                    saver=saver,
                    world_yaml=world_yaml_path(target),
                )
        finally:
            # Every stop that matters reaches here: a closed window ends the loop normally, and Ctrl+C or
            # a supervisor's SIGTERM is turned into QUITTING by _graceful_stop rather than raising through
            # the render thread. The recording is written *first* -- it is the primary artifact, and the
            # capture is derived from the same samples, so nothing about the export can be allowed to cost
            # it. Both still run before engine.shutdown(), because the export reads the live model: it
            # needs no world rebuild and no GL backend.
            #
            # All of it runs deaf to further stop signals. Two supervisors mean two of the same
            # signal, which the escalation counter reads as insisting -- and an interrupt raised in
            # here does not end a run early, it ends it *without its results*: the capture, and the
            # CSV a scoring plugin writes in shutdown(), are both produced below this line.
            with _deaf_to_stop_signals(engine.logger or log):
                written = recorder.close() if recorder is not None else None
                _export_capture_at_exit(engine, recorder, target, overrides, engine.logger or log)
                if profile:
                    print(engine.format_timing(), file=sys.stderr)
                engine.shutdown()

    # After the loop and after shutdown: the render is outside the measurement window entirely, which is
    # what lets --video cost the run nothing. Not in the finally, so an exception that ended the run is
    # never masked by a rendering problem on the way out.
    if video and written:
        takes = written if isinstance(written, list) else [written]
        for index, take in enumerate(takes, start=1):
            out = video if len(takes) == 1 else _numbered(video, index)
            _render_at_exit(take, out, video_size, engine.logger or log)
    return engine


def _export_capture_at_exit(engine, recorder, target, overrides, logger: logging.Logger) -> None:
    """Derive a browser run capture from what was just recorded, when asked.

    Opt-in via ``ROQSIM_CAPTURE_EXPORT_DIR``, alongside ``ROQSIM_RECORD``. Reported and never
    fatal: a run whose results are otherwise good must not fail because a viewer artifact could not be
    written. Deriving here rather than from the file afterwards is what avoids a second world build --
    the model is still live and the samples are still in memory.

    A ``TakeRecorder`` (a windowed run) may hold several numbered takes; only a single-take recording
    maps onto one capture directory, so anything else is skipped with a reason rather than silently
    exporting whichever take happened to be last.
    """
    out = os.environ.get("ROQSIM_CAPTURE_EXPORT_DIR")
    if not out or recorder is None:
        return
    if not hasattr(recorder, "replay"):
        # A windowed run's F9 takes are numbered, and a capture directory holds one run's motion; say so
        # rather than exporting whichever take happened to be last.
        logger.info(
            "run capture: skipped -- a windowed run records numbered takes, which do not map onto one "
            "capture. Export one afterwards with `roqsim export capture --state <take>.npz`."
        )
        return
    if not recorder.frames:
        return
    try:
        # Imported here, inside the guard: an ImportError is as non-fatal as an export failure, and
        # outside it would propagate out of the driver's `finally` and skip the rest of the teardown.
        from .export_capture import write_capture

        write_capture(
            engine.ctx.model,
            recorder.replay(engine.ctx),
            _session_path(out),
            world=target,
            overrides=overrides or {},
            seed=getattr(engine.ctx, "seed", None),
            logger=logger,
        )
    except Exception as err:  # noqa: BLE001 — a viewer artifact must not fail the run
        logger.warning("run capture export failed (%s); the recording itself is unaffected", err)


def _resolve_seed(seed: int | None, logger: logging.Logger, config_seed: int | None = None) -> int:
    """The run's noise seed, by precedence: explicit > world config > drawn.

    An explicitly passed seed wins because it is the more specific instruction -- the
    world states what a run normally uses, the caller states what THIS run uses. With
    neither, one is drawn and announced, exactly as before ``sim.seed`` existed.
    """
    if seed is not None:
        logger.info("seed: %d (given)", seed)
        return int(seed)
    if config_seed is not None:
        logger.info("seed: %d (from sim.seed)", config_seed)
        return int(config_seed)
    import secrets

    drawn = secrets.randbelow(2**31)
    logger.info("seed: %d (drawn -- pass --seed %d to repeat this run)", drawn, drawn)
    return drawn


def _numbered(path: str, index: int) -> str:
    """``run.webm`` -> ``run-2.webm`` for the second take, matching the recordings' own numbering."""
    p = Path(path)
    return str(p) if index == 1 else str(p.with_name(f"{p.stem}-{index}{p.suffix}"))


def _render_at_exit(recording: Path, video: str, size: str, logger: logging.Logger) -> None:
    """Render a just-finished recording to ``video``. Reported, never fatal.

    The run itself already succeeded and its recording is on disk, so a rendering failure must not turn
    a good run into a failed command -- the recording can always be rendered again by hand. Progress is
    printed because a silent multi-second pause at exit reads as a hang.
    """
    from .render import render_target

    # Printed *before* the pause, because a silent multi-second wait at exit reads as a hang. The
    # completion line is render_target's own, so it is not repeated here.
    logger.info("video: rendering %s -> %s ...", recording, video)
    try:
        render_target(None, video, size=size, state=recording)
    except Exception as err:  # noqa: BLE001 - deliberately broad: see the docstring
        logger.error(
            "video: could not render %s (%s: %s). The recording is intact -- retry with "
            "`roqsim render --state %s --out %s`.",
            video,
            type(err).__name__,
            err,
            recording,
            video,
        )


#: World-ish extensions an optional-value flag must not be given. `roqsim sim --record world.yaml` makes
#: `--record` swallow the target, and argparse then reports a *missing positional* -- true but useless,
#: and the kind of thing that survives in a checked-in Makefile for months.
_WORLDISH = (".yaml", ".yml", ".xml")


def _refuse_swallowed_target(argv: list[str]) -> None:
    """Refuse `--record <world>` before argparse can turn it into a confusing missing-target error."""
    for flag in ("--record", "--video"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv) and Path(argv[i + 1]).suffix.lower() in _WORLDISH:
                raise SystemExit(
                    f"roqsim sim: {flag} takes an output path, not {argv[i + 1]!r}. Put the world first:\n"
                    f"    roqsim sim {argv[i + 1]} {flag} <file>"
                )


def main(argv: list | None = None) -> int:
    # Opening a window: default the offscreen backend to egl and preload libGLEW (re-exec'ing once
    # if needed), before anything loads GL, to dodge MuJoCo's "gladLoadGL error" on camera worlds.
    # Done before argparse so the re-exec'd process, not this one, does the real work. No-op headless
    # or without a display; both defaults are overridable (MUJOCO_GL / ROQSIM_NO_GL_PRELOAD).
    _cli = sys.argv[1:] if argv is None else argv
    # Printing help opens no window, and the preload re-execs the process -- which on a help call is
    # pure noise (and doubles the time the command tree takes to answer a question about itself).
    _asking_for_help = bool({"-h", "--help"} & set(_cli))
    if "--headless" not in _cli and not _asking_for_help and has_display():
        prepare_viewer_gl()
    elif not _asking_for_help:
        # Headless (or no display): pick an offscreen backend that exists on *this*
        # machine. Left to MuJoCo's own default, a CPU-only node fails at import with a
        # message about a broken GL install -- the wrong diagnosis for a node that
        # simply has no GPU.
        #
        # Normally already done, and it has to be: mujoco is imported by this module's own
        # imports, long before this line runs, so a selection made HERE is made too late to bind
        # anything. `import roqsim` is what actually decides it (see roqsim.gl). Kept anyway --
        # it is idempotent, and it keeps the CLI correct if the package __init__ ever stops.
        select_offscreen_gl()

    _refuse_swallowed_target(_cli)

    parser = argparse.ArgumentParser(
        prog="roqsim sim",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "target",
        help="what to run: a world YAML, an MJCF scene (.xml), or a model reference "
        "(<pkg>:<name>, a bundled model name, or a path) shown alone in an empty room",
    )
    parser.add_argument("--headless", action="store_true", help="no viewer window (k8s)")
    parser.add_argument(
        "--ros",
        action="store_true",
        help="publish this world over ROS 2: appends the ros2_bridge plugin at load time. "
        "A checked-in world stays ROS-free so it runs standalone in a pip-only environment "
        "(where the bridge, a colcon package, is not registered) -- transport is how a run is "
        "deployed, not part of the experiment.",
    )
    parser.add_argument(
        "--tf-namespace",
        default=None,
        metavar="NS",
        help="publish /<NS>/tf and /<NS>/tf_static instead of the global topics (frames are "
        "unchanged). What Nav2's standard /tf->tf remap expects; without it a namespaced stack "
        "hangs waiting for a transform that is being published somewhere else. Implies --ros.",
    )
    parser.add_argument(
        "--sim-control",
        action="store_true",
        help="also serve the simulation_interfaces control plane (sim_interfaces), which a "
        "scenario's osc.sim actions are clients of. Off by default: only a scenario that "
        "touches entities needs it. Implies --ros.",
    )
    parser.add_argument(
        "--no-communication",
        action="store_true",
        help="run a world that declares a transport WITHOUT it: every transport plugin is dropped at "
        "load time (the ROS bridge is the usual one, but any transport goes), so a *_ros world opens "
        "in the viewer on a machine with no middleware installed. THE RUN THEN HAS NO COMMUNICATION: "
        "it publishes nothing (no clock, no TF, no sensor topics) and receives nothing (no cmd_vel, "
        "no goals), so an external stack sees no simulator at all and a robot it would have driven "
        "just stands there. For looking at the world; not for running its experiment. Only plugins "
        "identifiable as transport go -- an unresolvable ref that is not one still fails loudly, "
        "because there it would be a typo.",
    )
    parser.add_argument(
        "--pacing", default=None, help="'realtime' | 'asap' | a float factor (e.g. 4.0)"
    )
    parser.add_argument("--steps", type=int, default=None, help="stop after N steps")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N sim seconds")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print load-phase timing after setup and per-plugin hook timing at exit "
        "(timing is only collected with this flag)",
    )
    parser.add_argument(
        "--left-ui", action="store_true", help="show MuJoCo's left side panel at launch"
    )
    parser.add_argument(
        "--right-ui",
        action="store_true",
        help="show MuJoCo's right side panel -- the joint/actuator controls -- at launch",
    )
    parser.add_argument(
        "--manual-control",
        action="store_true",
        help="drive the robot by hand: every controller plugin stops writing data.ctrl and the "
        "viewer's control sliders take over (implies --right-ui, where the sliders live)",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="record MuJoCo state to this file, for later rendering with `roqsim render --state`. "
        "A path is required rather than defaulted: a recording is an artifact you will name again "
        "later, and a default would quietly overwrite the previous run's.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="seed the sensor noise, making the run reproducible (default: drawn and reported)",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const=_DEFAULT_VIDEO,
        default=None,
        metavar="PATH",
        help=f"implies --record, and renders the recording when the run ends (default: "
        f"{_DEFAULT_VIDEO}). Encoded at --capture-fps, so 1 s of file is 1 s of sim time; other "
        "speeds are `roqsim render --state --speed`.",
    )
    parser.add_argument(
        "--video-size", default="960x540", metavar="WxH", help="video frame size (default: 960x540)"
    )
    parser.add_argument(
        "--capture-fps",
        default=DEFAULT_FPS,
        metavar="N",
        help=f"samples per SIMULATED second (default: {DEFAULT_FPS}); accepts a fraction like 500/17. "
        "Snapped onto the world's physics step grid.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="PATH=VALUE",
        help="override a world value, e.g. --set plugins.floorplan.floor.reflectance=0.3 (repeatable)",
    )
    parser.add_argument(
        "--override",
        dest="override_files",
        action="append",
        metavar="FILE",
        help="a YAML file of world overrides -- the file spelling of --set, for anything "
        "structured enough that flattening it onto a command line loses it (repeatable; "
        "later files and --set win)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(verbose=args.verbose)

    pacing = args.pacing
    if pacing is not None and pacing not in ("realtime", "asap"):
        try:
            pacing = {"factor": float(pacing)}
        except ValueError:
            parser.error(f"invalid --pacing {pacing!r}")

    # Manual control needs the sliders, which live in the right panel and only exist in a window.
    # Headless would silently leave the robot with nobody driving it, so refuse rather than run it.
    if args.manual_control and args.headless:
        parser.error("--manual-control needs the viewer's control sliders; it can't run --headless")

    # --tf-namespace / --sim-control only mean anything with a bridge, so they imply one
    # rather than being silently ignored next to a world that publishes nothing.
    ros = args.ros or bool(args.tf_namespace) or args.sim_control
    transport = (
        {"ros": True, "tf_namespace": args.tf_namespace, "control": args.sim_control}
        if ros
        else None
    )

    # Adding a bridge and stripping one are opposite intents, and the order they would resolve in is
    # an implementation detail nobody should have to know. Refuse rather than pick a winner.
    if args.no_communication and ros:
        parser.error(
            "--no-communication contradicts --ros/--tf-namespace/--sim-control: pass one or the other"
        )

    try:
        run(
            args.target,
            headless=args.headless,
            pacing=pacing,
            max_steps=args.steps,
            seconds=args.seconds,
            profile=args.profile,
            left_ui=args.left_ui,
            # The sliders live in the right panel, so manual control is useless without it.
            right_ui=args.right_ui or args.manual_control,
            manual_control=args.manual_control,
            # Files first, --set last: a saved override set plus one ad-hoc tweak is the
            # obvious way to use the two together, and the tweak is what should win.
            overrides=deep_merge(
                overrides_from_files(args.override_files), overrides_from_dotlist(args.overrides)
            ),
            record=args.record,
            capture_fps=args.capture_fps,
            video=args.video,
            video_size=args.video_size,
            seed=args.seed,
            transport=transport,
            no_transport=args.no_communication,
        )
    except (DisplayError, ViewError, PluginError, ModelError, CaptureError) as err:
        print(f"roqsim sim: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
