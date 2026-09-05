"""Viewer camera setup: the free camera, UI panels, the tracking camera, and the chase cam."""

from __future__ import annotations

import contextlib
import math
import os
import threading
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np
import pytest

from roqsim import keys as key_catalogue
from roqsim.context import Entity, SimContext
from roqsim.rendering import view_forward
from roqsim.viewer import (
    DEFAULT_MUJOCO_GL,
    KEY_F10,
    HelpKeys,
    TrackingCamera,
    ViewError,
    WalkKeys,
    apply_view,
    close_viewer,
    ensure_gl_preload,
    launch_viewer,
    prepare_viewer_gl,
    setup_camera,
    ui_kwargs,
)

_XML = """
<mujoco>
  <worldbody>
    <body name="robot/base_link" pos="0 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeViewer:
    """Minimal stand-in for the passive-viewer handle."""

    def __init__(self):
        self.cam = SimpleNamespace(
            lookat=[0.0, 0.0, 0.0], distance=1.0, azimuth=0.0, elevation=0.0, type=0, trackbodyid=-1
        )
        self.synced = 0

    @contextlib.contextmanager
    def lock(self):
        yield

    def sync(self):
        self.synced += 1


@pytest.fixture
def ctx() -> SimContext:
    c = SimContext({})
    c.model = mujoco.MjModel.from_xml_string(_XML)
    c.data = mujoco.MjData(c.model)
    c.entities.add(Entity(name="robot", kind="robot", body="robot/base_link"))
    return c


def _set_yaw(ctx: SimContext, bid: int, yaw_deg: float) -> None:
    """Write the body's world rotation directly; mj_forward would need a matching qpos."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    ctx.data.xmat[bid] = [c, -s, 0, s, c, 0, 0, 0, 1]


# -- free camera + UI panels ---------------------------------------------------------------------


def test_apply_view_sets_all_camera_fields():
    v = _FakeViewer()
    apply_view(v, {"lookat": [1, 2, 3], "distance": 3.4, "azimuth": 130, "elevation": -20})
    assert v.cam.lookat == [1.0, 2.0, 3.0]
    assert v.cam.distance == 3.4
    assert v.cam.azimuth == 130.0
    assert v.cam.elevation == -20.0
    assert v.synced == 1


def test_apply_view_partial_keeps_defaults():
    v = _FakeViewer()
    apply_view(v, {"azimuth": 90})
    assert v.cam.azimuth == 90.0
    assert v.cam.distance == 1.0  # untouched default
    assert v.cam.lookat == [0.0, 0.0, 0.0]


def test_apply_view_none_is_noop():
    v = _FakeViewer()
    apply_view(v, None)
    apply_view(v, {})
    assert v.synced == 0  # no viewer mutation when nothing to apply


def test_ui_kwargs_hidden_by_default():
    assert ui_kwargs() == {"show_left_ui": False, "show_right_ui": False}
    assert ui_kwargs(left_ui=True) == {"show_left_ui": True, "show_right_ui": False}
    assert ui_kwargs(right_ui=True) == {"show_left_ui": False, "show_right_ui": True}


# -- setup_camera: free vs tracking dispatch -----------------------------------------------------


def test_setup_camera_free_returns_none(ctx):
    v = _FakeViewer()
    assert setup_camera(v, {"azimuth": 130}, ctx) is None
    assert v.cam.type == 0  # mjCAMERA_FREE
    assert v.cam.azimuth == 130.0


def test_setup_camera_tracking_returns_controller(ctx):
    v = _FakeViewer()
    cam = setup_camera(v, {"track": "robot"}, ctx)
    assert isinstance(cam, TrackingCamera)
    assert v.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING


def test_setup_camera_preview_autoframes_on_the_model(ctx):
    v = _FakeViewer()
    assert setup_camera(v, {}, ctx, preview=True) is None
    # Zoomed onto the 0.1 m box at the origin: close in, centred on it, not left at the default.
    assert v.cam.distance == pytest.approx(0.497, abs=0.05)
    assert list(v.cam.lookat) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert v.synced == 1


def test_setup_camera_explicit_view_beats_preview(ctx):
    v = _FakeViewer()
    setup_camera(v, {"azimuth": 130}, ctx, preview=True)
    assert v.cam.azimuth == 130.0
    assert v.cam.distance == 1.0  # view given -> autoframe stands down, default distance kept


# -- target resolution ---------------------------------------------------------------------------


def test_track_resolves_entity_name(ctx):
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    assert TrackingCamera({"track": "robot"}, ctx).track_body_id == bid


def test_track_falls_back_to_body_name(ctx):
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    assert TrackingCamera({"track": "robot/base_link"}, ctx).track_body_id == bid


def test_track_unknown_target_lists_entities(ctx):
    with pytest.raises(ViewError, match="robot"):
        TrackingCamera({"track": "nope"}, ctx)


def test_follow_heading_without_track_is_rejected(ctx):
    with pytest.raises(ViewError, match="follow_heading"):
        TrackingCamera({"follow_heading": True}, ctx)


# -- tracking camera -----------------------------------------------------------------------------


def test_apply_sets_tracking_camera(ctx):
    v = _FakeViewer()
    cam = TrackingCamera({"track": "robot", "distance": 3.4, "elevation": -20}, ctx)
    cam.apply(v)
    assert v.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert v.cam.trackbodyid == cam.track_body_id
    assert v.cam.distance == 3.4
    assert v.cam.elevation == -20.0


def test_update_is_noop_without_follow_heading(ctx):
    v = _FakeViewer()
    cam = TrackingCamera({"track": "robot", "azimuth": 130}, ctx)
    cam.apply(v)
    _set_yaw(ctx, cam.track_body_id, 90.0)
    cam.update(v)
    assert v.cam.azimuth == 130.0  # world-fixed: the robot turns, the camera does not


# -- chase cam -----------------------------------------------------------------------------------


def test_follow_heading_tracks_yaw_with_offset(ctx):
    v = _FakeViewer()
    cam = TrackingCamera({"track": "robot", "follow_heading": True, "azimuth": 180}, ctx)
    cam.apply(v)
    _set_yaw(ctx, cam.track_body_id, 0.0)
    cam.update(v)
    assert v.cam.azimuth == pytest.approx(180.0)  # behind a robot facing +x

    _set_yaw(ctx, cam.track_body_id, 90.0)
    cam.update(v)
    assert v.cam.azimuth == pytest.approx(270.0)  # stays behind through the turn


def test_mouse_drag_shifts_the_offset(ctx):
    v = _FakeViewer()
    cam = TrackingCamera({"track": "robot", "follow_heading": True, "azimuth": 180}, ctx)
    cam.apply(v)
    _set_yaw(ctx, cam.track_body_id, 0.0)
    cam.update(v)

    v.cam.azimuth += 30.0  # the user orbits 30 deg while the robot holds still
    cam.update(v)
    assert v.cam.azimuth == pytest.approx(210.0)  # drag absorbed, not stomped

    _set_yaw(ctx, cam.track_body_id, 45.0)
    cam.update(v)
    assert v.cam.azimuth == pytest.approx(255.0)  # new angle holds relative to the robot


def test_mouse_drag_across_the_wrap_seam(ctx):
    v = _FakeViewer()
    cam = TrackingCamera({"track": "robot", "follow_heading": True, "azimuth": 179}, ctx)
    cam.apply(v)
    _set_yaw(ctx, cam.track_body_id, 0.0)
    cam.update(v)

    # MuJoCo normalises azimuth into (-180, 180]: a 2 deg drag past 180 reads as -179.
    v.cam.azimuth = -179.0
    cam.update(v)
    assert v.cam.azimuth == pytest.approx(181.0)  # +2 deg, not -358


# --- ensure_gl_preload: the libGLEW re-exec guard -------------------------------------------------
# execv would replace the process, so every test here must hit a no-op branch. A test that lets it
# reach execv would take down the test runner -- so each asserts execv is NOT called.


def _no_execv(monkeypatch):
    called = []
    monkeypatch.setattr("roqsim.viewer.os.execv", lambda *a: called.append(a))
    return called


def test_gl_preload_noop_when_already_reexecd(monkeypatch):
    called = _no_execv(monkeypatch)
    monkeypatch.setenv("ROQSIM_GL_PRELOADED", "1")
    ensure_gl_preload()
    assert called == []  # sentinel present -> never loop


def test_gl_preload_noop_when_opted_out(monkeypatch):
    called = _no_execv(monkeypatch)
    monkeypatch.delenv("ROQSIM_GL_PRELOADED", raising=False)
    monkeypatch.setenv("ROQSIM_NO_GL_PRELOAD", "1")
    ensure_gl_preload()
    assert called == []


def test_gl_preload_noop_when_env_already_has_glew(monkeypatch):
    called = _no_execv(monkeypatch)
    monkeypatch.delenv("ROQSIM_GL_PRELOADED", raising=False)
    monkeypatch.delenv("ROQSIM_NO_GL_PRELOAD", raising=False)
    monkeypatch.setenv("LD_PRELOAD", "/some/libGLEW.so.2.2")
    ensure_gl_preload()
    assert called == []  # don't fight an existing preload


def test_gl_preload_noop_when_glew_not_found(monkeypatch):
    called = _no_execv(monkeypatch)
    monkeypatch.delenv("ROQSIM_GL_PRELOADED", raising=False)
    monkeypatch.delenv("ROQSIM_NO_GL_PRELOAD", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setattr("roqsim.viewer._resolve_glew", lambda: None)
    ensure_gl_preload()
    assert called == []  # can't help; GL_HELP covers the broken-GL-stack case


def test_gl_preload_reexecs_with_glew_and_sentinel(monkeypatch):
    called = _no_execv(monkeypatch)
    monkeypatch.delenv("ROQSIM_GL_PRELOADED", raising=False)
    monkeypatch.delenv("ROQSIM_NO_GL_PRELOAD", raising=False)
    monkeypatch.setenv("LD_PRELOAD", "/pre/existing.so")
    monkeypatch.setattr("roqsim.viewer._resolve_glew", lambda: "/usr/lib/libGLEW.so.2.2")
    ensure_gl_preload()
    assert len(called) == 1  # re-exec'd exactly once
    # absolute libGLEW prepended, existing preload preserved, sentinel set to break the loop
    assert os.environ["LD_PRELOAD"] == "/usr/lib/libGLEW.so.2.2 /pre/existing.so"
    assert os.environ["ROQSIM_GL_PRELOADED"] == "1"


# --- prepare_viewer_gl: backend default + libGLEW preload orchestration ---------------------------


def test_prepare_viewer_gl_defaults_mujoco_gl_to_egl(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr("roqsim.viewer.ensure_gl_preload", lambda: None)
    prepare_viewer_gl()
    assert os.environ["MUJOCO_GL"] == DEFAULT_MUJOCO_GL == "egl"


def test_prepare_viewer_gl_respects_explicit_mujoco_gl(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setattr("roqsim.viewer.ensure_gl_preload", lambda: None)
    prepare_viewer_gl()
    assert os.environ["MUJOCO_GL"] == "glfw"  # user override wins; not clobbered


def test_prepare_viewer_gl_also_preloads(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    calls = []
    monkeypatch.setattr("roqsim.viewer.ensure_gl_preload", lambda: calls.append(True))
    prepare_viewer_gl()
    assert calls == [True]  # the libGLEW preload still runs


# --- launch_viewer / close_viewer: ordered window teardown ----------------------------------------
#
# The window runs on threads MuJoCo owns and ``Handle.close()`` only *requests* their exit, so a
# process that closed and exited immediately used to hang (or segfault) in the race between that
# teardown and interpreter shutdown. close_viewer must therefore wait for those threads.


class _FakeHandle:
    """Stand-in for ``mujoco.viewer.Handle``: ``close()`` only signals, like the real one."""

    def __init__(self, stop: threading.Event | None = None, raises: bool = False):
        self.stop = stop or threading.Event()
        self.raises = raises
        self.closed = 0

    def close(self):
        self.closed += 1
        if self.raises:
            raise RuntimeError("native handle already gone")
        self.stop.set()


def _fake_launch(monkeypatch, *, stop: threading.Event):
    """Patch ``launch_passive`` with one that spawns a render-loop-ish thread. Returns the kwargs."""
    seen = {}

    def launch_passive(model, data, **kwargs):
        seen.update(kwargs)
        handle = _FakeHandle(stop)
        thread = threading.Thread(target=stop.wait, name="fake-render-loop", daemon=True)
        thread.start()
        handle.thread = thread
        return handle

    monkeypatch.setattr("mujoco.viewer.launch_passive", launch_passive)
    return seen


def test_close_viewer_waits_for_the_render_thread(monkeypatch):
    stop = threading.Event()
    seen = _fake_launch(monkeypatch, stop=stop)
    handle = launch_viewer(object(), object(), right_ui=True)
    panels = {k: v for k, v in seen.items() if k != "key_callback"}
    assert panels == ui_kwargs(False, True)  # panel switches still reach launch_passive
    assert callable(seen["key_callback"])  # ... alongside the WASD walk callback
    assert handle.thread.is_alive()

    close_viewer(handle)
    assert handle.closed == 1
    assert not handle.thread.is_alive()  # joined, not merely asked to stop


def test_close_viewer_forgets_the_handle_and_is_idempotent(monkeypatch):
    stop = threading.Event()
    _fake_launch(monkeypatch, stop=stop)
    handle = launch_viewer(object(), object())
    close_viewer(handle)
    close_viewer(handle)  # nothing left to join; a second close must not raise
    assert handle.closed == 2


def test_close_viewer_swallows_a_failing_close():
    # Teardown never raises: the caller is usually already unwinding a load failure.
    close_viewer(_FakeHandle(raises=True))


def test_close_viewer_gives_up_on_a_wedged_thread(monkeypatch, capsys):
    never = threading.Event()
    _fake_launch(monkeypatch, stop=never)
    handle = launch_viewer(object(), object())
    handle.stop = threading.Event()  # close() no longer releases the thread: it stays wedged
    close_viewer(handle, timeout=0.05)  # returns instead of blocking the process forever
    assert "did not shut down" in capsys.readouterr().err
    never.set()


_UP, _DOWN, _LEFT, _PAGE_UP = 265, 264, 263, 266  # GLFW keycodes, as MuJoCo reports them


class _Handle:
    """A stand-in viewer handle: the free camera and lock() WalkKeys touches, and nothing else.

    A class rather than a ``SimpleNamespace`` because the overlay text, like the walk state and the
    viewer's threads, is held against the handle in a ``WeakKeyDictionary`` -- as MuJoCo's own handle
    can be, and a ``SimpleNamespace`` cannot.
    """

    def __init__(self, camera):
        self.cam = camera
        self.lock = contextlib.nullcontext


def _walk_handle(**cam):
    """A stand-in viewer handle exposing just the free camera and lock() that WalkKeys touches."""
    camera = SimpleNamespace(
        lookat=[0.0, 0.0, 1.0],
        azimuth=0.0,
        elevation=0.0,
        distance=4.0,
        type=int(mujoco.mjtCamera.mjCAMERA_FREE),
    )
    for k, v in cam.items():
        setattr(camera, k, v)
    return _Handle(camera)


class _Clock:
    """A hand-cranked ``time.monotonic``: WalkKeys integrates real elapsed time, so the tests own it."""

    def __init__(self, t=100.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _held_frame(walk, handle, keycode):
    """One rendered frame with ``keycode`` still down: its auto-repeat lands, then the frame draws."""
    walk.key_callback(keycode)
    walk.apply(handle)


def test_walk_keys_fly_the_free_camera_at_a_speed():
    handle, walk, clock = (
        _walk_handle(),
        WalkKeys(speed=2.0, keys=False, pivot_radius=0, fly=True),
        _Clock(),
    )
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _held_frame(walk, handle, _UP)  # first frame only starts the clock
        clock.advance(0.25)
        _held_frame(walk, handle, _UP)  # 2 m/s for 0.25 s = 0.5 m forward
        assert handle.cam.lookat == pytest.approx([0.5, 0.0, 1.0])

        handle.cam.azimuth = 90.0  # turn: forward is now +y
        clock.advance(0.25)
        _held_frame(walk, handle, _UP)
        assert handle.cam.lookat == pytest.approx([0.5, 0.5, 1.0])

        handle.cam.elevation = -90.0  # look straight down: forward flies down with the view
        clock.advance(0.25)
        _held_frame(walk, handle, _UP)
        assert handle.cam.lookat == pytest.approx([0.5, 0.5, 0.5])


def test_walk_keys_stop_when_the_key_events_stop():
    # MuJoCo forwards no key release at all, so a direction is held only while its repeats arrive.
    walk = WalkKeys(speed=2.0, hold_s=0.35, keys=False, pivot_radius=0, fly=True)
    handle, clock = _walk_handle(), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _held_frame(walk, handle, _PAGE_UP)
        clock.advance(0.25)
        _held_frame(walk, handle, _PAGE_UP)
        assert handle.cam.lookat[2] == pytest.approx(1.5)  # rose 0.5 m

        clock.advance(0.5)  # key released: its repeats stop, the hold window lapses
        walk.apply(handle)
        clock.advance(0.5)
        walk.apply(handle)
        assert handle.cam.lookat[2] == pytest.approx(1.5)  # and the camera stays put


def test_walk_keys_ignore_other_keys_and_non_free_cameras():
    walk, clock = WalkKeys(speed=2.0, keys=False, pivot_radius=0, fly=True), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        for code in (ord(" "), ord("W"), 256):  # pause, MuJoCo's wireframe toggle, Esc
            walk.key_callback(code)
        assert walk.held(clock()) == []

        tracking = _walk_handle(type=int(mujoco.mjtCamera.mjCAMERA_TRACKING))
        _held_frame(walk, tracking, _UP)
        clock.advance(0.25)
        _held_frame(walk, tracking, _UP)
        assert tracking.cam.lookat == pytest.approx([0.0, 0.0, 1.0])  # tracking owns the pose


def test_walk_keys_normalise_two_directions_at_once():
    handle, walk, clock = (
        _walk_handle(),
        WalkKeys(speed=2.0, keys=False, pivot_radius=0, fly=True),
        _Clock(),
    )
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        for code in (_UP, _LEFT):
            walk.key_callback(code)
        walk.apply(handle)
        clock.advance(0.25)
        for code in (_UP, _LEFT):
            walk.key_callback(code)
        walk.apply(handle)
    # forward + strafe travels 0.5 m in total, not 0.5 m along each axis
    assert math.hypot(*handle.cam.lookat[:2]) == pytest.approx(0.5)


def test_walk_keys_cap_the_step_of_a_stalled_frame():
    # A frame that took seconds (a load, a breakpoint) must not teleport the camera across the world.
    walk = WalkKeys(speed=2.0, hold_s=0.35, keys=False, pivot_radius=0, fly=True)
    handle, clock = _walk_handle(), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _held_frame(walk, handle, _UP)
        clock.advance(30.0)
        _held_frame(walk, handle, _UP)
    assert handle.cam.lookat[0] == pytest.approx(2.0 * 0.35)  # clamped to one hold window


def test_walk_keys_chain_the_callers_key_callback():
    seen = []
    walk = WalkKeys(chain=seen.append, keys=False)
    walk.key_callback(ord("W"))
    walk.key_callback(_DOWN)
    assert seen == [ord("W"), _DOWN]  # every key, walk or not


def test_walk_keys_prefer_the_live_keyboard_state():
    # With the X key state available, the sparse key events are not consulted at all: what is
    # physically down is what flies, and letting go stops the camera on the next frame.
    handle, clock = _walk_handle(), _Clock()
    down: set[str] = set()
    walk = WalkKeys(
        speed=2.0,
        pivot_radius=0,
        fly=True,
        keys=SimpleNamespace(held=lambda: set(down), close=lambda: None),
    )
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        down.add("w")
        walk.apply(handle)
        clock.advance(0.25)
        walk.apply(handle)
        assert handle.cam.lookat == pytest.approx([0.5, 0.0, 1.0])

        down.clear()  # key released -- no event needed to notice
        clock.advance(0.25)
        walk.apply(handle)
        assert handle.cam.lookat == pytest.approx([0.5, 0.0, 1.0])


def test_walk_keys_hold_the_mouse_pivot_in_front_of_the_eye():
    # The rendered image is defined by the eye and the angles; pulling lookat in to the pivot radius
    # leaves it untouched while giving MuJoCo's orbit-drag a first-person pivot to turn about.
    handle = _walk_handle()
    handle.cam.lookat[:] = [10.0, 0.0, 1.0]
    handle.cam.distance = 40.0  # a world that framed its whole floorplan
    eye_before = np.array(handle.cam.lookat) - handle.cam.distance * view_forward(
        handle.cam.azimuth, handle.cam.elevation
    )
    walk = WalkKeys(keys=False, pivot_radius=1.5, fly=True)
    walk.apply(handle)

    assert handle.cam.distance == pytest.approx(1.5)
    assert handle.cam.lookat == pytest.approx(eye_before + 1.5 * view_forward(0.0, 0.0))
    eye_after = np.array(handle.cam.lookat) - handle.cam.distance * view_forward(
        handle.cam.azimuth, handle.cam.elevation
    )
    assert eye_after == pytest.approx(eye_before)  # the view did not move, only the pivot


def test_walk_keys_re_anchor_the_pivot_after_a_zoom():
    handle = _walk_handle()
    walk = WalkKeys(keys=False, pivot_radius=1.5, fly=True)
    walk.apply(handle)
    handle.cam.distance = 6.0  # MuJoCo's wheel zoom grows exactly this radius
    walk.apply(handle)
    assert handle.cam.distance == pytest.approx(1.5)  # ... and it is pulled back in each frame


def test_walk_keys_leave_a_tracking_cameras_pivot_alone():
    handle = _walk_handle(type=int(mujoco.mjtCamera.mjCAMERA_TRACKING), distance=8.0)
    WalkKeys(keys=False, pivot_radius=1.5, fly=True).apply(handle)
    assert handle.cam.distance == pytest.approx(
        8.0
    )  # MuJoCo drives lookat; the radius is the framing


def test_the_window_opens_in_mujocos_own_mouse_mode():
    # Nothing of ours touches the camera until F10 says so: the pivot is the world's framing and the
    # arrows do not travel, so drag, pan and zoom all keep the reach a mouse-only inspection needs.
    handle, walk, clock = _walk_handle(distance=40.0), WalkKeys(speed=2.0, keys=False), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _held_frame(walk, handle, _UP)
        clock.advance(0.25)
        _held_frame(walk, handle, _UP)
    assert handle.cam.distance == pytest.approx(40.0)
    assert handle.cam.lookat == pytest.approx([0.0, 0.0, 1.0])


def test_f10_switches_into_flight_and_back_out_without_moving_the_view():
    handle, walk, clock = _walk_handle(distance=40.0), WalkKeys(speed=2.0, keys=False), _Clock()
    eye = np.array(handle.cam.lookat) - 40.0 * view_forward(0.0, 0.0)
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        walk.key_callback(KEY_F10)
        walk.apply(handle)
        assert handle.cam.distance == pytest.approx(
            1.5
        )  # flight pulls the pivot in front of the eye
        clock.advance(0.25)
        _held_frame(walk, handle, _UP)
        assert handle.cam.lookat[0] == pytest.approx(
            eye[0] + 1.5 + 0.5
        )  # ... and the arrows travel

        flown = np.array(handle.cam.lookat) - 1.5 * view_forward(0.0, 0.0)
        clock.advance(1.0)  # past the debounce, so this is a second switch and not a repeat
        walk.key_callback(KEY_F10)
        walk.apply(handle)
    # Back in mouse mode the world's framing distance is the pivot again -- and with it MuJoCo's
    # distance-scaled pan and zoom -- while the eye stays exactly where the flight left it.
    assert handle.cam.distance == pytest.approx(40.0)
    assert np.array(handle.cam.lookat) - 40.0 * view_forward(0.0, 0.0) == pytest.approx(flown)


def test_a_held_f10_is_one_switch():
    # MuJoCo forwards auto-repeats and no release, so an undebounced toggle would flap while held.
    handle, walk, clock = _walk_handle(), WalkKeys(keys=False), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        for _ in range(5):
            walk.key_callback(KEY_F10)
            clock.advance(0.2)  # the ~5 Hz repeat rate measured from the window
        walk.apply(handle)
    assert walk.fly is True


def test_f10_is_remembered_while_the_world_drives_the_camera():
    # A switch pressed during a tracking shot must not be swallowed: there is no free camera to
    # re-parameterise, but the mode is what the person asked for and holds once the camera is free.
    tracking = _walk_handle(type=int(mujoco.mjtCamera.mjCAMERA_TRACKING), distance=8.0)
    walk, clock = WalkKeys(keys=False), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        walk.key_callback(KEY_F10)
        walk.apply(tracking)
    assert walk.fly is True
    assert tracking.cam.distance == pytest.approx(8.0)  # and its framing was left alone


def test_flight_entered_under_a_tracking_camera_still_has_a_framing_to_give_back():
    # The radius flight borrows is taken on its first *free* frame, not at the switch: a camera the
    # world was driving has no framing to lend, and leaving flight without one would strand the mouse
    # orbiting 1.5 m ahead.
    handle, walk, clock = _walk_handle(distance=40.0), WalkKeys(keys=False), _Clock()
    handle.cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        walk.key_callback(KEY_F10)
        walk.apply(handle)  # into flight, with nothing to borrow from

        handle.cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)  # Esc: the camera is released
        walk.apply(handle)
        assert handle.cam.distance == pytest.approx(1.5)

        clock.advance(1.0)
        walk.key_callback(KEY_F10)
        walk.apply(handle)
    assert handle.cam.distance == pytest.approx(40.0)


def test_f10_reaches_the_callers_key_callback_too():
    seen = []
    WalkKeys(chain=seen.append, keys=False).key_callback(KEY_F10)
    assert seen == [KEY_F10]  # the chain runs first, so no handler downstream loses the key


def test_the_mode_is_shown_in_the_window_and_taken_back_down():
    # A toggle you cannot see is a toggle you will get wrong -- but a label that never leaves would
    # sit in every screenshot, so the notice announces the mode and then clears itself.
    texts, cleared = [], []
    handle = _walk_handle()
    # The overlay is written a slot at a time, so a payload is a list of them; the notice is the one
    # slot up here, and its text2 is what the window shows on the right.
    handle.set_texts = lambda t: texts.append(t[-1][-1])
    handle.clear_texts = lambda: cleared.append(True)
    walk, clock = WalkKeys(keys=False), _Clock()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        walk.apply(handle)  # the opening notice: F10 is not a key anyone guesses
        assert len(texts) == 1 and "F10" in texts[0]
        clock.advance(10.0)
        walk.apply(handle)
        assert cleared == [True]

        walk.key_callback(KEY_F10)
        walk.apply(handle)
        assert len(texts) == 2 and "fly" in texts[1]


def test_a_window_without_the_overlay_api_still_navigates():
    # set_texts is cosmetic and best-effort, exactly like the loading splash: an installed MuJoCo
    # without it, or a window already closing, must cost nothing but the notice.
    walk = WalkKeys(speed=2.0, keys=False, pivot_radius=0, fly=True)
    handle, clock = _walk_handle(), _Clock()
    handle.set_texts = mock.Mock(side_effect=AttributeError("no overlay here"))
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _held_frame(walk, handle, _UP)
        clock.advance(0.25)
        _held_frame(walk, handle, _UP)
    assert handle.cam.lookat[0] == pytest.approx(0.5)


# -- the F1 key list ------------------------------------------------------------------------------


def _help_handle():
    """A stand-in window recording the overlay it is handed, one payload per change."""
    handle = _walk_handle()
    handle.payloads = []
    handle.set_texts = handle.payloads.append
    handle.clear_texts = lambda: handle.payloads.append([])
    return handle


def _last(handle):
    """The text of the last payload, both columns of every slot run together."""
    return " ".join(part for entry in handle.payloads[-1] for part in entry[2:])


def _press_f1(helper, handle, clock=None):
    helper.key_callback(key_catalogue.KEY_F1)
    if clock is not None:
        clock.advance(key_catalogue.DEBOUNCE_S * 2)  # past the auto-repeat guard
    helper.apply(handle)


def _helper(*bindings):
    return HelpKeys(bindings=bindings or key_catalogue.CATALOGUE)


def test_f1_lists_the_keys_and_leaves_the_list_up():
    # Unlike the mode notice, which clears itself: a list you cannot keep open is one you cannot fly
    # with. Simulate's own help is on screen at the same time, in the other corner.
    handle, clock = _help_handle(), _Clock()
    helper = _helper()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _press_f1(helper, handle, clock)
        assert "F10" in _last(handle) and "mouse / fly" in _last(handle)
        clock.advance(10.0)
        helper.apply(handle)
    assert "F10" in _last(handle), "the list took itself down"


def test_a_second_f1_takes_the_list_down():
    handle, clock = _help_handle(), _Clock()
    helper = _helper()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _press_f1(helper, handle, clock)
        _press_f1(helper, handle, clock)
    assert handle.payloads[-1] == []


def test_a_held_f1_is_one_toggle():
    # Auto-repeat delivers several presses from one held key; the list must not strobe.
    handle, clock = _help_handle(), _Clock()
    helper = _helper()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        for _ in range(5):
            helper.key_callback(key_catalogue.KEY_F1)
            clock.advance(0.2)
        helper.apply(handle)
    assert "F10" in _last(handle)


def test_the_list_and_the_mode_notice_are_on_screen_together():
    # set_texts replaces the whole overlay, so this is the regression the slot channel exists for:
    # switching camera mode with the list open must not wipe the list explaining the switch.
    handle, clock = _help_handle(), _Clock()
    helper, walk = _helper(), WalkKeys(keys=False)
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _press_f1(helper, handle, clock)
        walk.apply(handle)  # the opening mode notice
    shown = _last(handle)
    assert "F10" in shown and "mouse  ·  F10 to fly" in shown


def test_f1_reaches_the_callers_key_callback_too():
    seen = []
    HelpKeys(bindings=key_catalogue.CATALOGUE, chain=seen.append).key_callback(key_catalogue.KEY_F1)
    assert seen == [key_catalogue.KEY_F1]  # the chain runs first, so no handler downstream loses it


def test_the_list_names_only_the_keys_this_run_has():
    # A scenario-adapter window has no recorder and no view-save; a run from an MJCF scene has no
    # world YAML to save into. Neither may advertise a key it does not have.
    handle, clock = _help_handle(), _Clock()
    helper = _helper(*key_catalogue.CAMERA, key_catalogue.SHOW_HELP)
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _press_f1(helper, handle, clock)
    shown = _last(handle)
    assert "F10" in shown and "F1" in shown
    assert "F9" not in shown and "F8" not in shown


def test_a_window_without_the_overlay_api_still_takes_the_key():
    # Cosmetic and best-effort, like the splash and the notice.
    handle, clock = _help_handle(), _Clock()
    handle.set_texts = mock.Mock(side_effect=AttributeError("no overlay here"))
    helper = _helper()
    with mock.patch("roqsim.viewer.time.monotonic", clock):
        _press_f1(helper, handle, clock)
    assert helper.shown  # the state is sane, so a second press still turns it off


def test_a_window_lists_the_camera_keys_it_wired_up():
    """What ``launch_viewer`` assembles, not what a test hands in.

    The keys the window's own navigation mode is *for* went missing from the list once, because the
    handler that implements them declared nothing and every test until this one passed its bindings
    in by hand. The merge is the thing to pin: whatever a run wires, the list names.
    """
    listed = key_catalogue.merge(HelpKeys, WalkKeys())
    labels = [b.label for b in listed]
    assert "F10" in labels and "Up/Down" in labels and "Shift" in labels
    assert "F1" in labels


def test_a_window_with_a_recorder_and_a_saver_lists_those_too():
    from roqsim.capture import RecordToggle
    from roqsim.view_save import SaveViewKey

    listed = key_catalogue.merge(HelpKeys, WalkKeys(), RecordToggle(), SaveViewKey())
    assert {"F10", "F9", "F8", "F1"} <= {b.label for b in listed}


def test_a_run_with_nowhere_to_save_a_view_does_not_offer_the_key():
    from roqsim.view_save import SaveViewKey

    listed = key_catalogue.merge(HelpKeys, WalkKeys(), SaveViewKey(savable=False))
    assert "F8" not in {b.label for b in listed}
