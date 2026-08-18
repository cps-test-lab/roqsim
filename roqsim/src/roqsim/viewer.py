"""Interactive MuJoCo passive viewer, shared by the two drivers.

Both the standalone :mod:`roqsim.runner` (which owns the loop) and the scenario-execution adapter
:class:`roqsim.scenario_adapter.MujocoSim` (which is stepped by the framework) can show the same
on-screen window. The window management lives here so neither driver duplicates it:

* :func:`apply_view` maps the world's ``sim.view`` camera block onto the passive viewer, while
  :func:`ui_kwargs` maps the caller's own side-panel switches (run-level, never world config).
* :class:`TrackingCamera` / :func:`setup_camera` attach the camera to a robot (MuJoCo tracking) and,
  with ``follow_heading``, chase its heading; both drivers apply it the same way.
* :class:`PassiveViewer` wraps ``mujoco.viewer.launch_passive`` for the adapter, which -- unlike the
  runner -- cannot use a single ``with`` block (it opens in one call and syncs/closes in later ones).
* :class:`WalkKeys` gives that window a second navigation mode -- arrow-key *flight*, so a
  building-sized world can be flown through rather than only circled from outside -- and **F10**
  switches between the two. :func:`launch_viewer` wires one up for every driver; the driver's render
  loop then calls :func:`apply_walk` per frame. It shares :func:`roqsim.walk_delta` with the
  scene-review window, so both fly the same way.

  The window opens in MuJoCo's own mouse mode, untouched: orbit, pan and zoom all keep their full
  reach, which is what a mouse-only inspection needs. Flight is what F10 opts into, and the drag
  itself stays MuJoCo's -- it is C++ with no hook -- but is made to *feel* first-person by holding
  its pivot a metre or so in front of the eye (:func:`roqsim.rendering.set_orbit_radius`). Note the
  shape of that fix: a re-parameterisation the renderer cannot tell apart, not a correction.
  Post-correcting each drag from the driver loop was tried and makes the window flicker between the
  orbited and the corrected pose, because Simulate renders on its own clock; do not reintroduce it.
  The same property is what makes the mode switch itself free: entering and leaving flight only
  re-spells the eye, so neither transition moves the picture.
* :func:`launch_viewer` / :func:`close_viewer` are the open/close pair every driver must use, because
  ``Handle.close()`` alone leaves the window's teardown racing the process exit (see
  :func:`close_viewer`).

``mujoco.viewer`` is imported lazily so a headless/k8s process that never opens a window needs no
on-screen GL context.
"""

from __future__ import annotations

import ctypes.util
import logging
import math
import os
import subprocess
import sys
import threading
import time
import weakref

import mujoco
import numpy as np

from .gl import DEFAULT_MUJOCO_GL
from .key_state import KeyState
from .rendering import set_orbit_radius, walk_delta
from .window_title import retitle_window_async

log = logging.getLogger(__name__)


class DisplayError(RuntimeError):
    """No usable on-screen GL context for the interactive viewer (see :data:`GL_HELP`)."""


def has_display() -> bool:
    """True if an X ``DISPLAY`` is set, i.e. a window can plausibly be opened.

    ``mujoco.viewer.launch_passive`` hard-fails (native GLFW abort that exits the process, not a
    catchable Python exception) when there is no display, so callers that want a window must check
    this first and fail clearly instead of relying on ``try/except``. The GL *backend* (glfw vs the
    offscreen egl/osmesa) is a separate, self-healing choice made elsewhere.
    """
    return bool(os.environ.get("DISPLAY"))


def _resolve_glew() -> str | None:
    """Return an absolute path to the system libGLEW, or ``None`` if it can't be located.

    Prefer an absolute path over the bare soname: a bare ``libGLEW.so.2.2`` in ``LD_PRELOAD``
    is silently dropped by ``ld.so`` if it fails to resolve it in the re-exec'd process, which
    reintroduces the very ``gladLoadGL`` failure we are trying to prevent. An absolute path can't
    fail that way and exactly matches the known-good manual ``LD_PRELOAD=/usr/.../libGLEW.so``.

    Portable: :func:`ctypes.util.find_library` and ``ldconfig -p`` both consult the loader cache,
    so no arch- or distro-specific path is hard-coded.
    """
    cand = ctypes.util.find_library("GLEW")
    if cand and os.path.isabs(cand) and os.path.exists(cand):
        return cand  # some platforms already return a full path
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    want = cand or "libGLEW.so"
    for line in out.splitlines():
        # "  libGLEW.so.2.2 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGLEW.so.2.2"
        name = line.strip().split(" ", 1)[0]
        if "=>" in line and (name == want or name.startswith("libGLEW.so")):
            path = line.split("=>", 1)[1].strip()
            if os.path.exists(path):
                return path
    return cand if cand and os.path.exists(cand) else None


def ensure_gl_preload() -> None:
    """Preload the system libGLEW, re-exec'ing once, before opening the on-screen viewer.

    On many Linux + GL-driver combinations MuJoCo's windowed path aborts at startup with
    ``gladLoadGL error`` unless libGLEW is in the global symbol namespace first; the fix is
    ``LD_PRELOAD=<libGLEW>``. ``LD_PRELOAD`` is only read by the loader at process startup
    (a later ``os.environ`` write does not reach the lazy ``dlopen`` inside the viewer), so we
    set it and re-exec ourselves exactly once. A sentinel env var breaks the re-exec loop.

    Portable by construction: the library is located via :func:`_resolve_glew` (the loader cache),
    not a hard-coded multiarch path, so it works regardless of CPU arch, distro, or GPU. It is a
    silent no-op when GLEW is already preloaded, not installed, or opted out via
    ``ROQSIM_NO_GL_PRELOAD`` -- a machine whose GL stack is genuinely broken still fails loudly
    later with :data:`GL_HELP` guidance.

    Call this before opening a window (``not headless`` and :func:`has_display`); it must run
    before ``mujoco.viewer`` loads GL.
    """
    if os.environ.get("ROQSIM_GL_PRELOADED"):
        return  # already re-exec'd in this chain; never loop
    if os.environ.get("ROQSIM_NO_GL_PRELOAD"):
        return  # explicit opt-out
    existing = os.environ.get("LD_PRELOAD", "")
    if "libGLEW" in existing:
        return  # the environment already handles it; don't fight the user
    lib = _resolve_glew()
    if not lib:
        return  # can't help here; GL_HELP covers the "install the GL stack" case
    os.environ["LD_PRELOAD"] = f"{lib} {existing}".strip()
    os.environ["ROQSIM_GL_PRELOADED"] = "1"
    # Visible on stderr (logging is not configured yet at this point) so a re-exec -- and which
    # libGLEW it preloaded -- is observable; ROQSIM_NO_GL_PRELOAD=1 silences it by opting out.
    print(f"roqsim: preloading {lib} for the on-screen viewer (re-exec)", file=sys.stderr)
    # orig_argv, not argv: this re-runs the process, so it needs the command line the interpreter was
    # actually started with. sys.argv is a mutable display value -- the command tree rewrites argv[0]
    # so a tool's usage line names the subcommand the user typed -- and re-exec'ing that spelling
    # looks for a file called "roqsim sim".
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


def prepare_viewer_gl() -> None:
    """Apply roqsim's GL defaults before any GL is loaded, then (maybe) re-exec for the viewer.

    Two independently-overridable defaults, applied before opening a window:

    * ``MUJOCO_GL`` defaults to :data:`DEFAULT_MUJOCO_GL` (the offscreen/camera backend) -- override
      by setting ``MUJOCO_GL`` yourself.
    * libGLEW is preloaded for the glfw viewer via :func:`ensure_gl_preload` -- opt out with
      ``ROQSIM_NO_GL_PRELOAD``.

    Call before opening a window (``not headless`` and :func:`has_display`). The ``MUJOCO_GL``
    default is set before the possible re-exec in :func:`ensure_gl_preload`, so the re-exec'd
    process inherits it.

    In practice the ``setdefault`` is a no-op: :func:`roqsim.gl.select_offscreen_gl` already ran at
    ``import roqsim``, which is far enough ahead of ``import mujoco`` to matter -- and this is not.
    It is kept because it still decides the windowed case under ``ROQSIM_NO_GL_SELECT``, and
    because a default stated where the window is opened is the one a reader looks for.
    """
    os.environ.setdefault("MUJOCO_GL", DEFAULT_MUJOCO_GL)
    ensure_gl_preload()


GL_HELP = (
    "no usable on-screen GL context for the interactive viewer.\n"
    "  - On a headless/remote host, run headless (no viewer).\n"
    "  - Otherwise the OpenGL install is incomplete; try:\n"
    "      sudo apt install libglfw3 libgl1 libegl1 libglvnd0\n"
    "  - The viewer window is always glfw; MUJOCO_GL selects only the OFFSCREEN (camera)\n"
    "    backend, so MUJOCO_GL=egl does NOT make the viewer offscreen -- it just gives\n"
    "    cameras their own context. roqsim defaults MUJOCO_GL=egl; set it yourself to override.\n"
    "  Underlying error: {err}"
)


def apply_view(viewer, view: dict | None) -> None:
    """Point the free camera per the world's ``sim.view`` block (any subset of keys).

    Omitted keys keep MuJoCo's model-derived default, so a world can nudge just the azimuth. Mutating
    viewer camera state is done under ``viewer.lock()`` per the passive-viewer contract.
    """
    if not view:
        return
    with viewer.lock():
        cam = viewer.cam
        if "lookat" in view:
            cam.lookat[:] = [float(v) for v in view["lookat"]]
        if "distance" in view:
            cam.distance = float(view["distance"])
        if "azimuth" in view:
            cam.azimuth = float(view["azimuth"])
        if "elevation" in view:
            cam.elevation = float(view["elevation"])
    viewer.sync()


def ui_kwargs(left_ui: bool = False, right_ui: bool = False) -> dict:
    """Map the panel switches onto ``launch_passive``'s toggles (default: both hidden).

    roqsim opens with both Simulate side panels hidden for a clean view. Showing one is a
    *run-level* choice (the runner's ``--left-ui`` / ``--right-ui``), not world config: which camera
    a world wants is part of the scene, whereas wanting the panels is a property of one interactive
    session. Either way the user can still toggle them at runtime with Tab / Shift+Tab.
    """
    return {"show_left_ui": bool(left_ui), "show_right_ui": bool(right_ui)}


#: Viewer handle -> the threads ``launch_passive`` started for it. Weakly keyed, so a handle the
#: caller dropped without closing does not pin its threads here.
_VIEWER_THREADS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

#: How long :func:`close_viewer` waits for MuJoCo's render thread to finish. Teardown takes ~10 ms;
#: the cap only exists so a wedged GL driver degrades to the old racy exit instead of hanging here.
_CLOSE_TIMEOUT_S = 5.0


#: Camera travel speed under the arrow keys (m/s at real scene scale -- a brisk walk).
WALK_SPEED_MS = 2.5

#: Held-Shift multiplier on that speed, matching the scene-review window's sprint.
WALK_SPRINT = 3.0

#: How far in front of the eye MuJoCo's mouse pivot is held **in flight mode** (metres); see
#: :func:`roqsim.rendering.set_orbit_radius`. Close enough that a drag reads as turning your head rather
#: than circling a point across the room, far enough that MuJoCo's distance-scaled pan and zoom still
#: cover ground. 0 disables it, leaving the world's own framing as the pivot.
PIVOT_RADIUS_M = 1.5

#: GLFW keycode for F10 -- the navigation-mode toggle. Simulate claims F1-F7
#: (help/info/profiler/sensor/fullscreen/frame/label), roqsim's recording toggle is F9 and its view-save
#: is F8, which leaves F10; and only a function key will do, because Simulate binds every letter to a
#: visualization or rendering flag and the passive viewer's callback runs *in addition to* its own
#: handling. Same constraint that put travel on the arrows.
KEY_F10 = 299

#: Auto-repeat delivers several presses from one held F10 (~0.2 s apart); above that, below a
#: deliberate double-press. Matches :data:`roqsim.view_save.DEBOUNCE_S`.
_TOGGLE_DEBOUNCE_S = 0.4

#: How long the mode notice stays on screen after a switch. A toggle you cannot see is a toggle you
#: will get wrong -- but a label that never leaves would sit in every screenshot of the window.
_NOTICE_S = 2.5

#: What the notice says in either mode: the mode, then how to move in it and how to leave.
_NOTICE = {
    False: ("camera", "mouse  ·  F10 to fly"),
    True: ("camera", "fly  ·  arrows travel, PgUp/PgDn height, Shift 3x  ·  F10 for the mouse"),
}

#: Fallback only: how long after its last event a direction still counts as held, where the X key
#: state of :mod:`roqsim.key_state` is unavailable and the sparse event stream is all there is. The
#: camera then coasts this long after a key is let go -- shorter, and MuJoCo's gappy auto-repeat
#: reads as a release mid-flight.
_KEY_HOLD_S = 0.35

#: The token Shift carries through the held-key set -- a modifier, not a direction.
_SPRINT = "shift"

#: GLFW keycodes (what MuJoCo hands the key callback) -> the walk direction they mean.
#:
#: Letters are unusable in this window: MuJoCo's Simulate binds W/A/S/D/E/Q to visualization flags
#: (wireframe, static bodies, select point, ...) whether or not a side panel is open, so a WASD walk
#: would toggle rendering as it moved. The arrows and Page Up/Down are free while the simulation
#: runs -- Simulate's arrow bindings only step the physics while it is *paused*.
_WALK_KEYCODES = {
    265: "w",  # Up: forward along the view
    264: "s",  # Down: back
    263: "a",  # Left: strafe left
    262: "d",  # Right: strafe right
    266: "e",  # Page Up: rise
    267: "q",  # Page Down: descend
    340: _SPRINT,  # Left Shift: fly faster
    344: _SPRINT,  # Right Shift
}

#: The same keys as X keysym names, for the :mod:`roqsim.key_state` poll.
_WALK_KEYSYMS = {
    "Up": "w",
    "Down": "s",
    "Left": "a",
    "Right": "d",
    "Prior": "e",
    "Next": "q",
    "Shift_L": _SPRINT,
    "Shift_R": _SPRINT,
}


def _show_notice(handle, text: tuple[str, str]) -> None:
    """Put the mode notice in the window's lower-left corner. Cosmetic, so never raises.

    ``set_texts`` is the passive viewer's own overlay channel (the loading splash of :mod:`roqsim.splash`
    uses its ``set_images`` sibling, and nothing else in roqsim writes text), so this clobbers nobody. An
    installed MuJoCo without the API, or a window already closing, simply shows no notice -- the same
    best-effort stance the splash takes.
    """
    try:
        handle.set_texts(
            (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, *text)
        )
    except Exception as err:  # noqa: BLE001 — a notice never breaks navigation
        log.debug("camera-mode notice skipped: %s", err)


def _clear_notice(handle) -> None:
    """Take the mode notice back down. Cosmetic, so never raises."""
    try:
        handle.clear_texts()
    except Exception as err:  # noqa: BLE001
        log.debug("clearing the camera-mode notice failed: %s", err)


class WalkKeys:
    """The viewer's two navigation modes, and the F10 that switches between them.

    **Mouse mode** (the default) is MuJoCo's window exactly as it ships: drag orbits ``lookat``,
    right-drag pans, the wheel zooms, and this class does nothing at all to the camera. It is the mode
    a mouse-only inspection wants, because orbit, pan and zoom all keep their full reach.

    **Flight mode** trades that away for getting *inside* a building-sized world, which orbiting alone
    can only circle from outside. Moving ``lookat`` carries the eye with it (the free camera sits a
    fixed distance behind it), so mouse aims and keys travel: Up/Down fly along the view direction,
    Left/Right strafe level, Page Up/Down change height, and holding Shift flies :data:`WALK_SPRINT`
    times faster -- see :data:`_WALK_KEYCODES` for why not WASD. :meth:`apply` also keeps the orbit
    point :data:`PIVOT_RADIUS_M` in front of the eye (:func:`roqsim.rendering.set_orbit_radius`), which
    is what makes the drag read as a look rather than a swing around some point across the room; the
    price is MuJoCo's pan and zoom, which scale with ``distance`` and therefore shrink with it.

    :data:`KEY_F10` switches, and switching is *free*: the orbit radius is a re-parameterisation, not
    a correction (the rendered image is identical either way), so both transitions change the pivot
    and nothing that is drawn. Leaving flight restores the radius the camera had on entering it, which
    is the world's own framing -- and with it the reach of the mouse. Because it is a
    re-parameterisation there is also nothing for Simulate's own render loop to fight, unlike post-hoc
    rewriting of a drag, which flickers.

    Motion is *not* applied in the key callback, and where possible not driven by key events at all.
    The window forwards presses and auto-repeats but no releases, and the repeats are throttled and
    gappy (~5 Hz with second-long holes, measured against a 25 Hz keyboard), so a camera stepped per
    event hops, stalls mid-flight, and cannot tell a release from a gap. Instead
    :class:`roqsim.key_state.KeyState` reads the keyboard's live X11 bitmap, and :meth:`apply` -- called
    by the driver once per rendered frame -- integrates real elapsed time for whatever is down.

    Where that is unavailable (Wayland, macOS, no ``DISPLAY``, no libX11) it degrades to the event
    stream: :meth:`key_callback` timestamps each direction and it counts as held for
    :data:`_KEY_HOLD_S` after its last event. Hops and a short coast on release, but it still flies.
    """

    def __init__(
        self,
        speed: float = WALK_SPEED_MS,
        sprint: float = WALK_SPRINT,
        hold_s: float = _KEY_HOLD_S,
        chain=None,
        keys=None,
        pivot_radius: float = PIVOT_RADIUS_M,
        fly: bool = False,
    ):
        """``keys`` is the key-state source: ``None`` opens one (and falls back when it cannot),
        anything falsy forces the event-stream path, and an object with ``held()``/``close()``
        substitutes for it. ``pivot_radius`` of 0 leaves the mouse orbiting whatever the world framed
        even in flight, and ``fly`` chooses the mode the window opens in.
        """
        self.speed = float(speed)
        self.sprint = float(sprint)
        self.hold_s = float(hold_s)
        self.pivot_radius = float(pivot_radius)
        self.fly = bool(fly)
        self._chain = chain
        self._keys = KeyState.open(_WALK_KEYSYMS) if keys is None else keys
        self._seen: dict[str, float] = {}  # direction -> monotonic time of its last key event
        self._last_apply: float | None = None
        self._toggles = 0  # F10 presses the UI thread has accepted and no frame has consumed yet
        self._last_toggle = 0.0
        self._orbit_radius: float | None = None  # the mouse-mode radius flight is borrowing
        self._notice_until: float | None = None  # when the on-screen mode notice comes back down
        self._announced = False  # the opening notice, so the other mode is discoverable at all

    def key_callback(self, keycode: int) -> None:
        """The ``launch_passive`` key callback. Runs on MuJoCo's UI thread; only records a timestamp.

        ``chain`` (a caller's own callback) is invoked first with the raw keycode, so wiring this in
        never swallows anyone else's keys. The timestamp itself is the fallback hold signal, unused
        while the X key state is available; F10 is counted rather than acted on, because the mode
        switch touches the camera and that belongs to the thread calling :meth:`apply`.
        """
        if self._chain is not None:
            self._chain(keycode)
        code = int(keycode)
        if code == KEY_F10:
            now = time.monotonic()
            if now - self._last_toggle >= _TOGGLE_DEBOUNCE_S:  # one held key is not two switches
                self._last_toggle = now
                self._toggles += 1
            return
        key = _WALK_KEYCODES.get(code)
        if key is not None:
            self._seen[key] = time.monotonic()

    def held(self, now: float) -> list[str]:
        """The directions currently down: the X keyboard state, else the recent key events."""
        if self._keys:
            return sorted(self._keys.held())
        return [k for k, t in self._seen.items() if now - t < self.hold_s]

    def close(self) -> None:
        """Release the X connection, if one was opened."""
        if self._keys:
            self._keys.close()
        self._keys = None

    def _switch(self, cam) -> None:
        """Flip the mode, re-spelling ``cam`` for the one being entered. Renders the same either way.

        ``cam`` is ``None`` when the world is driving the camera itself; the mode is still remembered,
        so a switch pressed during a tracking shot is not swallowed, but there is no free camera to
        re-parameterise and MuJoCo owns the pivot anyway.
        """
        self.fly = not self.fly
        if self.fly:
            return  # the radius flight borrows is taken on its first free frame, in apply()
        if cam is not None and self._orbit_radius is not None:
            # Hand the framing distance back, or the camera would leave flight still orbiting 1.5 m
            # ahead, with the mouse's pan and zoom -- which MuJoCo scales by ``distance`` -- crawling.
            set_orbit_radius(cam, self._orbit_radius)
        self._orbit_radius = None

    def apply(self, handle) -> None:
        """Consume a pending F10, then, in flight mode, hold the pivot in and advance one frame.

        Call once per rendered frame, before ``sync``. In mouse mode nothing here touches the camera
        at all. Travel and the pivot are also skipped while the world drives the camera itself -- a
        tracking or fixed camera owns its pose, and flying it would fight that.
        """
        now = time.monotonic()
        previous, self._last_apply = self._last_apply, now
        keys = self.held(now)
        speed = self.speed * (self.sprint if _SPRINT in keys else 1.0)
        keys = [k for k in keys if k != _SPRINT]  # a modifier, not a direction to travel in
        switched, self._toggles = self._toggles % 2 == 1, 0  # two presses are back where we started
        with handle.lock():
            cam = handle.cam
            free = int(cam.type) == int(mujoco.mjtCamera.mjCAMERA_FREE)
            if switched:
                self._switch(cam if free else None)
            if free and self.fly:
                if self.pivot_radius:
                    if self._orbit_radius is None:
                        # Borrowed on the first free frame rather than at the switch, so that
                        # entering flight under a tracking camera still has a framing to give back
                        # once that camera is released.
                        self._orbit_radius = float(cam.distance)
                    # Pinned every frame, not just on entering flight: a zoom (which is MuJoCo
                    # shrinking or growing exactly this radius) would otherwise walk the pivot back
                    # out into the distance.
                    set_orbit_radius(cam, self.pivot_radius)
                # The first frame of a hold has no measured interval yet; the next one moves.
                if keys and previous is not None:
                    dt = min(now - previous, self.hold_s)  # a stalled frame must not teleport it
                    cam.lookat[:] = np.array(cam.lookat) + walk_delta(
                        cam.azimuth, keys, speed * dt, cam.elevation
                    )
        self._notice(handle, now, switched)

    def _notice(self, handle, now: float, switched: bool) -> None:
        """Show the mode in the window for :data:`_NOTICE_S`, on a switch and once at the start.

        Outside the viewer lock: the overlay is queued through the handle, not written to the camera.
        The opening notice is the only thing that makes the *other* mode discoverable -- F10 is not a
        key anyone guesses -- and it leaves on its own, so it stays out of screenshots of the window.
        """
        if switched or not self._announced:
            self._announced = True
            _show_notice(handle, _NOTICE[self.fly])
            self._notice_until = now + _NOTICE_S
        elif self._notice_until is not None and now >= self._notice_until:
            self._notice_until = None
            _clear_notice(handle)


#: Viewer handle -> its :class:`WalkKeys`, so a driver's render loop can find the one
#: :func:`launch_viewer` wired up without threading it through every call. Weakly keyed, like
#: :data:`_VIEWER_THREADS`.
_VIEWER_WALK: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def apply_walk(handle) -> None:
    """Advance ``handle``'s arrow-key camera travel by one frame; no-op if it has none."""
    walk = _VIEWER_WALK.get(handle)
    if walk is not None:
        walk.apply(handle)


def launch_viewer(
    model, data, *, left_ui: bool = False, right_ui: bool = False, key_callback=None, walk=True
):
    """``mujoco.viewer.launch_passive``, remembering the threads it started for :func:`close_viewer`.

    MuJoCo runs the window on its own daemon threads and hands back only a ``Handle``, so an ordered
    shutdown has to get at those threads; diffing ``threading.enumerate()`` around the call is the way
    to do that without reaching into ``mujoco.viewer`` internals. GL init failures propagate unchanged
    -- callers decide whether to translate them into :class:`DisplayError` or fall back.

    ``walk`` (default on) adds the arrow-key camera travel of :class:`WalkKeys`, chaining to
    ``key_callback`` when the caller passes one. The driver must call :func:`apply_walk` once per
    rendered frame for it to move -- key events alone are far too sparse to fly on.

    Always close the returned handle with :func:`close_viewer`, never ``Handle.close()``.
    """
    import mujoco.viewer  # lazy: headless never needs a display

    walker = WalkKeys(chain=key_callback) if walk else None
    if walker is not None:
        key_callback = walker.key_callback

    before = set(threading.enumerate())
    handle = mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, **ui_kwargs(left_ui, right_ui)
    )
    if walker is not None:
        _VIEWER_WALK[handle] = walker
    fresh = [t for t in threading.enumerate() if t not in before]
    if fresh:
        _VIEWER_THREADS[handle] = fresh
    return handle


def close_viewer(handle, *, timeout: float = _CLOSE_TIMEOUT_S) -> None:
    """Close the window and *wait* for MuJoCo to finish tearing it down. Never raises.

    ``Handle.close()`` only requests the exit: it sets ``exitrequest`` and returns immediately while
    the window, its GL context and the X drawable are destroyed later, on MuJoCo's own daemon thread.
    If the process walks into interpreter shutdown meanwhile, Python's ``atexit`` runs
    ``glfw.terminate`` under that thread's feet -- its in-flight ``glXSwapBuffers`` then targets a
    destroyed drawable, and Xlib's default ``GLXBadDrawable`` handler calls ``exit()`` from a second
    thread while the first is already finalizing. The two shutdowns deadlock and the process hangs
    after its last line of output (a segfault is the same race landing the other way). Both were
    reproducible on any run that opened the window and then exited promptly -- a world that fails to
    load, or ``--steps 1``.

    Joining the threads makes the teardown ordered, which costs ~10 ms.
    """
    threads = _VIEWER_THREADS.pop(handle, [])
    walk = _VIEWER_WALK.pop(handle, None)
    if walk is not None:
        walk.close()  # the arrow-key poll holds its own X connection
    try:
        handle.close()
    except Exception:  # noqa: BLE001 — teardown must never raise
        pass
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            # Nothing left to do but let the process exit the old way; say so rather than hang.
            print(
                f"roqsim: the viewer's {thread.name} did not shut down within {timeout:g}s; "
                "exiting anyway (the process may crash on the way out)",
                file=sys.stderr,
            )


class ViewError(RuntimeError):
    """The world's ``sim.view`` block names a track target that does not exist."""


def _wrap180(deg: float) -> float:
    """Fold an angle difference into (-180, 180] so a drag across the seam is not a 360 jump."""
    return (deg + 180.0) % 360.0 - 180.0


class TrackingCamera:
    """Keeps the camera attached to a robot: MuJoCo tracking, plus an optional yaw-following chase.

    ``track`` names the body to follow (an entity name -- so worlds say ``track: robot``, not the MJCF
    prefix -- or a raw body name). MuJoCo's ``mjCAMERA_TRACKING`` then slaves ``lookat`` to that body
    every frame while ``azimuth``/``elevation``/``distance`` stay mouse-controllable. With
    ``follow_heading: true``, :meth:`update` rewrites ``azimuth`` from the body's yaw each frame so the
    camera rides behind the robot; the configured ``azimuth`` becomes an offset (180 = directly
    behind) and any mouse drag is folded into it, so orbiting still works and holds relative to the
    robot.

    Constructed once the model exists (after ``engine.setup()``), because resolving ``track`` needs
    both the entity registry and the compiled model.
    """

    def __init__(self, view: dict | None, ctx) -> None:
        self.view = dict(view or {})
        self.ctx = ctx
        self.follow_heading = bool(self.view.get("follow_heading", False))
        self.track_body_id: int | None = None
        self._offset = float(self.view.get("azimuth", 0.0))
        self._last_written: float | None = None

        target = self.view.get("track")
        if target is None:
            if self.follow_heading:
                raise ViewError("sim.view.follow_heading needs a 'track' target")
            return
        self.track_body_id = self._resolve(str(target))
        if "lookat" in self.view:
            ctx.logger.warning(
                "sim.view: 'lookat' is ignored while tracking %r (MuJoCo drives it)", target
            )

    @property
    def active(self) -> bool:
        return self.track_body_id is not None

    @property
    def azimuth_offset(self) -> float:
        """The chase cam's angle *behind the robot*, kept current as the mouse drags it.

        Under ``follow_heading`` the live ``cam.azimuth`` is the robot's yaw plus this, rewritten every
        frame -- a world-frame angle that means nothing once the robot has turned. This is the half a
        world can actually state, so it is what :mod:`roqsim.view_save` writes back as ``azimuth``.
        """
        return self._offset

    def _resolve(self, target: str) -> int:
        """Entity name first (so worlds say ``track: robot``, not the MJCF prefix), then body name."""
        entity = self.ctx.entities.get(target)
        body = entity.body if entity is not None and entity.body else target
        bid = mujoco.mj_name2id(self.ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            known = ", ".join(sorted(self.ctx.entities.names())) or "<none>"
            raise ViewError(
                f"sim.view.track={target!r} is neither a known entity ({known}) nor a body "
                f"in the compiled model (looked for body {body!r})"
            )
        return bid

    def apply(self, handle) -> None:
        """Set the tracking camera's launch state. Called once, with the viewer open."""
        with handle.lock():
            cam = handle.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.track_body_id
            if "distance" in self.view:
                cam.distance = float(self.view["distance"])
            if "elevation" in self.view:
                cam.elevation = float(self.view["elevation"])
            if "azimuth" in self.view:
                cam.azimuth = float(self.view["azimuth"])
        handle.sync()

    def update(self, handle) -> None:
        """Re-aim the chase cam at the tracked body's heading. Called each frame before ``sync()``.

        Runs on the physics thread and only reads ``data.xmat``, so the single-writer rule holds. A
        mouse drag since the last frame shows up as a change to ``cam.azimuth``; we fold it into the
        offset instead of stomping it, so orbiting sticks and stays relative to the robot.
        """
        if not self.follow_heading:
            return
        xmat = self.ctx.data.xmat[self.track_body_id]
        yaw = math.degrees(math.atan2(xmat[3], xmat[0]))  # row-major 3x3: atan2(R10, R00)
        with handle.lock():
            cam = handle.cam
            if self._last_written is not None:
                self._offset += _wrap180(cam.azimuth - self._last_written)
            self._last_written = yaw + self._offset
            cam.azimuth = self._last_written


def setup_camera(handle, view: dict | None, ctx, *, preview: bool = False) -> TrackingCamera | None:
    """Apply ``sim.view`` to a freshly opened viewer ``handle``; wire tracking if the world asks.

    Returns a :class:`TrackingCamera` to call :meth:`~TrackingCamera.update` on each frame when the
    world tracks a robot, else ``None`` (a static free camera needs no per-frame work).

    ``preview`` (a single model/robot shown by itself) zooms the camera onto that model when no
    explicit ``sim.view`` is given -- an explicit view always wins.
    """
    camera = TrackingCamera(view, ctx)
    if camera.active:
        camera.apply(handle)
        return camera
    if preview and not view:
        from .rendering import preview_camera

        cam = preview_camera(ctx.model, ctx.data, ctx.entities.all())
        with handle.lock():
            handle.cam.lookat[:] = cam.lookat
            handle.cam.distance = cam.distance
            handle.cam.azimuth = cam.azimuth
            handle.cam.elevation = cam.elevation
        handle.sync()
        return None
    apply_view(handle, view)
    return None


class PassiveViewer:
    """A persistent passive viewer for a driver that doesn't own a ``with`` block.

    Open once (``PassiveViewer(ctx, view)``), :meth:`sync` after each physics step, and :meth:`close`
    at shutdown. It reads ``ctx.model``/``ctx.data`` for the window and ``ctx.entities`` to resolve a
    ``sim.view.track`` target. Opening translates any glad/GLFW/EGL init failure into a
    :class:`DisplayError` with actionable guidance (the same message the runner uses).

    ``view`` carries the camera only; the side panels are the caller's own switches (see
    :func:`ui_kwargs`) and default to hidden, which is what a scripted scenario run wants.
    """

    def __init__(
        self,
        ctx,
        view: dict | None = None,
        *,
        name: str | None = None,
        left_ui: bool = False,
        right_ui: bool = False,
    ):
        try:
            self._handle = launch_viewer(ctx.model, ctx.data, left_ui=left_ui, right_ui=right_ui)
        except Exception as err:  # noqa: BLE001 — any GL init failure maps to the same guidance
            raise DisplayError(GL_HELP.format(err=err)) from err
        retitle_window_async(ctx.model, name=name)
        self._camera = setup_camera(self._handle, view, ctx)

    def is_running(self) -> bool:
        return self._handle.is_running()

    def sync(self) -> None:
        if self._camera is not None:
            self._camera.update(self._handle)
        apply_walk(self._handle)
        self._handle.sync()

    def close(self) -> None:
        close_viewer(self._handle)
