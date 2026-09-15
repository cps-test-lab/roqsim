"""Replay a recorded run in the viewer window, and write down the moments worth drawing.

``roqsim sim run.npz`` watches a recording in the window a live run uses: the same free camera, the
same flight keys, the same visualization toggles. Nothing is simulated -- each frame is a state
restored out of the recording through :mod:`roqsim.recording`, so what the window shows is what
happened, not a re-run of it.

**Picking a shot is the reason this exists.** A figure needs one moment of one run seen from one
place, and neither half can be guessed from the command line: ``--at`` is a number nobody knows until
they have watched the run, and a camera is a pose nobody writes by hand. So a replay frames the
moment and then writes it down -- a :mod:`roqsim.shots` document that ``roqsim render`` draws later,
at whatever resolution the figure wants.

**The transport is two windows, because one cannot do it.** ``mujoco.viewer.launch_passive`` takes a
key callback and nothing else: there is no mouse callback and no way to add a widget, so a bar drawn
in that window could never be dragged and there would be nowhere to type a timestamp. The slider,
the time box and the buttons are therefore a small tkinter window beside it
(:mod:`roqsim.transport_window`), which also *drives* the replay -- one clock, one writer of
``MjData``. Without a display for it, or with the window declined, the overlay bar and the keys below
carry a replay on their own.

**The keys a replay binds** are the ones a live run has no use for here: F8 saves a camera into the
world a live run has, F9 toggles a recording take a replay has none of. A replay declares its own
meanings for them, and the window's F1 list shows what this run actually has.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from . import keys as keybind
from . import overlay
from .key_state import KeyState
from .playback import Timeline
from .recording import open_recording
from .shots import append_shot, read_shots, shot_document

log = logging.getLogger(__name__)

#: The overlay slot this module owns. Its own, so the camera-mode notice and the F1 list can be up
#: at the same time without either taking the bar down.
SLOT_BAR = "replay"

#: The overlay heading a replay's keys list under.
GROUP_REPLAY = "replay"

#: GLFW keycodes for the two function keys nothing in roqsim claims.
KEY_F11, KEY_F12 = 300, 301

#: How many samples a scrub key moves with Shift held. Coarse enough to cross a long recording,
#: small enough that the fine step is still one press away.
SCRUB_SPRINT = 10

SCRUB = keybind.KeyBinding(
    "replay.scrub",
    GROUP_REPLAY,
    "F11/F12",
    "scrub back / forward, Shift jumps",
    (keybind.Key(KEY_F11, "F11", "back"), keybind.Key(KEY_F12, "F12", "forward")),
)
PLAY = keybind.KeyBinding(
    "replay.play", GROUP_REPLAY, "F8", "play / pause", (keybind.Key(keybind.KEY_F8),)
)
SHOT = keybind.KeyBinding(
    "replay.shot", GROUP_REPLAY, "F9", "write this moment as a shot", (keybind.Key(keybind.KEY_F9),)
)

#: How often a replay with no transport window redraws while paused, so the camera stays draggable.
_IDLE_HZ = 60.0


class ReplayKeys:
    """The viewer keys a replay adds: scrub, play/pause, write a shot.

    Edges, not hold state. The passive viewer's key stream is gappy and drops releases, which is why
    camera travel polls X instead (:class:`roqsim.viewer.WalkKeys`) -- but a scrub step, a toggle and
    a capture are all edges, which is the one thing that stream is reliable for. Shift is the
    exception: it is a *held* modifier, so it is polled the way the walk keys poll it.

    :meth:`key_callback` runs on the UI thread and only counts; the driver takes what accumulated on
    its own thread, where the recording and the shot file are touched.
    """

    key_bindings = (SCRUB, PLAY, SHOT)

    def __init__(self, chain=None) -> None:
        self._chain = chain
        self._tokens = SCRUB.tokens()
        self._scrub = 0
        self._play = 0
        self._shot = 0
        self._last_play = 0.0
        self._last_shot = 0.0
        self._shift = KeyState.open({"Shift_L": "shift", "Shift_R": "shift"})

    def key_callback(self, keycode: int) -> None:
        """UI thread. Count what was pressed and return."""
        if self._chain is not None:
            self._chain(keycode)
        code = int(keycode)
        if (token := self._tokens.get(code)) is not None:
            step = SCRUB_SPRINT if self._shift_held() else 1
            self._scrub += step if token == "forward" else -step
        elif code == keybind.KEY_F8:
            self._play += self._accept("_last_play")
        elif code == keybind.KEY_F9:
            self._shot += self._accept("_last_shot")

    def _accept(self, field: str) -> int:
        """1 for a press that is not an auto-repeat of the one before it, else 0."""
        now = time.monotonic()
        if now - getattr(self, field) < keybind.DEBOUNCE_S:
            return 0
        setattr(self, field, now)
        return 1

    def _shift_held(self) -> bool:
        return bool(self._shift and "shift" in self._shift.held())

    def take_scrub(self) -> int:
        """The net samples asked for since the last call, collapsed into one move."""
        pending, self._scrub = self._scrub, 0
        return pending

    def take_play(self) -> bool:
        """Whether play/pause was pressed an odd number of times since the last call."""
        pending, self._play = self._play, 0
        return bool(pending % 2)

    def take_shot(self) -> bool:
        pending, self._shot = self._shot, 0
        return bool(pending)

    def close(self) -> None:
        if self._shift is not None:
            self._shift.close()
            self._shift = None


class Replay:
    """One opened recording shown in one window: where it is, and what a shot of it says.

    Free of tkinter and of any loop of its own, so the transport window and the bare-viewer fallback
    drive the same object and cannot disagree about which sample is showing.
    """

    def __init__(
        self,
        rec,
        handle,
        *,
        state: str | Path,
        shots: str | Path,
        project: str | Path = ".",
        png_dir: str | Path | None = None,
        render_size: str = "1920x1080",
        no_ceiling: bool = False,
        source: dict | None = None,
        keys: ReplayKeys | None = None,
        world: str | Path | None = None,
    ) -> None:
        self.rec = rec
        #: The world the recording was rebuilt from where its own provenance could not do it; a
        #: shot names it too, so its render rebuilds the same way.
        self.world = str(world) if world else None
        self.handle = handle
        self.timeline = Timeline(rec.times, float(rec.fps))
        self.keys = keys
        self.state = state
        self.shots = Path(shots)
        self.project = Path(project)
        self.png_dir = Path(png_dir) if png_dir else None
        self.render_size = render_size
        self.no_ceiling = no_ceiling
        self.source = source
        self.playing = False
        self.speed = 1.0
        #: Follow the camera the run was watched through, where it recorded one. A shot taken while
        #: following states no view, which is what leaves the render following it too.
        self.follow_recorded = bool(rec.has_camera)
        self._taken = [doc["id"] for doc in read_shots(self.shots)]

    @property
    def shot_ids(self) -> tuple[str, ...]:
        """The ids already in the shots file, this session's included -- what a new one avoids."""
        return tuple(self._taken)

    # -- what is on screen ---------------------------------------------------------------------

    @property
    def sample(self):
        """The recording's sample for the cursor. Re-poses the shared ``MjData``; never cache it."""
        return self.rec.at(self.timeline.time)

    def show(self) -> None:
        """Restore the cursor's sample into the window and draw the bar over it."""
        sample = self.sample
        if self.follow_recorded and sample.camera is not None:
            self._point_at(sample.camera)
        self._draw_bar()
        self.handle.sync()

    def _point_at(self, camera) -> None:
        """Put the window's free camera where the recorded one was, under the viewer's lock."""
        try:
            with self.handle.lock():
                cam = self.handle.cam
                cam.lookat[:] = camera.lookat
                cam.distance = camera.distance
                cam.azimuth = camera.azimuth
                cam.elevation = camera.elevation
        except Exception as err:  # noqa: BLE001 - a closing window must not take the replay down
            log.debug("replay: camera not applied: %s", err)

    def _draw_bar(self) -> None:
        """The playback bar, bottom right -- the corner the camera-mode notice does not use."""
        import mujoco

        overlay.set_text(
            self.handle,
            SLOT_BAR,
            font=mujoco.mjtFontScale.mjFONTSCALE_150,
            gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            text1="playing" if self.playing else "paused",
            text2=f"{self.timeline.bar()} {self.timeline.label()}",
        )

    # -- moving through it ---------------------------------------------------------------------

    def tick(self, elapsed: float) -> None:
        """Advance one frame: act on the keys, play if playing, and redraw.

        Redraws even while paused, so the window stays responsive to a camera drag -- the same reason
        the live loop syncs on a paused step.
        """
        if self.keys is not None:
            if steps := self.keys.take_scrub():
                self.playing = False
                self.timeline.step(steps)
            if self.keys.take_play():
                self.playing = not self.playing
            if self.keys.take_shot():
                self.add_shot()
        if self.playing:
            _index, hit_end = self.timeline.advance(elapsed, self.speed)
            if hit_end:
                self.playing = False
        self.show()

    def seek_time(self, when: float) -> int:
        index = self.timeline.seek_time(when)
        self.show()
        return index

    def seek_index(self, index: int) -> int:
        landed = self.timeline.seek_index(index)
        self.show()
        return landed

    def step(self, count: int) -> int:
        landed = self.timeline.step(count)
        self.show()
        return landed

    # -- writing a shot ------------------------------------------------------------------------

    def add_shot(self, label: str = "") -> dict:
        """Append the cursor's moment, framed as it is on screen, to the shots file."""
        sample = self.sample
        camera = None if self.follow_recorded else live_camera(self.handle)
        doc = shot_document(
            self.rec,
            sample,
            camera,
            state=self.state,
            project=self.project,
            label=label,
            size=self.render_size,
            no_ceiling=self.no_ceiling,
            source=self.source,
            taken=tuple(self._taken),
            world=self.world,
        )
        # The id is what names the file, and only shot_document can hand it out, so the directory is
        # joined on afterwards rather than guessed at beforehand.
        if self.png_dir is not None:
            doc["png"] = str(self.png_dir / f"{doc['id']}.png")
        count = append_shot(self.shots, doc)
        self._taken.append(doc["id"])
        log.info(
            "shot %s at %.3f s -> %s (%d in the file)", doc["id"], doc["at"], self.shots, count
        )
        return doc


def live_camera(handle):
    """A snapshot of the window's current camera, or ``None`` when there is no window left.

    ``mujoco.viewer`` hands ``handle.cam`` to its C++ ``Simulate`` by reference, so reading it is
    reading what the person is looking at, mouse drags and arrow-key flight included. Read under the
    viewer's lock, because the render thread reads the same struct while rasterising.
    """
    import mujoco

    if handle is None:
        return None
    try:
        with handle.lock():
            cam = handle.cam
            out = mujoco.MjvCamera()
            out.type, out.fixedcamid, out.trackbodyid = cam.type, cam.fixedcamid, cam.trackbodyid
            out.lookat[:] = cam.lookat
            out.distance, out.azimuth, out.elevation = cam.distance, cam.azimuth, cam.elevation
            return out
    except Exception:  # noqa: BLE001 - a closing window must not take the shot down
        return None


def is_recording(target: str) -> bool:
    """True when ``target`` names a run recording, which is what selects the replay path.

    The extension decides, as it does for ``roqsim render``'s output and for a mesh target: a
    recording is the only ``.npz`` anything hands ``roqsim sim``, and one that is not a roqsim
    recording is refused by name when it is opened rather than guessed at here.
    """
    return Path(target).suffix.lower() == ".npz"


def run_replay(
    state: str,
    *,
    at: float | None = None,
    view: dict | None = None,
    shots: str | Path | None = None,
    project: str | Path | None = None,
    png_dir: str | Path | None = None,
    render_size: str = "1920x1080",
    no_ceiling: bool = False,
    transport_window: bool = True,
    left_ui: bool = False,
    right_ui: bool = False,
    source: dict | None = None,
    world: str | Path | None = None,
) -> int:
    """Open ``state`` in the viewer and block until the window closes. Returns an exit code.

    ``view`` is a ``sim.view`` block (what ``--set sim.view.azimuth=90`` produces), merged over the
    recording world's own. ``world`` names the world to rebuild from instead of the recording's own
    provenance -- for a run whose world loaded files from beside itself that are not beside the
    recording; the provenance check still refuses one that does not match. Returns ``0`` once the
    window has been open, ``2`` with no display for it.
    """
    from .viewer import (
        GL_HELP,
        DisplayError,
        apply_viewer_keys,
        close_viewer,
        has_display,
        launch_viewer,
        setup_camera,
    )
    from .window_branding import brand_window_async

    if not has_display():
        log.error("no DISPLAY: replaying a run needs a graphical session.")
        return 2

    path = Path(state)
    rec = open_recording(path)
    model, ctx = rec.build(str(world) if world else None, no_ceiling=no_ceiling)
    # A stated camera merges over the recording world's own sim.view, exactly as it does for a render
    # of one moment, so opening a replay and rendering a shot of it frame the scene the same way.
    stated = dict(view or {})
    opening_view = {**(rec.view or {}), **stated}

    handler = ReplayKeys()
    try:
        handle = launch_viewer(
            model,
            ctx.data,
            left_ui=left_ui,
            right_ui=right_ui,
            key_callback=handler.key_callback,
            key_sources=(handler,),
        )
    except Exception as err:  # noqa: BLE001 - any GL init failure maps to the same guidance
        raise DisplayError(GL_HELP.format(err=err)) from err

    try:
        brand_window_async(model, name=path.name)
        setup_camera(handle, opening_view, ctx)
        replay = Replay(
            rec,
            handle,
            state=path,
            shots=shots or path.with_name("shots.yaml"),
            project=project or Path.cwd(),
            png_dir=png_dir,
            render_size=render_size,
            no_ceiling=no_ceiling,
            source=source,
            keys=handler,
            world=world,
        )
        # An explicit --view is a framing the caller chose, so it wins over following the recording's
        # own camera; without one, a recording that carries a camera opens through it.
        if stated:
            replay.follow_recorded = False
        # A stated --at is checked against the recording, which refuses one outside it by name.
        # Seeking clamps, which is right for a slider and wrong for a time somebody typed: a render
        # of the moment they asked for would silently be a render of the last one instead.
        if at is not None:
            rec.index_at(float(at))
        replay.seek_time(rec.span[0] if at is None else at)
        _drive(replay, handle, transport_window, apply_viewer_keys)
    finally:
        handler.close()
        close_viewer(handle)
    return 0


def _drive(replay: Replay, handle, transport_window: bool, apply_viewer_keys) -> None:
    """Hand the replay to the transport window, or run the bare-viewer loop over it."""
    if transport_window:
        try:
            from .transport_window import run_transport

            run_transport(replay, handle, apply_viewer_keys)
            return
        except ImportError as err:
            log.warning("no transport window (%s); replaying with the overlay bar and keys.", err)
    period = 1.0 / _IDLE_HZ
    last = time.perf_counter()
    while handle.is_running():
        now = time.perf_counter()
        elapsed, last = now - last, now
        replay.tick(elapsed)
        apply_viewer_keys(handle)
        time.sleep(max(0.0, period - (time.perf_counter() - now)))
