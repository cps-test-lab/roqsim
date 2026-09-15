"""The transport window: the playback bar a replay is scrubbed with.

``mujoco.viewer.launch_passive`` takes a key callback and nothing else -- no mouse callback, no way
to add a widget -- so the viewer window can *draw* a playback bar but never let anyone drag one, and
has nowhere to type a timestamp. This is that half of the interface: a small tkinter window beside
the viewer, holding the slider, the time box, the transport buttons and the button that writes a
shot. The 3D window keeps what it is good at, which is the camera.

**This window drives the replay.** Its ``after`` tick is the only clock: it advances the timeline,
restores the sample and syncs the viewer. The alternative -- a loop of its own in each window --
would be two writers of one ``MjData``, which is the hazard two windows would otherwise carry.

The rendering button does not render here. It runs ``roqsim render`` on the shot just written, in a
subprocess, so the picture comes from the exact command the figure will be built with: a shot whose
flags are wrong fails at the button rather than hours later.

Imported only when a replay asks for it, so a roqsim without tkinter still replays -- the overlay bar
and the keys carry it (see :mod:`roqsim.replay`).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

from .playback import format_time, parse_time
from .shots import render_args

log = logging.getLogger(__name__)

#: How often the window ticks. The recording's own rate is what playback follows; this is the
#: ceiling on how often the tick asks it to.
TICK_HZ = 60.0

#: Playback speeds the menu offers, as multiples of the recording's own rate.
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)

BG = "#1b1b1b"
PANEL = "#262626"
FG = "#e8e8e8"
MUTED = "#9a9a9a"
ACCENT = "#2f6f4f"


def run_transport(replay, handle, apply_viewer_keys) -> None:
    """Open the transport window over ``replay`` and block until either window closes."""
    app = _Transport(replay, handle, apply_viewer_keys)
    app.root.mainloop()


class _Transport:
    """The window, and the tick that drives the replay behind it."""

    def __init__(self, replay, handle, apply_viewer_keys) -> None:
        self.replay = replay
        self.handle = handle
        self.apply_viewer_keys = apply_viewer_keys
        self._syncing = False  # set while the code writes a widget, so its callback is not a seek
        self._render = None  # the Popen of a running `roqsim render`, while one is running
        self._rendering = None  # the shot that Popen is drawing
        self._last = time.perf_counter()

        self.root = tk.Tk()
        self.root.title(f"roqsim replay -- {Path(replay.state).name}")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._refresh()
        self.root.after(int(1000 / TICK_HZ), self._tick)

    # -- layout ---------------------------------------------------------------------------------

    def _build(self) -> None:
        transport = tk.Frame(self.root, bg=BG)
        transport.pack(fill="x", padx=8, pady=(8, 4))
        for label, command, width in (
            ("|<", lambda: self._seek_index(0), 3),
            ("<", lambda: self._step(-1), 3),
            ("> / ||", self._toggle_play, 7),
            (">", lambda: self._step(1), 3),
            (">|", lambda: self._seek_index(len(self.replay.timeline) - 1), 3),
        ):
            button = tk.Button(
                transport, text=label, width=width, command=command, bg=PANEL, fg=FG, relief="flat"
            )
            button.pack(side="left", padx=2)
            if label == "> / ||":
                self.play_button = button

        self.slider = tk.Scale(
            transport,
            from_=0,
            to=max(len(self.replay.timeline) - 1, 1),
            orient="horizontal",
            showvalue=False,
            command=self._on_slider,
            bg=BG,
            fg=FG,
            troughcolor=PANEL,
            highlightthickness=0,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

        self.time_var = tk.StringVar(master=self.root)
        entry = tk.Entry(
            transport,
            textvariable=self.time_var,
            width=9,
            bg=PANEL,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            justify="right",
        )
        entry.pack(side="left")
        entry.bind("<Return>", self._on_time_entered)
        tk.Label(transport, text="s", bg=BG, fg=MUTED).pack(side="left", padx=(2, 8))

        self.where = tk.Label(transport, text="", bg=BG, fg=MUTED, width=14, anchor="w")
        self.where.pack(side="left")

        self.speed_var = tk.StringVar(master=self.root, value="1.0x")
        speeds = tk.OptionMenu(
            transport, self.speed_var, *(f"{s}x" for s in SPEEDS), command=self._on_speed
        )
        speeds.configure(bg=PANEL, fg=FG, relief="flat", highlightthickness=0, width=5)
        speeds.pack(side="left", padx=(8, 0))

        picking = tk.Frame(self.root, bg=BG)
        picking.pack(fill="x", padx=8, pady=4)
        tk.Label(picking, text="label", bg=BG, fg=MUTED).pack(side="left")
        self.label_var = tk.StringVar(master=self.root)
        tk.Entry(
            picking,
            textvariable=self.label_var,
            bg=PANEL,
            fg=FG,
            insertbackground=FG,
            relief="flat",
        ).pack(side="left", fill="x", expand=True, padx=6)

        self.recorded_var = tk.BooleanVar(master=self.root, value=self.replay.follow_recorded)
        if self.replay.rec.has_camera:
            tk.Checkbutton(
                picking,
                text="recorded camera",
                variable=self.recorded_var,
                command=self._on_recorded,
                bg=BG,
                fg=FG,
                selectcolor=PANEL,
                activebackground=BG,
                activeforeground=FG,
                highlightthickness=0,
            ).pack(side="left", padx=6)

        self.add_button = tk.Button(
            picking, text="+ Add shot", command=self._add_shot, bg=ACCENT, fg=FG, relief="flat"
        )
        self.add_button.pack(side="left", padx=2)
        self.png_button = tk.Button(
            picking, text="Add + PNG", command=self._add_and_render, bg=PANEL, fg=FG, relief="flat"
        )
        self.png_button.pack(side="left", padx=2)

        self.shots = tk.Listbox(
            self.root, height=6, bg=PANEL, fg=FG, relief="flat", highlightthickness=0
        )
        self.shots.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        for shot in self.replay.shot_ids:
            self.shots.insert("end", shot)

        self.status = tk.Label(self.root, text=self._shots_note(), bg=BG, fg=MUTED, anchor="w")
        self.status.pack(fill="x", padx=8, pady=(0, 8))

    def _shots_note(self) -> str:
        return f"{len(self.replay.shot_ids)} shot(s) in {self.replay.shots}"

    # -- the clock ------------------------------------------------------------------------------

    def _tick(self) -> None:
        """Advance the replay, keep the viewer's own key work going, and reschedule."""
        if not self.handle.is_running():
            self._close()
            return
        now = time.perf_counter()
        elapsed, self._last = now - self._last, now
        self.replay.tick(elapsed)
        self.apply_viewer_keys(self.handle)
        self._poll_render()
        self._refresh()
        self.root.after(int(1000 / TICK_HZ), self._tick)

    def _refresh(self) -> None:
        """Write the timeline's position into the widgets without that reading as a seek."""
        timeline = self.replay.timeline
        self._syncing = True
        try:
            self.slider.set(timeline.index)
            self.time_var.set(format_time(timeline.time))
        finally:
            self._syncing = False
        self.where.configure(text=f"#{timeline.index} / {len(timeline) - 1}")
        self.play_button.configure(text="||" if self.replay.playing else "> ")

    # -- what the widgets do --------------------------------------------------------------------

    def _on_slider(self, value) -> None:
        if self._syncing:
            return
        self.replay.playing = False
        self.replay.seek_index(int(float(value)))
        self._refresh()

    def _on_time_entered(self, _event=None) -> None:
        try:
            when = parse_time(self.time_var.get())
        except ValueError as err:
            self.status.configure(text=str(err))
            return
        self.replay.playing = False
        self.replay.seek_time(when)
        # Rewritten with the time it landed on: the box must never claim a moment the recording
        # does not have.
        self._refresh()
        self.status.configure(text=self._shots_note())

    def _on_speed(self, _value=None) -> None:
        self.replay.speed = float(self.speed_var.get().rstrip("x"))

    def _on_recorded(self) -> None:
        """Turn following the recorded camera on or off, leaving the view where it is either way."""
        self.replay.follow_recorded = bool(self.recorded_var.get())
        self.replay.show()

    def _seek_index(self, index: int) -> None:
        self.replay.playing = False
        self.replay.seek_index(index)
        self._refresh()

    def _step(self, count: int) -> None:
        self.replay.playing = False
        self.replay.step(count)
        self._refresh()

    def _toggle_play(self) -> None:
        self.replay.playing = not self.replay.playing
        self._refresh()

    # -- writing shots --------------------------------------------------------------------------

    def _add_shot(self) -> dict:
        doc = self.replay.add_shot(self.label_var.get().strip())
        self.shots.insert("end", f"{doc['id']}  t={doc['at']:.3f}")
        self.shots.see("end")
        self.label_var.set("")
        self.status.configure(text=self._shots_note())
        return doc

    def _add_and_render(self) -> None:
        """Write the shot, then draw it with the command the figure will use."""
        if self._render is not None:
            return
        doc = self._add_shot()
        argv = [sys.executable, "-m", "roqsim.render", *render_args(doc)]
        Path(doc["png"]).parent.mkdir(parents=True, exist_ok=True)
        self._render = subprocess.Popen(  # noqa: S603 - our own interpreter, argv we built
            argv,
            cwd=str(self.replay.project),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._rendering = doc
        self.png_button.configure(state="disabled", text="rendering...")
        self.status.configure(text=f"rendering {doc['png']}")

    def _poll_render(self) -> None:
        if self._render is None or self._render.poll() is None:
            return
        code, stderr = self._render.returncode, (self._render.stderr.read() or b"")
        self._render = None
        self.png_button.configure(state="normal", text="Add + PNG")
        if code == 0:
            self.status.configure(text=f"wrote {self._rendering['png']}")
        else:
            last = stderr.decode(errors="replace").strip().splitlines()
            self.status.configure(text=f"render failed: {last[-1] if last else code}")
            log.error("render of %s failed (%s)", self._rendering["id"], code)

    def _close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:  # pragma: no cover - the window is already gone
            pass
