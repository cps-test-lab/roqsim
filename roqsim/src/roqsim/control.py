"""Thread-safe run-control shared between a driver and control-plane plugins.

The standalone runner consults this each loop; the ``simulation_interfaces`` plugin (on its executor
thread) and the ``run_control`` plugin's endpoints mutate it. State mirrors ``simulation_interfaces/SimulationState``. Default is ``PLAYING``,
so a world with no control plane behaves exactly as a free-running loop.

Under scenario-execution the framework owns stepping, so play/pause/step do not apply there (the
adapter ignores this object); ``GetSimulatorFeatures`` should reflect that.
"""

from __future__ import annotations

import threading

STOPPED = 0
PLAYING = 1
PAUSED = 2
QUITTING = 3

#: A state's name, as a caller outside the process spells it.
STATE_NAMES = {STOPPED: "stopped", PLAYING: "playing", PAUSED: "paused", QUITTING: "quitting"}


class RunControl:
    def __init__(self, state: int = PLAYING):
        self._lock = threading.Lock()
        self._state = state
        self._pending_steps = 0
        self._reset_requested = False
        # Called once the requested steps have been taken, or dropped by a change of state.
        self._on_steps_done: list = []
        # Called at the driver's next loop, after the step in flight (if any) has run.
        self._next_loop: list = []

    @property
    def state(self) -> int:
        with self._lock:
            return self._state

    def set_state(self, state: int) -> None:
        due, dropped = (), ()
        with self._lock:
            self._state = state
            if state == STOPPED:
                self._reset_requested = True
            # Steps are taken only while paused, so leaving the pause drops the ones not yet taken.
            if state != PAUSED:
                self._pending_steps = 0
                due, self._next_loop = self._next_loop, []
                dropped, self._on_steps_done = self._on_steps_done, []
        for fn in due:
            fn()
        for fn in dropped:
            fn(False)

    def request_steps(self, n: int, on_done=None) -> None:
        """Take *n* more steps while paused. ``on_done(completed)`` runs on the driver's thread
        at its next loop after the last of them ran -- or with ``completed=False`` when a change
        of state drops them first."""
        with self._lock:
            self._pending_steps += int(n)
            if on_done is not None:
                self._on_steps_done.append(on_done)

    def at_next_loop(self, fn) -> None:
        """Call ``fn()`` on the driver's thread at its next loop: after the step in flight, if any.

        What a command that changes the state uses to answer with the state it left -- the step
        it was drained in still runs after it.
        """
        with self._lock:
            self._next_loop.append(fn)

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def take_reset(self) -> bool:
        """Consume a pending reset request (called by the driver)."""
        with self._lock:
            r, self._reset_requested = self._reset_requested, False
            return r

    def should_step(self) -> bool:
        """True if the driver should advance one physics step now."""
        due = ()
        with self._lock:
            if self._next_loop:
                due, self._next_loop = self._next_loop, []
            if self._state == PLAYING:
                step = True
            elif self._state == PAUSED and self._pending_steps > 0:
                self._pending_steps -= 1
                if not self._pending_steps and self._on_steps_done:
                    done, self._on_steps_done = self._on_steps_done, []
                    self._next_loop.extend(lambda fn=fn: fn(True) for fn in done)
                step = True
            else:
                step = False
        for fn in due:
            fn()
        return step
