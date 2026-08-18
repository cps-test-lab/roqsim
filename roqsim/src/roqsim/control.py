"""Thread-safe run-control shared between a driver and control-plane plugins.

The standalone runner consults this each loop; the ``simulation_interfaces`` plugin (on its executor
thread) mutates it. State mirrors ``simulation_interfaces/SimulationState``. Default is ``PLAYING``,
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


class RunControl:
    def __init__(self, state: int = PLAYING):
        self._lock = threading.Lock()
        self._state = state
        self._pending_steps = 0
        self._reset_requested = False

    @property
    def state(self) -> int:
        with self._lock:
            return self._state

    def set_state(self, state: int) -> None:
        with self._lock:
            self._state = state
            if state == STOPPED:
                self._reset_requested = True
                self._pending_steps = 0

    def request_steps(self, n: int) -> None:
        with self._lock:
            self._pending_steps += int(n)

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
        with self._lock:
            if self._state == PLAYING:
                return True
            if self._state == PAUSED and self._pending_steps > 0:
                self._pending_steps -= 1
                return True
            return False
