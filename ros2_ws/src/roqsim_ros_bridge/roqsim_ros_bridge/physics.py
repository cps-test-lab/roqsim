"""Running a callable on the physics thread and waiting for it, from a ROS callback.

Every ROS service that *changes* the simulation needs the same three lines: post the change, block
until the physics thread has actually run it, and turn a timeout into a failed reply. It lives here
rather than on one plugin because the reason it exists is a contract, not a convenience:

**A post that timed out must never be reported as success.** A service that answers ``RESULT_OK``
before its change has run is indistinguishable, to its caller, from one that worked -- so a paused or
stalled simulator silently accepts commands and a scenario proceeds on the belief that the world
changed. The caller turns ``False`` into a failure; there is no optimistic branch.

Only for a *cross-thread* caller. Code already on the physics thread (a plugin hook) mutates
``model``/``data`` directly and must not post -- see the single-writer rule in
``docs/architecture.rst`` §7.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

#: Wall-clock seconds to wait for the physics thread. Generous enough for a slow step, short enough
#: that a paused simulator answers rather than hanging the caller's executor.
DEFAULT_TIMEOUT_S = 2.0


def run_on_physics(ctx, fn: Callable[[Any], None], timeout: float | None = None) -> bool:
    """Post *fn* to the physics thread and block until it has run. ``False`` on timeout.

    ``timeout`` defaults to :data:`DEFAULT_TIMEOUT_S`, read at call time so the module constant is
    actually the knob it looks like.
    """
    done = threading.Event()

    def wrapped(c):
        try:
            fn(c)
        finally:
            done.set()

    ctx.post(wrapped)
    return done.wait(DEFAULT_TIMEOUT_S if timeout is None else timeout)


def barrier(ctx, timeout: float | None = None) -> bool:
    """Wait until everything already queued has run. ``False`` on timeout.

    ``ctx.post`` is FIFO and drained at the start of ``pre_step``, so a no-op posted after a command
    tells the caller that the command has been applied -- without the caller having to know what the
    command touched. Two barriers therefore span a whole step: the second one runs in the *next*
    drain, i.e. after the ``post_step`` that follows the first, which is where a plugin that verifies
    its own work records the verdict.
    """
    return run_on_physics(ctx, lambda _c: None, timeout)
