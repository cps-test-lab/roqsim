"""The runtime goal interface: how anything outside the physics thread commands a navigator.

One handle serves every caller -- a scenario action, a ROS 2 action server, a test, an in-process
driver -- so a route sent from a `.osc` file and one sent over ``NavigateThroughPoses`` take the same
path through the simulator and cannot drift apart.

**Sequence numbers are the whole design.** A caller must be able to tell *its own* arrival from a
stale one, and "finished" alone cannot: a navigator that has completed its configured route is
already finished when a new route is queued, so a caller watching that flag would report an arrival
that never happened. Every route is stamped with a monotonically increasing number, returned
immediately; ``status`` reports which number is live. Equal and finished means you arrived. A larger
number means something preempted you.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class NavHandle:
    """Published on the blackboard under ``nav:<entity>``.

    ``send_goals`` is thread-safe: it stamps the route, marshals the change onto the physics thread
    via :meth:`~roqsim.context.SimContext.post`, and returns the sequence number synchronously --
    before the route has been applied. That is deliberate: an off-thread caller can hold the number
    it must poll for without ever touching ``data``.

    Only the list form exists. ``send_goals([p])`` is a single goal, and the OpenSCENARIO vocabulary
    already states the rule for exactly this case: one object is a one-element list. Two functions
    differing only by arity would be two things to keep in step.
    """

    name: str  # the ENTITY being navigated, not the component's address
    send_goals: Callable[[list], int]  # [(x, y[, yaw]), ...] -> sequence number
    start: Callable[[], int]  # release an `autostart: false` route -> sequence number
    cancel: Callable[[], int]  # stop where it stands -> sequence number
    status: Callable[[], tuple[int, bool, int, float]]  # (seq, finished, goals_left, dist_left)
    #: ``(x, y, yaw)``, ground truth, safe to call from any thread.
    #:
    #: It is the pose as of the navigator's **last tick**, so it lags the live body by at most one
    #: nav period (50 ms at the default 20 Hz). Deliberately a latched snapshot rather than a live
    #: read: reading ``data`` off the physics thread is the one thing the single-writer rule forbids,
    #: and a caller that needs the instantaneous pose is on the physics thread already and should
    #: read the model. Uniform across embodiments, which a post-emit refresh would not be -- a driven
    #: base has not moved yet when its twist is commanded, while a written pose has.
    pose: Callable[[], tuple[float, float, float]]


class Sequencer:
    """Hands out route sequence numbers, and remembers which one is live and whether it finished.

    Split out of the plugin because it is the part with a threading contract, and because the
    walker's own goal interface needs exactly the same one.
    """

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        #: The sequence number of the route currently applied (0 = none has been).
        self.applied = 0
        #: Latched on completion of the applied route. Latched rather than derived, because the
        #: navigator resumes its configured route on arrival -- so a state read a moment later would
        #: show it busy again, and a caller polling for its own completion would never see it.
        self.finished = False

    def next(self) -> int:
        """A fresh number, without touching physics state. Safe from any thread."""
        with self._lock:
            return next(self._counter)

    def apply(self, seq: int) -> None:
        """Called on the physics thread when the route stamped ``seq`` actually takes effect."""
        self.applied = seq
        self.finished = False

    def finish(self) -> None:
        self.finished = True
