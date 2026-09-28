"""A NavOutput that moves nothing and records everything.

Deliberately not in ``conftest.py``: it has to be reachable by a ``file.py:Class`` reference, which
is one of the three forms the registry must support, and a conftest is not a file a world can name.
"""

from __future__ import annotations

import numpy as np

from roqsim_nav.outputs import NavOutput, OutputUnavailable


class StubOutput(NavOutput):
    """Records what it was asked to do; reports a pose it was told to have."""

    kinematics = "holonomic"
    update_hz = None

    def __init__(self, config=None):
        super().__init__(config or {})
        self.attached = None
        self.emitted: list[tuple] = []
        self.stops = 0
        self.xy = np.zeros(2)
        self.yaw = 0.0

    def attach(self, ctx, entity):
        self.attached = entity.name

    def emit(self, ctx, pref_vel, yaw, dt):
        self.emitted.append((np.asarray(pref_vel).copy(), yaw, dt))
        self.xy = self.xy + np.asarray(pref_vel, dtype=float) * dt

    def pose(self, ctx):
        return float(self.xy[0]), float(self.xy[1]), self.yaw

    def stop(self, ctx):
        self.stops += 1


class SlowOutput(StubOutput):
    """Declares a tick rate, the way an animated embodiment does."""

    update_hz = 60.0


class NeverAvailable(NavOutput):
    """Refuses to attach, the way an output does when the entity is the wrong shape."""

    def attach(self, ctx, entity):
        raise OutputUnavailable("this entity is not the right shape for a stub")

    def emit(self, ctx, pref_vel, yaw, dt): ...

    def pose(self, ctx):
        return 0.0, 0.0, 0.0


class NotAnOutput:
    """Not a NavOutput subclass, so the registry must refuse it."""
