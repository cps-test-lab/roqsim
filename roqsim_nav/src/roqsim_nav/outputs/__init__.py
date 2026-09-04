"""How a navigator's motion reaches the physics -- the one thing that varies per embodiment.

Everything above this line is shared: the A* plan, the behaviour tree that follows it, the forward
caution probe, the goal interface. Below it, an entity is moved by whatever it actually is -- a
wheeled base takes a velocity command, a mocap prop takes a pose, a pedestrian takes a pose plus an
animated skeleton. That is the whole difference between them, so it is the whole of this interface.

Implementations are resolved by name from the ``roqsim_nav.outputs`` entry-point group (or by
``module:Class`` / ``file.py:Class``), which is what lets ``roqsim_walker`` supply the pedestrian
embodiment without this package importing it, and what lets an out-of-tree robot family add its own
without editing anything here. Nothing in ``roqsim_nav`` branches on an output's name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .._resolve import RegistryError, registered, resolve

ENTRY_POINT_GROUP = "roqsim_nav.outputs"


class NavOutput(ABC):
    """One embodiment: how this entity is moved, and where it currently is.

    Constructed with the navigator's resolved config, then :meth:`attach`-ed once the model is
    compiled. Every method runs on the physics thread.
    """

    #: Which planar velocities this embodiment can actually realise, so the navigator can shape a
    #: holonomic preferred velocity into something the thing can do. "holonomic" (any planar
    #: velocity), "unicycle" (drives and turns, cannot strafe), "ackermann" (cannot turn in place).
    #: A pose-written body is holonomic by construction; a wheeled base reports what its drive says.
    kinematics: str = "holonomic"

    #: The tick rate this embodiment wants, when the world does not say. ``None`` -> the navigator's
    #: default, which is chosen for planning. An embodiment whose output is *watched* rather than
    #: merely arrived at -- an animated gait -- asks for more, and says so here rather than having
    #: the navigator special-case it by name.
    update_hz: float | None = None

    def __init__(self, config: dict):
        self.config = config or {}

    @abstractmethod
    def attach(self, ctx, entity) -> None:
        """Resolve ids and handles for ``entity``, or raise :class:`OutputUnavailable`.

        Raising the specific type is what makes ``output: auto`` work: the navigator tries each
        candidate and reports every reason together, rather than the first one it hit.
        """

    @abstractmethod
    def emit(self, ctx, pref_vel: np.ndarray, yaw: float, dt: float) -> None:
        """Move the entity for ``dt``, given a world-frame preferred velocity. The only write."""

    @abstractmethod
    def pose(self, ctx) -> tuple[float, float, float]:
        """Ground-truth ``(x, y, yaw)``.

        Ground truth, not odometry, and deliberately: an opponent is apparatus, and closing its
        control loop on a dead-reckoned estimate would make its trajectory a function of wheel slip,
        hence of contacts, hence of the robot under test. It would also be in the wrong frame -- the
        planner's grid is in world coordinates.
        """

    def stop(self, ctx) -> None:
        """Come to rest: blocked by traffic, reset between episodes, or shutting down."""
        self.emit(ctx, np.zeros(2), self.pose(ctx)[2], 0.0)


class OutputUnavailable(RegistryError):
    """This embodiment cannot drive this entity, with the reason a world author can act on."""


def resolve_output(ref: str, base_dir: Path | None = None) -> type[NavOutput]:
    """Resolve ``ref`` to a :class:`NavOutput` subclass. See the module docstring for the forms."""
    return resolve(ref, group=ENTRY_POINT_GROUP, base=NavOutput, kind="output", base_dir=base_dir)


def available() -> list[str]:
    """Registered output names, for error messages and ``output: auto``."""
    return registered(ENTRY_POINT_GROUP)
