"""The state a navigating agent carries, and the contract the behaviour tree reads it through.

:class:`NavCore` and its py-trees leaves have always worked against a duck-typed object rather than a
type -- historically the walker controller's own ``_Walker``, which carries a great deal besides
(motion clips, mocap ids, a skeleton, an avoidance agent id). :class:`NavStateLike` writes that
implicit contract down: these eleven attributes are all the navigation layer touches, so any
embodiment can satisfy it, and a change to the walker's animation fields cannot silently alter what
navigation depends on.

:class:`NavState` is the plain implementation, used by every embodiment that has nothing else to
carry. The walker's richer object satisfies the same protocol structurally, which is what lets both
share one behaviour tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NavStateLike(Protocol):
    """What the planner and the behaviour tree read and write. Nothing else is theirs to touch."""

    #: Current world position, ``(2,)``. Written by the output, read by navigation.
    pos: np.ndarray
    #: Current heading (rad).
    yaw: float
    #: The active route, ``(N, 2)`` -- patrol waypoints or commanded goals.
    waypoints: np.ndarray
    #: Metres per second the route is followed at.
    speed: float
    #: Cycle the route forever rather than stopping at the last point.
    loop: bool
    #: How close to a goal counts as arrived (m).
    arrival_radius: float
    #: Per-waypoint ``(lo, hi)`` seconds to stand still on arrival, or ``None``.
    dwell: list | None
    #: Index of the waypoint currently targeted.
    goal_idx: int
    #: The planned sub-path to that waypoint (world xy), or ``None`` when one is needed.
    path: list | None
    #: Index of the next point along ``path``.
    path_idx: int
    #: Finished a non-looping route -> stand.
    done: bool


@dataclass
class NavState:
    """A navigating agent's state, for an embodiment with nothing else to carry.

    ``pos`` defaults to the origin rather than to ``None`` so that a freshly built state is already
    usable; the navigator overwrites it from ground truth before the first tick either way.
    """

    name: str
    waypoints: np.ndarray
    speed: float
    loop: bool = False
    arrival_radius: float = 0.25
    dwell: list | None = None
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = 0.0
    goal_idx: int = 1
    path: list | None = None
    path_idx: int = 0
    done: bool = False
    pref_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
