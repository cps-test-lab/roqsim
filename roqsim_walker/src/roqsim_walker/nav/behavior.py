"""Per-walker behaviour: navigate to goals, recover when stuck.

Ported from our earlier in-house nav prototype's ``mujoco_nav.pedestrian.behavior``.

The high-level decisions a walker makes each step live here, on top of the controller's global plan
(where to route) and ORCA (how to avoid collisions right now). The behaviour is a small py-trees
tree::

    Selector "root"
    |-- Sequence "recovery"     (higher priority)
    |   |-- IsStuck             condition: no progress over `stuck_time`
    |   '-- BackUpThenReplan    back away from the blocker, then clear the path
    '-- Sequence "navigate"
        |-- EnsurePath          plan (A*) to the current goal if there isn't one
        '-- FollowPath          steer to the next path point; advance goals (loop)

All leaves are thin wrappers over :class:`NavCore`, which holds the state and the actual logic. Each
tick the chosen branch writes ``NavCore.pref_vel`` (the ORCA preferred velocity); the controller
reads it after ticking.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import py_trees


@dataclass
class NavParams:
    """Navigation / recovery tuning (sane defaults; overridable per walker)."""

    inflation_radius: float = 0.0  # A* clearance; defaults to the walker radius
    waypoint_radius: float = 0.3  # "reached" radius for intermediate points
    stuck_time: float = 1.5  # s of no progress before declaring stuck
    stuck_eps: float = 0.10  # m travelled within stuck_time -> stuck
    backup_time: float = 0.5  # s to drive away from the blocker
    backup_speed: float = 0.4  # m/s during backup
    max_recovery: int = 4  # consecutive recoveries before skipping goal

    @classmethod
    def from_spec(cls, spec: dict, radius: float) -> NavParams:
        plan = spec.get("planner") or {}
        rec = spec.get("recovery") or {}
        return cls(
            inflation_radius=float(plan.get("inflation_radius", radius)),
            waypoint_radius=float(plan.get("waypoint_radius", 0.3)),
            stuck_time=float(rec.get("stuck_time", 1.5)),
            stuck_eps=float(rec.get("stuck_eps", 0.10)),
            backup_time=float(rec.get("backup_time", 0.5)),
            backup_speed=float(rec.get("backup_speed", 0.4)),
            max_recovery=int(rec.get("max_recovery", 4)),
        )


class NavCore:
    """State + logic for one walker's navigation, shared by the py-trees leaves.

    ``st`` is the controller's ``_Walker``; ``planner`` is a
    :class:`~roqsim_walker.nav.planner.GridPlanner` or ``None`` (then paths are straight lines to
    each goal -- recovery still applies).
    """

    def __init__(self, st, planner, params: NavParams):
        self.st = st
        self.planner = planner
        self.p = params
        self.pref_vel = np.zeros(2)
        # per-tick inputs, refreshed by the controller before each tick
        self._t = 0.0
        self._pos = np.array(st.pos, dtype=float)
        self._blocker = None  # nearest blocking agent xy, or None
        self.reset()

    # -- lifecycle -----------------------------------------------------------------------------
    def reset(self) -> None:
        st = self.st
        st.goal_idx = 1 % len(st.waypoints)
        st.path = None
        st.path_idx = 0
        st.done = False
        self._recover_until = 0.0
        self._recover_count = 0
        self._backup_vel = np.zeros(2)
        self._dwell_until = 0.0  # standing-idle-at-goal until this sim time
        self._hist = []  # list[(t, x, y)] for stuck detection
        self.pref_vel = np.zeros(2)

    def observe(self, t, pos, blocker) -> None:
        """Feed the current sim time, world position and nearest blocker (xy or None) before
        ticking."""
        self._t = float(t)
        self._pos = np.asarray(pos, dtype=float)
        self._blocker = None if blocker is None else np.asarray(blocker, dtype=float)
        self._hist.append((self._t, self._pos[0], self._pos[1]))
        cutoff = self._t - self.p.stuck_time
        while len(self._hist) > 2 and self._hist[0][0] < cutoff:
            self._hist.pop(0)

    # -- shared leaf logic ---------------------------------------------------------------------
    def is_stuck(self) -> bool:
        st = self.st
        if st.done or self._t < self._recover_until or self._t < self._dwell_until:
            return False  # not stuck -- deliberately dwelling at a goal
        if len(self._hist) < 2 or (self._t - self._hist[0][0]) < self.p.stuck_time:
            return False  # not enough history yet
        xs = np.array([[x, y] for _, x, y in self._hist])
        travelled = float(np.linalg.norm(xs - self._pos, axis=1).max())
        return travelled < self.p.stuck_eps

    def start_recovery(self) -> None:
        """Back away from the blocker (or reverse the last heading), then drop the path so the
        navigate branch replans from the unwedged pose. After ``max_recovery`` consecutive attempts
        (counter resets only when a goal is reached), give up on this goal and move on so we never
        loop forever."""
        st = self.st
        self._recover_count += 1
        if self._recover_count > self.p.max_recovery:
            self._recover_count = 0
            self._recover_until = 0.0
            self._backup_vel = np.zeros(2)
            self._advance_goal()  # skip the unreachable goal
            st.path = None
            st.path_idx = 0
            self.pref_vel = np.zeros(2)
            return
        if self._blocker is not None:
            away = self._pos - self._blocker
        else:  # reverse recent travel, else current yaw
            away = -np.array([math.cos(st.yaw), math.sin(st.yaw)])
        n = float(np.linalg.norm(away))
        away = away / n if n > 1e-6 else np.array([-math.cos(st.yaw), -math.sin(st.yaw)])
        self._backup_vel = away * self.p.backup_speed
        self._recover_until = self._t + self.p.backup_time
        st.path = None
        st.path_idx = 0

    def recovery_active(self) -> bool:
        return self._t < self._recover_until

    def needs_recovery(self) -> bool:
        """True while the recovery branch should own the tick: a backup already in progress, or
        freshly detected stuck."""
        return self.recovery_active() or self.is_stuck()

    def recovery_vel(self):
        self.pref_vel = self._backup_vel
        return self.pref_vel

    def ensure_path(self) -> bool:
        """Make sure ``st.path`` heads to the current goal. Returns False only when there is no goal
        to pursue (single-waypoint, non-looping, done)."""
        st = self.st
        if st.done:
            return False
        if st.path:
            return True
        goal = st.waypoints[st.goal_idx]
        if self.planner is not None:
            path = self.planner.plan((self._pos[0], self._pos[1]), (float(goal[0]), float(goal[1])))
        else:
            path = None
        # No planner, or unreachable -> straight line to the goal (ORCA/recovery then cope locally).
        st.path = path if path else [(float(goal[0]), float(goal[1]))]
        st.path_idx = 0
        return True

    def follow_path(self) -> bool:
        """Steer toward the next path point; advance through the path and on to the next goal
        (loop-aware). Resets the recovery counter on progress."""
        st = self.st
        if not st.path:
            self.pref_vel = np.zeros(2)
            return False
        # Skip reached intermediate points; the final point uses arrival_radius.
        while st.path_idx < len(st.path):
            tgt = np.array(st.path[st.path_idx], dtype=float)
            last = st.path_idx == len(st.path) - 1
            radius = st.arrival_radius if last else self.p.waypoint_radius
            if float(np.linalg.norm(tgt - self._pos)) < radius:
                if last:  # reached the goal
                    lo, hi = st.dwell[st.goal_idx] if st.dwell else (0.0, 0.0)
                    if hi > 0.0:  # pause and stand-idle here for a while
                        if self._dwell_until <= 0.0:
                            self._dwell_until = self._t + random.uniform(lo, hi)
                        if self._t < self._dwell_until:
                            self.pref_vel = np.zeros(2)
                            return True
                        self._dwell_until = 0.0  # done dwelling -> proceed
                        self._hist = []  # don't read the idle dwell as "stuck"
                    self._recover_count = 0
                    if not self._advance_goal():
                        self.pref_vel = np.zeros(2)
                        return True
                    st.path = None
                    st.path_idx = 0
                    return self.ensure_path() and self.follow_path()
                st.path_idx += 1
            else:
                break
        if st.path_idx >= len(st.path):
            self.pref_vel = np.zeros(2)
            return True
        tgt = np.array(st.path[st.path_idx], dtype=float)
        to = tgt - self._pos
        dist = float(np.linalg.norm(to))
        self.pref_vel = (to / dist * st.speed) if dist > 1e-6 else np.zeros(2)
        return True

    def _advance_goal(self) -> bool:
        st = self.st
        n = len(st.waypoints)
        if n <= 1:
            st.done = True
            return False
        nxt = st.goal_idx + 1
        if nxt < n:
            st.goal_idx = nxt
            return True
        if st.loop:
            st.goal_idx = 0
            return True
        st.done = True
        return False

    # -- progress reporting (consumed by goal-driven interfaces, e.g. NavigateThroughPoses) -----
    def remaining(self) -> tuple[int, float]:
        """``(goals_remaining, distance_remaining)`` along the current route.

        ``distance_remaining`` sums the leg to the active goal plus the straight-line legs between
        the goals still ahead -- the same estimate nav2's controllers report as feedback.
        """
        st = self.st
        if st.done:
            return 0, 0.0
        n = len(st.waypoints)
        idx = st.goal_idx
        goals_left = (n - idx) if not st.loop else 1
        dist = float(np.linalg.norm(np.asarray(st.waypoints[idx], float) - self._pos))
        if not st.loop:
            for i in range(idx, n - 1):
                a = np.asarray(st.waypoints[i], float)
                b = np.asarray(st.waypoints[i + 1], float)
                dist += float(np.linalg.norm(b - a))
        return int(goals_left), dist


# -- py-trees leaves ---------------------------------------------------------------------------
class _Leaf(py_trees.behaviour.Behaviour):
    def __init__(self, name, core):
        super().__init__(name)
        self.core = core


class _IsStuck(_Leaf):
    def update(self):
        s = py_trees.common.Status
        return s.SUCCESS if self.core.needs_recovery() else s.FAILURE


class _BackUpThenReplan(_Leaf):
    def update(self):
        s = py_trees.common.Status
        if not self.core.recovery_active():
            self.core.start_recovery()
        self.core.recovery_vel()
        return s.RUNNING if self.core.recovery_active() else s.SUCCESS


class _EnsurePath(_Leaf):
    def update(self):
        s = py_trees.common.Status
        return s.SUCCESS if self.core.ensure_path() else s.FAILURE


class _FollowPath(_Leaf):
    def update(self):
        s = py_trees.common.Status
        return s.SUCCESS if self.core.follow_path() else s.FAILURE


def build_tree(core: NavCore):
    """A ticked py-trees ``BehaviourTree`` for one walker."""
    recovery = py_trees.composites.Sequence(
        name="recovery",
        memory=False,
        children=[_IsStuck("IsStuck", core), _BackUpThenReplan("BackUp", core)],
    )
    navigate = py_trees.composites.Sequence(
        name="navigate",
        memory=False,
        children=[_EnsurePath("EnsurePath", core), _FollowPath("FollowPath", core)],
    )
    root = py_trees.composites.Selector(name="root", memory=False, children=[recovery, navigate])
    return py_trees.trees.BehaviourTree(root)
