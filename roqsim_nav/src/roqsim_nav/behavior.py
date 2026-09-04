"""Per-walker behaviour: navigate to goals, recover when stuck.

Ported from an earlier in-house navigation prototype.

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

import logging
import math
import random
from dataclasses import dataclass

import numpy as np
import py_trees

from .control import path_remaining, pure_pursuit

logger = logging.getLogger(__name__)

#: A tick may legitimately pass a few short goals; more than this means progress is misreported.
_MAX_GOALS_PER_TICK = 16

#: How close along the path counts as having reached a goal (m). A tolerance is required, not a
#: nicety: the mover converges on a goal asymptotically, so a bare ``travelled >= goal_at`` is only
#: satisfied in the limit and the mover parks a millimetre short of the last goal of a lap forever.
#: A centimetre is far below any sensible ``arrival_radius`` and far above float noise.
_PROGRESS_EPS = 0.01


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
    """State + logic for one agent's navigation, shared by the py-trees leaves.

    ``st`` is anything satisfying :class:`~roqsim_nav.state.NavStateLike` -- the plain
    :class:`~roqsim_nav.state.NavState`, or the walker controller's richer per-walker object.
    ``planner`` is a :class:`~roqsim_nav.planner.GridPlanner` or ``None`` (then paths are straight
    lines to each goal -- recovery still applies).

    ``uniform(lo, hi)`` is where the one random draw in this module -- a waypoint's dwell -- comes
    from. It is injected rather than taken from the ``random`` module so a caller can supply a
    counter-based stream (``ctx.rng_for``), which is what makes a dwelling agent replay exactly
    under the same seed instead of depending on how many draws happened before it. The default
    keeps the historical behaviour for a caller that passes nothing.
    """

    def __init__(self, st, planner, params: NavParams, *, uniform=random.uniform):
        self._uniform = uniform
        self.st = st
        self.planner = planner
        self.p = params
        self.pref_vel = np.zeros(2)
        # per-tick inputs, refreshed by the controller before each tick
        self._t = 0.0
        self._pos = np.array(st.pos, dtype=float)
        self._blocker = None  # nearest blocking agent xy, or None
        #: Arc curvature to the pure-pursuit carrot, for a base that steers by curvature rather than
        #: by heading error. Zero for the waypoint follower, which does not compute one.
        self.curvature = 0.0
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

    def forget_progress(self) -> None:
        """Discard the movement history stuck-detection reads, keeping only the latest sample.

        For a mover that is deliberately holding still -- yielding to traffic it must not drive into
        -- rather than failing to move. Both look identical to :meth:`is_stuck`, which only knows
        that the position has not changed for a while, so the layer that knows *why* has to say so.
        Without this a paused mover declares itself stuck after ``stuck_time`` and recovers out of a
        pause it was supposed to be waiting through.
        """
        self._hist = self._hist[-1:]
        self._recover_until = 0.0

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
                            self._dwell_until = self._t + self._uniform(lo, hi)
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

    def follow_path_pure_pursuit(self, lookahead: float, depth: int = 0) -> bool:
        """Track the path itself, rather than chasing the end of the current leg.

        The difference from :meth:`follow_path` is two changes that have to be made together, which
        is why this is a separate follower and not a switch inside that one:

        * **Steering** aims at a carrot a fixed ``lookahead`` along the route -- and the route
          continues past the current goal onto the ones after it, so the carrot rounds a corner
          before the mover reaches it. Cross-track error is then bounded by the lookahead instead of
          by how close the mover must get before it gives up on a waypoint.
        * **Progress** advances the goal when the mover *crosses the goal*, tested as passing the
          plane through it perpendicular to the approach -- not when it comes within
          ``arrival_radius`` of it. This is the half that cannot be omitted: pure pursuit
          deliberately does not drive at the goal, so a follower still waiting for proximity reaches
          that radius late or never, and the mover stalls short of the corner. (It did, measurably,
          when this was first tried as an override of the steering alone.)

        The final goal of a non-looping route is the exception and still uses ``arrival_radius``:
        there is nothing to round onto, and the mover has to stop *at* the goal rather than cross it.
        """
        st = self.st
        if not st.path:
            self.pref_vel = np.zeros(2)
            return False

        if self._reached_goal():
            lo, hi = st.dwell[st.goal_idx] if st.dwell else (0.0, 0.0)
            if hi > 0.0:
                if self._dwell_until <= 0.0:
                    self._dwell_until = self._t + self._uniform(lo, hi)
                if self._t < self._dwell_until:
                    self.pref_vel = np.zeros(2)
                    return True
                self._dwell_until = 0.0
                self._hist = []  # don't read the idle dwell as "stuck"
            self._recover_count = 0
            if not self._advance_goal():
                self.pref_vel = np.zeros(2)
                return True
            st.path = None
            st.path_idx = 0
            if not self.ensure_path():
                return False
            # A bounded loop rather than tail recursion: passing several goals in one tick is
            # legitimate (a short leg, a slow tick), but a logic error that makes every goal read as
            # reached should degrade into standing still with a warning, not blow the Python stack
            # from inside a physics step.
            if depth >= _MAX_GOALS_PER_TICK:
                logger.warning(
                    "navigator passed %d goals in one tick; holding. This means every goal is "
                    "reading as already reached, which is a progress-measurement bug, not a route.",
                    depth,
                )
                self.pref_vel = np.zeros(2)
                return True
            return self.follow_path_pure_pursuit(lookahead, depth=depth + 1)

        route = self.route_ahead()
        self.pref_vel, self.curvature = pure_pursuit(
            route, self._pos, float(st.yaw), lookahead=lookahead, speed=st.speed
        )
        return True

    def route_ahead(self) -> list[tuple[float, float]]:
        """The polyline still to be driven: the rest of the current path, then the goals after it.

        Continuing past the current goal is what lets the carrot round a corner instead of aiming at
        its end.

        **It extends exactly one goal past the current one, and no further.** One is all the carrot
        needs to round the upcoming corner, and stopping there is what keeps the polyline from
        crossing itself: the carrot and the projection are both found by projecting the mover onto
        this polyline, which is ambiguous the moment the same place appears on it twice. Carrying the
        whole remainder of a looping route does exactly that -- the route comes back round to where
        the mover is standing, the projection lands on that later pass, the remaining length reads as
        zero, and the end-of-path easing scales the mover's speed to nothing. It parks, on a route
        with no end. A bounded window cannot express that however the route is shaped.
        """
        st = self.st
        route = [tuple(float(v) for v in p) for p in st.path[st.path_idx :]]
        goals = [tuple(float(v) for v in p) for p in st.waypoints]
        after = goals[st.goal_idx + 1 :] or (goals[:1] if st.loop else [])
        route += after[:1]
        if len(route) < 2:
            # A single remaining point is not a polyline; anchor it at the mover so the carrot and
            # the projection are still defined.
            route = [tuple(float(v) for v in self._pos), *route]
        return route

    def _reached_goal(self) -> bool:
        """Whether the current goal counts as passed, measured as **progress along the route**.

        Not "am I within a radius of it" -- pure pursuit deliberately does not drive at the goal, so
        proximity is reached late or never and the mover stalls short of the corner.

        Not "have I crossed the plane through it" either, which is the obvious fix and is also
        wrong: a tracker that rounds a corner passes *inside* it, so it never crosses that plane at
        all. Both failures are silent in the same way -- the carrot keeps dragging the mover along
        the route while ``goal_idx`` sits still, so it appears to work while ``done`` never fires and
        progress reporting is nonsense.

        What does hold under cutting is arc length: the projection of the mover onto the route only
        ever moves forward, so "the projection is at or past the goal" is monotone and corner-shape
        independent. It is what the RPP family uses.
        """
        st = self.st
        goal = np.asarray(st.path[-1], dtype=float)
        if not st.loop and st.goal_idx >= len(st.waypoints) - 1:
            # Nothing to round onto: stop AT it, the way any goal-reaching controller must.
            return float(np.linalg.norm(goal - self._pos)) < st.arrival_radius
        # Progress is measured from a fixed ANCHOR behind the mover -- the point the current goal is
        # being approached from. Measuring from the route's own start does not work: with no
        # intermediate path points the goal IS the first route point, so the arc length to it is
        # zero and every goal reads as reached on the first tick.
        route = self._progress_route()
        goal_at = self._arc_length(route, len(st.path) - st.path_idx)
        travelled = self._arc_length(route, len(route) - 1) - path_remaining(route, self._pos)
        return travelled >= goal_at - _PROGRESS_EPS

    def _progress_route(self) -> list[tuple[float, float]]:
        """The polyline progress is measured along: anchor, then the path, then the goals after it.

        Built here rather than reused from :meth:`route_ahead`, which serves the *carrot* and anchors
        a one-point remainder at the mover's own position. That anchoring is right for steering and
        wrong for measuring: it shifts every index by one only sometimes, so the goal's position in
        the polyline stops being a function of the path length and the arithmetic silently points at
        the wrong point. One route per question is cheaper than one route with a caveat.
        """
        st = self.st
        goals = [tuple(float(v) for v in p) for p in st.waypoints]
        return [
            self._anchor(),
            *(tuple(float(v) for v in p) for p in st.path[st.path_idx :]),
            *goals[st.goal_idx + 1 :],
        ]

    def _anchor(self) -> tuple[float, float]:
        """The fixed point the current goal is being approached from."""
        st = self.st
        prev = (
            st.path[st.path_idx - 1]
            if st.path_idx >= 1
            else st.waypoints[(st.goal_idx - 1) % len(st.waypoints)]
        )
        return tuple(float(v) for v in prev)

    @staticmethod
    def _arc_length(route, upto: int) -> float:
        """Arc length from the start of ``route`` to its ``upto``-th point."""
        total = 0.0
        for i in range(max(0, min(upto, len(route) - 1))):
            a = np.asarray(route[i], dtype=float)
            b = np.asarray(route[i + 1], dtype=float)
            total += float(np.linalg.norm(b - a))
        return total

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


class _FollowPathPurePursuit(_Leaf):
    def __init__(self, name, core, lookahead: float):
        super().__init__(name, core)
        self.lookahead = float(lookahead)

    def update(self):
        s = py_trees.common.Status
        return s.SUCCESS if self.core.follow_path_pure_pursuit(self.lookahead) else s.FAILURE


def build_tree(core: NavCore, *, recovery: bool = True, lookahead: float | None = None):
    """A ticked py-trees ``BehaviourTree`` for one navigating agent.

    ``lookahead`` selects the path tracker: ``None`` keeps the waypoint follower (steer at the goal,
    advance on proximity), a distance switches to pure pursuit (steer at a carrot on the path,
    advance on crossing the goal). The pedestrian stack keeps the former by default -- a walker
    rounding corners by its arrival radius is what it has always looked like.

    ``recovery=False`` drops the higher-priority recovery branch, leaving ``navigate`` alone. That is
    the difference between an agent that gets itself unstuck -- backing up and re-planning, so its
    trajectory can differ from run to run -- and one whose path is planned once and then only ever
    followed or paused. Which is wanted is the experiment's call: a wandering pedestrian usually
    wants recovery, an obstacle whose motion must be identical in every repetition of a cell does
    not. The branch is omitted rather than disabled so that a tick costs nothing when it is off.
    """
    follow = (
        _FollowPath("FollowPath", core)
        if lookahead is None
        else _FollowPathPurePursuit("PurePursuit", core, lookahead)
    )
    navigate = py_trees.composites.Sequence(
        name="navigate", memory=False, children=[_EnsurePath("EnsurePath", core), follow]
    )
    if not recovery:
        return py_trees.trees.BehaviourTree(navigate)
    recover = py_trees.composites.Sequence(
        name="recovery",
        memory=False,
        children=[_IsStuck("IsStuck", core), _BackUpThenReplan("BackUp", core)],
    )
    root = py_trees.composites.Selector(name="root", memory=False, children=[recover, navigate])
    return py_trees.trees.BehaviourTree(root)
