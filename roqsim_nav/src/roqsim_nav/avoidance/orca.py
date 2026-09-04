"""ORCA (Optimal Reciprocal Collision Avoidance) as a :class:`~roqsim_nav.avoidance.AvoidanceModel`.

Reciprocal velocity obstacles: each agent takes half the responsibility for avoiding each other one,
so two agents heading at each other both step aside and neither has to know the other's intent. That
reciprocity is why a crowd of them does not deadlock the way a crowd of stop-for-anything movers
would.

Hoisted out of the pedestrian controller, where it was the only local-avoidance policy the substrate
had and was reachable only through a walker. Nothing here is pedestrian-specific.

Needs ``rvo2``, which is an optional extra because it publishes no wheel and must be built from
source. Without it this model refuses to load, and a world that named it is told so at load time
rather than quietly navigating without avoidance.
"""

from __future__ import annotations

import logging

import numpy as np

from . import NO_AGENT, AvoidanceModel

logger = logging.getLogger(__name__)

#: rvo2 defaults that are not per-agent: how many neighbours to consider, and the time horizon for
#: static obstacles (shorter than for agents -- a wall will not step aside, so reacting to one far
#: ahead only makes an agent hug the opposite side of a corridor).
_MAX_NEIGHBORS = 10
_OBSTACLE_HORIZON = 2.0


class OrcaModel(AvoidanceModel):
    """One rvo2 simulation for the world."""

    params_schema = ("neighbor_dist", "time_horizon", "radius", "max_speed")

    def __init__(self) -> None:
        self._sim = None
        self._dt = 0.002
        self._agents: dict[int, dict] = {}
        self._pending_static: list = []

    def configure(self, ctx, params: dict) -> None:
        try:
            import rvo2
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the 'orca' avoidance model needs rvo2, which is an optional extra because it "
                "publishes no wheel and is built from source: pip install 'roqsim_nav[avoidance]'. "
                "A world that must run without a compiler should declare no `avoidance:` entry -- "
                "movers then execute their own preferred velocity and nobody yields."
            ) from exc
        self._dt = float(ctx.model.opt.timestep) if ctx.model is not None else self._dt
        self._sim = rvo2.PyRVOSimulator(
            self._dt,
            float(params.get("neighbor_dist", 4.0)),
            _MAX_NEIGHBORS,
            float(params.get("time_horizon", 3.0)),
            _OBSTACLE_HORIZON,
            float(params.get("radius", 0.3)),
            float(params.get("max_speed", 1.5)),
        )
        for poly in self._pending_static:
            self._sim.addObstacle(poly)
        if self._pending_static:
            self._sim.processObstacles()
            self._pending_static = []

    def add_agent(self, key, *, radius, max_speed, yields, params):
        if self._sim is None:
            return NO_AGENT
        params = params or {}
        aid = self._sim.addAgent(
            (0.0, 0.0),
            float(params.get("neighbor_dist", 4.0)),
            _MAX_NEIGHBORS,
            float(params.get("time_horizon", 3.0)),
            _OBSTACLE_HORIZON,
            float(radius),
            # A non-yielding agent is pinned from ground truth every step, so its own speed limit is
            # never used; zero states plainly that ORCA is not to move it.
            float(max_speed) if yields else 0.0,
            (0.0, 0.0),
        )
        self._agents[aid] = {"key": key, "yields": bool(yields), "pref": np.zeros(2)}
        return aid

    def add_static(self, polygons) -> None:
        """CCW footprints; agents stay outside them.

        The same polygons the planner rasterized, so local avoidance cannot push an agent through a
        wall the global plan carefully went around. Queued when called before :meth:`configure`,
        because a world may declare its movers before its avoidance model.
        """
        polys = [p for p in polygons if len(p) >= 2]
        if not polys:
            return
        if self._sim is None:
            self._pending_static.extend(polys)
            return
        for poly in polys:
            self._sim.addObstacle(poly)
        self._sim.processObstacles()
        logger.info("ORCA: %d wall obstacle(s)", len(polys))

    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        if self._sim is None or aid == NO_AGENT:
            self._agents.setdefault(aid, {})["pref"] = np.asarray(pref_vel, dtype=float)
            return
        agent = self._agents[aid]
        agent["pref"] = np.asarray(pref_vel, dtype=float)
        agent["present"] = present
        # Position, every step: an agent must be where it really is, not where ORCA last integrated
        # it to.
        self._sim.setAgentPosition(aid, (float(pos[0]), float(pos[1])))
        if not agent.get("yields", True):
            # Velocity too, but ONLY for an agent this model may not move -- the robot under test, a
            # scripted prop. For a yielding one it would be catastrophic and silent: `doStep` writes
            # its answer into the agent's velocity and `getAgentVelocity` is how we read it back, so
            # overwriting that with the body's measured velocity discards the answer. Every mover
            # then executed zero and the whole world stood still, while ORCA reported no error and
            # the agents were plainly registered.
            self._sim.setAgentVelocity(aid, (float(vel[0]), float(vel[1])))
        if not present:
            # Absent entities deflect nobody, exactly as they are seen by no raycaster. Parking the
            # agent far away is how rvo2 expresses that without deleting and renumbering agents.
            self._sim.setAgentPosition(aid, (1e6, 1e6))
            self._sim.setAgentPrefVelocity(aid, (0.0, 0.0))
            return
        pref = agent["pref"] if agent.get("yields", True) else np.zeros(2)
        self._sim.setAgentPrefVelocity(aid, (float(pref[0]), float(pref[1])))

    def solve(self, dt: float) -> None:
        if self._sim is not None:
            self._sim.doStep()

    def result(self, aid) -> np.ndarray:
        agent = self._agents.get(aid, {})
        pref = agent.get("pref", np.zeros(2))
        if self._sim is None or aid == NO_AGENT or not agent.get("yields", True):
            # A non-yielding agent executes exactly what it wanted: this model is not allowed to
            # move it, and returning ORCA's answer would do so through the back door.
            return pref
        vx, vy = self._sim.getAgentVelocity(aid)
        return np.array([vx, vy], dtype=float)

    def reset(self) -> None:
        for agent in self._agents.values():
            agent["pref"] = np.zeros(2)
        if self._sim is not None:
            for aid in self._agents:
                if aid != NO_AGENT:
                    self._sim.setAgentVelocity(aid, (0.0, 0.0))
                    self._sim.setAgentPrefVelocity(aid, (0.0, 0.0))
