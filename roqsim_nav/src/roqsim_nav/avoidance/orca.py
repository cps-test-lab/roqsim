"""ORCA (Optimal Reciprocal Collision Avoidance) as a :class:`~roqsim_nav.avoidance.AvoidanceModel`.

Reciprocal velocity obstacles: each agent takes half the responsibility for avoiding each other one,
so two agents heading at each other both step aside and neither has to know the other's intent. That
reciprocity is why a crowd of them does not deadlock the way a crowd of stop-for-anything movers
would.

Nothing here is pedestrian-specific: any navigator can name it, not only a walker.

The solve is RVO2's, ported to Python in :mod:`._orca`; the neighbours are found by a scan, so an
agent can leave out the ones it ignores (:meth:`OrcaModel.exclude`).
"""

from __future__ import annotations

import logging

import numpy as np

from . import NO_AGENT, AvoidanceModel
from ._orca import Edges, agent_line, new_velocity, obstacle_lines

logger = logging.getLogger(__name__)

#: How many of the nearest agents each agent avoids, and the time horizon for static obstacles
#: (shorter than for agents -- a wall will not step aside, so reacting to one far ahead only makes
#: an agent hug the opposite side of a corridor).
_MAX_NEIGHBORS = 10
_OBSTACLE_HORIZON = 2.0


class OrcaModel(AvoidanceModel):
    """ORCA over every agent of the world, solved agent by agent each step."""

    params_schema = (
        "neighbor_dist",
        "time_horizon",
        "radius",
        "max_speed",
        "clearance",
        "clearance_time",
    )

    def __init__(self) -> None:
        self._agents: list[dict] = []
        self._edges = Edges()
        self._excluded: dict[int, set[int]] = {}
        self.neighbor_dist = 4.0
        self.time_horizon = 3.0
        #: The gap two agents keep passing (m), and how much more for each m/s they close at (s).
        #: ORCA keeps discs apart, so each agent's disc is its radius plus half the clearance plus
        #: ``clearance_time`` times its own speed: two agents keep ``clearance`` and
        #: ``clearance_time`` times the sum of their speeds, which is their closing speed head-on
        #: and past one standing.
        self.clearance = 0.24
        self.clearance_time = 0.06

    def configure(self, ctx, params: dict) -> None:
        self.neighbor_dist = float(params.get("neighbor_dist", self.neighbor_dist))
        self.time_horizon = float(params.get("time_horizon", self.time_horizon))
        self.clearance = float(params.get("clearance", self.clearance))
        self.clearance_time = float(params.get("clearance_time", self.clearance_time))

    def add_agent(self, key, *, radius, max_speed, yields, params):
        params = params or {}
        aid = len(self._agents)
        self._agents.append(
            {
                "key": key,
                "radius": float(radius),
                "disc": float(radius),
                # A non-yielding agent is set from ground truth every step and never moved here.
                "max_speed": float(max_speed) if yields else 0.0,
                "yields": bool(yields),
                "neighbor_dist": float(params.get("neighbor_dist", self.neighbor_dist)),
                "time_horizon": float(params.get("time_horizon", self.time_horizon)),
                "pos": np.zeros(2),
                "vel": np.zeros(2),
                "pref": np.zeros(2),
                "new": np.zeros(2),
                # Until its first submit an agent is nowhere: it deflects nobody and is not solved.
                "present": False,
            }
        )
        return aid

    def add_static(self, polygons) -> None:
        """CCW footprints; agents stay outside them.

        The same polygons the planner rasterized, so local avoidance cannot push an agent through a
        wall the global plan carefully went around.
        """
        polys = [p for p in polygons if len(p) >= 2]
        for poly in polys:
            self._edges.add(poly)
        if polys:
            logger.info("ORCA: %d wall obstacle(s)", len(polys))

    def exclude(self, aid, other) -> None:
        self._excluded.setdefault(aid, set()).add(other)

    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        if aid == NO_AGENT:
            return
        agent = self._agents[aid]
        agent["pref"] = np.asarray(pref_vel, dtype=float)
        agent["present"] = present
        # Position, every step: an agent is where it really is, not where the last solve would
        # have moved it.
        agent["pos"] = np.asarray(pos, dtype=float)[:2].copy()
        speed = float(np.hypot(vel[0], vel[1]))
        agent["disc"] = agent["radius"] + (self.clearance + self.clearance_time * speed) / 2.0
        if not agent["yields"]:
            # Velocity too, but ONLY for an agent this model may not move -- the robot under test, a
            # scripted prop. A yielding agent's velocity is the one its last solve chose, as ORCA
            # needs it: overwriting that with the body's measured velocity makes every agent that
            # has not yet moved plan from standing, and the world stands still.
            agent["vel"] = np.asarray(vel, dtype=float)[:2].copy()

    def solve(self, dt: float) -> None:
        agents = self._agents
        present = [i for i, a in enumerate(agents) if a["present"]]
        for i in present:
            agent = agents[i]
            if not agent["yields"]:
                continue
            pos, vel, disc = agent["pos"], agent["vel"], agent["disc"]
            reach = _OBSTACLE_HORIZON * agent["max_speed"] + disc
            walls = obstacle_lines(
                self._edges,
                self._edges.near(pos, reach * reach),
                pos,
                vel,
                disc,
                _OBSTACLE_HORIZON,
            )
            skip = self._excluded.get(i, ())
            range_sq = agent["neighbor_dist"] ** 2
            near = []
            for j in present:
                if j == i or j in skip:
                    continue
                d = agents[j]["pos"] - pos
                dist_sq = float(d[0] * d[0] + d[1] * d[1])
                if dist_sq < range_sq:
                    near.append((dist_sq, j))
            near.sort(key=lambda item: item[0])
            others = [
                agent_line(
                    pos,
                    vel,
                    disc,
                    agents[j]["pos"],
                    agents[j]["vel"],
                    agents[j]["disc"],
                    agent["time_horizon"],
                    dt,
                )
                for _, j in near[:_MAX_NEIGHBORS]
            ]
            agent["new"] = np.array(new_velocity(walls, others, agent["max_speed"], agent["pref"]))
        for agent in agents:
            if agent["yields"]:
                # What it chose becomes what it moves at, for the others' next solve and its own.
                agent["vel"] = agent["new"] if agent["present"] else np.zeros(2)

    def result(self, aid) -> np.ndarray:
        if aid == NO_AGENT:
            return np.zeros(2)
        agent = self._agents[aid]
        if not agent["yields"]:
            # A non-yielding agent executes exactly what it wanted: this model is not allowed to
            # move it, and returning ORCA's answer would do so through the back door.
            return agent["pref"]
        if not agent["present"]:
            return np.zeros(2)
        return np.array(agent["new"], dtype=float)

    def reset(self) -> None:
        for agent in self._agents:
            agent["pref"] = np.zeros(2)
            agent["vel"] = np.zeros(2)
            agent["new"] = np.zeros(2)
            # Nowhere until it is submitted again: a run does not start from where the last one
            # left its agents.
            agent["present"] = False
