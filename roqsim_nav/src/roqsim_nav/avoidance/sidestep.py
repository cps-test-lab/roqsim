"""Reciprocal sidestepping: enough to get two movers past each other, with no compiler.

The default, and it exists because the alternative was not one. ORCA is the better model and it
needs ``rvo2``, which publishes no wheel and is built from source -- so a plain ``pip install
roqsim_nav`` could only ever *stop* for traffic, and two movers meeting head-on stopped facing each
other for the rest of the trial. "Avoidance is available if you have a compiler" is not avoidance.

What it does, per agent, per solve:

* look at the other agents within ``neighbor_dist``;
* for each, extrapolate the straight-line closest approach at the current velocities, and treat it
  as a conflict when that approach falls inside both radii plus ``margin``, within ``horizon``
  seconds;
* push the preferred velocity **sideways**, more strongly the sooner and the tighter the conflict.

Two properties are what make it work rather than merely move things about:

**It picks a side by a rule, not by chance.** Both agents steer to their own right, so a head-on pair
resolves into a pass rather than into the mirror-image dance two greedy avoiders fall into. The rule
is the one road traffic and warehouse AGVs use, and it is the whole reason a symmetric encounter has
an asymmetric outcome. ``side: left`` mirrors it for a left-hand-traffic world; every agent in a
world must agree, which is why it is set once on the model rather than per agent.

**It is deterministic.** No sampling, no state carried between solves, no iteration order
dependence -- the same encounter replays identically, which is the point of an opponent.

What it is not: it does not reason about who yields to whom, it cannot squeeze through a gap it does
not fit, and it will not resolve a crowd the way a proper velocity-obstacle solver does. It is a
lower bound on useful, and ``orca`` is there when the encounter is the experiment rather than the
traffic around it. The forward caution probe still backs it up: sidestepping shapes a velocity, and
stopping is what happens when shaping is not enough.
"""

from __future__ import annotations

import numpy as np

from . import NO_AGENT, AvoidanceModel

#: Below this the relative motion is too slow for a closest-approach time to mean anything.
_STILL = 1e-6


class SidestepModel(AvoidanceModel):
    """Deterministic reciprocal sidestepping, in pure numpy."""

    params_schema = ("neighbor_dist", "horizon", "margin", "strength", "side")

    def __init__(self) -> None:
        self._agents: dict[int, dict] = {}
        self._result: dict[int, np.ndarray] = {}
        self._next = 0
        self.neighbor_dist = 4.0
        self.horizon = 4.0
        self.margin = 0.15
        self.strength = 1.6
        self.side = "right"

    def configure(self, ctx, params: dict) -> None:
        self.neighbor_dist = float(params.get("neighbor_dist", self.neighbor_dist))
        self.horizon = float(params.get("horizon", self.horizon))
        self.margin = float(params.get("margin", self.margin))
        self.strength = float(params.get("strength", self.strength))
        self.side = str(params.get("side", self.side))
        if self.side not in ("right", "left"):
            raise ValueError("avoidance 'side' must be 'right' or 'left'")

    def add_agent(self, key, *, radius, max_speed, yields, params):
        aid = self._next
        self._next += 1
        self._agents[aid] = {
            "key": key,
            "radius": float(radius),
            "max_speed": float(max_speed),
            "yields": bool(yields),
            "pos": np.zeros(2),
            "vel": np.zeros(2),
            "pref": np.zeros(2),
            "present": True,
        }
        return aid

    def add_static(self, polygons) -> None:
        """Walls are the planner's business here.

        Deliberately ignored rather than approximated: this model shapes a velocity around *movers*,
        and a sidestep that pushed an agent into a wall the planner had carefully routed around
        would be worse than none. The caution probe stops for anything this does not handle.
        """

    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        if aid == NO_AGENT:
            return
        agent = self._agents[aid]
        agent["pos"] = np.asarray(pos, dtype=float)[:2]
        agent["vel"] = np.asarray(vel, dtype=float)[:2]
        agent["pref"] = np.asarray(pref_vel, dtype=float)[:2]
        agent["present"] = bool(present)

    def solve(self, dt: float) -> None:
        self._result = {}
        for aid, me in self._agents.items():
            pref = me["pref"]
            speed = float(np.linalg.norm(pref))
            # A non-yielding agent is not this model's to move, and a stopped one has no course to
            # alter -- stopping is the caution probe's job, and overriding it here would undo it.
            if not me["yields"] or not me["present"] or speed < _STILL:
                self._result[aid] = pref
                continue
            push = np.zeros(2)
            for other, they in self._agents.items():
                if other == aid or not they["present"]:
                    continue
                push += self._avoid(me, they, pref)
            if not push.any():
                self._result[aid] = pref
                continue
            shaped = pref + push * speed
            # Keep the commanded SPEED: sidestepping changes where the mover is going, not how fast.
            # Letting the push add speed would make an agent accelerate out of a conflict, which
            # reads as panic and breaks the constant-speed assumption the follower is built on.
            norm = float(np.linalg.norm(shaped))
            self._result[aid] = pref if norm < _STILL else shaped / norm * speed

    def _avoid(self, me: dict, they: dict, pref: np.ndarray) -> np.ndarray:
        """The lateral push one neighbour contributes, or zero when there is no conflict."""
        rel_p = they["pos"] - me["pos"]
        distance = float(np.linalg.norm(rel_p))
        if distance > self.neighbor_dist or distance < _STILL:
            return np.zeros(2)
        # Our INTENDED velocity against their actual one: reacting to what they are doing rather
        # than to what they intend is what keeps the two sides of an encounter consistent.
        rel_v = pref - they["vel"]
        closing = float(rel_p @ rel_v)
        if closing <= 0.0:  # already separating
            return np.zeros(2)
        speed_sq = float(rel_v @ rel_v)
        if speed_sq < _STILL:
            return np.zeros(2)
        t_closest = closing / speed_sq
        if t_closest > self.horizon:
            return np.zeros(2)
        miss = float(np.linalg.norm(rel_p - rel_v * t_closest))
        clearance = me["radius"] + they["radius"] + self.margin
        if miss >= clearance:
            return np.zeros(2)
        # Steer to our own right (or left), consistently: both sides of a head-on encounter then
        # choose opposite directions in the world and the pair parts, where a "steer away from them"
        # rule leaves them mirroring each other.
        heading = pref / float(np.linalg.norm(pref))
        lateral = (
            np.array([heading[1], -heading[0]])
            if self.side == "right"
            else np.array([-heading[1], heading[0]])
        )
        urgency = 1.0 - t_closest / self.horizon
        tightness = 1.0 - miss / clearance
        return lateral * (self.strength * urgency * tightness)

    def result(self, aid) -> np.ndarray:
        if aid == NO_AGENT:
            return self._agents.get(aid, {}).get("pref", np.zeros(2))
        return self._result.get(aid, self._agents[aid]["pref"])

    def reset(self) -> None:
        self._result = {}
        for agent in self._agents.values():
            agent["vel"] = np.zeros(2)
            agent["pref"] = np.zeros(2)
