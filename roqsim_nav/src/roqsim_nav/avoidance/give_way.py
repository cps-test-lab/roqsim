"""Reciprocal give-way: rules that get two movers past each other.

The default model. Each agent changes course by itself, to a side the encounter decides (below);
``orca`` solves every agent's velocity jointly, and resolves a crowd this does not.

What it does, per agent, per solve:

* look at the other agents within ``neighbor_dist``;
* for each, extrapolate the straight-line closest approach at the current velocities, and treat it
  as a conflict when that approach falls inside both radii plus the clearance it keeps --
  ``clearance``, and ``clearance_time`` more for each m/s the two close at -- within ``horizon``
  seconds;
* alter course to one side, more strongly the sooner and the tighter the conflict.

**What comes out is a course change, not a lateral step.** The lateral push is an intermediate: the
result is renormalised back to the commanded speed, so the model emits the same speed on a rotated
heading and never asks a body to translate sideways. That matters because the avoidance layer is
shared by everything that moves -- a wheeled base cannot strafe, and a model that told it to would
be unusable on the majority of what this package drives. How the course change is realised is the
embodiment's business: a differential base turns and drives, a car steers, a walker or an
omnidirectional base may strafe if it likes.

Two properties are what make it work rather than merely move things about:

**It picks a side by the encounter, and by a rule where the encounter does not say.** A mover
crossing its way is passed behind, on the side it comes from; one standing, or going the same way,
is passed on the side away from it. Head-on, and for one standing on its line, both give way to
their own right, so a head-on pair resolves into a pass rather than into the mirror-image dance two
greedy avoiders fall into. ``side: left`` mirrors that rule for a left-hand-traffic world; every
agent in a world must agree, which is why it is set once on the model rather than per agent.

**It is deterministic.** No sampling, no state carried between solves, no iteration order
dependence -- the same encounter replays identically, which is the point of an opponent.

What it is not: it does not reason about who yields to whom, it cannot squeeze through a gap it does
not fit, and it will not resolve a crowd the way a proper velocity-obstacle solver does. It is a
lower bound on useful, and ``orca`` is there when the encounter is the experiment rather than the
traffic around it. The forward caution probe still backs it up: giving way shapes a velocity, and
stopping is what happens when shaping is not enough.
"""

from __future__ import annotations

import numpy as np

from . import NO_AGENT, AvoidanceModel

#: Below this the relative motion is too slow for a closest-approach time to mean anything.
_STILL = 1e-6

#: Slower than this, another agent is standing for the side it is passed on.
_STANDING = 0.1

#: How far from its line, beside its heading, a standing agent decides the side it is passed on;
#: nearer, the rule (``side``) does.
_ON_THE_LINE = 0.05

#: The cosine past which another agent is coming the other way, head-on; between it and its negative
#: it is crossing.
_HEAD_ON = 0.7


class GiveWayModel(AvoidanceModel):
    """Deterministic reciprocal give-way, in pure numpy."""

    params_schema = ("neighbor_dist", "horizon", "clearance", "clearance_time", "strength", "side")

    #: Parameters it once read, and what replaced each.
    retired_params = {
        "margin": "the gap it keeps is `clearance` (m), and `clearance_time` (s) more for each m/s "
        "the two close at"
    }

    def __init__(self) -> None:
        self._agents: dict[int, dict] = {}
        self._result: dict[int, np.ndarray] = {}
        self._next = 0
        self.neighbor_dist = 4.0
        self.horizon = 4.0
        #: The gap it keeps between two discs passing (m), and how much more for each m/s they
        #: close at (s).
        self.clearance = 0.24
        self.clearance_time = 0.06
        self.strength = 1.6
        self.side = "right"
        self._excluded: dict[int, set[int]] = {}

    def configure(self, ctx, params: dict) -> None:
        self.neighbor_dist = float(params.get("neighbor_dist", self.neighbor_dist))
        self.horizon = float(params.get("horizon", self.horizon))
        self.clearance = float(params.get("clearance", self.clearance))
        self.clearance_time = float(params.get("clearance_time", self.clearance_time))
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
        and a course change that pushed an agent into a wall the planner had carefully routed around
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
            skip = self._excluded.get(aid, ())
            for other, they in self._agents.items():
                if other == aid or not they["present"] or other in skip:
                    continue
                push += self._avoid(me, they, pref)
            if not push.any():
                self._result[aid] = pref
                continue
            shaped = pref + push * speed
            # Keep the commanded SPEED: giving way changes where the mover is going, not how fast.
            # This is also what makes the result embodiment-neutral -- same speed, rotated heading,
            # which every base can execute, rather than a sideways translation only some can.
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
        clearance = (
            me["radius"]
            + they["radius"]
            + self.clearance
            + self.clearance_time * float(np.sqrt(speed_sq))
        )
        if miss >= clearance:
            return np.zeros(2)
        heading = pref / float(np.linalg.norm(pref))
        right = np.array([heading[1], -heading[0]])
        rule = right if self.side == "right" else -right
        their = they["vel"]
        their_speed = float(np.linalg.norm(their))
        if their_speed > _STANDING and float(heading @ their) / their_speed < -_HEAD_ON:
            # Head-on: give way to our own right (or left), consistently, so both sides choose
            # opposite directions in the world and the pair parts, where a "steer away from them"
            # rule leaves them mirroring each other.
            lateral = rule
        elif their_speed > _STANDING and float(heading @ their) / their_speed < _HEAD_ON:
            # Crossing: pass behind it, on the side it comes from.
            lateral = -right if float(their @ right) > 0.0 else right
        else:
            # Standing, or going our way: pass on the side away from it, by the rule on its line.
            off = float(rel_p @ right)
            lateral = rule if abs(off) < _ON_THE_LINE else (-right if off > 0.0 else right)
        urgency = 1.0 - t_closest / self.horizon
        tightness = 1.0 - miss / clearance
        return lateral * (self.strength * urgency * tightness)

    def result(self, aid) -> np.ndarray:
        if aid == NO_AGENT:
            return self._agents.get(aid, {}).get("pref", np.zeros(2))
        return self._result.get(aid, self._agents[aid]["pref"])

    def exclude(self, aid, other) -> None:
        self._excluded.setdefault(aid, set()).add(other)

    def reset(self) -> None:
        self._result = {}
        for agent in self._agents.values():
            agent["vel"] = np.zeros(2)
            agent["pref"] = np.zeros(2)
