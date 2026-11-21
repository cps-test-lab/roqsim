# SPDX-License-Identifier: Apache-2.0
"""A humanoid's moves: one entity's procedures, started through its handle, each an event.

The ``humanoid`` plugin is nested under the entity whose body it moves -- a walker with
``hands: true``, or a humanoid robot's ``spawn_model`` with ``humanoid_robot`` -- and drives that
body (``humanoid_body:<entity>``, :class:`~roqsim_motion.body.HumanoidBody`) through the motion
core. It is a :class:`~roqsim.machine.MachinePlugin` that registers no entity of its own, so its
moves are programs with the machine contract unchanged: ``start(move, **args)`` returning a
sequence number, ``status()`` with ``finished`` and ``error``, ``busy``, ``cancel``, an event at
each step between ``program_start`` and ``program_end``, and the endpoints ``start``, ``status``
and ``cancel`` on the entity, so ``machine_run`` and the control socket drive a humanoid as they
drive a press. ``humanoid:<entity>`` publishes the same handle with a method per move.

**The line** (``docs/architecture.rst``, "Machines and scenarios"): a move is what the humanoid
does by itself. Which moves a worker makes, in which order and for what is a scenario's; a move never
decides what comes next. A move that cannot be done fails with the reason in its status, and the
humanoid stands where the move left it: a hold stays held, so a cancelled carry does not drop what
it carries.

Its keys are declared (``roqsim plugins describe humanoid``); nested under a walker::

    - walker: {walker: MaleVisitorWalk, hands: true}
      name: worker
      components:
        - navigator: {output: walker, arrival_radius: 0.05, avoidance: {stop: false}}
        - humanoid:
            graspable: [tote_*, cabinet]  # entities whose grip sites the hands may hold (fnmatch);
                                          # a hold is a weld per hand and grip site, set at build
            hold: {max_force: 300.0, slip: 0.06}  # past either (low-passed), a hold lets go and fails
            press: {max_force: 40.0}              # a press pushes no harder; a press move's
                                                  # `max_force` overrides it for that press

Refused when the world loads: a ``humanoid`` under a walker with ``hands: false`` (its moves need
hands: set ``hands: true`` on the walker), one beside a navigator whose ``arrival_radius`` is over
0.1 m (an approach would stop that far from where it works), and one beside a navigator with a patrol (the
navigator would walk the humanoid back onto its route after a move; send it with ``navigate``
between moves instead). Refused at build: a ``graspable`` pattern no grip site matches.
"""

from __future__ import annotations

import fnmatch
import math

import mujoco
import numpy as np

from roqsim.machine import MACHINE_SCHEMA, MachineError, MachineHandle, MachinePlugin, program
from roqsim.schema import Field
from roqsim_nav.footprint import Footprint, Piece, capsule_piece, convex_hull, geom_piece
from roqsim_nav.lattice import NoWay, plan_way_steps

from . import quat as Q
from .body import SIDES, HumanoidBody
from .contact import (
    HOLD_SOLREF,
    Hold,
    add_holds,
    check_solref,
    contact_force,
    touching_body,
)
from .targets import TargetError, make_target

HANDS = {"left": ("l",), "right": ("r",), "both": ("l", "r")}
#: How near a hand is to its goal when it has arrived (m, rad), and how long it may take after its
#: goal has (s).
ARRIVED = 0.01
ARRIVED_ROT = math.radians(6.0)
SETTLE = 2.0
#: How far in front of what it presses or grips a hand comes first (m), and how far from a grip
#: site a hold is made (m): the weld takes up the rest.
PRE = 0.08
HOLD_REACH = 0.02
#: The speed a press goes in at (m/s), and how far past the button's stroke it may go looking for it.
PRESS_SPEED = 0.05
PRESS_SEARCH = 0.04
#: The heights a humanoid looks for what is in front of where it stands at (m), foot to knee; how far
#: it keeps clear is its body's (HumanoidBody.stand_clear).
CLEAR_HEIGHTS = (0.1, 0.25, 0.4)
#: Below this height (m) a handle is taken as its frame says, not turned for carrying: off the floor.
CARRY_TURN_ABOVE = 0.45
#: How near a pull or push brings what it holds to its distance (m), and how far past the distance
#: its hand may go to get it there, at least (m; half the distance, if that is more).
ALONG_TOL = 0.003
ALONG_EXTRA = 0.05
#: A stroke that would leave a slide this near its stop goes to the stop (m): a hand's width.
ALONG_STOP = 0.03
#: How far a hand reaches from the body's root before a move refuses it without trying (m).
FAR = 1.4
#: How far above the highest thing under it -- a surface, a rim, a drawer's walls -- what is carried
#: goes before it is lowered (m).
CLEAR = 0.05
#: How near a humanoid's navigator must bring it to where it stands to work (m): its arrival_radius.
ARRIVAL = 0.1
#: How far apart (m) the surfaces under the corners of what is set down may be: more, and it would
#: stand on two things, or half off one -- a drawer's wall and its deck, a bench's edge and the floor.
LEVEL = 0.015
#: How far the grips of a load in both hands may stand above one another (m), and how often the
#: hand that holds it turns to bring them level before a carry fails.
LEVEL_GRIPS = 0.01
LEVEL_TRIES = 8
LEVEL_SETTLE = 0.3
LEVEL_GAIN = 0.5
#: How far a carry lifts what it holds straight up before drawing it in toward the body (m): over a
#: drawer's stops, a comb's teeth, a lip round what it stood on.
LIFT_FREE = 0.06
#: How near the place (m) what is set down comes over it before it is lowered: a drawer's deck
#: holds a box in stops with millimetres to spare, a centimetre from the fence over it.
OVER_AT = 0.004
#: How far (m) what is lowered may stay above where the hands lower it before it counts as resting
#: on what it touches.
CAUGHT = 0.03
#: The most a place turns what it sets down to stand square to the place (rad).
SQUARE_MAX = math.radians(20)
#: How fast a body carrying something turns on the spot (rad/s).
CARRY_TURN = 1.0
#: How far in front of a target a body stands to work at it, where the target states no stance and
#: the move gives no ``standoff`` (m).
STANDOFF = 0.6
#: How far a carry may step straight back at a time, by default (m): the limit of a step back on
#: its way (``roqsim_nav.lattice``); 0, never.
BACK_MAX = 0.5
#: How far a body walking a way may stray from it before the move fails (m), on top of the margin
#: the way was planned with (``HumanoidBody.follow_error``): what is checked is what is walked.
FOLLOW_TOL = 0.03
#: The radius of a holding arm in a carry's footprint, from shoulder to hand (m), and the height
#: (m) within which the body's geoms from the floor up make one piece of it.
ARM_RADIUS = 0.05
BODY_LAYER = 0.3
#: How many times a leg or a step back that ends short of where it goes is aimed on before the move
#: fails, and how near its end it must stand (m).
BACK_TRIES = 3
AT_END = 0.01
#: The time constant (s) a hold's force is judged over.
FORCE_TAU = 0.15

HOLD = {
    "max_force": Field(float, default=300.0, minimum=1.0, unit="N", doc="a hold lets go past this"),
    "slip": Field(
        float,
        default=0.06,
        minimum=0.001,
        unit="m",
        doc="a hold lets go when its frames part this far",
    ),
    "solref": Field(
        list, default=list(HOLD_SOLREF), length=2, doc="the hold's (timeconst, dampratio)"
    ),
}

#: How hard a press may push: the default of a press move's ``max_force``.
PRESS = {
    "max_force": Field(float, default=40.0, minimum=0.1, unit="N", doc="a press's force at most"),
}


def _side_word(side: str) -> str:
    return {"l": "left", "r": "right"}[side]


class HumanoidHandle(MachineHandle):
    """The machine handle with a method per move; each returns the start's sequence number."""

    def approach(self, target: str | dict, **kw) -> int:
        return self.start("approach", target=target, **kw)

    def reach(self, hand: str, target: str | dict, **kw) -> int:
        return self.start("reach", hand=hand, target=target, **kw)

    def press(self, hand: str, target: str, **kw) -> int:
        return self.start("press", hand=hand, target=target, **kw)

    def grasp(self, hand: str, target: str, **kw) -> int:
        return self.start("grasp", hand=hand, target=target, **kw)

    def release(self, hand: str = "both") -> int:
        return self.start("release", hand=hand)

    def pull(self, hand: str, distance: float, **kw) -> int:
        return self.start("pull", hand=hand, distance=distance, **kw)

    def push(self, hand: str, distance: float, **kw) -> int:
        return self.start("push", hand=hand, distance=distance, **kw)

    def carry(self, to: str | dict, **kw) -> int:
        return self.start("carry", to=to, **kw)

    def place(self, at: str | dict, **kw) -> int:
        return self.start("place", at=at, **kw)

    def play(self, clip: str, **kw) -> int:
        return self.start("play", clip=clip, **kw)

    def stop_clip(self, clip: str = "") -> int:
        return self.start("stop_clip", clip=clip)

    def look_at(self, target: str | list = "") -> int:
        return self.start("look_at", target=target)


class HumanoidPlugin(MachinePlugin):
    """A humanoid's moves (see the module docstring)."""

    registers_entity = False
    requires_owner = True
    ENTITY_KIND = "humanoid"
    HANDLE_CLASS = HumanoidHandle
    CONFIG_SCHEMA = {
        **MACHINE_SCHEMA,
        "graspable": Field(
            list, default=[], doc="entities whose grip sites the hands may hold (fnmatch)"
        ),
        "hold": Field(dict, schema=HOLD, doc="how hard a hold may pull and how far it may slip"),
        "press": Field(dict, schema=PRESS, doc="how hard a press may push"),
    }

    #: The force cap a press move and ``humanoid_press`` call ``max_force`` has the same name here.
    CONFIG_HINTS = {"press_force": "press: {max_force: <N>}, the name a press move's cap has too"}

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.body: HumanoidBody | None = None
        self.holding: dict[str, Hold | None] = {s: None for s in SIDES}
        self._force = {s: 0.0 for s in SIDES}
        self._apart = {s: 0.0 for s in SIDES}
        self._riding = {s: False for s in SIDES}  # a second hand riding on what the first holds
        self._stance = ""  # where the last walk stood: a stance site, or its stand-off
        self._turns = {
            s: 0.0 for s in SIDES
        }  # how far each hand turned on the handle it took (rad)
        self._holds: dict = {}

    # -- load and build -----------------------------------------------------------------------
    @classmethod
    def expand(cls, spec, world, base_dir):
        owner = next((w for w in world if w.address == spec.entity), None)
        if (
            owner is not None
            and owner.ref == "walker"
            and not (owner.config or {}).get("hands", False)
        ):
            raise ValueError(
                f"humanoid {spec.address}: its moves need hands -- set 'hands: true' on walker "
                f"{owner.address}"
            )
        if owner is not None and owner.ref == "spawn_model":
            # a robot's bodies are named by its prefix: they are its own, as a walker's are
            spec.config.setdefault("prefix", str((owner.config or {}).get("prefix", "")))
        navs = [c for c in (owner.children if owner is not None else []) if c.ref == "navigator"]
        for nav in navs:
            radius = float((nav.config or {}).get("arrival_radius", 0.25))
            if radius > ARRIVAL:
                raise ValueError(
                    f"humanoid {spec.address}: its navigator's arrival_radius is {radius} m, so an "
                    f"approach may stop that far from where it works; set 'arrival_radius' to at most "
                    f"{ARRIVAL} (0.05 stands within a hand's width)"
                )
        for sibling in owner.children if owner is not None else []:
            if sibling.ref == "navigator" and (
                (sibling.config or {}).get("goals") or (sibling.config or {}).get("loop")
            ):
                raise ValueError(
                    f"humanoid {spec.address}: its navigator has a patrol ('goals'/'loop'), which "
                    f"would walk it back onto its route after a move; drop the patrol and send it "
                    f"with `navigate` between moves"
                )
        return []

    def build_late(self, spec: mujoco.MjSpec, ctx) -> None:
        """One hold per hand grip site and graspable grip site; refuses a pattern that matches none."""
        names = [site.name for site in spec.sites]
        hands = [f"{self.entity}/grip_{s}" for s in SIDES]
        missing = [h for h in hands if h not in names]
        if missing:
            raise MachineError(
                f"humanoid {self.address}: {self.entity} has no grip sites ({', '.join(missing)}); "
                f"a walker needs 'hands: true', a robot a 'humanoid_robot' nested beside this"
            )
        error = check_solref(spec.option, tuple(self.settings.hold.solref))
        if error:
            raise MachineError(f"humanoid {self.address}: {error}")
        objects = []
        for pattern in self.settings.graspable:
            found = [
                n for n in names if _graspable(n, pattern) and not n.startswith(f"{self.entity}/")
            ]
            if not found:
                raise MachineError(
                    f"humanoid {self.address}: graspable {pattern!r} matches no grip site (a site "
                    f"named '<entity>/grip', '<entity>/grip_<x>' or '<entity>/<part>/grip')"
                )
            objects += [n for n in found if n not in objects]
        add_holds(spec, hands, objects, tuple(self.settings.hold.solref))
        self._objects = objects
        # What the humanoid may hold does not collide with its own body: a box held against the
        # thighs would be pressed into them by the hold. Its hands still press buttons, which are no
        # graspable.
        own = [b.name for b in spec.bodies if self._own(b.name)]
        held = {spec.site(o).parent.name for o in objects}
        for obj in sorted(held):
            for b in own:
                ex = spec.add_exclude()
                ex.bodyname1, ex.bodyname2 = obj, b

    def _own(self, body: str) -> bool:
        """Whether a body of the model is this humanoid's: under its entity's name, or its prefix."""
        prefix = str(self.config.get("prefix", ""))
        return body.startswith(f"{self.entity}/") or bool(prefix and body.startswith(prefix))

    def configure(self, ctx) -> None:
        super().configure(ctx)
        ctx.blackboard.set(f"humanoid:{self.machine_entity}", self.handle)
        m = ctx.model
        self._holds = {
            (side, obj): Hold.find(m, f"{self.entity}/grip_{side}", obj)
            for side in SIDES
            for obj in getattr(self, "_objects", [])
        }

    def on_reset(self, ctx) -> None:
        super().on_reset(ctx)
        self.body = ctx.blackboard.get(f"humanoid_body:{self.entity}")
        if self.body is None:
            raise MachineError(
                f"humanoid {self.address}: {self.entity} published no body (humanoid_body:"
                f"{self.entity}); nest it under a walker with 'hands: true' or a robot with "
                f"'humanoid_robot'"
            )
        self.holding = {s: None for s in SIDES}
        self._force = {s: 0.0 for s in SIDES}
        self._apart = {s: 0.0 for s in SIDES}
        self._riding = {s: False for s in SIDES}
        self._turns = {s: 0.0 for s in SIDES}

    # -- helpers ----------------------------------------------------------------------------
    @property
    def _m(self):
        return self._ctx.model

    @property
    def _d(self):
        return self._ctx.data

    def _sides(self, hand: str) -> tuple:
        if hand not in HANDS:
            self.fail(f"no hand {hand!r}: the hands are {', '.join(HANDS)}")
        return HANDS[hand]

    def _target(self, spec, offset=None):
        try:
            return make_target(self._m, self._ctx.entities, spec, offset)
        except TargetError as exc:
            self.fail(str(exc))

    def _site_id(self, name) -> int:
        """The site named ``name``, or -1: for no such site, and for a target that is a world pose."""
        if not isinstance(name, str):
            return -1
        return mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_SITE, name)

    def _grip_sites(self, target: str, sides) -> dict:
        """The grip site each side holds ``target`` by."""
        if self._site_id(target) >= 0:
            if len(sides) > 1:
                self.fail(f"{target} is one grip site; one hand holds it")
            return {sides[0]: target}
        out = {}
        for side in sides:
            for name in (f"{target}/grip_{side}", f"{target}/grip"):
                if self._site_id(name) >= 0:
                    out[side] = name
                    break
            else:
                self.fail(
                    f"{target} has no grip site for the {_side_word(side)} hand ({target}/grip_{side} or {target}/grip)"
                )
        if len(sides) > 1 and len(set(out.values())) < len(out):
            self.fail(f"{target} has one grip site; one hand holds it")
        return out

    def _far(self, pos) -> None:
        x, y, _ = self.body.root()
        d = math.hypot(pos[0] - x, pos[1] - y)
        if d > FAR:
            self.fail(
                f"out of reach: the target is {d:.2f} m from where it stands -- approach first"
            )

    def _move_hands(
        self, goals, speed: float, what: str, rot_tol: float = ARRIVED_ROT, pos_tol: float = ARRIVED
    ):
        """Move each side's grip frame to ``goals[side]()`` -- a callable giving ``(pos, quat)`` now,
        re-read every step so a hand follows a moving target -- at ``speed`` m/s, then wait for the
        body to arrive. Fails naming how far a hand was left, when it cannot get there."""
        d = self._d
        start = {s: self.body.grip(d, s) for s in goals}
        dist = max(float(np.linalg.norm(goals[s]()[0] - start[s][0])) for s in goals)
        duration = max(dist / max(speed, 1e-3), 0.3)
        t0 = self._ctx.sim_time
        while True:
            u = min((self._ctx.sim_time - t0) / duration, 1.0)
            u = u * u * (3.0 - 2.0 * u)  # ease in and out
            for s, goal in goals.items():
                gp, gq = goal()
                p = start[s][0] + u * (gp - start[s][0])
                q = None if gq is None else Q.slerp(start[s][1], gq, u)
                self.body.set_hand(s, p, q)
            self._ride()
            if u >= 1.0:
                break
            yield
        t1 = self._ctx.sim_time
        while True:
            errs = {}
            self._ride()
            for s, goal in goals.items():
                gp, gq = goal()
                self.body.set_hand(s, gp, gq)
                p, q = self.body.grip(d, s)
                errs[s] = (
                    float(np.linalg.norm(p - gp)),
                    0.0 if gq is None else Q.angle_between(q, gq),
                )
            if all(
                e[0] < pos_tol and e[1] < rot_tol and self.body.settled(s) for s, e in errs.items()
            ):
                return
            if self._ctx.sim_time - t1 > SETTLE:
                side, (e, a) = max(errs.items(), key=lambda kv: kv[1][0])
                self.fail(
                    f"out of reach: the {_side_word(side)} hand stays {e:.3f} m and "
                    f"{math.degrees(a):.0f} deg from {what} -- approach first"
                )
            yield

    def _site_goal(self, site: int, back: float = 0.0, turn: float = 0.0):
        """A goal on a site's frame, ``back`` m out along its -z: a hand comes onto a grip from where
        its fingers point from, above a handle its fingers close down over. ``turn`` (rad) turns the
        hand on the handle, about the palm's normal, its fingers away from the body."""

        def goal():
            p = self._d.site_xpos[site]
            r = self._d.site_xmat[site].reshape(3, 3)
            if turn:
                x, y, _ = self.body.root()
                away = np.array([p[0] - x, p[1] - y, 0.0])
                best = None
                for sign in (1.0, -1.0):
                    q = Q.mul(Q.from_matrix(r), Q.axis_angle("x", sign * turn))
                    score = float(Q.to_matrix(q)[:, 2] @ away)
                    if best is None or score > best[0]:
                        best = (score, Q.to_matrix(q))
                r = best[1]
            return p - back * r[:, 2], Q.from_matrix(r)

        return goal

    def _ride(self) -> None:
        """Keep a hand that rides on what the other holds on its own grip site."""
        for side, h in self.holding.items():
            if h is not None and self._riding[side]:
                self.body.set_hand(
                    side, *self._site_goal(h.obj_site, turn=self._turns.get(side, 0.0))()
                )

    def _check_holds(self) -> None:
        """Let go of a hold, and fail, once it has pulled past ``max_force`` for a while (its force
        low-passed over :data:`FORCE_TAU`: a body posed at its own rate moves a hand in steps, and
        one step is no pull) or its frames have parted past ``slip`` for as long."""
        hold = self.settings.hold
        self._ride()
        a = min(self._ctx.dt / FORCE_TAU, 1.0)
        for side, h in self.holding.items():
            if h is None:
                continue
            if self._riding[side]:
                continue
            self._force[side] += a * (
                h.force(self._d, h.slide_axis(self._m, self._d)) - self._force[side]
            )
            self._apart[side] += a * (h.separation(self._d) - self._apart[side])
            force, apart = self._force[side], self._apart[side]
            if force > hold.max_force or apart > hold.slip:
                for s2, h2 in self.holding.items():  # a hand riding on it lets go with it
                    if h2 is not None and h2.obj_body == h.obj_body:
                        h2.set(self._d, False)
                        self.holding[s2] = None
                        self._riding[s2] = False
                self.fail(
                    f"lost hold: the {_side_word(side)} hand pulled {force:.0f} N (at most "
                    f"{hold.max_force:.0f}) and slipped {apart:.3f} m (at most {hold.slip})"
                )

    def _held(self, hand: str):
        sides = [s for s in self._sides(hand) if self.holding[s] is not None]
        if not sides:
            self.fail(
                f"not holding with the {hand} hand" if hand != "both" else "not holding anything"
            )
        return sides

    def _stances(self, target) -> list:
        """The stances a target states: sites ``<target>/stand`` and ``<target>/stand_<k>`` (and
        for a grip site ``<x>/grip``, those of ``<x>``), on the floor, their +x the way to face --
        ``[(name, (x, y), yaw), ...]``."""
        if not isinstance(target, str):
            return []
        m, d = self._m, self._d
        bases = {target}
        if target.rsplit("/", 1)[-1].startswith("grip") and "/" in target:
            bases.add(target.rsplit("/", 1)[0])
        out = []
        for sid in range(m.nsite):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
            base, _, last = name.rpartition("/")
            if base in bases and (last == "stand" or last.startswith("stand_")):
                x = d.site_xmat[sid].reshape(3, 3)[:, 0]
                out.append((name, d.site_xpos[sid][:2].copy(), math.atan2(x[1], x[0])))
        return out

    def _stand_for(self, target: str, standoff: float | None, facing):
        """Where to stand, and which way to face, to work on ``target``: at its stance (a slot's
        ``…/stand`` site, the nearest of several) where it states one and neither ``standoff`` nor
        ``facing`` is given; else ``standoff`` (0.6 m left out) from its front. A grip site the
        model does not have (``<slot>/grip`` of a place or press slot) is refused before anything
        moves, not walked to by its slot's stance."""
        if isinstance(target, str) and "/" in target and self._site_id(target) < 0:
            base, last = target.rsplit("/", 1)
            if last.startswith("grip") and self._stances(base):
                self.fail(f"unknown target {target}: {base} states no grip, only where to stand")
        stances = self._stances(target) if standoff is None and facing is None else []
        if stances:
            x, y, _ = self.body.root()
            name, xy, yaw = min(stances, key=lambda s: math.hypot(s[1][0] - x, s[1][1] - y))
            self._stance = name
            return np.array(xy, float), yaw
        if standoff is None:
            standoff = STANDOFF
        self._stance = f"{float(standoff):.2f} m from its front"
        d, m = self._d, self._m
        gl, gr = self._site_id(f"{target}/grip_l"), self._site_id(f"{target}/grip_r")
        site = self._site_id(target)
        handle = -1  # the body a hand goes to: what stands proud of the rest, so not in the way
        if gl >= 0 and gr >= 0:
            handle = int(m.site_bodyid[gl])
            # Carried in front: the left grip on the left, the right on the right.
            pl, pr = d.site_xpos[gl], d.site_xpos[gr]
            point = (pl + pr) / 2
            v = pl - pr
            direction = np.array([v[1], -v[0]])
        elif site >= 0:
            point = d.site_xpos[site].copy()
            x = d.site_xmat[site].reshape(3, 3)[:, 0]
            name = target.rsplit("/", 1)[-1]
            # a grip site's +x runs from the palm to the object; any other site faces out along it
            direction = (
                np.array([x[0], x[1]]) if name.startswith("grip") else -np.array([x[0], x[1]])
            )
            if name.startswith("grip"):
                handle = int(m.site_bodyid[site])
        else:
            point, _ = self._target(target).resolve(self._m, d)
            x, y, _ = self.body.root()
            direction = np.array([point[0] - x, point[1] - y])
        if facing is not None:
            direction = np.array([math.cos(facing), math.sin(facing)])
        n = float(np.linalg.norm(direction))
        if n < 1e-9:
            self.fail(f"cannot face {target}: no direction to it")
        direction /= n
        stand = np.array([point[0], point[1]]) - standoff * direction
        # Coming to take something by its handles, stand clear of what it stands on or in: a
        # stand-off from a tote in the middle of a shelf would put the legs in the shelf. Step back
        # until nothing is nearer than the body keeps clear in front; the arms and the bend reach
        # the rest.
        # (Coming to a surface, its stand-off is the scene's, as near as setting down needs.)
        if handle >= 0:
            free = self._clear_ahead(stand, direction, standoff, handle)
            clear = self.body.stand_clear
            if free < clear:
                stand = stand - (clear - free) * direction
        return stand, math.atan2(direction[1], direction[0])

    def _clear_ahead(self, stand, direction, reach: float, handle: int) -> float:
        """How far in front of ``stand`` (toward ``direction``, up to ``reach``) the nearest solid
        thing is, between foot and knee height: rays past the humanoid's own bodies, what it holds
        and ``handle`` (and what is welded to it)."""
        m, d = self._m, self._d
        skip = {int(m.body_weldid[h.obj_body]) for h in self.holding.values() if h is not None}
        if handle >= 0:
            skip.add(int(m.body_weldid[handle]))
        ray = np.array([direction[0], direction[1], 0.0])
        geomid = np.zeros(1, np.int32)
        nearest = reach
        for z in CLEAR_HEIGHTS:
            start = np.array([stand[0], stand[1], z])
            gone = 0.0
            while gone < nearest:
                dist = mujoco.mj_ray(m, d, start, ray, None, 1, -1, geomid)
                if dist < 0 or gone + dist >= nearest:
                    break
                g = int(geomid[0])
                body = int(m.geom_bodyid[g])
                solid = bool(m.geom_contype[g] or m.geom_conaffinity[g])
                if solid and int(m.body_weldid[body]) not in skip and not self.body.owns(m, body):
                    nearest = gone + dist
                    break
                gone += dist + 1e-4
                start = start + (dist + 1e-4) * ray
        return nearest

    def _via(self, via) -> list:
        """The floor points a move walks through before its stand-off, checked: each ``[x, y]``,
        finite."""
        points = []
        for p in via or []:
            try:
                x, y = (float(v) for v in p)
            except (TypeError, ValueError):
                self.fail(f"via: a point is [x, y] on the floor, not {p!r}")
            if not (math.isfinite(x) and math.isfinite(y)):
                self.fail(f"via: a point is [x, y] on the floor, not {p!r}")
            points.append((x, y))
        return points

    def _back_max(self, back_max) -> float:
        """``back_max`` as a distance (m), failing the move where it is none."""
        if not (math.isfinite(float(back_max)) and float(back_max) >= 0.0):
            self.fail(f"back_max: a distance of 0 m or more, not {back_max!r}")
        return float(back_max)

    def _leg(self, nav, stand, what: str, while_walking=None, via=()):
        """Walk through ``via`` to ``stand``, facing where it walks; carrying (``while_walking``),
        it turns at a carrying pace."""
        self.body.face(None)  # it faces where it walks until it arrives
        self.body.limit_turn(CARRY_TURN if while_walking is not None else None)
        try:
            yield from self._route(nav, stand, what, while_walking, via)
        finally:
            self.body.limit_turn(None)

    def _walk(self, stand, yaw, what: str, via=()):
        """Walk empty-handed through ``via`` to ``stand``, on the navigator's own plan, and face
        ``yaw``."""
        nav = self._ctx.blackboard.get(f"nav:{self.entity}:handle")
        if nav is None:
            self.fail(f"{self.entity} has no navigator to walk to {what} with")
        yield from self._leg(nav, stand, what, None, via)
        yield from self._face(yaw, what, None)
        yield from self._come_to_rest(what, None)
        self.step(
            "arrived", at=[float(stand[0]), float(stand[1])], yaw=float(yaw), stance=self._stance
        )

    def _face(self, yaw, what: str, each):
        """Turn the standing body to ``yaw`` and wait until it faces it (within 3 degrees)."""
        self.body.face(yaw)
        t0 = self._ctx.sim_time
        while abs(
            math.atan2(math.sin(self.body.root()[2] - yaw), math.cos(self.body.root()[2] - yaw))
        ) > math.radians(3):
            if each is not None:
                each()
            if self._ctx.sim_time - t0 > 5.0:
                self.fail(
                    f"cannot face {what}: it turned to {math.degrees(self.body.root()[2]):.0f} deg "
                    f"of {math.degrees(yaw):.0f}"
                )
            yield

    def _footprint_now(self) -> tuple[Footprint, list, str]:
        """The body and what its hands hold, as they are now, as a footprint in the body's frame --
        its collision geoms, every solid geom of what it holds and of what rides in it, and each
        holding arm from shoulder to hand -- with the weld roots of what is held (not obstacles to
        its own way) and the name of what it holds."""
        m, d = self._m, self._d
        root = self.body.root()
        x, y, yaw = root
        c, s = math.cos(-yaw), math.sin(-yaw)
        held = sorted({h.obj_body for h in self.holding.values() if h is not None})
        roots = {int(m.body_weldid[b]) for b in held}
        for b in held:
            roots |= {int(m.body_weldid[r]) for r in self._contents(b)}
        # the body's own parts are its footprint, not obstacles to it (a robot's links have DOFs)
        own = {int(m.body_weldid[b]) for b in range(1, m.nbody) if self.body.owns(m, b)}
        pieces = []
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            body = int(m.geom_bodyid[g])
            mine = self.body.owns(m, body)
            if not mine and int(m.body_weldid[body]) not in roots:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            # a part parked below the floor (a press proxy) has no piece
            label = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or f"body {body}"
            piece = geom_piece(m, d, g, root, f"{'body' if mine else 'held'}:{label}")
            if piece is not None:
                piece.body = mine
                pieces.append(piece)
        for side, h in self.holding.items():
            if h is None:
                continue
            ends = []
            for point in (self.body.shoulder(d, side), self.body.grip(d, side)[0]):
                rel = np.asarray(point, float)
                ends.append(
                    (
                        (rel[0] - x) * c - (rel[1] - y) * s,
                        (rel[0] - x) * s + (rel[1] - y) * c,
                        rel[2],
                    )
                )
            pieces.append(capsule_piece(*ends, ARM_RADIUS, f"arm:{_side_word(side)}"))
        # A few pieces, each a convex hull of parts at about one height, as tall as they are:
        # the body's geoms grouped from the floor up (a group takes the geoms that start within
        # BODY_LAYER of its lowest -- legs, hips, trunk), and each thing it holds whole. A hull
        # is conservative, and a few pieces are a few rasters and polygons to check.
        merged = []
        body = sorted((p for p in pieces if p.what.startswith("body:")), key=lambda p: p.z[0])
        while body:
            lo = body[0].z[0]
            group = [p for p in body if p.z[0] < lo + BODY_LAYER]
            body = body[len(group) :]
            hull = convex_hull(np.concatenate([p.polygon for p in group]))
            merged.append(Piece(hull, (lo, max(p.z[1] for p in group)), "body", body=True))
        held_pieces = [p for p in pieces if p.what.startswith("held:")]
        if held_pieces:
            hull = convex_hull(np.concatenate([p.polygon for p in held_pieces]))
            band = (min(p.z[0] for p in held_pieces), max(p.z[1] for p in held_pieces))
            merged.append(Piece(hull, band, "what it holds"))
        merged += [p for p in pieces if p.what.startswith("arm:")]
        pieces = merged
        name = ", ".join(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body {b}" for b in held
        )
        margin = self.body.follow_error(self._nav_speed())
        return Footprint(pieces, margin=margin), sorted(roots | own - {0}), name

    def _nav_speed(self) -> float:
        """The speed the body's navigator walks it at (m/s); 1 where it states none. A way's legs
        are sent ``exact``, which walks them at this speed: an avoidance model that may command a
        body up to its ``max_speed`` does not steer an exact route."""
        nav = self._ctx.blackboard.get(f"nav:{self.entity}")
        config = getattr(nav, "config", None) or {}
        return float(config.get("speed", 1.0))

    def _dock(self, target, held) -> list:
        """The bodies a body may stand against at ``target``: the target's own (its body and
        everything under it in the tree) and the surface under it -- what a worker stands at to
        work."""
        m, d = self._m, self._d
        try:
            t = make_target(m, self._ctx.entities, target)
        except TargetError:
            return []  # a slot named by its stances alone: nothing to stand against but the floor
        names = set()
        if t.kind != "world" and t.body > 0:
            for b in range(1, m.nbody):
                x = b
                while x > 0 and x != t.body:
                    x = int(m.body_parentid[x])
                if x == t.body:
                    names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body {b}")
        point, _ = t.resolve(m, d)
        under = held[0] if held else -1
        _top, body = self._surface(point, under)
        if body > 0:
            names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or f"body {body}")
        return sorted(names)

    def _walk_way(self, stand, yaw, what: str, carried, via=(), back_max=BACK_MAX, long_way=False):
        """Holding something: plan a way for the body and what it holds to ``stand`` facing
        ``yaw`` (:func:`roqsim_nav.lattice.plan_way`), through ``via``, and walk exactly that way --
        its legs exact on the navigator, its turns on the spot, its steps back -- what it holds
        held in the hands all along."""
        nav = self._ctx.blackboard.get(f"nav:{self.entity}:handle")
        if nav is None:
            self.fail(f"{self.entity} has no navigator to walk to {what} with")
        fp, skip, held = self._footprint_now()
        label = what if isinstance(what, str) else "where it is sent"
        each = lambda: self._hold_carried(carried)  # noqa: E731
        # while the way is planned the body stands as it is, its heading kept: turning, it would
        # swing what it holds
        self.body.keep_heading(self.body.root()[2])
        try:
            # planned in steps, the simulation stepping on between them: the body stands holding
            # what it holds, and no tick waits for the whole search
            steps = plan_way_steps(
                self._m,
                self._d,
                fp,
                self.body.root(),
                (float(stand[0]), float(stand[1]), float(yaw)),
                via=via,
                back_max=back_max,
                long_way=long_way,
                skip_roots=skip,
                what=held,
                to=label,
                dock=self._dock(what, [h.obj_body for h in self.holding.values() if h is not None]),
            )
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    way = done.value
                    break
                each()
                yield
        except NoWay as err:
            if fp.margin > 0.0:
                self.fail(
                    f"{err}; planned {fp.margin:.3f} m clear of everything, how far the body "
                    "strays from its way"
                )
            self.fail(str(err))
        finally:
            # however the planning ends -- a way, none, a failure while it ran, a cancel -- the
            # heading is the body's own again
            self.body.keep_heading(None)
        self.step("planned", **way.summary())
        tol = FOLLOW_TOL + fp.margin
        for seg in way.segments:
            if seg.kind == "walk":
                yield from self._walk_leg(nav, seg.poses, label, each, tol)
            elif seg.kind == "turn":
                if abs(seg.angle) > math.pi + 1e-6:
                    self.step(
                        "turning_long_way",
                        to=float(seg.poses[-1][2]),
                        turn=float(seg.angle),
                    )
                yield from self._spin(seg.angle, each)
            elif seg.kind == "back":
                yield from self._step_back(seg.distance, each, label, tol)
        yield from self._face(yaw, what, each)
        yield from self._come_to_rest(what, each)
        self.step(
            "arrived", at=[float(stand[0]), float(stand[1])], yaw=float(yaw), stance=self._stance
        )

    def _walk_leg(self, nav, poses, what: str, each, tol: float):
        """Walk one leg of a way through its poses, exactly (no plan, no steering), facing where it
        walks at a carrying pace; fail where the body leaves the leg by more than ``tol``; then
        stand on its end within :data:`AT_END`, aimed on by what it lacks (a body drawn behind its
        root comes to rest short) :data:`BACK_TRIES` times at most, and fail where it still stands
        short."""
        points = [(float(p[0]), float(p[1])) for p in poses]
        line = np.asarray(points)
        end = points[-1]
        self.body.face(None)
        self.body.limit_turn(CARRY_TURN)
        try:
            check = lambda: self._on_leg(line, tol, what, each)  # noqa: E731
            yield from self._route(nav, end, what, check, points[1:-1], exact=True)
            for k in range(BACK_TRIES + 1):
                yield from self._come_to_rest(what, each)
                x, y, _ = self.body.root()
                short = math.hypot(end[0] - x, end[1] - y)
                if short < AT_END:
                    break
                if k == BACK_TRIES:
                    self.fail(
                        f"stopped {short:.3f} m short of a leg's end on its way to {what}, at "
                        f"({x:.2f}, {y:.2f}), aimed on {BACK_TRIES} times"
                    )
                # the last few millimetres, straight on; aimed past by what it lacks
                ux, uy = (end[0] - x) / short, (end[1] - y) / short
                aim = (end[0] + ux * min(short, 0.04), end[1] + uy * min(short, 0.04))
                yield from self._route(nav, aim, what, check, exact=True)
        finally:
            self.body.limit_turn(None)

    def _on_leg(self, line, tol: float, what: str, each) -> None:
        """Hold what is carried, and fail if the body has left ``line`` (a polyline) by ``tol``."""
        each()
        x, y, _ = self.body.root()
        p = np.array([x, y])
        a, b = line[:-1], line[1:]
        ab = b - a
        t = np.clip(
            np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12),
            0,
            1,
        )
        off = float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1))) if len(a) else 0.0
        if off > tol:
            self.fail(
                f"left its way to {what} by {off:.3f} m at ({x:.2f}, {y:.2f}); it keeps to "
                f"{tol:.3f} m of it"
            )

    def _spin(self, angle: float, each):
        """Turn on the spot by ``angle`` (rad, signed) at :data:`CARRY_TURN`, a quarter turn at a
        time so that it goes round the way it is told, calling ``each`` every step."""
        heading0 = self.body.root()[2]
        legs = max(1, int(math.ceil(abs(angle) / (math.pi / 2))))
        for k in range(1, legs + 1):
            # on through each quarter but the last, not stopping at it
            tol = math.radians(3 if k == legs else 25)
            yield from self._turn_to(heading0 + angle * k / legs, each, tol)

    def _come_to_rest(self, what: str, while_walking):
        """Wait for a body drawn behind its root to come to rest: until then what it carries goes
        along with it, swinging out round a corner it turns at, and its hands are not where a move
        that follows would start them."""
        t0 = self._ctx.sim_time
        while not self.body.at_rest():
            if self._ctx.sim_time - t0 > 3.0:
                self.fail(f"cannot stand at {what}: the body did not come to rest in 3 s")
            if while_walking is not None:
                while_walking()
            yield

    def _route(self, nav, stand, what: str, while_walking, via=(), exact: bool = False):
        """Walk the navigator through ``via`` to ``stand``. A move that ends before the route does --
        cancelled, or failed while walking -- cancels the route, so the body stops where it is rather than walking
        on without the move; a route another sender replaced is theirs and left running."""
        goals = [*via, (float(stand[0]), float(stand[1]))]
        seq = nav.send_goals(goals, exact=True) if exact else nav.send_goals(goals)
        ended = False
        try:
            self.step("walking", to=[float(stand[0]), float(stand[1])])
            while True:
                if while_walking is not None:
                    while_walking()
                status = nav.status()
                if status[0] == seq and status[1]:
                    ended = True
                    if status[4]:
                        self.fail(f"no route to {what}: {status[4]}")
                    break
                if status[0] > seq:
                    ended = True
                    self.fail(f"the route to {what} was replaced by another")
                yield
        finally:
            # a route another sender's the navigator has applied since the last poll is theirs and
            # left running; one sent and not yet applied is not seen here, and is cancelled too
            if not ended and nav.status()[0] <= seq:
                nav.cancel()

    # -- the moves ----------------------------------------------------------------------------
    @program
    def approach(
        self,
        target: str | dict,
        standoff: float | None = None,
        facing: float | None = None,
        via: list | None = None,
        back_max: float = BACK_MAX,
        long_way: bool = False,
    ):
        """Walk to where a target says to stand -- its stance, a slot's ``<target>/stand`` site (the
        nearest of several), facing its +x -- or, where it states none or ``standoff`` or
        ``facing`` is given, to ``standoff`` (0.6 m left out) in front of it facing it: a grip
        site's or a face's front, a tote's long side (its left grip on the left), or the target as
        seen from here. ``arrived`` says which (``stance``). ``via``:
        floor points ``[x, y]`` it walks through first, in order -- a way of the scene's, round
        what stands between. Holding something, it walks a way planned for the body and what it
        holds, as ``carry`` does: ``back_max`` and ``long_way`` as there."""
        points = self._via(via)
        back_max = self._back_max(back_max)
        stand, yaw = self._stand_for(target, standoff, facing)
        if any(h is not None for h in self.holding.values()):
            # holding something: it goes along, on a way planned for the body and it
            carried = self._carried()
            yield from self._walk_way(
                stand, yaw, target, carried, via=points, back_max=back_max, long_way=bool(long_way)
            )
        else:
            yield from self._walk(stand, yaw, target, via=points)

    @program
    def reach(
        self,
        hand: str,
        target: str | dict,
        offset: list | None = None,
        orient: bool = True,
        pose: str = "open",
        speed: float = 0.8,
    ):
        """Put a hand, or both, on a target (a grip site's frame, or a pose with an offset in its
        frame); the hand stays there until another move moves it or ``release`` frees it."""
        sides = self._sides(hand)
        if (
            self._site_id(target) < 0
            and offset is None
            and all(
                self._site_id(f"{target}/grip_{s}") >= 0 or self._site_id(f"{target}/grip") >= 0
                for s in sides
            )
        ):
            sites = self._grip_sites(target, sides)
            goals = {s: self._site_goal(self._site_id(sites[s])) for s in sides}
        else:
            t = self._target(target, offset)
            goals = {
                s: (
                    lambda t=t: (lambda pq: (pq[0], pq[1] if orient else None))(
                        t.resolve(self._m, self._d)
                    )
                )
                for s in sides
            }
        for s in sides:
            self._far(goals[s]()[0])
            self.body.set_pose(s, pose)
        yield from self._move_hands(goals, speed, target)
        self.step("reached", hand=hand, target=target)

    @program
    def press(
        self,
        hand: str,
        target: str,
        max_force: float | None = None,
        travel: float | None = None,
        pose: str = "point",
    ):
        """Press a button (or anything on a joint) with a fingertip until it has travelled its press,
        then let go: in from the front of its face along the face's -x, the hand turned so the
        pressing finger points along that axis (a hand with fingers; :meth:`HumanoidBody.press_aim`),
        the palm turned no more than that needs, and its drawn fingertip on the face."""
        (side,) = self._sides(hand) if hand != "both" else self.fail("a press is one hand's")
        site = self._site_id(target)
        if site < 0:
            self.fail(
                f"press {target}: a press needs a site on the button's face (e.g. '<button>/button')"
            )
        m, d = self._m, self._d
        body_id = int(m.site_bodyid[site])
        jnt = int(m.body_jntadr[body_id])
        if jnt < 0:
            self.fail(f"press {target}: its body has no joint, so nothing can be pressed")
        qadr = int(m.jnt_qposadr[jnt])
        stroke = float(m.jnt_range[jnt][1] - m.jnt_range[jnt][0]) if m.jnt_limited[jnt] else 0.01
        need = travel if travel is not None else 0.6 * stroke
        limit = max_force if max_force is not None else self.settings.press.max_force
        q0 = float(d.qpos[qadr])
        self._far(d.site_xpos[site])
        self.body.set_pose(side, pose)
        normal = d.site_xmat[site].reshape(3, 3)[:, 0].copy()  # out of the face
        aim_q = self._press_rotation(side, pose, -normal)

        def tip_goal(depth):
            def goal():
                aim = d.site_xpos[site] + normal * (0.0 - depth)
                offset = self.body.grip(d, side)[0] - self.body.tip(d, side)
                return aim + offset, aim_q

            return goal

        # judged by the fingertip's place: the tip goal follows the hand as it turns, and a hand its
        # wrist cannot turn all the way still presses with the finger where it is
        yield from self._move_hands({side: tip_goal(-PRE)}, 0.8, target, rot_tol=math.pi)
        t0 = self._ctx.sim_time
        while not self.body.hand_done(side):  # the finger points before it goes in
            if self._ctx.sim_time - t0 > SETTLE:
                self.fail(
                    f"press {target}: the {_side_word(side)} hand's {pose!r} pose did not form in {SETTLE} s"
                )
            self.body.set_hand(side, *tip_goal(-PRE)())
            yield
        self.body.press_proxy(d, side, True)
        depth, touched = -PRE, False
        while True:
            depth += PRESS_SPEED * self._ctx.dt
            goal = tip_goal(depth)
            self.body.set_hand(side, *goal())
            force = contact_force(m, d, body_id, exclude=None)
            if force > 0.0 and not touched:
                touched = True
                self.step("contact", target=target)
            moved = abs(float(d.qpos[qadr]) - q0)
            if moved >= need:
                self.step("pressed", target=target, travel=moved, force=force)
                break
            if force >= limit or depth > stroke + PRESS_SEARCH:
                self.body.press_proxy(d, side, False)
                self.fail(f"button did not move: {moved:.4f} of {need:.4f} m at {force:.1f} N")
            yield
        while depth > -PRE:
            depth -= 2 * PRESS_SPEED * self._ctx.dt
            self.body.set_hand(side, *tip_goal(depth)())
            yield
        self.body.press_proxy(d, side, False)
        self.body.free_hand(side)
        self.body.set_pose(side, "open")
        self.step("released", target=target)

    def _press_rotation(self, side: str, pose: str, into):
        """The grip frame's world rotation that points ``side``'s pressing finger along ``into``, the
        palm turned from where it faces as little as that needs; ``None`` for a hand that presses as
        it is turned."""
        along = self.body.press_aim(side, pose)
        if along is None:
            return None
        into = np.asarray(into, float) / np.linalg.norm(into)
        _, _, yaw = self.body.root()
        # the palm turned as little as aiming the finger needs: from where it faces now (a hand at
        # rest faces the thigh, so it points thumb up), else down, else toward the body's front
        now = Q.to_matrix(self.body.grip(self._d, side)[1])[:, 0]
        for palm in (
            now,
            np.array([0.0, 0.0, -1.0]),
            np.array([math.cos(yaw), math.sin(yaw), 0.0]),
        ):
            palm = palm - np.dot(palm, into) * into
            if np.linalg.norm(palm) > 0.3:
                break
        palm /= np.linalg.norm(palm)
        # the grip frame's own (finger, palm) pair, made orthonormal, onto the world's
        a1 = np.asarray(along, float)
        a2 = np.array([1.0, 0.0, 0.0]) - a1[0] * a1  # the grip frame's +x is out of the palm
        a2 /= np.linalg.norm(a2)
        local = np.column_stack([a1, a2, np.cross(a1, a2)])
        world = np.column_stack([into, palm, np.cross(into, palm)])
        return Q.from_matrix(world @ local.T)

    @program
    def grasp(self, hand: str, target: str, grip: str = "power"):
        """Take a target's grip site(s) in a hand or both: in from the front of the grip, close the
        fingers to its shape, and hold it -- a weld, so the physics carries what is held."""
        sides = self._sides(hand)
        if (
            self._site_id(target) < 0
            and not any(self._site_id(f"{target}/{n}") >= 0 for n in ("grip", "grip_l", "grip_r"))
            and self._ctx.entities.get(target) is None
        ):
            self.fail(f"nothing to grasp: unknown target {target!r}")
        sites = self._grip_sites(target, sides)
        holds = {}
        for s in sides:
            h = self._holds.get((s, sites[s]))
            if h is None:
                self.fail(
                    f"nothing to grasp: {target} is not graspable (graspable: {list(self.settings.graspable)})"
                )
            if self.holding[s] is not None:
                self.fail(f"the {_side_word(s)} hand is holding already; release it first")
            holds[s] = h
        ids = {s: self._site_id(sites[s]) for s in sides}
        # a hold of the position only (an object on a joint) does not care how the hand is turned
        tol = math.radians(25) if all(holds[s].whole for s in sides) else math.pi
        for s in sides:
            self._far(self._d.site_xpos[ids[s]])
            self.body.set_pose(s, "open")
        # a free object a hand will carry is taken with the hand turned on its handle as the body
        # carries (HumanoidBody.carry_turn): the hold keeps it so. Not off the floor: a hand reaching
        # down to a handle below the knees comes onto it fingers down, as its frame says.
        low = min(float(self._d.site_xpos[ids[s]][2]) for s in sides) < CARRY_TURN_ABOVE
        turn = self.body.carry_turn if all(holds[s].whole for s in sides) and not low else 0.0
        goal = {
            s: (lambda back=0.0, s=s: self._site_goal(ids[s], back=back, turn=turn)) for s in sides
        }
        yield from self._move_hands(
            {s: goal[s](PRE) for s in sides}, 0.8, target, rot_tol=tol, pos_tol=0.03
        )
        # judged by position: the hold makes the frames one, turning a free object to the hand
        # within what a hold takes up as it is made (see below)
        yield from self._move_hands(
            {s: goal[s]() for s in sides}, 0.3, target, rot_tol=tol, pos_tol=HOLD_REACH
        )
        self.step("closing", hand=hand, target=target)
        d = self._d
        for s in sides:
            obj = holds[s].obj_body
            self.body.set_pose(s, grip, touching=self._touching(s, obj, ids[s]))
            self.body.gripper(d, s, True)
        t0 = self._ctx.sim_time
        while not all(self.body.hand_done(s) for s in sides) and self._ctx.sim_time - t0 < 1.0:
            for s in sides:
                self.body.set_hand(s, *goal[s]()())
            yield
        for s in sides:
            apart = holds[s].separation(d)
            if apart > HOLD_REACH:
                self.fail(
                    f"hold not made: the {_side_word(s)} hand is {apart:.3f} m from {sites[s]}"
                )
        ride = self.body.two_hands == "ride"
        for i, s in enumerate(sides):
            # Each hand holds the object as it is in that hand now. Of two hands on one rigid object
            # the first holds it whole; the second rides on its own grip site, or holds it by its
            # position only, as the body says (HumanoidBody.two_hands).
            if i == 0 or not ride:
                holds[s].set(d, True, whole=None if i == 0 else False)
            self.holding[s] = holds[s]
            self._riding[s] = i > 0 and ride
            self._turns[s] = turn
            self._force[s] = self._apart[s] = 0.0
            self.body.set_hand(s, *self.body.grip(d, s))
        self.step("held", hand=hand, target=target)

    def _touching(self, side, obj_body: int, site: int):
        """A finger stops where its tip meets the held object (within 4 mm, along the palm's
        normal)."""
        m, d = self._m, self._d
        geomid = np.zeros(1, np.int32)

        def touching_(finger: int) -> bool:
            tips = self.body.fingertips(d, side)
            if tips is None:
                return True
            normal = d.site_xmat[site].reshape(3, 3)[:, 0].copy()
            dist = mujoco.mj_ray(m, d, tips[finger], normal, None, 1, -1, geomid)
            return 0.0 <= dist < 0.004 and int(m.geom_bodyid[geomid[0]]) == obj_body

        return touching_

    @program
    def release(self, hand: str = "both"):
        """Let go: the hold off, the fingers open, the hand back to the gait."""
        sides = self._held(hand)
        for s in sides:
            self.holding[s].set(self._d, False)
            self.holding[s] = None
            self._riding[s] = False
            self.body.set_pose(s, "open")
            self.body.gripper(self._d, s, False)
        yield from self.wait(0.3)
        for s in sides:
            self.body.free_hand(s)
        self.step("opened", hand=hand)

    def _along(self, sides, distance: float, axis, speed: float, sign: float, what: str):
        d = self._d
        h0 = (
            next(self.holding[s] for s in sides if not self._riding[s])
            if any(not self._riding[s] for s in sides)
            else self.holding[sides[0]]
        )
        obj0 = d.site_xpos[h0.obj_site].copy()
        if axis is None:
            body = h0.obj_body
            jnt = int(self._m.body_jntadr[body])
            if jnt >= 0 and self._m.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_SLIDE:
                axis = d.xaxis[jnt].copy()
            else:
                axis = -d.site_xmat[h0.obj_site].reshape(3, 3)[:, 0]
            # a pull comes toward the humanoid, a push goes away from it
            x, y, _ = self.body.root()
            if np.dot(axis[:2], np.array([x, y]) - obj0[:2]) < 0:
                axis = -axis
            axis = sign * np.asarray(axis, float)
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        starts = {s: self.body.grip(d, s) for s in sides}
        self.step(
            "moving", hand="both" if len(sides) > 1 else _side_word(sides[0]), distance=distance
        )
        at_stop = self._stop_check(h0.obj_body, axis)
        # a lagging hand's stroke that would leave a slide short of its stop by less than a hand's
        # width goes to it: a drawer pushed nearly shut is pushed shut, not left open by the lag
        left = self._to_stop(h0.obj_body, axis)
        to_stop = (
            not self.body.hands_track_goals and left is not None and left - distance < ALONG_STOP
        )
        duration = abs(distance) / max(speed, 1e-3)
        t0 = self._ctx.sim_time
        while True:
            u = min((self._ctx.sim_time - t0) / duration, 1.0)
            u = u * u * (3.0 - 2.0 * u)
            for s in sides:
                self.body.set_hand(s, starts[s][0] + u * distance * axis, starts[s][1])
            if at_stop():
                # a drawer at its end stop is all the way out (or shut): done, not stuck
                self._stop_at_stop(sides)
                break
            self._check_holds()
            if u >= 1.0:
                break
            yield
        # The stroke is the object's, not the hand's: a hand that lags its goal (a servo under load)
        # goes on, a little further at a time, until what it holds has moved the distance or is at
        # its stop -- at most ALONG_EXTRA past the distance.
        extra, cap = 0.0, max(ALONG_EXTRA, 0.5 * abs(distance))
        while not at_stop() and extra < cap:
            moved = float(np.dot(d.site_xpos[h0.obj_site] - obj0, axis))
            if moved >= distance - ALONG_TOL and not to_stop:
                break
            extra = min(extra + 0.25 * speed * self._ctx.dt, cap)
            for s in sides:
                self.body.set_hand(s, starts[s][0] + (distance + extra) * axis, starts[s][1])
            self._check_holds()
            yield
        if at_stop():
            self._stop_at_stop(sides)
        yield from self.wait(0.3)
        self._check_holds()
        moved = float(np.dot(d.site_xpos[h0.obj_site] - obj0, axis))
        if at_stop():
            return moved
        if moved < 0.8 * distance:
            self.fail(f"stuck: {what} moved {moved:.3f} of {distance:.3f} m")
        return moved

    def _stop_at_stop(self, sides) -> None:
        """At a stop, a posed hand's goal comes back onto the hand, so it pushes no further; a
        servo-driven hand's stays, since a goal moved back onto a hand that lags it would pull the
        drawer back by that lag."""
        if self.body.hands_track_goals:
            for s in sides:
                self.body.set_hand(s, *self.body.grip(self._d, s))

    def _to_stop(self, body: int, axis):
        """How far ``body``'s limited slide joint is from its limit in the direction of ``axis`` (m),
        or ``None`` for a body on no limited slide."""
        m, d = self._m, self._d
        jnt = int(m.body_jntadr[body])
        if jnt < 0 or m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_SLIDE or not m.jnt_limited[jnt]:
            return None
        q = float(d.qpos[int(m.jnt_qposadr[jnt])])
        lo, hi = (float(v) for v in m.jnt_range[jnt])
        return hi - q if float(np.dot(d.xaxis[jnt], axis)) > 0 else q - lo

    def _stop_check(self, body: int, axis):
        """A test for ``body``'s slide joint being at its limit in the direction of ``axis``
        (within 5 mm); always false for a body on no limited slide."""
        m, d = self._m, self._d
        jnt = int(m.body_jntadr[body])
        if jnt < 0 or m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_SLIDE or not m.jnt_limited[jnt]:
            return lambda: False
        qadr = int(m.jnt_qposadr[jnt])
        lo, hi = (float(v) for v in m.jnt_range[jnt])
        direction = float(np.dot(d.xaxis[jnt], axis))

        def at_stop() -> bool:
            q = float(d.qpos[qadr])
            return (direction > 0 and q > hi - 0.005) or (direction < 0 and q < lo + 0.005)

        return at_stop

    @program
    def pull(self, hand: str, distance: float, speed: float = 0.2, axis: list | None = None):
        """Pull what a hand holds toward the humanoid by ``distance`` m -- what it holds moves that far,
        the hand going on past it for a hand that lags -- along its slide joint for a drawer, else
        straight out of its grip. A drawer that reaches its end stop first is out."""
        sides = self._held(hand)
        moved = yield from self._along(sides, distance, axis, speed, 1.0, "what it pulls")
        self.step("pulled", distance=moved)

    @program
    def push(self, hand: str, distance: float, speed: float = 0.2, axis: list | None = None):
        """Push what a hand holds away from the humanoid by ``distance`` m -- what it holds moves that
        far, as a pull's -- a drawer shut, done when it reaches its stop."""
        sides = self._held(hand)
        moved = yield from self._along(sides, distance, axis, speed, -1.0, "what it pushes")
        self.step("pushed", distance=moved)

    def _carried(self):
        """The held hands' grip poses in the body's frame, to hold them there while it walks."""
        d = self._d
        x, y, yaw = self.body.root()
        inv = Q.axis_angle("z", -yaw)
        rel = {}
        for s in SIDES:
            if self.holding[s] is not None and not self._riding[s]:
                p, q = self.body.grip(d, s)
                rel[s] = (Q.rotate(inv, p - np.array([x, y, 0.0])), Q.mul(inv, q))
        return rel

    def _hold_carried(self, rel) -> None:
        x, y, yaw = self.body.root()
        rot = Q.axis_angle("z", yaw)
        for s, (p, q) in rel.items():
            self.body.set_hand(s, np.array([x, y, 0.0]) + Q.rotate(rot, p), Q.mul(rot, q))
        self._check_holds()

    @program
    def carry(
        self,
        to: str | dict,
        standoff: float | None = None,
        height: float | None = None,
        via: list | None = None,
        facing: float | None = None,
        back_max: float = BACK_MAX,
        long_way: bool = False,
    ):
        """Carry what the hands hold to a target: lift it to a carrying height in front of the body
        (``height``: the hands' height, m; default a hand's breadth under the pelvis, or high enough
        for what is held to clear the surface the target is on by :data:`CLEAR`), and walk to where
        the target says to stand, or a stand-off in front of it, holding it there, facing as
        ``approach`` faces -- on a way planned for the body and what it holds
        (:func:`roqsim_nav.lattice.plan_way`: every pose of both checked in height layers),
        through the floor points ``via`` in order, and walked exactly. The way steps straight back
        at most ``back_max`` (m; 0: never) at a time, and turns past half a turn only with
        ``long_way``, saying so in a ``turning_long_way`` event. A ``planned`` event says what the
        way cost; where there is none, it fails naming where it got to and what is in the way."""
        points = self._via(via)
        back_max = self._back_max(back_max)
        sides = self._held("both")
        d = self._d
        m = self._m
        obj = self.holding[sides[0]].obj_body
        if height is None:
            # a hand's breadth under the pelvis, or clear of the surface it goes to
            height = self.body.carry_height()
            lead = next(s for s in sides if not self._riding[s])
            dest = self._target(to)
            point, _ = dest.resolve(m, d)
            top = self._top(point, obj, self._on(dest))
            low = self._bottom(obj)[2]
            below = self.body.grip(d, lead)[0][2] - low
            height = max(height, top + below + CLEAR)
            # and clear of what it was lifted from: what it holds by CLEAR, and the arms that hold
            # it, which the way checks as capsules of ARM_RADIUS down to the hands, as well
            here = d.xpos[obj].copy()
            under = self._top(np.array([here[0], here[1], low]), obj, below=low + 0.01)
            grip_z = float(self.body.grip(d, lead)[0][2])
            height = max(height, grip_z + 0.03, under + CLEAR + below, under + CLEAR + ARM_RADIUS)
        x, y, yaw = self.body.root()
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        rel = {}
        for s in [s for s in sides if not self._riding[s]]:
            p, q = self.body.grip(d, s)
            # in front of the body, at the carrying height, the hands as far apart as they are
            ahead = float(np.dot(p - [x, y, 0.0], heading))
            pull_in = max(0.0, ahead - 0.42)
            rel[s] = lambda p=p, q=q, pull_in=pull_in: (
                np.array([p[0], p[1], height]) - pull_in * heading,
                q,
            )
        # Straight up off what it stood on first, by LIFT_FREE at most, before it comes in toward the
        # body: drawn in at once, it would drag what it stands in (a drawer's stops) along with it.
        low_z = min(float(self.body.grip(d, s)[0][2]) for s in rel)
        free = min(LIFT_FREE, max(0.0, height - low_z))
        if free > 0.005:
            up = {s: (lambda p=self.body.grip(d, s): (p[0] + [0.0, 0.0, free], p[1])) for s in rel}
            yield from self._move_hands(
                up, 0.3, "the carrying height", rot_tol=math.radians(25), pos_tol=0.05
            )
        # what is held turns with the hands, so the lift is judged by where they are
        yield from self._move_hands(
            rel, 0.3, "the carrying height", rot_tol=math.radians(25), pos_tol=0.05
        )
        self._check_holds()
        yield from self._level(sides, rel)
        # the height the hands holding came to (a riding one too): a lift ends within a few
        # centimetres of where it aims
        got = float(np.mean([self.body.grip(d, s)[0][2] for s in sides]))
        self.step("lifted", height=round(got, 3))
        carried = self._carried()
        stand, yaw = self._stand_for(to, standoff, facing)
        yield from self._walk_way(
            stand, yaw, to, carried, via=points, back_max=back_max, long_way=bool(long_way)
        )

    def _level(self, sides, rel):
        """Hold what both hands hold level: where a hand rides on the load the hand that holds it
        does not, its far grip is no more than LEVEL_GRIPS higher or lower than the holding hand's
        -- the holding hand turned about the way across between them until it is, LEVEL_TRIES
        times at most, then failing naming the tilt. The load hangs from the holding hand, so the
        hand's turn the arm leaves it, a few degrees, tilts a long load by as much at its far end."""
        d = self._d
        lead = [s for s in rel if not self._riding[s]]
        riding = [s for s in sides if self._riding[s]]
        if len(lead) != 1 or not riding:
            return
        s0 = lead[0]
        near, far = self.holding[s0].obj_site, self.holding[riding[0]].obj_site
        tilt = 0.0
        goal_p, goal_q = rel[s0]()
        for _ in range(LEVEL_TRIES):
            a, b = d.site_xpos[near], d.site_xpos[far]
            across = np.array([b[0] - a[0], b[1] - a[1], 0.0])
            span = float(np.linalg.norm(across))
            if span < 1e-6:
                return
            tilt = math.atan2(float(b[2] - a[2]), span)
            if abs(b[2] - a[2]) <= LEVEL_GRIPS:
                return
            axis = np.cross(across / span, [0.0, 0.0, 1.0])
            # its goal turned the other way about that axis, by the tilt: the far grip comes down
            # to level, whatever turn the arm leaves the hand short of its goal by
            goal_q = Q.mul(Q.axis_angle(axis, -LEVEL_GAIN * tilt), goal_q)
            yield from self._move_hands(
                {s0: lambda p=goal_p, q=goal_q: (p, q)},
                0.3,
                "level",
                rot_tol=math.radians(25),
                pos_tol=0.05,
            )
            yield from self.wait(LEVEL_SETTLE)  # the hand and what it holds come to rest
            self._check_holds()
        a, b = d.site_xpos[near], d.site_xpos[far]
        if abs(b[2] - a[2]) > LEVEL_GRIPS:
            name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, self.holding[s0].obj_body)
            self.fail(
                f"cannot hold {name} level: its far grip stays {abs(b[2] - a[2]):.3f} m "
                f"{'higher' if b[2] > a[2] else 'lower'} than the near one "
                f"({math.degrees(tilt):.1f} deg)"
            )

    def _extent(self, held: int):
        """The held object's world box now: its lowest and highest corners, of its solid geoms."""
        m, d = self._m, self._d
        corners = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != held or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(d.geom_xpos[g] + rot @ (c + h * [sx, sy, sz]))
        if not corners:
            p = d.xpos[held].copy()
            return p, p
        return np.min(corners, axis=0), np.max(corners, axis=0)

    def _bottom(self, held: int) -> np.ndarray:
        """Where the held object stands on: its ``/bottom`` site, or under its origin at the bottom
        of its solid geoms (:meth:`_extent`), wherever its origin is."""
        m, d = self._m, self._d
        site = self._site_id(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, held)}/bottom")
        if site >= 0:
            return d.site_xpos[site].copy()
        p = d.xpos[held].copy()
        p[2] = float(self._extent(held)[0][2])
        return p

    def _step_back(self, distance: float, each, what: str, tol: float):
        """Step straight back by ``distance`` (m), keeping the heading, on the navigator exactly
        (unplanned, unsteered), calling ``each`` every step and failing where the body leaves the
        straight step by more than ``tol`` -- the step the way was checked on; then come to rest. A
        step that falls short is aimed on by what it lacks (:data:`BACK_TRIES` times at most), to
        stand within :data:`AT_END` of where the way goes on from; one still short then fails."""
        nav = self._ctx.blackboard.get(f"nav:{self.entity}:handle")
        if nav is None:
            self.fail(f"{self.entity} has no navigator to step back with")
        x0, y0, h = self.body.root()
        back = np.array([-math.cos(h), -math.sin(h)])
        self.step("stepping_back", distance=float(distance))
        self.body.keep_heading(h)
        self.body.limit_turn(CARRY_TURN)
        try:
            aim = distance
            for k in range(BACK_TRIES):
                goal = (x0 + aim * back[0], y0 + aim * back[1])
                line = np.array([(x0, y0), goal])
                check = lambda line=line: self._on_leg(line, tol, what, each)  # noqa: E731
                yield from self._route(nav, goal, "a step back", check, exact=True)
                yield from self._come_to_rest("a step back", check)
                x, y, _ = self.body.root()
                short = distance - float(np.dot([x - x0, y - y0], back))
                if short < AT_END:
                    break
                if k == BACK_TRIES - 1:
                    self.fail(
                        f"stepped back {distance - short:.3f} m of {distance:.3f} on its way to "
                        f"{what}, aimed on {BACK_TRIES} times"
                    )
                aim += short
        finally:
            self.body.keep_heading(None)
            self.body.limit_turn(None)

    def _turn_to(self, yaw: float, each=None, tol: float = math.radians(3)):
        self.body.limit_turn(CARRY_TURN)
        self.body.face(yaw)
        t0 = self._ctx.sim_time
        try:
            while True:
                heading = self.body.root()[2]
                if abs(math.atan2(math.sin(yaw - heading), math.cos(yaw - heading))) < tol:
                    return
                if self._ctx.sim_time - t0 > 2 * math.pi / CARRY_TURN + 2.0:
                    self.fail(
                        f"cannot turn to {math.degrees(yaw):.0f} deg: it stays at {math.degrees(heading):.0f}"
                    )
                if each is not None:
                    each()
                yield
        finally:
            self.body.limit_turn(None)

    def _surface(
        self,
        point,
        held: int,
        accept=None,
        inside=None,
        above: float = 1.5,
        below: float | None = None,
    ):
        """The top of whatever is under ``point``, and its body: a ray cast down from ``above`` m
        over it, or from the height ``below`` (m) where given, so what is higher is not under it
        (a ray starting inside something does not see it), past what is held (and what is welded
        to it), what rides in or on it (:meth:`_contents`), the humanoid's own bodies, anything
        that collides with nothing, and with ``accept`` every body it refuses."""
        m, d = self._m, self._d
        start = np.array([point[0], point[1], point[2] + above if below is None else float(below)])
        geomid = np.zeros(1, np.int32)
        inside = self._contents(held) if inside is None else inside
        while True:
            dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), None, 1, held, geomid)
            if dist < 0:
                return np.asarray(point, float), -1
            hit = start + [0.0, 0.0, -dist]
            g = int(geomid[0])
            body = int(m.geom_bodyid[g])
            solid = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            part_of_held = int(m.body_weldid[body]) == int(m.body_weldid[held])
            if (
                solid
                and not part_of_held
                and int(m.body_rootid[body]) not in inside
                and not self.body.owns(m, body)
                and (accept is None or accept(body))
            ):
                return np.array([point[0], point[1], hit[2]]), body
            start = hit - [0.0, 0.0, 1e-4]

    def _contents(self, held: int) -> set:
        """What rides in or on the held object: the free bodies (by their root) inside its extent,
        and those touching it, or touching one that does, standing on it -- above its origin, which
        is its underside or its middle. The parts in a box are not a surface it is set down on,
        whether they rest on its floor at the moment or are jolted off it by the walk."""
        m, d = self._m, self._d
        base = float(d.xpos[held][2])
        held_root = int(m.body_rootid[held])

        def free(root: int) -> bool:
            j = int(m.body_jntadr[root])
            return j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE

        touching_ = {}
        for i in range(d.ncon):
            c = d.contact[i]
            a = int(m.body_rootid[m.geom_bodyid[c.geom1]])
            b = int(m.body_rootid[m.geom_bodyid[c.geom2]])
            touching_.setdefault(a, set()).add(b)
            touching_.setdefault(b, set()).add(a)
        found, todo = set(), [held_root]
        # inside the held object's extent: its geoms' world boxes, as the footprint reads them
        corners = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != held or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(d.geom_xpos[g] + rot @ (c + h * [sx, sy, sz]))
        if corners:
            lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
            for root in set(int(r) for r in m.body_rootid):
                if root in (0, held_root) or not free(root) or self.body.owns(m, root):
                    continue
                p = d.xpos[root]
                if (lo[:2] <= p[:2]).all() and (p[:2] <= hi[:2]).all() and base < p[2] <= hi[2]:
                    found.add(root)
                    todo.append(root)
        while todo:
            for other in touching_.get(todo.pop(), ()):
                if other in found or other == held_root or not free(other):
                    continue
                if float(d.xpos[other][2]) <= base:
                    continue
                found.add(other)
                todo.append(other)
        return found

    def _footprint(self, held: int, spacing: float | None = None) -> list:
        """The held object's horizontal extent around its origin, as it is turned now: the corners
        of its geoms' boxes, inset by a centimetre, and its origin; with ``spacing`` (m), a grid
        over that extent as well, so a wall thinner than the object is found under it."""
        m, d = self._m, self._d
        pts = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != held or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append(d.geom_xpos[g] + rot @ (c + h * [sx, sy, 0.0]) - d.xpos[held])
        if not pts:
            return [np.zeros(3)]
        pts = np.array(pts)
        lo, hi = pts[:, :2].min(axis=0) + 0.01, pts[:, :2].max(axis=0) - 0.01
        if spacing:
            xs = np.linspace(lo[0], hi[0], max(2, int(math.ceil((hi[0] - lo[0]) / spacing)) + 1))
            ys = np.linspace(lo[1], hi[1], max(2, int(math.ceil((hi[1] - lo[1]) / spacing)) + 1))
        else:
            xs, ys = (lo[0], hi[0]), (lo[1], hi[1])
        return [np.zeros(3)] + [np.array([x, y, 0.0]) for x in xs for y in ys]

    def _on(self, target):
        """A test for a body being the target's, or under it in the tree: what is set down on a
        drawer clears the drawer's walls, not the cabinet's top above it."""
        m = self._m
        if target.kind == "world":
            return None
        root = target.body

        def accept(body: int) -> bool:
            while body > 0:
                if body == root:
                    return True
                body = int(m.body_parentid[body])
            return body == root

        return accept

    def _top(
        self, point, held: int, accept=None, spacing: float | None = 0.04, below=None
    ) -> float:
        """The highest surface under the held object's footprint (:meth:`_footprint`, a grid at
        ``spacing``) if it stood with its origin over ``point``; with ``accept``, of the bodies it
        accepts; with ``below``, of what is lower than that height (m)."""
        tops = []
        inside = self._contents(held)
        for off in self._footprint(held, spacing):
            p, under = self._surface(
                np.asarray(point, float) + off, held, accept, inside, below=below
            )
            if under >= 0:
                tops.append(float(p[2]))
        return max(tops) if tops else float(point[2])

    def _square_to(self, target, held: int) -> float:
        """How far to turn the held object about the vertical (rad) to stand square to ``target``'s
        frame -- the nearest quarter turn of its heading; 0 for a world pose with no heading
        stated, or a turn of more than SQUARE_MAX, which is not a matter of squaring."""
        if not target.heading:
            return 0.0
        _, tq = target.resolve(self._m, self._d)
        tm = Q.to_matrix(tq)
        om = self._d.xmat[held].reshape(3, 3)
        rel = math.atan2(om[1, 0], om[0, 0]) - math.atan2(tm[1, 0], tm[0, 0])
        rel = math.atan2(math.sin(rel), math.cos(rel))
        off = rel - round(rel / (math.pi / 2)) * (math.pi / 2)
        return -off if abs(off) <= SQUARE_MAX else 0.0

    def _path_top(self, a, b, held: int, z: float, below=None) -> float:
        """The highest surface under the held object's footprint (:meth:`_footprint`) with its
        origin anywhere on the way from ``a`` to ``b`` (floor points), whatever it belongs to, and
        with ``below`` lower than that height (m): in steps finer than the footprint's grid, so
        that together they pass over a wall 2 cm thick."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / 0.015)))
        return max(
            self._top(np.array([*(a + k / steps * (b - a)), z]), held, below=below)
            for k in range(steps + 1)
        )

    @program
    def place(self, at: str | dict, release: bool = True, speed: float = 0.15):
        """Set what the hands hold down on a target: over it, then lowered until it rests on what is
        under it, then (``release``) let go."""
        held = self._held("both")
        m, d = self._m, self._d
        obj = self.holding[held[0]].obj_body
        target = self._target(at)
        point, _ = target.resolve(m, d)

        def low():
            return self._bottom(obj)

        sides = [s for s in held if not self._riding[s]]
        starts = {s: self.body.grip(d, s) for s in sides}
        # Toward the target, as far as the hands reach in front of the body: set down on the near
        # part of a surface (a body, an entity), as a worker does, not stretched out to its middle.
        x, y, yaw = self.body.root()
        ahead = np.array([math.cos(yaw), math.sin(yaw)])
        shift = np.array([point[0], point[1]]) - low()[:2]
        for s in sides:
            # how far in front of the body the hand would be: a box held by its ends has its hands
            # to the sides as well, which is no further to reach forward; how far it reaches at the
            # height it sets it down at
            reach = self.body.place_reach(d, s, float(point[2]) + float(starts[s][0][2] - low()[2]))
            out = float((starts[s][0][:2] + shift - [x, y]) @ ahead)
            forward = float(shift @ ahead)
            if out > reach and forward > 1e-9:
                shift *= max(0.0, 1.0 - (out - reach) / forward)
        where = low()[:2] + shift
        # What it is set over is lower than it is carried, or than the place where that is higher
        # (a shelf above a carried load): a lintel or a roof over the way is above it, not a
        # surface to lift it over.
        ceiling = max(float(self._extent(obj)[1][2]), float(point[2])) + CLEAR
        surface, under = self._surface(np.array([where[0], where[1], point[2]]), obj, below=ceiling)
        if (
            target.kind != "world"
            and under >= 0
            and int(m.body_rootid[under]) != int(m.body_rootid[target.body])
        ):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, under) or "the floor"
            self.fail(
                f"out of reach: under the hands' reach is {name}, not {at} -- approach {at} closer "
                f"(a smaller standoff, facing it)"
            )
        origin = d.xpos[obj][:2] + shift
        top = max(
            float(surface[2]),
            self._top(
                np.array([origin[0], origin[1], point[2]]), obj, self._on(target), below=ceiling
            ),
        )
        # What it is set down on is under all of it: the surfaces under its corners where it goes
        # are one, or it would stand on a drawer's wall and its deck, or half off a bench's edge.
        under_corners = [
            self._surface(
                np.array([origin[0] + off[0], origin[1] + off[1], point[2]]), obj, below=ceiling
            )
            for off in self._footprint(obj)
        ]
        # a corner over nothing solid (a floor that collides with nothing) stands on nothing
        heights = [float(p[2]) if b >= 0 else -math.inf for p, b in under_corners]
        if max(heights) - min(heights) > LEVEL:
            hi = max(under_corners, key=lambda pb: pb[0][2])[1]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, hi) if hi > 0 else "the floor"
            self.fail(
                f"out of reach: set down where the hands reach, it would stand on {name} at "
                f"{max(heights):.2f} m and "
                + (f"on {min(heights):.2f} m" if math.isfinite(min(heights)) else "over nothing")
                + f", not on {at} -- approach {at} closer (a smaller standoff, facing it)"
            )
        # Over it, clear of the highest thing it passes over on the way there (a drawer's front, a
        # rim the target's footprint does not cover): straight up first where that is higher than
        # it is carried, then across, so it never cuts down through one.
        here = d.xpos[obj][:2].copy()
        clear = max(top, self._path_top(here, here + shift, obj, point[2], below=ceiling)) + CLEAR
        rise = clear - low()[2]
        if rise > 0.0:
            up = np.array([0.0, 0.0, rise])
            goals = {s: (lambda s=s: (starts[s][0] + up, starts[s][1])) for s in sides}
            yield from self._move_hands(goals, 0.3, at, rot_tol=math.radians(25), pos_tol=0.03)
        over = np.array([shift[0], shift[1], rise])
        # Square to a place that has a frame (a site's, a body's): turned on the way, about its own
        # origin, to the nearest quarter turn of the place's heading -- the hold keeps a load as it
        # was taken, and a box a few degrees off a drawer's nest stands on its stops.
        turn = self._square_to(target, obj)
        spin = Q.axis_angle("z", turn)
        centre = d.xpos[obj].copy()
        goals = {
            s: (
                lambda s=s: (
                    centre + Q.rotate(spin, starts[s][0] - centre) + over,
                    Q.mul(spin, starts[s][1]),
                )
            )
            for s in sides
        }
        yield from self._move_hands(goals, 0.3, at, rot_tol=math.radians(25), pos_tol=0.03)
        # The hold is a spring: what it holds comes over the place a moment after the hands do, and
        # lowered before it has, it would come down on a rim beside the place.
        # Nor do the hands always come all the way to their goals: they are led on by what is
        # still missing, until it is over the place.
        t0 = self._ctx.sim_time
        lead = np.zeros(3)
        while float(np.linalg.norm(low()[:2] - where)) > OVER_AT and self._ctx.sim_time - t0 < 1.5:
            miss = where - low()[:2]
            lead[:2] += min(1.0, self._ctx.dt / 0.1) * miss
            for s in sides:
                p, q = goals[s]()
                self.body.set_hand(s, p + lead, q)
            self._check_holds()
            yield
        missing = float(np.linalg.norm(low()[:2] - where))
        if missing > 2 * OVER_AT:
            self.fail(
                f"out of reach: the hands bring it to {missing:.3f} m of {at} and no nearer -- "
                f"approach {at} closer (a smaller standoff)"
            )
        self.step("lowering", at=at)
        starts = {s: (goals[s]()[0] + lead, goals[s]()[1]) for s in sides}
        lead = np.zeros(3)
        down = 0.0
        start_low = float(low()[2])
        # what it touches that is not the surface: itself, the humanoid, what rides in it
        inside = self._contents(obj)
        skip = {obj} | {
            b for b in range(m.nbody) if self.body.owns(m, b) or int(m.body_rootid[b]) in inside
        }
        while True:
            hit = touching_body(m, d, obj, skip)
            if hit >= 0 and float(low()[2]) <= top + LEVEL:
                break
            if hit >= 0 and float(low()[2]) > start_low - down + CAUGHT:
                # held up above the place by what it touches -- a rim or a wall it came down on,
                # not one it grazes on the way down: it would be let go on it
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, hit) or "the world"
                self.fail(
                    f"caught on {name}: {float(low()[2]) - top:.3f} m above {at} it rests on it "
                    f"-- approach {at} square to it"
                )
            down += speed * self._ctx.dt
            if down > start_low - float(surface[2]) + 0.2:
                self.fail(f"nothing under it at {at}: lowered {down:.2f} m and touched nothing")
            # led on over the place as it goes down, as it was led onto it
            lead[:2] += min(1.0, self._ctx.dt / 0.1) * (where - low()[:2])
            for s in sides:
                self.body.set_hand(s, starts[s][0] - [0.0, 0.0, down] + lead, starts[s][1])
            self._check_holds()
            yield
        yield from self.wait(0.2)
        self.step("set_down", at=at)
        if release:
            for s in held:
                self.holding[s].set(d, False)
                self.holding[s] = None
                self._riding[s] = False
                self.body.set_pose(s, "open")
                self.body.gripper(d, s, False)
            yield from self.wait(0.3)
            for s in held:
                self.body.free_hand(s)
            self.step("released", at=at)

    def _body_call(self, method: str, *args, **kw):
        """A call on the body that fails the move, with the body's reason, where it refuses."""
        try:
            return getattr(self.body, method)(*args, **kw)
        except (NotImplementedError, ValueError) as exc:
            self.fail(str(exc))

    @program
    def play(
        self,
        clip: str,
        joints: list | None = None,
        weight: float = 1.0,
        loop: bool = False,
        fade: float = 0.3,
        speed: float = 1.0,
    ):
        """Play a clip over the body's own motion on some joints (groups and joint names; all by
        default): once, ending when it has faded out (``played``); or with ``loop``, ending once it
        has faded in (``looping``) and playing on until ``stop_clip``. A play cancelled or failed fades its clip out. Hands and
        moves come over it."""
        layer = self._body_call(
            "play",
            clip,
            joints=joints or ["all"],
            weight=weight,
            loop=loop,
            fade=fade,
            speed=speed,
        )
        self.step("playing", clip=clip, layer=layer)
        done = False
        try:
            if loop:
                yield from self.wait(fade)
            else:
                yield from self.until(lambda: self.body.clip_ended(layer))
            done = True
        finally:
            if not done:
                self.body.stop_clip(layer)
        self.step("looping" if loop else "played", clip=clip, layer=layer)

    @program
    def stop_clip(self, clip: str = ""):
        """Fade out the clips playing ``clip`` (its name as played), or every clip, and end once they
        are gone."""
        layers = self._body_call("stop_clip", clip)
        yield from self.until(lambda: all(self.body.clip_ended(i) for i in layers))
        self.step("stopped", clip=clip, layers=layers)

    @program
    def look_at(self, target: str | list = ""):
        """Turn the head toward a target (a site, body or entity by name, or a world point
        ``[x, y, z]``),
        following it as it moves, and end once it is on it; ``""`` turns it back to the body's own
        motion."""
        self._body_call("set_gaze", target or None)
        yield from self.until(self.body.looking, timeout=SETTLE * 2, what="the head turning")
        self.step("looking", target=target)


def _graspable(site: str, pattern: str) -> bool:
    """Whether ``site`` is a grip site of an entity ``pattern`` matches: ``<e>/grip``,
    ``<e>/grip_<x>`` or ``<e>/<part>/grip``."""
    parts = site.split("/")
    if not parts[-1].startswith("grip"):
        return False
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts))]
    return any(fnmatch.fnmatchcase(p, pattern) for p in prefixes)
