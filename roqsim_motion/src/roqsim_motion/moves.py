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
        - navigator: {output: walker, avoidance: {stop: false}}
        - humanoid:
            graspable: [tote_*, cabinet]  # entities whose grip sites the hands may hold (fnmatch);
                                          # a hold is a weld per hand and grip site, set at build
            hold: {max_force: 300.0, slip: 0.06}  # past either (low-passed), a hold lets go and fails

Refused when the world loads: a ``humanoid`` under a walker with ``hands: false`` (its moves need
hands: set ``hands: true`` on the walker), and one beside a navigator with a patrol (the
navigator would walk the humanoid back onto its route after a move; send it with ``navigate``
between moves instead). Refused at build: a ``graspable`` pattern no grip site matches.
"""

from __future__ import annotations

import fnmatch
import math

import mujoco
import numpy as np

from roqsim.machine import MachineError, MachineHandle, MachinePlugin, program
from roqsim.schema import Field

from . import quat as Q
from .body import SIDES, HumanoidBody
from .contact import HOLD_SOLREF, Hold, add_holds, check_solref, contact_force, touching
from .targets import TargetError, make_target

HANDS = {"left": ("l",), "right": ("r",), "both": ("l", "r")}
#: How near a hand is to its goal when it has arrived (m, rad), and how long it may take after its
#: goal has (s).
ARRIVED = 0.01
ARRIVED_ROT = math.radians(6.0)
SETTLE = 2.0
#: How far in front of what it presses or grips a hand comes first (m).
PRE = 0.08
#: The speed a press goes in at (m/s), and how far past the button's stroke it may go looking for it.
PRESS_SPEED = 0.05
PRESS_SEARCH = 0.04
#: How far a hand reaches from the body's root before a move refuses it without trying (m).
FAR = 1.4
#: How far in front of the body a hand sets something down, at most (m).
PLACE_REACH = 0.55
#: How fast a body carrying something turns on the spot (rad/s).
CARRY_TURN = 1.0
#: The time constant (s) a hold's force is judged over.
FORCE_TAU = 0.15

HOLD = {
    "max_force": Field(float, default=300.0, minimum=1.0, unit="N", doc="a hold lets go past this"),
    "slip": Field(float, default=0.06, minimum=0.001, unit="m", doc="a hold lets go when its frames part this far"),
    "solref": Field(list, default=list(HOLD_SOLREF), length=2, doc="the hold's (timeconst, dampratio)"),
}


def _side_word(side: str) -> str:
    return {"l": "left", "r": "right"}[side]


class HumanoidHandle(MachineHandle):
    """The machine handle with a method per move; each returns the start's sequence number."""

    def approach(self, target: str, **kw) -> int:
        return self.start("approach", target=target, **kw)

    def reach(self, hand: str, target: str, **kw) -> int:
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

    def carry(self, to: str, **kw) -> int:
        return self.start("carry", to=to, **kw)

    def place(self, at: str, **kw) -> int:
        return self.start("place", at=at, **kw)


class HumanoidPlugin(MachinePlugin):
    """A humanoid's moves (see the module docstring)."""

    registers_entity = False
    requires_owner = True
    ENTITY_KIND = "humanoid"
    CONFIG_SCHEMA = {
        "graspable": Field(list, default=[], doc="entities whose grip sites the hands may hold (fnmatch)"),
        "hold": Field(dict, schema=HOLD, doc="how hard a hold may pull and how far it may slip"),
        "press_force": Field(float, default=40.0, minimum=0.1, unit="N", doc="a press's force at most"),
        "on_error": Field(str, default="report", choices=["raise", "report"], doc="a defect in a move"),
        "home": Field(dict, doc="(a machine's; a humanoid drives no actuator)"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        config = {"on_error": "report", **(config or {})}
        super().__init__(config, name=name, entity=entity, label=label)
        self.body: HumanoidBody | None = None
        self.holding: dict[str, Hold | None] = {s: None for s in SIDES}
        self._force = {s: 0.0 for s in SIDES}
        self._apart = {s: 0.0 for s in SIDES}
        self._riding = {s: False for s in SIDES}  # a second hand riding on what the first holds
        self._holds: dict = {}

    # -- load and build -----------------------------------------------------------------------
    @classmethod
    def expand(cls, spec, world, base_dir):
        owner = next((w for w in world if w.address == spec.entity), None)
        if owner is not None and owner.ref == "walker" and not (owner.config or {}).get("hands", False):
            raise ValueError(
                f"humanoid {spec.address}: its moves need hands -- set 'hands: true' on walker "
                f"{owner.address}"
            )
        for sibling in (owner.children if owner is not None else []):
            if sibling.ref == "navigator" and ((sibling.config or {}).get("goals") or (sibling.config or {}).get("loop")):
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
            found = [n for n in names if _graspable(n, pattern) and not n.startswith(f"{self.entity}/")]
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
        self.handle = HumanoidHandle(self)
        ctx.blackboard.set(f"machine:{self.machine_entity}", self.handle)
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

    def _site_id(self, name: str) -> int:
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
                self.fail(f"{target} has no grip site for the {_side_word(side)} hand ({target}/grip_{side} or {target}/grip)")
        if len(sides) > 1 and len(set(out.values())) < len(out):
            self.fail(f"{target} has one grip site; one hand holds it")
        return out

    def _far(self, pos) -> None:
        x, y, _ = self.body.root()
        d = math.hypot(pos[0] - x, pos[1] - y)
        if d > FAR:
            self.fail(f"out of reach: the target is {d:.2f} m from where it stands -- approach first")

    def _move_hands(self, goals, speed: float, what: str, rot_tol: float = ARRIVED_ROT):
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
                errs[s] = (float(np.linalg.norm(p - gp)), 0.0 if gq is None else Q.angle_between(q, gq))
            if all(e[0] < ARRIVED and e[1] < rot_tol and self.body.settled(s) for s, e in errs.items()):
                return
            if self._ctx.sim_time - t1 > SETTLE:
                side, (e, a) = max(errs.items(), key=lambda kv: kv[1][0])
                self.fail(
                    f"out of reach: the {_side_word(side)} hand stays {e:.3f} m and "
                    f"{math.degrees(a):.0f} deg from {what} -- approach first"
                )
            yield

    def _site_goal(self, site: int, back: float = 0.0):
        """A goal on a site's frame, ``back`` m out along its -x (in front of what it is on)."""

        def goal():
            p = self._d.site_xpos[site]
            r = self._d.site_xmat[site].reshape(3, 3)
            return p - back * r[:, 0], Q.from_matrix(r)

        return goal

    def _ride(self) -> None:
        """Keep a hand that rides on what the other holds on its own grip site."""
        for side, h in self.holding.items():
            if h is not None and self._riding[side]:
                self.body.set_hand(side, *self._site_goal(h.obj_site)())

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
            self._force[side] += a * (h.force(self._d) - self._force[side])
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
            self.fail(f"not holding with the {hand} hand" if hand != "both" else "not holding anything")
        return sides

    def _stand_for(self, target: str, standoff: float, facing):
        """Where to stand, and which way to face, to work on ``target``."""
        d = self._d
        gl, gr = self._site_id(f"{target}/grip_l"), self._site_id(f"{target}/grip_r")
        site = self._site_id(target)
        if gl >= 0 and gr >= 0:
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
            direction = np.array([x[0], x[1]]) if name.startswith("grip") else -np.array([x[0], x[1]])
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
        return stand, math.atan2(direction[1], direction[0])

    def _walk(self, stand, yaw, what: str, while_walking=None):
        nav = self._ctx.blackboard.get(f"nav:{self.entity}:handle")
        if nav is None:
            self.fail(f"{self.entity} has no navigator to walk to {what} with")
        self.body.face(None)  # it faces where it walks until it arrives
        seq = nav.send_goals([(float(stand[0]), float(stand[1]))])
        self.step("walking", to=[float(stand[0]), float(stand[1])])
        while True:
            if while_walking is not None:
                while_walking()
            status = nav.status()
            if status[0] == seq and status[1]:
                if status[4]:
                    self.fail(f"no route to {what}: {status[4]}")
                break
            if status[0] > seq:
                self.fail(f"the route to {what} was replaced by another")
            yield
        if while_walking is not None:
            # carrying: turn at a carrying pace
            yield from self._turn(yaw, while_walking)
        self.body.face(yaw)
        t0 = self._ctx.sim_time
        while abs(math.atan2(math.sin(self.body.root()[2] - yaw), math.cos(self.body.root()[2] - yaw))) > math.radians(3):
            if while_walking is not None:
                while_walking()
            if self._ctx.sim_time - t0 > 5.0:
                self.fail(f"cannot face {what}: it turned to {math.degrees(self.body.root()[2]):.0f} deg of {math.degrees(yaw):.0f}")
            yield
        self.step("arrived", at=[float(stand[0]), float(stand[1])], yaw=float(yaw))

    # -- the moves ----------------------------------------------------------------------------
    @program
    def approach(self, target: str, standoff: float = 0.6, facing: float | None = None):
        """Walk to a stand-off in front of a target and face it: a grip site's or a face's front, a
        tote's long side (its left grip on the left), or the target as seen from here."""
        stand, yaw = self._stand_for(target, standoff, facing)
        if any(h is not None for h in self.holding.values()):
            # holding something: it goes along, as in a carry
            carried = self._carried()
            x, y, _ = self.body.root()
            if math.hypot(stand[0] - x, stand[1] - y) > 0.05:
                yield from self._turn(math.atan2(stand[1] - y, stand[0] - x), lambda: self._hold_carried(carried))
            yield from self._walk(stand, yaw, target, while_walking=lambda: self._hold_carried(carried))
        else:
            yield from self._walk(stand, yaw, target)

    @program
    def reach(self, hand: str, target: str, offset: list | None = None, orient: bool = True, pose: str = "open",
              speed: float = 0.8):
        """Put a hand, or both, on a target (a grip site's frame, or a pose with an offset in its
        frame); the hand stays there until another move moves it or ``release`` frees it."""
        sides = self._sides(hand)
        if self._site_id(target) < 0 and offset is None and all(
            self._site_id(f"{target}/grip_{s}") >= 0 or self._site_id(f"{target}/grip") >= 0 for s in sides
        ):
            sites = self._grip_sites(target, sides)
            goals = {s: self._site_goal(self._site_id(sites[s])) for s in sides}
        else:
            t = self._target(target, offset)
            goals = {s: (lambda t=t: (lambda pq: (pq[0], pq[1] if orient else None))(t.resolve(self._m, self._d)))
                     for s in sides}
        for s in sides:
            self._far(goals[s]()[0])
            self.body.set_pose(s, pose)
        yield from self._move_hands(goals, speed, target)
        self.step("reached", hand=hand, target=target)

    @program
    def press(self, hand: str, target: str, max_force: float | None = None, travel: float | None = None,
              pose: str = "point"):
        """Press a button (or anything on a joint) with a fingertip until it has travelled its press,
        then let go: in from the front of its face along the face's -x."""
        (side,) = self._sides(hand) if hand != "both" else self.fail("a press is one hand's")
        site = self._site_id(target)
        if site < 0:
            self.fail(f"press {target}: a press needs a site on the button's face (e.g. '<button>/button')")
        m, d = self._m, self._d
        body_id = int(m.site_bodyid[site])
        jnt = int(m.body_jntadr[body_id])
        if jnt < 0:
            self.fail(f"press {target}: its body has no joint, so nothing can be pressed")
        qadr = int(m.jnt_qposadr[jnt])
        stroke = float(m.jnt_range[jnt][1] - m.jnt_range[jnt][0]) if m.jnt_limited[jnt] else 0.01
        need = travel if travel is not None else 0.6 * stroke
        limit = max_force if max_force is not None else self.settings.press_force
        q0 = float(d.qpos[qadr])
        self._far(d.site_xpos[site])
        self.body.set_pose(side, pose)
        normal = d.site_xmat[site].reshape(3, 3)[:, 0].copy()  # out of the face

        def tip_goal(depth):
            def goal():
                aim = d.site_xpos[site] + normal * (0.0 - depth)
                offset = self.body.grip(d, side)[0] - self.body.tip(d, side)
                return aim + offset, None
            return goal

        yield from self._move_hands({side: tip_goal(-PRE)}, 0.8, target)
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

    @program
    def grasp(self, hand: str, target: str, grip: str = "power"):
        """Take a target's grip site(s) in a hand or both: in from the front of the grip, close the
        fingers to its shape, and hold it -- a weld, so the physics carries what is held."""
        sides = self._sides(hand)
        if self._site_id(target) < 0 and not any(
            self._site_id(f"{target}/{n}") >= 0 for n in ("grip", "grip_l", "grip_r")
        ) and self._ctx.entities.get(target) is None:
            self.fail(f"nothing to grasp: unknown target {target!r}")
        sites = self._grip_sites(target, sides)
        holds = {}
        for s in sides:
            h = self._holds.get((s, sites[s]))
            if h is None:
                self.fail(f"nothing to grasp: {target} is not graspable (graspable: {list(self.settings.graspable)})")
            if self.holding[s] is not None:
                self.fail(f"the {_side_word(s)} hand is holding already; release it first")
            holds[s] = h
        ids = {s: self._site_id(sites[s]) for s in sides}
        for s in sides:
            self._far(self._d.site_xpos[ids[s]])
            self.body.set_pose(s, "open")
        yield from self._move_hands({s: self._site_goal(ids[s], back=PRE) for s in sides}, 0.8, target,
                                    rot_tol=math.radians(25))
        # judged by position: the hold makes the frames one, turning a free object to the hand
        yield from self._move_hands({s: self._site_goal(ids[s]) for s in sides}, 0.3, target, rot_tol=math.radians(25))
        self.step("closing", hand=hand, target=target)
        m, d = self._m, self._d
        for s in sides:
            obj = holds[s].obj_body
            self.body.set_pose(s, grip, touching=self._touching(s, obj, ids[s]))
            self.body.gripper(d, s, True)
        t0 = self._ctx.sim_time
        while not all(self.body.hand_done(s) for s in sides) and self._ctx.sim_time - t0 < 1.0:
            for s in sides:
                self.body.set_hand(s, *self._site_goal(ids[s])())
            yield
        for s in sides:
            apart = holds[s].separation(d)
            if apart > 0.02:
                self.fail(f"hold not made: the {_side_word(s)} hand is {apart:.3f} m from {sites[s]}")
        for i, s in enumerate(sides):
            # Two hands on one rigid object: one weld carries it, the other hand rides on its own grip
            # site. Two welds would squeeze it between them by every millimetre the hands are off.
            holds[s].set(d, i == 0)
            self.holding[s] = holds[s]
            self._riding[s] = i > 0
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
        h0 = next(self.holding[s] for s in sides if not self._riding[s]) if any(
            not self._riding[s] for s in sides) else self.holding[sides[0]]
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
        self.step("moving", hand="both" if len(sides) > 1 else _side_word(sides[0]), distance=distance)
        duration = abs(distance) / max(speed, 1e-3)
        t0 = self._ctx.sim_time
        while True:
            u = min((self._ctx.sim_time - t0) / duration, 1.0)
            u = u * u * (3.0 - 2.0 * u)
            for s in sides:
                self.body.set_hand(s, starts[s][0] + u * distance * axis, starts[s][1])
            self._check_holds()
            if u >= 1.0:
                break
            yield
        yield from self.wait(0.3)
        self._check_holds()
        moved = float(np.dot(d.site_xpos[h0.obj_site] - obj0, axis))
        if moved < 0.8 * distance:
            self.fail(f"stuck: {what} moved {moved:.3f} of {distance:.3f} m")
        return moved

    @program
    def pull(self, hand: str, distance: float, speed: float = 0.2, axis: list | None = None):
        """Pull what a hand holds toward the humanoid by ``distance`` m: along its slide joint for a
        drawer, else straight out of its grip."""
        sides = self._held(hand)
        moved = yield from self._along(sides, distance, axis, speed, 1.0, "what it pulls")
        self.step("pulled", distance=moved)

    @program
    def push(self, hand: str, distance: float, speed: float = 0.2, axis: list | None = None):
        """Push what a hand holds away from the humanoid by ``distance`` m (a drawer shut)."""
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
    def carry(self, to: str, standoff: float = 0.6, height: float | None = None):
        """Carry what the hands hold to a target: lift it to a carrying height in front of the body
        (``height``: the hands' height, m; default a hand's breadth under the pelvis, or high enough
        for what is held to clear the surface the target is on by 5 cm), turn toward
        the way it leaves, walk to a stand-off in front of the target holding it there, and face the
        target."""
        sides = self._held("both")
        d = self._d
        m = self._m
        obj = self.holding[sides[0]].obj_body
        if height is None:
            # a hand's breadth under the pelvis, or above the surface it goes to, clear by 5 cm
            height = self.body.carry_height()
            lead = next(s for s in sides if not self._riding[s])
            point, _ = self._target(to).resolve(m, d)
            surface, under = self._surface(point, obj)
            if under > 0:
                bottom = self._site_id(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, obj)}/bottom")
                low = d.site_xpos[bottom][2] if bottom >= 0 else d.xpos[obj][2]
                below = self.body.grip(d, lead)[0][2] - low
                height = max(height, float(surface[2]) + below + 0.05)
        x, y, yaw = self.body.root()
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        rel = {}
        for s in [s for s in sides if not self._riding[s]]:
            p, q = self.body.grip(d, s)
            # in front of the body, at the carrying height, the hands as far apart as they are
            ahead = float(np.dot(p - [x, y, 0.0], heading))
            pull_in = max(0.0, ahead - 0.42)
            rel[s] = (lambda p=p, q=q, pull_in=pull_in: (
                np.array([p[0], p[1], height]) - pull_in * heading, q))
        # what is held turns with the hands, so the lift is judged by where they are
        yield from self._move_hands(rel, 0.3, "the carrying height", rot_tol=math.radians(25))
        self._check_holds()
        self.step("lifted", height=height)
        carried = self._carried()
        stand, yaw = self._stand_for(to, standoff, None)
        # Turn toward the way it leaves at a carrying pace first: a body turning on the spot at its
        # walking rate swings what it holds faster than a hold follows.
        x, y, heading = self.body.root()
        leave = math.atan2(stand[1] - y, stand[0] - x)
        yield from self._turn(leave, lambda: self._hold_carried(carried))
        yield from self._walk(stand, yaw, to, while_walking=lambda: self._hold_carried(carried))

    def _turn(self, yaw: float, each=None):
        """Turn the standing body to ``yaw`` at :data:`CARRY_TURN`, calling ``each`` every step."""
        while True:
            heading = self.body.root()[2]
            err = math.atan2(math.sin(yaw - heading), math.cos(yaw - heading))
            if abs(err) < math.radians(3):
                return
            step = math.copysign(min(abs(err), CARRY_TURN * self._ctx.dt), err)
            self.body.face(heading + step)
            if each is not None:
                each()
            yield

    def _surface(self, point, held: int):
        """The top of whatever is under ``point``, and its body: a ray cast down from 1.5 m above
        it, past what is held and the humanoid's own bodies."""
        m, d = self._m, self._d
        start = np.array([point[0], point[1], point[2] + 1.5])
        geomid = np.zeros(1, np.int32)
        while True:
            dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), None, 1, held, geomid)
            if dist < 0:
                return np.asarray(point, float), -1
            hit = start + [0.0, 0.0, -dist]
            body = int(m.geom_bodyid[geomid[0]])
            if not self.body.owns(m, body):
                return np.array([point[0], point[1], hit[2]]), body
            start = hit - [0.0, 0.0, 1e-4]

    @program
    def place(self, at: str, release: bool = True, speed: float = 0.15):
        """Set what the hands hold down on a target: over it, then lowered until it rests on what is
        under it, then (``release``) let go."""
        held = self._held("both")
        m, d = self._m, self._d
        obj = self.holding[held[0]].obj_body
        target = self._target(at)
        point, _ = target.resolve(m, d)
        bottom = self._site_id(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, obj)}/bottom")
        low = (lambda: d.site_xpos[bottom].copy()) if bottom >= 0 else (lambda: d.xpos[obj].copy())
        sides = [s for s in held if not self._riding[s]]
        starts = {s: self.body.grip(d, s) for s in sides}
        # Toward the target, as far as the hands reach in front of the body: set down on the near
        # part of a surface, as a worker does, not stretched out to its middle.
        x, y, _ = self.body.root()
        shift = np.array([point[0], point[1]]) - low()[:2]
        for s in sides:
            g = starts[s][0][:2] + shift
            out = float(np.linalg.norm(g - [x, y]))
            if out > PLACE_REACH:
                shift *= max(0.0, 1.0 - (out - PLACE_REACH) / max(float(np.linalg.norm(shift)), 1e-9))
        where = low()[:2] + shift
        surface, under = self._surface(np.array([where[0], where[1], point[2]]), obj)
        if target.kind != "world" and under >= 0 and int(m.body_rootid[under]) != int(m.body_rootid[target.body]):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, under) or "the floor"
            self.fail(
                f"out of reach: under the hands' reach is {name}, not {at} -- approach {at} closer "
                f"(a smaller standoff, facing it)"
            )
        over = np.array([shift[0], shift[1], surface[2] + 0.05 - low()[2]])
        goals = {s: (lambda s=s: (starts[s][0] + over, starts[s][1])) for s in sides}
        yield from self._move_hands(goals, 0.3, at, rot_tol=math.radians(25))
        self.step("lowering", at=at)
        starts = {s: self.body.grip(d, s) for s in sides}
        down = 0.0

        def ignore():
            return {obj} | {b for b in range(m.nbody) if self.body.owns(m, b)}

        skip = ignore()
        while not touching(m, d, obj, skip):
            down += speed * self._ctx.dt
            if down > 0.3:
                self.fail(f"nothing under it at {at}: lowered 0.3 m and touched nothing")
            for s in sides:
                self.body.set_hand(s, starts[s][0] - [0.0, 0.0, down], starts[s][1])
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


def _graspable(site: str, pattern: str) -> bool:
    """Whether ``site`` is a grip site of an entity ``pattern`` matches: ``<e>/grip``,
    ``<e>/grip_<x>`` or ``<e>/<part>/grip``."""
    parts = site.split("/")
    if not parts[-1].startswith("grip"):
        return False
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts))]
    return any(fnmatch.fnmatchcase(p, pattern) for p in prefixes)
