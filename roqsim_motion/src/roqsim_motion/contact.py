# SPDX-License-Identifier: Apache-2.0
"""Contact where it matters: holds set up when the world is built, switched while it runs.

**A hold** is a weld between a hand's body and the body of an object's grip site, one per (hand,
grip site), added before compile (:func:`add_holds`, from a ``build_late``) and inactive. Holding
switches it on (``data.eq_active``) with the pose the object has in the hand at that moment -- the
weld's relative pose, written into the weld's own row of ``model.eq_data`` by :meth:`Hold.set`, the
one model field a hold writes, owned by the plugin that built the row -- and letting go switches it
off. Nothing is recompiled. The object stays in the hand as it was taken, so a hand a few degrees
off a handle does not turn the object by them, and a drawer is pulled and a box carried by the
physics. Its force is bounded by whoever switched it on: :meth:`Hold.force` and
:meth:`Hold.separation` read the weld each step, and a move lets go past its ``max_force`` or
``slip``.

**Pressing** is the body's: a walker's hands press through proxies it poses while a move needs them
and parks under the floor otherwise (``roqsim_walker.handling``); a robot's through its own links.
:func:`contact_force` reads how hard a body is pushed, whoever pushes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.solref import solref_floor

#: A hold's softness: a weld's time constant (s) and damping ratio, and its impedance. Stiff enough to
#: carry a box within millimetres, soft enough that its force is a spring's, bounded by the move.
HOLD_SOLREF = (0.02, 1.0)
HOLD_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def hold_name(hand_site: str, object_site: str) -> str:
    return f"hold:{hand_site}->{object_site}"


def spec_site(spec: mujoco.MjSpec, name: str):
    """The site ``name`` of ``spec``. ``MjSpec.site`` looks names up in a map the last compile
    built, so it misses a site added after a compile made while the world was built (a plugin that
    compiles to measure the world, as the ceiling's removal does); this walks the sites instead."""
    site = spec.site(name)
    if site is None:
        site = next((s for s in spec.sites if s.name == name), None)
    if site is None:
        raise KeyError(f"no site {name!r}")
    return site


def _jointed(spec: mujoco.MjSpec, site: str) -> bool:
    """Whether the body ``site`` is on moves on a joint other than a free one (a drawer's slide)."""
    body = spec_site(spec, site).parent
    joints = list(body.joints)
    return bool(joints) and all(j.type != mujoco.mjtJoint.mjJNT_FREE for j in joints)


def add_holds(
    spec: mujoco.MjSpec, hand_sites: list[str], object_sites: list[str], solref=HOLD_SOLREF
) -> list[str]:
    """One inactive weld per (hand site, object site) between their bodies, named by
    :func:`hold_name`; the names.

    A free object (a tote) is held whole: it turns with the hand. An object on a joint (a drawer on
    its slide) is held by its position only (the weld's torque scale 0): it cannot turn, and a hand a
    few degrees off its handle would otherwise pull on it all the time it holds."""
    names = []
    for hand in hand_sites:
        for obj in object_sites:
            eq = spec.add_equality()
            eq.name = hold_name(hand, obj)
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = spec_site(
                spec, obj
            ).parent.name  # the object first: the weld keeps it on the hand
            eq.name2 = spec_site(spec, hand).parent.name
            eq.active = False
            eq.solref = list(solref)
            eq.solimp = list(HOLD_SOLIMP)
            if _jointed(spec, obj):
                data = list(eq.data)
                data[10] = 0.0  # torquescale: position only
                eq.data = data
            names.append(eq.name)
    return names


def check_solref(option, solref=HOLD_SOLREF) -> str:
    """Why a hold's ``solref`` is too stiff for the model's step, or ``""``: judged where every
    ``solref`` is (:func:`roqsim.solref.solref_floor`; ``option`` is ``spec.option`` or
    ``model.opt``)."""
    floor = solref_floor(option, solref, HOLD_SOLIMP)
    if floor is not None and solref[0] < floor:
        return (
            f"a hold's solref time constant {solref[0]} s is under {floor:g} s, the floor for a "
            f"{float(option.timestep)} s step (roqsim.solref.solref_floor)"
        )
    return ""


@dataclass
class Hold:
    """One weld: whether it holds, how hard it pulls and how far it has slipped."""

    eq: int
    obj_site: int
    hand_site: int
    obj_body: int
    hand_body: int
    whole: bool = True  # holds the object's rotation too (a free object), or its position only
    model: object = None

    @classmethod
    def find(cls, model, hand_site: str, object_site: str) -> Hold | None:
        eq = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, hold_name(hand_site, object_site)
        )
        if eq < 0:
            return None
        o = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, object_site)
        h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, hand_site)
        return cls(
            eq,
            o,
            h,
            int(model.site_bodyid[o]),
            int(model.site_bodyid[h]),
            bool(model.eq_data[eq][10] > 0.0),
            model,
        )

    def active(self, data) -> bool:
        return bool(data.eq_active[self.eq])

    def set(self, data, on: bool, whole: bool | None = None) -> None:
        """Hold the object where it is in the hand now (``on``), or let it go. ``whole=False``
        holds it by its position only (a second hand on an object the first holds whole); by
        default as the hold was built."""
        if on:
            b1, b2 = self.obj_body, self.hand_body
            qi = np.zeros(4)
            mujoco.mju_negQuat(qi, data.xquat[b1])
            pos = np.zeros(3)
            mujoco.mju_rotVecQuat(pos, data.xpos[b2] - data.xpos[b1], qi)
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, qi, data.xquat[b2])
            row = self.model.eq_data[self.eq]
            row[0:3] = 0.0  # the anchor: the hand body's origin
            row[3:6] = pos  # the hand's pose in the object's frame, as it is
            row[6:10] = quat
            row[10] = 1.0 if (self.whole if whole is None else whole) else 0.0
        data.eq_active[self.eq] = 1 if on else 0

    def distance(self, data) -> float:
        """How far the hand's grip site is from the object's (m)."""
        return float(np.linalg.norm(data.site_xpos[self.obj_site] - data.site_xpos[self.hand_site]))

    def _rows(self, data):
        return np.nonzero(
            (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == self.eq)
        )[0]

    def separation(self, data) -> float:
        """How far the held object has slipped from where it was taken (m): the weld's position
        error while it holds, else :meth:`distance`."""
        rows = self._rows(data)
        if not self.active(data) or len(rows) == 0:
            return self.distance(data)
        return float(np.linalg.norm(data.efc_pos[rows[:3]]))

    def force(self, data, axis=None) -> float:
        """The weld's translational constraint force now (N): the norm of its first three rows, or
        with ``axis`` (world, unit) its component along it -- what pulls an object on a slide along
        the slide, the rest being the joint's reaction to a hand a little off its line."""
        rows = self._rows(data)
        if len(rows) == 0:
            return 0.0
        f = data.efc_force[rows[:3]]
        if axis is None:
            return float(np.linalg.norm(f))
        return float(abs(np.dot(f, axis)))

    def slide_axis(self, model, data):
        """The world axis of the slide the held object is on, or ``None`` (a free object)."""
        jnt = int(model.body_jntadr[self.obj_body])
        if jnt < 0 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_SLIDE:
            return None
        return data.xaxis[jnt].copy()


def contact_force(model, data, body: int, exclude: set | None = None) -> float:
    """The summed normal force (N) of the contacts on ``body``, those with ``exclude``'s bodies left
    out."""
    total = 0.0
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if body not in (b1, b2):
            continue
        other = b2 if b1 == body else b1
        if exclude and other in exclude:
            continue
        mujoco.mj_contactForce(model, data, i, f)
        total += abs(float(f[0]))
    return total


def touching(model, data, body: int, ignore: set) -> bool:
    """Whether ``body`` touches any body not in ``ignore``."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if body in (b1, b2) and (b2 if b1 == body else b1) not in ignore:
            return True
    return False
