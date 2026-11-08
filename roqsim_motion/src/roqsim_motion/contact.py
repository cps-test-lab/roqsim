# SPDX-License-Identifier: Apache-2.0
"""Contact where it matters: holds set up when the world is built, switched while it runs.

**A hold** is a weld between a hand's grip site and an object's grip site, one per (hand, grip site),
added before compile (:func:`add_holds`, from a ``build_late``) and inactive. Holding sets it active
(``data.eq_active``), letting go clears it: nothing is recompiled and no model field is written. The
weld pulls the two frames into one, so a drawer is pulled and a box carried by the physics. Its
force is bounded by whoever switched it on: :meth:`Hold.force` and :meth:`Hold.separation` read the
weld each step, and a move lets go past its ``max_force`` or ``slip``.

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


def add_holds(spec: mujoco.MjSpec, hand_sites: list[str], object_sites: list[str], solref=HOLD_SOLREF) -> list[str]:
    """One inactive weld per (hand site, object site), named by :func:`hold_name`; the names."""
    names = []
    for hand in hand_sites:
        for obj in object_sites:
            eq = spec.add_equality()
            eq.name = hold_name(hand, obj)
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE
            eq.name1 = obj  # the object first: the weld pulls it onto the hand
            eq.name2 = hand
            eq.active = False
            eq.solref = list(solref)
            eq.solimp = list(HOLD_SOLIMP)
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
    """One weld: whether it holds, how hard it pulls and how far apart its frames are."""

    eq: int
    obj_site: int
    hand_site: int
    obj_body: int

    @classmethod
    def find(cls, model, hand_site: str, object_site: str) -> Hold | None:
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, hold_name(hand_site, object_site))
        if eq < 0:
            return None
        o = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, object_site)
        h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, hand_site)
        return cls(eq, o, h, int(model.site_bodyid[o]))

    def active(self, data) -> bool:
        return bool(data.eq_active[self.eq])

    def set(self, data, on: bool) -> None:
        data.eq_active[self.eq] = 1 if on else 0

    def separation(self, data) -> float:
        return float(np.linalg.norm(data.site_xpos[self.obj_site] - data.site_xpos[self.hand_site]))

    def force(self, data) -> float:
        """The weld's translational constraint force now (N): the norm of its first three rows."""
        rows = np.nonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == self.eq))[0]
        if len(rows) == 0:
            return 0.0
        return float(np.linalg.norm(data.efc_force[rows[:3]]))


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
