# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Which bodies a trial's start state puts inside one another, and how deep.

A world whose reset state interpenetrates -- an arm's ``home`` that puts its tool inside the table,
a prop spawned into another -- does not fail to load. The contact solver resolves the overlap on the
first steps instead, with a force that grows with the depth: normal forces of kilonewtons, bodies
flung, and a run that then fails downstream in a way that looks like a controller or a protocol
fault. Nothing in the run's own record says the cause was the pose it started from.

:func:`interpenetrations` reads that cause off the contacts MuJoCo computed for the start state
(``mj_forward`` after every plugin's ``on_reset``, which is where :meth:`roqsim.engine.Engine.reset`
leaves ``data``), and reports every pair of sides that overlaps by more than its tolerance --
naming both sides, the entity each belongs to, and the depth.

What is not reported, and why:

* **A pair MuJoCo does not act on.** A pair excluded by ``contype``/``conaffinity``, by an
  ``<exclude>``, by parent-child filtering or by two bodies with no degree of freedom between them
  never enters ``data.contact``, or enters it without a constraint row (``efc_address < 0``). Its
  overlap exerts no force, so it is what the author asked for, not a fault.
* **A contact within tolerance** -- a box resting on a table. See :func:`contact_tolerance`.
* **An overlap MuJoCo generates no contact for.** A geom that passes all the way through a thin one
  (a capsule through a slab) can get none; the solver then does not act on it either.

A side is a geom or, for a MuJoCo flex contact, a flex: there ``contact.geom`` is ``-1`` and the
side is named from ``contact.flex`` rather than by indexing the geom arrays with ``-1`` (which would
name the model's last geom).

This module only reads; it never refuses. The engine logs what it finds at reset, and
``roqsim check`` reports it as a warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: The absolute floor of the tolerance, in metres. A start pose is written by hand, snapped from a
#: settled run, or placed by a primitive against a mesh's hull, and each of those is off by a few
#: millimetres without anything being wrong; a pose that is actually wrong buries a part by
#: centimetres. 5 mm separates the two, and at MuJoCo's default contact parameters it is above both
#: the solimp width (1 mm) and the depth at which the contact asks for one g (about 3.5 mm).
DEFAULT_TOLERANCE = 0.005

#: The acceleration scale of :func:`contact_tolerance`'s stiffness term (m/s^2). Standard gravity
#: rather than the world's own, so a zero-g world is judged by the same yardstick as any other.
STANDARD_GRAVITY = 9.80665

#: What to change, stated once for the engine's log line and for ``roqsim check``.
HINT = (
    "move the pose that places them there -- an arm's `home` (spawn_arm) or the model's keyframe, "
    "a robot's or a prop's spawn pose -- until they no longer overlap at reset; if the overlap is "
    "intended, exclude the pair (contype/conaffinity or an <exclude>) so the solver does not "
    "separate it on the first steps"
)

# MuJoCo clamps solimp's dmin/dmax into [mjMINIMP, mjMAXIMP] before using them.
_MIN_IMP = 0.0001
_MAX_IMP = 0.9999


@dataclass(frozen=True)
class Side:
    """One side of a contact: a geom, or a flex."""

    kind: str  # "geom" | "flex"
    id: int
    name: str  # the MJCF name; "" when the element has none
    body: str  # the body it hangs on (for a flex, the body of its first moving vertex)
    entity: str | None  # the registered entity whose subtree holds that body

    def label(self) -> str:
        what = f"{self.kind} '{self.name}'" if self.name else f"unnamed {self.kind} #{self.id}"
        if not self.name and self.body:
            what += f" of body '{self.body}'"
        if self.entity:
            what += f" (entity '{self.entity}')"
        return what


@dataclass(frozen=True)
class Interpenetration:
    """Two sides overlapping beyond tolerance: the deepest of their contacts, and how many."""

    first: Side
    second: Side
    depth: float  # m, -dist of the deepest contact between the two
    tolerance: float  # m, that contact's tolerance
    contacts: int

    def describe(self) -> str:
        count = f", {self.contacts} contacts" if self.contacts > 1 else ""
        return (
            f"{self.first.label()} and {self.second.label()} interpenetrate by "
            f"{self.depth * 1e3:.1f} mm at reset (tolerance {self.tolerance * 1e3:.1f} mm{count})"
        )


def contact_tolerance(model, solref, solimp, floor: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """The depth each contact may have at reset without being reported, from its own parameters.

    The largest of three depths, each one at which a contact is still *holding* rather than
    *ejecting*:

    * *floor* (:data:`DEFAULT_TOLERANCE`) -- the placement slack of a pose written by hand.
    * ``solimp[2]``, the contact's **width**: the depth over which MuJoCo ramps its impedance from
      ``dmin`` to ``dmax``, so a contact inside it is still being eased in by design.
    * ``g / k``: the depth at which the contact's spring alone asks for one standard gravity of
      acceleration, with ``k`` MuJoCo's reference stiffness for the contact's ``solref`` --
      ``1 / (dmax^2 * timeconst^2 * dampratio^2)``, the time constant clamped to two steps as MuJoCo
      clamps it, or ``-solref[0] / dmax^2`` in the direct form. A resting contact carries its weight
      at a fraction of this depth; a contact several times deeper is pushed out at several g on the
      first step. A contact softer than the default is meant to sink further, and this term is what
      lets the tolerance follow it.

    *solref* is ``(n, 2)`` and *solimp* ``(n, 5)``, as ``data.contact`` carries them -- which are the
    global override's values when ``sim.contact_override`` enabled it.
    """
    solref = np.asarray(solref, dtype=float).reshape(-1, 2)
    solimp = np.asarray(solimp, dtype=float).reshape(-1, 5)
    dmax = np.clip(solimp[:, 1], _MIN_IMP, _MAX_IMP)
    width = solimp[:, 2]

    timeconst = solref[:, 0]
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        timeconst = np.maximum(timeconst, 2.0 * model.opt.timestep)
    dampratio = solref[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(
            solref[:, 0] > 0,
            1.0 / (dmax**2 * timeconst**2 * dampratio**2),
            -solref[:, 0] / dmax**2,
        )
        # A contact with no stiffness never pushes back, so no depth of it is violent.
        stiffness_depth = np.where(k > 0, STANDARD_GRAVITY / k, np.inf)
    return np.maximum(np.maximum(floor, width), stiffness_depth)


def interpenetrations(
    model, data, entities=None, floor: float = DEFAULT_TOLERANCE
) -> list[Interpenetration]:
    """Every pair of sides in ``data.contact`` overlapping beyond tolerance, deepest first.

    Reads the contacts as they stand -- call it after ``mj_forward`` on the state to be judged.
    *entities* (a :class:`roqsim.context.EntityRegistry`) attributes each side to the entity whose
    body subtree holds it; without one, sides are named by geom or flex and body only. Contacts
    between the same two sides are folded into one entry carrying the deepest, so a box sunk into a
    table is one finding rather than one per corner.
    """
    n = int(data.ncon)
    if not n:
        return []
    contact = data.contact
    depth = -np.asarray(contact.dist[:n], dtype=float)
    acted_on = np.asarray(contact.efc_address[:n]) >= 0
    tolerance = contact_tolerance(model, contact.solref[:n], contact.solimp[:n], floor)
    deep = np.flatnonzero(acted_on & (depth > tolerance))
    if not deep.size:
        return []

    geom = np.asarray(contact.geom[:n])
    flex = np.asarray(contact.flex[:n])
    owner = _entity_by_body(model, entities)
    sides: dict[tuple[str, int], Side] = {}
    pairs: dict[tuple, list] = {}
    for i in deep:
        a = _side(model, int(geom[i, 0]), int(flex[i, 0]), owner, sides)
        b = _side(model, int(geom[i, 1]), int(flex[i, 1]), owner, sides)
        key = tuple(sorted(((a.kind, a.id), (b.kind, b.id))))
        entry = pairs.get(key)
        if entry is None:
            pairs[key] = [a, b, float(depth[i]), float(tolerance[i]), 1]
            continue
        entry[4] += 1
        if depth[i] > entry[2]:
            entry[2], entry[3] = float(depth[i]), float(tolerance[i])
    found = [Interpenetration(a, b, d, t, c) for a, b, d, t, c in pairs.values()]
    return sorted(found, key=lambda f: -f.depth)


def summary(found: list[Interpenetration], limit: int = 3) -> str:
    """One log line naming the deepest *limit* findings, and how many more there are."""
    shown = "; ".join(f.describe() for f in found[:limit])
    more = f"; and {len(found) - limit} more pair(s)" if len(found) > limit else ""
    return (
        "the start state interpenetrates, and the contact solver will push the bodies apart on "
        f"the first steps: {shown}{more}. Fix: {HINT}"
    )


def _side(model, geom: int, flex: int, owner: dict[int, str], cache: dict) -> Side:
    kind, index = ("geom", geom) if geom >= 0 else ("flex", flex)
    side = cache.get((kind, index))
    if side is not None:
        return side
    if kind == "geom":
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        body = int(model.geom_bodyid[index])
    else:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, index) or ""
        body = _flex_body(model, index)
    side = Side(
        kind=kind,
        id=index,
        name=name,
        body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "",
        entity=_owner_of(model, body, owner),
    )
    cache[(kind, index)] = side
    return side


def _flex_body(model, flex: int) -> int:
    """The body a flex's first moving vertex hangs on (the world body if none moves)."""
    start = int(model.flex_vertadr[flex])
    bodies = np.asarray(model.flex_vertbodyid[start : start + int(model.flex_vertnum[flex])])
    moving = bodies[bodies > 0]
    return int(moving[0]) if moving.size else 0


def _entity_by_body(model, entities) -> dict[int, str]:
    if entities is None:
        return {}
    owner: dict[int, str] = {}
    for entity in entities.all():
        if not entity.body:
            continue
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if body > 0:
            owner[body] = entity.name
    return owner


def _owner_of(model, body: int, owner: dict[int, str]) -> str | None:
    """The entity whose base body is *body* or its nearest ancestor -- the most specific owner."""
    while body > 0:
        if body in owner:
            return owner[body]
        body = int(model.body_parentid[body])
    return None
