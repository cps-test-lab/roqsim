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

"""Which contacts belong to a watched entity -- the rule its observables share.

An external collision of an entity is a contact with **exactly one** side in that entity's
kinematic subtree and neither side ignored. Several plugins report different things about that one
set of contacts -- whether it happened (:mod:`roqsim.plugins.contact_monitor`), how hard
(:mod:`roqsim.plugins.contact_impulse`) -- and two observables over one geometry must never be able
to disagree about *which* contacts they are describing. So the rule is resolved once here and each
plugin asks this object, rather than each carrying its own copy of a predicate that is only equal
by inspection.

The subtree walk itself is :func:`roqsim.presence.entity_geom_ids`, which is what decides the
extent of an entity everywhere else in the substrate; what this module adds is the ignore list, the
per-step filter, and failing loudly where a mistyped body would otherwise leave a meter watching
nothing and reporting a clean run forever.

**A contact side is a geom or a flex.** A contact involving a MuJoCo flex carries ``geom = -1`` on
the flex's side and the flex's id in ``contact.flex`` (the vertex in ``contact.vert``, or the
element in ``contact.elem``). Indexing a per-geom mask with that ``-1`` reads the model's *last*
geom, so a flex touching anything would be attributed to whatever geom happened to be compiled
last. The scope therefore carries a mask per flex beside the mask per geom, and reads each side
from the one that side is: its geom where ``geom >= 0``, else its flex. The flexes an entity owns
are :func:`roqsim.flex.entity_flex_ids` -- those whose DOF bodies all lie in its subtree -- so an
entity that is only a flex is watchable, and ``ignore`` may name a flex as well as a geom.
:func:`side_name` names a side for a report, a flex side as ``flex:<name>[v<i>]``.

What it deliberately does not decide: a force threshold. ``contact_monitor`` applies its own
``min_force`` to the contacts this returns, because a verdict must reject numerical grazing and an
integral must not.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from .flex import entity_flex_ids
from .presence import entity_geom_ids

_log = logging.getLogger(__name__)

_NO_CONTACTS = np.zeros(0, dtype=np.int64)


def side_mask(
    geom: np.ndarray, flex: np.ndarray, geom_mask: np.ndarray, flex_mask: np.ndarray
) -> np.ndarray:
    """Per contact side, the value of the mask for what that side IS: its geom, else its flex.

    *geom* and *flex* are a step's ``contact.geom`` / ``contact.flex`` (``(n, 2)``, or any equal
    shape). A side with ``geom >= 0`` is that geom; a side with ``geom == -1`` is the flex in the
    same slot. Two boolean-indexed gathers rather than one lookup, so a ``-1`` never reaches a mask
    as an index -- numpy would read it as the last entry. A side that names neither a geom nor a
    flex is not a contact MuJoCo produces, and raises rather than being read as anything.
    """
    if not flex_mask.size:
        # A model without a flex, where every side is a geom: one lookup is the whole filter.
        return geom_mask[geom]
    is_geom = geom >= 0
    out = np.empty(geom.shape, dtype=bool)
    out[is_geom] = geom_mask[geom[is_geom]]
    flexes = flex[~is_geom]
    if flexes.size and int(flexes.min()) < 0:
        raise RuntimeError("a contact side names neither a geom nor a flex")
    out[~is_geom] = flex_mask[flexes]
    return out


def side_name(model, geom: int, flex: int, vert: int = -1, elem: int = -1) -> str:
    """One side of a contact, named for a report.

    A geom by its name (``geom<id>`` when it has none); a flex as ``flex:<name>``, with the vertex
    (``[v<i>]``) or element (``[e<i>]``) that touched, indices local to the flex. An unnamed flex
    is ``flex:#<id>``. Pass a side's ``contact.geom``, ``.flex``, ``.vert`` and ``.elem`` entries.
    """
    geom, flex, vert, elem = int(geom), int(flex), int(vert), int(elem)
    if geom >= 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"
    if flex < 0:
        raise ValueError("a contact side names neither a geom nor a flex")
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex) or f"#{flex}"
    where = f"[v{vert}]" if vert >= 0 else f"[e{elem}]" if elem >= 0 else ""
    return f"flex:{name}{where}"


def contact_side_names(model, contact) -> tuple[str, str]:
    """Both sides of one ``data.contact[i]``, named by :func:`side_name`."""
    geom, flex, vert, elem = contact.geom, contact.flex, contact.vert, contact.elem
    return (
        side_name(model, geom[0], flex[0], vert[0], elem[0]),
        side_name(model, geom[1], flex[1], vert[1], elem[1]),
    )


@dataclass(frozen=True)
class ContactScope:
    """The watched entity's geoms and flexes, the ignored ones, and the filter they are for."""

    body: str  # the resolved base body, for log lines and error messages
    watched: np.ndarray  # per-geom mask: in the watched entity's subtree
    ignored: np.ndarray  # per-geom mask: never counts
    watched_flex: np.ndarray  # per-flex mask: owned by the watched entity
    ignored_flex: np.ndarray  # per-flex mask: never counts

    def qualifying(self, geom: np.ndarray, flex: np.ndarray) -> np.ndarray:
        """Mask over a step's contacts: exactly one side watched, neither side ignored.

        *geom* and *flex* are ``(n, 2)``: ``data.contact.geom[:n]`` and ``data.contact.flex[:n]``.
        Equal sides are either a pair the entity is not in at all or a self-contact -- a flex
        touching itself included -- and neither is an external collision.
        """
        watched = side_mask(geom, flex, self.watched, self.watched_flex)
        ignored = side_mask(geom, flex, self.ignored, self.ignored_flex)
        return (watched[:, 0] ^ watched[:, 1]) & ~(ignored[:, 0] | ignored[:, 1])

    def indices(self, data) -> np.ndarray:
        """Indices into ``data.contact`` of this step's qualifying contacts, ascending.

        Masks rather than a Python loop over ``data.ncon``: a world's contacts are dominated by
        pairs the watched entity is not in -- props on the floor, a crowd's feet -- and touching
        each of them from Python costs more than the physics step that produced them. Ascending, so
        a caller that wants the *first* contact of a step gets the one MuJoCo listed first.
        """
        n = int(data.ncon)
        if not n:
            return _NO_CONTACTS
        con = data.contact
        return np.flatnonzero(self.qualifying(con.geom[:n], con.flex[:n]))


def resolve_base_body(entity, body: str = "") -> str:
    """The body an entity's observables watch: an explicit override, else its registered base.

    ``base_link`` is the last resort and only reached for an entity that registered no body of its
    own; a plugin that watches nothing must fail rather than guess, which is why the callers here
    treat an unresolvable name as fatal.
    """
    prefix = entity.meta.get("prefix", "") if entity else ""
    if body:
        return prefix + body
    return entity.body if entity and entity.body else prefix + "base_link"


def resolve_contact_scope(
    model,
    entity,
    *,
    plugin: str,
    body: str = "",
    ignore: Iterable[str] = ("floor",),
    ignore_prefixes: Sequence[str] = (),
) -> ContactScope:
    """Resolve the watched subtree and the ignore list into masks, or fail loudly.

    *plugin* names the caller in the errors and the log line, since what a missing body means is
    the caller's story. The watched set is the subtree's geoms and the flexes it owns
    (:func:`roqsim.flex.entity_flex_ids`); *ignore* and *ignore_prefixes* match geom and flex names
    alike. Raises :class:`RuntimeError` where the base body does not resolve or its subtree carries
    neither a geom nor a flex: a meter watching nothing reports a clean run forever, which is
    indistinguishable from a trial that touched nothing and would be averaged in as one.
    """
    ignore = list(ignore)
    body_name = resolve_base_body(entity, body)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
        raise RuntimeError(f"{plugin}: base body {body_name!r} not found")

    watched = np.zeros(model.ngeom, dtype=bool)
    watched[entity_geom_ids(model, body_name)] = True
    watched_flex = np.zeros(model.nflex, dtype=bool)
    watched_flex[entity_flex_ids(model, body_name)] = True
    if not watched.any() and not watched_flex.any():
        raise RuntimeError(
            f"{plugin}: body {body_name!r} and its subtree carry no geoms or flexes to watch"
        )

    def _ignored(kind, count: int) -> np.ndarray:
        mask = np.zeros(count, dtype=bool)
        for i in range(count):
            name = mujoco.mj_id2name(model, kind, i) or ""
            if name in ignore or any(name.startswith(p) for p in ignore_prefixes):
                mask[i] = True
        return mask

    ignored = _ignored(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    ignored_flex = _ignored(mujoco.mjtObj.mjOBJ_FLEX, model.nflex)

    missing = [
        n
        for n in ignore
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) < 0
        and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n) < 0
    ]
    if missing:
        # Not fatal (a world may legitimately have no `floor` geom), but never silent: an unmatched
        # ignore entry is how a ground plane starts counting as a collision, and how the weight a
        # robot rests on it with becomes the largest impulse of the trial.
        _log.warning(
            "%s: ignore entry has no matching geom or flex: %s", plugin, ", ".join(missing)
        )

    _log.info(
        "%s: watching %d geoms and %d flexes of %r, ignoring %d geoms and %d flexes",
        plugin,
        int(watched.sum()),
        int(watched_flex.sum()),
        body_name,
        int(ignored.sum()),
        int(ignored_flex.sum()),
    )
    return ContactScope(
        body=body_name,
        watched=watched,
        ignored=ignored,
        watched_flex=watched_flex,
        ignored_flex=ignored_flex,
    )
