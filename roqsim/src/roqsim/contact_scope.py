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

from .presence import entity_geom_ids

_log = logging.getLogger(__name__)

_NO_CONTACTS = np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class ContactScope:
    """The watched entity's geoms, the ignored ones, and the filter both are for."""

    body: str  # the resolved base body, for log lines and error messages
    watched: np.ndarray  # per-geom mask: in the watched entity's subtree
    ignored: np.ndarray  # per-geom mask: never counts

    def qualifying(self, geom1: np.ndarray, geom2: np.ndarray) -> np.ndarray:
        """Mask over a step's contacts: exactly one side watched, neither side ignored.

        Equal sides are either a pair the entity is not in at all or a self-contact, and neither is
        an external collision.
        """
        return (self.watched[geom1] ^ self.watched[geom2]) & ~(
            self.ignored[geom1] | self.ignored[geom2]
        )

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
        return np.flatnonzero(self.qualifying(con.geom1[:n], con.geom2[:n]))


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
    the caller's story. Raises :class:`RuntimeError` where the base body does not resolve or its
    subtree carries no geoms: a meter watching nothing reports a clean run forever, which is
    indistinguishable from a trial that touched nothing and would be averaged in as one.
    """
    ignore = list(ignore)
    body_name = resolve_base_body(entity, body)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
        raise RuntimeError(f"{plugin}: base body {body_name!r} not found")

    watched = np.zeros(model.ngeom, dtype=bool)
    watched[entity_geom_ids(model, body_name)] = True
    if not watched.any():
        raise RuntimeError(f"{plugin}: body {body_name!r} and its subtree carry no geoms to watch")

    ignored = np.zeros(model.ngeom, dtype=bool)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name in ignore or any(name.startswith(p) for p in ignore_prefixes):
            ignored[gid] = True

    missing = [n for n in ignore if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) < 0]
    if missing:
        # Not fatal (a world may legitimately have no `floor` geom), but never silent: an unmatched
        # ignore entry is how a ground plane starts counting as a collision, and how the weight a
        # robot rests on it with becomes the largest impulse of the trial.
        _log.warning("%s: ignore entry has no matching geom: %s", plugin, ", ".join(missing))

    _log.info(
        "%s: watching %d geoms of %r, ignoring %d",
        plugin,
        int(watched.sum()),
        body_name,
        int(ignored.sum()),
    )
    return ContactScope(body=body_name, watched=watched, ignored=ignored)
