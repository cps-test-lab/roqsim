# SPDX-License-Identifier: Apache-2.0
"""Hand targets: a world pose, or a body, a site or an entity of the scene with an offset in its
frame, resolved every tick so a hand follows what it is on.

Spelled as a move takes it::

    "drawer_1/grip"                      # a site of the model, by its name (or a body's)
    "tote_1"                             # an entity: its body
    {"position": {"x": 1.2, "y": 0.4, "z": 0.9}, "orientation": {"yaw": 1.57}}   # a world pose

and an ``offset`` as ``(x, y, z)`` or ``(x, y, z, roll, pitch, yaw)`` in the target's frame. A name
is looked up as a site, then a body, then an entity, and one that is none of them is refused,
naming what was looked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.pose import PoseError, parse_pose

from . import quat as Q


class TargetError(ValueError):
    """A target that names nothing in the scene, or is not a pose."""


@dataclass
class Target:
    """A resolvable frame: ``kind`` is ``site``, ``body`` or ``world``; ``ident`` its id."""

    name: str
    kind: str
    ident: int
    pos: np.ndarray  # a world pose's position, or the offset's
    quat: np.ndarray  # a world pose's rotation, or the offset's

    def resolve(self, model, data) -> tuple[np.ndarray, np.ndarray]:
        """The target's world ``(position, rotation)`` now."""
        if self.kind == "world":
            return self.pos.copy(), self.quat.copy()
        if self.kind == "site":
            p = data.site_xpos[self.ident]
            r = Q.from_matrix(data.site_xmat[self.ident])
        else:
            p = data.xpos[self.ident]
            r = data.xquat[self.ident].copy()
        return p + Q.rotate(r, self.pos), Q.mul(r, self.quat)

    @property
    def body(self) -> int:
        """The body the target is on (``-1`` for a world pose)."""
        return -1 if self.kind == "world" else self._body

    _body: int = -1
    #: Whether the target states a heading: a site's or a body's frame, or a world pose given with
    #: a rotation; a world pose with only a position has none.
    heading: bool = True


def _offset(offset) -> tuple[np.ndarray, np.ndarray]:
    if offset is None:
        return np.zeros(3), Q.IDENTITY.copy()
    values = [float(v) for v in offset]
    if len(values) not in (3, 6):
        raise TargetError(f"an offset is (x, y, z) or (x, y, z, roll, pitch, yaw), not {offset!r}")
    rot = Q.from_rpy(*values[3:]) if len(values) == 6 else Q.IDENTITY.copy()
    return np.array(values[:3]), rot


def make_target(model, entities, spec, offset=None) -> Target:
    """Resolve ``spec`` (a name or a world pose) against ``model`` and the entity registry once;
    :meth:`Target.resolve` then reads it every tick."""
    if isinstance(spec, Target):
        return spec
    off_pos, off_rot = _offset(offset)
    if isinstance(spec, dict):
        try:
            pos, rot = parse_pose(spec)
        except PoseError as exc:
            raise TargetError(f"target {spec!r}: {exc}") from exc
        if pos[2] is None:
            raise TargetError(
                f"target {spec!r}: a world pose for a hand states its height, 'position.z'; "
                "there is no resting height for a hand to fall back on"
            )
        p = np.array([float(v) for v in pos])
        r = np.array(rot, float)
        return Target(
            str(spec),
            "world",
            -1,
            p + Q.rotate(r, off_pos),
            Q.mul(r, off_rot),
            heading="orientation" in (spec["pose"] if isinstance(spec.get("pose"), dict) else spec),
        )
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        p, r = np.asarray(spec[0], float), np.asarray(spec[1], float)
        if r.shape == (3,):
            r = Q.from_rpy(*r)
        return Target(str(spec), "world", -1, p + Q.rotate(r, off_pos), Q.mul(r, off_rot))
    name = str(spec)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        t = Target(name, "site", sid, off_pos, off_rot)
        t._body = int(model.site_bodyid[sid])
        return t
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    entity = entities.get(name) if entities is not None and bid < 0 else None
    if bid < 0 and entity is not None and entity.body:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
    if bid >= 0:
        t = Target(name, "body", bid, off_pos, off_rot)
        t._body = bid
        return t
    raise TargetError(
        f"unknown target {name!r}: no site, body or entity of that name (a site or body is named as "
        f"in the model, e.g. 'drawer_1/grip'; an entity by the world's `name:`)"
    )
