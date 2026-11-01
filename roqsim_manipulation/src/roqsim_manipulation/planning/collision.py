# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Whether an arm configuration, with what the arm holds, touches what it should not.

The check runs on a **copy** of the compiled model and of the data it was made from, so it sees the
world as it stands now -- a door open or shut, the parts still in a box, a look shown or hidden --
and never writes the live run. What moves is the arm (every body in the subtree of its base, the
gripper included) and the parts it holds, each carried at the pose it was gripped in relative to
the arm's site. A configuration is in collision when any geom of what moves comes within
``clearance`` of a geom it is not allowed to touch: MuJoCo's own collision detection, with the
moving geoms' margin set to the clearance, finds every such pair at once, broad phase included.

What may touch is said by geom pairs (``allow``): a held part and the gripper's fingers, a part and
the support it is being lifted off. The arm's own links are checked against each other as MuJoCo
checks them (adjacent bodies excluded), so a gripper swung into the forearm is a collision too. The
arm's base does not move, and what its base and first link touch where the arm stands -- its mount,
the floor or stand it is bolted to -- is the mounting, allowed from the start.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import mujoco
import numpy as np

from .kinematics import ArmKinematics


class CollisionError(RuntimeError):
    """A configuration that must be free is not."""


def subtree_bodies(model: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


def body_geoms(model: mujoco.MjModel, bodies: Iterable[int]) -> set[int]:
    bodies = set(bodies)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}


class CollisionChecker:
    """Collision queries for an arm over a snapshot of the world.

    Args:
        model, data: the live world; both are copied, so later changes to them are not seen
        joints: the arm's joint ids, in chain order
        site: the site held parts are carried relative to
        base: the arm's base body; everything in its subtree moves with the arm
        clearance: how near a moving geom may come to anything (m)
    """

    def __init__(self, model, data, joints, site, base, *, clearance: float = 0.005) -> None:
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = data.qpos
        self.data.mocap_pos[:] = data.mocap_pos
        self.data.mocap_quat[:] = data.mocap_quat
        self.data.qvel[:] = 0.0
        self.kin = ArmKinematics(self.model, joints, site, self.data)
        self.kin.scratch = self.data
        self.clearance = float(clearance)
        self.arm_bodies = subtree_bodies(self.model, base)
        self.arm_geoms = body_geoms(self.model, self.arm_bodies)
        #: held part body -> (its free joint's qpos address, pose in the site frame: pos, 3x3)
        self._held: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
        self._held_geoms: set[int] = set()
        #: The pairs that touch where the arm is mounted, allowed for good (:meth:`_allow_mount`).
        self._mount: set[frozenset[int]] = set()
        self._allowed: set[frozenset[int]] = set()
        self._margin0 = self.model.geom_margin.copy()
        self._set_margins()
        self._allow_mount(base)

    # -- what moves and what may touch ----------------------------------------------------------
    def _allow_mount(self, base: int) -> None:
        """Allow what the base and its first link touch where the arm stands now: the mount."""
        m = self.model
        near = {base} | {b for b in self.arm_bodies if m.body_parentid[b] == base}
        near_geoms = body_geoms(m, near)
        mujoco.mj_kinematics(m, self.data)
        mujoco.mj_collision(m, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in near_geoms and b not in self.arm_geoms:
                    self._mount.add(frozenset((a, b)))
        self._allowed |= self._mount

    def _set_margins(self) -> None:
        for g in self.arm_geoms | self._held_geoms:
            self.model.geom_margin[g] = self.clearance

    def hold(self, body: int, data: mujoco.MjData) -> None:
        """Carry ``body`` (a free body) with the site, at the pose ``data`` has it in relative to it."""
        m = self.model
        jnt = m.body_jntadr[body]
        if jnt < 0 or m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"body {m.body(body).name!r} has no free joint, so it cannot be carried"
            )
        site_pos, site_rot = (
            data.site_xpos[self.kin.site],
            data.site_xmat[self.kin.site].reshape(3, 3),
        )
        rel_pos = site_rot.T @ (data.xpos[body] - site_pos)
        rel_rot = site_rot.T @ data.xmat[body].reshape(3, 3)
        self._held[body] = (int(m.jnt_qposadr[jnt]), rel_pos, rel_rot)
        geoms = body_geoms(m, subtree_bodies(m, body))
        self._held_geoms |= geoms
        # A held part touches the fingers holding it; that is the grip, not a collision.
        for g in geoms:
            for a in self.arm_geoms:
                self._allowed.add(frozenset((g, a)))
        self._set_margins()

    def release(self, body: int, data: mujoco.MjData) -> None:
        """Stop carrying ``body``: it stays where ``data`` has it now, its contacts with the arm
        count again, and it keeps no clearance of its own."""
        if body not in self._held:
            raise ValueError(f"body {self.model.body(body).name!r} is not held")
        adr = self._held.pop(body)[0]
        self.data.qpos[adr : adr + 7] = data.qpos[adr : adr + 7]
        geoms = body_geoms(self.model, subtree_bodies(self.model, body))
        self._held_geoms -= geoms
        self._allowed = {p for p in self._allowed if not (p & geoms and p & self.arm_geoms)}
        for g in geoms:
            self.model.geom_margin[g] = self._margin0[g]

    def allow(self, geoms_a: Iterable[int], geoms_b: Iterable[int]) -> None:
        """Let every geom of ``geoms_a`` touch every geom of ``geoms_b``."""
        for a in geoms_a:
            for b in geoms_b:
                self._allowed.add(frozenset((int(a), int(b))))

    def disallow_all_extra(self) -> None:
        """Forget every pair :meth:`allow` added, keeping the mount and the grips."""
        held = self._held_geoms
        self._allowed = self._mount | {p for p in self._allowed if p & held and p & self.arm_geoms}

    # -- queries ----------------------------------------------------------------------------------
    def pose(self, q) -> None:
        """Pose the arm at ``q`` and every held part with it."""
        self.kin.set(q)
        site_pos = self.data.site_xpos[self.kin.site]
        site_rot = self.data.site_xmat[self.kin.site].reshape(3, 3)
        for adr, rel_pos, rel_rot in self._held.values():
            self.data.qpos[adr : adr + 3] = site_pos + site_rot @ rel_pos
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (site_rot @ rel_rot).reshape(-1))
            self.data.qpos[adr + 3 : adr + 7] = quat
        mujoco.mj_kinematics(self.model, self.data)

    def contacts(self, q) -> list[tuple[int, int, float]]:
        """``(geom, geom, distance)`` of every pair within the clearance that is not allowed."""
        self.pose(q)
        mujoco.mj_collision(self.model, self.data)
        moving = self.arm_geoms | self._held_geoms
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 not in moving and g2 not in moving:
                continue
            if frozenset((g1, g2)) in self._allowed:
                continue
            if c.dist >= self.clearance:
                continue
            out.append((g1, g2, float(c.dist)))
        return out

    def free(self, q) -> bool:
        return not self.contacts(q)

    def describe(self, q) -> str:
        """The offending pairs at ``q``, by name, for an error message."""
        m = self.model
        names = []
        for g1, g2, dist in self.contacts(q)[:6]:
            n1 = m.geom(g1).name or f"geom {g1} of {m.body(m.geom_bodyid[g1]).name}"
            n2 = m.geom(g2).name or f"geom {g2} of {m.body(m.geom_bodyid[g2]).name}"
            names.append(f"{n1} / {n2} at {dist * 1000:.1f} mm")
        return "; ".join(names) or "nothing"

    def segment_free(self, qa, qb, step: float = 0.02) -> bool:
        """Whether the straight joint-space segment from ``qa`` to ``qb`` is free, at ``step`` rad."""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        n = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / step)))
        for k in range(1, n + 1):
            if not self.free(qa + (qb - qa) * k / n):
                return False
        return True
