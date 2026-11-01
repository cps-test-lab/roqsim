# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Forward and inverse kinematics of a site on an arm, on a scratch copy of the world's data.

The arm is named by its joints (MuJoCo joint ids of hinge or slide joints, in chain order) and the
site whose pose is wanted -- a gripper's ``pinch``, a tool's ``tcp``. Every query works on a scratch
``MjData`` of its own, so the live run is never touched; the other joints of the world (a door, a
part) stay where the data it was made from had them.

**Inverse kinematics** is damped least squares on the site's 6-D error, with the step clamped to the
joint limits. It starts from a seed -- the arm's current configuration -- and, a solve that reaches
the target being one of several (an elbow up or down, a wrist flipped), returns the one it reaches
from the seed: the branch nearest where the arm is. A solve that does not converge restarts from the
seed perturbed by the generator it is given, and of every solution found the one nearest the seed
is returned. A target out of reach raises :class:`IKError` naming the residual, never returns a
best effort.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


class IKError(RuntimeError):
    """No configuration reaches the target within the tolerance."""


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation vector (axis * angle, world frame) turning ``current`` into ``target``."""
    rel = target @ current.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rel.reshape(-1))
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


@dataclass
class ArmKinematics:
    """Kinematics of ``site`` over the arm joints ``joints`` of ``model``.

    Args:
        model: the compiled world
        joints: joint ids of the arm, in chain order
        site: the site id whose pose is solved for
        data: the data the scratch copy starts from (the other joints' positions); default the
            model's reference configuration
    """

    model: mujoco.MjModel
    joints: list[int]
    site: int
    data: mujoco.MjData | None = None

    def __post_init__(self) -> None:
        m = self.model
        self.joints = [int(j) for j in self.joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.dofadr = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.lower = np.array(
            [m.jnt_range[j, 0] if m.jnt_limited[j] else -np.inf for j in self.joints]
        )
        self.upper = np.array(
            [m.jnt_range[j, 1] if m.jnt_limited[j] else np.inf for j in self.joints]
        )
        self.scratch = mujoco.MjData(m)
        if self.data is not None:
            self.scratch.qpos[:] = self.data.qpos
            self.scratch.qvel[:] = 0.0
            self.scratch.mocap_pos[:] = self.data.mocap_pos
            self.scratch.mocap_quat[:] = self.data.mocap_quat
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    @property
    def n(self) -> int:
        return len(self.joints)

    def set(self, q) -> None:
        """Pose the scratch data at ``q`` (positions and the frames derived from them)."""
        self.scratch.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """The site's world position and 3x3 rotation at ``q``."""
        self.set(q)
        return (
            self.scratch.site_xpos[self.site].copy(),
            self.scratch.site_xmat[self.site].reshape(3, 3).copy(),
        )

    def jacobian(self, q=None) -> np.ndarray:
        """The site's 6 x n Jacobian over the arm joints (translation rows first), at ``q``."""
        if q is not None:
            self.set(q)
        mujoco.mj_jacSite(self.model, self.scratch, self._jacp, self._jacr, self.site)
        return np.vstack([self._jacp[:, self.dofadr], self._jacr[:, self.dofadr]])

    def within_limits(self, q) -> bool:
        return bool(np.all(q >= self.lower - 1e-9) and np.all(q <= self.upper + 1e-9))

    def _solve(self, pos, rot, q0, *, tol_pos, tol_rot, iterations, damping, rotation):
        q = np.clip(np.asarray(q0, float).copy(), self.lower, self.upper)
        for _ in range(iterations):
            cur_p, cur_r = self.fk(q)
            ep = pos - cur_p
            er = rotation_error(cur_r, rot) if rotation else np.zeros(3)
            if np.linalg.norm(ep) <= tol_pos and np.linalg.norm(er) <= tol_rot:
                return q
            jac = self.jacobian()
            if not rotation:
                jac, err = jac[:3], ep
            else:
                err = np.concatenate([ep, er])
            jjt = jac @ jac.T + (damping**2) * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jjt, err)
            step = np.max(np.abs(dq))
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.lower, self.upper)
        return None

    def ik(
        self,
        pos,
        rot=None,
        seed=None,
        *,
        rng: np.random.Generator | None = None,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        iterations: int = 200,
        restarts: int = 30,
        spread: float = 0.6,
        damping: float = 1e-3,
        accept=None,
    ) -> np.ndarray:
        """A configuration placing the site at ``pos`` with rotation ``rot`` (3x3; ``None``: any).

        From ``seed`` first; then, if that fails or ``accept`` (a predicate on q) refuses it, from
        the seed perturbed by up to ``spread`` rad per joint, ``restarts`` times, keeping the
        solution nearest the seed. Raises :class:`IKError` when none is found.
        """
        pos = np.asarray(pos, float)
        rot = None if rot is None else np.asarray(rot, float).reshape(3, 3)
        seed = np.asarray(self.scratch.qpos[self.qadr] if seed is None else seed, float)
        kw = {
            "tol_pos": tol_pos,
            "tol_rot": tol_rot,
            "iterations": iterations,
            "damping": damping,
            "rotation": rot is not None,
        }
        found = []
        q = self._solve(pos, rot, seed, **kw)
        if q is not None and (accept is None or accept(q)):
            return q
        if restarts and rng is None:
            raise ValueError(
                "ik: the seed did not give a solution, and the restarts draw from a generator: "
                "pass rng= (or restarts=0)"
            )
        for _ in range(restarts):
            start = seed + rng.uniform(-spread, spread, size=self.n)
            q = self._solve(pos, rot, start, **kw)
            if q is not None and (accept is None or accept(q)):
                found.append(q)
        if not found:
            cur_p, cur_r = self.fk(seed)
            raise IKError(
                f"no configuration within the joint limits reaches {np.round(pos, 4).tolist()} "
                f"(from the seed the site is {np.linalg.norm(pos - cur_p):.3f} m away); "
                "out of reach, or every solution is refused"
            )
        return min(found, key=lambda x: float(np.linalg.norm(x - seed)))
