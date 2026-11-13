"""The robot's own model: the nominal arm plus the load it was told, never the world's truth.

A Franka controller compensates gravity and estimates external torques from *its* model: the arm's
nominal dynamics and the end effector and load configured in Desk and through ``setLoad``. A tool
heavier than configured is therefore a constant external torque on the robot, and that is how it
shows here: this model is a private MuJoCo copy of the arm, built from the same MJCF, with the
joint armature, damping and friction removed (they act on the motor side, before the link-side
torque sensor) and one body at the flange carrying the configured end effector and load.

Frames are libfranka's: ``O`` the base (``<prefix>fr3_link0``), ``F`` the flange (``fr3_link8``),
``NE`` the nominal end effector (Desk), ``EE`` the end effector (``setEE``), ``K`` the stiffness
frame (``setK``). Poses are 4 x 4 homogeneous matrices in ``O``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def combine_loads(loads) -> tuple[float, np.ndarray, np.ndarray]:
    """One rigid body equivalent to several: ``[(mass, com (3,), inertia about com (3, 3)), ...]``.

    All in the flange frame. libfranka adds ``m_ee`` and ``m_load`` the same way into ``m_total``.
    """
    m = sum(float(mi) for mi, _, _ in loads)
    if m <= 0.0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = sum(float(mi) * np.asarray(ci, float) for mi, ci, _ in loads) / m
    inertia = np.zeros((3, 3))
    for mi, ci, Ii in loads:
        d = np.asarray(ci, float) - c
        inertia += np.asarray(Ii, float).reshape(3, 3) + float(mi) * (
            d @ d * np.eye(3) - np.outer(d, d)
        )
    return m, c, inertia


class RobotModel:
    """The arm's nominal model with a configurable load at the flange; one per simulated arm."""

    def __init__(self, mjcf: Path, gravity_in_base: np.ndarray):
        spec = mujoco.MjSpec.from_file(str(mjcf))
        for act in list(spec.actuators):
            spec.delete(act)
        for geom in list(spec.geoms):
            geom.contype = geom.conaffinity = 0
        load = spec.body("fr3_link8").add_body(name="__configured_load")
        load.mass = 1e-9
        load.inertia = [1e-12, 1e-12, 1e-12]
        load.explicitinertial = True
        for key in list(spec.keys):
            spec.delete(key)
        self.model = spec.compile()
        self.model.opt.gravity[:] = gravity_in_base
        self.model.dof_armature[:] = 0.0
        self.model.dof_damping[:] = 0.0
        self.model.dof_frictionloss[:] = 0.0
        self.data = mujoco.MjData(self.model)
        self._still = mujoco.MjData(self.model)  # the same pose at rest: its bias is gravity alone
        self._load = self.model.body("__configured_load").id
        self._flange = self.model.body("fr3_link8").id
        self.nv = self.model.nv
        self.set_load(0.0, np.zeros(3), np.zeros((3, 3)))

    def set_load(self, mass: float, com, inertia) -> None:
        """The configured end effector and load together, in the flange frame (``combine_loads``)."""
        m = self.model
        m.body_mass[self._load] = max(float(mass), 1e-9)
        m.body_ipos[self._load] = np.asarray(com, float)
        evals, evecs = np.linalg.eigh(np.asarray(inertia, float).reshape(3, 3))
        if np.linalg.det(evecs) < 0:
            evecs[:, 0] *= -1
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, evecs.reshape(9))
        m.body_iquat[self._load] = quat
        m.body_inertia[self._load] = np.maximum(evals, 1e-12)

    def update(self, q, dq) -> None:
        d, still = self.data, self._still
        d.qpos[:] = q
        d.qvel[:] = dq
        mujoco.mj_forward(self.model, d)
        still.qpos[:] = q
        mujoco.mj_forward(self.model, still)

    # -- what libfranka's Model offers, at the state of the last update ---------------------------

    def mass(self) -> np.ndarray:
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        return M

    def coriolis(self) -> np.ndarray:
        """C(q, dq) dq: the bias torque without gravity."""
        return self.data.qfrc_bias.copy() - self.gravity()

    def gravity(self) -> np.ndarray:
        """g(q): the torque that holds the arm and its configured load at rest."""
        return self._still.qfrc_bias.copy()

    def flange_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[self._flange].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self._flange]
        return T

    def zero_jacobian(self, F_T_X: np.ndarray) -> np.ndarray:
        """6 x 7 geometric Jacobian of frame X (given in the flange frame), linear rows first, in O."""
        T = self.flange_pose() @ F_T_X
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, T[:3, 3], self._flange)
        return np.vstack([jacp, jacr])
