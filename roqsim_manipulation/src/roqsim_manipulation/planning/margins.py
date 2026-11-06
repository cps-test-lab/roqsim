# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""How much room a pose leaves: to each joint's limits, to a singularity, to the drives' torque.

A cell that "works" can still only just reach a place: a joint a few degrees from its stop, a wrist
nearly straight, a shoulder carrying most of what it can. :func:`pose_margins` measures each of
these for one configuration, so a place the cell only just reaches shows as such.

- **Joint limits:** each joint's distance to the nearer end of its range, in rad and as a fraction
  of half the range.
- **Singularity:** the Jacobian of the site (6 x n, translation in m, rotation in rad) -- its smallest
  singular value and Yoshikawa's manipulability ``sqrt(det(J Jᵀ))`` -- and, for a six-joint arm
  of the UR kind (its second, third and fourth axes parallel), the three singularities by name: the **wrist**
  (``|sin|`` of the angle between the first and the last wrist axes; 0 when they line up), the
  **elbow** (``|sin|`` of the elbow joint; 0 when the arm is stretched out or folded back) and the
  **shoulder** (the wrist centre's distance from the base axis, m; 0 when it is on it).
- **Torque:** the joint torques that hold the pose against gravity, with a payload at a point of a
  body (the gripper and the part it holds), by inverse dynamics; and their ratio to each joint's
  limit, the force range of the actuator driving it.

:class:`ArmDynamics` gives the same inverse dynamics for a moving arm, which is what
:func:`~.timing.time_parameterize` checks a trajectory's torques with.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .kinematics import ArmKinematics


class ArmDynamics:
    """Inverse dynamics of the arm joints, with a payload, on a copy of the model.

    The copy has contacts and constraints disabled and no gravity compensation, so the torque is
    what the joint's drive must deliver: inertia, Coriolis and gravity, the payload included.

    Args:
        model: the compiled world
        joints: the arm's joint ids
        payload: ``(body id, mass kg, position in the body's frame)`` added to that body, or None
        data: the data the copy's other joints are taken from
    """

    def __init__(self, model, joints, payload=None, data=None) -> None:
        m = copy.copy(model)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        m.body_gravcomp[:] = 0.0
        if payload is not None:
            body, mass, pos = payload
            mass, pos = float(mass), np.asarray(pos, float)
            m0, c0 = float(m.body_mass[body]), m.body_ipos[body].copy()
            total = m0 + mass
            # A point mass at `pos`: the centres of mass combine; the body's rotational inertia is
            # kept, which for a part of a few hundred grams at the gripper is the smaller term.
            m.body_ipos[body] = (m0 * c0 + mass * pos) / total
            m.body_mass[body] = total
        self.model = m
        self.data = mujoco.MjData(m)
        if data is not None:
            self.data.qpos[:] = data.qpos
        self.joints = [int(j) for j in joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.dofadr = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.limits = joint_torque_limits(model, self.joints)

    def torque(self, q, qd=None, qdd=None) -> np.ndarray:
        d = self.data
        d.qpos[self.qadr] = q
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        if qd is not None:
            d.qvel[self.dofadr] = qd
        if qdd is not None:
            d.qacc[self.dofadr] = qdd
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse[self.dofadr].copy()


def joint_torque_limits(model, joints) -> np.ndarray:
    """Each joint's torque limit: the force range of the actuator that drives it (``inf`` if none)."""
    out = np.full(len(joints), np.inf)
    for k, j in enumerate(joints):
        for a in range(model.nu):
            if (
                model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[a, 0] == j
                and model.actuator_forcelimited[a]
            ):
                lo, hi = model.actuator_forcerange[a]
                out[k] = min(abs(lo), abs(hi)) * float(model.actuator_gear[a, 0] or 1.0)
    return out


@dataclass
class PoseMargins:
    """The margins of one configuration; see the module docstring for each."""

    q: list[float]
    #: per joint, rad to the nearer end of its range
    limit_rad: list[float]
    #: per joint, that distance as a fraction of half the range (1 in the middle, 0 at a stop)
    limit_fraction: list[float]
    sigma_min: float
    manipulability: float
    #: The singularity measures by name, or None for an arm that is not of the UR kind
    wrist: float | None
    elbow: float | None
    shoulder_m: float | None
    #: per joint, the static torque with the payload (N*m) and its ratio to the limit
    torque: list[float] = field(default_factory=list)
    torque_ratio: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def pose_margins(kin: ArmKinematics, q, dynamics: ArmDynamics | None = None) -> PoseMargins:
    """The margins of configuration ``q`` of the arm ``kin`` describes."""
    q = np.asarray(q, float)
    lower, upper = kin.lower, kin.upper
    dist = np.minimum(q - lower, upper - q)
    half = (upper - lower) / 2
    frac = np.where(np.isfinite(half), dist / half, 1.0)
    jac = kin.jacobian(q)
    sv = np.linalg.svd(jac, compute_uv=False)
    sigma_min = float(sv[-1]) if len(sv) else 0.0
    manip = float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))
    wrist = elbow = shoulder = None
    d = kin.scratch
    axes = [d.xaxis[j] for j in kin.joints]
    anchors = [d.xanchor[j] for j in kin.joints]
    # The named measures read a UR-kind arm's joints by position; another arm has others there.
    ur_kind = kin.n == 6 and all(
        np.linalg.norm(np.cross(axes[i], axes[i + 1])) < 1e-6 for i in (1, 2)
    )
    if ur_kind:
        wrist = float(np.linalg.norm(np.cross(axes[3], axes[5])))
        elbow = float(abs(np.sin(q[2])))
        # The wrist centre: where the second wrist axis meets the first, its anchor point.
        centre = anchors[4]
        base_point, base_axis = anchors[0], axes[0] / np.linalg.norm(axes[0])
        rel = centre - base_point
        shoulder = float(np.linalg.norm(rel - np.dot(rel, base_axis) * base_axis))
    tau, ratio = [], []
    if dynamics is not None:
        t = dynamics.torque(q)
        tau = t.tolist()
        ratio = (np.abs(t) / dynamics.limits).tolist()
    return PoseMargins(
        q=q.tolist(),
        limit_rad=dist.tolist(),
        limit_fraction=frac.tolist(),
        sigma_min=sigma_min,
        manipulability=manip,
        wrist=wrist,
        elbow=elbow,
        shoulder_m=shoulder,
        torque=tau,
        torque_ratio=ratio,
    )
