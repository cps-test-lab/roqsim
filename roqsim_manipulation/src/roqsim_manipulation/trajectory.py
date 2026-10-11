# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A joint trajectory sampled on simulated time, and the feed-forward that makes a position servo track it.

``JointTrajectory`` holds knots: times, positions and optionally velocities and accelerations, the
fields of a ``trajectory_msgs/JointTrajectory``. It is sampled the way ros2_control's
JointTrajectoryController samples one. Between two knots the curve is:

- the quintic matching position, velocity and acceleration at both ends, when the knots carry
  accelerations;
- the cubic Hermite, when they carry velocities only;
- a straight line, when they carry positions only.

A trajectory sampled at its own knots reproduces them exactly; a piecewise quintic sampled at its own
segment boundaries is reproduced exactly everywhere.

``servo_feedforward`` is what an affine position servo (``force = p * ctrl - p * q - d * qd``, the
MJCF ``<position>`` actuator and ``roqsim.actuators``' ``position``/``impedance`` laws) must be handed
so that, on the planned state, its force is the force the motion needs:

    ctrl = q_d + (d * qd_d + tau_ff) / p

``velocity`` feed-forward cancels the damping term's lag (``d * qd / p``, which a position target alone
leaves behind at speed). ``dynamics`` adds the inverse dynamics as well:

    tau_ff = M(q_d) qdd_d + c(q_d, qd_d) - g_drive(q_d)

Here ``g_drive`` is the gravity the model compensates outside the servo (``qfrc_gravcomp``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

FEEDFORWARD = ("none", "velocity", "dynamics")


@dataclass
class JointTrajectory:
    times: NDArray[np.float64]  # (n,), strictly increasing, from 0
    positions: NDArray[np.float64]  # (n, j)
    velocities: NDArray[np.float64] | None = None  # (n, j)
    accelerations: NDArray[np.float64] | None = None  # (n, j); only with velocities

    @classmethod
    def build(cls, times, positions, velocities=None, accelerations=None) -> JointTrajectory:
        """Validate and freeze knots; a malformed trajectory raises ``ValueError`` naming the fault."""
        t = np.asarray(times, dtype=float).reshape(-1)
        q = np.asarray(positions, dtype=float)
        if q.ndim != 2 or q.shape[0] != t.size or t.size < 1:
            raise ValueError(
                f"positions must be (knots, joints) with one row per time; got {q.shape} for {t.size} times"
            )
        if np.any(np.diff(t) <= 0.0) or t[0] < 0.0:
            raise ValueError("times must start at or after 0 and increase strictly")
        qd = None if velocities is None else np.asarray(velocities, dtype=float)
        qdd = None if accelerations is None else np.asarray(accelerations, dtype=float)
        for name, arr in (("velocities", qd), ("accelerations", qdd)):
            if arr is not None and arr.shape != q.shape:
                raise ValueError(f"{name} must match positions' shape {q.shape}, got {arr.shape}")
        if qdd is not None and qd is None:
            raise ValueError("accelerations need velocities")
        if not (
            np.all(np.isfinite(q))
            and (qd is None or np.all(np.isfinite(qd)))
            and (qdd is None or np.all(np.isfinite(qdd)))
        ):
            raise ValueError("knots must be finite")
        return cls(t, q, qd, qdd)

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, t: float):
        """``(q, qd, qdd)`` at ``t`` seconds after the start; the first knot before it, the last after."""
        tt, q = self.times, self.positions
        n, j = q.shape
        zeros = np.zeros(j)
        if t <= tt[0]:
            if t < tt[0] or n == 1:
                return q[0].copy(), zeros, zeros.copy()
        if t > tt[-1] or (t == tt[-1] and self.velocities is None):
            return q[-1].copy(), zeros, zeros.copy()
        if t == tt[-1]:
            acc = zeros if self.accelerations is None else self.accelerations[-1].copy()
            return q[-1].copy(), self.velocities[-1].copy(), acc
        i = int(np.searchsorted(tt, t, side="right") - 1)
        h = tt[i + 1] - tt[i]
        u = (t - tt[i]) / h
        p0, p1 = q[i], q[i + 1]
        if self.velocities is None:
            v = (p1 - p0) / h
            return p0 + (p1 - p0) * u, v, zeros
        v0, v1 = self.velocities[i] * h, self.velocities[i + 1] * h
        if self.accelerations is None:
            h00, h10, h01, h11 = (
                2 * u**3 - 3 * u**2 + 1,
                u**3 - 2 * u**2 + u,
                -2 * u**3 + 3 * u**2,
                u**3 - u**2,
            )
            d00, d10, d01, d11 = (
                6 * u**2 - 6 * u,
                3 * u**2 - 4 * u + 1,
                -6 * u**2 + 6 * u,
                3 * u**2 - 2 * u,
            )
            e00, e10, e01, e11 = 12 * u - 6, 6 * u - 4, -12 * u + 6, 6 * u - 2
            pos = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
            vel = (d00 * p0 + d10 * v0 + d01 * p1 + d11 * v1) / h
            acc = (e00 * p0 + e10 * v0 + e01 * p1 + e11 * v1) / (h * h)
            return pos, vel, acc
        a0, a1 = self.accelerations[i] * h * h, self.accelerations[i + 1] * h * h
        u2, u3, u4, u5 = u * u, u**3, u**4, u**5
        b = (
            1 - 10 * u3 + 15 * u4 - 6 * u5,
            u - 6 * u3 + 8 * u4 - 3 * u5,
            0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5,
            10 * u3 - 15 * u4 + 6 * u5,
            -4 * u3 + 7 * u4 - 3 * u5,
            0.5 * u3 - u4 + 0.5 * u5,
        )
        db = (
            -30 * u2 + 60 * u3 - 30 * u4,
            1 - 18 * u2 + 32 * u3 - 15 * u4,
            u - 4.5 * u2 + 6 * u3 - 2.5 * u4,
            30 * u2 - 60 * u3 + 30 * u4,
            -12 * u2 + 28 * u3 - 15 * u4,
            1.5 * u2 - 4 * u3 + 2.5 * u4,
        )
        ddb = (
            -60 * u + 180 * u2 - 120 * u3,
            -36 * u + 96 * u2 - 60 * u3,
            1 - 9 * u + 18 * u2 - 10 * u3,
            60 * u - 180 * u2 + 120 * u3,
            -24 * u + 84 * u2 - 60 * u3,
            3 * u - 12 * u2 + 10 * u3,
        )
        terms = (p0, v0, a0, p1, v1, a1)
        pos = sum(c * x for c, x in zip(b, terms, strict=True))
        vel = sum(c * x for c, x in zip(db, terms, strict=True)) / h
        acc = sum(c * x for c, x in zip(ddb, terms, strict=True)) / (h * h)
        return pos, vel, acc


def affine_servo_gains(model, actuator_id: int) -> tuple[float, float] | None:
    """``(p, d)`` of an affine position servo on a joint with unit gear, or ``None`` if it is not one."""
    if int(model.actuator_gaintype[actuator_id]) != int(mujoco.mjtGain.mjGAIN_FIXED):
        return None
    if int(model.actuator_biastype[actuator_id]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
        return None
    if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
        return None
    if abs(float(model.actuator_gear[actuator_id][0]) - 1.0) > 1e-12:
        return None
    p = float(model.actuator_gainprm[actuator_id][0])
    if p <= 0.0 or abs(float(model.actuator_biasprm[actuator_id][1]) + p) > 1e-9 * max(1.0, p):
        return None
    return p, -float(model.actuator_biasprm[actuator_id][2])


def servo_feedforward(
    model, scratch, base_qpos, qadr, vadr, q, qd, qdd, mode: str
) -> NDArray[np.float64]:
    """The torque feed-forward ``tau_ff`` of the module docstring (zero for ``velocity``).

    ``scratch`` is a private ``MjData``; ``base_qpos`` fills the degrees of freedom the trajectory does
    not own (the rest of the world, as it stands).
    """
    if mode != "dynamics":
        return np.zeros(len(vadr))
    scratch.qpos[:] = base_qpos
    scratch.qvel[:] = 0.0
    scratch.qpos[qadr] = q
    scratch.qvel[vadr] = qd
    mujoco.mj_forward(model, scratch)
    acc = np.zeros(model.nv)
    acc[vadr] = qdd
    out = np.zeros(model.nv)
    mujoco.mj_mulM(model, scratch, out, acc)
    tau = out + scratch.qfrc_bias - scratch.qfrc_gravcomp
    return tau[vadr]
