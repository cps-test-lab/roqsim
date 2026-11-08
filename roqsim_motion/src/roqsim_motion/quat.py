# SPDX-License-Identifier: Apache-2.0
"""Quaternions (w, x, y, z) and the few rotations the motion core needs, in plain numpy."""

from __future__ import annotations

import math

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def inv(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, v) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    vv = np.asarray(v, dtype=float)
    return vv + 2.0 * np.cross(u, np.cross(u, vv) + w * vv)


def axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(AXES[axis] if isinstance(axis, str) else axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), *(axis * s)])


def between(a, b) -> np.ndarray:
    """The shortest rotation taking direction ``a`` to direction ``b``."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle(axis, math.pi)
    q = np.array([1.0 + c, *np.cross(a, b)])
    return q / np.linalg.norm(q)


def normalize(q) -> np.ndarray:
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def to_rotvec(q) -> np.ndarray:
    """The rotation vector (axis * angle) of ``q``, the shorter way round."""
    q = normalize(q)
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(s, q[0])
    return q[1:] / s * angle


def angle_between(a, b) -> float:
    """The angle (rad) of the rotation from ``a`` to ``b``."""
    return float(np.linalg.norm(to_rotvec(mul(b, inv(a)))))


def slerp(a, b, t: float) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return normalize(a + t * (b - a))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll about x, then pitch about y, then yaw about z, in the fixed frame (as a world YAML)."""
    return mul(axis_angle("z", yaw), mul(axis_angle("y", pitch), axis_angle("x", roll)))


def to_matrix(q) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def from_matrix(m) -> np.ndarray:
    m = np.asarray(m, float).reshape(3, 3)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return normalize(q)


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def euler(q, order: str) -> list[float]:
    """Angles ``(a1, a2, a3)`` with ``q = R(order[0], a1) R(order[1], a2) R(order[2], a3)``, each
    about the moving frame's axis; ``order`` is three distinct axes of ``xyz``."""
    from scipy.spatial.transform import Rotation

    w, x, y, z = normalize(q)
    return list(Rotation.from_quat([x, y, z, w]).as_euler(order.upper()))


def from_euler(order: str, angles) -> np.ndarray:
    q = IDENTITY.copy()
    for axis, a in zip(order, angles, strict=True):
        q = mul(q, axis_angle(axis, a))
    return q
