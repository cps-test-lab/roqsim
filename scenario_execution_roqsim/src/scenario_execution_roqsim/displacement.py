# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""What "it moved" means, as arithmetic. No MuJoCo, no ROS, no scenario-execution.

Separated from the actions so the one part with a right and a wrong answer can be tested as a table,
in any venv, without building a world.

**Net displacement, not path length.** An entity that travels out and comes back has moved zero here.
That is the opposite of ``osc.ros``' ``odometry_distance_traveled``, which integrates, and the
difference is worth knowing before choosing between them: on a curved approach the integral is larger,
sometimes much larger. Displacement is what a fault trigger wants ("is it clear of the table yet"),
integrated path is what an odometry budget wants.
"""

from __future__ import annotations

import numpy as np

#: Translation measures. `distance` and `planar` are magnitudes; the axis modes are SIGNED, so the
#: sign of the threshold says which way (`z: 0.05` risen, `z: -0.05` fallen). Signed axes rather than
#: nine enum values (up/down/abs x 3) -- and the sign rule is the first thing the .osc comment says,
#: because someone will read `mode: z, threshold: 0.05` expecting |dz|.
AXES = {"x": 0, "y": 1, "z": 2}
MAGNITUDE_MODES = ("distance", "planar")
MODES = MAGNITUDE_MODES + tuple(AXES)


def displacement(mode: str, start: np.ndarray, now: np.ndarray) -> float:
    """The measured quantity for *mode*, in metres.

    For a magnitude mode: how far it is from where it started. For an axis mode: the SIGNED component,
    so the caller compares it against a signed threshold and both directions are expressible.
    """
    delta = np.asarray(now, dtype=float) - np.asarray(start, dtype=float)
    if mode == "distance":
        return float(np.linalg.norm(delta))
    if mode == "planar":
        return float(np.linalg.norm(delta[:2]))
    try:
        return float(delta[AXES[mode]])
    except KeyError:
        raise ValueError(f"unknown displacement mode {mode!r}; known: {', '.join(MODES)}") from None


def satisfied(mode: str, threshold: float, measured: float) -> bool:
    """Has *measured* reached *threshold*?

    Magnitude modes compare upward. An axis mode compares in the direction the threshold's sign
    names -- which is what makes one parameter express "risen 5 cm" and "fell 5 cm" without a second
    one that could contradict it.
    """
    if mode in MAGNITUDE_MODES:
        return measured >= threshold
    return measured <= threshold if threshold < 0 else measured >= threshold


def rotation_angle(start: np.ndarray, now: np.ndarray) -> float:
    """The geodesic angle between two quaternions, in radians, in ``[0, pi]``.

    The one honest scalar for "how far has it turned": the angle of the single rotation that takes one
    orientation to the other, whatever axis that is about. Not roll/pitch/yaw differences, which are
    three numbers that do not compose and which gimbal-lock; and not a quaternion component, which is
    not an angle.

    ``(w, x, y, z)`` order, as everything here uses. ``|<q0, q1>|`` because ``q`` and ``-q`` are the
    same orientation -- without the absolute value a turn past 180 deg reads as a turn back.
    """
    q0 = _unit(np.asarray(start, dtype=float))
    q1 = _unit(np.asarray(now, dtype=float))
    dot = float(np.clip(abs(float(np.dot(q0, q1))), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _unit(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        # A zero quaternion is not an orientation. Identity is the only non-lying answer, and it
        # cannot arise from a compiled model -- MuJoCo normalises xquat.
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm
