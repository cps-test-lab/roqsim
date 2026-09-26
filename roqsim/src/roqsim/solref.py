"""The ``solref`` floor: the stiffest contact time constant MuJoCo will actually use.

With ``refsafe`` enabled (MuJoCo's default) a contact whose ``solref`` time constant is shorter
than the step can resolve is not refused and not reported: MuJoCo raises it to a floor before the
solver sees it. The model compiles, the run finishes, and the contact behaves as though the floor
had been stated while the world's own configuration still reads the tighter value. Every place
roqsim judges a ``solref`` against the timestep asks :func:`solref_floor`, so the refusal of
``sim.contact_override``, the ``flex-solref`` warning of ``roqsim check`` and the interpenetration
tolerance cannot disagree about where it lies.

**Where the floor is depends on the integrator**, because MuJoCo imposes it in two different ways:

* Under ``euler``, ``rk4``, ``implicit`` and ``implicitfast`` the time constant itself is raised to
  ``2 * timestep``, for every constraint row, whatever the damping ratio and ``solimp``.
* Under ``discrete`` the time constant is left alone and a contact (or limit) row whose spring the
  step cannot resolve -- ``timestep**2 * K * I > 1``, with ``K = 1 / (dmax**2 * timeconst**2 *
  dampratio**2)`` MuJoCo's reference stiffness and ``I`` the impedance at the contact's current
  depth -- has its stiffness lowered to ``1 / (timestep**2 * I)``, damping ratio kept. That is a
  time constant of ``timestep * sqrt(I) / (dmax * dampratio)``. ``I`` runs from ``solimp[0]`` at
  the surface to ``solimp[1]`` at the ``solimp`` width, so the floor stated here is the one at the
  larger of the two: a time constant at or above it runs as stated at every depth. At the default
  ``solimp`` (0.9, 0.95) and a damping ratio of 1 it is ``timestep / sqrt(0.95)``, about 1.026
  steps; it halves when the damping ratio doubles.

With ``refsafe`` disabled there is no floor. A ``solref`` whose entries are not both positive is
MuJoCo's direct ``(-stiffness, -damping)`` form, which has none either; callers pass only the
standard form. Measured on MuJoCo 3.14.0 by reading the stiffness the solver uses (``efc_KBIP``)
and pinned by ``tests/test_solref_floor.py``.

This is not a flex rule -- it applies to every contact -- which is why it lives here rather than
beside the flex analysis that also reads it (:mod:`roqsim.flex_modes`).
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import mujoco
import numpy as np

#: MuJoCo's impedance bounds (``mjMINIMP`` / ``mjMAXIMP``); it clamps ``solimp[0:2]`` to them.
_MIN_IMP = 0.0001
_MAX_IMP = 0.9999


def integrator_name(option) -> str:
    """The integrator of *option* by the name ``sim.integrator`` uses (``implicitfast``, ``discrete``)."""
    return mujoco.mjtIntegrator(int(option.integrator)).name.removeprefix("mjINT_").lower()


def solref_floor(option, solref, solimp):
    """The time constant below which MuJoCo raises *solref*'s, ``None`` with ``refsafe`` disabled.

    *option* is a compiled model's ``model.opt`` or a spec's ``spec.option`` before compile: the
    floor reads its timestep, integrator and ``refsafe`` flag. *solref* is a standard-form
    ``(timeconst, dampratio)`` and *solimp* the contact's ``(dmin, dmax, width, midpoint, power)``;
    only ``discrete`` reads them. Both may be arrays of such rows, ``(n, 2)`` and ``(n, 5)``, and the
    floor is then one per row. See the module docstring for the rule.
    """
    if int(option.disableflags) & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        return None
    timestep = float(option.timestep)
    solref = np.asarray(solref, dtype=float)
    solimp = np.asarray(solimp, dtype=float)
    if int(option.integrator) != mujoco.mjtIntegrator.mjINT_DISCRETE:
        floor = np.full(solref.shape[:-1], 2.0 * timestep)
    else:
        dmin = np.clip(solimp[..., 0], _MIN_IMP, _MAX_IMP)
        dmax = np.clip(solimp[..., 1], _MIN_IMP, _MAX_IMP)
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = timestep * np.sqrt(np.maximum(dmin, dmax)) / (dmax * solref[..., 1])
    return float(floor) if floor.ndim == 0 else floor


def floor_rule(option) -> str:
    """How :func:`solref_floor` is set under *option*'s integrator, for a message that names it."""
    if int(option.integrator) != mujoco.mjtIntegrator.mjINT_DISCRETE:
        return f"2 * timestep under {integrator_name(option)}"
    return "timestep * sqrt(max(solimp[0], solimp[1])) / (solimp[1] * dampratio) under discrete"


def floor_text(floor: float) -> str:
    """*floor* to six significant digits, rounded up, so that a world stating the text clears it."""
    exact = Decimal(repr(floor))
    step = Decimal(1).scaleb(exact.adjusted() - 5)
    return f"{exact.quantize(step, rounding=ROUND_CEILING).normalize():f}"
