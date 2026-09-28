"""The rates a world can hold, and putting a requested rate on them.

A sample and a publication both land on a physics step, so the rates a world can hold are exactly
``physics_rate / k`` for integer ``k >= 1``. A request between two of them is served at one of them
for the whole run -- a constant that is not the requested one -- so a caller that means a rate snaps
it here and says which one it got: :class:`roqsim.capture.StateRecorder` for a recording,
:meth:`roqsim.bridge.BridgeBase._rate_gate` for a published endpoint.

Arithmetic over exact rationals and nothing else: no numpy, no mujoco, no recording. That is what
lets the transport half of the substrate (``roqsim.bridge``, and every out-of-tree backend built on
it) reach the same grid as the recording half without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

#: Denominator bound for recovering a timestep's intended exact value from its float. A world writes
#: ``timestep: 0.002``, which is not exactly representable; ``Fraction(0.002)`` is a 60-digit monster
#: whereas ``limit_denominator(1e9)`` is exactly ``1/500``. Verified to recover the intent for every
#: timestep in use here, including ``1/240``.
_DT_DENOM_LIMIT = 10**9

#: How far a snap may move the rate before it is worth saying so, and before it is worth a warning.
#: One set of bands for every caller on this grid: two would make the same move loud where it hits a
#: recording and silent where it hits a topic.
SNAP_QUIET = 0.001  # 0.1%: the caller got what they asked for
SNAP_NOTABLE = 0.01  # 1%: above this, name the neighbours


class RateError(ValueError):
    """A rate that cannot exist in this world (see the message)."""


def parse_rate(value: str | int | float | Fraction) -> Fraction:
    """Parse a rate as an exact :class:`~fractions.Fraction`: ``25``, ``29.412``, ``500/17``, ``1/3``.

    Accepting a fraction literally is what makes every rate this module *prints* re-enterable -- a
    snap message suggests rates like ``500/17``, and a suggestion you cannot type back is not a
    suggestion. It also removes the only reason to write a repeating decimal by hand.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(_DT_DENOM_LIMIT)
    try:
        return Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError) as err:
        raise RateError(
            f"{value!r}: expected a number or a fraction, e.g. 25, 29.412, 500/17, 1/3"
        ) from err


def physics_rate(dt: float) -> Fraction:
    """A world's step rate as an exact rational, recovered from its float timestep."""
    if dt <= 0:
        raise RateError(f"timestep must be positive, got {dt!r}")
    return 1 / Fraction(dt).limit_denominator(_DT_DENOM_LIMIT)


@dataclass(frozen=True)
class GridRate:
    """A rate that exists in this world: once per ``every`` steps, i.e. exactly ``hz`` per sim second."""

    hz: Fraction  # the effective rate -- what is declared to ffmpeg, published at, and recorded
    every: int  # k: once per this many physics steps
    requested: Fraction  # what the caller asked for, kept so the report can compare
    physics: Fraction  # the world's step rate, kept for the message

    @property
    def deviation(self) -> float:
        """How far the snap moved the rate, as a fraction of the request."""
        return abs(float(self.hz - self.requested) / float(self.requested))

    @property
    def period(self) -> float:
        """Seconds of *simulated* time between firings."""
        return float(1 / self.hz)

    def rational(self) -> str:
        """The rate as an exact rational, e.g. ``500/17``.

        Never a rounded decimal: ``29.41`` given to anything that multiplies it up -- ffmpeg's ``-r``,
        a reader deriving sample times -- drifts, which is the whole defect this module exists to
        avoid.
        """
        return f"{self.hz.numerator}/{self.hz.denominator}"

    def neighbours(self, count: int = 3) -> list[GridRate]:
        """Achievable rates either side of this one, nearest first -- the suggestions in a report."""
        out: list[GridRate] = []
        for offset in _spiral(count):
            k = self.every + offset
            if k >= 1 and k != self.every:
                out.append(type(self)(self.physics / k, k, self.requested, self.physics))
        return out[:count]

    def exact_step_rate(self) -> Fraction:
        """A step rate that would serve the REQUEST exactly: the nearest multiple of it to this one.

        The other half of what a report can offer. Moving the rate keeps the world and changes the
        number a result quotes; moving the timestep keeps the number and changes the world -- which is
        the one to reach for when the requested rate is the paper's and not ours.
        """
        return self.requested * self.every


def _spiral(count: int):
    """Step offsets nearest-first: 1, -1, 2, -2, ... so suggestions stay close to what was asked."""
    for i in range(1, count + 2):
        yield i
        yield -i


def snap_rate(rate: str | int | float | Fraction, dt: float) -> GridRate:
    """The nearest rate on this world's grid, refusing nothing that is positive.

    NEAREST, not rounded down: the snapped rate may be the faster neighbour, so a caller that meant
    its request as a ceiling has to say so itself. How far the move is worth announcing is the
    caller's too, in :data:`SNAP_QUIET` and :data:`SNAP_NOTABLE` bands.

    A request faster than the world steps gets the step rate -- the fastest rate that exists here --
    rather than an error, because a caller with no flag to hand back to a person (a bridge binding an
    endpoint) still has to go on with some rate. :func:`roqsim.capture.snap_fps` is this plus the two
    refusals that a typed flag can afford.
    """
    requested = parse_rate(rate)
    if requested <= 0:
        raise RateError(f"a rate must be positive, got {float(requested):g}")
    physics = physics_rate(dt)
    every = max(1, round(float(physics / requested)))
    return GridRate(physics / every, every, requested, physics)
