"""The run's noise seed: one precedence, shared by every driver.

A seed is **driver-owned**. :class:`~roqsim.context.SimContext` only holds it and
:meth:`~roqsim.context.SimContext.rng_for` only keys on it; deciding *which* seed a run uses is a
driver's job, and it is the same decision for every driver. The decision lives here, in one place,
so that the standalone driver (``roqsim sim``) and an embedding one
(``roqsim.scenario_adapter``) cannot answer it differently. Anything that builds an
:class:`~roqsim.engine.Engine` and steps it is a driver and belongs on this path.

:func:`resolve_seed` is deliberately **pure**: it reads no environment and no config file. Where a
driver's explicit seed comes from -- a command-line flag, an environment variable set by whoever
deployed the run -- is that driver's business, and it passes the value in.
"""

from __future__ import annotations

import logging

#: Environment variable an embedding driver reads its explicit seed from, for the shape that has no
#: command line. Named here beside the precedence it feeds rather than at the point of use, the way
#: :mod:`roqsim.logging_setup` names its own.
SEED_ENV = "ROQSIM_SEED"


#: Seed for a **preview**: a settle-and-look path (a web export, the scene-builder window) that
#: builds and steps a world to show it rather than to measure it. Those are drivers too, so they
#: must resolve a seed like any other; a fixed one is right because a preview is not a trial, and
#: pinning it makes the picture reproducible. Never use it for a run that produces data.
PREVIEW_SEED = 0


class SeedError(RuntimeError):
    """Randomness was drawn from a run whose seed no driver resolved.

    Its own error class, like every other roqsim failure domain, so a driver can catch exactly this
    and a caller can tell it from a config or plugin fault.
    """


def resolve_seed(seed: int | None, logger: logging.Logger, config_seed: int | None = None) -> int:
    """The run's noise seed, by precedence: explicit > world config > drawn.

    An explicitly passed seed wins because it is the more specific instruction -- the
    world states what a run normally uses, the caller states what THIS run uses. With
    neither, one is drawn and announced, exactly as before ``sim.seed`` existed.

    Always returns an ``int``, never ``None``: a driver assigns the result to
    :attr:`~roqsim.context.SimContext.seed`, and an unset seed is what
    :meth:`~roqsim.context.SimContext.rng_for` now refuses to draw from.
    """
    if seed is not None:
        logger.info("seed: %d (given)", seed)
        return int(seed)
    if config_seed is not None:
        logger.info("seed: %d (from sim.seed)", config_seed)
        return int(config_seed)
    import secrets

    drawn = secrets.randbelow(2**31)
    # Names both channels because both reproduce this run from either driver: `sim.seed` in the world
    # travels with it, and `roqsim sim <world> --seed N` is how a recorded run is replayed by hand.
    logger.info(
        "seed: %d (drawn -- pass --seed %d, or set sim.seed, to repeat this run)", drawn, drawn
    )
    return drawn
