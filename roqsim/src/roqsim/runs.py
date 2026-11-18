# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A run's identity: which execution of a world, between two resets, this is.

A run is one execution of a world from a reset to the next. Its identity is a string the runner
gives it -- a campaign runner names the configuration and the run number, ``"cfg-a/3"`` -- and the
random draws are keyed on it (:meth:`roqsim.context.SimContext.rng_for`), so two runs never draw
the same values whichever process they share, and the same seed and run draw the same values in
any process, whatever ran before.

A reset that names no run gets a name all the same, from :class:`RunNames`: ``"#0"``, ``"#1"``, ...
in a process that never named one, and ``"X#1"``, ``"X#2"``, ... after a reset named ``"X"``. The
unnamed runs of a process have names of their own, and every name can be given back to run it
again: ``roqsim sim --seed S --run '#2'`` draws what the third unnamed run under seed ``S`` drew.
Giving a name twice runs one run twice, with the same draws, which is what giving it means.

The names are the **driver's**, not an engine's: a driver that builds a new engine for a new world
hands the new engine the :class:`RunNames` it already holds, so the unnamed runs of a process keep
counting across the rebuild.
"""

from __future__ import annotations

import hashlib

#: The environment variable a deployment names a process's first run with, where nothing passes a
#: name to the reset: the scenario adapter and ``roqsim sim`` read it.
RUN_ENV = "ROQSIM_RUN"

#: What separates a named run from the unnamed ones after it.
SEPARATOR = "#"


def check_run(run) -> str:
    """``run`` as a run name, or a ``ValueError`` naming what is wrong with it."""
    if not isinstance(run, str):
        raise ValueError(f"a run is named by a string, not {type(run).__name__} {run!r}")
    if not run:
        raise ValueError("a run's name is empty: name it, or pass none for the next unnamed one")
    return run


def run_key(run: str) -> int:
    """The 64-bit number a run occupies in the draws' counter: a stable hash of its name.

    Not Python's ``hash()``, which is salted per process and would make a draw irreproducible
    across processes -- the property this key exists for.
    """
    return int.from_bytes(hashlib.blake2b(run.encode(), digest_size=8).digest(), "little")


class RunNames:
    """The names of a driver's runs, one per reset.

    ``first`` names the first run when its reset names none (``--run``, ``ROQSIM_RUN``).
    """

    def __init__(self, first: str | None = None) -> None:
        self._first = None if first is None else check_run(first)
        self._base = ""
        self._unnamed = 0

    def next(self, run: str | None = None) -> str:
        """The name of the run a reset starts: ``run`` when given, else the next unnamed one."""
        if run is None and self._first is not None:
            run, self._first = self._first, None
        if run is not None:
            self._first = None
            self._base, self._unnamed = check_run(run), 1
            return run
        name = f"{self._base}{SEPARATOR}{self._unnamed}"
        self._unnamed += 1
        return name
