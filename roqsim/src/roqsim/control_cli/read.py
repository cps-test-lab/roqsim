"""Read an out endpoint of a running simulation, once.

See :mod:`roqsim.control_cli` for how the simulator is found and how values are printed.
"""

from __future__ import annotations

from . import run


def main(argv: list | None = None) -> int:
    return run("read", argv, __doc__)
