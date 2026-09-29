"""Send a command (waiting for its outcome) or a stream value to a running simulation.

See :mod:`roqsim.control_cli` for how the simulator is found and how values are printed.
"""

from __future__ import annotations

from . import run


def main(argv: list | None = None) -> int:
    return run("call", argv, __doc__)
