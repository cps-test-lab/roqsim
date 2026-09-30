"""Print the values a running simulation publishes under a path prefix, as they come.

See :mod:`roqsim.control_cli` for how the simulator is found and how values are printed.
"""

from __future__ import annotations

from . import run


def main(argv: list | None = None) -> int:
    return run("sub", argv, __doc__)
