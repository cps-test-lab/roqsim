"""List a running simulation's endpoints: path, kind, rate and what each is.

See :mod:`roqsim.control_cli` for how the simulator is found and how values are printed.
"""

from __future__ import annotations

from . import run


def main(argv: list | None = None) -> int:
    return run("endpoints", argv, __doc__)
