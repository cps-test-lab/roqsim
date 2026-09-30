"""Describe one endpoint of a running simulation: its doc, payload and an example call.

See :mod:`roqsim.control_cli` for how the simulator is found and how values are printed.
"""

from __future__ import annotations

from . import run


def main(argv: list | None = None) -> int:
    return run("describe", argv, __doc__)
