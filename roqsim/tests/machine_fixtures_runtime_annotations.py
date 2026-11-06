# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A machine in a module whose annotations are real types: no ``from __future__ import
annotations``, so ``float | None`` is a :class:`types.UnionType` rather than a string."""

from machine_fixtures import Press

from roqsim.machine import program


class TypedPress(Press):
    """Machine: the press, with a program whose parameter is a real union."""

    @program
    def nudge(self, depth: float | None = None):
        """Wait a step, with an optional depth."""
        yield
