# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""``roqsim splat`` -- the tools of the Gaussian-splat renderer."""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("splat")
def splat() -> None:
    """Place a captured room (a 3DGS splat) in a world drawn by the splat renderer."""


splat.add_command(tool("roqsim_render_splat.frame", "frame"))
splat.add_command(tool("roqsim_render_splat.collision", "collision"))
