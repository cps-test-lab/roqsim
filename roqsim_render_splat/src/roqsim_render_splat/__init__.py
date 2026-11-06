# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A Gaussian-splat renderer for roqsim: a captured room as every camera sees it.

``sim.renderer: splat`` in a world; see :mod:`roqsim_render_splat.renderer`. Importing this package
imports neither torch nor gsplat: only a renderer that opens does.
"""

from pathlib import Path

#: The example worlds this package ships (``roqsim.worlds``): ``roqsim_render_splat:<world>``.
WORLDS_DIR = Path(__file__).parent / "worlds"
