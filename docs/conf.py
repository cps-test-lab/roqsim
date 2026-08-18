"""Sphinx configuration for the roqsim user documentation."""

import os
import sys

# Local doc-generation extensions (docs/_ext): they introspect the installed roqsim* packages at
# build time to generate the plugin/model/world/asset catalogs, so the docs can't drift from what is
# actually registered. See docs/_ext/_render.py.
sys.path.insert(0, os.path.abspath("_ext"))

project = "roqsim"
copyright = "2026, Frederik Pasch"
author = "Frederik Pasch"
release = "0.1.0"

extensions = ["plugin_docs", "model_docs", "world_docs", "asset_docs"]
templates_path = []

# All docs are reStructuredText (the architecture doc is architecture.rst), so the built site is
# self-contained with no extra parsers.
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

try:  # modern, styled theme (installed by `make venv`)
    import furo  # noqa: F401

    html_theme = "furo"
except Exception:  # pragma: no cover
    html_theme = "alabaster"

html_static_path = ["_static"]

# Brand artwork lives in docs/_static, not in the roqsim package: only splash.jpg is loaded at
# runtime, so the wheel has no reason to carry the logos. See docs/_static/CREDITS.txt.
html_logo = "_static/roqsim-transparent.png"
html_favicon = "_static/favicon.ico"
