# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``.. roqsim-view-presets::`` -- the table of a views file's presets, from the installed
:data:`roqsim.views.PRESETS`, so the docs cannot drift from what ``roqsim render --views`` reads."""

from __future__ import annotations

from _render import parse_rst
from docutils.parsers.rst import Directive

from roqsim.views import presets_rst


class ViewPresetsDirective(Directive):
    has_content = False

    def run(self):
        return parse_rst(self, presets_rst())


def setup(app):
    app.add_directive("roqsim-view-presets", ViewPresetsDirective)
    return {"version": "0.1", "parallel_read_safe": True}
