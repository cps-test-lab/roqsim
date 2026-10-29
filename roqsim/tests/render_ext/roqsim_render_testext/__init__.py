# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer from a package roqsim does not know: ``test_render_extension.py`` puts it on the path."""

from __future__ import annotations

import numpy as np

from roqsim.render_api import Frame, Renderer, RendererCapabilities, Setting


class TestExtRenderer(Renderer):
    """Fills each view with the colour its settings name, its first row encoding the snapshot step."""

    capabilities = RendererCapabilities(
        images=frozenset({"rgb", "depth"}),
        camera_models=frozenset({"pinhole", "free"}),
        batched=True,
        deterministic=True,
    )
    SETTINGS = {"colour": Setting([200, 100, 50], "the colour every pixel gets")}

    def open(self, scene):
        self.cameras = [c.id for c in scene.cameras]

    def render(self, snapshot, requests):
        out = []
        for r in requests:
            h, w = r.camera.height, r.camera.width
            rgb = np.empty((h, w, 3), np.uint8)
            rgb[:] = self.settings["colour"]
            depth = np.full((h, w), 2.0, np.float32) if "depth" in r.images else None
            out.append(Frame(r.id, snapshot.sim_time, snapshot.step, rgb=rgb, depth=depth))
        return out
