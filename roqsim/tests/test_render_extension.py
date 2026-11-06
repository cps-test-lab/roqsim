# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer registered by another package is found and used with no edit to roqsim.

``render_ext/`` holds a package with an installed-distribution record (a ``.dist-info`` with an
``entry_points.txt``) registering ``testext`` under ``roqsim.renderers``. A fresh process with that
directory on its path is all it takes: the world names the renderer, the cameras draw through it,
and ``roqsim renderers list`` shows it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

EXT = Path(__file__).with_name("render_ext")


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(EXT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )


def test_roqsim_renderers_lists_a_renderer_from_another_package():
    out = _run(
        """
        import json
        from roqsim.renderers import describe_renderer, list_renderers
        print(json.dumps([list_renderers(), describe_renderer("testext")]))
        """
    )
    assert out.returncode == 0, out.stderr
    listed, described = json.loads(out.stdout)
    names = {r["name"]: r for r in listed["renderers"]}
    assert names["testext"]["package"] == "roqsim-render-testext"
    assert described["settings"]["colour"]["default"] == [200, 100, 50]
    assert described["capabilities"]["images"] == ["depth", "rgb"]


def test_a_world_naming_it_draws_its_cameras_through_it():
    out = _run(
        """
        import json
        import mujoco
        from roqsim.config import load_config_from_dict
        from roqsim.engine import Engine
        from roqsim.plugin import Plugin

        class Cam(Plugin):
            def build(self, spec, ctx):
                spec.worldbody.add_camera(name="c", pos=[1, 0, 1])

        import __main__
        __main__.Cam = Cam
        cfg = load_config_from_dict({
            "sim": {"renderer": "testext", "renderers": {"testext": {"colour": [1, 2, 3]}}},
            "components": [{"__main__:Cam": {}}],
        })
        engine = Engine(cfg)
        engine.ctx.seed = 1
        engine.setup()
        engine.reset()
        service = engine.ctx.render
        view = service.view("c", camera="c", width=8, height=6, images=("rgb", "depth"))
        (frame,) = service.render_now([(view, ("rgb", "depth"))], engine.ctx.data)
        print(json.dumps({
            "renderer": type(service.renderer()).__name__,
            "rgb": frame.rgb[0, 0].tolist(),
            "depth": float(frame.depth[0, 0]),
            "stamp": frame.sim_time,
        }))
        engine.shutdown()
        """
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {"renderer": "TestExtRenderer", "rgb": [1, 2, 3], "depth": 2.0, "stamp": 0.0}


def test_it_is_not_installed_in_this_process():
    """The extension is reached only through the path the test gives the child process: nothing in
    roqsim names it."""
    from roqsim.registry import renderer_names

    assert "testext" not in renderer_names()
