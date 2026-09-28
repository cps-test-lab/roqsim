# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim export urdf/srdf/mesh/moveit`` take ``--set``/``--override`` for a world, and refuse the
world-only options for a bare MJCF or a model rather than ignoring them."""

from __future__ import annotations

import pytest

from roqsim import exit_status, export_mesh, export_moveit, export_srdf, export_urdf
from roqsim.engine import Engine

_MJCF = "<mujoco><worldbody><body name='box'><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
_WORLD = """
sim: {timestep: 0.002}
components:
  - spawn_model: {model: box.xml, motion: static}
    name: box
"""
#: Arguments each exporter requires besides its source and --out.
_REQUIRED = {
    export_urdf: [],
    export_srdf: [
        "--urdf",
        "robot.urdf",
        "--name",
        "robot",
        "--arm-base",
        "a",
        "--arm-tip",
        "b",
        "--gripper-joint",
        "g",
        "--gripper-open",
        "0",
        "--gripper-close",
        "1",
    ],
    export_mesh: [],
    export_moveit: [],
}
_OUT = {export_urdf: "r.urdf", export_srdf: "r.srdf", export_mesh: "r.stl", export_moveit: "cfg"}


class _Compiled(Exception):
    """Raised once the world is compiled: what is exported from it is not under test here."""


@pytest.fixture
def compiled_timestep(monkeypatch):
    """The ``opt.timestep`` of the world the command compiles; the command stops there."""
    seen = []
    setup = Engine.setup

    def spy(self, *a, **kw):
        setup(self, *a, **kw)
        seen.append(self.ctx.model.opt.timestep)
        raise _Compiled

    monkeypatch.setattr(Engine, "setup", spy)
    return seen


@pytest.mark.parametrize("module", list(_REQUIRED), ids=lambda m: m.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize("form", ["set", "override"])
def test_an_override_reaches_the_compiled_world(tmp_path, compiled_timestep, module, form):
    (tmp_path / "box.xml").write_text(_MJCF, encoding="utf-8")
    world = tmp_path / "world.yaml"
    world.write_text(_WORLD, encoding="utf-8")
    if form == "set":
        flags = ["--set", "sim.timestep=0.0005"]
    else:
        (tmp_path / "run.yaml").write_text("sim: {timestep: 0.0005}\n", encoding="utf-8")
        flags = ["--override", str(tmp_path / "run.yaml")]
    argv = ["--world", str(world), "--out", str(tmp_path / _OUT[module]), *_REQUIRED[module]]
    with pytest.raises(_Compiled):
        module.main([*argv, *flags])
    assert compiled_timestep == [0.0005]


_WORLD_ONLY = [
    ["--set", "sim.timestep=0.001"],
    ["--override", "run.yaml"],
    ["--skip-plugins", "box"],
]
_REFUSING = [
    (export_urdf, "--mjcf"),
    (export_srdf, "--mjcf"),
    (export_mesh, "--mjcf"),
    (export_mesh, "--model"),
    (export_moveit, "--mjcf"),
]


@pytest.mark.parametrize(
    "module,source", _REFUSING, ids=[f"{m.__name__.rsplit('.', 1)[-1]}{s}" for m, s in _REFUSING]
)
@pytest.mark.parametrize("flags", _WORLD_ONLY, ids=lambda f: f[0])
def test_a_world_option_without_a_world_is_refused_by_name(tmp_path, capsys, module, source, flags):
    scene = tmp_path / "scene.xml"
    scene.write_text(_MJCF, encoding="utf-8")
    out = tmp_path / _OUT[module]
    argv = [source, str(scene), "--out", str(out), *_REQUIRED[module], *flags]
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv)
    assert exit_info.value.code == exit_status.BAD_INPUT
    err = capsys.readouterr().err
    assert f"{flags[0]} acts on a world YAML, and {source}" in err
    assert not out.exists(), "nothing is exported"
