# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim export web --mjcf`` refuses ``--set``/``--override`` rather than exporting without them."""

from __future__ import annotations

import pytest

from roqsim import export_web

_MJCF = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"


@pytest.mark.parametrize(
    "flags", [["--set", "sim.timestep=0.001"], ["--override", "run.overrides.yaml"]]
)
def test_an_override_with_a_bare_mjcf_is_bad_input(tmp_path, capsys, flags):
    scene = tmp_path / "scene.xml"
    scene.write_text(_MJCF, encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exit_info:
        export_web.main(["--mjcf", str(scene), "--out", str(out), *flags])
    assert exit_info.value.code == 2
    assert "--mjcf compiles a bare MJCF" in capsys.readouterr().err
    assert not out.exists(), "nothing is exported"


def test_a_bare_mjcf_without_overrides_still_exports(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(_MJCF, encoding="utf-8")
    out = tmp_path / "out"
    assert export_web.main(["--mjcf", str(scene), "--out", str(out)]) == 0
    assert (out / "scene.json").exists()
