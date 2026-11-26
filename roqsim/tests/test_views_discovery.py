# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The conventions of a views file are found from the installation, without the docs: by a person or
an agent that only runs commands. Each part is made from :data:`roqsim.views.PRESETS`, so none can
drift from the others."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import mujoco
import numpy as np
import pytest
import yaml

from roqsim import views as V
from roqsim.views_cli import main


def _roqsim(*args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "roqsim.commands", *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=cwd,
        env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
    )


# -- (b) `roqsim views describe` -----------------------------------------------------------------


@pytest.mark.parametrize("preset", sorted(V.PRESETS))
def test_describe_gives_a_ready_example_file_for_every_preset(preset, tmp_path, capsys):
    assert main(["describe", preset]) == 0
    text = capsys.readouterr().out
    # The whole output is a views file: the conventions are comments.
    path = tmp_path / "views.yaml"
    path.write_text(text)
    assert yaml.safe_load(text)["convention"] == preset
    (view,) = V.load_views(path)
    # The example camera, whatever its convention: 2 m from the origin, 1 m up, looking along +x.
    reading = view.describe()
    assert reading["position"] == pytest.approx([-2.0, 0.0, 1.0])
    assert reading["toward"] == "+x" and reading["warnings"] == []
    assert reading["principal_offset_px"] == {"right": 0.0, "below": 0.0}
    # Each key the convention names is explained beside it.
    for key in V.KEYS:
        assert V.PRESETS[preset].keys()[key] in text


def test_the_example_files_draw(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".3 .3 .3" pos="0 0 1"'
        ' rgba="1 0 0 1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames = {}
    for preset in V.PRESETS:
        doc, _text = V.example_file(preset)
        (view,) = V.parse_views(doc)
        ((_, frames[preset]),) = list(V.render_views((model, data), [view]))
    ref = frames["mujoco"].rgb
    assert (ref[..., 0] > 100).mean() > 0.01  # the box, ahead
    for preset, frame in frames.items():
        assert np.array_equal(frame.rgb, ref), preset


def test_describe_as_json_holds_the_keys_the_presets_and_the_example(capsys):
    assert main(["describe", "opencv", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["keys"] == V.KEYS
    assert {n: {k: p[k] for k in V.KEYS} for n, p in out["presets"].items()} == {
        n: c.keys() for n, c in V.PRESETS.items()
    }
    from_data = V.parse_views(out["example"])[0]
    from_text = V.parse_views(yaml.safe_load(out["example_text"]))[0]
    assert from_data == from_text


def test_describe_without_a_preset_lists_every_preset_and_key(capsys):
    assert main(["describe"]) == 0
    text = capsys.readouterr().out
    for name, c in V.PRESETS.items():
        assert re.search(rf"# +{name} +{c.axes} / {c.rotation} / {c.pixel_origin} ", text), name
    for values in V.KEYS.values():
        for value in values:
            assert value in text
    assert yaml.safe_load(text) is None  # comments only


def test_describe_is_a_roqsim_command():
    out = _roqsim("views", "describe", "mujoco")
    assert out.returncode == 0 and "convention: mujoco" in out.stdout


# -- (d) an unknown preset -----------------------------------------------------------------------


def test_describe_refuses_an_unknown_preset_naming_the_presets_and_itself():
    out = _roqsim("views", "describe", "colmpa")
    assert out.returncode == 2
    for want in (*V.PRESETS, *V.KEYS, "did you mean 'colmap'", "roqsim views describe"):
        assert want in out.stderr


def test_the_docs_table_is_made_from_the_presets():
    rst = "\n".join(V.presets_rst())
    for name, c in V.PRESETS.items():
        assert f"``{name}``" in rst and c.source in rst


# -- (a) `roqsim render --help` ------------------------------------------------------------------


def test_render_help_names_every_preset_one_line_each(capsys):
    from roqsim import render

    with pytest.raises(SystemExit):
        render.main(["--help"])
    help_text = capsys.readouterr().out
    lines = help_text.splitlines()
    for name, c in V.PRESETS.items():
        mine = [ln for ln in lines if re.match(rf"\s+{name}\s+\S", ln)]
        assert len(mine) == 1, f"{name}: {mine}"
        assert c.summary() in mine[0]
    assert "roqsim views describe" in help_text


def test_a_preset_added_gets_its_help_line(monkeypatch, capsys):
    from roqsim import render

    monkeypatch.setitem(
        V.PRESETS, "blender", V.Convention("mujoco", "camera_to_world", "corner", "blender", "x")
    )
    with pytest.raises(SystemExit):
        render.main(["--help"])
    assert re.search(r"^\s+blender\s+x: ", capsys.readouterr().out, re.M)


# -- (c) `--check` says how it read the file -----------------------------------------------------

SCENE = """
<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="5 5 .1"/></worldbody></mujoco>
"""


def _check(tmp_path, convention, entry):
    from roqsim import render

    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)
    views = tmp_path / "views.json"
    views.write_text(json.dumps({"convention": convention, "views": [entry]}))
    return render.main([str(scene), "--views", str(views), "--out", str(tmp_path / "o"), "--check"])


def _colmap_entry(**override):
    doc, _ = V.example_file("colmap")
    return {**doc["views"][0], **override}


def test_check_says_per_view_where_it_stands_looks_and_what_is_up(tmp_path, capsys):
    assert _check(tmp_path, "colmap", _colmap_entry()) == 0
    out = capsys.readouterr()
    line = next(ln for ln in out.err.splitlines() if ln.startswith("view_0001"))
    assert "stands at x -2.000, y 0.000, z 1.000 m" in line
    assert "toward +x, level" in line
    assert "up is +0.00 +0.00 +1.00, toward +z, rolled 0°" in line
    assert "principal point 0.0 px right of and 0.0 px above the image centre" in line
    assert "WARNING" not in line
    (reading,) = json.loads(out.out)["reading"]
    assert reading["position"] == [-2.0, 0.0, 1.0] and reading["toward"] == "+x"


def test_check_shows_a_wrong_preset_before_anything_is_drawn(tmp_path, capsys):
    """The colmap example read with MuJoCo's axes: the camera looks backwards and stands on its
    head, and --check says so before anything is drawn."""
    entry = _colmap_entry()
    entry["position"] = [-2.0, 0.0, 1.0]
    del entry["t"]
    assert (
        _check(
            tmp_path,
            {"axes": "mujoco", "rotation": "world_to_camera", "pixel_origin": "corner"},
            entry,
        )
        == 0
    )
    line = next(ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("view_0001"))
    assert "toward -x" in line and "WARNING upside down" in line


def test_check_says_where_an_off_centre_principal_point_lies(tmp_path, capsys):
    assert _check(tmp_path, "colmap", _colmap_entry(cx=330.0, cy=230.0)) == 0
    line = next(ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("view_0001"))
    assert "principal point 10.0 px right of and 10.0 px above the image centre" in line


# -- (d) the refusal, through the command --------------------------------------------------------


def test_render_refuses_an_unknown_preset_naming_the_presets_and_the_command(tmp_path, capsys):
    assert _check(tmp_path, "colmpa", _colmap_entry()) == 2
    err = capsys.readouterr().err
    for want in (*V.PRESETS, *V.KEYS, "did you mean 'colmap'", "roqsim views describe"):
        assert want in err
