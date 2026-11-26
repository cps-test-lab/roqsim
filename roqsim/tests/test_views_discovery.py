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
