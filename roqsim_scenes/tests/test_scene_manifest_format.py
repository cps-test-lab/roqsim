"""A scene manifest states its format and version, and the bake refuses one it cannot read.

``scene.json`` is the stage-1 output every importer writes and the bake reads. It also shares its
file name with the web scene descriptor ``roqsim export web`` writes, so a manifest that says what it
is can be told from that file, and one written to a later contract is refused by name rather than
baked with the keys that happen to overlap. Absent means version 1, the layout every manifest
written before the stamp existed has.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from roqsim.textures import UVScaler
from roqsim_scenes import scene_manifest as sm
from roqsim_scenes.cli import floorplan_to_world as fw
from roqsim_scenes.cli import scene_to_mjcf


def _scene(tmp_path, **stamp):
    manifest = {
        **stamp,
        "name": "s",
        "bounds_min": [0.0, 0.0, 0.0],
        "bounds_max": [1.0, 1.0, 1.0],
        "objects": [],
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(manifest))
    return str(path)


def _bake(scene_json):
    return scene_to_mjcf.build_spec(scene_json, {}, [], UVScaler(prefix="t_"))


def test_the_writer_stamps_format_and_version():
    manifest = fw.scene_manifest("r", (0.0, 0.0, 4.0, 3.0), 2.5, n_walls=1)
    assert (manifest["format"], manifest["version"]) == (sm.FORMAT, sm.FORMAT_VERSION)


def test_an_unstamped_manifest_bakes(tmp_path):
    assert _bake(_scene(tmp_path)).modelname == "s"


def test_the_current_version_bakes(tmp_path):
    assert _bake(_scene(tmp_path, format=sm.FORMAT, version=sm.FORMAT_VERSION)).modelname == "s"


def test_a_newer_version_is_refused_naming_both(tmp_path):
    scene = _scene(tmp_path, format=sm.FORMAT, version=sm.FORMAT_VERSION + 1)
    with pytest.raises(
        ValueError, match=rf"version {sm.FORMAT_VERSION + 1}.*up to {sm.FORMAT_VERSION}"
    ):
        _bake(scene)


def test_another_format_is_refused_naming_it(tmp_path):
    """The web descriptor shares the file name; baking it would fail far from the cause."""
    scene = _scene(tmp_path, format="roqsim.web_scene", version=1)
    with pytest.raises(ValueError, match=r"'roqsim\.web_scene'.*roqsim_scenes\.scene_manifest"):
        _bake(scene)


def test_the_blender_writer_stamps_the_same_pair():
    """usd_to_scene runs under Blender's Python, which cannot import this package, so it writes the
    pair literally; this keeps its literals equal to the shared constants."""
    src = (Path(scene_to_mjcf.__file__).with_name("usd_to_scene.py")).read_text()
    assert re.search(rf'"format": "{re.escape(sm.FORMAT)}"', src)
    assert re.search(rf'"version": {sm.FORMAT_VERSION}\b', src)
