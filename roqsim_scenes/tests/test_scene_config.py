"""A bake config is read by name, so a key nothing reads is refused rather than dropped.

``ground_z: 0.1`` misspelt as ``groud_z`` would otherwise bake a floor at the scene's lowest point;
``physical_size`` misspelt in a material would tile a texture at the default size. Both files look
applied while the geometry says something else, which is the failure an allowlist exists to remove.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from roqsim_scenes.cli import scene_to_mjcf


def test_a_misspelt_top_level_key_is_refused_with_the_nearest_named():
    with pytest.raises(
        ValueError, match=r"unknown key\(s\) 'groud_z' \(did you mean 'ground_z'\?\)"
    ):
        scene_to_mjcf.check_config({"groud_z": 0.1}, "scene.yaml")


def test_a_far_off_key_is_refused_without_a_guess():
    with pytest.raises(ValueError, match=r"unknown key\(s\) 'banana'; it takes collision, floor"):
        scene_to_mjcf.check_config({"banana": 1}, "scene.yaml")


def test_a_material_entry_key_is_refused_where_it_sits():
    config = {"materials": [{"match": "Wall_*", "rgba": [1, 1, 1, 1], "physcal_size": 2}]}
    with pytest.raises(ValueError, match=r"materials\[0\]: unknown key\(s\) 'physcal_size'"):
        scene_to_mjcf.check_config(config, "scene.yaml")


def test_floor_and_light_blocks_are_checked():
    with pytest.raises(ValueError, match=r"light: unknown key\(s\) 'hieght'"):
        scene_to_mjcf.check_config({"light": {"hieght": 3}}, "scene.yaml")
    with pytest.raises(ValueError, match=r"floor: unknown key\(s\) 'colour'"):
        scene_to_mjcf.check_config({"floor": {"colour": [1, 1, 1]}}, "scene.yaml")


def test_the_shared_look_passes():
    shared = Path(scene_to_mjcf.__file__).with_name("floorplan.scene.yaml")
    scene_to_mjcf.check_config(yaml.safe_load(shared.read_text()), str(shared))


def test_loading_a_file_checks_it(tmp_path):
    (tmp_path / "scene.json").write_text("{}")
    (tmp_path / "scene.yaml").write_text("collison: none\n")
    with pytest.raises(ValueError, match=r"did you mean 'collision'"):
        scene_to_mjcf._load_config(str(tmp_path / "scene.json"), None)
