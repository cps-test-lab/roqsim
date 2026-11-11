# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The configuration a user writes, and every mistake said in their terms. No GPU needed."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import roqsim_render_splat
from roqsim.config import load_config, load_config_from_dict
from roqsim.plugin import PluginError
from roqsim.renderers import describe_renderer, list_renderers
from roqsim_render_splat.renderer import SplatRenderer

EXAMPLE = Path(roqsim_render_splat.WORLDS_DIR) / "splat_room_example.yaml"
EXAMPLE_PLY = Path(roqsim_render_splat.WORLDS_DIR) / "splat_room_example" / "room.ply"


def _world(splat: dict | None = None, **settings) -> dict:
    block = {"splat": splat if splat is not None else {"file": "room.ply"}, **settings}
    return {"sim": {"renderer": "splat", "renderers": {"splat": block}}, "components": []}


@pytest.fixture
def room(tmp_path) -> Path:
    shutil.copy(EXAMPLE_PLY, tmp_path / "room.ply")
    return tmp_path


def test_the_example_world_loads_as_shipped():
    cfg = load_config(EXAMPLE)
    assert cfg.renderer == "splat"
    block = cfg.renderers["splat"]["splat"]
    assert (
        block["file"] == "roqsim_render_splat:splat_room_example/room.ply" and block["scale"] == 0.5
    )


def test_a_world_that_extends_the_example_elsewhere_finds_its_capture(tmp_path):
    """The example names its capture by package, so a world in another directory inherits it."""
    child = tmp_path / "mine" / "room.yaml"
    child.parent.mkdir()
    child.write_text("extends: roqsim_render_splat:splat_room_example\n")
    block = load_config(child).renderers["splat"]["splat"]  # loading checks the file is there
    assert block["file"] == "roqsim_render_splat:splat_room_example/room.ply"


def test_the_minimum_a_world_states_is_the_file(room):
    load_config_from_dict(_world({"file": "room.ply"}), base_dir=room)


def test_a_file_from_a_package_resolves_by_reference(tmp_path):
    load_config_from_dict(
        _world({"file": "roqsim_render_splat:splat_room_example/room.ply"}), base_dir=tmp_path
    )


def test_an_unknown_renderer_names_the_installed_ones_and_the_package_that_adds_it():
    with pytest.raises(PluginError) as err:
        load_config_from_dict({"sim": {"renderer": "splats"}, "components": []})
    message = str(err.value)
    assert "unknown renderer 'splats'" in message
    # Every renderer installed is named; which others are installed is not this package's to say.
    installed = message.split("Installed: ", 1)[1].split(". ", 1)[0].split(", ")
    assert {"mujoco", "splat"} <= set(installed)
    assert "roqsim_render_splats" in message


def test_a_mistyped_key_names_the_key_the_block_and_the_valid_ones(room):
    with pytest.raises(PluginError) as err:
        load_config_from_dict(_world(sh_degre=2), base_dir=room)
    message = str(err.value)
    assert "sim.renderers.splat: unknown key 'sh_degre'" in message
    assert "valid keys: background, device, max_pass_mib, min_alpha, min_opacity" in message
    assert "did you mean 'sh_degree'?" in message


def test_a_mistyped_key_in_the_splat_block_is_named_with_it(room):
    with pytest.raises(
        PluginError, match=r"unknown key 'scael' in splat; valid keys: file, pose, scale"
    ):
        load_config_from_dict(_world({"file": "room.ply", "scael": 2}), base_dir=room)


def test_a_mistyped_key_in_the_pose_is_named_with_it(room):
    with pytest.raises(
        PluginError,
        match=r"unknown key 'rotation' in splat.pose; valid keys: orientation, position",
    ):
        load_config_from_dict(
            _world({"file": "room.ply", "pose": {"rotation": {"yaw": 1}}}), base_dir=room
        )
    with pytest.raises(PluginError, match=r"'splat.pose.orientation' has no key\(s\) \['yaww'\]"):
        load_config_from_dict(
            _world({"file": "room.ply", "pose": {"orientation": {"yaww": 1}}}), base_dir=room
        )


def test_a_missing_file_names_the_path_it_resolved(tmp_path):
    with pytest.raises(PluginError) as err:
        load_config_from_dict(_world({"file": "rooms/lab.ply"}), base_dir=tmp_path)
    assert f"'splat.file': splat file not found: {tmp_path / 'rooms' / 'lab.ply'}" in str(err.value)


def test_a_mesh_ply_is_said_to_be_a_mesh_not_a_splat(tmp_path):
    (tmp_path / "room.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\n"
        b"property float y\nproperty float z\nelement face 1\n"
        b"property list uchar int vertex_indices\nend_header\n" + bytes(3 * 12 + 13)
    )
    with pytest.raises(PluginError, match=r"holds vertex and face; .* \(this looks like a mesh\)"):
        load_config_from_dict(_world(), base_dir=tmp_path)


def test_a_point_cloud_ply_names_the_splat_fields_it_lacks(tmp_path):
    (tmp_path / "room.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\n"
        b"property float y\nproperty float z\nproperty uchar red\nend_header\n" + bytes(13)
    )
    with pytest.raises(PluginError) as err:
        load_config_from_dict(_world(), base_dir=tmp_path)
    assert "is not a 3DGS splat: its vertices lack f_dc_0, f_dc_1, f_dc_2, opacity" in str(
        err.value
    )


def test_an_ascii_ply_is_refused_naming_its_format(tmp_path):
    (tmp_path / "room.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(PluginError, match="a ascii PLY; a 3DGS splat is binary_little_endian"):
        load_config_from_dict(_world(), base_dir=tmp_path)


def test_a_world_without_its_capture_says_what_to_write(tmp_path):
    world = {"sim": {"renderer": "splat", "renderers": {"splat": {}}}, "components": []}
    with pytest.raises(PluginError, match=r"'splat' is required: .*roqsim splat frame"):
        load_config_from_dict(world, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"sh_degree": 4}, "'sh_degree' is 0, 1, 2, 3 or null, got 4"),
        ({"background": [0, 0]}, "'background' is three numbers 0-255"),
        ({"min_opacity": 2}, "'min_opacity' is a number from 0 to 1, got 2"),
        ({"device": "cpu"}, "'device' is a CUDA device"),
        ({"max_pass_mib": 0}, "'max_pass_mib' is a number > 0, got 0"),
    ],
)
def test_a_wrong_value_names_what_it_takes(room, settings, message):
    with pytest.raises(PluginError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_config_from_dict(_world(**settings), base_dir=room)


def test_an_unset_scale_is_refused(room):
    with pytest.raises(PluginError, match="'splat.scale' is metres per unit of the capture"):
        load_config_from_dict(_world({"file": "room.ply", "scale": None}), base_dir=room)


def test_the_renderer_is_listed_with_every_setting_its_default_and_meaning():
    names = [r["name"] for r in list_renderers()["renderers"]]
    assert "splat" in names
    described = describe_renderer("splat")
    assert described["package"] == "roqsim_render_splat"
    assert set(described["settings"]) == set(SplatRenderer.SETTINGS)
    for key, setting in described["settings"].items():
        assert setting["meaning"], key
        assert setting["default"] == SplatRenderer.SETTINGS[key].default
    caps = described["capabilities"]
    assert caps["images"] == ["depth", "rgb"] and caps["draws_static"] is False


def test_the_example_world_s_comments_name_the_tool_that_proposes_the_pose():
    text = EXAMPLE.read_text()
    assert "roqsim splat frame" in text
    yaml.safe_load(text)  # and it is YAML a user can copy


def test_the_shipped_example_room_is_what_the_generator_makes(tmp_path):
    """The example's ``room.ply`` and ``collision.xml`` are generated; this is how they were made."""
    from roqsim_render_splat.synthetic import write_shipped_example

    ply, xml = write_shipped_example(tmp_path)
    shipped = EXAMPLE_PLY.parent
    assert ply.read_bytes() == (shipped / "room.ply").read_bytes()
    assert xml.read_text() == (shipped / "collision.xml").read_text()
