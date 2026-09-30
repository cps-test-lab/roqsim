# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Reaching a MODEL DEFAULT component's config from outside the document.

``spawn_robot`` pulls a model's components in from its manifest, so a world that just spawns a robot
never names its lidar, and an override must still reach it. Resolved against the parsed YAML, before
expansion, ``plugins.lidar.rays`` would name nothing and be refused. The refusal is right --
silently ignoring a swept parameter lets a campaign look healthy while changing nothing -- but it
would leave a model default unreachable except through a stub entry whose only job is to exist.

Expansion happens while the document loads, so an override resolves against what will actually
run. This is the file that says so.
"""

import pytest

from roqsim.config import PluginError, load_config_from_dict, overrides_from_dotlist

pytest.importorskip("roqsim_mobile", reason="turtlebot4 manifest lives in roqsim_mobile")

BARE = {
    "sim": {"world": "empty_room"},
    "components": [{"spawn_robot": {"model": "turtlebot4"}, "name": "robot"}],
}


def _components(overrides=None):
    return {s.address: s.config for s in load_config_from_dict(BARE, overrides=overrides).plugins}


def test_a_model_default_is_addressable_with_nothing_declared():
    """The headline: no stub, no entry, and the lidar the manifest supplies is reachable."""
    lidar = _components({"components": {"robot.rplidar.lidar": {"range_stddev": 0.05}}})["robot.rplidar.lidar"]
    assert lidar["range_stddev"] == 0.05


def test_the_manifest_still_supplies_everything_the_override_did_not_name():
    """An override is partial, like a declaration: it sets keys, it does not replace a component."""
    lidar = _components({"components": {"robot.rplidar.lidar": {"range_stddev": 0.05}}})["robot.rplidar.lidar"]
    assert lidar["rays"] == 360  # the turtlebot4 manifest's override: its datasheet's 1 deg resolution
    assert lidar["max_range"] == 12.0
    assert lidar["frame_id"] == "rplidar_link"


def test_the_dotlist_spelling_means_the_same_thing():
    """`--set` and an override document are two spellings of one assignment; a campaign writes the
    first and a saved override set the second, and they must not diverge."""
    by_set = _components(overrides_from_dotlist(["components.robot.rplidar.lidar.rays=720"]))
    by_doc = _components({"components": {"robot.rplidar.lidar": {"rays": 720}}})
    assert by_set["robot.rplidar.lidar"]["rays"] == by_doc["robot.rplidar.lidar"]["rays"] == 720


def test_a_structural_override_changes_which_manifest_expands():
    """`model:` is read by expansion, and the override lands before it -- so this swaps the robot,
    not just a value on it. The husky ships no depth camera; the turtlebot4 does."""
    swapped = _components(overrides_from_dotlist(["components.robot.model=husky_a200"]))
    assert "robot.diff_drive" in swapped
    assert "robot.oakd.oakd_camera" not in swapped


def test_an_override_that_names_no_component_is_still_refused():
    """The property that must survive: a swept parameter that reaches nothing has to say so, or a
    campaign changes nothing while every run looks healthy."""
    with pytest.raises(PluginError, match="matches no component"):
        _components({"components": {"nosuch": {"x": 1}}})


def test_the_refusal_names_what_the_document_actually_has():
    """A bare `lidar` is a no-match rather than an ambiguity, and the useful answer is the address
    it should have used."""
    with pytest.raises(PluginError) as exc:
        _components({"components": {"lidar": {"rays": 4}}})
    assert "robot.rplidar.lidar" in str(exc.value)


def test_reaching_an_injected_component_leaves_nothing_on_its_owner():
    """The first override pass sees only the document, where `rplidar` is not yet a component, and
    writes it as a key on `robot`. The value belongs to the lidar alone: the spawn's config -- and
    the run record built from it -- must not carry a key nothing reads."""
    config = _components(overrides_from_dotlist(["components.robot.rplidar.lidar.rays=90"]))
    assert config["robot.rplidar.lidar"]["rays"] == 90
    assert "rplidar" not in config["robot"]
    assert "lidar" not in config["robot.rplidar"]


def test_the_camera_address_the_device_mount_replaced_is_refused_naming_the_new_one():
    """The TurtleBot 4's camera is `robot.oakd.oakd_camera`, on its mounted `oakd_pro` device. A
    world or campaign still switching `robot.oakd_camera` off would otherwise leave the camera on
    while every run looked configured."""
    with pytest.raises(PluginError, match=r"components\.robot\.oakd\.oakd_camera"):
        _components(overrides_from_dotlist(["components.robot.oakd_camera.enabled=false"]))
