# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Reaching a MODEL DEFAULT component's config from outside the document.

``spawn_robot`` pulls a model's components in from its manifest, so a world that just spawns a robot
never names its lidar. It used to be unable to reach one either: overrides resolved against the
parsed YAML, before expansion, so ``plugins.lidar.rays`` named nothing and was refused. The refusal
was right -- silently ignoring a swept parameter lets a campaign look healthy while changing nothing
-- but it left a model default unreachable, and the documented way out was a stub entry whose only
job was to exist.

Expansion now happens while the document loads, so an override resolves against what will actually
run. This is the file that says so, and it is the reason for the whole change.
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
    lidar = _components({"components": {"robot.lidar": {"range_stddev": 0.05}}})["robot.lidar"]
    assert lidar["range_stddev"] == 0.05


def test_the_manifest_still_supplies_everything_the_override_did_not_name():
    """An override is partial, like a declaration: it sets keys, it does not replace a component."""
    lidar = _components({"components": {"robot.lidar": {"range_stddev": 0.05}}})["robot.lidar"]
    assert lidar["rays"] == 360
    assert lidar["max_range"] == 12.0
    assert lidar["frame_id"] == "rplidar_link"


def test_the_dotlist_spelling_means_the_same_thing():
    """`--set` and an override document are two spellings of one assignment; a campaign writes the
    first and a saved override set the second, and they must not diverge."""
    by_set = _components(overrides_from_dotlist(["components.robot.lidar.rays=720"]))
    by_doc = _components({"components": {"robot.lidar": {"rays": 720}}})
    assert by_set["robot.lidar"]["rays"] == by_doc["robot.lidar"]["rays"] == 720


def test_a_structural_override_changes_which_manifest_expands():
    """`model:` is read by expansion, and the override lands before it -- so this swaps the robot,
    not just a value on it. The husky ships no depth camera; the turtlebot4 does."""
    swapped = _components(overrides_from_dotlist(["components.robot.model=husky_a200"]))
    assert "robot.diff_drive" in swapped
    assert "robot.oakd_camera" not in swapped


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
    assert "robot.lidar" in str(exc.value)
