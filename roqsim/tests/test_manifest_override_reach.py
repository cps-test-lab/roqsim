# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Reaching a MODEL DEFAULT plugin's config from outside the world.

``spawn_robot`` pulls a model's default plugins in from its manifest, so a world that
just spawns a robot never names them. Two separate questions follow, and they have
different answers:

1. Can a world set ONE key of a default without restating the rest? (It can — declare it
   as a component of the spawn, and the manifest merges underneath, per key.)
2. Can an ``--override`` / ``--set`` reach a default the world never declared at all?
   (It cannot — overrides are applied to the raw YAML, before manifest expansion.)

The second is what decides whether a campaign can sweep a model default, so it is
pinned here rather than left to be rediscovered.
"""

import pytest

from roqsim.config import PluginError, apply_overrides

pytest.importorskip("roqsim_mobile", reason="turtlebot4 manifest lives in roqsim_mobile")

WORLD_BARE = {
    "sim": {"world": "empty_room"},
    "components": [{"spawn_robot": {"model": "turtlebot4"}, "name": "robot"}],
}


def _expanded(raw):
    """The plugin specs that actually run, manifest defaults included."""
    from roqsim.config import load_config_from_dict

    return {spec.ref: spec.config for spec in load_config_from_dict(raw).plugins}


def test_a_world_may_set_one_key_of_a_model_default(tmp_path):
    """The manifest merges UNDER the owner's entry, so a one-key declaration keeps the
    model's own geometry instead of silently discarding it.

    The entry sits inside the spawn, which is what says it belongs to that robot -- there is no
    `robot:` key to spell, and therefore no way to spell it wrong."""
    raw = {
        "sim": {"world": "empty_room"},
        "components": [
            {
                "spawn_robot": {"model": "turtlebot4"},
                "name": "robot",
                "components": [{"lidar": {"range_stddev": 0.05}}],
            }
        ],
    }
    lidar = _expanded(raw)["lidar"]
    assert lidar["range_stddev"] == 0.05  # what the world asked for
    assert lidar["rays"] == 360  # ...and the manifest's, not lost
    assert lidar["max_range"] == 12.0
    assert lidar["frame_id"] == "rplidar_link"


def test_an_override_cannot_reach_a_default_the_world_never_declared():
    """The actual limit: overrides resolve against the world's own plugin list, which
    at that point has no `lidar` entry -- so a campaign sweeping `plugins.lidar.*`
    against a bare world is REFUSED rather than silently ignored."""
    with pytest.raises(PluginError):
        apply_overrides(WORLD_BARE, {"plugins": {"lidar": {"range_stddev": 0.05}}})


def test_the_refusal_names_the_fix_when_a_model_could_supply_the_plugin():
    """A bare 'matches no plugin' is a dead end when the plugin DOES run -- it just came
    from a model manifest. The world spawns a model, so say which, or the reader concludes
    the plugin does not exist."""
    with pytest.raises(PluginError) as exc:
        apply_overrides(WORLD_BARE, {"plugins": {"lidar": {"range_stddev": 0.05}}})
    assert "turtlebot4" in str(exc.value)  # which model might be supplying it


def test_the_refusal_stays_terse_when_no_model_is_spawned():
    """No model in the world means no manifest could be hiding the plugin, so the
    hint would be noise."""
    raw = {"sim": {}, "components": [{"ros2_bridge": {}}]}
    with pytest.raises(PluginError) as exc:
        apply_overrides(raw, {"plugins": {"lidar": {"range_stddev": 0.05}}})
    assert "model" not in str(exc.value)


def test_the_refusal_names_the_owner_to_nest_under():
    """The old advice -- a stub at the top of the document -- no longer does anything: an entry
    belongs to the entry it sits under, so a lidar declared beside the robot is not the robot's.
    The message has to name the owner, or it sends the reader somewhere that silently fails."""
    with pytest.raises(PluginError) as exc:
        apply_overrides(WORLD_BARE, {"plugins": {"lidar": {"range_stddev": 0.05}}})
    msg = str(exc.value)
    assert "components:" in msg
    assert "'robot'" in msg  # the entry to nest it under, by label
