# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Reaching a MODEL DEFAULT plugin's config from outside the world.

``spawn_robot`` pulls a model's default plugins in from its manifest, so a world that
just spawns a robot never names them. Two separate questions follow, and they have
different answers:

1. Can a world set ONE key of a default without restating the rest? (It can — the
   manifest merges underneath a world's declaration, per key.)
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
    "plugins": [{"spawn_robot": {"model": "turtlebot4", "name": "robot"}}],
}


def _expanded(raw):
    """The plugin specs that actually run, manifest defaults included."""
    from roqsim.config import _expand_plugins, load_config_from_dict

    cfg = load_config_from_dict(raw)
    return {spec.ref: spec.config for spec, _cls in _expand_plugins(cfg)}


def test_a_world_may_set_one_key_of_a_model_default(tmp_path):
    """The manifest merges UNDER the world's entry, so a one-key declaration keeps the
    model's own geometry instead of silently discarding it."""
    raw = {
        "sim": {"world": "empty_room"},
        "plugins": [
            {"spawn_robot": {"model": "turtlebot4", "name": "robot"}},
            {"lidar": {"robot": "robot", "range_stddev": 0.05}},
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
    from a model manifest. The world spawns a model, so say so and name the one-line
    stub, or the reader concludes the plugin does not exist."""
    with pytest.raises(PluginError) as exc:
        apply_overrides(WORLD_BARE, {"plugins": {"lidar": {"range_stddev": 0.05}}})
    msg = str(exc.value)
    assert "turtlebot4" in msg  # which model might be supplying it
    assert "lidar: {robot: robot}" in msg  # the exact line to add


def test_the_refusal_stays_terse_when_no_model_is_spawned():
    """No model in the world means no manifest could be hiding the plugin, so the
    hint would be noise."""
    raw = {"sim": {}, "plugins": [{"ros2_bridge": {}}]}
    with pytest.raises(PluginError) as exc:
        apply_overrides(raw, {"plugins": {"lidar": {"range_stddev": 0.05}}})
    assert "model" not in str(exc.value)


def test_a_stub_declaration_makes_a_default_addressable():
    """One line in the world -- the plugin named and wired, nothing restated -- is what
    turns a model default into a campaign factor."""
    raw = {
        "sim": {"world": "empty_room"},
        "plugins": [
            {"spawn_robot": {"model": "turtlebot4", "name": "robot"}},
            {"lidar": {"robot": "robot"}},  # the stub
        ],
    }
    merged = apply_overrides(raw, {"plugins": {"lidar": {"range_stddev": 0.05}}})
    lidar = _expanded(merged)["lidar"]
    assert lidar["range_stddev"] == 0.05
    assert lidar["rays"] == 360  # manifest still fills the rest
