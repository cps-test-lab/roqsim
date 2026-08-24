# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``enabled: false`` is how a component is removed.

Turning one off rather than deleting it is what makes "is this sensor present" a value a campaign can
sweep instead of a structural edit to the world file -- and it leaves a trace: the component is still
addressable, still in the record saying what was turned off, and a later override can turn it back on.
"""

import pytest

from roqsim.config import PluginError, instantiate_plugins, load_config_from_dict

pytest.importorskip("roqsim_mobile", reason="turtlebot4 manifest lives in roqsim_mobile")

BARE = {
    "sim": {"world": "empty_room"},
    "components": [{"spawn_robot": {"model": "turtlebot4"}, "name": "robot"}],
}


def _cfg(overrides=None):
    return load_config_from_dict(BARE, overrides=overrides)


def test_a_manifest_component_can_be_switched_off_from_outside():
    """No entry, no stub: a model default is a campaign factor rather than a file edit."""
    cfg = _cfg({"components": {"robot.oakd_camera": {"enabled": False}}})
    assert {s.address: s.enabled for s in cfg.plugins}["robot.oakd_camera"] is False
    assert "OakdCameraPlugin" not in [type(p).__name__ for p in instantiate_plugins(cfg)]


def test_the_component_stays_addressable_and_in_the_record():
    """The reason this is a flag rather than a deletion: the run's record can still say what was
    turned off, and a later override can turn it back on."""
    cfg = _cfg({"components": {"robot.lidar": {"enabled": False}}})
    assert "robot.lidar" in {s.address for s in cfg.plugins}
    back = _cfg({"components": {"robot.lidar": {"enabled": False}}, "sim": {}})
    assert {s.address: s.enabled for s in back.plugins}["robot.lidar"] is False


def test_disabling_an_owner_disables_what_it_owns():
    """Computed, not left to the reader. Without it the entity is never registered and every
    component aimed at it resolves an unprefixed name, failing much later with a message about
    MJCF rather than about the switch that caused it."""
    cfg = _cfg({"components": {"robot": {"enabled": False}}})
    assert all(not s.enabled for s in cfg.plugins)
    assert instantiate_plugins(cfg) == []


def test_enabled_must_be_a_boolean_wherever_it_is_written():
    with pytest.raises(PluginError, match="true or false"):
        load_config_from_dict({"sim": {}, "components": [{"dummy": {}, "enabled": "yes"}]})
    with pytest.raises(PluginError, match="true or false"):
        _cfg({"components": {"robot.lidar": {"enabled": "no"}}})


def test_the_ceiling_plugin_refuses_enabled_by_name():
    """It is SUBTRACTIVE -- it opens a roof by removing geometry -- so `enabled: false` would leave
    the ceiling standing, the opposite of what the key used to mean. Silence there would be a
    world that quietly stopped doing what it said."""
    cfg = load_config_from_dict(
        {"sim": {}, "components": [{"ceiling": {"enabled": False, "above_z": 2.6}}]}
    )
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    assert "Use `keep:`" in str(exc.value)
