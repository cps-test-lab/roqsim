# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Two things an override does beyond setting one key on one component.

``*`` fans out over components, so a fleet-wide sweep does not have to enumerate entities the world
may not have when the campaign is written. ``components:`` on an address ADDS under it, so a campaign
can bring its own instrumentation to a world it does not edit.
"""

import pytest

from roqsim.config import PluginError, instantiate_plugins, load_config_from_dict

pytest.importorskip("roqsim_mobile", reason="turtlebot4 manifest lives in roqsim_mobile")

FLEET = {
    "sim": {"world": "empty_room"},
    "components": [
        {"spawn_robot": {"model": "turtlebot4"}, "name": "r1"},
        {"spawn_robot": {"model": "turtlebot4"}, "name": "r2"},
    ],
}


def _cfg(overrides):
    return load_config_from_dict(FLEET, overrides=overrides)


# -- wildcards ---------------------------------------------------------------------------------


def test_one_assignment_reaches_every_robots_sensor():
    cfg = _cfg({"components": {"*.lidar": {"range_stddev": 0.05}}})
    hit = {s.address: s.config["range_stddev"] for s in cfg.plugins if s.ref == "lidar"}
    assert hit == {"r1.lidar": 0.05, "r2.lidar": 0.05}


def test_it_reaches_components_no_document_declared():
    """Which is the point: neither robot's lidar is written anywhere -- both come from the model."""
    assert all(
        s.entity
        for s in _cfg({"components": {"*.lidar": {"rays": 720}}}).plugins
        if s.ref == "lidar"
    )


def test_the_manifest_still_fills_in_the_rest():
    lidar = next(
        s for s in _cfg({"components": {"*.lidar": {"rays": 720}}}).plugins if s.ref == "lidar"
    )
    assert lidar.config["rays"] == 720 and lidar.config["max_range"] == 12.0


def test_a_wildcard_that_reaches_nothing_is_refused_like_a_typo():
    """The property this is worth having at all: a sweep that changed nothing would otherwise look
    exactly like one that worked."""
    with pytest.raises(PluginError, match="matches no component"):
        _cfg({"components": {"*.nosuch": {"x": 1}}})


def test_the_refusal_explains_what_a_wildcard_addresses():
    """Because the mistake it catches -- a component name that is not one here -- looks like a
    config key, and the reader needs to be told which of the two roqsim read it as."""
    with pytest.raises(PluginError) as exc:
        _cfg({"components": {"*": {"pos": [1, 2]}}})
    assert "fans out over components" in str(exc.value)


def test_a_wildcard_matches_exactly_one_segment():
    """Not a subtree. `*.lidar` reaches each robot's lidar and stops there -- everything after the
    address is a path into that component's config, however deep."""
    cfg = _cfg({"components": {"*.lidar.topics.scan": "/s"}})
    scans = {s.address: s.config["topics"]["scan"] for s in cfg.plugins if s.ref == "lidar"}
    assert scans == {"r1.lidar": "/s", "r2.lidar": "/s"}


# -- adding ------------------------------------------------------------------------------------


def test_an_override_can_add_a_component_under_an_owner():
    """A campaign brings its own instrumentation without editing the world it measures."""
    cfg = _cfg({"components": {"r1": {"components": [{"contact_monitor": {"min_force": 2.0}}]}}})
    added = next(s for s in cfg.plugins if s.ref == "contact_monitor")
    assert added.address == "r1.contact_monitor"
    assert added.entity == "r1"  # wired by position, like anything the document declared
    assert added.config["min_force"] == 2.0
    assert len(instantiate_plugins(cfg)) == len(cfg.plugins)


def test_an_added_component_behaves_as_though_the_document_declared_it():
    """It goes back through the same walk, so it is wired, checked and merged identically -- adding
    a `lidar` sets keys on the model's, exactly as declaring one in the document would, rather than
    starting a second sensor beside it."""
    cfg = _cfg({"components": {"r1": {"components": [{"lidar": {"rays": 999}}]}}})
    lidars = [
        (s.address, s.config["rays"], s.config["max_range"])
        for s in cfg.plugins
        if s.ref == "lidar" and s.entity == "r1"
    ]
    assert lidars == [("r1.lidar", 999, 12.0)]


def test_adding_takes_a_list_of_entries():
    """A mapping there is the document's own `components:` shape misremembered, and the message has
    to say which shape is wanted rather than reporting a reserved key."""
    with pytest.raises(PluginError, match="LIST of entries"):
        _cfg({"components": {"r1": {"components": {"contact_monitor": {}}}}})
