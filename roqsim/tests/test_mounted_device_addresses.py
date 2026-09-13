# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""An address that stops short of a mounted device is refused at load, with the address meant.

A robot whose scanner is a device its manifest mounts (``robot.rplidar.lidar``) has no ``lidar`` of
its own. Unrefused, two spellings aimed at that lidar would load cleanly and do nothing -- or fail far
from the cause:

* an override ``components.robot.lidar.rays`` stops where the tree ends, at ``robot``, and would write
  a ``lidar`` key there that nothing reads while the real lidar keeps its value;
* a world nesting ``- lidar: {...}`` under the robot would load a second lidar beside the device's,
  which fails only once the model compiles, naming a site rather than the mount.
"""

import copy

import pytest

from roqsim.config import (
    PluginError,
    instantiate_plugins,
    load_config_from_dict,
    overrides_from_dotlist,
)

pytest.importorskip("roqsim_mobile", reason="turtlebot4 manifest lives in roqsim_mobile")
pytest.importorskip("roqsim_sensors", reason="the rplidar_a1 device lives in roqsim_sensors")


def _world(robot_config=None, children=None):
    entry = {"spawn_robot": {"model": "turtlebot4", **(robot_config or {})}, "name": "robot"}
    if children is not None:
        entry["components"] = children
    return {"sim": {"world": "empty_room"}, "components": [entry]}


def _load(doc, overrides=None):
    return load_config_from_dict(copy.deepcopy(doc), overrides=overrides)


# -- a key that names no setting -------------------------------------------------------------------


def test_an_override_stopping_at_the_robot_names_the_mounted_lidar():
    with pytest.raises(PluginError) as exc:
        _load(_world(), overrides_from_dotlist(["components.robot.lidar.rays=720"]))
    message = str(exc.value)
    assert "'lidar' is not a spawn_robot key" in message
    assert "components.robot.rplidar.lidar" in message


def test_the_same_key_written_in_the_world_is_refused_the_same_way():
    with pytest.raises(PluginError, match=r"components\.robot\.rplidar\b"):
        _load(_world({"rplidar": {"frame_id": "laser"}}))


def test_a_key_naming_no_component_is_refused_when_the_plugins_are_validated():
    cfg = _load(_world({"lidar_rays": 720}))
    with pytest.raises(PluginError, match="'lidar_rays' is not a setting of this component"):
        instantiate_plugins(cfg)


def test_a_standalone_mount_names_its_own_device_component():
    doc = {
        "sim": {"world": "empty_room"},
        "components": [
            {"spawn_sensor": {"model": "rplidar_a1", "lidar": {"rays": 90}}, "name": "scan"}
        ],
    }
    with pytest.raises(PluginError) as exc:
        _load(doc)
    assert "'lidar' is not a spawn_sensor key" in str(exc.value)
    assert "components.scan.lidar" in str(exc.value)


def test_a_misspelt_mount_key_is_refused():
    doc = {
        "sim": {"world": "empty_room"},
        "components": [
            {"spawn_sensor": {"model": "rplidar_a1", "show_fovv": True}, "name": "scan"}
        ],
    }
    with pytest.raises(PluginError, match="'show_fovv' is not a setting"):
        instantiate_plugins(_load(doc))


def test_the_full_address_still_loads_and_validates():
    cfg = _load(_world(), overrides_from_dotlist(["components.robot.rplidar.lidar.rays=720"]))
    configs = {s.address: s.config for s in cfg.plugins}
    assert configs["robot.rplidar.lidar"]["rays"] == 720
    assert "lidar" not in configs["robot"] and "rplidar" not in configs["robot"]
    instantiate_plugins(cfg)


# -- an entry nested under the carrier instead of under its mount ---------------------------------


def test_a_stale_nested_lidar_names_the_mount_to_nest_it_under():
    with pytest.raises(PluginError) as exc:
        _load(_world(children=[{"lidar": {"range_stddev": 0.01}}]))
    message = str(exc.value)
    assert "nest it under 'robot.rplidar' (spawn_sensor, name: rplidar)" in message
    assert "robot.rplidar.lidar" in message


def test_the_same_override_nested_under_the_mount_loads():
    doc = _world(
        children=[
            {
                "spawn_sensor": {},
                "name": "rplidar",
                "components": [{"lidar": {"range_stddev": 0.01}}],
            }
        ]
    )
    cfg = _load(doc)
    configs = {s.address: s.config for s in cfg.plugins}
    assert configs["robot.rplidar.lidar"]["range_stddev"] == 0.01
    assert configs["robot.rplidar.lidar"]["site"] == "scan"  # the device still supplies the rest
    assert "robot.lidar" not in configs
    instantiate_plugins(cfg)


@pytest.mark.parametrize("name", [None, "lidar_rear"])
def test_a_second_lidar_that_names_a_site_of_the_robot_is_not_refused(name):
    """A real second sensor states where it hangs on the carrier; `shell_link` is a frame the
    turtlebot4 manifest declares, which becomes a site at build."""
    child = {"lidar": {"site": "shell_link", "frame_id": "second_scan"}}
    if name:
        child["name"] = name
    cfg = _load(_world(children=[child]))
    assert f"robot.{name or 'lidar'}" in {s.address for s in cfg.plugins}
    assert "robot.rplidar.lidar" in {s.address for s in cfg.plugins}


def test_a_robot_with_its_defaults_off_is_not_second_guessed():
    """Without the manifest nothing is mounted, so a lidar under the robot is the only one."""
    cfg = _load(_world({"default_plugins": False}, children=[{"lidar": {"range_stddev": 0.01}}]))
    assert "robot.lidar" in {s.address for s in cfg.plugins}
