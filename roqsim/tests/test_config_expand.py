"""Config: Plugin.expand splices extra plugin specs into the pipeline (per-plugin dedupe is the
producing plugin's job -- see test_manifest.py -- so core just splices what expand returns)."""

from __future__ import annotations

import textwrap

import pytest

from roqsim.config import PluginSpec, instantiate_plugins, load_config_from_dict
from roqsim.manifest import expand_manifest
from roqsim.plugin import Plugin, PluginError


class Child(Plugin):
    pass


class Parent(Plugin):
    """Stand-in for a spawn plugin: injects a Child wired to its entity."""

    @classmethod
    def expand(cls, spec: PluginSpec, world, base_dir):
        # Wired to the entry's LABEL, the way a real spawn plugin names the entity it registers.
        return [PluginSpec(ref=CHILD, name=None, config={"robot": spec.label})]


PARENT = f"{__name__}:Parent"
CHILD = f"{__name__}:Child"


def _kinds(plugins):
    return [type(p).__name__ for p in plugins]


def test_expand_injects_spec_after_parent():
    cfg = load_config_from_dict({"plugins": [{PARENT: {}, "name": "r1"}]})
    plugins = instantiate_plugins(cfg)
    assert _kinds(plugins) == ["Parent", "Child"]
    assert plugins[1].config == {"robot": "r1"}  # wired to the parent's entity


def test_plain_plugin_expands_to_nothing():
    cfg = load_config_from_dict({"plugins": [{CHILD: {}}]})
    assert _kinds(instantiate_plugins(cfg)) == ["Child"]


def test_each_parent_injects_its_own_child():
    cfg = load_config_from_dict(
        {
            "plugins": [
                {PARENT: {}, "name": "alice"},
                {PARENT: {}, "name": "bob"},
            ]
        }
    )
    plugins = instantiate_plugins(cfg)
    assert _kinds(plugins) == ["Parent", "Child", "Parent", "Child"]
    assert plugins[1].config["robot"] == "alice"
    assert plugins[3].config["robot"] == "bob"


# -- recursive expansion: a carrier mounting a device that has a manifest of its own ---------------


class Carrier(Plugin):
    """Stand-in for spawn_robot: registers an entity and expands its model's manifest."""

    provides_entity = True

    @classmethod
    def expand(cls, spec, world, base_dir):
        return expand_manifest(spec, world, base_dir=base_dir)


class Device(Carrier):
    """Stand-in for spawn_sensor: a mountable entity whose model has a manifest too."""


class Capture(Plugin):
    requires_owner = True


CARRIER = f"{__name__}:Carrier"
DEVICE = f"{__name__}:Device"
CAPTURE = f"{__name__}:Capture"


def _model(tmp_path, stem, manifest):
    (tmp_path / f"{stem}.xml").write_text("<mujoco/>")
    (tmp_path / f"{stem}.manifest.yaml").write_text(textwrap.dedent(manifest))
    return str(tmp_path / f"{stem}.xml")


def _device_model(tmp_path, stem="scanner", rays=360):
    return _model(
        tmp_path,
        stem,
        f"""
        components:
          - {CAPTURE}: {{rays: {rays}, range_max: 20.0, frame: laser}}
            name: lidar
          - {CAPTURE}: {{rate: 100}}
            name: imu
        """,
    )


def _robot_model(tmp_path, device, lidar_override=None, stem="robot"):
    nested = (
        f"\n            components:\n              - {CAPTURE}: {lidar_override}\n                name: lidar"
        if lidar_override
        else ""
    )
    return _model(
        tmp_path,
        stem,
        f"""
        components:
          - {CAPTURE}: {{wheel_radius: 0.1}}
            name: drive
          - {DEVICE}: {{model: {device}}}
            name: scan_front{nested}
          - {CAPTURE}: {{}}
            name: bumper
        """,
    )


def _load(entries):
    return load_config_from_dict({"components": entries})


def test_expansion_is_depth_first_and_follows_the_document_shape(tmp_path):
    robot = _robot_model(tmp_path, _device_model(tmp_path))
    cfg = _load([{CARRIER: {"model": robot, "prefix": "r_"}, "name": "robot"}])
    assert [s.address for s in cfg.plugins] == [
        "robot",
        "robot.drive",
        "robot.scan_front",
        "robot.scan_front.lidar",
        "robot.scan_front.imu",
        "robot.bumper",
    ]
    by = {s.address: s for s in cfg.plugins}
    # The device takes its carrier's prefix as attach_prefix, and only a plain component takes it
    # as prefix: the device's own components are the device's to prefix.
    assert by["robot.scan_front"].config["attach_prefix"] == "r_"
    assert "prefix" not in by["robot.scan_front"].config
    assert by["robot.drive"].config["prefix"] == "r_"
    assert by["robot.scan_front.lidar"].entity == "robot.scan_front"
    assert _kinds(instantiate_plugins(cfg)) == [
        "Carrier",
        "Capture",
        "Device",
        "Capture",
        "Capture",
        "Capture",
    ]


def test_a_manifest_entry_keeps_its_nested_components(tmp_path):
    """A robot manifest overrides part of a mounted device by nesting under it."""
    robot = _robot_model(tmp_path, _device_model(tmp_path), lidar_override="{rays: 720}")
    cfg = _load([{CARRIER: {"model": robot}, "name": "robot"}])
    lidar = next(s for s in cfg.plugins if s.address == "robot.scan_front.lidar")
    assert lidar.config["rays"] == 720  # the robot manifest's value
    assert lidar.config["range_max"] == 20.0  # the device manifest fills the rest
    assert [s.address for s in cfg.plugins].count("robot.scan_front.lidar") == 1


def test_nearer_wins_world_over_robot_manifest_over_device_manifest(tmp_path):
    robot = _robot_model(tmp_path, _device_model(tmp_path), lidar_override="{rays: 720, frame: a}")
    cfg = _load(
        [
            {
                CARRIER: {"model": robot},
                "name": "robot",
                "components": [
                    {
                        DEVICE: {},
                        "name": "scan_front",
                        "components": [{CAPTURE: {"rays": 90}, "name": "lidar"}],
                    }
                ],
            }
        ]
    )
    lidar = next(s for s in cfg.plugins if s.address == "robot.scan_front.lidar")
    assert lidar.config["rays"] == 90  # world
    assert lidar.config["frame"] == "a"  # robot manifest, over the device's "laser"
    assert lidar.config["range_max"] == 20.0  # device manifest
    addresses = [s.address for s in cfg.plugins]
    assert addresses.count("robot.scan_front") == 1
    # The world declared the device but not its imu: the device manifest still brings it, after it.
    assert addresses.index("robot.scan_front") < addresses.index("robot.scan_front.imu")


def test_a_world_declared_device_is_built_before_the_robot_manifests_children_of_it(tmp_path):
    """The robot manifest's `lidar` under `scan_front` is owned by the WORLD's scan_front entry,
    which comes after the robot's own expansion -- so it waits for that owner rather than building
    before the mount it hangs from."""
    robot = _robot_model(tmp_path, _device_model(tmp_path), lidar_override="{rays: 720}")
    cfg = _load(
        [
            {
                CARRIER: {"model": robot},
                "name": "robot",
                "components": [{DEVICE: {"pos": [1, 0, 0]}, "name": "scan_front"}],
            }
        ]
    )
    addresses = [s.address for s in cfg.plugins]
    assert addresses.index("robot.scan_front") < addresses.index("robot.scan_front.lidar")
    lidar = next(s for s in cfg.plugins if s.address == "robot.scan_front.lidar")
    assert lidar.config["rays"] == 720


def test_two_robots_mounting_the_same_device_label_do_not_collide(tmp_path):
    robot = _robot_model(tmp_path, _device_model(tmp_path))
    cfg = _load(
        [
            {CARRIER: {"model": robot, "prefix": "a_"}, "name": "alice"},
            {CARRIER: {"model": robot, "prefix": "b_"}, "name": "bob"},
        ]
    )
    addresses = [s.address for s in cfg.plugins]
    assert "alice.scan_front.lidar" in addresses and "bob.scan_front.lidar" in addresses
    assert len(addresses) == len(set(addresses))
    by = {s.address: s for s in cfg.plugins}
    assert by["alice.scan_front"].config["attach_prefix"] == "a_"
    assert by["bob.scan_front"].config["attach_prefix"] == "b_"


def test_an_override_reaches_a_component_of_a_mounted_device(tmp_path):
    robot = _robot_model(tmp_path, _device_model(tmp_path))
    cfg = load_config_from_dict(
        {"components": [{CARRIER: {"model": robot}, "name": "robot"}]},
        overrides={"components": {"robot": {"scan_front": {"lidar": {"rays": 1440}}}}},
    )
    lidar = next(s for s in cfg.plugins if s.address == "robot.scan_front.lidar")
    assert lidar.config["rays"] == 1440
    # ...and the record carries it at depth, and reads back the same.
    record = cfg.as_record()
    again = type(cfg).from_record(record)
    assert [(s.address, s.entity) for s in again.plugins] == [
        (s.address, s.entity) for s in cfg.plugins
    ]


def test_disabling_a_robot_disables_its_mounted_device_and_what_the_device_owns(tmp_path):
    robot = _robot_model(tmp_path, _device_model(tmp_path))
    cfg = load_config_from_dict(
        {"components": [{CARRIER: {"model": robot}, "name": "robot"}]},
        overrides={"components": {"robot": {"scan_front": {"enabled": False}}}},
    )
    by = {s.address: s for s in cfg.plugins}
    assert not by["robot.scan_front"].enabled
    assert not by["robot.scan_front.lidar"].enabled
    assert by["robot.drive"].enabled


def test_a_manifest_mounting_its_own_model_is_a_named_cycle(tmp_path):
    loop = tmp_path / "loop.xml"
    _model(
        tmp_path,
        "loop",
        f"""
        components:
          - {DEVICE}: {{model: {loop}}}
            name: again
        """,
    )
    with pytest.raises(PluginError, match=r"cycle.*robot \(.*loop.xml\) -> robot.again"):
        _load([{CARRIER: {"model": str(loop)}, "name": "robot"}])


def test_expansion_deeper_than_the_bound_is_refused(tmp_path):
    # Ten distinct models, each mounting the next: no cycle, only depth.
    for i in range(10, -1, -1):
        nxt = (
            f"\n          - {DEVICE}: {{model: {tmp_path / f'm{i + 1}.xml'}}}\n            name: d"
            if i < 10
            else ""
        )
        _model(tmp_path, f"m{i}", f"components:{nxt or ' []'}\n")
    with pytest.raises(PluginError, match="deeper than 8"):
        _load([{CARRIER: {"model": str(tmp_path / "m0.xml")}, "name": "robot"}])


def test_components_under_an_entry_that_registers_no_entity_are_refused(tmp_path):
    robot = _model(
        tmp_path,
        "bad",
        f"""
        components:
          - {CAPTURE}: {{}}
            name: lidar
            components:
              - {CAPTURE}: {{}}
        """,
    )
    with pytest.raises(PluginError, match="registers no entity"):
        _load([{CARRIER: {"model": robot}, "name": "robot"}])
