"""A robot's manifest components survive world inheritance (``extends`` / ``disable``).

``disable:`` turns an inherited entry off rather than deleting it, so a world that disables the base
world's ``robot`` and declares its own ``robot`` holds two entries under one label: a dead one and a
live one. The live robot must get exactly the components it would get in a world that declared it
directly -- nothing the dead one's model supplies, and nothing lost to it.
"""

from __future__ import annotations

import textwrap

import pytest

from roqsim.config import instantiate_plugins, load_config
from roqsim.plugin import PluginError


def _write(dir_, name: str, text: str):
    path = dir_ / name
    path.write_text(textwrap.dedent(text))
    return path


def _base(dir_, model: str = "husky_a200", drive: str | None = None):
    """The base world: one robot labelled ``robot``, optionally with a declared drive override."""
    text = f"sim:\n  world: empty_room\ncomponents:\n  - spawn_robot: {{model: {model}}}\n    name: robot\n"
    if drive is not None:
        text += f"    components:\n      - diff_drive: {drive}\n"
    path = dir_ / "base.yaml"
    path.write_text(text)
    return path


def _live(cfg):
    """``{address: ref}`` of every component that will run."""
    return {s.address: s.ref for s in cfg.plugins if s.enabled}


def _by_address(cfg, address):
    [spec] = [s for s in cfg.plugins if s.enabled and s.address == address]
    return spec


def _direct(dir_, robot: str = "spawn_robot: {model: turtlebot3_waffle}"):
    """The same robot declared in a world of its own: what the inherited one must equal."""
    return load_config(
        _write(
            dir_,
            "direct.yaml",
            f"""
            sim:
              world: empty_room
            components:
              - {robot}
                name: robot
            """,
        )
    )


def test_redeclared_robot_keeps_its_manifest_components(tmp_path):
    """The regression: every component of the re-declared robot's model was silently dropped."""
    _base(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle}
            name: robot
        """,
    )
    cfg = load_config(child)
    live = _live(cfg)
    assert live == _live(_direct(tmp_path))
    assert live["robot.diff_drive"] == "diff_drive"
    assert live["robot.lds01"] == "spawn_sensor"
    assert live["robot.lds01.lidar"] == "lidar"
    # The live drive is the turtlebot's, not a husky default the dead entry's manifest filled in.
    assert _by_address(cfg, "robot.diff_drive").config["wheel_radius"] == 0.033
    # And what runs is that set: the robot, its drive, its scanner and the scanner's lidar.
    refs = [type(p).__name__ for p in instantiate_plugins(cfg)]
    assert refs.count("SpawnRobotPlugin") == 1
    assert "DiffDrivePlugin" in refs


def test_redeclared_robot_of_the_same_model_keeps_its_components(tmp_path):
    """Same model on both sides: the dead entry's injected copies must not stand in for the live."""
    _base(tmp_path, model="turtlebot3_waffle")
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle}
            name: robot
        """,
    )
    assert _live(load_config(child)) == _live(_direct(tmp_path))


def test_dead_robot_does_not_fill_the_live_robots_overrides(tmp_path):
    """A partial override on the live robot merges its OWN manifest's defaults, not the dead one's.

    The base world's robot is a husky with a declared drive override; the child's is a turtlebot
    with one. Both are ``robot.diff_drive``, and the husky's manifest expands first.
    """
    _base(tmp_path, drive="{max_linear_vel: 0.5}")
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle}
            name: robot
            components:
              - diff_drive: {max_linear_vel: 0.1}
        """,
    )
    cfg = load_config(child)
    drive = _by_address(cfg, "robot.diff_drive").config
    assert drive["max_linear_vel"] == 0.1
    assert drive["wheel_radius"] == 0.033
    assert drive["left_actuator"] == "left_wheel_motor"
    assert "slip_factor" not in drive
    assert _live(cfg) == _live(_direct(tmp_path))


def test_extends_without_disable_keeps_the_inherited_robots_components(tmp_path):
    _base(tmp_path, model="turtlebot3_waffle")
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        components:
          - dummy: {}
            name: greeter
        """,
    )
    live = _live(load_config(child))
    assert {k: v for k, v in live.items() if k != "greeter"} == _live(_direct(tmp_path))


def test_overriding_the_inherited_robot_by_name_is_refused(tmp_path):
    """Two live entries under one label is an ambiguity, and it is refused rather than resolved."""
    _base(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        components:
          - spawn_robot: {model: turtlebot3_waffle}
            name: robot
        """,
    )
    with pytest.raises(PluginError, match="two components labelled 'robot'"):
        load_config(child)


def test_overriding_the_inherited_robot_by_set_keeps_its_components(tmp_path):
    """Replacing the model by override lands before expansion, so the new model's manifest runs."""
    _base(tmp_path)
    cfg = load_config(
        tmp_path / "base.yaml",
        overrides={"components": {"robot": {"model": "turtlebot3_waffle"}}},
    )
    assert _live(cfg) == _live(_direct(tmp_path))


def test_default_plugins_false_still_suppresses_injection(tmp_path):
    _base(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle, default_plugins: false}
            name: robot
        """,
    )
    assert _live(load_config(child)) == {"robot": "spawn_robot"}


def test_a_default_the_live_robot_disables_stays_off(tmp_path):
    """``enabled: false`` on a live robot's component is still how a manifest default is turned off."""
    _base(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle}
            name: robot
            components:
              - spawn_sensor: {}
                name: lds01
                enabled: false
        """,
    )
    live = _live(load_config(child))
    assert "robot.diff_drive" in live
    assert "robot.lds01" not in live
    assert "robot.lds01.lidar" not in live


def test_the_replaced_robot_stays_in_the_record_and_out_of_reach(tmp_path):
    """The replaced entry is kept, turned off, with only what its document declared for it."""
    _base(tmp_path, drive="{max_linear_vel: 0.5}")
    _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle}
            name: robot
        """,
    )
    cfg = load_config(
        tmp_path / "child.yaml",
        overrides={"components": {"robot": {"diff_drive": {"max_linear_vel": 0.2}}}},
    )
    dead = [s for s in cfg.plugins if not s.enabled]
    assert [(s.address, s.config.get("model")) for s in dead] == [
        ("robot", "husky_a200"),
        ("robot.diff_drive", None),
    ]
    # No husky wheel geometry: the replaced robot's manifest was never merged into anything.
    assert "wheel_radius" not in dead[1].config
    assert _by_address(cfg, "robot.diff_drive").config["max_linear_vel"] == 0.2


def test_a_mount_on_the_live_robot_takes_the_live_robots_prefix(tmp_path):
    """A mounted device reads its carrier's prefix by address; the live carrier is the one it gets."""
    path = tmp_path / "base.yaml"
    path.write_text(
        "sim:\n  world: empty_room\ncomponents:\n"
        "  - spawn_robot: {model: husky_a200, prefix: old_}\n    name: robot\n"
    )
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - spawn_robot: {model: turtlebot3_waffle, prefix: new_}
            name: robot
        """,
    )
    cfg = load_config(child)
    assert _by_address(cfg, "robot.lds01").config["attach_prefix"] == "new_"
    assert _by_address(cfg, "robot.diff_drive").config["prefix"] == "new_"


def test_a_disabled_robot_with_no_replacement_still_expands(tmp_path):
    """Only a REPLACED entry is left unexpanded: a plain disable keeps its components addressable."""
    _base(tmp_path, model="turtlebot3_waffle")
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        disable: [robot]
        components:
          - dummy: {}
            name: greeter
        """,
    )
    cfg = load_config(
        child, overrides={"components": {"robot": {"lds01": {"lidar": {"rays": 90}}}}}
    )
    assert _live(cfg) == {"greeter": "dummy"}
    [lidar] = [s for s in cfg.plugins if s.address == "robot.lds01.lidar"]
    assert not lidar.enabled
    assert lidar.config["rays"] == 90
