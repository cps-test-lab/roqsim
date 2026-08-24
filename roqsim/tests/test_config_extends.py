"""World inheritance: the ``extends`` / ``disable`` composition keys (see config._resolve_inheritance)."""

from __future__ import annotations

import textwrap

import pytest

from roqsim.config import load_config
from roqsim.plugin import PluginError


def _write(dir_, name: str, text: str):
    path = dir_ / name
    path.write_text(textwrap.dedent(text))
    return path


def _parent(dir_, **sim):
    """A parent world with a relative ``sim.world`` and two named furniture plugins."""
    world = sim.pop("world", "scene/scene.xml")
    extra = "".join(f"  {k}: {v}\n" for k, v in sim.items())
    return _write(
        dir_,
        "parent.yaml",
        f"""
        sim:
          world: {world}
          pacing: realtime
        {extra}plugins:
          - spawn_model: {{model: industrial_table}}
            name: table_1
          - spawn_model: {{model: industrial_table}}
            name: table_2
          - dummy: {{}}
            name: greeter
        """,
    )


def test_child_plugins_appended_after_parent(tmp_path):
    _parent(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: parent.yaml
        plugins:
          - spawn_robot: {model: oli, prefix: oli_}
            name: oli
        """,
    )
    cfg = load_config(child)
    # `declared` is what the two documents said; `plugins` is that plus what their models' manifests
    # contribute, which is not what this test is about.
    refs = [s.ref for s in cfg.declared]
    assert refs == [
        "spawn_model",
        "spawn_model",
        "dummy",
        "spawn_robot",
    ]  # parent first, child last


def test_sim_block_deep_merged_child_wins(tmp_path):
    _parent(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: parent.yaml
        sim:
          timestep: 0.001
        plugins: []
        """,
    )
    cfg = load_config(child)
    assert cfg.timestep == 0.001  # child adds it
    assert cfg.pacing == "realtime"  # inherited from the parent


def test_parent_relative_world_is_absolutized(tmp_path):
    # Parent lives in a subdir; its relative sim.world must resolve against the PARENT dir, not the
    # child's, once inherited.
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "scene").mkdir()
    (sub / "scene" / "scene.xml").write_text("<mujoco/>")
    _parent(sub)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: sub/parent.yaml
        plugins: []
        """,
    )
    cfg = load_config(child)
    assert cfg.sim["world"] == str((sub / "scene" / "scene.xml").resolve())


def test_disable_drops_named_plugin(tmp_path):
    _parent(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: parent.yaml
        disable: [table_2, greeter]
        plugins: []
        """,
    )
    cfg = load_config(child)
    # Both matched by their LABEL -- one key, whether the entry named itself or fell back to its ref.
    assert [s.label for s in cfg.plugins] == ["table_1"]


def test_disable_unknown_selector_raises(tmp_path):
    _parent(tmp_path)
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: parent.yaml
        disable: [does_not_exist]
        plugins: []
        """,
    )
    with pytest.raises(PluginError, match="matched no inherited plugin"):
        load_config(child)


def test_disable_without_extends_raises(tmp_path):
    child = _write(
        tmp_path,
        "child.yaml",
        """
        disable: [foo]
        plugins: []
        """,
    )
    with pytest.raises(PluginError, match="'disable' requires 'extends'"):
        load_config(child)


def test_extends_cycle_raises(tmp_path):
    _write(tmp_path, "a.yaml", "extends: b.yaml\nplugins: []\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\nplugins: []\n")
    with pytest.raises(PluginError, match="cycle detected"):
        load_config(tmp_path / "a.yaml")


def test_extends_missing_target_raises(tmp_path):
    child = _write(tmp_path, "child.yaml", "extends: nope.yaml\nplugins: []\n")
    with pytest.raises(PluginError, match="world YAML not found"):
        load_config(child)


def test_extends_package_ref_resolves(tmp_path):
    # A real installed world resolves through its `<package>:<world>` ref and brings its own plugin
    # stack; the child appends a robot on top. Uses a shipped world on purpose: the point is that the
    # entry-point lookup works, which a tmp_path fixture cannot exercise.
    child = _write(
        tmp_path,
        "child.yaml",
        """
        extends: roqsim_scenes:depot
        sim:
          timestep: 0.001
        plugins:
          - spawn_robot: {model: turtlebot4, prefix: robot_, pos: [-8.0, 0.0]}
            name: robot
        """,
    )
    cfg = load_config(child)
    assert cfg.timestep == 0.001
    assert cfg.declared[-1].ref == "spawn_robot"
    assert cfg.sim["world"].endswith("worlds/depot/depot.xml")
    assert len(cfg.plugins) > 1  # the parent's own plugins are inherited
