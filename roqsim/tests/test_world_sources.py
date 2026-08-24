# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""What a world *depends on*, which is more than the files it is written in.

A caller that caches something compiled from a world -- an export, a baked scene -- needs
the whole set or its staleness check is wrong in the silent direction. The YAML chain and
the MJCF's assets are found by walking; what a plugin points at can only be asked of the
plugin, and that is what these pin.
"""

import textwrap

from roqsim.config import world_sources
from roqsim.plugin import Plugin


class _NamesAFile(Plugin):
    def sources(self):
        return [self.config["file"]]


class _Explodes(Plugin):
    def sources(self):
        raise RuntimeError("this plugin's sources() is broken")


def _world(tmp_path, body: str):
    path = tmp_path / "world.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_a_plugins_own_file_is_a_source(tmp_path):
    """The case the YAML walk cannot see: a path in a plugin's config, not in the world."""
    asset = tmp_path / "rooms.stl"
    asset.write_bytes(b"solid\n")
    world = _world(
        tmp_path,
        f"""\
        plugins:
          - "{__name__}:_NamesAFile":
              file: {asset}
    """,
    )
    assert asset.resolve() in world_sources(world)


def test_an_unresolvable_plugin_does_not_sink_the_answer(tmp_path):
    """A ROS world in a pip-only environment cannot resolve its bridge.

    Enumerating dependencies is best-effort by contract, and raising here would pre-empt the
    caller's own error with one about a plugin they did not ask about.
    """
    world = _world(
        tmp_path,
        """\
        plugins:
          - not_a_real_plugin_anywhere: {}
    """,
    )
    assert world_sources(world) == [world.resolve()]


def test_a_plugin_whose_sources_raises_is_skipped(tmp_path):
    world = _world(
        tmp_path,
        f"""\
        plugins:
          - "{__name__}:_Explodes": {{}}
    """,
    )
    assert world_sources(world) == [world.resolve()]


def test_a_source_that_does_not_exist_is_dropped(tmp_path):
    """An optional file a plugin names but that is absent is not a dependency."""
    world = _world(
        tmp_path,
        f"""\
        plugins:
          - "{__name__}:_NamesAFile":
              file: {tmp_path / "absent.stl"}
    """,
    )
    assert world_sources(world) == [world.resolve()]


def test_a_manifest_supplied_components_file_counts_as_an_input(tmp_path, monkeypatch):
    """`prop_trajectory` implements `expand` for one reason: to hand its `sources()` the directory
    the world lives in. Expansion used to happen inside the engine, so this walk never saw it and
    the CSV resolved against the CALLER's working directory -- against what its docstring promises.
    """
    (tmp_path / "route.csv").write_text("t,x,y\n0,0,0\n1,1,0\n")
    world = tmp_path / "w.yaml"
    world.write_text(
        "sim: {}\n"
        "components:\n"
        "  - prop_trajectory: {model: industrial_table, path: route.csv}\n"
        "    name: mover\n"
    )
    monkeypatch.chdir(tmp_path.parent)  # a caller who is not standing in the world's directory
    assert tmp_path / "route.csv" in world_sources(world)
