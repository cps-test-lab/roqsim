# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""The one entry-point scan every registry reads."""

from __future__ import annotations

import pytest

from roqsim import commands, models, registry, world
from roqsim.entry_points import entry_points
from roqsim.plugin import PluginError


def test_a_group_is_scanned_once_and_read_by_every_registry():
    """Plugins, models and worlds all answer from the same cached scan, so a package's registration
    is seen the same way whichever door asks -- and the scan is not paid for again per plugin."""
    plugins = entry_points(registry.ENTRY_POINT_GROUP)
    assert entry_points(registry.ENTRY_POINT_GROUP) is plugins
    assert {ep.name for ep in plugins} >= {"spawn_model", "dummy"}
    assert registry._entry_points is entry_points
    assert models._entry_points is entry_points
    assert commands.entry_points is entry_points
    assert world._world_entry_points() == entry_points(world.WORLDS_ENTRY_POINT_GROUP)


def test_an_unregistered_group_is_empty_rather_than_an_error():
    assert entry_points("roqsim.no_such_group") == ()


class _EP:
    def __init__(self, value: str, dist: str) -> None:
        self.name, self.value = "twice", value
        self.dist = type("Dist", (), {"name": dist})()

    def load(self):
        raise AssertionError("an ambiguous name is refused before anything is loaded")


def test_a_name_two_packages_register_is_refused_naming_both(monkeypatch):
    """Whichever the metadata lists first would win, and a world would get another plugin silently."""
    eps = (_EP("pkg_a.plugins:Twice", "pkg-a"), _EP("pkg_b.plugins:Twice", "pkg-b"))
    monkeypatch.setattr(registry, "_entry_points", lambda group: eps)
    with pytest.raises(PluginError, match=r"more than one installed package") as refused:
        registry.resolve_plugin("twice")
    assert "pkg_a.plugins:Twice (pkg-a)" in str(refused.value)
    assert "pkg_b.plugins:Twice (pkg-b)" in str(refused.value)
    with pytest.raises(PluginError, match="renderer 'twice'"):
        registry.resolve_renderer("twice")


def test_one_entry_point_listed_twice_is_one_registration(monkeypatch):
    """An editable install and its metadata can both list a package: the same target is one."""
    from roqsim.plugin import Plugin

    class Twice(Plugin):
        pass

    class Listed(_EP):
        def load(self):
            return Twice

    monkeypatch.setattr(
        registry, "_entry_points", lambda group: (Listed("m:T", "pkg"), Listed("m:T", "pkg"))
    )
    assert registry.resolve_plugin("twice") is Twice
