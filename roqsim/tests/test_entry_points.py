# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""The one entry-point scan every registry reads."""

from __future__ import annotations

from roqsim import models, registry, world
from roqsim.entry_points import entry_points


def test_a_group_is_scanned_once_and_read_by_every_registry():
    """Plugins, models and worlds all answer from the same cached scan, so a package's registration
    is seen the same way whichever door asks -- and the scan is not paid for again per plugin."""
    plugins = entry_points(registry.ENTRY_POINT_GROUP)
    assert entry_points(registry.ENTRY_POINT_GROUP) is plugins
    assert {ep.name for ep in plugins} >= {"spawn_model", "dummy"}
    assert registry._entry_points is entry_points
    assert models._entry_points is entry_points
    assert world._world_entry_points() == entry_points(world.WORLDS_ENTRY_POINT_GROUP)


def test_an_unregistered_group_is_empty_rather_than_an_error():
    assert entry_points("roqsim.no_such_group") == ()
