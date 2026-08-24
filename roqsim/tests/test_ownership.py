# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Ownership is the shape of the document: a component belongs to the entry it is nested under.

There is nothing for a plugin to parse and therefore nothing to spell wrong. The two ways a document
can contradict itself -- an attacher with nothing to attach to, an owner that owns nothing -- are
refused by name, because both used to be silent and both cost a debugging session.
"""

import pytest

from roqsim.config import PluginError, instantiate_plugins, load_config_from_dict
from roqsim.plugin import Plugin


class Spawner(Plugin):
    provides_entity = True


class Attacher(Plugin):
    requires_owner = True


SPAWNER = f"{__name__}:Spawner"
ATTACHER = f"{__name__}:Attacher"


def _cfg(entries):
    return load_config_from_dict({"sim": {}, "components": entries})


def test_a_nested_entry_is_wired_to_the_entry_it_sits_under():
    cfg = _cfg([{SPAWNER: {}, "name": "robot", "components": [{ATTACHER: {}, "name": "lidar"}]}])
    assert [(s.address, s.entity) for s in cfg.plugins] == [
        ("robot", None),
        ("robot.lidar", "robot"),
    ]
    lidar = instantiate_plugins(cfg)[1]
    assert lidar.entity == "robot"


def test_the_owner_builds_before_what_attaches_to_it():
    """Build order falls out of the document's shape -- a spawn must run before its sensors."""
    cfg = _cfg(
        [
            {SPAWNER: {}, "name": "a", "components": [{ATTACHER: {}, "name": "s"}]},
            {SPAWNER: {}, "name": "b"},
        ]
    )
    assert [s.address for s in cfg.plugins] == ["a", "a.s", "b"]


def test_two_owners_may_each_have_a_component_of_the_same_kind():
    """Labels are unique within one owner, not across the world: `a.lidar` and `b.lidar` coexist."""
    cfg = _cfg(
        [
            {SPAWNER: {}, "name": "a", "components": [{ATTACHER: {}, "name": "lidar"}]},
            {SPAWNER: {}, "name": "b", "components": [{ATTACHER: {}, "name": "lidar"}]},
        ]
    )
    assert [s.address for s in cfg.plugins if s.entity] == ["a.lidar", "b.lidar"]


def test_an_attacher_at_the_top_of_a_document_is_refused_with_the_fix():
    """It has nothing to attach to. This used to fall back to the literal name 'robot' and run
    ALONGSIDE the default it meant to replace -- a config that silently had no effect."""
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(_cfg([{ATTACHER: {}}]))
    msg = str(exc.value)
    assert "nested under" in msg and "components:" in msg


def test_components_on_an_entry_that_registers_no_entity_is_refused():
    """There is nothing for the children to attach to, so wiring them would name no entity."""
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(_cfg([{"dummy": {}, "components": [{ATTACHER: {}}]}]))
    assert "registers no entity" in str(exc.value)


def test_a_components_block_must_be_a_list():
    with pytest.raises(PluginError) as exc:
        _cfg([{SPAWNER: {}, "components": {"lidar": {}}}])
    assert "must be a list" in str(exc.value)


def test_the_label_of_an_unnamed_entry_is_its_ref_not_its_class_name():
    """`Plugin.name` falls back to the class name; a label must not, or a dotted ref would put a
    dot in an address and an entry-point ref would address by a name nothing in the YAML uses."""
    (spec,) = _cfg([{"dummy": {}}]).plugins
    assert spec.label == "dummy"
    (spec,) = _cfg([{SPAWNER: {}}]).plugins
    assert spec.label == "Spawner"  # module:Class ref -> the class part, which carries no dot


# -- addresses and the invariants that keep them unambiguous ----------------------------------


def test_an_address_is_qualified_all_the_way_down():
    """Labels are unique among siblings only, so an address carries the whole path: two robots each
    carrying an `arm` are two entities, not one hiding the other."""
    cfg = _cfg(
        [
            {SPAWNER: {}, "name": "r1", "components": [{SPAWNER: {}, "name": "arm"}]},
            {SPAWNER: {}, "name": "r2", "components": [{SPAWNER: {}, "name": "arm"}]},
        ]
    )
    assert [s.address for s in cfg.plugins] == ["r1", "r1.arm", "r2", "r2.arm"]
    assert [s.entity for s in cfg.plugins] == [None, "r1", None, "r2"]


def test_a_top_level_entrys_address_is_just_its_label():
    """So no existing world's entity names change: qualification only appears where nesting does."""
    assert [s.address for s in _cfg([{SPAWNER: {}, "name": "robot"}]).plugins] == ["robot"]


def test_two_components_of_one_owner_may_not_share_a_label():
    with pytest.raises(PluginError) as exc:
        _cfg([{SPAWNER: {}, "name": "r", "components": [{ATTACHER: {}}, {ATTACHER: {}}]}])
    assert "two components labelled" in str(exc.value)


def test_the_same_label_under_two_different_owners_is_fine():
    cfg = _cfg(
        [
            {SPAWNER: {}, "name": "a", "components": [{ATTACHER: {}, "name": "lidar"}]},
            {SPAWNER: {}, "name": "b", "components": [{ATTACHER: {}, "name": "lidar"}]},
        ]
    )
    assert [s.address for s in cfg.plugins if s.entity] == ["a.lidar", "b.lidar"]


@pytest.mark.parametrize("bad", ["a.b", "a b", "a*b", "a#b", "a[0]", "a=b", "a:b", "a/b"])
def test_a_name_an_address_cannot_spell_is_refused(bad):
    """`.` separates address segments; the rest are shapes the grammar may grow into, kept free so
    adding one later cannot invalidate a document that was legal when it was written."""
    with pytest.raises(PluginError, match="an address cannot spell"):
        _cfg([{"dummy": {}, "name": bad}])


def test_a_child_label_may_not_collide_with_its_owners_config_key():
    """`components.robot.pos` has to mean one thing."""
    with pytest.raises(PluginError) as exc:
        _cfg(
            [
                {
                    SPAWNER: {"pos": [0, 0]},
                    "name": "robot",
                    "components": [{ATTACHER: {}, "name": "pos"}],
                }
            ]
        )
    assert "would be ambiguous" in str(exc.value)
