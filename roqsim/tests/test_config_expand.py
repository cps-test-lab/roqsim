"""Config: Plugin.expand splices extra plugin specs into the pipeline (per-plugin dedupe is the
producing plugin's job -- see test_manifest.py -- so core just splices what expand returns)."""

from __future__ import annotations

from roqsim.config import PluginSpec, instantiate_plugins, load_config_from_dict
from roqsim.plugin import Plugin


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
