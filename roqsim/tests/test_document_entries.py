# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The document key that holds a list of entries is ``components:``, and a world has no other
top-level key than the ones it reads: any other is refused, naming it."""

import pytest

from roqsim.config import PluginError, document_entries, load_config, load_config_from_dict


def test_the_entries_are_read_from_components():
    assert [
        s.ref for s in load_config_from_dict({"sim": {}, "components": [{"dummy": {}}]}).plugins
    ] == ["dummy"]


def test_a_document_without_components_has_no_entries():
    assert document_entries({"sim": {}}) == []


@pytest.mark.parametrize("key", ["plugins", "robots"])
def test_an_unknown_top_level_key_is_refused_naming_the_file(tmp_path, key):
    """Nothing reads it, so its entries would vanish without a word."""
    world = tmp_path / "w.yaml"
    world.write_text(f"sim: {{}}\n{key}:\n  - dummy: {{}}\n")
    with pytest.raises(PluginError, match=rf"{world}: unknown key\(s\) '{key}'"):
        load_config(world)


def test_an_unknown_top_level_key_in_an_extended_world_is_refused(tmp_path):
    (tmp_path / "parent.yaml").write_text("sim: {}\nplugins: []\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\ncomponents: []\n")
    with pytest.raises(PluginError, match=r"parent\.yaml: unknown key\(s\) 'plugins'"):
        load_config(child)


def test_an_override_under_an_unknown_top_level_key_is_refused():
    with pytest.raises(PluginError, match=r"world override: unknown key\(s\) 'plugins'"):
        load_config_from_dict(
            {"sim": {}, "components": [{"dummy": {}}]}, overrides={"plugins": {"dummy": {}}}
        )


def test_a_misspelt_top_level_key_names_the_one_it_meant():
    with pytest.raises(PluginError, match=r"'componets' \(did you mean 'components'\?\)"):
        load_config_from_dict({"componets": []})


def test_the_refusal_says_where_a_value_that_varies_per_run_goes():
    """A parameter block is the key most often invented here; a world has none."""
    with pytest.raises(PluginError) as exc:
        load_config_from_dict({"parameters": {"box_offset_y": 0.0}, "components": []})
    msg = str(exc.value)
    assert "'parameters'" in msg
    assert "components.<name>.<key>" in msg


@pytest.mark.parametrize(
    "override",
    [{"box_offset_y": 0.03}, {"extends": "other.yaml"}, {"disable": ["a"]}, {"version": 2}],
)
def test_an_override_rooted_outside_sim_and_components_is_refused(override):
    """Inheritance and the version stamp are consumed before overrides merge, so an override there
    would be read by nothing."""
    (root,) = override
    with pytest.raises(PluginError, match=rf"world override: unknown key\(s\) '{root}'"):
        load_config_from_dict({"sim": {}, "components": []}, overrides=override)
