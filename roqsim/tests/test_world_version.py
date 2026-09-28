"""A world states its version, and a newer one is refused by name.

A world written for a later roqsim can use a key that means something else here. Read with the keys
that happen to overlap, it runs a different experiment while looking correct. Absent means 1.
"""

from __future__ import annotations

import pytest

from roqsim.config import WORLD_VERSION, PluginError, load_config, load_config_from_dict


def _write(path, text):
    path.write_text(text)
    return path


def test_absent_means_version_1(tmp_path):
    cfg = load_config_from_dict({"sim": {"timestep": 0.002}}, tmp_path)
    assert cfg.sim == {"timestep": 0.002}


def test_the_current_version_loads_and_is_not_passed_on(tmp_path):
    cfg = load_config_from_dict({"version": WORLD_VERSION, "sim": {"timestep": 0.002}}, tmp_path)
    assert "version" not in cfg.raw
    assert cfg.sim == {"timestep": 0.002}


def test_a_newer_version_is_refused_naming_both(tmp_path):
    with pytest.raises(
        PluginError, match=rf"world version {WORLD_VERSION + 1}.*reads up to {WORLD_VERSION}"
    ):
        load_config_from_dict({"version": WORLD_VERSION + 1}, tmp_path)


@pytest.mark.parametrize("bad", [0, -1, "1", 1.5, True, None])
def test_a_version_that_is_not_a_positive_integer_is_refused(tmp_path, bad):
    with pytest.raises(PluginError, match="version"):
        load_config_from_dict({"version": bad}, tmp_path)


def test_every_document_in_an_extends_chain_is_checked(tmp_path):
    """A parent is read too, so a parent stating a newer version is refused like a leaf."""
    _write(tmp_path / "parent.yaml", f"version: {WORLD_VERSION + 1}\nsim: {{timestep: 0.002}}\n")
    leaf = _write(tmp_path / "leaf.yaml", f"version: {WORLD_VERSION}\nextends: parent.yaml\n")
    with pytest.raises(PluginError, match=r"parent\.yaml.*version"):
        load_config(leaf)


def test_a_chain_of_current_documents_loads(tmp_path):
    _write(tmp_path / "parent.yaml", f"version: {WORLD_VERSION}\nsim: {{timestep: 0.002}}\n")
    leaf = _write(tmp_path / "leaf.yaml", "extends: parent.yaml\n")
    assert load_config(leaf).sim["timestep"] == 0.002
