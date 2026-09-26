"""A world states the format it was written to, and a newer one is refused by name.

A campaign archives its worlds, and a world written for a later roqsim can use a key that means
something else here, or that this version does not know. Read with the keys that happen to overlap,
it loads and runs a different experiment while looking correct. ``format:`` is the top-level stamp;
absent means 1, the format every world written before the stamp existed is in.
"""

from __future__ import annotations

import pytest

from roqsim.config import WORLD_FORMAT, PluginError, load_config, load_config_from_dict


def _write(path, text):
    path.write_text(text)
    return path


def test_absent_means_the_first_format(tmp_path):
    cfg = load_config_from_dict({"sim": {"timestep": 0.002}}, tmp_path)
    assert cfg.sim == {"timestep": 0.002}


def test_the_current_format_loads_and_is_not_passed_on(tmp_path):
    cfg = load_config_from_dict({"format": WORLD_FORMAT, "sim": {"timestep": 0.002}}, tmp_path)
    assert "format" not in cfg.raw
    assert cfg.sim == {"timestep": 0.002}


def test_a_newer_format_is_refused_naming_both(tmp_path):
    with pytest.raises(
        PluginError, match=rf"format {WORLD_FORMAT + 1}.*reads up to {WORLD_FORMAT}"
    ):
        load_config_from_dict({"format": WORLD_FORMAT + 1}, tmp_path)


@pytest.mark.parametrize("bad", [0, -1, "1", 1.5, True, None])
def test_a_format_that_is_not_a_positive_integer_is_refused(tmp_path, bad):
    with pytest.raises(PluginError, match="format"):
        load_config_from_dict({"format": bad}, tmp_path)


def test_every_document_in_an_extends_chain_is_checked(tmp_path):
    """A parent is read too, so a parent written to a newer format is refused like a leaf."""
    _write(tmp_path / "parent.yaml", f"format: {WORLD_FORMAT + 1}\nsim: {{timestep: 0.002}}\n")
    leaf = _write(tmp_path / "leaf.yaml", f"format: {WORLD_FORMAT}\nextends: parent.yaml\n")
    with pytest.raises(PluginError, match=r"parent\.yaml.*format"):
        load_config(leaf)


def test_a_chain_of_current_documents_loads(tmp_path):
    _write(tmp_path / "parent.yaml", f"format: {WORLD_FORMAT}\nsim: {{timestep: 0.002}}\n")
    leaf = _write(tmp_path / "leaf.yaml", "extends: parent.yaml\n")
    assert load_config(leaf).sim["timestep"] == 0.002
