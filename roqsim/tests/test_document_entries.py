# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The document key that holds a list of entries: ``components:``.

``plugins:`` is the former spelling, still accepted so worlds and manifests can be swept over
separately from the behaviour change. Both in one document is refused rather than merged -- two
spellings of one key in one file is a merge nobody can predict.
"""

import pytest

from roqsim.config import PluginError, document_entries, load_config_from_dict, with_transport


def _refs(raw):
    return [s.ref for s in load_config_from_dict(raw).plugins]


def test_either_spelling_loads_the_same_world():
    assert _refs({"sim": {}, "components": [{"dummy": {}}]}) == ["dummy"]
    assert _refs({"sim": {}, "plugins": [{"dummy": {}}]}) == ["dummy"]


def test_both_spellings_in_one_document_are_refused_by_name():
    """Silently preferring one would make the other's entries vanish without a word."""
    with pytest.raises(PluginError) as exc:
        document_entries({"components": [], "plugins": []}, "w.yaml")
    msg = str(exc.value)
    assert "w.yaml" in msg
    assert "components" in msg and "plugins" in msg


def test_a_document_with_neither_key_has_no_entries():
    assert document_entries({"sim": {}}) == []


def test_the_loader_normalises_onto_the_current_spelling():
    """Anything reading the loaded document back -- describe, the exporters -- sees one key."""
    out = with_transport({"plugins": [{"floorplan": {}}]}, ros=True)
    assert "plugins" not in out
    assert [k for e in out["components"] for k in e] == ["floorplan", "ros2_bridge"]
