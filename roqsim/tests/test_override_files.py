# SPDX-License-Identifier: Apache-2.0
"""``--override``: the file spelling of ``--set``.

A command line cannot carry anything structured -- a list of obstacle instances, a nested
plugin config -- without losing it to quoting and word splitting. A file can, and it is also
something a run's results can keep and a person can replay from.
"""

import pytest
import yaml

from roqsim.config import apply_overrides, deep_merge, overrides_from_dotlist, overrides_from_files
from roqsim.plugin import PluginError


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.dump(data))
    return str(path)


def test_a_file_is_the_same_nested_mapping_set_builds(tmp_path):
    path = _write(tmp_path, "o.yaml",
                  {"plugins": {"floorplan": {"floor": {"reflectance": 0.3}}}})
    assert (overrides_from_files([path])
            == overrides_from_dotlist(["plugins.floorplan.floor.reflectance=0.3"]))


def test_it_carries_what_a_command_line_cannot(tmp_path):
    """The reason the file form exists: a list of instances survives it intact."""
    instances = [{"pos": [2.1, -3.4], "size": [0.5, 0.5, 1.0]},
                 {"pos": [5.8, -1.2], "size": [0.5, 0.5, 1.0]}]
    path = _write(tmp_path, "o.yaml", {"plugins": {"boxes": {"instances": instances}}})
    assert overrides_from_files([path])["plugins"]["boxes"]["instances"] == instances


def test_several_files_merge_with_the_later_one_winning(tmp_path):
    base = _write(tmp_path, "base.yaml",
                  {"plugins": {"floorplan": {"size": 3.0, "floor": {"reflectance": 0.2}}}})
    tweak = _write(tmp_path, "tweak.yaml", {"plugins": {"floorplan": {"size": 4.2}}})
    merged = overrides_from_files([base, tweak])
    assert merged["plugins"]["floorplan"] == {"size": 4.2, "floor": {"reflectance": 0.2}}


def test_set_wins_over_a_file(tmp_path):
    """A saved override set plus one ad-hoc tweak: the tweak is what should win."""
    base = _write(tmp_path, "base.yaml", {"sim": {"pacing": "realtime"}})
    merged = deep_merge(overrides_from_files([base]),
                        overrides_from_dotlist(["sim.pacing=asap"]))
    assert merged["sim"]["pacing"] == "asap"


def test_an_empty_document_contributes_nothing(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert overrides_from_files([str(path)]) == {}


def test_no_files_is_no_overrides():
    assert overrides_from_files(None) == {}
    assert overrides_from_files([]) == {}


def test_a_document_that_is_not_a_mapping_is_refused(tmp_path):
    path = _write(tmp_path, "list.yaml", [1, 2])
    with pytest.raises(PluginError, match="must contain a mapping"):
        overrides_from_files([path])


def test_a_missing_file_is_refused_by_name(tmp_path):
    """Loudly: an override silently skipped is a run against a world nobody asked for."""
    with pytest.raises(PluginError, match="could not be read"):
        overrides_from_files([str(tmp_path / "nope.yaml")])


def test_a_broken_document_is_refused_by_name(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("plugins: [unclosed\n")
    with pytest.raises(PluginError, match="not valid YAML"):
        overrides_from_files([str(path)])


def test_the_result_is_what_apply_overrides_takes(tmp_path):
    """The whole point: the file lands in the same call ``--set`` funnels into."""
    world = {"plugins": [{"floorplan": {"name": "floorplan", "size": 3.0}}]}
    path = _write(tmp_path, "o.yaml", {"plugins": {"floorplan": {"size": 4.2}}})
    applied = apply_overrides(world, overrides_from_files([path]))
    assert applied["plugins"][0]["floorplan"]["size"] == 4.2
    # Modified in place, not appended: a second floorplan would be a different world.
    assert len(applied["plugins"]) == 1
