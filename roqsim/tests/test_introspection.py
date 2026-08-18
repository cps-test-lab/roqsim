"""Reporting the roqsim.plugins registry as JSON: what a caller with no repo access can learn.

The point of this module is to answer "what plugins exist, and what does each one's Config::
block accept" without importing roqsim -- so its own tests hold it to that promise: every
plugin already installed in this dev environment must parse cleanly, and a name that isn't
registered must report an error rather than a crash or a silently empty result.
"""

from __future__ import annotations

from roqsim.introspection import (
    _parse_config_block,
    get_plugin_details,
    list_plugins,
)


def test_list_plugins_includes_dummy_with_a_doc():
    """dummy is core roqsim's own no-op plugin: always registered, a stable fixture."""
    catalog = list_plugins()
    dummy = next((item for item in catalog["items"] if item["name"] == "dummy"), None)
    assert dummy is not None, "expected the always-registered 'dummy' plugin"
    assert dummy["kind"] == "plugin"
    assert dummy["doc"]


def test_list_plugins_sorted_by_name():
    names = [item["name"] for item in list_plugins()["items"]]
    assert names == sorted(names)


def test_get_plugin_details_dummy_has_size_field():
    details = get_plugin_details("dummy")
    assert "error" not in details
    fields = {p["name"]: p for p in details["parameters"]}
    assert "size" in fields


def test_get_plugin_details_contact_monitor_min_force_doc_intact():
    details = get_plugin_details("contact_monitor")
    fields = {p["name"]: p for p in details["parameters"]}
    assert "min_force" in fields
    assert "contacts below this normal force are ignored" in fields["min_force"]["doc"]


def test_get_plugin_details_unknown_name_is_error_not_exception():
    result = get_plugin_details("not_a_real_plugin_xyz")
    assert "error" in result


class TestParseConfigBlock:
    """Pure parsing, independent of any installed plugin."""

    def test_single_line_comment_per_field(self):
        doc = (
            "Summary.\n\nConfig::\n\n    my_plugin:\n"
            '      size: 0.1        # radius in metres\n'
            "      enabled: true    # turn the effect on\n"
        )
        fields = _parse_config_block(doc)
        assert [f["name"] for f in fields] == ["size", "enabled"]
        assert fields[0]["example"] == "0.1"
        assert fields[0]["doc"] == "radius in metres"

    def test_wrapped_comment_extends_previous_field_not_a_new_one(self):
        # Regression guard: a bare "# ..." continuation line (no leading "name:")
        # must extend the previous field's doc, not end the block early and lose
        # every field that follows it.
        doc = (
            "Summary.\n\nConfig::\n\n    my_plugin:\n"
            "      enabled: true      # first part of a long explanation\n"
            "                         # continues here\n"
            "      above_z: 2.5       # a second, later field\n"
        )
        fields = _parse_config_block(doc)
        assert [f["name"] for f in fields] == ["enabled", "above_z"]
        assert fields[0]["doc"] == "first part of a long explanation continues here"

    def test_trailing_prose_after_block_is_not_mistaken_for_a_field(self):
        doc = (
            "Summary.\n\nConfig::\n\n    my_plugin:\n"
            "      size: 0.1   # radius\n"
            "\n"
            "Set size to change the radius. See docs for details.\n"
        )
        fields = _parse_config_block(doc)
        assert [f["name"] for f in fields] == ["size"]

    def test_no_config_block_returns_empty_list(self):
        assert _parse_config_block("Just a summary, no Config:: block.") == []

    def test_field_with_no_trailing_comment_has_none_doc(self):
        doc = "Summary.\n\nConfig::\n\n    my_plugin:\n      size: 0.1\n"
        fields = _parse_config_block(doc)
        assert fields[0]["doc"] is None

    def test_parenthetical_header_recognized(self):
        # "Config (in addition to X's Y/Z)::" -- oakd_camera.py's and
        # seyond_robin_w1g.py's real header shape, not a bare "Config::".
        doc = (
            "Summary.\n\n"
            "Config (in addition to base's fields)::\n\n"
            "    my_plugin:\n      extra: true   # an extra field\n"
        )
        fields = _parse_config_block(doc)
        assert [f["name"] for f in fields] == ["extra"]

