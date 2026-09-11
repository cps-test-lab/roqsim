"""Reporting the roqsim.plugins registry as JSON: what a caller with no repo access can learn.

The point of this module is to answer "what plugins exist, and what does each one's Config::
block accept" without importing roqsim -- so its own tests hold it to that promise: every
plugin already installed in this dev environment must parse cleanly, and a name that isn't
registered must report an error rather than a crash or a silently empty result.
"""

from __future__ import annotations

from roqsim.introspection import (
    _config_header_span,
    _own_or_module_doc,
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



# -- the shapes a real Config:: block is written in --------------------------------

def _fields(doc):
    return {f["name"]: f for f in _parse_config_block(doc)}


def test_a_header_whose_qualifier_wraps_still_opens_the_block():
    """The ``::`` need not sit on the same line as the word.

    A plugin that lists the keys it inherits qualifies the header, and the qualifier is
    routinely longer than a line. Requiring both on one line reported such a plugin as having
    no configuration at all -- while its keys sat documented directly underneath.
    """
    doc = (
        "Sensor plugin: a fan of rays.\n\n"
        "Config (in addition to ``lidar_common``'s ``namespace``/``site``/\n"
        "``range_min``/``max_range``/``rate_hz``)::\n\n"
        "    lidar:\n"
        "      rays: 360\n"
        "      angle_min: 0.0\n"
    )
    assert set(_fields(doc)) == {"rays", "angle_min"}


def test_prose_opening_with_the_word_is_not_a_header():
    doc = (
        "Config is read from the world YAML and validated on load.\n\n"
        "Some other paragraph.\n"
    )
    assert _parse_config_block(doc) == []


def test_nested_keys_are_reported_at_the_path_a_world_yaml_writes_them_at():
    """A key opening a mapping used to END the block, hiding every key after it."""
    doc = (
        "Scene plugin: report coverage.\n\n"
        "Config::\n\n"
        "    sensor_coverage_probe:\n"
        "      sensors: auto            # 'auto' = every camera\n"
        "      sample:\n"
        "        volume: true\n"
        "        resolution: 0.25\n"
        "      out: coverage            # output directory\n"
    )
    fields = _fields(doc)
    assert set(fields) == {"sensors", "sample", "sample.volume", "sample.resolution", "out"}
    # The key after the nested mapping is reached, and reported at the top level.
    assert fields["out"]["doc"] == "output directory"
    # The mapping itself carries no example -- its children are the value.
    assert fields["sample"]["example"] is None


def test_the_line_naming_the_plugin_is_not_itself_a_field():
    doc = "Plugin.\n\nConfig::\n\n    contact_monitor:\n      min_force: 1.0\n"
    assert set(_fields(doc)) == {"min_force"}


def test_a_block_written_as_a_components_list_entry_parses_too():
    """World YAML's ``components:`` takes a list, and some blocks are written that way."""
    doc = (
        "Spawn a sensor.\n\n"
        "Config::\n\n"
        "    - spawn_sensor:\n"
        "        model: d435            # bundled model name\n"
        "        prefix: \"\"\n"
    )
    assert set(_fields(doc)) == {"model", "prefix"}


def test_a_class_docstring_pointing_at_the_module_does_not_hide_the_block():
    """``\"\"\"See the module docstring.\"\"\"`` used to win and publish the pointer."""
    import sys
    import types

    module = types.ModuleType("_roqsim_test_pointer_module")
    module.__doc__ = "IMU sensor.\n\nConfig::\n\n    imu:\n      rate_hz: 100.0\n"
    sys.modules[module.__name__] = module
    try:
        class Pointer:
            """See the module docstring."""
        Pointer.__module__ = module.__name__
        doc = _own_or_module_doc(Pointer)
        assert set(_fields(doc)) == {"rate_hz"}
    finally:
        del sys.modules[module.__name__]


def test_a_class_docstring_that_documents_config_itself_still_wins():
    import sys
    import types

    module = types.ModuleType("_roqsim_test_module_block")
    module.__doc__ = "Module.\n\nConfig::\n\n    x:\n      from_module: 1\n"
    sys.modules[module.__name__] = module
    try:
        class OwnBlock:
            """Class.

            Config::

                x:
                  from_class: 1
            """
        OwnBlock.__module__ = module.__name__
        assert set(_fields(_own_or_module_doc(OwnBlock))) == {"from_class"}
    finally:
        del sys.modules[module.__name__]


def test_every_installed_plugin_with_a_config_block_reports_at_least_one_key():
    """A block that is present but unreadable is the failure this guards.

    It reads as "this plugin takes no configuration", which is a wrong answer rather than a
    missing one -- and the reader has no way to tell the two apart.
    """
    silent = []
    for item in list_plugins()["items"]:
        details = get_plugin_details(item["name"])
        if "error" in details:
            continue
        doc = details.get("doc") or ""
        if _config_header_span(doc.splitlines()) is not None and not details["parameters"]:
            silent.append(item["name"])
    assert not silent, f"Config:: block present but no keys parsed: {silent}"
