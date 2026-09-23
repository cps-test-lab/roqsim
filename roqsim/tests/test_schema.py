# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim.schema``: one declaration, checked at load and published to a caller.

The rules worth pinning are the ones a hand-written check gets right by accident and a shared one
must get right on purpose: a YAML ``1`` satisfies a float, a ``true`` does NOT, an unknown key is
only an error where a plugin says its list is complete, and every message names its key.
"""

from __future__ import annotations

import pytest

from roqsim.plugin import Plugin
from roqsim.schema import INJECTED_KEYS, Field, describe, validate

SCHEMA = {
    "mass": Field(float, required=True, minimum=0.0, unit="kg", doc="what it weighs"),
    "mode": Field(str, default="soft", choices=("soft", "rigid")),
    "pos": Field(list, length=3, unit="m"),
    "count": Field(int, default=1, minimum=1, maximum=8),
    "loud": Field(bool, default=False),
}


def _errors(config, **kwargs):
    return validate(SCHEMA, config, **kwargs)


# -- types --------------------------------------------------------------------------------------


def test_an_integer_satisfies_a_float_because_yaml_writes_one_for_one_metre():
    assert _errors({"mass": 1}) == []
    assert _errors({"mass": 1.5}) == []


def test_a_bool_does_not_satisfy_a_number():
    """Python says True == 1; a world that wrote `count: true` did not mean one of something."""
    errors = _errors({"mass": 1.0, "count": True})
    assert any("'count' must be int" in e for e in errors)


def test_a_wrong_type_is_reported_once_and_stops_the_other_checks_on_that_key():
    """A range check against a string is noise on top of the error the reader has to fix."""
    errors = _errors({"mass": "heavy"})
    assert errors == ["'mass' must be float, got str ('heavy')"]


# -- a key that takes more than one shape --------------------------------------------------------

UNION = {"gain": Field((float, dict), default=0.0, minimum=0.0, unit="W")}


def test_a_union_accepts_each_of_its_shapes():
    for value in (0.5, 2, {"shoulder": 0.1}, {}):
        assert validate(UNION, {"gain": value}) == [], value


def test_a_union_refuses_what_is_none_of_them_and_names_every_shape():
    assert validate(UNION, {"gain": "lots"}) == ["'gain' must be float or dict, got str ('lots')"]
    assert validate(UNION, {"gain": [0.1]}) == ["'gain' must be float or dict, got list ([0.1])"]


def test_a_bool_is_still_not_a_number_inside_a_union():
    assert validate(UNION, {"gain": True}) == ["'gain' must be float or dict, got bool (True)"]


def test_a_bound_applies_to_the_number_and_not_to_the_mapping():
    """A mapping has no order against 0; its entries are the plugin's to check."""
    assert validate(UNION, {"gain": -1.0}) == ["'gain' must be >= 0.0 W, got -1.0"]
    assert validate(UNION, {"gain": {"shoulder": -1.0}}) == []


def test_a_union_is_published_as_a_list_of_its_names():
    (gain,) = describe(UNION)
    assert gain["type"] == ["float", "dict"]
    assert describe({"x": Field(float)})[0]["type"] == "float"


# -- rules --------------------------------------------------------------------------------------


def test_a_required_key_is_named_with_the_reason_it_exists():
    errors = _errors({})
    assert errors == ["'mass' is required -- what it weighs"]


def test_bounds_choices_and_lengths_all_name_the_key_and_the_limit():
    errors = _errors({"mass": -1.0, "mode": "springy", "pos": [0, 0], "count": 99})
    assert "'mass' must be >= 0.0 kg, got -1.0" in errors
    assert "'mode' must be one of soft, rigid, got 'springy'" in errors
    assert "'pos' must have exactly 3 entries, got 2" in errors
    assert "'count' must be <= 8, got 99" in errors


def test_every_problem_is_reported_at_once():
    """A world with three mistakes should take one run to find them, not three."""
    assert len(_errors({"mode": "springy", "count": 0})) == 3  # missing mass, bad mode, bad count


def test_a_schema_that_is_both_required_and_defaulted_is_itself_an_error():
    """Refused where it is READ: a caller cannot act on 'required, default 3'."""
    bad = {"x": Field(float, required=True, default=3.0)}
    assert any("cannot both be true" in e for e in validate(bad, {"x": 1.0}))


# -- unknown keys -------------------------------------------------------------------------------


def test_validate_refuses_an_unknown_key_only_when_asked():
    assert _errors({"mass": 1.0, "wobble": 3}) == []
    strict = _errors({"mass": 1.0, "wobble": 3}, strict_keys=True)
    assert any("'wobble' is not a setting" in e for e in strict)


def test_the_keys_something_else_injected_are_never_unknown():
    """A manifest adds `prefix`, a spawn fills the entity: rejecting those would break adoption."""
    config = {"mass": 1.0, **{key: "x" for key in INJECTED_KEYS}}
    assert _errors(config, strict_keys=True) == []


def test_a_near_miss_is_suggested_and_a_distant_one_is_not():
    close = _errors({"mass": 1.0, "modes": "soft"}, strict_keys=True)
    assert any("did you mean 'mode'?" in e for e in close)
    far = _errors({"mass": 1.0, "banana": 1}, strict_keys=True)
    assert any("banana" in e and "did you mean" not in e for e in far)


# -- what is published ---------------------------------------------------------------------------


def test_the_published_form_carries_what_prose_cannot():
    fields = {f["name"]: f for f in describe(SCHEMA)}
    assert fields["mass"] == {
        "name": "mass",
        "type": "float",
        "required": True,
        "minimum": 0.0,
        "unit": "kg",
        "doc": "what it weighs",
    }
    # A key with a default publishes the default rather than leaving a caller to find it in code.
    assert fields["mode"]["default"] == "soft"
    assert fields["mode"]["choices"] == ["soft", "rigid"]
    assert "default" not in fields["mass"], "a required key has none, and must not imply one"


def test_declaration_order_is_kept():
    assert [f["name"] for f in describe(SCHEMA)] == list(SCHEMA)


# -- the plugin side ------------------------------------------------------------------------------


class _Declared(Plugin):
    CONFIG_SCHEMA = SCHEMA


class _Open(Plugin):
    CONFIG_SCHEMA = SCHEMA
    STRICT_KEYS = False
    OPEN_KEYS = "the rest is passed on to something that checks it"


class _Undeclared(Plugin):
    pass


def test_a_plugin_without_a_schema_is_unaffected_by_the_rule():
    assert _Undeclared({}).validate_schema({"anything": 1}) == []
    assert _Undeclared({}).config_errors({"anything": 1}) == []


def test_a_plugin_with_one_gets_the_checks_and_is_strict_without_asking():
    """Strict is the default: a schema says what the config is, so a key outside it is a typo."""
    assert Plugin.STRICT_KEYS is True
    assert _Declared({}).validate_schema({"mass": 2.0}) == []
    assert any("not a setting" in e for e in _Declared({}).validate_schema({"mass": 2.0, "x": 1}))


def test_a_plugin_that_opens_its_schema_passes_an_unknown_key():
    assert _Open({}).validate_schema({"mass": 2.0, "x": 1}) == []
    assert any("'mass' is required" in e for e in _Open({}).validate_schema({"x": 1}))


def test_the_catalog_publishes_a_declared_schema_and_says_when_it_is_strict():
    from roqsim.introspection import get_plugin_details

    payload = get_plugin_details("payload")
    assert {f["name"] for f in payload["schema"]} == {"mass", "body", "robot"}
    mass = next(f for f in payload["schema"] if f["name"] == "mass")
    assert mass["required"] is True and mass["unit"] == "kg"
    assert payload["strict_keys"] is True
    assert "open_keys" not in payload

    ceiling = get_plugin_details("ceiling")
    assert ceiling["strict_keys"] is True
    keep = next(f for f in ceiling["schema"] if f["name"] == "keep")
    assert keep["type"] == "bool" and keep["default"] is True


def test_an_open_schema_publishes_why_it_is_open(monkeypatch):
    """A caller told `strict_keys: false` should also be told what may pass, and why."""
    from roqsim.introspection import get_plugin_details
    from roqsim.plugins.payload import PayloadPlugin

    monkeypatch.setattr(PayloadPlugin, "STRICT_KEYS", False)
    monkeypatch.setattr(PayloadPlugin, "OPEN_KEYS", "a manifest adds keys this plugin passes on")
    payload = get_plugin_details("payload")
    assert payload["strict_keys"] is False
    assert payload["open_keys"] == "a manifest adds keys this plugin passes on"


def test_present_is_never_unknown_because_the_base_class_owns_it():
    """Read and checked for every plugin by `validate_presence`, which refuses it with the reason
    on a plugin that registers no entity -- so a schema must not refuse it a second time."""
    assert "present" in INJECTED_KEYS
    from roqsim.plugins.ceiling import CeilingPlugin

    errors = CeilingPlugin({}).config_errors({"present": False})
    assert len(errors) == 1 and "registers none" in errors[0], errors


def test_a_plugin_without_one_publishes_no_schema_key_at_all():
    """Absent rather than empty: a caller must be able to tell 'no declaration' from 'no keys'."""
    from roqsim.introspection import get_plugin_details

    assert "schema" not in get_plugin_details("dummy")


# -- the two adopters still behave --------------------------------------------------------------


def test_payload_still_refuses_what_only_it_knows_about():
    from roqsim.plugins.payload import PayloadPlugin

    errors = PayloadPlugin({"mass": 1.0, "offset": [0, 0, 1]}).config_errors(
        {"mass": 1.0, "offset": [0, 0, 1]}
    )
    assert any("'offset' is not supported" in e for e in errors)
    assert PayloadPlugin({"mass": 1.0}).config_errors({"mass": 1.0}) == []
    assert any("required" in e for e in PayloadPlugin({}).config_errors({}))


def test_ceiling_catches_a_misspelt_key_now():
    """The reason it can afford STRICT_KEYS: `above_Z` would otherwise leave the ceiling standing
    and look like the plugin not working."""
    from roqsim.plugins.ceiling import CeilingPlugin

    errors = CeilingPlugin({}).config_errors({"keep": False, "above_Z": 2.0})
    assert any("did you mean 'above_z'?" in e for e in errors)


def test_ceiling_keeps_the_rule_the_schema_has_no_word_for():
    from roqsim.plugins.ceiling import CeilingPlugin

    assert any("finite" in e for e in CeilingPlugin({}).config_errors({"above_z": float("inf")}))


def test_ceiling_still_explains_the_reserved_enabled_key():
    from roqsim.plugins.ceiling import CeilingPlugin

    errors = CeilingPlugin({}).config_errors({"enabled": False})
    assert any("reserved sibling" in e for e in errors)


# -- declaring a schema is what enforces it ---------------------------------------------------


class _Forgetful(Plugin):
    """A plugin that declares a schema and writes no validator of its own.

    The case the rule exists for: nothing here calls the checker, and a world is still held to the
    declaration. Before, this plugin published a contract through the catalog and checked none of
    it -- which is the docstring the schema replaces, wearing a type.
    """

    CONFIG_SCHEMA = {"mass": Field(float, required=True, minimum=0.0, unit="kg")}


def test_a_declared_schema_is_checked_without_the_plugin_asking():
    assert _Forgetful({}).config_errors({}) == ["'mass' is required"]
    assert _Forgetful({}).config_errors({"mass": -1.0}) == ["'mass' must be >= 0.0 kg, got -1.0"]
    assert _Forgetful({}).config_errors({"mass": 2.0}) == []


def test_the_schema_and_the_plugin_s_own_rules_are_both_reported():
    """One run finds both, in that order -- the schema's mechanical error first."""

    class _Both(Plugin):
        CONFIG_SCHEMA = {"above_z": Field(float, default=2.0)}

        def validate_config(self, config):
            return ["a rule only this plugin knows"]

    assert _Both({}).config_errors({"above_z": "high"}) == [
        "'above_z' must be float, got str ('high')",
        "a rule only this plugin knows",
    ]


def test_a_validator_that_raises_is_reported_rather_than_escaping():
    """The schema's errors survive it: a broken validator must not hide the ones already found."""

    class _Broken(Plugin):
        CONFIG_SCHEMA = {"mass": Field(float, required=True)}

        def validate_config(self, config):
            raise RuntimeError("boom")

    errors = _Broken({}).config_errors({})
    assert errors[0] == "'mass' is required"
    assert "validate_config raised: boom" in errors[1]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"above_Z": 2.0}, "did you mean 'above_z'"),
        # Mistyped, too: the plugin reads its settings rather than converting them in `__init__`,
        # so a wrong type reaches the checker that names it instead of raising out of a float().
        ({"above_z": "high"}, "'above_z' must be float"),
    ],
)
def test_the_whole_path_raises_for_a_world(config, expected):
    """Through instantiate_plugins, which is what a world actually meets."""
    from roqsim.config import PluginError, instantiate_plugins, load_config_from_dict

    cfg = load_config_from_dict({"sim": {}, "plugins": [{"ceiling": config, "name": "roof"}]})
    with pytest.raises(PluginError, match=expected):
        instantiate_plugins(cfg)


# -- settings: the config read through the schema ------------------------------------------------


def test_settings_fill_the_declared_default_for_a_key_left_out():
    settings = _Declared({"mass": 2.0}).settings
    assert settings.mass == 2.0
    assert settings.mode == "soft" and settings.count == 1 and settings.loud is False
    assert settings.pos is None, "no default declared: absent reads as None"


def test_settings_read_a_yaml_integer_as_the_float_the_schema_accepted():
    value = _Declared({"mass": 2}).settings.mass
    assert value == 2.0 and isinstance(value, float)
    assert isinstance(_Declared({"count": 3}).settings.count, int), "an int key stays an int"


def test_settings_refuse_a_name_the_schema_does_not_declare():
    with pytest.raises(AttributeError, match="declares no setting 'mas'. Declared: mass, mode"):
        _ = _Declared({}).settings.mas


def test_settings_are_read_only():
    settings = _Declared({"mass": 1.0}).settings
    with pytest.raises(AttributeError, match="read-only"):
        settings.mass = 3.0
    with pytest.raises(AttributeError, match="read-only"):
        del settings.mass


def test_a_mutable_default_is_a_fresh_copy_per_read():
    """A list default mutated by one reader must not become every other instance's default."""
    schema = {"rays": Field(list, default=[32, 24])}

    class _Rays(Plugin):
        CONFIG_SCHEMA = schema

    _Rays({}).settings.rays.append(99)
    assert _Rays({}).settings.rays == [32, 24]
    assert schema["rays"].default == [32, 24]


def test_a_value_of_the_wrong_type_reads_as_given_and_is_reported_by_the_check():
    """A view that fell back to the default would run the plugin on a value nobody stated."""
    plugin = _Declared({"mass": "heavy"})
    assert plugin.settings.mass == "heavy"
    assert plugin.config_errors(plugin.config) == ["'mass' must be float, got str ('heavy')"]


def test_settings_for_reads_the_config_it_is_given():
    """What a validator uses: the config it is asked about, not the instance's own."""
    plugin = _Declared({"mass": 1.0})
    assert plugin.settings_for({"mass": 5.0}).mass == 5.0
    assert plugin.settings.mass == 1.0


def test_a_plugin_without_a_schema_has_no_settings():
    with pytest.raises(AttributeError, match="declares no CONFIG_SCHEMA"):
        _ = _Undeclared({}).settings


def test_a_schema_plugin_publishes_its_parameters_from_the_declaration():
    """The list a caller reads and the list validation runs on are one list."""
    from roqsim.introspection import get_plugin_details
    from roqsim.plugins.energy_monitor import EnergyMonitorPlugin

    details = get_plugin_details("energy_monitor")
    assert [p["name"] for p in details["parameters"]] == list(EnergyMonitorPlugin.CONFIG_SCHEMA)
    idle = next(p for p in details["parameters"] if p["name"] == "idle_w")
    assert idle["example"] == "0.0" and idle["doc"].startswith("W; ")
    mass = next(p for p in get_plugin_details("payload")["parameters"] if p["name"] == "mass")
    assert mass["example"] is None and mass["doc"].startswith("required, kg; ")


def test_the_docs_page_renders_a_schema_as_a_config_block():
    from roqsim.introspection import _parse_config_block, schema_config_block
    from roqsim.plugins.ceiling import CeilingPlugin

    block = schema_config_block("ceiling", CeilingPlugin)
    assert block[0] == "Config (declared in ``CONFIG_SCHEMA`` -- unknown keys are refused)::"
    parsed = _parse_config_block("\n".join(block))
    assert [f["name"] for f in parsed] == ["keep", "above_z"]
    assert parsed[1]["example"] == "2.5"
