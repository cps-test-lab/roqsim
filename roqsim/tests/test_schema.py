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


def test_an_unknown_key_passes_unless_the_plugin_says_its_list_is_complete():
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
    STRICT_KEYS = True


class _Undeclared(Plugin):
    pass


def test_a_plugin_without_a_schema_is_unaffected_by_the_rule():
    assert _Undeclared({}).validate_schema({"anything": 1}) == []
    assert _Undeclared({}).config_errors({"anything": 1}) == []


def test_a_plugin_with_one_gets_the_checks_and_its_strictness():
    assert _Declared({}).validate_schema({"mass": 2.0}) == []
    assert any("not a setting" in e for e in _Declared({}).validate_schema({"mass": 2.0, "x": 1}))


def test_the_catalog_publishes_a_declared_schema_and_says_when_it_is_strict():
    from roqsim.introspection import get_plugin_details

    payload = get_plugin_details("payload")
    assert {f["name"] for f in payload["schema"]} == {"mass", "body", "robot"}
    mass = next(f for f in payload["schema"] if f["name"] == "mass")
    assert mass["required"] is True and mass["unit"] == "kg"
    assert payload["strict_keys"] is False

    ceiling = get_plugin_details("ceiling")
    assert ceiling["strict_keys"] is True
    keep = next(f for f in ceiling["schema"] if f["name"] == "keep")
    assert keep["type"] == "bool" and keep["default"] is True


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


def test_the_whole_path_raises_for_a_world():
    """Through instantiate_plugins, which is what a world actually meets.

    A misspelt key rather than a mistyped one, because a plugin reads its config in ``__init__``
    and instances are built before anything is validated -- so ``above_z: high`` raises out of
    ``float()`` first, and never reaches the checker that would have named it.
    """
    from roqsim.config import PluginError, instantiate_plugins, load_config_from_dict

    cfg = load_config_from_dict(
        {"sim": {}, "plugins": [{"ceiling": {"above_Z": 2.0}, "name": "roof"}]}
    )
    with pytest.raises(PluginError, match="did you mean 'above_z'"):
        instantiate_plugins(cfg)
