"""Config: per-plugin validate_config aggregation and fail-fast behaviour."""

from __future__ import annotations

import pytest

from roqsim.config import instantiate_plugins, load_config_from_dict
from roqsim.plugin import Plugin, PluginError


class Strict(Plugin):
    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if "required_key" not in config:
            errors.append("missing 'required_key'")
        if config.get("n", 1) < 0:
            errors.append("'n' must be >= 0")
        return errors


REF = f"{__name__}:Strict"


def test_valid_config_instantiates():
    cfg = load_config_from_dict({"plugins": [{REF: {"required_key": 1, "n": 2}}]})
    plugins = instantiate_plugins(cfg)
    assert len(plugins) == 1


def test_aggregates_all_errors():
    cfg = load_config_from_dict({"plugins": [{REF: {"n": -1}}]})
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    msg = str(exc.value)
    # both errors from the single plugin are reported together
    assert "missing 'required_key'" in msg
    assert "'n' must be >= 0" in msg


def test_errors_are_namespaced_by_plugin():
    cfg = load_config_from_dict(
        {
            "plugins": [
                {REF: {}, "name": "first"},
                {REF: {"required_key": 1}, "name": "second"},
            ]
        }
    )
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    msg = str(exc.value)
    assert "[first" in msg
    assert "[second" not in msg  # second is valid


def test_missing_plugin_key_errors():
    # An entry with only the reserved 'name' key (no plugin ref) is rejected.
    with pytest.raises(PluginError, match="exactly one plugin-ref key"):
        load_config_from_dict({"plugins": [{"name": "x"}]})


def test_multiple_plugin_keys_error():
    # An entry may carry at most one plugin ref (plus the optional 'name').
    with pytest.raises(PluginError, match="exactly one plugin-ref key"):
        load_config_from_dict({"plugins": [{REF: {}, "other_ref": {}}]})


def test_a_reserved_sibling_written_inside_the_config_is_refused():
    """The near miss the shape check alone lets through.

    A `name` inside the config reaches no plugin, so without this the document loads with every
    entry keeping the plugin's ref as its label -- and the mistake surfaces far from its cause, as a
    duplicate label, as an override that matches nothing, or not at all.
    """
    with pytest.raises(PluginError) as exc:
        load_config_from_dict({"plugins": [{REF: {"required_key": 1, "name": "obs_3"}}]})
    msg = str(exc.value)
    assert "'name'" in msg and "reserved sibling key" in msg
    assert "components[0]" in msg  # which entry
    assert "obs_3" in msg  # ... spelled the way it should have been written


def test_the_refusal_names_every_misplaced_reserved_key():
    with pytest.raises(PluginError) as exc:
        load_config_from_dict({"plugins": [{REF: {"name": "x", "components": []}}]})
    msg = str(exc.value)
    assert "'components', 'name'" in msg and "are reserved sibling keys" in msg


def test_enabled_inside_a_config_is_left_to_the_plugin():
    """The one reserved sibling a plugin may intercept itself, because for a subtractive plugin
    (``ceiling``) the sibling means the opposite of what writing it into the config intends -- so a
    generic "move it out one level" would be wrong advice. Here it reaches validate_config."""
    cfg = load_config_from_dict({"plugins": [{REF: {"required_key": 1, "enabled": False}}]})
    assert cfg.plugins[0].config["enabled"] is False


def test_a_reserved_sibling_inside_a_nested_entrys_config_is_refused():
    """Children are entries too, and the address in the message says which one."""
    with pytest.raises(PluginError) as exc:
        load_config_from_dict(
            {
                "plugins": [
                    {REF: {"required_key": 1}, "components": [{REF: {"name": "inner"}}]},
                ]
            }
        )
    assert "components[0].components[0]" in str(exc.value)


def test_the_sibling_spelling_is_what_labels_the_entry():
    cfg = load_config_from_dict(
        {"plugins": [{REF: {"required_key": 1}, "name": "obs_3"}, {REF: {"required_key": 1}}]}
    )
    # An entry that omits it answers to its ref -- here the class part of a `module:Class` ref.
    assert [spec.label for spec in cfg.plugins] == ["obs_3", "Strict"]


def test_view_accepts_the_camera_keys():
    view = {
        "lookat": [0, 0, 1],
        "distance": 3.0,
        "azimuth": 90,
        "elevation": -20,
        "track": "robot",
        "follow_heading": True,
    }
    assert load_config_from_dict({"sim": {"view": view}, "plugins": []}).view == view


def test_view_rejects_unknown_keys():
    # sim.view is the camera only; a typo (or a run-level switch that doesn't belong in a world)
    # must fail rather than be silently dropped.
    with pytest.raises(PluginError, match="unknown key\\(s\\) right_ui"):
        load_config_from_dict({"sim": {"view": {"right_ui": True}}, "plugins": []})
    with pytest.raises(PluginError, match="unknown key\\(s\\) azimut"):
        load_config_from_dict({"sim": {"view": {"azimut": 90}}, "plugins": []})


def test_view_rejects_malformed_values():
    # A string where three numbers belong used to travel all the way to apply_view, which iterates
    # the characters of `"1,2,0"` and fails with `could not convert string to float: '.'` -- naming
    # a decimal point, no key and no file. Reject it here, where the key can be named.
    with pytest.raises(PluginError, match="sim.view.lookat: expected 3 numbers"):
        load_config_from_dict({"sim": {"view": {"lookat": "1,2,0"}}, "plugins": []})
    with pytest.raises(PluginError, match="sim.view.lookat: expected 3 numbers"):
        load_config_from_dict({"sim": {"view": {"lookat": [1, 2]}}, "plugins": []})
    with pytest.raises(PluginError, match="sim.view.distance: expected a number"):
        load_config_from_dict({"sim": {"view": {"distance": "near"}}, "plugins": []})
    with pytest.raises(PluginError, match="sim.view.track: expected an entity or body name"):
        load_config_from_dict({"sim": {"view": {"track": 3}}, "plugins": []})
