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
