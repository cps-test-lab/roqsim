"""Config: per-plugin validate_config aggregation and fail-fast behaviour."""

from __future__ import annotations

import pytest

from roqsim.config import SIM_KEYS, SIM_OPTION_KEYS, instantiate_plugins, load_config_from_dict
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
    cfg = load_config_from_dict({"components": [{REF: {"required_key": 1, "n": 2}}]})
    plugins = instantiate_plugins(cfg)
    assert len(plugins) == 1


def test_aggregates_all_errors():
    cfg = load_config_from_dict({"components": [{REF: {"n": -1}}]})
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    msg = str(exc.value)
    # both errors from the single plugin are reported together
    assert "missing 'required_key'" in msg
    assert "'n' must be >= 0" in msg


def test_errors_are_namespaced_by_plugin():
    cfg = load_config_from_dict(
        {
            "components": [
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


class ConvertsInInit(Plugin):
    """Reads a setting in its constructor, before validate_config has looked at it."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.speed = float(self.config.get("speed", 1.0))

    def validate_config(self, config: dict) -> list[str]:
        speed = config.get("speed", 1.0)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            return [f"'speed' must be a number, got {speed!r}"]
        return []


class NeedsItsConstructor(ConvertsInInit):
    """A validator that reads what only the constructor sets."""

    def validate_config(self, config: dict) -> list[str]:
        return [] if self.speed > 0 else ["'speed' must be > 0"]


CONVERTS = f"{__name__}:ConvertsInInit"


def test_a_constructor_that_raises_on_its_config_is_reported_with_the_validators_message():
    cfg = load_config_from_dict(
        {"components": [{CONVERTS: {"speed": "fast"}, "name": "mover"}, {REF: {}, "name": "other"}]}
    )
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    msg = str(exc.value)
    assert f"[mover ({CONVERTS})] 'speed' must be a number, got 'fast'" in msg
    assert f"[mover ({CONVERTS})] could not be constructed: ValueError" in msg
    # Every other plugin's errors are still in the same report.
    assert "[other" in msg and "missing 'required_key'" in msg
    # The constructor's own traceback is kept.
    assert isinstance(exc.value.__cause__, ValueError)


def test_a_validator_that_needs_the_constructor_reports_only_the_constructor():
    ref = f"{__name__}:NeedsItsConstructor"
    cfg = load_config_from_dict({"components": [{ref: {"speed": [1]}, "name": "mover"}]})
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    msg = str(exc.value)
    assert f"[mover ({ref})] could not be constructed: TypeError" in msg
    # Its validator raised on the half-built instance; that says nothing of the config.
    assert "validate_config raised" not in msg


def test_missing_plugin_key_errors():
    # An entry with only the reserved 'name' key (no plugin ref) is rejected.
    with pytest.raises(PluginError, match="exactly one plugin-ref key"):
        load_config_from_dict({"components": [{"name": "x"}]})


def test_multiple_plugin_keys_error():
    # An entry may carry at most one plugin ref (plus the optional 'name').
    with pytest.raises(PluginError, match="exactly one plugin-ref key"):
        load_config_from_dict({"components": [{REF: {}, "other_ref": {}}]})


def test_a_reserved_sibling_written_inside_the_config_is_refused():
    """The near miss the shape check alone lets through.

    A `name` inside the config reaches no plugin, so without this the document loads with every
    entry keeping the plugin's ref as its label -- and the mistake surfaces far from its cause, as a
    duplicate label, as an override that matches nothing, or not at all.
    """
    with pytest.raises(PluginError) as exc:
        load_config_from_dict({"components": [{REF: {"required_key": 1, "name": "obs_3"}}]})
    msg = str(exc.value)
    assert "'name'" in msg and "reserved sibling key" in msg
    assert "components[0]" in msg  # which entry
    assert "obs_3" in msg  # ... spelled the way it should have been written


def test_the_refusal_names_every_misplaced_reserved_key():
    with pytest.raises(PluginError) as exc:
        load_config_from_dict({"components": [{REF: {"name": "x", "components": []}}]})
    msg = str(exc.value)
    assert "'components', 'name'" in msg and "are reserved sibling keys" in msg


def test_enabled_inside_a_config_is_left_to_the_plugin():
    """The one reserved sibling a plugin may intercept itself, because for a subtractive plugin
    (``ceiling``) the sibling means the opposite of what writing it into the config intends -- so a
    generic "move it out one level" would be wrong advice. Here it reaches validate_config."""
    cfg = load_config_from_dict({"components": [{REF: {"required_key": 1, "enabled": False}}]})
    assert cfg.plugins[0].config["enabled"] is False


def test_a_reserved_sibling_inside_a_nested_entrys_config_is_refused():
    """Children are entries too, and the address in the message says which one."""
    with pytest.raises(PluginError) as exc:
        load_config_from_dict(
            {
                "components": [
                    {REF: {"required_key": 1}, "components": [{REF: {"name": "inner"}}]},
                ]
            }
        )
    assert "components[0].components[0]" in str(exc.value)


def test_the_sibling_spelling_is_what_labels_the_entry():
    cfg = load_config_from_dict(
        {"components": [{REF: {"required_key": 1}, "name": "obs_3"}, {REF: {"required_key": 1}}]}
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
    assert load_config_from_dict({"sim": {"view": view}, "components": []}).view == view


#: A value each ``sim`` key accepts.
_SIM_SAMPLES = {
    "cone": "elliptic",
    "contact_override": {"solref": [0.02, 1]},
    "dedup_assets": False,
    "density": 1.2,
    "gravity": [0, 0, 0],
    "impratio": 10,
    "integrator": "rk4",
    "iterations": 50,
    "ls_iterations": 20,
    "name": "Lab",
    "noslip_iterations": 3,
    "pacing": "asap",
    "renderer": "mujoco",
    "renderers": {"mujoco": {}},
    "seed": 7,
    "solver": "newton",
    "sync": {"enabled": False},
    "timestep": 0.002,
    "view": {"azimuth": 90},
    "viscosity": 1.8e-5,
    "wind": [1, 0, 0],
    "world": "world.xml",
}


def test_sim_samples_cover_the_keys():
    assert set(_SIM_SAMPLES) == SIM_KEYS


def test_each_option_key_is_a_sim_key_and_a_mujoco_option_field():
    """The engine sets these on ``spec.option`` by their own names, and the loader admits them."""
    import mujoco

    option = mujoco.MjSpec().option
    assert set(SIM_OPTION_KEYS) <= SIM_KEYS
    assert [k for k in SIM_OPTION_KEYS if not hasattr(option, k)] == []


@pytest.mark.parametrize("key", sorted(SIM_KEYS))
def test_sim_accepts_each_key(key):
    cfg = load_config_from_dict({"sim": {key: _SIM_SAMPLES[key]}, "components": []})
    assert cfg.sim == {key: _SIM_SAMPLES[key]}


@pytest.mark.parametrize("key", ["foo", "headless"])
def test_sim_refuses_unknown_keys(key):
    with pytest.raises(PluginError, match=f"sim: unknown key\\(s\\) '{key}'"):
        load_config_from_dict({"sim": {key: 1, "pacing": "asap"}, "components": []})


def test_sim_refuses_unknown_key_from_an_override():
    with pytest.raises(PluginError, match="sim: unknown key\\(s\\) 'foo'"):
        load_config_from_dict({"components": []}, overrides={"sim": {"foo": 1}})


def test_sim_unknown_key_names_the_nearest():
    with pytest.raises(PluginError, match="'timestp' \\(did you mean 'timestep'\\?\\)"):
        load_config_from_dict({"sim": {"timestp": 0.002}, "components": []})


def test_view_rejects_unknown_keys():
    # sim.view is the camera only; a typo (or a run-level switch that doesn't belong in a world)
    # must fail rather than be silently dropped.
    with pytest.raises(PluginError, match="unknown key\\(s\\) right_ui"):
        load_config_from_dict({"sim": {"view": {"right_ui": True}}, "components": []})
    with pytest.raises(PluginError, match="unknown key\\(s\\) azimut"):
        load_config_from_dict({"sim": {"view": {"azimut": 90}}, "components": []})


def test_view_rejects_malformed_values():
    # Unchecked, a string where three numbers belong travels all the way to apply_view, which iterates
    # the characters of `"1,2,0"` and fails with `could not convert string to float: '.'` -- naming
    # a decimal point, no key and no file. Reject it here, where the key can be named.
    with pytest.raises(PluginError, match="sim.view.lookat: expected 3 numbers"):
        load_config_from_dict({"sim": {"view": {"lookat": "1,2,0"}}, "components": []})
    with pytest.raises(PluginError, match="sim.view.lookat: expected 3 numbers"):
        load_config_from_dict({"sim": {"view": {"lookat": [1, 2]}}, "components": []})
    with pytest.raises(PluginError, match="sim.view.distance: expected a number"):
        load_config_from_dict({"sim": {"view": {"distance": "near"}}, "components": []})
    with pytest.raises(PluginError, match="sim.view.track: expected an entity or body name"):
        load_config_from_dict({"sim": {"view": {"track": 3}}, "components": []})


@pytest.mark.parametrize(
    ("sim", "message"),
    [
        ({"cone": "eliptic"}, "sim.cone: unknown value 'eliptic'"),
        ({"gravity": [0, 0]}, "sim.gravity: expected 3 numbers"),
        ({"gravity": "down"}, "sim.gravity: expected 3 numbers"),
        ({"wind": 3.0}, "sim.wind: expected 3 numbers"),
        ({"iterations": 2.5}, "sim.iterations: expected a whole number"),
        ({"noslip_iterations": -1}, "sim.noslip_iterations: expected a whole number"),
        ({"density": "air"}, "sim.density: expected a number >= 0"),
        ({"timestep": 0}, "sim.timestep: expected a number of seconds > 0"),
        ({"dedup_assets": "no"}, "sim.dedup_assets: expected true or false"),
        ({"pacing": "fast"}, "sim.pacing: invalid pacing 'fast'"),
        ({"pacing": {"factor": 0}}, "sim.pacing: invalid pacing"),
        ({"pacing": {"factr": 2}}, "sim.pacing: invalid pacing"),
    ],
)
def test_a_malformed_sim_value_is_refused_at_load(sim, message):
    """Each would otherwise load, and fail only once every plugin had built, as a bare error from
    a MuJoCo setter or an enum lookup that names neither the key nor the file -- or, for pacing,
    once the standalone driver divides by the factor."""
    with pytest.raises(PluginError, match=message.replace("(", "\\(").replace("[", "\\[")):
        load_config_from_dict({"sim": sim, "components": []})


def test_the_engine_maps_exactly_the_names_the_loader_admits():
    """Two tables name each enum -- the loader's, which refuses, and the engine's, which maps to
    MuJoCo -- so a name in one and not the other is a world that loads and then fails to build."""
    from roqsim import engine
    from roqsim.config import SIM_CONES, SIM_INTEGRATORS, SIM_SOLVERS

    assert set(engine._CONES) == set(SIM_CONES)
    assert set(engine._SOLVERS) == set(SIM_SOLVERS)
    assert set(engine._INTEGRATORS) == set(SIM_INTEGRATORS) - {"auto"}


@pytest.mark.parametrize("factor", [0, -1.0, True, "4"])
def test_the_pacer_refuses_a_factor_that_is_not_a_positive_number(factor):
    """``--pacing 0`` reaches the pacer as ``{factor: 0.0}``, whose wait divides by it."""
    from roqsim.clock import Pacer

    with pytest.raises(ValueError, match="positive number"):
        Pacer.from_config({"factor": factor}, dt=0.002)


def test_a_number_yaml_read_as_text_is_named_as_such():
    import yaml

    sim = yaml.safe_load("viscosity: 1e-5")
    assert sim == {"viscosity": "1e-5"}
    with pytest.raises(PluginError, match="the string '1e-5' \\(YAML reads an exponent"):
        load_config_from_dict({"sim": sim, "components": []})
