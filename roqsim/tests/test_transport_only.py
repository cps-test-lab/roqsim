"""A scene-only consumer drops the plugins that build no scene -- `Plugin.transport_only` and refs
this environment cannot resolve -- which is what makes a `*_ros` world renderable without ROS.

The property is declared by the plugin, never listed here: a name list in core would serve only the
bridges we happen to ship (see roqsim/CLAUDE.md, "A capability is declared by the plugin").
"""

from __future__ import annotations

import pytest

from roqsim.bridge import BridgeBase
from roqsim.config import (
    drop_transport,
    drop_transport_plugins,
    instantiate_plugins,
    load_config_from_dict,
)
from roqsim.plugin import Plugin, PluginError


class Geometry(Plugin):
    """Stand-in for anything that contributes to the model."""


class Transport(Plugin):
    transport_only = True


GEOMETRY = f"{__name__}:Geometry"
TRANSPORT = f"{__name__}:Transport"


def _refs(cfg):
    return [spec.ref for spec in cfg.plugins]


def test_a_bridge_declares_itself_transport_only():
    """On the base class, so a third-party transport is covered without touching roqsim."""
    assert BridgeBase.transport_only is True
    assert Plugin.transport_only is False  # the default: a plugin is assumed to build something


def test_transport_plugin_is_dropped_and_reported():
    cfg = load_config_from_dict({"plugins": [{GEOMETRY: {}}, {TRANSPORT: {}}]})
    transport, unavailable = drop_transport_plugins(cfg)
    assert _refs(cfg) == [GEOMETRY]
    assert transport == [TRANSPORT]
    assert unavailable == []


def test_unresolvable_ref_is_dropped_separately():
    """The reported case: `ros2_bridge` lives in a colcon package, absent from a pip-only venv."""
    cfg = load_config_from_dict({"plugins": [{"ros2_bridge": {}}, {GEOMETRY: {}}]})
    transport, unavailable = drop_transport_plugins(cfg)
    assert _refs(cfg) == [GEOMETRY]
    assert transport == []
    assert unavailable == ["ros2_bridge"]


def test_a_named_instance_is_reported_by_name_and_ref():
    """So a world with several bridges says which one went."""
    cfg = load_config_from_dict({"plugins": [{TRANSPORT: {}, "name": "gt_bridge"}]})
    transport, _ = drop_transport_plugins(cfg)
    assert transport == [f"gt_bridge ({TRANSPORT})"]


def test_a_scene_of_only_geometry_is_untouched():
    cfg = load_config_from_dict({"plugins": [{GEOMETRY: {}}, {"dummy": {}}]})
    assert drop_transport_plugins(cfg) == ([], [])
    assert _refs(cfg) == [GEOMETRY, "dummy"]


# -- what the consumers do with it ---------------------------------------------------------------


def _world(tmp_path, body: str):
    world = tmp_path / "w.yaml"
    world.write_text(body)
    return str(world)


def test_render_builds_a_world_whose_bridge_is_not_installed(tmp_path):
    from roqsim import render

    target = _world(tmp_path, "plugins:\n  - ros2_bridge: {}\n  - sim_interfaces: {}\n")
    model, _data, _ctx, _view, _cam = render.build_target(target, None)
    assert model is not None  # the empty_room world definition, built without ROS


def test_render_can_still_demand_the_simulators_strict_build(tmp_path):
    """`roqsim sim` uses that build: a missing bridge there must stay a loud failure."""
    from roqsim import render

    target = _world(tmp_path, "plugins:\n  - ros2_bridge: {}\n")
    with pytest.raises(PluginError, match="unknown plugin 'ros2_bridge'"):
        render.build_target(target, None, skip_transport=False)


def test_render_warns_about_what_it_could_not_load(tmp_path, caplog):
    from roqsim import render

    target = _world(tmp_path, "plugins:\n  - ros2_bridge: {}\n")
    with caplog.at_level("WARNING"):
        render.build_target(target, None)
    assert "ros2_bridge" in caplog.text  # a typo'd ref surfaces the same way


def test_a_registered_but_unimportable_plugin_is_dropped_too(monkeypatch):
    """The case that actually bit: the entry point *exists*, and importing it is what fails.

    ``test_unresolvable_ref_is_dropped_separately`` covers a pip-only venv, where ``ros2_bridge`` is not
    registered at all and resolution stops early with a PluginError. A campaign image is the opposite:
    the ``roqsim_ros_bridge`` wheel *is* installed, so the entry point resolves and ``ep.load()`` runs --
    then dies on ``import rclpy`` because a geometry-only export never sourced ROS. That ImportError is
    not a PluginError, so it sailed straight through ``drop_transport_plugins`` and killed
    ``roqsim export web`` inside the aux container, on the one plugin the function was trying to discard.
    """
    from roqsim import registry

    class _EP:
        name = "explodes"

        def load(self):
            raise ModuleNotFoundError("No module named 'rclpy'")

    monkeypatch.setattr(registry, "_entry_points", lambda group: (_EP(),))
    cfg = load_config_from_dict({"plugins": [{"explodes": {}}, {GEOMETRY: {}}]})
    transport, unavailable = drop_transport_plugins(cfg)
    assert _refs(cfg) == [GEOMETRY], "the unimportable plugin was not dropped"
    assert transport == []
    assert unavailable == ["explodes"]


# -- `roqsim sim --no-communication`: the same subtraction, held to a simulation's standard ----------


def test_no_communication_drops_a_declared_transport():
    cfg = load_config_from_dict({"plugins": [{GEOMETRY: {}}, {TRANSPORT: {}}]})
    assert drop_transport(cfg) == [TRANSPORT]
    assert _refs(cfg) == [GEOMETRY]


def test_no_communication_drops_a_bridge_this_environment_cannot_resolve():
    """The pip-only case: the class that would declare `transport_only` is what is missing."""
    cfg = load_config_from_dict(
        {"plugins": [{"ros2_bridge": {}}, {"sim_interfaces": {}}, {GEOMETRY: {}}]}
    )
    assert drop_transport(cfg) == ["ros2_bridge", "sim_interfaces"]
    assert _refs(cfg) == [GEOMETRY]


def test_no_communication_keeps_an_unresolvable_ref_that_is_not_a_bridge():
    """The difference from the render path, and the reason this is a separate function.

    Dropping every ref that fails to resolve would swallow a misspelt geometry plugin -- harmless for
    a picture, a silently different experiment for a run.
    """
    cfg = load_config_from_dict({"plugins": [{"groud_truth_pose": {}}, {"ros2_bridge": {}}]})
    assert drop_transport(cfg) == ["ros2_bridge"]
    assert _refs(cfg) == ["groud_truth_pose"], "a typo must survive to fail the build"


def test_no_communication_reports_a_named_instance_by_name_and_ref():
    cfg = load_config_from_dict({"plugins": [{TRANSPORT: {}, "name": "gt_bridge"}]})
    assert drop_transport(cfg) == [f"gt_bridge ({TRANSPORT})"]


def test_a_missing_dependency_is_still_loud_for_a_simulation(monkeypatch):
    """`roqsim sim` needs its transport, so resolution must keep raising -- with the reason, not a bare
    ImportError traceback from somewhere inside the plugin."""
    from roqsim import registry

    class _EP:
        name = "explodes"

        def load(self):
            raise ModuleNotFoundError("No module named 'rclpy'")

    monkeypatch.setattr(registry, "_entry_points", lambda group: (_EP(),))
    with pytest.raises(PluginError, match="registered by an installed package"):
        registry.resolve_plugin("explodes")


# -- what the failure says when transport is the ONLY thing missing -------------------------------


@pytest.fixture
def no_entry_points(monkeypatch):
    """No short name resolves, whatever this machine has installed.

    The bridge's resolvability is exactly what differs between a pip-only venv and a ROS-sourced one,
    so a test that let the environment decide it would assert the opposite thing depending on who ran
    it. ``module:Class`` refs are unaffected -- they never consult the entry points.
    """
    from roqsim import registry

    monkeypatch.setattr(registry, "_entry_points", lambda group: ())


def _build(raw):
    from roqsim.config import instantiate_plugins

    instantiate_plugins(load_config_from_dict(raw))


def test_a_world_failing_only_on_its_bridges_names_both_ways_out(no_entry_points):
    """The reported experience: `unknown plugin 'ros2_bridge'` sent the reader hunting for a typo."""
    with pytest.raises(PluginError) as err:
        _build({"plugins": [{GEOMETRY: {}}, {"ros2_bridge": {}}, {"sim_interfaces": {}}]})
    message = str(err.value)
    assert "ros2_bridge, sim_interfaces: transport, not scene" in message
    assert "ros2_ws/install/setup.bash" in message, "the way to run it as authored"
    assert "--no-communication" in message, "the way to run it without"


def test_a_typo_alongside_a_bridge_gets_the_plain_report(no_entry_points):
    """The hint would be actively misleading here: --no-communication cannot fix a misspelt plugin."""
    with pytest.raises(PluginError) as err:
        _build({"plugins": [{"ros2_bridge": {}}, {"groud_truth_pose": {}}]})
    message = str(err.value)
    assert "groud_truth_pose" in message and "ros2_bridge" in message, "both are reported"
    assert "--no-communication" not in message


def test_every_unresolvable_ref_is_reported_at_once():
    """Resolution used to stop at the first, costing an edit-and-rerun cycle per typo.

    No fixture needed: these two names are not registered in any environment.
    """
    with pytest.raises(PluginError) as err:
        _build({"plugins": [{"nope_one": {}}, {"nope_two": {}}]})
    assert "nope_one" in str(err.value) and "nope_two" in str(err.value)


def test_a_single_failure_keeps_the_original_wording():
    """A plain typo is the common case; it should not gain a list to read."""
    with pytest.raises(PluginError, match="^unknown plugin 'groud_truth_pose'"):
        _build({"plugins": [{"groud_truth_pose": {}}]})


# -- expansion runs while the document loads, and must not make loading strict ------------------


def test_a_document_whose_bridge_is_not_installed_still_LOADS(no_entry_points):
    """Expansion moved into the load path, so a ref that will not import is met earlier than it
    used to be. Loading has to stay tolerant: a consumer that wants the *scene* -- render, the
    exporters, describe -- must still get one for a world it cannot run."""
    cfg = load_config_from_dict({"components": [{GEOMETRY: {}}, {"ros2_bridge": {}}]})
    assert _refs(cfg) == [GEOMETRY, "ros2_bridge"]
    assert [ref for ref, _msg in cfg.unresolved] == ["ros2_bridge"]


def test_but_running_it_is_still_refused_with_the_same_report(no_entry_points):
    """The refusal moves file position, not observable behaviour -- including the case that says
    this is a ROS world rather than a typo, and names the two ways on."""
    cfg = load_config_from_dict({"components": [{GEOMETRY: {}}, {"ros2_bridge": {}}]})
    with pytest.raises(PluginError) as exc:
        instantiate_plugins(cfg)
    msg = str(exc.value)
    assert "transport, not scene" in msg
    assert "--no-communication" in msg


def test_dropping_the_transport_drops_its_deferred_failure_with_it(no_entry_points):
    """Otherwise a scene-only consumer removes the bridge it cannot import and is then refused
    for it anyway -- the exact situation dropping it exists to avoid."""
    cfg = load_config_from_dict({"components": [{GEOMETRY: {}}, {"ros2_bridge": {}}]})
    drop_transport_plugins(cfg)
    assert cfg.unresolved == []
    assert [type(p).__name__ for p in instantiate_plugins(cfg)] == ["Geometry"]
