# SPDX-License-Identifier: Apache-2.0
"""``with_transport``: the seam that lets a checked-in world stay ROS-free.

A world that declares ``ros2_bridge`` cannot be run by ``roqsim sim`` in a pip-only
environment -- the bridge is registered by a colcon package that is not there -- so
keeping it out of the file is what makes the world standalone-runnable. Transport is
therefore added at load time, by whoever deploys the run.
"""

import pytest

from roqsim.config import (drop_transport_plugins, entry_ref, load_config,
                        with_transport)


def _refs(raw):
    return [entry_ref(entry) for entry in raw["plugins"]]


def _world(tmp_path, body):
    path = tmp_path / "w.yaml"
    path.write_text(body)
    return path


def test_it_appends_the_bridge():
    raw = {"plugins": [{"spawn_robot": {"model": "husky_a200"}}]}
    assert _refs(with_transport(raw)) == ["spawn_robot", "ros2_bridge"]


def test_control_is_opt_in():
    """The simulation_interfaces control plane is a handful of extra services that only
    a scenario touching entities needs."""
    raw = {"plugins": []}
    assert _refs(with_transport(raw)) == ["ros2_bridge"]
    assert _refs(with_transport(raw, control=True)) == ["ros2_bridge", "sim_interfaces"]


def test_ros_false_is_a_no_op():
    raw = {"plugins": [{"floorplan": {}}]}
    assert with_transport(raw, ros=False) is raw


def test_it_is_idempotent():
    """A caller should not have to know whether the author already put one there."""
    once = with_transport({"plugins": [{"floorplan": {}}]})
    assert with_transport(once) == once


def test_a_world_that_already_has_a_transport_is_left_alone():
    raw = {"plugins": [{"ros2_bridge": {"tf_namespace": "robot"}}]}
    assert with_transport(raw, tf_namespace="other") is raw


def test_it_does_not_mutate_the_input():
    """The caller's parsed world is reused elsewhere (world_sources, the exporters)."""
    raw = {"plugins": [{"floorplan": {}}]}
    with_transport(raw)
    assert _refs(raw) == ["floorplan"]


def test_tf_namespace_is_topic_only():
    """Nav2 launches with the standard /tf->tf remap; without this both it and
    scenario-execution's listener look under /<ns>/tf while the bridge publishes
    globally, and init_nav2 hangs waiting for a transform."""
    out = with_transport({"plugins": []}, tf_namespace="robot")
    assert out["plugins"][0]["ros2_bridge"] == {"tf_namespace": "robot"}


def test_it_is_the_inverse_of_dropping(tmp_path):
    """What ``roqsim render`` and the exporters strip is exactly what this adds."""
    world = _world(tmp_path, "sim: {}\nplugins:\n  - dummy: {}\n")
    cfg = load_config(world, transport={"ros": True})
    assert [spec.ref for spec in cfg.plugins] == ["dummy", "ros2_bridge"]

    transport, unavailable = drop_transport_plugins(cfg)
    # ros2_bridge is registered by a colcon package, so in a pip-only test environment it
    # is 'unavailable' rather than 'transport' -- either way it is dropped, and the world
    # left behind is the one that was checked in.
    assert transport or unavailable
    assert [spec.ref for spec in cfg.plugins] == ["dummy"]


def test_a_checked_in_world_stays_ros_free(tmp_path):
    """The property the whole seam exists for: loading without transport leaves the file
    exactly as authored, so it runs where the bridge is not installed."""
    world = _world(tmp_path, "sim: {}\nplugins:\n  - dummy: {}\n")
    assert [spec.ref for spec in load_config(world).plugins] == ["dummy"]


def test_transport_is_applied_after_overrides(tmp_path):
    """Overrides address plugins by name and cannot append one, so ordering matters:
    an override naming ros2_bridge before it exists would be an error."""
    world = _world(tmp_path, "sim: {}\nplugins:\n  - dummy: {}\n")
    cfg = load_config(world, {"sim": {"pacing": "asap"}}, {"ros": True})
    assert cfg.pacing == "asap"
    assert [spec.ref for spec in cfg.plugins] == ["dummy", "ros2_bridge"]


def test_overriding_a_plugin_that_does_not_exist_still_fails(tmp_path):
    """The injection must not turn a typo into a silent no-op."""
    from roqsim.plugin import PluginError
    world = _world(tmp_path, "sim: {}\nplugins:\n  - dummy: {}\n")
    with pytest.raises(PluginError, match="matches no plugin"):
        load_config(world, {"plugins": {"nosuch": {}}})
