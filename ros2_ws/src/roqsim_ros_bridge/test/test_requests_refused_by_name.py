"""A request field the control plane cannot honour is refused, naming it, never read past.

``GetEntities`` narrows by name, category, tags and bounds, and ``SetEntityState`` states its pose
in a frame. A handler that skips a field answers ``RESULT_OK`` for a different question: every
entity for a narrowed query, a pose in the world frame for one stated in another. The messages are
the real ones, so a field the service adds or renames shows up here.
"""

from __future__ import annotations

from types import SimpleNamespace

from simulation_interfaces.msg import EntityCategory, EntityFilters, Result
from simulation_interfaces.srv import GetEntities, SetEntityState, SpawnEntity

import roqsim_ros_bridge.sim_interfaces as mod
from roqsim_ros_bridge.sim_interfaces import SimInterfacesPlugin

NAMES = ["turtlebot", "obstacle_0", "obstacle_1"]


def _plugin():
    plugin = SimInterfacesPlugin()
    entities = SimpleNamespace(
        names=lambda present_only=False: list(NAMES),
        get=lambda name: SimpleNamespace(body="b", meta={}, present=True),
    )
    plugin._ctx = SimpleNamespace(entities=entities)
    return plugin


def _get(filters=None):
    req = GetEntities.Request()
    if filters is not None:
        req.filters = filters
    return _plugin()._get_entities(req, GetEntities.Response())


def test_no_filter_lists_every_present_entity():
    resp = _get()
    assert resp.result.result == Result.RESULT_OK
    assert list(resp.entities) == NAMES


def test_a_name_filter_narrows_the_list():
    resp = _get(EntityFilters(filter="^obstacle_"))
    assert resp.result.result == Result.RESULT_OK
    assert list(resp.entities) == ["obstacle_0", "obstacle_1"]


def test_a_filter_that_is_not_a_regex_is_refused():
    resp = _get(EntityFilters(filter="obstacle_("))
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "'filter'" in resp.result.error_message


def test_a_category_filter_is_refused_by_name():
    filters = EntityFilters(categories=[EntityCategory(category=EntityCategory.CATEGORY_ROBOT)])
    resp = _get(filters)
    assert resp.result.result == Result.RESULT_FEATURE_UNSUPPORTED
    assert "'categories'" in resp.result.error_message
    assert not list(resp.entities)


def test_tags_and_bounds_are_refused_by_name():
    filters = EntityFilters()
    filters.tags.tags = ["shelf"]
    filters.bounds.type = filters.bounds.TYPE_BOX
    resp = _get(filters)
    assert resp.result.result == Result.RESULT_FEATURE_UNSUPPORTED
    assert "'tags'" in resp.result.error_message
    assert "'bounds'" in resp.result.error_message


def _set(req, monkeypatch):
    written = []
    monkeypatch.setattr(mod, "run_on_physics", lambda ctx, fn, timeout=None: (fn(ctx), True)[1])
    monkeypatch.setattr(mod, "place_body", lambda *a, **k: written.append(a) or True)
    return _plugin()._set_entity_state(req, SetEntityState.Response()), written


def test_a_state_in_a_frame_the_simulator_does_not_know_is_refused(monkeypatch):
    req = SetEntityState.Request(entity="obstacle_0")
    req.state.header.frame_id = "odom"
    req.state.pose.position.x = 1.0
    resp, written = _set(req, monkeypatch)
    assert resp.result.result == SetEntityState.Response.INVALID_POSE
    assert "'odom'" in resp.result.error_message
    assert not written


def test_the_world_frame_by_name_is_accepted(monkeypatch):
    req = SetEntityState.Request(entity="obstacle_0")
    req.state.header.frame_id = "world"
    resp, written = _set(req, monkeypatch)
    assert resp.result.result == Result.RESULT_OK
    assert written


def test_a_zero_quaternion_is_refused_rather_than_taken_as_the_identity(monkeypatch):
    req = SetEntityState.Request(entity="obstacle_0")
    req.state.pose.orientation.w = 0.0
    resp, written = _set(req, monkeypatch)
    assert resp.result.result == SetEntityState.Response.INVALID_POSE
    assert "quaternion" in resp.result.error_message
    assert not written


def test_a_near_unit_quaternion_is_normalised(monkeypatch):
    req = SetEntityState.Request(entity="obstacle_0")
    req.state.pose.orientation.w = 2.0
    resp, written = _set(req, monkeypatch)
    assert resp.result.result == Result.RESULT_OK
    (args,) = written
    assert args[3] == (1.0, 0.0, 0.0, 0.0)


def test_a_non_finite_pose_is_invalid(monkeypatch):
    req = SetEntityState.Request(entity="obstacle_0")
    req.state.pose.position.x = float("nan")
    resp, written = _set(req, monkeypatch)
    assert resp.result.result == SetEntityState.Response.INVALID_POSE
    assert not written


def test_spawning_a_present_entity_is_name_not_unique():
    """The name is held by a listed entity and renaming is never allowed here, which
    SpawnEntity.srv answers with its own code rather than the generic failure."""
    req = SpawnEntity.Request(uri="obstacle_0")
    resp = _plugin()._spawn_entity(req, SpawnEntity.Response())
    assert resp.result.result == SpawnEntity.Response.NAME_NOT_UNIQUE
    assert "already present" in resp.result.error_message


def test_a_non_finite_twist_is_refused(monkeypatch):
    req = SetEntityState.Request(entity="obstacle_0")
    req.state.twist.angular.z = float("inf")
    resp, written = _set(req, monkeypatch)
    assert resp.result.result == Result.RESULT_OPERATION_FAILED
    assert "NaN or infinite" in resp.result.error_message
    assert not written
