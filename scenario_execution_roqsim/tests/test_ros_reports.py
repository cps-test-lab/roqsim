# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A report over ROS: found through the bridge's endpoint map, read from the topic the map names.

ROS-free, like ``test_ros_pending_reasons``: the access is duck-typed over a node, so what can go
wrong -- a map misread, a topic re-derived instead of taken from the map, a field that does not
travel read as something else, a wait that does not say what it waits for -- is catchable without a
ROS installation. The map a real bridge sends is pinned against this reader by the bridge's own
test (``roqsim_ros_bridge/test/test_endpoint_map.py``), which runs where ROS is sourced.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from roqsim.bridge import ENDPOINT_MAP
from scenario_execution_roqsim.access import AccessError
from scenario_execution_roqsim.access import ros as ros_access


class _Msg:
    def __init__(self, data):
        self.data = data


class _Node:
    """Records the subscriptions an access makes; answers the graph queries a reason asks."""

    def __init__(self, publishers=0):
        self.subscriptions: list[tuple[object, str]] = []
        self.callbacks: dict[str, object] = {}
        self._publishers = publishers

    def create_subscription(self, msg_type, topic, callback, qos, callback_group=None):
        self.subscriptions.append((msg_type, topic))
        self.callbacks[topic] = callback
        return object()

    def count_publishers(self, topic):
        return self._publishers

    def resolve_topic_name(self, name):
        return name if name.startswith("/") else "/cell1/" + name

    def get_topic_names_and_types(self):
        return [("/cell1/ur5e/force_limit", ["std_msgs/msg/Bool"])]


class _Access(ros_access.RosAccess):
    """A RosAccess without rclpy: the map subscription is stood in for, the map fed by hand."""

    def __init__(self, node):  # noqa: D107 - no ROS node wanted here
        self._node = node
        self._group = None
        self._endpoint_map = ros_access._EndpointMap()
        self._endpoint_map_sub = object()
        self._report_values = {}


@pytest.fixture
def fake_msgs(monkeypatch):
    """A message package the access can import by the type string the map carries."""

    class Bool:
        @staticmethod
        def get_fields_and_field_types():
            return {"data": "boolean"}

    class LaserScan:
        @staticmethod
        def get_fields_and_field_types():
            return {"header": "std_msgs/Header", "ranges": "sequence<float>"}

    pkg = types.ModuleType("fake_msgs")
    msg = types.ModuleType("fake_msgs.msg")
    msg.Bool, msg.LaserScan = Bool, LaserScan
    monkeypatch.setitem(sys.modules, "fake_msgs", pkg)
    monkeypatch.setitem(sys.modules, "fake_msgs.msg", msg)
    return msg


def _map(*endpoints, owners=None):
    return _Msg(json.dumps({"owners": owners, "endpoints": list(endpoints)}))


FORCE_LIMIT = {
    "owner": "ur5e",
    "name": "force_limit",
    "topic": "/cell1/ur5e/safety_stop",  # a `topics:` rename: not derivable from the name
    "type": "fake_msgs.msg.Bool",
    "field": "tripped",
}


def test_the_map_topic_is_the_one_the_bridge_advertises_on():
    """Repeated as a literal so the access imports no simulator; this is what keeps them one name."""
    assert ros_access.RosAccess.ENDPOINT_MAP_TOPIC == ENDPOINT_MAP


def test_nothing_is_known_before_the_map_arrives_and_the_wait_says_where_it_looks():
    access = _Access(_Node())
    call = access.entity_report("ur5e", "force_limit", "tripped")
    assert call.poll() is None
    reason = call.pending_reason()
    assert "/cell1/roqsim/endpoints" in reason, "the resolved name, so a namespace mismatch shows"
    assert "/cell1/ur5e/force_limit" in reason, "and what the graph does carry"


def test_the_report_is_read_from_the_topic_the_map_names(fake_msgs):
    """Not from a topic re-derived from the entity and endpoint names: the map's is exact."""
    node = _Node(publishers=1)
    access = _Access(node)
    access._endpoint_map.store(_map(FORCE_LIMIT))
    call = access.entity_report("ur5e", "force_limit", "")

    assert call.poll() is None, "subscribed, no message yet"
    assert node.subscriptions == [(fake_msgs.Bool, "/cell1/ur5e/safety_stop")]
    assert call.pending_reason() == "no message on /cell1/ur5e/safety_stop yet"

    node.callbacks["/cell1/ur5e/safety_stop"](_Msg(True))
    reading = call.poll()
    assert reading.value is True
    assert reading.field == "tripped", "a bare report is its published field"
    assert reading.source == "/cell1/ur5e/safety_stop"


def test_a_topic_nothing_publishes_says_so(fake_msgs):
    access = _Access(_Node(publishers=0))
    access._endpoint_map.store(_map(FORCE_LIMIT))
    call = access.entity_report("ur5e", "force_limit", "tripped")
    call.poll()
    assert call.pending_reason().endswith("and nothing publishes it")


def test_two_reports_on_one_topic_share_one_subscription(fake_msgs):
    node = _Node()
    access = _Access(node)
    access._endpoint_map.store(_map(FORCE_LIMIT))
    access.entity_report("ur5e", "force_limit", "").poll()
    access.entity_report("ur5e", "force_limit", "tripped").poll()
    assert len(node.subscriptions) == 1


def test_a_field_that_does_not_travel_is_refused_naming_the_one_that_does(fake_msgs):
    """In a stepped run `force_limit.force` is readable; over ROS only `tripped` is published, and
    reading `force` as `tripped` would compare the wrong quantity without a word."""
    access = _Access(_Node())
    access._endpoint_map.store(_map(FORCE_LIMIT))
    call = access.entity_report("ur5e", "force_limit", "force")
    with pytest.raises(AccessError, match=r"publishes only its field 'tripped'.*stepped run"):
        call.poll()


def test_a_field_named_on_a_report_published_whole_is_refused(fake_msgs):
    access = _Access(_Node())
    access._endpoint_map.store(_map({**FORCE_LIMIT, "field": None}))
    with pytest.raises(AccessError, match="one value with no fields"):
        access.entity_report("ur5e", "force_limit", "tripped").poll()


def test_a_structured_publication_is_refused_rather_than_compared(fake_msgs):
    access = _Access(_Node())
    access._endpoint_map.store(
        _map({**FORCE_LIMIT, "name": "scan", "type": "fake_msgs.msg.LaserScan", "field": None})
    )
    with pytest.raises(AccessError, match="structured message"):
        access.entity_report("ur5e", "scan", "").poll()


@pytest.mark.parametrize(
    "entity,report,message",
    [
        ("ur5e", "force_limt", r"no report 'ur5e.force_limt'.*publishes: force_limit"),
        ("ur10e", "force_limit", r"no reports for an entity 'ur10e'.*ur5e: force_limit"),
    ],
)
def test_a_name_the_map_does_not_carry_raises_listing_what_it_does(entity, report, message):
    access = _Access(_Node())
    access._endpoint_map.store(_map(FORCE_LIMIT))
    with pytest.raises(AccessError, match=message):
        access.entity_report(entity, report, "").poll()


def test_an_entity_no_bridge_heard_from_serves_is_waited_for_not_refused(fake_msgs):
    """Two bridges split by owner: the one serving `ur10e` may simply not have been heard from yet,
    so its absence from the other's map is not an answer."""
    access = _Access(_Node())
    access._endpoint_map.store(_map(FORCE_LIMIT, owners=["ur5e"]))
    call = access.entity_report("ur10e", "force_limit", "")
    assert call.poll() is None
    assert "serves entity 'ur10e'" in call.pending_reason()

    access._endpoint_map.store(_map({**FORCE_LIMIT, "owner": "ur10e"}, owners=["ur10e"]))
    assert call.poll() is None, "found in the second map, now subscribed"
    assert ("ur5e", "force_limit") in access._endpoint_map.entries, "the maps are merged"


def test_a_map_that_is_not_the_bridges_shape_is_reported_not_swallowed():
    """Kept by the callback and raised from the tick: raising in a subscription callback would stop
    the executor, and every other subscription with it."""
    access = _Access(_Node())
    access._endpoint_map.store(_Msg("not json"))
    with pytest.raises(AccessError, match="not the shape"):
        access.entity_report("ur5e", "force_limit", "").poll()
