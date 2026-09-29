"""find_endpoint: an endpoint addressed by its entity or its component, through roqsim.paths."""

from __future__ import annotations

import pytest

from scenario_execution_roqsim.access import AccessError, find_endpoint

ROWS = [
    {"path": "robot/lidar/override", "owner": "robot", "name": "override", "kind": "command"},
    {"path": "robot/lidar2/override", "owner": "robot", "name": "override", "kind": "command"},
    {"path": "ur5e/force_torque/tare", "owner": "ur5e", "name": "tare", "kind": "command"},
    {"path": "ur5e/force_torque/wrench", "owner": "ur5e", "name": "wrench", "kind": "out"},
]


def test_an_endpoint_is_found_by_its_owner_or_by_its_component():
    assert find_endpoint(ROWS, "ur5e", "tare", kind="in") is ROWS[2]
    assert find_endpoint(ROWS, "ur5e", "force_torque/tare", kind="in") is ROWS[2]
    assert find_endpoint(ROWS, "ur5e.force_torque", "tare", kind="in") is ROWS[2]
    assert find_endpoint(ROWS, "robot.lidar2", "override", kind="in") is ROWS[1]


def test_one_owner_with_two_of_a_name_is_refused_naming_both():
    with pytest.raises(AccessError, match="robot/lidar/override, robot/lidar2/override"):
        find_endpoint(ROWS, "robot", "override", kind="in")


def test_an_unknown_endpoint_names_the_nearest():
    with pytest.raises(AccessError, match=r"no command or stream 'tar' of 'ur5e'\. Did you mean"):
        find_endpoint(ROWS, "ur5e", "tar", kind="in")
    with pytest.raises(AccessError, match="no report 'tare' of 'ur5e'"):
        find_endpoint(ROWS, "ur5e", "tare", kind="out")
