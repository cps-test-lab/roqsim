# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A delete over ROS names its entity in the field ``DeleteEntity.srv`` actually has.

``SpawnEntity`` calls it ``name`` and ``DeleteEntity`` calls it ``entity``. A generated ROS message
refuses an attribute it does not declare, so writing the spawn's field into a delete request fails
at the call -- and only over ROS, since a stepped run never builds a request at all. ROS-free, like
the calls themselves: the request type is duck-typed, so a stand-in with the same slots is enough.
"""

from __future__ import annotations

from scenario_execution_roqsim.access import ros as ros_access


class _DeleteRequest:
    __slots__ = ("entity",)


class _DeleteType:
    Request = _DeleteRequest


def _access():
    access = object.__new__(ros_access.RosAccess)
    access._delete_type = _DeleteType
    access._delete_client = object()
    access._result_ok = 1
    access._advertised_services = lambda: ()
    return access


def test_a_delete_request_carries_the_entity_field():
    call = _access().set_entity_presence("crate", False)
    assert call._request.entity == "crate"
