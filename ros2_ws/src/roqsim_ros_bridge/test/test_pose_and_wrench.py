# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Poses and wrenches across the wire: quaternion order, and the full orientation.

Two failures sit behind these. A plugin declared ``WrenchStamped`` and no converter existed, so the
reflective fallback raised at publish time -- the wrench a contact task is built on never reached
ROS at all. And the pose decoder projected every orientation down to a yaw, which suits an airframe
and discards exactly what a Cartesian controller commanded to hold its tool upright needs.
"""

from __future__ import annotations

import math

import pytest

from roqsim_ros_bridge.registry import get_converter, get_decoder


class _Vec3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Quat(_Vec3):
    def __init__(self):
        super().__init__()
        self.w = 1.0


class _Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class _Wrench:
    def __init__(self):
        self.force, self.torque = _Vec3(), _Vec3()


class _WrenchMsg:
    """Stand-in for geometry_msgs.msg.WrenchStamped."""

    def __init__(self):
        self.header, self.wrench = _Header(), _Wrench()


class _Pose:
    def __init__(self):
        self.position, self.orientation = _Vec3(), _Quat()


class _PoseMsg:
    """Stand-in for geometry_msgs.msg.PoseStamped."""

    def __init__(self):
        self.header, self.pose = _Header(), _Pose()


# -- wrench --------------------------------------------------------------------------------------


def test_a_wrench_has_a_converter_at_all():
    """The bug itself: a declared type with no converter falls back to ``msg.data``, which a wrench
    does not have, and raises at the first publish rather than at configure."""
    fill = get_converter("geometry_msgs.msg.WrenchStamped")
    msg = _WrenchMsg()
    fill(msg, ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]), None, {})
    assert (msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z) == (1.0, 2.0, 3.0)
    assert (msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z) == (0.1, 0.2, 0.3)


def test_a_wrench_carries_the_frame_it_was_resolved_in():
    """A wrench in the sensor frame consumed as the world's makes a controller drift sideways under
    load and look like a friction problem. The header is how a subscriber can tell."""
    fill = get_converter("geometry_msgs.msg.WrenchStamped")
    msg = _WrenchMsg()
    fill(msg, ([0.0] * 3, [0.0] * 3), None, {"frame_id": "tool0"})
    assert msg.header.frame_id == "tool0"


def test_a_commanded_wrench_decodes_to_the_readers_own_shape():
    msg = _WrenchMsg()
    msg.wrench.force.x, msg.wrench.force.z = 1.5, -8.0
    msg.wrench.torque.y = 0.25
    force, torque = get_decoder("geometry_msgs.msg.WrenchStamped")(msg)
    assert tuple(force) == (1.5, 0.0, -8.0)
    assert tuple(torque) == (0.0, 0.25, 0.0)


# -- pose ----------------------------------------------------------------------------------------


def test_a_pose_decodes_with_its_orientation_intact():
    """Not a yaw. A tool commanded to stay upright is commanded in the part a yaw throws away."""
    msg = _PoseMsg()
    msg.pose.position.x, msg.pose.position.z = 0.4, 0.3
    # +90 deg about Y -- pure pitch, which a yaw projection loses entirely.
    msg.pose.orientation.w = msg.pose.orientation.y = math.sqrt(0.5)
    msg.pose.orientation.x = msg.pose.orientation.z = 0.0

    position, quat = get_decoder("geometry_msgs.msg.PoseStamped")(msg)

    assert tuple(position) == (0.4, 0.0, 0.3)
    assert quat[0] == pytest.approx(math.sqrt(0.5)), "w comes first, MuJoCo's order"
    assert quat[2] == pytest.approx(math.sqrt(0.5)), "the pitch must survive the decode"


def test_the_quaternion_is_reordered_on_the_way_out():
    """MuJoCo puts w first and ROS puts it last. Passed straight through, the result is still a
    valid-looking orientation and not the one anybody commanded."""
    fill = get_converter("geometry_msgs.msg.PoseStamped")
    msg = _PoseMsg()
    fill(msg, ([0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]), None, {})  # (w, x, y, z)
    o = msg.pose.orientation
    assert (o.w, o.x, o.y, o.z) == (0.1, 0.2, 0.3, 0.4)


def test_a_pose_survives_a_round_trip():
    fill = get_converter("geometry_msgs.msg.PoseStamped")
    msg = _PoseMsg()
    original = ([0.4, -0.1, 0.3], [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
    fill(msg, original, None, {})

    position, quat = get_decoder("geometry_msgs.msg.PoseStamped")(msg)
    assert list(position) == pytest.approx(original[0])
    assert list(quat) == pytest.approx(original[1])


# -- the guard that would have caught the wrench --------------------------------------------------


def test_every_out_topic_type_a_shipped_plugin_declares_has_a_converter():
    """A declared out type with no converter reaches the reflective ``msg.data`` fallback, which
    raises for any structured message -- at the first PUBLISH, not at configure, so a world starts,
    runs, and produces nothing on that topic.

    The sibling of ``test_every_service_a_shipped_plugin_declares_has_a_handler``. Direction is read
    off the same ``Endpoint(...)`` call as the type, because an `in` endpoint needs a DECODER and
    would otherwise be reported here as a missing converter.
    """
    import ast
    import importlib
    from importlib.metadata import entry_points

    from roqsim_ros_bridge.registry import CONVERTERS

    def out_types(tree):
        """Type hints of every ``Endpoint(direction="out", ...)`` literal in a module."""
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Endpoint"):
                continue
            kwargs = {k.arg: k.value for k in node.keywords}
            direction = kwargs.get("direction")
            if not (isinstance(direction, ast.Constant) and direction.value == "out"):
                continue  # `in` endpoints are served by a decoder, not a converter
            backend = kwargs.get("backend")
            for hint in ast.walk(backend) if backend is not None else ():
                if not isinstance(hint, ast.Dict):
                    continue
                for key, value in zip(hint.keys, hint.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "type"
                        and isinstance(value, ast.Constant)
                    ):
                        yield value.value

    declared: dict = {}
    for entry in entry_points(group="roqsim.plugins"):
        module_name = entry.value.split(":")[0]
        try:
            source = importlib.import_module(module_name).__file__
        except Exception:  # noqa: BLE001 - an optional extra's plugin is not this test's business
            continue
        if not source:
            continue
        with open(source, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for type_path in out_types(tree):
            declared.setdefault(type_path, set()).add(entry.name)

    assert declared, "the scan found no out-endpoint type hints; has the declaration shape changed?"
    # std_msgs primitives are served by the reflective fallback by design -- they have `data`.
    missing = {
        type_path: sorted(plugins)
        for type_path, plugins in declared.items()
        if type_path not in CONVERTERS and not type_path.startswith("std_msgs.")
    }
    assert not missing, (
        f"declared by a plugin and converted by nobody: {missing}. The fallback needs a 'data' "
        "field, so these raise at the first publish. Add a converter in roqsim_ros_bridge.registry."
    )
