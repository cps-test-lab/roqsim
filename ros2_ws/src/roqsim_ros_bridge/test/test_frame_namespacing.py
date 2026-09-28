"""The bridge namespace prefixes a robot's frames and leaves the frames all robots share bare."""

from __future__ import annotations

import pytest

from roqsim_ros_bridge.frames import GLOBAL_FRAMES, namespaced


@pytest.mark.parametrize("name", ["base_link", "odom", "tool0"])
def test_a_robot_frame_is_namespaced(name):
    assert namespaced("robot_b", name) == f"robot_b/{name}"


@pytest.mark.parametrize("name", sorted(GLOBAL_FRAMES))
def test_a_global_frame_is_not(name):
    assert namespaced("robot_b", name) == name


def test_no_namespace_leaves_every_frame_bare():
    assert namespaced("", "base_link") == "base_link"
