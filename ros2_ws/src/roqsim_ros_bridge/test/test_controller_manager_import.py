# SPDX-License-Identifier: Apache-2.0
"""A controller_manager surface that cannot be imported fails the bridge's start.

``controller_manager_msgs`` is a declared dependency of the bridge, so an import that fails there is a
broken install. A bridge that started anyway would serve a world with controllers and no controller
services, and the run would look healthy.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
pytest.importorskip("rclpy")

from roqsim.context import SimContext  # noqa: E402
from roqsim_ros_bridge import ros2_bridge  # noqa: E402
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge  # noqa: E402


def test_a_failed_controller_manager_import_fails_the_start_and_names_it(monkeypatch):
    monkeypatch.setattr(ros2_bridge, "load_extensions", lambda: None)
    # None in sys.modules makes the import raise ImportError naming the module.
    monkeypatch.setitem(sys.modules, "roqsim_ros_bridge.controller_manager", None)
    ctx = SimContext(config={})
    bridge = Ros2Bridge({"domain_id": 100 + os.getpid() % 100, "clock_rate_hz": 0})
    try:
        with pytest.raises(ImportError, match="controller_manager surface cannot start") as info:
            bridge._setup(ctx)
    finally:
        bridge._teardown(ctx)

    assert "roqsim_ros_bridge.controller_manager" in str(info.value)
    assert info.value.name == "roqsim_ros_bridge.controller_manager"
    assert isinstance(info.value.__cause__, ImportError)
