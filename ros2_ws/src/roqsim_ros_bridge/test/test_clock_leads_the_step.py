# SPDX-License-Identifier: Apache-2.0
"""``/clock`` reaches a step's time before any message stamped with it does.

Every output of a step carries that step's sim time. Published before the clock tick, a scan
arrives stamped ahead of its subscriber's clock -- a message from the future to tf2 and to a message
filter. No ROS graph is needed to see the order: the bridge's publishers are replaced by recorders.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")

from roqsim.bridge import _Output, _RateGate  # noqa: E402
from roqsim.context import Endpoint  # noqa: E402
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge  # noqa: E402


class _Recorder:
    def __init__(self, log, name):
        self._log, self._name = log, name

    def publish(self, msg):
        self._log.append(self._name)


def test_the_clock_is_published_before_the_steps_outputs():
    log: list[str] = []
    bridge = Ros2Bridge({})
    bridge._ready = True
    bridge._context = SimpleNamespace(ok=lambda: True)
    bridge._clock_pub = _Recorder(log, "clock")
    scan = Endpoint(name="scan", direction="out", owner="robot", read=lambda: object())
    bridge._outputs = [_Output(endpoint=scan, handle="scan", gate=_RateGate(-1.0))]
    bridge._publish = lambda handle, payload, stamp: log.append(handle)
    bridge._peer_gate = _RateGate(1e-9)  # the graph check is not what is under test
    bridge._peer_gate.due(0.0)
    bridge.post_step(SimpleNamespace(sim_time=0.25))
    assert log == ["clock", "scan"]
