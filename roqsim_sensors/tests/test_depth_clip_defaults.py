"""A depth camera's working range is judged against the defaults it runs with.

Each device states its own ``clip_near``/``clip_far``. A world that sets one of them alone is checked
against the device's other one, so a range that leaves no depth at all is refused when the world
loads rather than published as a stream of "no return".
"""

from __future__ import annotations

import pytest

from roqsim_sensors.plugins.oakd_camera import OakDCameraPlugin
from roqsim_sensors.plugins.realsense_d435 import RealsenseD435Plugin
from roqsim_sensors.plugins.realsense_d455 import RealsenseD455Plugin
from roqsim_sensors.plugins.zivid import ZividPlugin

DEVICES = [
    (OakDCameraPlugin, 0.3, 100.0),
    (RealsenseD435Plugin, 0.28, 3.0),
    (RealsenseD455Plugin, 0.6, 6.0),
    (ZividPlugin, 1.3, 5.0),
]


@pytest.mark.parametrize(("cls", "near", "far"), DEVICES, ids=lambda v: getattr(v, "__name__", ""))
def test_a_device_runs_with_the_range_it_states(cls, near, far):
    plugin = cls({})
    assert (plugin.clip_near, plugin.clip_far) == (near, far)
    assert plugin.validate_config({}) == []


@pytest.mark.parametrize(("cls", "near", "far"), DEVICES, ids=lambda v: getattr(v, "__name__", ""))
def test_a_near_clip_past_the_devices_far_clip_is_refused(cls, near, far):
    config = {"clip_near": far + 1.0}
    errors = cls(config).validate_config(config)
    assert any("'clip_far'" in e and "'clip_near'" in e for e in errors), errors


@pytest.mark.parametrize(("cls", "near", "far"), DEVICES, ids=lambda v: getattr(v, "__name__", ""))
def test_a_far_clip_short_of_the_devices_near_clip_is_refused(cls, near, far):
    config = {"clip_far": near / 2.0}
    errors = cls(config).validate_config(config)
    assert any("'clip_far'" in e and "'clip_near'" in e for e in errors), errors


def test_the_16uc1_ceilings_read_the_devices_own_far_clip():
    """The D435 reaches 3 m, inside both 16UC1 ceilings, so it needs no clip_far to opt in."""
    config = {"depth_encoding": "16UC1"}
    assert RealsenseD435Plugin(config).validate_config(config) == []
    errors = OakDCameraPlugin(config).validate_config(config)
    assert any("65.535" in e for e in errors), errors
