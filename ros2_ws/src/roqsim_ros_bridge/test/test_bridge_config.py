"""The ``ros2_bridge`` plugin's config check: a key it does not honour is refused, not ignored.

``gt`` is not an option of the bridge. A world that sets it expects its outputs under a prefix, so
loading it must fail with the reason rather than run with every topic somewhere else.
"""

from pathlib import Path

import pytest

from roqsim.config import instantiate_plugins, load_config_from_dict
from roqsim.plugin import PluginError
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge


def test_a_gt_block_is_refused_when_the_world_loads():
    world = {"components": [{"ros2_bridge": {"gt": {"prefix": "/gt", "exempt": ["odom"]}}}]}
    with pytest.raises(PluginError, match="'gt' is not a key of ros2_bridge"):
        instantiate_plugins(load_config_from_dict(world, base_dir=Path(".")))


def test_a_config_without_it_passes():
    bridge = Ros2Bridge({"namespace": "sim", "strip_namespace": "robot"})
    assert bridge.config_errors(bridge.config) == []
