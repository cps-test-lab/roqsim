"""Every config key this package's plugins read is one their catalog entry publishes.

``python3 -m roqsim.introspection describe <name>`` is how a caller outside the image learns what a
component takes, and a key read but not listed there reads, to that caller, as a key the plugin
ignores. The check is roqsim's own (:func:`roqsim.introspection.undeclared_config_reads`); these
plugins register outside the roqsim venv, so its tree-wide test does not reach them.
"""

from roqsim.introspection import undeclared_config_reads
from roqsim.registry import ENTRY_POINT_GROUP, _entry_points


def test_every_bridge_plugin_publishes_every_key_it_reads():
    ours = [
        ep for ep in _entry_points(ENTRY_POINT_GROUP) if ep.value.startswith("roqsim_ros_bridge.")
    ]
    assert ours, "no roqsim.plugins entry from roqsim_ros_bridge is installed: is ros2_ws sourced?"
    undeclared = {ep.name: undeclared_config_reads(ep.load()) for ep in ours}
    assert not any(undeclared.values()), (
        f"read but not in the plugin's Config:: block: {undeclared}"
    )
