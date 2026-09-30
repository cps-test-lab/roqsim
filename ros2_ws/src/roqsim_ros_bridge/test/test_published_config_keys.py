"""Every plugin this package registers publishes the config keys it reads.

``roqsim plugins describe`` is what a caller checks a world against before running it, so a key the
bridge reads and does not publish is one that check refuses in a valid world. roqsim's own guard
covers the plugins installed beside it; these register from this package, so they are held here,
by the same scan (:func:`roqsim.introspection.undeclared_config_reads`).
"""

from importlib.metadata import entry_points

import pytest

from roqsim.introspection import undeclared_config_reads
from roqsim.registry import ENTRY_POINT_GROUP

#: Keys a plugin reads only to refuse them with a reason; not settings, so not published.
REFUSED = {
    ("ros2_bridge", "gt"): "outputs are not moved under a ground-truth prefix",
}

PLUGINS = sorted(
    (ep.name, ep.load())
    for ep in entry_points(group=ENTRY_POINT_GROUP)
    if ep.value.startswith("roqsim_ros_bridge.")
)


def test_the_package_registers_plugins():
    """An entry the scan could not find would be one it skipped."""
    assert PLUGINS, "no roqsim.plugins entry of roqsim_ros_bridge is installed"


@pytest.mark.parametrize(("name", "cls"), PLUGINS, ids=[name for name, _ in PLUGINS])
def test_a_plugin_publishes_every_key_it_reads(name, cls):
    missing = [key for key in undeclared_config_reads(cls) if (name, key) not in REFUSED]
    assert not missing, (
        f"{name} reads {missing}, which it does not publish (roqsim plugins describe {name}): add "
        f"them to its Config:: block or CONFIG_SCHEMA"
    )


@pytest.mark.parametrize(("name", "key"), sorted(REFUSED))
def test_a_key_read_to_be_refused_is_refused(name, key):
    """An exemption that outlives its refusal would hide a key read as a setting."""
    cls = dict(PLUGINS)[name]
    errors = cls({}).config_errors({key: 1})
    assert any(f"'{key}'" in e for e in errors), (name, key, errors)
