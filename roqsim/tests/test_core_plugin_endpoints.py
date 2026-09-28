"""The core plugins declare their endpoints on typed methods: what ``plugins describe`` publishes.

Each plugin's own tests check what its endpoints carry at run time; this pins the declared shape --
names, kinds, the result type with its units, and a user-facing first docstring line.
"""

from __future__ import annotations

import pytest

from roqsim.introspection import get_plugin_details

#: plugin -> {endpoint name: (kind, result type, {field: unit})}
EXPECTED = {
    "bumper": {"bumper/{item}": ("out", "bool", {})},
    "clearance_monitor": {
        "clearance": ("out", "ClearanceReport", {"current": "m", "minimum": "m", "at_time": "s"})
    },
    "contact_impulse": {
        "contact_impulse": (
            "out",
            "ContactImpulseReport",
            {"impulse_ns": "N*s", "peak_normal_n": "N", "contact_time_s": "s", "normal_n": "N"},
        )
    },
    "contact_location": {
        "contact_location": (
            "out",
            "ContactLocation",
            {"x": "m", "y": "m", "z": "m", "extent": "m", "time": "s"},
        )
    },
    "contact_monitor": {"contact": ("out", "ContactReport", {"first_time": "s"})},
    "energy_monitor": {
        "battery": (
            "out",
            "EnergyReport",
            {"energy_j": "J", "power_w": "W", "voltage": "V", "current_a": "A"},
        )
    },
    "joint_state_publisher": {"joint_states": ("out", "tuple[list[str], array, array, array]", {})},
    "model_override": {
        "override": ("command", "none", {}),
        "override_state": ("out", "OverrideReport", {"since": "s"}),
        "override_verified": ("out", "OverrideReport", {"since": "s"}),
    },
    "spawn_model": {"{item}_pose": ("out", None, {})},
}


@pytest.mark.parametrize("plugin", sorted(EXPECTED))
def test_a_core_plugin_describes_its_endpoints(plugin):
    rows = get_plugin_details(plugin)["endpoints"]
    by_name = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    assert set(by_name) == set(EXPECTED[plugin])
    for name, (kind, result, units) in EXPECTED[plugin].items():
        for row in by_name[name]:
            assert row["kind"] == kind
            assert row["backends"] == ["ros2"]
            assert row["doc"], f"{plugin}/{name} has no docstring for describe to show"
            if result is not None:
                assert row["result"]["type"] == result
            fields = {f["name"]: f for f in row["result"].get("fields", [])}
            for field, unit in units.items():
                assert fields[field]["unit"] == unit, (plugin, name, field)


def test_the_bumper_is_one_family_over_its_zones():
    (row,) = get_plugin_details("bumper")["endpoints"]
    assert row["family"] is True


def test_model_override_takes_setbools_data():
    rows = {r["name"]: r for r in get_plugin_details("model_override")["endpoints"]}
    assert rows["override"]["params"] == [
        {
            "name": "data",
            "type": "bool",
            "doc": "true applies the override, false restores",
            "required": True,
        }
    ]


def test_spawn_model_declares_a_streamed_and_a_static_pose_each_switched_by_publish_tf():
    rows = get_plugin_details("spawn_model")["endpoints"]
    assert len(rows) == 2 and all(r["conditional"] and r["family"] for r in rows)
