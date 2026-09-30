"""The navigator's config is its schema: every key it reads is declared, and no other is accepted."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import roqsim_nav
import roqsim_walker
from roqsim.config import instantiate_plugins, load_config, load_config_from_dict
from roqsim.introspection import get_plugin_details
from roqsim.plugin import PluginError
from roqsim_nav.caution import CautionProbe
from roqsim_nav.plugins import navigator
from roqsim_nav.plugins.navigator import NavigatorPlugin

REPO = Path(__file__).resolve().parents[2]
MOCAP_BOX = """<mujoco model="box">
  <worldbody><body name="box"><geom type="box" size=".2 .2 .2"/></body></worldbody>
</mujoco>"""


def _load(tmp_path, **navigator):
    """Instantiate a world with one mocap prop carrying a navigator configured as given."""
    model = tmp_path / "box.xml"
    model.write_text(MOCAP_BOX)
    world = {
        "sim": {},
        "components": [
            {
                "spawn_model": {"model": str(model), "motion": "driven"},
                "name": "cart",
                "components": [{"navigator": {"speed": 0.5, "goals": [[1.0, 0.0]], **navigator}}],
            }
        ],
    }
    return instantiate_plugins(load_config_from_dict(world, base_dir=tmp_path))


def test_a_complete_config_loads(tmp_path):
    _load(
        tmp_path,
        output="mocap",
        dwell=[0.0, [1.0, 2.0]],
        yaw_rate=2.0,
        obstacle_height=[0.05, 0.6],
        planner={"inflation_radius": 0.3, "waypoint_radius": 0.3},
        recovery={"enabled": True, "stuck_time": 1.0, "stuck_eps": 0.1, "max_recovery": 2},
        avoidance={"stop": True, "lookahead": 0.8, "width": 0.5, "ignore": []},
        action_names={"navigate_through_poses": "patrol"},
        actions=["navigate_through_poses"],
    )


@pytest.mark.parametrize(
    ("config", "names"),
    [
        ({"sped": 1.0}, "'sped' is not a setting of this component -- did you mean 'speed'?"),
        # Spellings the navigator once read. Each is now an unknown key like any other.
        ({"traffic": "ignore"}, "'traffic' is not a setting of this component"),
        ({"action_name": "patrol"}, "did you mean 'action_names'?"),
        ({"caution": {"lookahead": 1.0}}, "'caution' is not a setting of this component"),
        ({"avoidance": {"on_blocked": "replan"}}, "'avoidance.on_blocked' is not a key of"),
        # A typo inside each nested block.
        ({"avoidance": {"lookahed": 1.0}}, "'avoidance.lookahed' is not a key of 'avoidance'"),
        ({"planner": {"inflation_radus": 0.3}}, "did you mean 'inflation_radius'?"),
        ({"recovery": {"stuck": 1.0}}, "'recovery.stuck' is not a key of 'recovery'"),
        ({"action_names": {"navigate_to_poses": "x"}}, "'action_names.navigate_to_poses' is not"),
        ({"actions": ["navigate_to_poses"]}, "'actions' names 'navigate_to_poses'"),
    ],
)
def test_a_key_the_navigator_does_not_read_is_refused_by_name(tmp_path, config, names):
    with pytest.raises(PluginError) as exc:
        _load(tmp_path, **config)
    assert names in str(exc.value)


def test_a_nested_value_of_the_wrong_type_is_refused(tmp_path):
    with pytest.raises(PluginError, match=r"'avoidance\.width' must be float"):
        _load(tmp_path, avoidance={"width": "wide"})


def test_the_schema_covers_every_key_of_the_probe():
    assert set(navigator._PROBE_FIELDS) == set(CautionProbe.KEYS)


def test_the_catalog_publishes_the_schema_as_strict():
    details = get_plugin_details("navigator")
    assert details["strict_keys"] is True
    names = [row["name"] for row in details["schema"]]
    assert names == list(NavigatorPlugin.CONFIG_SCHEMA)
    assert "action_name" not in names and "traffic" not in names


def test_the_catalog_publishes_each_nested_block_s_keys():
    schema = {row["name"]: row for row in get_plugin_details("navigator")["schema"]}
    for block, keys in (
        ("avoidance", navigator.AVOIDANCE_SCHEMA),
        ("planner", navigator.PLANNER_SCHEMA),
        ("recovery", navigator.RECOVERY_SCHEMA),
        ("action_names", NavigatorPlugin.ACTIONS),
    ):
        assert [f["name"] for f in schema[block]["fields"]] == list(keys), block
    (stuck,) = [f for f in schema["recovery"]["fields"] if f["name"] == "stuck_time"]
    assert stuck == {
        "name": "stuck_time",
        "type": "float",
        "required": False,
        "default": 1.5,
        "unit": "s",
        "doc": "window progress is measured over",
    }


def test_settings_read_a_nested_default():
    settings = NavigatorPlugin({"speed": 0.5, "recovery": {"enabled": False}}).settings
    assert settings.recovery.enabled is False and settings.recovery.stuck_time == 1.5
    assert settings.action_names.start_route == "start_route"


# -- every shipped world that configures a navigator ----------------------------------------------
WALKER_WORLDS = Path(roqsim_walker.__file__).parent / "worlds"
WALKER_NAV2 = REPO / "ros2_ws/src/roqsim_walker_ros/worlds/walker_nav2.yaml"


def _navigators(plugins):
    return [p for p in plugins if isinstance(p, NavigatorPlugin)]


@pytest.mark.parametrize(
    "world",
    [Path(roqsim_nav.WORLDS_DIR) / "nav_opponents.yaml", WALKER_WORLDS / "walker_patrol.yaml"],
    ids=lambda p: p.stem,
)
def test_a_shipped_world_loads(world):
    assert _navigators(instantiate_plugins(load_config(str(world))))


def test_the_ros_walker_world_loads_and_names_its_action():
    """Its bridge entries are ROS packages, so the walker is loaded on its own."""
    doc = yaml.safe_load(WALKER_NAV2.read_text())
    doc["components"] = [c for c in doc["components"] if "walker" in c]
    (nav,) = _navigators(
        instantiate_plugins(load_config_from_dict(doc, base_dir=WALKER_NAV2.parent))
    )
    assert nav.config["action_names"] == {"navigate_through_poses": "navigate_through_poses"}
