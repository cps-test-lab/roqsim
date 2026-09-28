"""Every robot mounts its devices by name, and every device sits where ``robot_device_poses.json`` says.

The fixture holds, per robot, each device body's parent and its pose in that parent, as compiled
from the robot's manifest. A device a manifest moves fails here, naming the robot and the body;
a deliberate move updates the fixture with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import (
    PluginError,
    instantiate_plugins,
    load_config_from_dict,
    overrides_from_dotlist,
)
from roqsim.context import SimContext
from roqsim.manifest import load_manifest, manifest_mounts, resolve_mount
from roqsim.models import ModelError, resolve_model

POSES = json.loads((Path(__file__).parent / "robot_device_poses.json").read_text())
SPAWNS = ("spawn_robot", "spawn_sensor")


def _compiled(model: str, *components) -> mujoco.MjModel:
    """The robot and its devices as spawned, built by the spawn plugins alone.

    *components* are nested under the robot, as a world declares them.
    """
    pytest.importorskip("roqsim_mobile", reason="spawn_robot lives in roqsim_mobile")
    try:
        resolve_model(model)
    except ModelError:  # the package shipping this robot is not installed
        pytest.skip(f"{model} is not installed")
    robot = {"spawn_robot": {"model": model, "prefix": "r_"}, "name": "r"}
    if components:
        robot["components"] = list(components)
    cfg = load_config_from_dict({"sim": {}, "components": [robot]})
    spawns = {s.address for s in cfg.plugins if s.ref in SPAWNS}
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    ctx = SimContext(cfg.raw)
    for plugin in instantiate_plugins(cfg):
        if plugin.address in spawns:
            plugin.build(spec, ctx)
    return spec.compile()


@pytest.mark.parametrize("model", sorted(POSES))
def test_every_device_body_is_where_it_was(model):
    m = _compiled(model)
    devices = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b
        for b in range(m.nbody)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) in POSES[model]
    }
    assert sorted(devices) == sorted(POSES[model])
    for name, b in devices.items():
        want = POSES[model][name]
        parent = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])
        assert parent == want["parent"], name
        assert np.allclose(m.body_pos[b], want["pos"], atol=1e-9), name
        assert np.allclose(m.body_quat[b], want["quat"], atol=1e-9), name


@pytest.mark.parametrize("model", sorted(POSES))
def test_a_robot_mounts_each_device_by_name_and_every_mount_resolves(model):
    pytest.importorskip("roqsim_mobile", reason="spawn_robot lives in roqsim_mobile")
    try:
        path = resolve_model(model).path
    except ModelError:  # the package shipping this robot is not installed
        pytest.skip(f"{model} is not installed")
    devices = [e["spawn_sensor"] for e in load_manifest(path) if "spawn_sensor" in e]
    assert devices
    for device in devices:
        assert "mount" in device and not {"parent_frame", "pos", "rpy"} & set(device), device
    for mount in manifest_mounts(path):
        resolve_mount(path, mount["name"])


def test_a_world_puts_another_device_on_a_robots_mount_without_its_offset():
    """The TurtleBot 4's OAK-D swapped for a D435 on the same mount: the D435 sits where the OAK-D did."""
    m = _compiled(
        "turtlebot4",
        {"spawn_sensor": {"model": "realsense_d435", "mount": "oakd"}, "name": "d435"},
        {"spawn_sensor": {}, "name": "oakd", "enabled": False},
    )
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_d435_mount")
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_oakd_mount") < 0
    want = POSES["turtlebot4"]["r_oakd_mount"]
    assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]) == want["parent"]
    assert np.allclose(m.body_pos[b], want["pos"], atol=1e-9)
    assert np.allclose(m.body_quat[b], want["quat"], atol=1e-9)


@pytest.mark.parametrize("key", ["pos=[0,0,0.5]", "rpy=[0,0,0]", "parent_frame=base_link"])
def test_an_override_cannot_move_a_device_off_the_robots_mount(key):
    """A late override would land after the mount resolved and leave `mount:` naming where it is not."""
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 manifest lives in roqsim_mobile")
    with pytest.raises(PluginError, match=r"read while that component expands"):
        load_config_from_dict(
            {"sim": {}, "components": [{"spawn_robot": {"model": "turtlebot4"}, "name": "robot"}]},
            overrides=overrides_from_dotlist([f"components.robot.rplidar.{key}"]),
        )
