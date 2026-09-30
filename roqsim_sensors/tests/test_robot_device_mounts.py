"""Every robot hangs its devices from its frames, and every device sits where ``robot_device_poses.json`` says.

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

from roqsim.config import instantiate_plugins, load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.frames import parse_frames
from roqsim.manifest import load_manifest, manifest_frames, resolve_parent_frame
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


def _manifest(model: str):
    pytest.importorskip("roqsim_mobile", reason="spawn_robot lives in roqsim_mobile")
    try:
        return resolve_model(model).path
    except ModelError:  # the package shipping this robot is not installed
        pytest.skip(f"{model} is not installed")


@pytest.mark.parametrize("model", sorted(POSES))
def test_each_device_hangs_from_a_frame_of_its_robot(model):
    path = _manifest(model)
    devices = [e["spawn_sensor"] for e in load_manifest(path) if "spawn_sensor" in e]
    assert devices
    for device in devices:
        assert "parent_frame" in device, device
        resolve_parent_frame(path, device["parent_frame"])


@pytest.mark.parametrize("model", sorted(POSES))
def test_every_frame_a_robot_declares_is_published(model):
    """A frame is a link of the robot description: a site, and the child of a static transform."""
    path = _manifest(model)
    frames = {f.name for f in parse_frames(manifest_frames(path), model)}
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {},
                "components": [{"spawn_robot": {"model": model, "prefix": "r_"}, "name": "r"}],
            }
        )
    )
    engine.setup()
    for name in frames:
        assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_SITE, f"r_{name}") >= 0, name
    children = {
        t.child for e in engine.ctx.interface.all() if e.name == "frames" for t in e.read().transforms
    }
    assert frames <= children, model


def test_a_world_puts_another_device_on_a_robots_frame():
    """The TurtleBot 4's OAK-D swapped for a D435 at its joint: the D435 sits where the OAK-D did."""
    at = {"parent_frame": "oakd_camera_bracket", "pose": {"position": {"x": 0.0584, "z": 0.09676}}}
    m = _compiled(
        "turtlebot4",
        {"spawn_sensor": {"model": "realsense_d435", **at}, "name": "d435"},
        {"spawn_sensor": {}, "name": "oakd", "enabled": False},
    )
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_d435_mount")
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_oakd_mount") < 0
    want = POSES["turtlebot4"]["r_oakd_mount"]
    assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]) == want["parent"]
    assert np.allclose(m.body_pos[b], want["pos"], atol=1e-9)
    assert np.allclose(m.body_quat[b], want["quat"], atol=1e-9)
