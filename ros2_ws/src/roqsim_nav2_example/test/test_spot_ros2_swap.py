"""The simulated Spot is a drop-in for the real one, for a consumer written against spot_ros2.

Starts the simulated Spot behind the ROS 2 bridge and runs ``spot_ros2_consumer.py`` against it,
unchanged: no remapping, no ``use_sim_time``, nothing that knows it talks to a simulator. Its
expectations are ``spot_ros2_interface.json``, pinned to the spot_ros2 commit it names.

``spot_nav2.yaml`` is the Spot a world gets by default: the cameras and their tf are spot_ros2's
there. ``spot_ros2_interface.yaml`` adds the driver's state topics; the one field that still
differs is the odometry's child frame (``base_link`` for ``body``, an identity apart), listed in
:data:`KNOWN_DIFFERENCES` so a change to it is seen.

Skips without ROS, and without the Spot walking policy (``python -m
roqsim_quadruped.policy.fetch_policy``), which both worlds run.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("tf2_ros")
policy = pytest.importorskip("roqsim_quadruped.policy")

HERE = Path(__file__).parent
WORLDS = HERE.parent / "worlds"
CONSUMER = HERE / "spot_ros2_consumer.py"
INTERFACE = json.loads((HERE / "spot_ros2_interface.json").read_text())
POLICY = os.environ.get("SPOT_POLICY_PATH") or str(policy.DEFAULT_POLICY)
#: What the consumer finds different from the real robot, and why it stays: see the world's header.
KNOWN_DIFFERENCES = {"odometry.child_frame_id": "base_link"}

pytestmark = pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)


def _bridge(world: str) -> subprocess.Popen:
    if shutil.which("ros2") is None:
        pytest.skip("ros2 CLI not on PATH (ROS not sourced)")
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "roqsim_ros_bridge.run_bridge",
            "--world",
            str(WORLDS / world),
            "--headless",
        ],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _consume(*args: str) -> dict:
    # A clean environment for the consumer: nothing of the simulator's, no sim time.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROQSIM", "MUJOCO"))}
    done = subprocess.run(
        [sys.executable, str(CONSUMER), *args, "--timeout", "120"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(done.stdout)
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def default_spot():
    proc = _bridge("spot_nav2.yaml")
    time.sleep(5.0)
    yield proc
    _stop(proc)


@pytest.mark.parametrize("station", INTERFACE["stations"])
def test_a_spot_ros2_perception_consumer_runs_unchanged(default_spot, station):
    report = _consume("--station", station)
    assert report.get("ok"), report


def test_the_state_interface_world_matches_spot_ros2_but_for_the_odometry_child():
    proc = _bridge("spot_ros2_interface.yaml")
    try:
        time.sleep(5.0)
        report = _consume("--state")
    finally:
        _stop(proc)
    failed = {k: v["seen"] for k, v in report["checks"].items() if not v["ok"]}
    assert failed == KNOWN_DIFFERENCES, report


def test_the_state_interface_world_names_every_joint_as_spot_ros2_does():
    import yaml

    world = yaml.safe_load((WORLDS / "spot_ros2_interface.yaml").read_text())
    robot = next(c for c in world["components"] if "spawn_robot" in c)
    names = next(c for c in robot["components"] if "spot_locomotion" in c)["spot_locomotion"]
    assert sorted(names["joint_names"].values()) == sorted(
        INTERFACE["state"]["joint_states"]["names"]
    )
