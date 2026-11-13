"""The real libfranka, at the pinned release, against the simulated FR3's FCI.

This is the strongest form of "in sync": Franka's own client library -- its protocol, its control
loop, its rate limiter and filter, its Pinocchio ``Model`` built from the URDF the robot serves --
runs unmodified against the simulated arm (``franka_build.py`` builds it). Each behaviour is checked
from the client's side, as a program on the robot would see it, through ``franka_clients/fci_client.cpp``.
"""

from __future__ import annotations

import json
import socket
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from franka_build import BuildError, build_client, runtime_env
from roqsim_manipulation.franka.protocol import CONNECT_REQUEST, CONNECT_RESPONSE, HEADER, Command

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

CB = {
    "lower_torque": [20.0] * 7,
    "upper_torque": [40.0] * 7,
    "lower_force": [20.0] * 6,
    "upper_force": [60.0] * 6,
}


@pytest.fixture(scope="module")
def client():
    try:
        return build_client()
    except BuildError as exc:
        pytest.fail(f"libfranka could not be built for the in-sync test: {exc}")


class Sim:
    """A world stepped in a thread of its own, as `roqsim sim` steps it; ``hooks`` run per step."""

    def __init__(self, arms: dict[str, str]):
        world = {
            "sim": {"timestep": 0.001},
            "components": [
                {
                    "spawn_arm": {
                        "model": "fr3",
                        "prefix": f"{name}_",
                        "pose": {"position": {"y": 0.5 if name == "left" else -0.5}},
                    },
                    "name": name,
                    "components": [
                        {
                            "franka_fci": {
                                "collision_behavior": CB,
                                "bind": addr,
                                "client_timeout_s": 20.0,
                            }
                        }
                    ],
                }
                for name, addr in arms.items()
            ],
        }
        self.engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
        self.engine.ctx.seed = 0
        self.engine.setup()
        self.engine.reset()
        self.hooks = []
        self.error: BaseException | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop:
                for hook in self.hooks:
                    hook(self.engine)
                self.engine.step()
        except BaseException as exc:  # noqa: BLE001 - re-raised by close()
            self.error = exc

    def close(self):
        self._stop = True
        self._thread.join(timeout=30)
        self.engine.shutdown()
        if self.error is not None:
            raise self.error


@pytest.fixture
def sim():
    s = Sim({"left": "127.0.0.2"})
    yield s
    s.close()


def run(client, *args, timeout=120) -> dict:
    r = subprocess.run(
        [str(client), *map(str, args)],
        capture_output=True,
        text=True,
        env=runtime_env(),
        timeout=timeout,
    )
    lines = [line for line in r.stdout.splitlines() if line.startswith("{")]
    assert lines, f"fci_client {args}: rc {r.returncode}\n{r.stdout}\n{r.stderr}"
    return json.loads(lines[-1])


def test_connect_read_and_the_model_libfranka_builds(client, sim):
    out = run(client, "info", "127.0.0.2")
    assert out["server_version"] == 10 and out["robot_mode"] == 1
    from roqsim_manipulation.franka.robot_model import RobotModel

    from roqsim.models import resolve_model

    robot = RobotModel(Path(resolve_model("fr3").path), np.array([0.0, 0.0, -9.81]))
    robot.update(np.array(out["q"]), np.zeros(7))
    assert np.allclose(out["gravity"], robot.gravity(), atol=1e-4)
    assert np.allclose(np.array(out["mass"]).reshape(7, 7).T, robot.mass(), atol=1e-4)
    assert np.allclose(np.array(out["flange"]).reshape(4, 4).T, robot.flange_pose(), atol=1e-5)
    J = np.array(out["zero_jacobian"]).reshape(7, 6).T
    assert np.allclose(J, robot.zero_jacobian(np.eye(4)), atol=1e-5)
    assert np.array(out["upper_velocity"]) == pytest.approx(
        __import__("roqsim_manipulation.franka.limits", fromlist=["x"]).upper_joint_velocity_limits(
            out["q"]
        ),
        abs=1e-5,
    )


def test_one_callback_is_one_millisecond_of_simulated_time(client, sim):
    out = run(client, "torque_hold", "127.0.0.2", 1000)
    assert out["ok"] and out["callbacks"] == 1000
    assert out["last_ms"] - out["first_ms"] == 999
    assert out["min_period"] == pytest.approx(0.001) and out["max_period"] == pytest.approx(0.001)
    assert out["robot_mode"] == 1


def test_libfrankas_joint_motion_example_runs(client, sim):
    out = run(client, "joint_motion", "127.0.0.2", 2.0)
    assert out["ok"]
    delta = np.pi / 8 * (1 - np.cos(np.pi / 2.5 * 2.0))
    expected = np.array(out["q0"]) + delta * np.array([0, 0, 0, 1, 1, 0, 1])
    assert np.allclose(out["q"], expected, atol=5e-3)


def test_a_jump_is_a_discontinuity_and_recovery_clears_it(client, sim):
    out = run(client, "joint_motion", "127.0.0.2", 1.0, 0.05)
    assert not out["ok"] and "joint_motion_generator_velocity_discontinuity" in out["error"]
    assert run(client, "recover", "127.0.0.2")["robot_mode"] == 1


def test_a_torque_step_is_refused_unless_libfranka_limits_its_rate(client, sim):
    out = run(client, "torque_hold", "127.0.0.2", 300, 60, 0, 3.0)
    assert not out["ok"] and "controller_torque_discontinuity" in out["error"]
    run(client, "recover", "127.0.0.2")
    out = run(client, "torque_hold", "127.0.0.2", 300, 60, 1, 3.0)
    assert out["ok"], out


def test_a_push_above_the_threshold_ends_the_motion_with_a_cartesian_reflex(client, sim):
    flange = sim.engine.ctx.model.body("left_fr3_link8").id

    def push(engine):
        if engine.ctx.sim_time > 0.3:
            engine.ctx.data.xfrc_applied[flange, :3] = [0.0, 0.0, -80.0]

    sim.hooks.append(push)
    out = run(client, "torque_hold", "127.0.0.2", 3000, 60)
    assert not out["ok"] and "cartesian_reflex" in out["error"]


def test_a_client_of_another_protocol_version_is_refused(sim):
    with socket.create_connection(("127.0.0.2", 1337), timeout=5) as s:
        s.sendall(
            HEADER.pack(Command.CONNECT, 0, HEADER.size + CONNECT_REQUEST.size)
            + CONNECT_REQUEST.pack(9, 50000)
        )
        data = b""
        while len(data) < HEADER.size + CONNECT_RESPONSE.size:
            data += s.recv(64)
    status, version = CONNECT_RESPONSE.unpack(data[HEADER.size :])
    assert status == 1 and version == 10  # kIncompatibleLibraryVersion, and the robot's own version


def test_two_arms_two_connections_one_clock(client):
    sim = Sim({"left": "127.0.0.2", "right": "127.0.0.3"})
    try:
        out = run(client, "dual", "127.0.0.2", "127.0.0.3")
    finally:
        sim.close()
    (lf, ll, ln), (rf, rl, rn) = out["left"], out["right"]
    assert ln == rn == 500
    assert ll - lf == rl - rf == 499  # each arm's callbacks are consecutive milliseconds
    assert abs(lf - rf) < 50  # and run over the same simulated time
