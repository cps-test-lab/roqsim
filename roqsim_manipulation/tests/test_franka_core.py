"""The simulated FR3's controller, in process: gravity, the estimate, the reflexes, the 1 kHz tick.

The estimate is checked against forces applied to the arm, and a wrong configured load against the
bias it must cause (``J^T dm g``): the estimate is built from the link-side joint torques and the
robot's model, as on the robot, so both show up exactly as the robot would show them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from roqsim_manipulation.franka.protocol import MoveStatus

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

CB = {
    "lower_torque": [20.0] * 7,
    "upper_torque": [40.0] * 7,
    "lower_force": [10.0] * 6,
    "upper_force": [25.0] * 6,
}


def _engine(timestep=0.001, **fci):
    cfg = {"collision_behavior": CB, **fci}
    world = {
        "sim": {"timestep": timestep},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": cfg}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")), preview=True)
    engine.setup()
    engine.reset()
    return engine


def _steps(engine, n):
    for _ in range(n):
        engine.step()


def _handle(engine):
    return engine.ctx.blackboard.get("franka_fci:left")


def _flange(engine):
    return engine.ctx.model.body("left_fr3_link8").id


def test_the_arm_holds_its_pose_on_gravity_compensation():
    engine = _engine()
    q0 = engine.ctx.data.qpos[:7].copy()
    _steps(engine, 1000)
    s = _handle(engine).state()
    assert np.max(np.abs(np.array(s["q"]) - q0)) < 1e-4
    assert np.max(np.abs(s["O_F_ext_hat_K"])) < 0.05
    engine.shutdown()


@pytest.mark.parametrize("force", [[0, 0, -20.0], [15.0, 0, 0], [0, -10.0, 5.0]])
def test_the_estimate_reads_a_force_on_the_flange_as_the_robot_reports_it(force):
    """The environment pushes the flange with F; the robot reports what it applies: -F."""
    engine = _engine()
    engine.ctx.data.xfrc_applied[_flange(engine), :3] = force
    _steps(engine, 600)
    s = _handle(engine).state()
    assert np.allclose(
        s["O_F_ext_hat_K"][:3], -np.asarray(force), atol=0.02 * max(np.abs(force)) + 0.05
    )
    engine.shutdown()


def test_a_wrong_configured_load_is_the_bias_it_causes():
    """Told 0.2 kg that is not there: the arm reads a constant wrench of that weight at the flange."""
    engine = _engine(load={"mass": 0.2, "com": [0, 0, 0], "inertia": [0] * 9})
    _steps(engine, 800)
    s = _handle(engine).state()
    F = np.array(s["O_F_ext_hat_K"][:3])
    # The model expects 0.2 kg more weight than the arm carries: it reads as if the environment
    # lifted the flange by 0.2 g, so the robot's force on the environment is 0.2 g downward.
    assert F == pytest.approx([0.0, 0.0, -0.2 * 9.81], abs=0.03)
    engine.shutdown()


def test_the_true_load_configured_reads_zero():
    engine = _engine(
        load={"mass": 0.5, "com": [0, 0, 0.05], "inertia": [1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3]}
    )
    flange = _flange(engine)
    # the same 0.5 kg as a real weight at the configured centre of mass
    engine.ctx.data.xfrc_applied[flange, :3] = [0, 0, -0.5 * 9.81]
    _steps(engine, 800)
    s = _handle(engine).state()
    assert (
        np.max(np.abs(s["O_F_ext_hat_K"][:3])) < 0.25
    )  # the weight's lever arm differs by 5 cm only
    engine.shutdown()


def _hold_law(seconds, tau=None):
    ticks = {"n": 0}

    def law(state, period):
        ticks["n"] += 1
        out = {"tau_J_d": list(tau if tau is not None else np.zeros(7))}
        if ticks["n"] >= seconds * 1000:
            out["torque_command_finished"] = True
        return out

    return law, ticks


def test_in_process_torque_control_ends_with_success():
    engine = _engine()
    law, ticks = _hold_law(0.2)
    motion = _handle(engine).control(law)
    _steps(engine, 250)
    assert motion.done and motion.ending.status == MoveStatus.SUCCESS
    assert ticks["n"] == 200
    engine.shutdown()


def test_a_push_above_the_threshold_is_a_cartesian_reflex():
    engine = _engine()
    law, _ = _hold_law(2.0)
    motion = _handle(engine).control(law)
    engine.ctx.data.xfrc_applied[_flange(engine), :3] = [0, 0, -40.0]
    _steps(engine, 400)
    assert motion.done and motion.ending.status == MoveStatus.REFLEX_ABORTED
    assert "cartesian_reflex" in motion.ending.errors
    s = _handle(engine).state()
    assert s["robot_mode"] == 4 and any(s["errors"])
    engine.ctx.data.xfrc_applied[_flange(engine), :3] = 0
    _steps(engine, 10)
    assert _handle(engine).recover()
    _steps(engine, 2)
    assert _handle(engine).state()["robot_mode"] == 1 and not any(_handle(engine).state()["errors"])
    engine.shutdown()


def test_a_push_below_the_threshold_is_a_contact_not_a_reflex():
    engine = _engine()
    law, _ = _hold_law(0.4)
    motion = _handle(engine).control(law)
    engine.ctx.data.xfrc_applied[_flange(engine), :3] = [0, 0, -15.0]
    _steps(engine, 300)
    s = _handle(engine).state()
    assert (
        not motion.done and s["cartesian_contact"][2] == 1.0 and s["cartesian_collision"][2] == 0.0
    )
    engine.shutdown()


def test_a_torque_jump_is_a_controller_torque_discontinuity():
    engine = _engine()
    law, _ = _hold_law(1.0, tau=[0, 0, 0, 0, 0, 0, 5.0])
    motion = _handle(engine).control(
        lambda s, p: law(s, p) if s["message_id"] > 3 else {"tau_J_d": [0] * 7}
    )
    _steps(engine, 20)
    assert motion.done and "controller_torque_discontinuity" in motion.ending.errors
    engine.shutdown()


def test_a_timestep_that_does_not_divide_a_millisecond_is_refused():
    with pytest.raises(Exception, match="1 ms"):
        _engine(timestep=0.0007)


def test_at_a_third_of_a_millisecond_one_tick_is_three_steps():
    engine = _engine(timestep=1e-3 / 3)
    law, ticks = _hold_law(1.0)
    _handle(engine).control(law)
    _steps(engine, 300)
    assert ticks["n"] == 100
    assert engine.ctx.sim_time == pytest.approx(0.1, abs=1e-9)
    engine.shutdown()


def test_collision_behavior_is_required():
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {}}],
            }
        ],
    }
    with pytest.raises(Exception, match="collision_behavior"):
        Engine(load_config_from_dict(world, base_dir=Path(".")), preview=True).setup()


def test_a_run_is_reproducible():
    def run():
        engine = _engine(tau_J_noise=0.05)
        _handle(engine).control(
            lambda s, p: {"tau_J_d": [0, min(0.5, s["message_id"] * 0.0005), 0, 0, 0, 0, 0]}
        )
        engine.ctx.data.xfrc_applied[_flange(engine), :3] = [0, 0, -5.0]
        _steps(engine, 300)
        s = _handle(engine).state()
        engine.shutdown()
        return s["q"], s["tau_ext_hat_filtered"]

    assert run() == run()


def test_a_preview_serves_no_fci():
    """A render or an export builds the world to look at it, and takes no address from a run."""
    engine = _engine(bind="127.0.0.9")
    assert not any(getattr(p, "server", None) for p in engine.plugins)
    engine.shutdown()


def _read_reply(tcp, header, response, engine=None):
    """One reply off the command connection. With ``engine``, step it while waiting: the server's
    thread queues a command for the physics thread, which answers it at a later tick."""
    data = b""
    tcp.settimeout(0.01 if engine is not None else 5)
    for _ in range(5000):
        if len(data) >= header.size + response.size:
            break
        try:
            chunk = tcp.recv(256)
        except TimeoutError:
            if engine is None:
                raise
            engine.step()
            continue
        assert chunk, "the server closed the connection"
        data += chunk
    assert len(data) >= header.size + response.size, "no reply within 5000 steps"
    return header.unpack(data[: header.size]), response.unpack(
        data[header.size : header.size + response.size]
    )


def test_a_reset_answers_a_running_network_motion_as_aborted():
    """libfranka blocks on a Move's outcome (throwOnMotionError); a reset that ends the motion must
    send it, or the client waits forever."""
    import socket

    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.6"
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {"collision_behavior": CB, "bind": address}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    tcp = socket.create_connection((address, P.COMMAND_PORT), timeout=5)
    try:
        connect = P.CONNECT_REQUEST.pack(P.VERSION, udp.getsockname()[1])
        tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 1, P.HEADER.size + len(connect)) + connect)
        _, (status, _) = _read_reply(tcp, P.HEADER, P.CONNECT_RESPONSE)
        assert status == P.ConnectStatus.SUCCESS
        move = P.MOVE_REQUEST.pack(
            P.MoveControllerMode.JOINT_IMPEDANCE,
            P.MoveMotionGeneratorMode.JOINT_POSITION,
            *([10.0] * 6),
            False,
            *([2.0] * 7),
        )
        tcp.sendall(P.HEADER.pack(P.Command.MOVE, 2, P.HEADER.size + len(move)) + move)
        (command, command_id, _), (status,) = _read_reply(tcp, P.HEADER, P.STATUS_RESPONSE, engine)
        assert (command, command_id, status) == (P.Command.MOVE, 2, P.MoveStatus.MOTION_STARTED)
        _steps(engine, 3)
        engine.reset()
        (command, command_id, _), (status,) = _read_reply(tcp, P.HEADER, P.STATUS_RESPONSE)
        assert (command, command_id, status) == (P.Command.MOVE, 2, P.MoveStatus.ABORTED)
    finally:
        tcp.close()
        udp.close()
        engine.shutdown()


def test_a_setting_still_queued_at_a_reset_is_refused_and_does_not_reach_the_next_run():
    """A client's setLoad the physics has not yet taken when the world resets: the reset answers
    it as refused, and the next run carries the world's configured load."""
    import socket
    import time

    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.8"
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {"collision_behavior": CB, "bind": address}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    fci = _handle(engine)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    tcp = socket.create_connection((address, P.COMMAND_PORT), timeout=5)
    try:
        connect = P.CONNECT_REQUEST.pack(P.VERSION, udp.getsockname()[1])
        tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 1, P.HEADER.size + len(connect)) + connect)
        _, (status, _) = _read_reply(tcp, P.HEADER, P.CONNECT_RESPONSE)
        assert status == P.ConnectStatus.SUCCESS
        load = P.SET_LOAD_REQUEST.pack(2.0, 0.0, 0.0, 0.1, *np.eye(3).reshape(9) * 1e-3)
        tcp.sendall(P.HEADER.pack(P.Command.SET_LOAD, 2, P.HEADER.size + len(load)) + load)
        (plugin,) = [p for p in engine.plugins if type(p).__name__ == "FrankaFCIPlugin"]
        deadline = time.monotonic() + 5
        while plugin.server.requests.empty():  # the server's thread queues it; no step takes it
            assert time.monotonic() < deadline, "the server never queued the request"
            time.sleep(0.001)
        engine.reset()
        (command, command_id, _), (status,) = _read_reply(tcp, P.HEADER, P.STATUS_RESPONSE)
        assert (command, command_id) == (P.Command.SET_LOAD, 2)
        assert status == P.GetterSetterStatus.COMMAND_NOT_POSSIBLE_REJECTED
        _steps(engine, 3)
        assert fci.core.m_load == 0.0
    finally:
        tcp.close()
        udp.close()
        engine.shutdown()


def _connect(address, P):
    import socket

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    tcp = socket.create_connection((address, P.COMMAND_PORT), timeout=5)
    connect = P.CONNECT_REQUEST.pack(P.VERSION, udp.getsockname()[1])
    tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 1, P.HEADER.size + len(connect)) + connect)
    _, (status, _) = _read_reply(tcp, P.HEADER, P.CONNECT_RESPONSE)
    assert status == P.ConnectStatus.SUCCESS
    return tcp, udp


def test_a_dropped_clients_queued_setting_neither_applies_nor_answers_the_next_client():
    """libfranka numbers its commands from 0 on every connection: a setLoad a client queued
    before it dropped must not act for the client that connects next, nor answer an id that
    client uses itself."""
    import time

    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.11"
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {"collision_behavior": CB, "bind": address}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    fci = _handle(engine)
    (plugin,) = [p for p in engine.plugins if type(p).__name__ == "FrankaFCIPlugin"]
    a_tcp, a_udp = _connect(address, P)
    load = P.SET_LOAD_REQUEST.pack(2.0, 0.0, 0.0, 0.1, *np.eye(3).reshape(9) * 1e-3)
    a_tcp.sendall(P.HEADER.pack(P.Command.SET_LOAD, 2, P.HEADER.size + len(load)) + load)
    deadline = time.monotonic() + 5
    while plugin.server.requests.empty():  # the server's thread queues it; no step takes it
        assert time.monotonic() < deadline, "the server never queued the request"
        time.sleep(0.001)
    a_tcp.close()
    a_udp.close()
    b_tcp, b_udp = _connect(address, P)
    try:
        b_tcp.sendall(P.HEADER.pack(P.Command.AUTOMATIC_ERROR_RECOVERY, 2, P.HEADER.size))
        (command, command_id, _), _ = _read_reply(b_tcp, P.HEADER, P.STATUS_RESPONSE, engine)
        assert (command, command_id) == (P.Command.AUTOMATIC_ERROR_RECOVERY, 2)
        assert fci.core.m_load == 0.0
    finally:
        b_tcp.close()
        b_udp.close()
        engine.shutdown()


def _messages(tcp, P, n, engine, tries=300):
    """Up to *n* messages off the command connection, stepping *engine* while none is there."""
    out, data = [], b""
    tcp.settimeout(0.01)
    for _ in range(tries):
        while len(data) >= P.HEADER.size:
            command, command_id, size = P.HEADER.unpack(data[: P.HEADER.size])
            if len(data) < size:
                break
            out.append((P.Command(command), command_id, data[P.HEADER.size : size]))
            data = data[size:]
        if len(out) >= n:
            break
        try:
            chunk = tcp.recv(4096)
        except TimeoutError:
            engine.step()
            continue
        if not chunk:
            break
        data += chunk
    return out


def test_a_dropped_clients_move_outcome_does_not_reach_the_next_client():
    """A client's Move runs and the client drops; the next one connects and sends a Move under
    the same id (libfranka numbers from 0 again) before the physics takes the stop. The next
    client hears only its own Move, and its motion is the one the FCI serves."""
    import time

    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.10"
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {"collision_behavior": CB, "bind": address}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    (plugin,) = [p for p in engine.plugins if type(p).__name__ == "FrankaFCIPlugin"]
    move = P.MOVE_REQUEST.pack(
        P.MoveControllerMode.JOINT_IMPEDANCE,
        P.MoveMotionGeneratorMode.JOINT_POSITION,
        *([10.0] * 6),
        False,
        *([2.0] * 7),
    )
    a_tcp, a_udp = _connect(address, P)
    a_tcp.sendall(P.HEADER.pack(P.Command.MOVE, 2, P.HEADER.size + len(move)) + move)
    (command, command_id, _), (status,) = _read_reply(a_tcp, P.HEADER, P.STATUS_RESPONSE, engine)
    assert (command, command_id, status) == (P.Command.MOVE, 2, P.MoveStatus.MOTION_STARTED)
    _steps(engine, 3)
    a_tcp.close()
    a_udp.close()
    deadline = time.monotonic() + 5
    while plugin.server.requests.empty():  # the server's own StopMove for the dropped client
        assert time.monotonic() < deadline, "the server never noticed the drop"
        time.sleep(0.001)
    b_tcp, b_udp = _connect(address, P)
    try:
        b_tcp.sendall(P.HEADER.pack(P.Command.MOVE, 2, P.HEADER.size + len(move)) + move)
        deadline = time.monotonic() + 5
        while plugin.server.requests.qsize() < 2:
            assert time.monotonic() < deadline, "the server never queued the Move"
            time.sleep(0.001)
        heard = _messages(b_tcp, P, 2, engine, tries=30)
        statuses = [P.MoveStatus(P.STATUS_RESPONSE.unpack(p[:1])[0]) for _, _, p in heard]
        assert statuses == [P.MoveStatus.MOTION_STARTED], statuses
        assert plugin.core.motion is not None and plugin.core.motion.owner is plugin.server._conn
    finally:
        b_tcp.close()
        b_udp.close()
        engine.shutdown()


def test_a_setting_that_is_not_finite_is_refused_and_the_run_goes_on():
    """NaN in a client's setting, or a negative load mass, is refused as an invalid argument:
    taken, it would stop the physics (a load's inertia, a frame) or make MuJoCo reset its data
    unseen (an impedance)."""
    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.12"
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {"collision_behavior": CB, "bind": address}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    core = _handle(engine).core
    K_theta, K_x = core.K_theta.copy(), core.K_x.copy()
    nan = float("nan")
    eye = list(np.eye(4).T.reshape(16))
    requests = [
        (P.Command.SET_LOAD, P.SET_LOAD_REQUEST.pack(1.0, 0.0, 0.0, 0.1, *([nan] * 9))),
        (P.Command.SET_LOAD, P.SET_LOAD_REQUEST.pack(nan, 0.0, 0.0, 0.1, *([0.0] * 9))),
        (P.Command.SET_LOAD, P.SET_LOAD_REQUEST.pack(-1.0, 0.0, 0.0, 0.1, *([0.0] * 9))),
        (P.Command.SET_NE_TO_EE, P.SET_NE_TO_EE_REQUEST.pack(*([nan] + eye[1:]))),
        (P.Command.SET_EE_TO_K, P.SET_EE_TO_K_REQUEST.pack(*([nan] + eye[1:]))),
        (P.Command.SET_JOINT_IMPEDANCE, P.SET_JOINT_IMPEDANCE_REQUEST.pack(*([nan] * 7))),
        (P.Command.SET_CARTESIAN_IMPEDANCE, P.SET_CARTESIAN_IMPEDANCE_REQUEST.pack(*([nan] * 6))),
    ]
    tcp, udp = _connect(address, P)
    try:
        for command_id, (command, payload) in enumerate(requests, start=2):
            tcp.sendall(P.HEADER.pack(command, command_id, P.HEADER.size + len(payload)) + payload)
            (c, i, _), (status,) = _read_reply(tcp, P.HEADER, P.STATUS_RESPONSE, engine)
            assert (c, i) == (command, command_id)
            assert status == P.GetterSetterStatus.INVALID_ARGUMENT_REJECTED, command
        _steps(engine, 5)
        assert core.m_load == 0.0 and np.array_equal(core.NE_T_EE, np.eye(4))
        assert np.array_equal(core.EE_T_K, np.eye(4))
        assert np.array_equal(core.K_theta, K_theta) and np.array_equal(core.K_x, K_x)
    finally:
        tcp.close()
        udp.close()
        engine.shutdown()


def test_a_client_that_cancels_its_motion_has_it_preempted_and_the_run_goes_on():
    """libfranka cancels a motion whose control callback threw: it sends StopMove, answers no
    further state, and waits for the reply. The strict wait for its command ends on that
    StopMove, the motion is preempted, and the run goes on instead of timing out."""
    import socket
    import threading
    import time

    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.13"
    cfg = {"collision_behavior": CB, "bind": address, "client_timeout_s": 2.0}
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": cfg}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    heard: list = []

    def client():
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("127.0.0.1", 0))
        udp.settimeout(5)
        tcp = socket.create_connection((address, P.COMMAND_PORT), timeout=10)
        try:
            c = P.CONNECT_REQUEST.pack(P.VERSION, udp.getsockname()[1])
            tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 1, P.HEADER.size + len(c)) + c)
            _read_reply(tcp, P.HEADER, P.CONNECT_RESPONSE)
            move = P.MOVE_REQUEST.pack(
                P.MoveControllerMode.JOINT_IMPEDANCE,
                P.MoveMotionGeneratorMode.JOINT_POSITION,
                *([10.0] * 6),
                False,
                *([2.0] * 7),
            )
            tcp.sendall(P.HEADER.pack(P.Command.MOVE, 2, P.HEADER.size + len(move)) + move)
            _read_reply(tcp, P.HEADER, P.STATUS_RESPONSE)  # motion started
            answered = 0
            while answered < 20:
                data, server = udp.recvfrom(65536)
                state = P.unpack_robot_state(data)
                if state["robot_mode"] != P.RobotMode.MOVE:
                    continue
                command = {"message_id": state["message_id"], "q_c": state["q_d"]}
                udp.sendto(P.pack_robot_command(command), server)
                answered += 1
            while P.unpack_robot_state(udp.recvfrom(65536)[0])["robot_mode"] != P.RobotMode.MOVE:
                pass  # the next state: the callback throws on it instead of answering
            tcp.sendall(P.HEADER.pack(P.Command.STOP_MOVE, 3, P.HEADER.size))
            tcp.settimeout(5)
            data = b""
            deadline = time.monotonic() + 5
            while len(heard) < 2 and time.monotonic() < deadline:
                data += tcp.recv(256)
                while len(data) >= P.HEADER.size + 1:
                    command, command_id, size = P.HEADER.unpack(data[: P.HEADER.size])
                    heard.append((P.Command(command), command_id, data[P.HEADER.size]))
                    data = data[size:]
        finally:
            tcp.close()
            udp.close()

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        while thread.is_alive() and time.monotonic() - start < 15:
            engine.step()
    finally:
        thread.join(5)
        engine.shutdown()
    assert (P.Command.STOP_MOVE, 3, P.StopMoveStatus.SUCCESS) in heard, heard
    assert (P.Command.MOVE, 2, P.MoveStatus.PREEMPTED) in heard, heard


def test_a_command_that_is_not_finite_ends_the_motion_and_the_arm_holds():
    """Every limit check is a comparison, which NaN passes; such a command ends the motion as
    aborted, naming the field, instead of reaching the actuators."""
    engine = _engine()
    handle = _handle(engine)
    q0 = np.array(handle.state()["q"])
    ticks = {"n": 0}

    def law(state, period):
        ticks["n"] += 1
        q = list(state["q_d"])
        if ticks["n"] == 5:
            q[3] = float("nan")
        return {"q_c": q}

    motion = handle.control(law, controller="joint_impedance", generator="joint_position")
    _steps(engine, 200)
    assert motion.done and motion.ending.status == MoveStatus.ABORTED, motion.ending
    assert any("q_c" in e for e in motion.ending.errors), motion.ending.errors
    assert np.all(np.isfinite(engine.ctx.data.ctrl))
    assert np.abs(np.array(handle.state()["q"]) - q0).max() < 2e-3
    engine.shutdown()


def test_the_handle_takes_no_settings_while_a_motion_runs():
    """As the robot refuses a setting over the network while it moves, the handle raises."""
    engine = _engine()
    handle = _handle(engine)
    law, _ = _hold_law(0.1)
    handle.control(law)
    _steps(engine, 2)
    with pytest.raises(RuntimeError, match="no settings while it moves"):
        handle.set_load(1.0, [0, 0, 0], np.eye(3) * 1e-3)
    _steps(engine, 150)
    handle.set_load(1.0, [0, 0, 0], np.eye(3) * 1e-3)  # idle again: taken
    assert handle.core.m_load == 1.0
    engine.shutdown()


def test_a_move_in_a_mode_the_protocol_does_not_define_is_refused():
    """As the robot refuses it, rather than raising on the physics thread and ending the run."""
    import socket

    from roqsim_manipulation.franka import protocol as P

    address = "127.0.0.7"
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "fr3", "prefix": "left_"},
                "name": "left",
                "components": [{"franka_fci": {"collision_behavior": CB, "bind": address}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    tcp = socket.create_connection((address, P.COMMAND_PORT), timeout=5)
    try:
        connect = P.CONNECT_REQUEST.pack(P.VERSION, udp.getsockname()[1])
        tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 1, P.HEADER.size + len(connect)) + connect)
        _read_reply(tcp, P.HEADER, P.CONNECT_RESPONSE)
        move = P.MOVE_REQUEST.pack(99, 0, *([10.0] * 6), False, *([2.0] * 7))
        tcp.sendall(P.HEADER.pack(P.Command.MOVE, 3, P.HEADER.size + len(move)) + move)
        (command, command_id, _), (status,) = _read_reply(tcp, P.HEADER, P.STATUS_RESPONSE, engine)
        assert (command, command_id, status) == (
            P.Command.MOVE,
            3,
            P.MoveStatus.INVALID_ARGUMENT_REJECTED,
        )
    finally:
        tcp.close()
        udp.close()
        engine.shutdown()


def _ramp(t, duration=1.0):
    """0 to 1 over *duration*, at rest at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * min(t, duration) / duration))


def _cartesian_position(s0, t):
    T = list(s0["O_T_EE"])
    T[12] += 0.05 * _ramp(t)  # O_T_EE is column-major: x of the translation
    return {"O_T_EE_c": T}


def _cartesian_velocity(s0, t):
    return {"O_dP_EE_c": [0.05 * np.pi / 2 * np.sin(np.pi * min(t, 1.0)), 0, 0, 0, 0, 0]}


def _joint_position(s0, t):
    q = list(s0["q"])
    q[3] += 0.2 * _ramp(t)
    return {"q_c": q}


def _joint_velocity(s0, t):
    return {"dq_c": [0, 0, 0, 0.2 * np.pi / 2 * np.sin(np.pi * min(t, 1.0)), 0, 0, 0]}


@pytest.mark.parametrize(
    ("controller", "generator", "command", "moves"),
    [
        ("cartesian_impedance", "cartesian_position", _cartesian_position, "x"),
        ("joint_impedance", "cartesian_position", _cartesian_position, "x"),
        ("cartesian_impedance", "cartesian_velocity", _cartesian_velocity, "x"),
        ("joint_impedance", "cartesian_velocity", _cartesian_velocity, "x"),
        ("joint_impedance", "joint_position", _joint_position, "q4"),
        ("cartesian_impedance", "joint_position", _joint_position, "q4"),
        ("joint_impedance", "joint_velocity", _joint_velocity, "q4"),
    ],
)
def test_each_generator_under_each_controller_moves_the_arm_as_commanded(
    controller, generator, command, moves
):
    """A 50 mm step along x of the end effector, or 0.2 rad of joint 4, over one second, ends
    with success where it was commanded; the internal impedance leaves a few millimetres."""
    engine = _engine()
    handle = _handle(engine)
    s0 = handle.state()
    ticks = {"n": 0}

    def law(state, period):
        ticks["n"] += 1
        out = command(s0, ticks["n"] * 1e-3)
        if ticks["n"] >= 1000:
            out["motion_generation_finished"] = True
        return out

    motion = handle.control(law, controller=controller, generator=generator)
    _steps(engine, 1100)
    s = handle.state()
    engine.shutdown()
    assert motion.done and motion.ending.status == MoveStatus.SUCCESS, motion.ending
    dp = np.array(s["O_T_EE"])[12:15] - np.array(s0["O_T_EE"])[12:15]
    dq = np.array(s["q"]) - np.array(s0["q"])
    if moves == "x":
        assert dp[0] == pytest.approx(0.05, abs=0.006)
        assert np.all(np.abs(dp[1:]) < 0.002)
    else:
        assert dq[3] == pytest.approx(0.2, abs=0.015)
        assert np.all(np.abs(np.delete(dq, 3)) < 0.015)


def test_an_external_controller_ignores_its_generators_motion():
    """Torque control with a joint-position generator: the commanded torque moves the arm, and
    the generator's q_c does not."""
    engine = _engine()
    handle = _handle(engine)
    s0 = handle.state()
    ticks = {"n": 0}

    def law(state, period):
        ticks["n"] += 1
        out = {**_joint_position(s0, ticks["n"] * 1e-3), "tau_J_d": [0.0] * 7}
        if ticks["n"] >= 500:
            out["motion_generation_finished"] = True
        return out

    motion = handle.control(law, controller="external", generator="joint_position")
    _steps(engine, 600)
    s = handle.state()
    engine.shutdown()
    assert motion.done and motion.ending.status == MoveStatus.SUCCESS, motion.ending
    assert np.all(np.abs(np.array(s["q"]) - np.array(s0["q"])) < 1e-3)


def test_every_error_flag_is_raised_or_named_as_not_raised():
    """``NOT_RAISED`` is what the docs say the simulated robot leaves out: every other flag of
    libfranka's must be named where the core or the plugin raises it."""
    import ast
    import inspect

    from roqsim_manipulation.franka import core
    from roqsim_manipulation.franka.protocol import ERRORS
    from roqsim_manipulation.plugins import franka_fci

    named = set()
    for module in (core, franka_fci):
        tree = ast.parse(inspect.getsource(module))
        skip = {
            id(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "NOT_RAISED" for t in n.targets)
        }
        stack = [tree]
        while stack:
            node = stack.pop()
            if id(node) in skip:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                named.add(node.value)
            stack.extend(ast.iter_child_nodes(node))
    not_raised = set(core.NOT_RAISED)
    assert not_raised <= set(ERRORS), sorted(not_raised - set(ERRORS))
    assert len(not_raised) == len(core.NOT_RAISED), "a flag is named twice"
    assert not (not_raised & named), f"raised and named as not raised: {sorted(not_raised & named)}"
    missing = [e for e in ERRORS if e not in not_raised and e not in named]
    assert not missing, f"neither raised nor named as not raised: {missing}"


def test_a_load_of_no_mass_is_taken_and_a_negative_one_refused():
    engine = _engine()
    handle = _handle(engine)
    assert handle.set_load(0.0, [0, 0, 0], np.zeros((3, 3)))
    assert not handle.set_load(-0.5, [0, 0, 0], np.zeros((3, 3)))
    assert handle.core.m_load == 0.0
    engine.shutdown()
    with pytest.raises(ValueError, match="cannot be negative"):
        _engine(load={"mass": -1.0})
