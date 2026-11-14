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
