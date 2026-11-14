"""The simulated FCI's network side on its own: the handshake, the robot model and lock-step UDP.

A client here speaks the wire format of ``roqsim_manipulation.franka.protocol`` by hand, so the
server is tested without a controller or a client library behind it.
"""

from __future__ import annotations

import socket
import threading

import pytest
from roqsim_manipulation.franka import protocol as P
from roqsim_manipulation.franka.server import FCIServer

ADDRESS = "127.0.0.4"


@pytest.fixture
def server():
    s = FCIServer(ADDRESS, "<robot name='fr3'/>", resend_s=0.05, first_command_wait_s=0.05)
    yield s
    s.close()


def _connect(version: int, udp: socket.socket) -> tuple[socket.socket, tuple]:
    tcp = socket.create_connection((ADDRESS, P.COMMAND_PORT), timeout=5)
    payload = P.CONNECT_REQUEST.pack(version, udp.getsockname()[1])
    tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 0, P.HEADER.size + len(payload)) + payload)
    reply = b""
    while len(reply) < P.HEADER.size + P.CONNECT_RESPONSE.size:
        chunk = tcp.recv(64)
        assert chunk, "the server closed the connection before it answered Connect"
        reply += chunk
    return tcp, P.CONNECT_RESPONSE.unpack(reply[P.HEADER.size :])


def _udp() -> socket.socket:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(5)
    return udp


def test_the_robots_version_is_accepted_and_another_refused(server):
    udp = _udp()
    tcp, (status, version) = _connect(P.VERSION, udp)
    assert (status, version) == (P.ConnectStatus.SUCCESS, P.VERSION)
    tcp.close()
    tcp, (status, version) = _connect(P.VERSION - 1, udp)
    assert (status, version) == (P.ConnectStatus.INCOMPATIBLE_LIBRARY_VERSION, P.VERSION)
    tcp.close()


def test_get_robot_model_serves_the_urdf(server):
    tcp, _ = _connect(P.VERSION, _udp())
    tcp.sendall(P.HEADER.pack(P.Command.GET_ROBOT_MODEL, 1, P.HEADER.size))
    data = b""
    while b"/>" not in data:
        data += tcp.recv(256)
    command, command_id, size = P.HEADER.unpack(data[: P.HEADER.size])
    assert (command, command_id) == (P.Command.GET_ROBOT_MODEL, 1)
    assert data[P.HEADER.size] == P.CommandStatus.SUCCESS
    assert data[P.HEADER.size + 1 : size].decode() == "<robot name='fr3'/>"
    tcp.close()


def test_a_setter_reaches_the_physics_thread_queue(server):
    tcp, _ = _connect(P.VERSION, _udp())
    payload = P.SET_JOINT_IMPEDANCE_REQUEST.pack(*([100.0] * 7))
    tcp.sendall(
        P.HEADER.pack(P.Command.SET_JOINT_IMPEDANCE, 7, P.HEADER.size + len(payload)) + payload
    )
    request = server.requests.get(timeout=5)
    assert (request.command, request.command_id) == (P.Command.SET_JOINT_IMPEDANCE, 7)
    assert P.SET_JOINT_IMPEDANCE_REQUEST.unpack(request.payload) == (100.0,) * 7
    tcp.close()


def _state(message_id: int) -> bytes:
    state = {name: (0 if n == 1 else [0] * n) for name, _, n in P.ROBOT_STATE_FIELDS}
    state["message_id"] = message_id
    return P.pack_robot_state(state)


def test_lock_step_waits_for_the_answer_to_its_state_and_resends_meanwhile(server):
    udp = _udp()
    tcp, _ = _connect(P.VERSION, udp)
    server.send_state(_state(5))
    seen = []

    def client():
        # the first copy arrives at once, a resend after resend_s; then the answer
        for _ in range(2):
            data, addr = udp.recvfrom(4096)
            seen.append(P.unpack_robot_state(data)["message_id"])
        udp.sendto(P.pack_robot_command({"message_id": 4, "tau_J_d": [9] * 7}), addr)  # stale
        udp.sendto(P.pack_robot_command({"message_id": 5, "tau_J_d": [1] * 7}), addr)

    t = threading.Thread(target=client)
    t.start()
    cmd = server.await_command(5, _state(5), strict=True, client_timeout_s=5)
    t.join()
    assert seen == [5, 5]
    assert cmd["message_id"] == 5 and cmd["tau_J_d"] == [1.0] * 7
    tcp.close()


def test_a_client_that_never_answers_stops_the_run(server):
    tcp, _ = _connect(P.VERSION, _udp())
    with pytest.raises(TimeoutError, match="stopped answering"):
        server.await_command(1, _state(1), strict=True, client_timeout_s=0.3)
    assert server.await_command(2, _state(2), strict=False, client_timeout_s=0.3) is None
    tcp.close()


def test_a_malformed_connect_is_dropped_and_the_connection_still_served(server):
    """A wrong-size Connect is logged and skipped, as any wrong-size command: it must not end the
    connection's thread and leave the server holding a client it no longer reads."""
    udp = _udp()
    tcp = socket.create_connection((ADDRESS, P.COMMAND_PORT), timeout=5)
    tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 0, P.HEADER.size + 1) + b"\0")
    payload = P.CONNECT_REQUEST.pack(P.VERSION, udp.getsockname()[1])
    tcp.sendall(P.HEADER.pack(P.Command.CONNECT, 1, P.HEADER.size + len(payload)) + payload)
    reply = b""
    while len(reply) < P.HEADER.size + P.CONNECT_RESPONSE.size:
        chunk = tcp.recv(64)
        assert chunk, "the server closed the connection after a malformed Connect"
        reply += chunk
    assert P.HEADER.unpack(reply[: P.HEADER.size])[1] == 1
    assert P.CONNECT_RESPONSE.unpack(reply[P.HEADER.size :])[0] == P.ConnectStatus.SUCCESS
    tcp.close()


def test_a_closing_old_client_leaves_its_successor_connected(server):
    """One client at a time: a new connection replaces the old, and the old one's closing (its
    thread noticing later) does not drop the new client."""
    old, _ = _connect(P.VERSION, _udp())
    new_udp = _udp()
    new, (status, _) = _connect(P.VERSION, new_udp)
    assert status == P.ConnectStatus.SUCCESS
    old.close()
    new.sendall(P.HEADER.pack(P.Command.GET_ROBOT_MODEL, 2, P.HEADER.size))
    data = b""
    while b"/>" not in data:
        chunk = new.recv(256)
        assert chunk, "the new client was dropped"
        data += chunk
    assert server.client_udp == ("127.0.0.1", new_udp.getsockname()[1])
    new.close()


def test_only_the_connected_clients_commands_are_taken(server):
    udp = _udp()
    tcp, _ = _connect(P.VERSION, udp)
    stranger = _udp()
    stranger.sendto(
        P.pack_robot_command({"message_id": 3, "tau_J_d": [9] * 7}), server._udp.getsockname()
    )
    assert server.await_command(3, _state(3), strict=False, client_timeout_s=1) is None
    udp.sendto(
        P.pack_robot_command({"message_id": 3, "tau_J_d": [1] * 7}), server._udp.getsockname()
    )
    assert (
        server.await_command(3, _state(3), strict=False, client_timeout_s=1)["tau_J_d"] == [1.0] * 7
    )
    tcp.close()


def test_a_client_gone_during_a_strict_wait_ends_the_wait(server):
    """The motion ends through the StopMove the disconnect queues; the run goes on."""
    tcp, _ = _connect(P.VERSION, _udp())
    tcp.close()
    assert server.await_command(1, _state(1), strict=True, client_timeout_s=5) is None
    assert server.requests.get(timeout=5).command == P.Command.STOP_MOVE
