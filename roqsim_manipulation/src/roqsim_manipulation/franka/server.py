"""The robot's side of the FCI on the network, in lock-step with the physics.

An unmodified libfranka (and pylibfranka, and franka_ros2 on top of it) connects to
``<bind>:1337`` as it connects to a robot. The TCP thread here reads commands and queues them; the
physics thread (the plugin that is the arm's controller) executes them at the next tick, answers them, sends the
robot state over UDP, and -- while a motion runs -- waits for the command answering that state
before it steps the physics again. One callback of the client is therefore one millisecond of
simulated time however slow the physics is, and a run is reproducible.

**The start of a motion.** libfranka's control loop reads one state after the motion has started
before it sends its first command (``src/control_loop.cpp``, ``loop()``), so the state that first
shows the motion running cannot be waited on. Until a motion's first command has arrived the server
waits only ``first_command_wait_s`` of wall time for one, then sends the next state; from the first
command on it waits for every one.

**Keeping the client alive.** libfranka gives up after one second of wall time without a state
(``src/network.h``, ``udp_timeout``). While the server waits for a command it re-sends the current
state every ``resend_s``; libfranka discards it as not newer, and keeps waiting.
"""

from __future__ import annotations

import logging
import queue
import select
import socket
import threading
import time
from dataclasses import dataclass

from .protocol import (
    COMMAND_PORT,
    CONNECT_REQUEST,
    CONNECT_RESPONSE,
    HEADER,
    REQUESTS,
    ROBOT_COMMAND,
    STATUS_RESPONSE,
    VERSION,
    Command,
    CommandStatus,
    ConnectStatus,
    unpack_robot_command,
)

log = logging.getLogger(__name__)


@dataclass
class Request:
    command: Command
    command_id: int
    payload: bytes


class FCIServer:
    """One arm's FCI: a TCP command port and a UDP state stream, bound to one address."""

    def __init__(
        self,
        bind: str,
        robot_model_urdf: str,
        *,
        port: int = COMMAND_PORT,
        resend_s: float = 0.1,
        first_command_wait_s: float = 0.02,
    ):
        self.bind = bind
        self.port = port
        self.urdf = robot_model_urdf
        self.resend_s = resend_s
        self.first_command_wait_s = first_command_wait_s
        self.requests: queue.Queue[Request] = queue.Queue()
        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp.bind((bind, port))
        self._tcp.listen(1)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.bind((bind, 0))
        self._conn: socket.socket | None = None
        # Guards the swap of the connected client: a closing connection's thread must not drop the
        # client that replaced it.
        self._conn_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self.client_udp: tuple[str, int] | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._accept, name=f"fci:{bind}", daemon=True)
        self._thread.start()

    # -- TCP ----------------------------------------------------------------------------------------

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, addr = self._tcp.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._conn_lock:
                if self._conn is not None:  # one client at a time, as the robot serves one
                    self._drop_client("a new client connected", self._conn)
                self._conn = conn
            threading.Thread(target=self._serve, args=(conn, addr[0]), daemon=True).start()

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _serve(self, conn: socket.socket, host: str) -> None:
        while not self._closed:
            head = self._recv_exact(conn, HEADER.size)
            if head is None:
                break
            command, command_id, size = HEADER.unpack(head)
            payload = self._recv_exact(conn, size - HEADER.size) if size > HEADER.size else b""
            if payload is None:
                break
            try:
                command = Command(command)
            except ValueError:
                log.warning("fci %s: unknown command %d", self.bind, command)
                continue
            if command == Command.CONNECT:
                if len(payload) != CONNECT_REQUEST.size:
                    log.warning(
                        "fci %s: CONNECT payload of %d bytes, expected %d",
                        self.bind,
                        len(payload),
                        CONNECT_REQUEST.size,
                    )
                    continue
                version, udp_port = CONNECT_REQUEST.unpack(payload)
                ok = version == VERSION
                if ok:  # streaming to the client before it hears the reply it may act on
                    self.client_udp = (host, udp_port)
                self.reply(
                    command,
                    command_id,
                    CONNECT_RESPONSE.pack(
                        ConnectStatus.SUCCESS if ok else ConnectStatus.INCOMPATIBLE_LIBRARY_VERSION,
                        VERSION,
                    ),
                )
                continue
            if command == Command.GET_ROBOT_MODEL:
                self.reply(
                    command,
                    command_id,
                    STATUS_RESPONSE.pack(CommandStatus.SUCCESS) + self.urdf.encode(),
                )
                continue
            expected = REQUESTS[command]
            if expected is not None and len(payload) != expected.size:
                log.warning(
                    "fci %s: %s payload of %d bytes, expected %d",
                    self.bind,
                    command.name,
                    len(payload),
                    expected.size,
                )
                continue
            self.requests.put(Request(command, command_id, payload))
        with self._conn_lock:
            self._drop_client("the client closed the connection", conn)

    def _drop_client(self, why: str, conn: socket.socket) -> None:
        """Close ``conn`` and forget the client, if ``conn`` is still the connected one; the caller
        holds ``_conn_lock``."""
        if self._conn is not conn:
            return
        log.info("fci %s: %s", self.bind, why)
        self._conn, self.client_udp = None, None
        try:
            conn.close()
        except OSError:
            pass
        self.requests.put(Request(Command.STOP_MOVE, -1, b""))  # a motion without a client ends

    def reply(self, command: Command, command_id: int, payload: bytes) -> None:
        conn = self._conn
        if conn is None or command_id < 0:
            return
        msg = HEADER.pack(int(command), command_id, HEADER.size + len(payload)) + payload
        with self._send_lock:
            try:
                conn.sendall(msg)
            except OSError:
                pass

    @property
    def connected(self) -> bool:
        return self.client_udp is not None

    # -- UDP ----------------------------------------------------------------------------------------

    def send_state(self, data: bytes) -> None:
        if self.client_udp is not None:
            try:
                self._udp.sendto(data, self.client_udp)
            except OSError:
                pass

    def await_command(
        self, message_id: int, state: bytes, *, strict: bool, client_timeout_s: float
    ) -> dict | None:
        """The ``RobotCommand`` answering ``message_id``, from the connected client's address.

        ``strict``: wait for it, re-sending ``state`` every ``resend_s``, and raise after
        ``client_timeout_s`` of wall time; otherwise wait at most ``first_command_wait_s``.

        ``None`` when none came within ``first_command_wait_s`` (not strict), or when the client
        disconnected meanwhile (either way). A disconnect ends the motion through the ``StopMove``
        it queues, and the run goes on, as a robot whose client went away does.
        """
        start = time.monotonic()
        last_send = start
        limit = client_timeout_s if strict else self.first_command_wait_s
        while self.client_udp is not None:
            now = time.monotonic()
            if now - start > limit:
                if strict:
                    raise TimeoutError(
                        f"FCI {self.bind}: no command for state {message_id} within "
                        f"{client_timeout_s:.0f} s of wall time; the client stopped answering"
                    )
                return None
            if strict and now - last_send > self.resend_s:
                self.send_state(state)
                last_send = now
            ready, _, _ = select.select([self._udp], [], [], min(0.005, self.resend_s))
            if not ready:
                continue
            data, sender = self._udp.recvfrom(4096)
            if sender != self.client_udp or len(data) != ROBOT_COMMAND.size:
                continue
            cmd = unpack_robot_command(data)
            if cmd["message_id"] == message_id:
                return cmd
        return None

    def close(self) -> None:
        """Release the port: a blocked ``accept`` and ``recv`` wake on shutdown, not on close."""
        self._closed = True
        for s in (self._conn, self._tcp, self._udp):
            if s is None:
                continue
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass
        self._thread.join(timeout=5)
