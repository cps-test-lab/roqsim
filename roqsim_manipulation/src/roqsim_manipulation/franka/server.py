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
    #: The connection that sent it; None for the server's own StopMove when a client drops.
    conn: socket.socket | None = None


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
        # The last state streamed, and when: kept alive while the physics does not step.
        self._last_state: bytes | None = None
        self._last_sent = 0.0
        self._thread = threading.Thread(target=self._accept, name=f"fci:{bind}", daemon=True)
        self._thread.start()
        self._keeper = threading.Thread(
            target=self._keep_alive, name=f"fci:{bind}:keep-alive", daemon=True
        )
        self._keeper.start()

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
                    with self._conn_lock:  # a replaced client's late Connect redirects nothing
                        if conn is self._conn:
                            self.client_udp = (host, udp_port)
                self.reply(
                    command,
                    command_id,
                    CONNECT_RESPONSE.pack(
                        ConnectStatus.SUCCESS if ok else ConnectStatus.INCOMPATIBLE_LIBRARY_VERSION,
                        VERSION,
                    ),
                    to=conn,
                )
                continue
            if command == Command.GET_ROBOT_MODEL:
                self.reply(
                    command,
                    command_id,
                    STATUS_RESPONSE.pack(CommandStatus.SUCCESS) + self.urdf.encode(),
                    to=conn,
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
            self.requests.put(Request(command, command_id, payload, conn))
        with self._conn_lock:
            self._drop_client("the client closed the connection", conn)

    def _drop_client(self, why: str, conn: socket.socket) -> None:
        """Close ``conn`` and forget the client, if ``conn`` is still the connected one; the caller
        holds ``_conn_lock``."""
        if self._conn is not conn:
            return
        log.info("fci %s: %s", self.bind, why)
        self._conn, self.client_udp = None, None
        try:  # shutdown wakes the connection's blocked recv, so its thread ends; close alone does not
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass
        self.requests.put(Request(Command.STOP_MOVE, -1, b""))  # a motion without a client ends

    def _stop_queued_by(self, conn: socket.socket) -> bool:
        """Whether *conn* has a ``StopMove`` waiting in the queue for the physics thread."""
        with self.requests.mutex:
            return any(
                r.command == Command.STOP_MOVE and r.conn is conn for r in self.requests.queue
            )

    def is_current(self, req: Request) -> bool:
        """Whether *req* is the server's own or came from the client connected now: libfranka
        numbers its commands from 0 on every connection, so a dropped client's request carried
        out for the next one would answer an id the next client uses itself."""
        return req.conn is None or req.conn is self._conn

    def reply(
        self, command: Command, command_id: int, payload: bytes, to: socket.socket | None = None
    ) -> None:
        """Answer *command_id* on the connection *to* that sent it, if it is still connected; a
        reply never reaches a client that did not ask."""
        conn = self._conn
        if conn is None or command_id < 0 or (to is not None and to is not conn):
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

    def send_state(self, data: bytes, to: tuple[str, int] | None = None) -> None:
        """Stream a state to ``to``, or to the connected client; a client the TCP thread drops
        meanwhile gets nothing, and the physics goes on. The state is kept, for
        :meth:`_keep_alive` to send again while no newer one comes."""
        self._last_state, self._last_sent = data, time.monotonic()
        target = self.client_udp if to is None else to  # one read: the TCP thread may clear it
        if target is not None:
            try:
                self._udp.sendto(data, target)
            except OSError:
                pass

    def await_command(
        self,
        message_id: int,
        state: bytes,
        *,
        strict: bool,
        client_timeout_s: float,
        owner: socket.socket | None = None,
    ) -> dict | None:
        """The ``RobotCommand`` answering ``message_id``, from ``owner``'s address: the connection
        whose motion it is (the one connected now, by default).

        ``strict``: wait for it, re-sending ``state`` every ``resend_s``, and raise after
        ``client_timeout_s`` of wall time; otherwise wait at most ``first_command_wait_s``.

        ``None`` when none came within ``first_command_wait_s`` (not strict), or when ``owner`` is
        no longer the connected client (either way): dropped or replaced, its motion ends through
        the ``StopMove`` the drop queues, and the run goes on, as a robot whose client went away
        does. A client that replaced it is not waited for: it did not start the motion. ``None``
        too once ``owner`` has queued a ``StopMove``: libfranka cancels a motion whose control
        callback threw by sending one and answering no further state, and waits for its reply.
        """
        with self._conn_lock:
            owner = self._conn if owner is None else owner
            target = self.client_udp if owner is self._conn else None
        start = time.monotonic()
        last_send = start
        limit = client_timeout_s if strict else self.first_command_wait_s
        while target is not None and self._conn is owner:
            if self._stop_queued_by(owner):
                return None
            now = time.monotonic()
            if now - start > limit:
                if strict:
                    raise TimeoutError(
                        f"FCI {self.bind}: no command for state {message_id} within "
                        f"{client_timeout_s:.0f} s of wall time; the client stopped answering"
                    )
                return None
            if strict and now - last_send > self.resend_s:
                self.send_state(state, to=target)  # to the motion's own client, not a successor
                last_send = now
            ready, _, _ = select.select([self._udp], [], [], min(0.005, self.resend_s))
            if not ready:
                continue
            data, sender = self._udp.recvfrom(4096)
            if sender != target or len(data) != ROBOT_COMMAND.size:
                continue
            cmd = unpack_robot_command(data)
            if cmd["message_id"] == message_id:
                return cmd
        return None

    def _keep_alive(self) -> None:
        """Send the last state again every ``resend_s`` while the physics sends none: a paused
        run, one stepped on request, a slow tick, the time before the first step. libfranka ends
        a client that hears no state for its UDP timeout (1 s), and discards a state that is not
        newer than the last it read, so the repeat keeps it connected and changes nothing else."""
        while not self._closed:
            time.sleep(self.resend_s)
            data = self._last_state
            if data is None or time.monotonic() - self._last_sent < self.resend_s:
                continue
            target = self.client_udp
            if target is not None:
                try:
                    self._udp.sendto(data, target)
                except OSError:
                    pass
                self._last_sent = time.monotonic()

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
        self._keeper.join(timeout=5)
