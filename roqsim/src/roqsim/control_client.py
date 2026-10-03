"""Talk to a running simulation from Python: read, call and subscribe to its endpoints.

::

    from roqsim.control_client import Client

    sim = Client()                                   # ROQSIM_CONTROL, or the only one running
    sim.endpoints()                                  # [{'path': 'robot/lidar/scan', 'kind': 'out', ...}]
    sim.read("robot/diff_drive/odom")                # the value, numpy arrays included
    sim.read("sim/run_control/state", field="sim_time")
    sim.call("robot/diff_drive/cmd_vel", {"vx": 0.3})   # parameters by name
    sim.call("grip_fault/override", True)            # {'applied': True, 'verified': True, ...}
    sim.pause(); sim.step(10); sim.resume()
    for path, t, value in sim.subscribe("robot/lidar"):
        ...

The server is :class:`roqsim.ipc.bridge.IpcBridge`, which ``roqsim sim`` starts by default. A
refused request raises :class:`~roqsim.ipc.ControlError` with the server's message -- for a command
the producer refused, its own exception text. One :class:`Client` is one socket: use it from one
thread, or give each thread its own.

The module-level functions below (:func:`list_endpoints` ...) are the same calls returning plain
JSON with errors as ``{"error": ...}`` and large arrays summarised, for a caller that is a tool
server rather than a program.
"""

from __future__ import annotations

import time
from typing import Any

from .ipc import ControlError, discover, wire

__all__ = ["Client", "ControlError", "Subscription"]

#: How long a request waits for a reply beyond the server's own timeout (seconds).
_GRACE_S = 2.0
_NO_VALUE = object()


def _zmq():
    try:
        import zmq
    except ImportError as err:
        raise ControlError(
            f"talking to a simulator needs pyzmq ({err}): pip install 'roqsim[ipc]'", "unreachable"
        ) from None
    return zmq


class Client:
    """A connection to one running simulator. See the module docstring."""

    def __init__(self, control: str | None = None, *, timeout: float = 5.0):
        zmq = _zmq()
        try:
            self.uri = discover(control)
        except ValueError as err:
            raise ControlError(str(err), "bad_request") from None
        self.timeout = float(timeout)
        self._zctx = zmq.Context()
        self._sock = self._zctx.socket(zmq.DEALER)
        self._sock.linger = 0
        self._sock.connect(self.uri)
        self._next = 0
        self._late: set[int] = set()  # requests sent and not yet collected
        self._replies: dict[int, tuple[dict, Any]] = {}
        self._pub: str | None = None

    def close(self) -> None:
        self._zctx.destroy(linger=0)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the protocol -------------------------------------------------------------------------------
    def request(self, op: str, value: Any = _NO_VALUE, *, timeout: float | None = None, **fields):
        """Send one request and return its value; raises :class:`ControlError` on a refusal."""
        wait = self.timeout if timeout is None else float(timeout)
        rid = self.send(op, value, timeout=wait, **fields)
        deadline = time.monotonic() + wait + _GRACE_S
        while True:
            done, result = self.poll(rid, max(0.0, deadline - time.monotonic()))
            if done:
                return result
            if time.monotonic() >= deadline:
                self._late.discard(rid)
                raise ControlError(
                    f"no reply from {self.uri} within {wait + _GRACE_S:g} s: is a simulator "
                    "serving it? `roqsim ls` lists the running ones.",
                    "unreachable",
                )

    def send(self, op: str, value: Any = _NO_VALUE, *, timeout: float | None = None, **fields):
        """Send a request without waiting; returns its id for :meth:`poll`."""
        self._next += 1
        header = {
            "op": op,
            "id": self._next,
            "timeout": self.timeout if timeout is None else timeout,
        }
        frames = wire.pack({**header, **fields}, None if value is _NO_VALUE else value)
        self._sock.send_multipart(frames, copy=False)
        self._late.add(self._next)
        return self._next

    def poll(self, rid: int, wait: float = 0.0) -> tuple[bool, Any]:
        """``(True, value)`` once request *rid* is answered, ``(False, None)`` until then.

        Waits up to *wait* seconds (0: not at all). A refusal raises :class:`ControlError`. Replies
        to other requests of this client that arrive meanwhile are kept for their own ``poll``.
        """
        zmq = _zmq()
        deadline = time.monotonic() + wait
        while rid not in self._replies:
            left = deadline - time.monotonic()
            if not self._sock.poll(max(0, int(left * 1000)), zmq.POLLIN):
                return False, None
            reply, result = wire.unpack(self._sock.recv_multipart(copy=False))
            if reply.get("id") in self._late:
                self._replies[reply["id"]] = (reply, result)
        self._late.discard(rid)
        reply, result = self._replies.pop(rid)
        if not reply.get("ok"):
            raise ControlError(
                reply.get("message", "refused"), reply.get("error", "error"), result or {}
            )
        return True, result

    def hello(self) -> dict:
        """Who is serving: pid, world, URIs, run state and the number of endpoints."""
        info = self.request("hello")
        self._pub = info.get("pub")
        return info

    def endpoints(self, prefix: str = "") -> list[dict]:
        """Every endpoint (or those under *prefix*): path, kind, rate and first doc line."""
        found = self.request("describe", path=prefix)
        return found if isinstance(found, list) else [found]

    def describe(self, path: str) -> dict | list:
        """One endpoint in full -- doc, kind, rate, confirmation, other transports' names -- or the
        list under a prefix."""
        return self.request("describe", path=path)

    def read(self, path: str, field: str = "", *, timeout: float | None = None) -> Any:
        """The current value of an ``out`` endpoint (read on the physics thread), or one field."""
        return self.request("read", path=path, field=field, timeout=timeout)

    def call(self, path: str, value: Any = None, *, timeout: float | None = None) -> dict:
        """Write a command and wait for its outcome, or put a value in a stream.

        *value* is the parameters by name, a mapping, for an endpoint that declares them
        (``describe`` lists them), or a bare value for one that declares exactly one; an endpoint
        built by hand takes its payload as it is.

        A command replies ``{"applied": True, "result": <what it returned>}``, plus
        ``verified``/``confirmation`` where the endpoint names one that confirms it. A stream
        replies ``{"queued": True}``.
        """
        return self.request("call", value, path=path, timeout=timeout)

    def subscribe(self, prefix: str = "") -> Subscription:
        """The ``out`` endpoints under *prefix*, as they are published (at each one's rate)."""
        if self._pub is None:
            self.hello()
        return Subscription(self._pub, prefix)

    # -- run control ----------------------------------------------------------------------------------
    def state(self) -> dict:
        return self.read("sim/run_control/state")

    def pause(self) -> dict:
        return self.call("sim/run_control/pause")["result"]

    def resume(self) -> dict:
        return self.call("sim/run_control/resume")["result"]

    def step(self, n: int = 1, *, timeout: float | None = None) -> dict:
        """Take *n* steps while paused; returns once they ran, with the sim time reached."""
        return self.call("sim/run_control/step", {"n": int(n)}, timeout=timeout)["result"]

    def reset(self, *, timeout: float | None = None) -> dict:
        return self.call("sim/run_control/reset", timeout=timeout)["result"]


class Subscription:
    """Published values under one prefix: iterate for ``(path, sim_time, value)``."""

    def __init__(self, pub: str, prefix: str = ""):
        zmq = _zmq()
        self._zctx = zmq.Context()
        self._sock = self._zctx.socket(zmq.SUB)
        self._sock.linger = 0
        self._sock.connect(pub)
        self._sock.setsockopt(zmq.SUBSCRIBE, prefix.encode())

    def get(self, timeout: float | None = None) -> tuple[str, float, Any] | None:
        """The next value, or ``None`` if none arrives within *timeout* seconds."""
        zmq = _zmq()
        if timeout is not None and not self._sock.poll(int(timeout * 1000), zmq.POLLIN):
            return None
        frames = self._sock.recv_multipart(copy=False)
        header, value = wire.unpack(frames[1:])
        return header["path"], float(header["t"]), value

    def __iter__(self):
        while True:
            yield self.get()

    def close(self) -> None:
        self._zctx.destroy(linger=0)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# -- plain-JSON calls (a tool server's) -----------------------------------------------------------
#: Arrays larger than this are summarised in the plain-JSON calls.
MAX_ITEMS = 64


def _plainly(fn, control: str = ""):
    try:
        with Client(control or None) as sim:
            return {"value": wire.plain(fn(sim), max_items=MAX_ITEMS)}
    except ControlError as err:
        return {"error": str(err), "kind": err.kind}


def list_endpoints(prefix: str = "", control: str = "") -> dict:
    """The endpoints of the running simulator (under *prefix*): path, kind, rate, first doc line."""
    return _plainly(lambda sim: sim.endpoints(prefix), control)


def describe_endpoint(path: str, control: str = "") -> dict:
    """One endpoint in full: doc, kind, rate, what confirms it, its name on other transports."""
    return _plainly(lambda sim: sim.describe(path), control)


def read_endpoint(path: str, field: str = "", control: str = "") -> dict:
    """The current value of an out endpoint, or one (dotted) field of it."""
    return _plainly(lambda sim: sim.read(path, field), control)


def call_endpoint(path: str, value: Any = None, control: str = "") -> dict:
    """Send a command (waits for its outcome and confirmation) or a stream value."""
    return _plainly(lambda sim: sim.call(path, value), control)


def pause(control: str = "") -> dict:
    """Pause the running simulation; commands still run while it is paused."""
    return _plainly(lambda sim: sim.pause(), control)


def resume(control: str = "") -> dict:
    """Resume a paused simulation."""
    return _plainly(lambda sim: sim.resume(), control)


def step(n: int = 1, control: str = "") -> dict:
    """Take n steps of a paused simulation; returns the sim time reached."""
    return _plainly(lambda sim: sim.step(n), control)
