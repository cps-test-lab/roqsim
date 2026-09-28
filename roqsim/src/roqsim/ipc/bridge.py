"""The ``ipc`` bridge: every endpoint of a running simulation, served on demand over ZeroMQ.

A second transport beside ROS 2, and the one ``roqsim sim`` starts by default (``--control``). It
carries every payload as it is, so it wires every endpoint, hand-built ones included
(:meth:`IpcBridge._hints_for`); an endpoint opts out with ``ipc=None``. Each endpoint has a path built
from the plugin that registered it and its name -- ``robot.lidar`` + ``scan`` is ``robot/lidar/scan``.
``describe`` gives an endpoint's schema -- its payload type, parameters and units, and the
attribute or config key its rate and presence come from -- and what each other bridge made of it
(its ROS topic, type and QoS).

**Nothing runs per physics step while nobody asks.** Two sockets:

* a ROUTER at the control URI answers requests (``hello``, ``describe``, ``read``, ``call``) on a
  background thread. A ``read`` runs the endpoint's read on the physics thread through
  :meth:`~roqsim.context.SimContext.submit` and replies with it; a ``call`` writes a command and
  waits on its :class:`~roqsim.context.CommandFuture` -- a timeout is an error reply, never a
  success -- or puts a stream's value in its slot.
* an XPUB at :func:`roqsim.ipc.pub_uri` publishes the ``out`` endpoints some client subscribed to,
  from ``post_step`` at the endpoint's rate. ZeroMQ reports each subscription prefix to this
  socket, so the bridge publishes exactly the endpoints under a subscribed prefix and nothing when
  there is none. An endpoint's ``has_subscribers`` also answers yes for an IPC subscription.

A command may name an ``out`` endpoint that confirms it (``Endpoint.confirm``): the reply then
carries that endpoint's value as read in the ``post_step`` of the step that applied the command.
While the run is paused no step runs, so the reply says ``applied`` and ``verified: false`` rather
than stepping or reporting the verdict from before the change.

Messages are :mod:`roqsim.ipc.wire`: JSON, with numpy arrays as raw frames beside it. pyzmq is the
``roqsim[ipc]`` extra and is imported only here, when the bridge starts.
"""

from __future__ import annotations

import collections
import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import control as ctl
from ..bridge import BridgeBase, _RateGate
from ..context import CommandFuture, endpoint_kind
from ..document import nearest
from ..endpoint import ParameterError
from ..plugin import PluginError
from ..rates import snap_rate
from . import PROTOCOL, pub_uri, register, unregister, wire

if TYPE_CHECKING:
    from ..context import Endpoint, SimContext

#: How long a request waits for the physics thread when it names no timeout (seconds).
DEFAULT_TIMEOUT_S = 5.0
#: Threads that wait on the physics thread for requests, so one slow call does not hold the rest.
WORKERS = 4


def path_of(ep: Endpoint) -> str:
    """``<producer address, dots as slashes>/<name>``: ``robot.lidar`` + ``scan`` -> ``robot/lidar/scan``."""
    producer = ep.producer or ep.owner
    return f"{producer.replace('.', '/')}/{ep.name}" if producer else ep.name


class _Refusal(Exception):
    def __init__(self, kind: str, message: str, **detail):
        super().__init__(message)
        self.kind = kind
        self.detail = detail


class _Confirmation:
    """A command waiting for the ``post_step`` after it applied, to read its confirming endpoint."""

    __slots__ = ("cancelled", "done", "endpoint", "future", "value")

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.future: CommandFuture | None = None
        self.done = threading.Event()
        self.value: Any = None
        self.cancelled = False


def _select(value: Any, field: str) -> Any:
    """*field* of *value*, dotted, through mappings, attributes and list indices."""
    for part in field.split("."):
        if isinstance(value, dict):
            if part not in value:
                raise _Refusal(
                    "bad_request", f"no field {part!r}; it has: {', '.join(map(str, value))}"
                )
            value = value[part]
        elif isinstance(value, list | tuple) and part.lstrip("-").isdigit():
            value = value[int(part)]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            names = sorted(k for k in getattr(value, "__dict__", {}) if not k.startswith("_"))
            names = names or list(getattr(value, "__dataclass_fields__", {}))
            raise _Refusal(
                "bad_request",
                f"no field {part!r} in a {type(value).__name__}"
                + (f"; it has: {', '.join(names)}" if names else ""),
            )
    return value


class IpcBridge(BridgeBase):
    """Serve every endpoint over ZeroMQ at ``uri``; see the module docstring."""

    BACKEND = "ipc"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._uri = str(self.config.get("uri", ""))
        self._world = str(self.config.get("world", ""))
        self._paths: dict[str, Endpoint] = {}
        self._bound_inputs: list[tuple[Endpoint, Any]] = []
        self._inputs: dict[str, Any] = {}
        self._confirm_of: dict[str, Endpoint] = {}
        self._gates: dict[int, _RateGate] = {}
        # Read on the physics thread every step, replaced whole by the server thread.
        self._active: list = []
        self._subscribed: frozenset[str] = frozenset()
        self._prefixes: set[str] = set()
        self._confirms: collections.deque[_Confirmation] = collections.deque()
        self._zmq = None
        self._zctx = None
        self._push = None  # the physics thread's socket to the server thread
        self._inproc = f"inproc://roqsim-ipc-{id(self)}"
        self._requests: queue.SimpleQueue = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._registration = None
        self._late = threading.Lock()

    def validate_config(self, config: dict) -> list[str]:
        return [] if config.get("uri") else ["ipc_bridge needs a `uri` (ipc://<path> or tcp://...)"]

    # -- setup and teardown (physics thread) ------------------------------------------------------
    def _setup(self, ctx: SimContext) -> None:
        try:
            import zmq
        except ImportError as err:
            raise PluginError(
                f"the ipc bridge needs pyzmq, which is not installed ({err}). Install the extra: "
                "pip install 'roqsim[ipc]' -- or run with --control none."
            ) from None
        self._zmq = zmq
        try:
            self._registration = register(self._uri, pub_uri(self._uri), self._world)
        except RuntimeError as err:
            raise PluginError(str(err)) from None
        self._zctx = zmq.Context()
        try:
            self._router = self._socket(zmq.ROUTER, self._uri)
            self._xpub = self._socket(zmq.XPUB, pub_uri(self._uri))
        except zmq.ZMQError as err:
            self._teardown(ctx)
            raise PluginError(f"cannot serve the control socket at {self._uri}: {err}") from None
        self._pull = self._zctx.socket(zmq.PULL)
        self._pull.bind(self._inproc)
        self._push = self._zctx.socket(zmq.PUSH)
        self._push.linger = 0
        self._push.connect(self._inproc)
        server = threading.Thread(target=self._serve, name="roqsim-ipc", daemon=True)
        self._threads.append(server)
        for i in range(WORKERS):
            self._threads.append(
                threading.Thread(target=self._work, name=f"roqsim-ipc-{i}", daemon=True)
            )

    def _socket(self, kind, uri: str):
        sock = self._zctx.socket(kind)
        sock.linger = 0
        sock.bind(uri)
        return sock

    def _bind(self, ctx: SimContext) -> None:
        super()._bind(ctx)
        bound = [out.endpoint for out in self._outputs] + [ep for ep, _ in self._bound_inputs]
        for ep in bound:
            path = path_of(ep)
            other = self._paths.get(path)
            if other is not None:
                raise PluginError(
                    f"two endpoints would share the control path {path!r}: {other.name!r} of "
                    f"{other.producer or other.owner!r} and {ep.name!r} of "
                    f"{ep.producer or ep.owner!r}. Rename one, or keep one off this transport "
                    "with ipc=None."
                )
            self._paths[path] = ep
        self._inputs = {path_of(ep): cb for ep, cb in self._bound_inputs}
        self._gates = {id(out.endpoint): out.gate for out in self._outputs}
        for path, ep in self._paths.items():
            if ep.direction == "in" and ep.confirm:
                sibling = self._paths.get(path.rpartition("/")[0] + "/" + ep.confirm)
                if sibling is None or sibling.direction != "out":
                    raise PluginError(
                        f"command {path!r} is confirmed by {ep.confirm!r}, which its producer "
                        "does not publish as an out endpoint"
                    )
                self._confirm_of[path] = sibling
        # An IPC subscriber is a subscriber: a producer that skips work nobody asked for (a camera
        # render) asks has_subscribers, which another transport may already have set.
        for out in self._outputs:
            prev = out.endpoint.has_subscribers
            if prev is not None:
                out.endpoint.has_subscribers = lambda p=prev, path=out.handle: (
                    p() or path in self._subscribed
                )
        for thread in self._threads:
            thread.start()

    def _teardown(self, ctx: SimContext) -> None:
        if self._push is not None and self._threads and self._threads[0].is_alive():
            self._push.send_multipart([b"S"])
            self._threads[0].join(timeout=2.0)
        for _ in self._threads[1:]:
            self._requests.put(None)
        if self._zctx is not None:
            self._zctx.destroy(linger=0)
            self._zctx = None
            for uri in (self._uri, pub_uri(self._uri)):
                if uri.startswith("ipc://"):
                    Path(uri[len("ipc://") :]).unlink(missing_ok=True)
        unregister(self._registration)
        self._registration = None

    # -- binding hooks ------------------------------------------------------------------------------
    def _hints_for(self, ep: Endpoint) -> dict | None:
        """Every endpoint, with its ``ipc`` hints or none; a hint block of ``None`` keeps it off."""
        return ep.backend.get(self.BACKEND, {})

    def _make_output(self, ep: Endpoint, hints: dict) -> str:
        return path_of(ep)

    def _make_input(self, ep: Endpoint, hints: dict, on_payload) -> None:
        self._bound_inputs.append((ep, on_payload))

    def _rate_gate(self, ctx: SimContext, rate_hz: float, subject: str) -> _RateGate:
        # On the grid like every publication, but silent: an endpoint is published here only when a
        # client subscribes, so a snap is not news at start-up. ROS reports its own.
        model = getattr(ctx, "model", None)
        if rate_hz <= 0.0 or model is None:
            return _RateGate(rate_hz)
        snapped = snap_rate(rate_hz, float(model.opt.timestep))
        return _RateGate(float(snapped.hz), every=snapped.every)

    def _record_rate(self, *args, **kwargs) -> None:
        # Not recorded: nothing here is published unless someone subscribes during the run.
        return

    # -- per step (physics thread) ------------------------------------------------------------------
    def post_step(self, ctx: SimContext) -> None:
        active = self._active
        if active:
            t = ctx.sim_time
            for out in active:
                if out.gate.due(t):
                    payload = out.endpoint.read()
                    if payload is not None:
                        frames = wire.pack({"path": out.handle, "t": float(t)}, payload)
                        self._push.send_multipart([b"P", out.handle.encode(), *frames], copy=False)
        if self._confirms:
            self._settle_confirmations()

    def _settle_confirmations(self) -> None:
        keep = []
        while self._confirms:
            waiter = self._confirms.popleft()
            if waiter.cancelled:
                continue
            if waiter.future is None or not waiter.future.done():
                keep.append(waiter)
                continue
            try:
                waiter.value = wire.encode(waiter.endpoint.read())
            except Exception as exc:  # noqa: BLE001 - the caller gets it, not the loop
                waiter.value = exc
            waiter.done.set()
        self._confirms.extend(keep)

    # -- the server thread -------------------------------------------------------------------------
    def _serve(self) -> None:
        zmq = self._zmq
        poller = zmq.Poller()
        for sock in (self._router, self._xpub, self._pull):
            poller.register(sock, zmq.POLLIN)
        try:
            while True:
                for sock, _ in poller.poll():
                    if sock is self._router:
                        self._requests.put(self._router.recv_multipart(copy=False))
                    elif sock is self._xpub:
                        self._subscription(self._xpub.recv())
                    else:
                        frames = self._pull.recv_multipart(copy=False)
                        kind = frames[0].bytes
                        if kind == b"S":
                            return
                        target = self._router if kind == b"R" else self._xpub
                        target.send_multipart(frames[1:], copy=False)
        except zmq.ZMQError:
            return  # the context was destroyed under us: shutting down

    def _subscription(self, msg: bytes) -> None:
        if not msg:
            return
        prefix = msg[1:].decode(errors="replace")
        if msg[0] == 1:
            self._prefixes.add(prefix)
        elif msg[0] == 0:
            self._prefixes.discard(prefix)
        prefixes = tuple(self._prefixes)
        active = [o for o in self._outputs if o.handle.startswith(prefixes)] if prefixes else []
        self._subscribed = frozenset(o.handle for o in active)
        self._active = active

    # -- worker threads -----------------------------------------------------------------------------
    def _work(self) -> None:
        zmq = self._zmq
        try:
            push = self._zctx.socket(zmq.PUSH)
            push.linger = 0
            push.connect(self._inproc)
        except (zmq.ZMQError, AttributeError):
            return
        while True:
            frames = self._requests.get()
            if frames is None:
                push.close()
                return
            identity, body = frames[0], frames[1:]
            try:
                push.send_multipart([b"R", identity, *self._answer(body)], copy=False)
            except zmq.ZMQError:
                return

    def _answer(self, body: list) -> list:
        rid = None
        try:
            header, value = wire.unpack(body)
            rid = header.get("id")
            op = header.get("op")
            handler = {
                "hello": self._hello,
                "describe": self._describe,
                "read": self._read,
                "call": self._call,
            }.get(op)
            if handler is None:
                raise _Refusal(
                    "bad_request", f"unknown op {op!r}; one of: hello, describe, read, call"
                )
            result = handler(header, value)
            return wire.pack({"id": rid, "ok": True}, result)
        except _Refusal as refusal:
            return wire.pack(
                {"id": rid, "ok": False, "error": refusal.kind, "message": str(refusal)},
                refusal.detail or None,
            )
        except Exception as exc:  # noqa: BLE001 - a malformed request must not kill a worker
            return wire.pack(
                {"id": rid, "ok": False, "error": "bad_request", "message": f"{exc}"}, None
            )

    # -- operations ---------------------------------------------------------------------------------
    def _hello(self, _header: dict, _value) -> dict:
        ctx = self._ctx
        return {
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "world": self._world,
            "uri": self._uri,
            "pub": pub_uri(self._uri),
            "state": ctl.STATE_NAMES.get(ctx.control.state, str(ctx.control.state)),
            "endpoints": len(self._paths),
        }

    def _refresh(self) -> None:
        """Take in the ``out`` endpoints registered after bind (``on_demand``, such as an entity's
        pose): read on request only, so serving them late loses nothing."""
        with self._late:
            known = {id(ep) for ep in self._paths.values()}
            for ep in self._ctx.interface.all():
                if id(ep) in known or ep.direction != "out" or ep.read is None:
                    continue
                if self._hints_for(ep) is None:
                    continue
                self._paths.setdefault(path_of(ep), ep)

    def _endpoint(self, path: str) -> Endpoint:
        ep = self._paths.get(path)
        if ep is None:
            self._refresh()
            ep = self._paths.get(path)
        if ep is not None:
            return ep
        guess = nearest(path, self._paths)
        parent = path.rstrip("/")
        siblings: list[str] = []
        while parent and not siblings:
            parent = parent.rpartition("/")[0]
            siblings = sorted(p for p in self._paths if parent and p.startswith(parent + "/"))
        if not siblings:
            siblings = sorted({p.split("/")[0] for p in self._paths})
        message = f"no endpoint {path!r}."
        if guess:
            message += f" Did you mean {guess!r}?"
        where = f"under {parent!r}" if parent else "at the top"
        message += f" {where}: {', '.join(siblings[:30])}" + (" ..." if len(siblings) > 30 else "")
        raise _Refusal("unknown_path", message, suggestion=guess, siblings=siblings)

    def _entry(self, path: str, ep: Endpoint, full: bool) -> dict:
        kind = endpoint_kind(ep)
        doc = ep.doc.strip()
        entry: dict[str, Any] = {"path": path, "kind": kind}
        if kind == "out":
            gate = self._gates.get(id(ep))
            entry["rate_hz"] = float(gate.rate_hz) if gate is not None else float(ep.rate_hz)
        entry["doc"] = doc if full else (doc.splitlines()[0] if doc else "")
        if not full:
            return entry
        entry.update(owner=ep.owner, producer=ep.producer, name=ep.name, namespace=ep.namespace)
        # Where a decorated endpoint's rate, presence and family come from: an attribute or config key.
        if isinstance(ep.options.get("rate"), dict):
            entry["rate_from"] = ep.options["rate"]["from"]
        for key in ("when", "family"):
            if key in ep.options:
                entry[key] = ep.options[key]
        if path in self._confirm_of:
            entry["confirm"] = path_of(self._confirm_of[path])
        # The typed schema, where the endpoint declares one (roqsim.endpoint).
        if ep.payload_type is not None:
            entry["payload"] = ep.payload_type.describe()
        if ep.params is not None:
            entry["params"] = [p.describe() for p in ep.params]
        if ep.result is not None:
            entry["result"] = ep.result.describe()
        entry["bridges"] = {
            bridge.BACKEND: named
            for bridge in self._ctx.interface.bridges
            if bridge is not self and (named := bridge.bound_name(ep)) is not None
        }
        return entry

    def _describe(self, header: dict, _value) -> Any:
        self._refresh()
        path = str(header.get("path") or "").strip("/")
        if not path:
            return [self._entry(p, ep, False) for p, ep in sorted(self._paths.items())]
        if path in self._paths:
            return self._entry(path, self._paths[path], True)
        under = sorted(p for p in self._paths if p.startswith(path + "/"))
        if under:
            return [self._entry(p, self._paths[p], False) for p in under]
        self._endpoint(path)  # raises with suggestions
        return None

    def _timeout(self, header: dict) -> float:
        value = header.get("timeout")
        return DEFAULT_TIMEOUT_S if value is None else max(0.0, float(value))

    def _read(self, header: dict, _value) -> Any:
        path = str(header.get("path") or "")
        ep = self._endpoint(path)
        if ep.direction != "out":
            raise _Refusal(
                "wrong_kind", f"{path} is a {endpoint_kind(ep)}, not an out endpoint: use call"
            )
        field = str(header.get("field") or "")
        timeout = self._timeout(header)

        def read(_ctx):
            payload = ep.read()
            return wire.encode(_select(payload, field) if field else payload)

        try:
            return self._ctx.submit(read).result(timeout)
        except TimeoutError:
            raise _Refusal(
                "timeout",
                f"the physics thread did not read {path} within {timeout:g} s -- the simulation "
                "is stalled or shutting down",
            ) from None
        except _Refusal:
            raise
        except Exception as exc:  # noqa: BLE001 - the producer's failure is the reply
            raise _Refusal("refused", f"{exc}", exception=type(exc).__name__) from None

    def _call(self, header: dict, value) -> dict:
        path = str(header.get("path") or "")
        ep = self._endpoint(path)
        if ep.direction != "in":
            raise _Refusal("wrong_kind", f"{path} is an out endpoint: use read")
        write = self._inputs[path]
        if endpoint_kind(ep) == "stream":
            try:
                write(value)
            except ParameterError as exc:
                raise _Refusal("bad_request", str(exc)) from None
            return {"queued": True}
        timeout = self._timeout(header)
        confirm = self._confirm_of.get(path)
        waiter = _Confirmation(confirm) if confirm is not None else None
        outcome = write(value)
        if not isinstance(outcome, CommandFuture):
            raise _Refusal("refused", f"{path} could not be queued: the bridge is not running")
        if waiter is not None:
            waiter.future = outcome
            self._confirms.append(waiter)
        try:
            result = self._outcome(path, outcome, timeout)
        except _Refusal:
            if waiter is not None:
                waiter.cancelled = True
            raise
        reply = {"applied": True, "result": result}
        if waiter is None:
            return reply
        confirm_path = path_of(confirm)
        if self._ctx.control.state != ctl.PLAYING:
            waiter.cancelled = True
            reply.update(
                verified=False,
                note=f"applied, unverified: the simulation is not stepping, and {confirm_path} "
                "records its verdict in the step after the command",
            )
            return reply
        if not waiter.done.wait(timeout):
            waiter.cancelled = True
            raise _Refusal(
                "timeout",
                f"{path} applied, but no step ran within {timeout:g} s to confirm it "
                f"({confirm_path})",
            )
        if isinstance(waiter.value, Exception):
            raise _Refusal("refused", f"reading {confirm_path} failed: {waiter.value}")
        reply.update(verified=True, confirmation=waiter.value)
        return reply

    def _outcome(self, path: str, future: CommandFuture, timeout: float) -> Any:
        """What the command returned -- waiting on a future it returned in turn -- or a refusal."""
        try:
            result = future.result(timeout)
            if isinstance(result, CommandFuture):
                result = result.result(timeout)
        except TimeoutError:
            raise _Refusal(
                "timeout",
                f"no outcome for {path} within {timeout:g} s: the physics thread did not get to it "
                "(stalled or shutting down), so it may or may not apply later",
            ) from None
        except ParameterError as exc:  # refused before it was queued
            raise _Refusal("bad_request", str(exc)) from None
        except Exception as exc:  # noqa: BLE001 - the producer's own refusal is the reply
            raise _Refusal("refused", f"{exc}", exception=type(exc).__name__) from None
        return wire.encode(result)
