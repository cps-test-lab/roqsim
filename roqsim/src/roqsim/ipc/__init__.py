"""Where a running simulation's control socket is, and how a client finds it.

``roqsim sim`` serves every endpoint over ZeroMQ (:class:`roqsim.ipc.bridge.IpcBridge`) at one URI:

* ``ipc://<path>`` -- a Unix socket; the default is ``roqsim-control.sock`` in the run directory
  (``RUN_OUTPUT_DIR``, else ``OUTPUT_DIR``, else the working directory).
* ``tcp://<host>:<port>`` -- ``tcp://:<port>`` binds ``127.0.0.1``; name another address to open it
  beyond this machine.

The PUB socket that carries subscriptions is derived from it: ``<path>.pub`` for ``ipc://``, port + 1
for ``tcp://``. Each running simulator also leaves a small JSON file in a per-user runtime directory
(:func:`runtime_dir`), which is what ``roqsim ls`` lists and what lets a client find "the only one
running" without being told.

A client resolves the address in this order (:func:`discover`): the URI it was given, then
``ROQSIM_CONTROL``, then the run directory's socket if a running simulator serves it, then the only
simulator running. Several
running and none named is refused with the list, never guessed.

Nothing here imports ZeroMQ: finding a simulator costs a directory listing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

#: The file name of the default control socket in the run directory.
SOCKET_NAME = "roqsim-control.sock"
#: The environment variable naming the control URI, for ``roqsim sim`` and for a client alike.
ENV = "ROQSIM_CONTROL"
#: What ``--control`` / ``ROQSIM_CONTROL`` take to serve nothing.
NONE = "none"
#: The run directory's anchors, most specific first (shared with ``roqsim sim``'s output paths).
OUTPUT_DIR_VARS = ("RUN_OUTPUT_DIR", "OUTPUT_DIR")
#: A Unix socket path longer than this cannot be bound (``sun_path`` holds 108 bytes with its NUL).
IPC_PATH_MAX = 107
#: The protocol version a server states in ``hello``.
PROTOCOL = 1

log = logging.getLogger(__name__)


def path_of(ep) -> str:
    """An endpoint's path: ``<producer address, dots as slashes>/<name>``.

    ``robot.lidar`` + ``scan`` is ``robot/lidar/scan``; an endpoint with no producer (the core's
    entity poses) is placed under its owner (``sim`` + ``entities/robot/pose``).
    """
    producer = ep.producer or ep.owner
    return f"{producer.replace('.', '/')}/{ep.name}" if producer else ep.name


class ControlError(RuntimeError):
    """No simulator to talk to, or one that refused the request. ``kind`` names which.

    ``kind`` is ``unreachable`` (nothing answers at the address), ``not_found`` (no simulator could
    be found), ``ambiguous`` (several run and none was named), ``unknown_path``, ``wrong_kind``,
    ``refused`` (the producer raised; the message is its own), ``timeout`` or ``bad_request``.
    """

    def __init__(self, message: str, kind: str = "error", detail: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.detail = detail or {}


def run_dir() -> Path:
    """The directory a run's outputs land in: the first output variable set, else the CWD."""
    for var in OUTPUT_DIR_VARS:
        base = os.environ.get(var)
        if base:
            return Path(base)
    return Path.cwd()


def runtime_dir() -> Path:
    """Per-user directory where each running simulator leaves its registration file."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base and Path(base).is_dir():
        return Path(base) / "roqsim"
    return Path(tempfile.gettempdir()) / f"roqsim-{os.getuid()}"


def default_uri() -> str:
    """``ipc://`` + the control socket in the run directory."""
    return "ipc://" + str((run_dir() / SOCKET_NAME).resolve())


def normalize(value: str) -> str | None:
    """A control address as a full URI, or ``None`` for :data:`NONE`.

    ``tcp://:5555`` (or ``tcp://5555``) is ``tcp://127.0.0.1:5555``; a relative ``ipc://`` path and a
    bare path are made absolute, so the URI printed and registered is one another process can use.
    """
    value = str(value).strip()
    if value.lower() == NONE:
        return None
    if value.startswith("tcp://"):
        rest = value[len("tcp://") :]
        host, sep, port = rest.rpartition(":")
        if not sep:
            host, port = "", rest
        if not port.isdigit():
            raise ValueError(f"{value!r}: a tcp:// control address needs a port, e.g. tcp://:5555")
        return f"tcp://{host or '127.0.0.1'}:{port}"
    path = value[len("ipc://") :] if value.startswith("ipc://") else value
    if not path:
        raise ValueError("an ipc:// control address needs a path")
    return "ipc://" + str(Path(path).expanduser().resolve())


def pub_uri(uri: str) -> str:
    """The PUB socket beside control URI *uri*: ``<path>.pub`` for ipc, port + 1 for tcp."""
    if uri.startswith("tcp://"):
        host, _, port = uri[len("tcp://") :].rpartition(":")
        return f"tcp://{host}:{int(port) + 1}"
    return uri + ".pub"


def bind_uri(value: str | None, logger: logging.Logger | None = None) -> str | None:
    """The URI ``roqsim sim`` serves: *value* (``--control``), else ``ROQSIM_CONTROL``, else the
    default in the run directory. ``None`` when either says :data:`NONE`.

    A default whose path is too long for a Unix socket (a deep run directory) moves to
    :func:`runtime_dir`, with a warning naming both -- the printed ``control:`` line and the
    registration carry the address it moved to. A path given explicitly is refused instead.
    """
    explicit = value if value is not None else os.environ.get(ENV) or None
    uri = normalize(explicit) if explicit is not None else default_uri()
    if uri is None or not uri.startswith("ipc://"):
        return uri
    for path in (uri[len("ipc://") :], pub_uri(uri)[len("ipc://") :]):
        if len(path.encode()) > IPC_PATH_MAX:
            if explicit is not None:
                raise ValueError(
                    f"{uri}: a Unix socket path is at most {IPC_PATH_MAX} bytes and {path!r} is "
                    f"{len(path.encode())}. Name a shorter path, or a tcp:// address."
                )
            moved = "ipc://" + str(runtime_dir() / f"{os.getpid()}.sock")
            (logger or log).warning(
                "control: the run directory's socket path %s is too long for a Unix socket; "
                "serving at %s instead",
                uri,
                moved,
            )
            runtime_dir().mkdir(parents=True, exist_ok=True)
            return moved
    return uri


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running() -> list[dict]:
    """The simulators running for this user, oldest first. Registrations of dead processes are
    removed on the way."""
    found = []
    directory = runtime_dir()
    if not directory.is_dir():
        return found
    for path in directory.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            pid = int(entry["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _alive(pid):
            path.unlink(missing_ok=True)
            continue
        found.append(entry)
    return sorted(found, key=lambda e: e.get("started", 0.0))


def register(uri: str, pub: str, world: str) -> Path:
    """Record this process as serving *uri*; returns the file to :func:`unregister`.

    Refuses when another live simulator already registered *uri*: binding it would take the
    address from under that one, and its clients would silently start talking to this one.
    """
    for entry in running():
        if entry.get("uri") == uri and int(entry["pid"]) != os.getpid():
            raise RuntimeError(
                f"{uri} is already served by another simulator (pid {entry['pid']}, "
                f"{entry.get('world') or 'no world named'}). Stop it, or serve this run elsewhere "
                f"with --control <uri> (or --control none)."
            )
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{os.getpid()}.json"
    entry = {
        "pid": os.getpid(),
        "uri": uri,
        "pub": pub,
        "world": world,
        "cwd": str(Path.cwd()),
        "started": time.time(),
    }
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def unregister(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def discover(explicit: str | None = None) -> str:
    """The control URI a client should talk to. See the module docstring for the order."""
    if explicit:
        uri = normalize(explicit)
        if uri is None:
            raise ControlError("control address 'none' names no simulator", "not_found")
        return uri
    env = os.environ.get(ENV)
    if env and env.strip().lower() != NONE:
        return normalize(env)
    live = running()
    local = "ipc://" + str((run_dir() / SOCKET_NAME).resolve())
    if any(entry.get("uri") == local for entry in live):
        # Only a registered one: a process killed outright leaves its socket file behind.
        return local
    if len(live) == 1:
        return live[0]["uri"]
    if not live:
        raise ControlError(
            f"no running simulator found: none registered in {runtime_dir()}, no {SOCKET_NAME} "
            f"in {run_dir()}, and {ENV} is not set. Start one with `roqsim sim <world>`, or name "
            "its address with --control.",
            "not_found",
        )
    listed = "; ".join(f"{e['uri']} (pid {e['pid']}, {e.get('world', '')})" for e in live)
    raise ControlError(
        f"{len(live)} simulators are running; name one with --control or {ENV}: {listed}",
        "ambiguous",
    )
