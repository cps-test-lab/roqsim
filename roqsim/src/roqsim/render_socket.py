# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A renderer in another process, behind a socket: the out-of-process side of the renderer contract.

:class:`SocketRenderer` is a client. It stands behind the render service like any renderer, and
forwards the contract over a stream socket to a server that does the drawing -- Isaac Sim, a GPU
splat renderer in its own environment, a stub in a test. roqsim's process imports nothing of the
server's. It is registered as ``socket``; a package for one particular server subclasses it and
declares that server's capabilities.

**Messages.** Each is a 4-byte big-endian length, that many bytes of a UTF-8 JSON header, then the
raw bytes of every array the header lists under ``blobs`` (``name``, ``dtype``, ``shape``), in order.
Arrays travel as their bytes, never through JSON::

    -> hello   {version}                                  <- hello  {capabilities?}
    -> open    {scene: {cameras, body_names, moving, settings}}  + blob scene.glb (optional)
                                                          <- opened
    -> render  {sim_time, step, requests: [{id, width, height, model, fovy, near, far, images,
                pos, quat}]}  + blobs xpos, xquat          <- frames {frames: [{id, sim_time, step}]}
                                                              + blobs <id>/<image> per image
    -> close                                              <- closed
    any reply may instead be  error {message}

A request's ``pos``/``quat`` are the camera's world pose (``quat`` ``w, x, y, z``; MuJoCo's camera
frame, looking down its -z with +y up). The scene, sent once, is the world as glTF 2.0 written by
``roqsim export gltf``'s writer (:mod:`roqsim.export_gltf`), so there is one walk of the scene, and
the body table names the bodies the per-tick ``xpos``/``xquat`` rows belong to.

**Failing loudly.** A server that cannot be reached within ``connect_timeout_s``, one that does not
answer a tick within ``timeout_s``, one that closes the connection or answers ``error``, raises
:class:`~roqsim.render_api.RenderError` for the tick that needed the frames.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .render_api import (
    CAMERA_MODELS,
    IMAGES,
    CameraRequest,
    Frame,
    Renderer,
    RendererCapabilities,
    RenderError,
    SceneDescription,
    Setting,
    StateSnapshot,
)

#: The protocol's version; a server answering ``hello`` with another is refused.
PROTOCOL_VERSION = 1

_HEADER = struct.Struct(">I")


def send_message(sock: socket.socket, header: dict, blobs: Sequence[tuple[str, np.ndarray]] = ()):
    """Send one message: the header, then the bytes of every ``(name, array)`` in *blobs*."""
    arrays = [(name, np.ascontiguousarray(a)) for name, a in blobs]
    header = dict(header)
    header["blobs"] = [
        {"name": name, "dtype": a.dtype.str, "shape": list(a.shape)} for name, a in arrays
    ]
    body = json.dumps(header).encode("utf-8")
    sock.sendall(_HEADER.pack(len(body)) + body)
    for _, a in arrays:
        sock.sendall(memoryview(a).cast("B"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("the peer closed the connection")
        got += k
    return bytes(buf)


def recv_message(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    """Receive one message: its header and its arrays by name."""
    (length,) = _HEADER.unpack(_recv_exactly(sock, _HEADER.size))
    header = json.loads(_recv_exactly(sock, length).decode("utf-8"))
    blobs = {}
    for spec in header.get("blobs", []):
        dtype = np.dtype(spec["dtype"])
        shape = tuple(spec["shape"])
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        blobs[spec["name"]] = np.frombuffer(_recv_exactly(sock, nbytes), dtype=dtype).reshape(shape)
    return header, blobs


def parse_address(address: str) -> tuple[int, object]:
    """``unix:/path`` or ``host:port`` as a socket family and address."""
    if address.startswith("unix:"):
        return socket.AF_UNIX, address[len("unix:") :]
    host, sep, port = address.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"address {address!r}: 'unix:/path/to/socket' or 'host:port'")
    return socket.AF_INET, (host or "127.0.0.1", int(port))


class SocketRenderer(Renderer):
    """A renderer in another process, reached over a socket (see the module docstring).

    The generic client's capabilities are what its settings state (``images``, ``camera_models``,
    ``deterministic``); a package for one server subclasses it and declares them instead.
    """

    capabilities = RendererCapabilities(
        images=frozenset({"rgb"}),
        camera_models=frozenset({"pinhole"}),
        composes_moving_bodies=True,
        draws_static=True,
        batched=True,
        deterministic=False,
        in_process=False,
        decorations=False,
    )
    SETTINGS = {
        "address": Setting(
            None, "where the server listens: 'unix:/path' or 'host:port' (required)"
        ),
        "command": Setting(
            None,
            "a command that starts the server, as a list; '{address}' in it is replaced by the "
            "address. Started at the first frame, stopped when the run ends. Null: the server is "
            "already running",
        ),
        "connect_timeout_s": Setting(30.0, "how long to wait for the server to accept, in s"),
        "timeout_s": Setting(30.0, "how long one tick may take to answer, in s"),
        "scene": Setting(
            "gltf", "what the server is sent once: 'gltf' (the world as GLB) | 'none'"
        ),
        "images": Setting(["rgb"], "the images the server renders (rgb, depth, segmentation)"),
        "camera_models": Setting(["pinhole"], "the camera models it renders (pinhole, ...)"),
        "deterministic": Setting(False, "whether its frame is a function of the state alone"),
    }

    @classmethod
    def capabilities_for(cls, settings: Mapping) -> RendererCapabilities:
        if cls is not SocketRenderer:
            return cls.capabilities
        merged = {k: s.default for k, s in cls.SETTINGS.items()} | dict(settings or {})
        return RendererCapabilities(
            images=frozenset(merged["images"]),
            camera_models=frozenset(merged["camera_models"]),
            composes_moving_bodies=True,
            draws_static=True,
            batched=True,
            deterministic=bool(merged["deterministic"]),
            in_process=False,
            decorations=False,
        )

    @classmethod
    def validate_config(cls, settings) -> list[str]:
        errors = super().validate_config(settings)
        if not isinstance(settings, Mapping):
            return errors
        address = settings.get("address")
        if not address:
            errors.append("'address' is required: 'unix:/path/to/socket' or 'host:port'")
        else:
            try:
                parse_address(str(address))
            except ValueError as err:
                errors.append(f"'address': {err}")
        command = settings.get("command")
        if command is not None and (
            not isinstance(command, list) or not all(isinstance(c, str) for c in command)
        ):
            errors.append("'command' must be a list of strings")
        if settings.get("scene", "gltf") not in ("gltf", "none"):
            errors.append(f"'scene' is 'gltf' or 'none', got {settings['scene']!r}")
        for key, allowed in (("images", IMAGES), ("camera_models", CAMERA_MODELS)):
            value = settings.get(key, [])
            if not isinstance(value, list) or set(value) - set(allowed):
                errors.append(f"'{key}' is a list of {', '.join(allowed)}, got {value!r}")
        for key in ("connect_timeout_s", "timeout_s"):
            value = settings.get(key, 1.0)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"'{key}' must be a number of seconds > 0, got {value!r}")
        return errors

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen | None = None
        self.server_capabilities: dict = {}

    # -- lifecycle -------------------------------------------------------------------------------

    def open(self, scene: SceneDescription) -> None:
        address = str(self.settings["address"])
        command = self.settings.get("command")
        if command:
            self._proc = subprocess.Popen([c.replace("{address}", address) for c in command])
        self._sock = self._connect(address, float(self.settings["connect_timeout_s"]))
        self._sock.settimeout(float(self.settings["timeout_s"]))
        hello = self._call({"op": "hello", "version": PROTOCOL_VERSION}, (), "hello")
        if int(hello.get("version", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
            raise RenderError(
                f"renderer at {address} speaks protocol {hello.get('version')}, "
                f"this client {PROTOCOL_VERSION}"
            )
        self.server_capabilities = hello.get("capabilities") or {}
        blobs = []
        if self.settings.get("scene", "gltf") == "gltf":
            blobs.append(("scene.glb", np.frombuffer(self._scene_glb(scene), dtype=np.uint8)))
        description = {
            "cameras": [
                {
                    "id": c.id,
                    "width": c.width,
                    "height": c.height,
                    "model": c.model,
                    "fovy": c.fovy,
                    "near": c.near,
                    "far": c.far,
                    "images": sorted(c.images),
                }
                for c in scene.cameras
            ],
            "body_names": list(scene.body_names),
            "moving": [bool(m) for m in np.asarray(scene.moving)],
            "settings": {k: v for k, v in scene.settings.items() if k != "command"},
        }
        self._call({"op": "open", "scene": description}, blobs, "opened")

    def _connect(self, address: str, timeout: float) -> socket.socket:
        family, where = parse_address(address)
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RenderError(
                    f"the renderer process exited with status {self._proc.returncode} before it "
                    f"accepted a connection at {address}"
                )
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.connect(where)
                return sock
            except OSError as err:
                sock.close()
                last = err
                time.sleep(0.05)
        raise RenderError(
            f"no renderer accepted a connection at {address} within {timeout:g} s: {last}"
        )

    @staticmethod
    def _scene_glb(scene: SceneDescription) -> bytes:
        """The world as GLB, through the same walk and writer ``roqsim export gltf`` uses."""
        import mujoco  # noqa: PLC0415

        from .export_gltf import export_gltf  # noqa: PLC0415 -- only a renderer that wants it

        model = scene.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scene.glb"
            export_gltf(model, data, out)
            return out.read_bytes()

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError, ConnectionError, ValueError):
                send_message(sock, {"op": "close"})
                recv_message(sock)
            sock.close()
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    # -- per tick --------------------------------------------------------------------------------

    def _call(self, header: dict, blobs, expect: str) -> dict:
        if self._sock is None:
            raise RenderError("the socket renderer is not open")
        try:
            send_message(self._sock, header, blobs)
            reply, self._reply_blobs = recv_message(self._sock)
        except (OSError, ConnectionError, ValueError) as err:
            raise RenderError(
                f"the renderer at {self.settings['address']} failed during {header['op']!r}: {err}"
            ) from err
        if reply.get("op") == "error":
            raise RenderError(f"the renderer at {self.settings['address']}: {reply.get('message')}")
        if reply.get("op") != expect:
            raise RenderError(
                f"the renderer at {self.settings['address']} answered {reply.get('op')!r} to "
                f"{header['op']!r}, expected {expect!r}"
            )
        return reply

    def render(self, snapshot: StateSnapshot, requests: Sequence[CameraRequest]) -> list[Frame]:
        header = {
            "op": "render",
            "sim_time": snapshot.sim_time,
            "step": snapshot.step,
            "requests": [
                {
                    "id": r.id,
                    "width": r.camera.width,
                    "height": r.camera.height,
                    "model": r.camera.model,
                    "fovy": r.fovy,
                    "near": r.camera.near,
                    "far": r.camera.far,
                    "images": sorted(r.images),
                    "pos": [float(v) for v in r.pos],
                    "quat": [float(v) for v in r.quat],
                }
                for r in requests
            ],
        }
        reply = self._call(
            header,
            (("xpos", np.asarray(snapshot.xpos)), ("xquat", np.asarray(snapshot.xquat))),
            "frames",
        )
        frames = []
        answered = {f["id"]: f for f in reply.get("frames", [])}
        for r in requests:
            meta = answered.get(r.id)
            if meta is None:
                raise RenderError(f"the renderer answered no frame for camera {r.id!r}")
            frame = Frame(r.id, float(meta["sim_time"]), int(meta["step"]))
            for kind in r.images:
                image = self._reply_blobs.get(f"{r.id}/{kind}")
                if image is not None:
                    setattr(frame, kind, np.array(image))
            frames.append(frame)
        return frames


def serve_forever(address: str, handler, *, ready=None) -> None:
    """Serve the protocol at *address* with *handler*, one connection at a time, until it closes.

    *handler* is an object with ``hello() -> dict``, ``open(scene, blobs)``, ``render(header,
    blobs) -> (frames, blobs)`` and ``close()``. A server written in another environment needs
    only this module's framing (``send_message``/``recv_message``), which uses the standard library
    and numpy alone.
    """
    family, where = parse_address(address)
    with socket.socket(family, socket.SOCK_STREAM) as server:
        if family == socket.AF_UNIX:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(where)
        server.bind(where)
        server.listen(1)
        if ready is not None:
            ready()
        conn, _ = server.accept()
        with conn:
            while True:
                try:
                    header, blobs = recv_message(conn)
                except ConnectionError:
                    return
                op = header.get("op")
                try:
                    if op == "hello":
                        send_message(
                            conn, {"op": "hello", "version": PROTOCOL_VERSION, **handler.hello()}
                        )
                    elif op == "open":
                        handler.open(header["scene"], blobs)
                        send_message(conn, {"op": "opened"})
                    elif op == "render":
                        frames, out = handler.render(header, blobs)
                        send_message(conn, {"op": "frames", "frames": frames}, out)
                    elif op == "close":
                        handler.close()
                        send_message(conn, {"op": "closed"})
                        return
                    else:
                        send_message(conn, {"op": "error", "message": f"unknown op {op!r}"})
                except Exception as err:  # noqa: BLE001 -- a server reports, the client raises
                    send_message(conn, {"op": "error", "message": f"{type(err).__name__}: {err}"})
