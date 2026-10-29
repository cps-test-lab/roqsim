# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer server for the out-of-process tests: synthetic images over roqsim's render socket.

    python render_stub_server.py ADDRESS [REPORT.json]

Every image is a function of the request and the snapshot: red is the step modulo 256, green the
length of the camera id, blue a column gradient; depth is 1 m plus a millimetre per step. With
REPORT, what it was sent at open (the cameras, the bodies, the scene's size) is written there.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from roqsim.render_socket import serve_forever


class Stub:
    def __init__(self, report: str | None) -> None:
        self.report = report

    def hello(self) -> dict:
        return {"capabilities": {"images": ["rgb", "depth"]}}

    def open(self, scene, blobs) -> None:
        if self.report:
            glb = blobs.get("scene.glb")
            with open(self.report, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cameras": [c["id"] for c in scene["cameras"]],
                        "bodies": len(scene["body_names"]),
                        "glb_bytes": 0 if glb is None else int(glb.size),
                        "glb_magic": "" if glb is None else bytes(glb[:4]).decode("ascii"),
                    },
                    f,
                )

    def render(self, header, blobs):
        frames, out = [], []
        for r in header["requests"]:
            h, w = r["height"], r["width"]
            if "rgb" in r["images"]:
                rgb = np.empty((h, w, 3), np.uint8)
                rgb[..., 0] = header["step"] % 256
                rgb[..., 1] = len(r["id"]) % 256
                rgb[..., 2] = (np.arange(w) % 256)[None, :]
                out.append((f"{r['id']}/rgb", rgb))
            if "depth" in r["images"]:
                depth = np.full((h, w), 1.0 + 0.001 * header["step"], np.float32)
                out.append((f"{r['id']}/depth", depth))
            frames.append({"id": r["id"], "sim_time": header["sim_time"], "step": header["step"]})
        return frames, out

    def close(self) -> None:
        pass


if __name__ == "__main__":
    serve_forever(sys.argv[1], Stub(sys.argv[2] if len(sys.argv) > 2 else None))
