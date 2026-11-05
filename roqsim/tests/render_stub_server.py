# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer server for the out-of-process tests: synthetic images over roqsim's render socket.

    python render_stub_server.py ADDRESS [REPORT.json] [OPEN_DELAY_S]

Every image is a function of the request and the snapshot: red is the step modulo 256, green the
length of the camera id, blue a column gradient; depth is 1 m plus a millimetre per step. With
REPORT, what it was sent at open (the cameras, the bodies, the scene's size) is written there; with
OPEN_DELAY_S, it takes that long to answer ``open``.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from roqsim.presence import ABSENT_GEOM_GROUP
from roqsim.render_socket import serve_forever


class Stub:
    def __init__(self, report: str | None, open_delay_s: float = 0.0) -> None:
        self.report = report
        self.open_delay_s = open_delay_s

    def hello(self) -> dict:
        return {"capabilities": {"images": ["rgb", "depth"]}}

    def open(self, scene, blobs) -> None:
        # A server that takes long to open.
        time.sleep(self.open_delay_s)
        if self.report:
            glb = blobs.get("scene.glb")
            with open(self.report, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cameras": [c["id"] for c in scene["cameras"]],
                        "intrinsics": {c["id"]: c["intrinsics"] for c in scene["cameras"]},
                        "bodies": len(scene["body_names"]),
                        "glb_bytes": 0 if glb is None else int(glb.size),
                        "glb_magic": "" if glb is None else bytes(glb[:4]).decode("ascii"),
                    },
                    f,
                )

    def render(self, header, blobs):
        if self.report:
            with open(self.report, encoding="utf-8") as f:
                report = json.load(f)
            report["render_blobs"] = sorted(blobs)
            # What it draws: a flex in the absent group or at zero alpha is not drawn.
            hidden = (blobs["model/flex_group"] == ABSENT_GEOM_GROUP) | (
                blobs["model/flex_rgba"][:, 3] == 0.0
            )
            report.setdefault("flexes_drawn", []).append(int((~hidden).sum()))
            report["geomgroups"] = [r["geomgroup"] for r in header["requests"]]
            report["request_intrinsics"] = {r["id"]: r["intrinsics"] for r in header["requests"]}
            with open(self.report, "w", encoding="utf-8") as f:
                json.dump(report, f)
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
    report = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    serve_forever(sys.argv[1], Stub(report, float(sys.argv[3]) if len(sys.argv) > 3 else 0.0))
