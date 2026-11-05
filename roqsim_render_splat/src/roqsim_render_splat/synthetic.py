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

"""A synthetic room as a splat, and its collision model: what the example world and the tests draw.

The room is laid out in the world's frame and metres -- a floor, four walls and a table, each a box
-- and written as a splat in a *capture's* frame: placed by a pose and a scale the way a real
capture is (by default the capture's ``-y`` is up and one unit is half a metre), so that the
example world's ``splat:`` block is exercised as a real one is, and ``roqsim splat frame`` has a
frame to find. The same boxes are written as the world's collision MJCF, so what the robot meets is
what the camera sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ply import Splat, write_splat

#: Degree-0 spherical harmonics' constant: colour = 0.5 + SH_C0 * f_dc.
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the world: centre and half sizes in metres, its colour 0-1."""

    name: str
    centre: tuple[float, float, float]
    half: tuple[float, float, float]
    rgb: tuple[float, float, float]
    #: Whether its surface is checkered, so a camera sees it move.
    checker: bool = False
    #: The faces a capture saw, as ``+x`` .. ``-z`` (the face whose normal points that way); a
    #: capture holds the surfaces facing into the room, never a wall's back or a floor's underside.
    faces: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z")


def example_room() -> list[Box]:
    """A 4 m x 3 m room, 2.4 m high, with a table: the example world's room."""
    grey, wall_a, wall_b = (0.55, 0.55, 0.52), (0.80, 0.78, 0.70), (0.45, 0.62, 0.75)
    return [
        Box("floor", (0.0, 0.0, -0.05), (2.0, 1.5, 0.05), grey, checker=True, faces=("+z",)),
        Box("wall_north", (0.0, 1.55, 1.2), (2.0, 0.05, 1.2), wall_a, faces=("-y",)),
        Box("wall_south", (0.0, -1.55, 1.2), (2.0, 0.05, 1.2), wall_a, faces=("+y",)),
        Box("wall_east", (2.05, 0.0, 1.2), (0.05, 1.5, 1.2), wall_b, faces=("-x",)),
        Box("wall_west", (-2.05, 0.0, 1.2), (0.05, 1.5, 1.2), (0.85, 0.45, 0.30), faces=("+x",)),
        Box("table", (0.9, 0.5, 0.375), (0.45, 0.3, 0.375), (0.35, 0.22, 0.12)),
    ]


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_quat(m: np.ndarray) -> np.ndarray:
    """``w, x, y, z`` of a rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def _surface(box: Box, spacing: float):
    """Points on *box*'s surface, their normals' axis, and their colours."""
    c, h = np.asarray(box.centre), np.asarray(box.half)
    pts, axes, cols = [], [], []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        nu = max(2, int(round(2 * h[u] / spacing)) + 1)
        nv = max(2, int(round(2 * h[v] / spacing)) + 1)
        gu, gv = np.meshgrid(
            np.linspace(-h[u], h[u], nu), np.linspace(-h[v], h[v], nv), indexing="ij"
        )
        for sign in (-1.0, 1.0):
            if f"{'+' if sign > 0 else '-'}{'xyz'[axis]}" not in box.faces:
                continue
            p = np.zeros((gu.size, 3))
            p[:, u], p[:, v] = gu.ravel(), gv.ravel()
            p[:, axis] = sign * h[axis]
            p += c
            col = np.tile(np.asarray(box.rgb, dtype=float), (len(p), 1))
            if box.checker:
                cells = (np.floor(p[:, 0] / 0.5) + np.floor(p[:, 1] / 0.5)).astype(int) % 2
                col[cells == 1] *= 0.6
            pts.append(p)
            axes.append(np.full(len(p), axis))
            cols.append(col)
    return np.concatenate(pts), np.concatenate(axes), np.concatenate(cols)


def room_splat(
    boxes: list[Box],
    pose_quat=(1.0, 0.0, 0.0, 0.0),
    pose_pos=(0.0, 0.0, 0.0),
    scale: float = 1.0,
    spacing: float = 0.08,
) -> Splat:
    """The surfaces of *boxes* as flat Gaussians, in the frame of a capture placed at *pose_pos*,
    *pose_quat* (``w, x, y, z``) with *scale* metres per unit."""
    rot = _quat_matrix(pose_quat)
    pts, axes, cols = [], [], []
    for box in boxes:
        p, a, c = _surface(box, spacing)
        pts.append(p)
        axes.append(a)
        cols.append(c)
    world, axis, colour = np.concatenate(pts), np.concatenate(axes), np.concatenate(cols)
    means = (rot.T @ (world - np.asarray(pose_pos)).T).T / scale
    # Each Gaussian is a disc in its face's plane: wide along the face, thin along the normal,
    # oriented in the world (identity) and so by the inverse of the pose in the capture.
    sigma = np.full((len(world), 3), 0.6 * spacing)
    sigma[np.arange(len(world)), axis] = 0.004
    quat = _matrix_quat(rot.T)
    return Splat(
        means=means.astype(np.float32),
        quats=np.tile(quat, (len(world), 1)).astype(np.float32),
        log_scales=np.log(sigma / scale).astype(np.float32),
        opacity_logits=np.full(len(world), np.log(0.97 / 0.03), dtype=np.float32),
        sh=((colour - 0.5) / SH_C0).astype(np.float32)[:, None, :],
        sh_degree=0,
    )


def collision_mjcf(boxes: list[Box], name: str) -> str:
    """The MJCF of *boxes* as the world's static collision geometry (the floor as a plane)."""
    geoms = []
    for box in boxes:
        if box.name == "floor":
            geoms.append(
                '    <geom name="floor" type="plane" size="0 0 0.05" rgba=".55 .55 .52 1"/>'
            )
            continue
        pos = " ".join(f"{v:g}" for v in box.centre)
        size = " ".join(f"{v:g}" for v in box.half)
        rgba = " ".join(f"{v:g}" for v in box.rgb) + " 1"
        geoms.append(
            f'    <geom name="{box.name}" type="box" pos="{pos}" size="{size}" rgba="{rgba}"/>'
        )
    return (
        f'<mujoco model="{name}">\n'
        "  <!-- What the robot collides with. The splat renderer does not draw it; MuJoCo's does. -->\n"
        "  <worldbody>\n"
        '    <light pos="0 0 2.3" dir="0 0 -1" diffuse=".7 .7 .7"/>\n'
        + "\n".join(geoms)
        + "\n  </worldbody>\n</mujoco>\n"
    )


def write_example(directory: str | Path, pose_quat, pose_pos, scale: float) -> tuple[Path, Path]:
    """Write the example room's ``room.ply`` and ``collision.xml`` into *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    boxes = example_room()
    ply = write_splat(directory / "room.ply", room_splat(boxes, pose_quat, pose_pos, scale))
    xml = directory / "collision.xml"
    xml.write_text(collision_mjcf(boxes, "splat_room_example"))
    return ply, xml


#: Where the example world places its synthetic capture: the capture's -y up, a 20-degree yaw, one
#: unit half a metre, as ``worlds/splat_room_example.yaml`` states it.
EXAMPLE_POSE = {"position": (0.0, 0.0, 1.0), "rpy": (-1.5708, 0.0, 0.35), "scale": 0.5}


def write_shipped_example(directory: str | Path) -> tuple[Path, Path]:
    """The example world's ``room.ply`` and ``collision.xml``, as the package ships them."""
    from roqsim.pose import rpy_to_quat  # noqa: PLC0415

    return write_example(
        directory,
        rpy_to_quat(*EXAMPLE_POSE["rpy"]),
        EXAMPLE_POSE["position"],
        EXAMPLE_POSE["scale"],
    )
