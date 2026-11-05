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

"""Find a captured room's floor, up axis and walls in its splat, and propose the pose that places it.

A capture comes in a frame of its own: an up axis its software chose (often ``-y``), a floor
anywhere, walls at any angle, units of no size. This reads the opaque Gaussians' centres and
reports what a world needs:

- **the floor**: the largest plane with (nearly) every opaque Gaussian on one side of it, found
  by RANSAC and refitted; its normal, pointing to that side, is **up**, reported as the nearest
  axis of the capture and the tilt from it;
- **the walls**: planes square to the floor; their mean direction (modulo 90 degrees) gives the
  **yaw** that lays the walls along the world's x and y;
- **the extent**: the footprint along the walls and the height above the floor, in capture units
  (1st to 99th percentile across, 99.9th up), and the ceiling where one was captured;
- **the pose** that puts the floor at z = 0, the walls along x and y and the footprint's centre at
  the origin, printed as the block to paste into the world.

The **scale is not guessed**: a capture does not carry it. ``--scale`` states it, ``--known-height``
derives it from a measured floor-to-ceiling height (or the height of the room's tallest captured
surface). Without either the block says ``scale: null``, which the world refuses until it is set.
What it proposes is a proposal: a bowl-shaped floor, a missing ceiling or furniture larger than the
floor make the user correct it, and the numbers it prints are what to check against.

    roqsim splat frame room.ply
    roqsim splat frame room.ply --known-height 2.6 --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .ply import SplatFileError, read_splat

#: Gaussians at least this opaque are what the frame is read from.
SOLID_OPACITY = 0.5
#: The most centres a fit reads (a fixed subsample: the same file gives the same proposal).
MAX_POINTS = 200_000
#: A plane with at least this share of the points off it on one side is a candidate floor.
ONE_SIDED = 0.9


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane ``n . p = d`` through *points*."""
    centre = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centre, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return normal, float(normal @ centre)


def find_planes(points: np.ndarray, threshold: float, count: int = 8, seed: int = 0) -> list[dict]:
    """The *count* largest planes in *points*, by RANSAC, each refitted to its inliers."""
    rng = np.random.default_rng(seed)
    remaining = points
    planes = []
    for _ in range(count):
        if len(remaining) < 100:
            break
        best, best_n = None, 0
        for _ in range(300):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            inliers = np.abs(remaining @ normal - normal @ a) < threshold
            n = int(inliers.sum())
            if n > best_n:
                best, best_n = inliers, n
        if best is None:
            break
        normal, offset = _fit_plane(remaining[best])
        inliers = np.abs(remaining @ normal - offset) < threshold
        normal, offset = _fit_plane(remaining[inliers])
        side = points @ normal - offset
        above = float((side > threshold).mean())
        below = float((side < -threshold).mean())
        if below > above:  # point the normal at the side most points are on
            normal, offset, above, below = -normal, -offset, below, above
        planes.append(
            {
                "normal": normal,
                "offset": offset,
                "inliers": int(inliers.sum()),
                "above": above,
                "below": below,
            }
        )
        remaining = remaining[~inliers]
    return planes


def _axis_name(v: np.ndarray) -> tuple[str, float]:
    """The capture axis nearest *v* (``+x`` .. ``-z``) and the angle between them, in degrees."""
    i = int(np.argmax(np.abs(v)))
    name = ("+" if v[i] > 0 else "-") + "xyz"[i]
    return name, math.degrees(math.acos(min(1.0, abs(float(v[i])))))


def _rotation_to_z(up: np.ndarray) -> np.ndarray:
    """The smallest rotation taking *up* to +z."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(up @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def _rpy(m: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw (fixed-axis XYZ, as a world's pose reads them) of rotation *m*."""
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 1 - 1e-9:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: yaw carries it all
        roll, yaw = 0.0, math.atan2(-m[0, 1], m[1, 1])
    return roll, pitch, yaw


def propose(
    path: str | Path, scale: float | None = None, known_height: float | None = None
) -> dict:
    """What the splat at *path* says about its frame, and the pose that places it."""
    splat = read_splat(path)
    opacity = 1.0 / (1.0 + np.exp(-splat.opacity_logits.astype(np.float64)))
    solid = splat.means[opacity >= SOLID_OPACITY].astype(np.float64)
    if len(solid) < 1000:
        raise SplatFileError(
            f"{path}: only {len(solid)} Gaussians are at least {SOLID_OPACITY} opaque; too few to "
            "find a floor in"
        )
    if len(solid) > MAX_POINTS:
        solid = solid[np.random.default_rng(0).choice(len(solid), MAX_POINTS, replace=False)]
    spread = np.percentile(solid, 99, axis=0) - np.percentile(solid, 1, axis=0)
    threshold = 0.005 * float(np.linalg.norm(spread))
    planes = find_planes(solid, threshold)
    floors = [p for p in planes if p["above"] >= ONE_SIDED * (p["above"] + p["below"])]
    if not floors:
        raise SplatFileError(
            f"{path}: no plane has the room on one side of it; is this a room captured from inside?"
        )
    floor = max(floors, key=lambda p: p["inliers"])
    up = floor["normal"]
    to_z = _rotation_to_z(up)
    # Walls: planes square to the floor; their directions modulo 90 degrees, weighted by size.
    walls = [p for p in planes if abs(float(p["normal"] @ up)) < math.sin(math.radians(10))]
    if walls:
        angles = []
        weights = []
        for wall in walls:
            n = to_z @ wall["normal"]
            angles.append(4.0 * math.atan2(n[1], n[0]))
            weights.append(wall["inliers"])
        mean = math.atan2(
            float(np.average(np.sin(angles), weights=weights)),
            float(np.average(np.cos(angles), weights=weights)),
        )
        wall_angle = mean / 4.0
    else:
        wall_angle = 0.0
    yaw_fix = np.array(
        [
            [math.cos(-wall_angle), -math.sin(-wall_angle), 0.0],
            [math.sin(-wall_angle), math.cos(-wall_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = yaw_fix @ to_z  # capture -> world directions
    levelled = solid @ rot.T
    floor_z = float(floor["offset"])  # up . p = offset on the floor; up maps to +z
    height = levelled[:, 2] - floor_z
    lo = np.percentile(levelled[:, :2], 1, axis=0)
    hi = np.percentile(levelled[:, :2], 99, axis=0)
    centre = (lo + hi) / 2.0
    top = float(np.percentile(height, 99.9))
    ceilings = [
        p
        for p in planes
        if float(p["normal"] @ up) < -math.cos(math.radians(10))
        and p["inliers"] > 0.05 * len(solid)
    ]
    ceiling = None
    if ceilings:
        c = max(ceilings, key=lambda p: p["inliers"])
        ceiling = float(-c["offset"]) - floor_z  # its normal points down: -up . p = offset
    room_height = ceiling if ceiling is not None else top
    if scale is None and known_height is not None:
        scale = float(known_height) / room_height
    axis, tilt = _axis_name(up)
    roll, pitch, yaw = _rpy(rot)
    result = {
        "file": str(path),
        "gaussians": len(splat),
        "solid_gaussians_read": len(solid),
        "up": {
            "capture_axis": axis,
            "tilt_deg": round(tilt, 2),
            "vector": [round(float(v), 5) for v in up],
        },
        "floor_offset_units": round(floor_z, 4),
        "wall_yaw_deg": round(math.degrees(wall_angle), 2),
        "walls_found": len(walls),
        "footprint_units": [round(float(v), 3) for v in hi - lo],
        "height_units": round(top, 3),
        "ceiling_units": None if ceiling is None else round(ceiling, 3),
        "scale": scale,
        "scale_source": "--scale"
        if scale is not None and known_height is None
        else ("--known-height" if known_height is not None else None),
    }
    s = 1.0 if scale is None else scale
    position = [-s * centre[0], -s * centre[1], -s * floor_z]
    result["pose"] = {
        "position": {k: round(float(v), 4) + 0.0 for k, v in zip("xyz", position, strict=True)},
        "orientation": {
            "roll": round(roll, 4) + 0.0,
            "pitch": round(pitch, 4) + 0.0,
            "yaw": round(yaw, 4) + 0.0,
        },
    }
    if scale is not None:
        result["footprint_m"] = [round(float(v) * scale, 3) for v in hi - lo]
        result["height_m"] = round(room_height * scale, 3)
    return result


def splat_block(result: dict, file_ref: str) -> str:
    """The ``splat:`` block of a world, with what it rests on as comments."""
    p, o = result["pose"]["position"], result["pose"]["orientation"]
    scale = result["scale"]
    lines = [
        "splat:",
        f"  file: {file_ref}",
        "  pose:",
        f"    position: {{x: {p['x']}, y: {p['y']}, z: {p['z']}}}  # floor at z = 0, footprint centred",
        f"    orientation: {{roll: {o['roll']}, pitch: {o['pitch']}, yaw: {o['yaw']}}}  "
        f"# up was the capture's {result['up']['capture_axis']} "
        f"(tilted {result['up']['tilt_deg']} deg); walls along x and y",
    ]
    if scale is None:
        lines.append(
            "  scale: null  # UNKNOWN: a capture has no units. Measure something and pass "
            "--known-height or --scale"
        )
        lines.append("  # the position above assumes scale 1; run again with the scale")
    else:
        lines.append(f"  scale: {scale:.6g}  # metres per unit ({result['scale_source']})")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim splat frame",
        description="Find a captured room's floor, up axis, walls and extent in its 3DGS .ply, "
        "and print the `splat:` block that places it in a world.",
    )
    parser.add_argument("ply", help="the capture: a standard 3DGS .ply")
    parser.add_argument("--scale", type=float, help="metres per unit of the capture, if known")
    parser.add_argument(
        "--known-height",
        type=float,
        metavar="M",
        help="the room's floor-to-ceiling height in metres (or of its tallest captured surface), "
        "to derive the scale from",
    )
    parser.add_argument(
        "--file-ref", help="how the world names the file (default: the path as given)"
    )
    parser.add_argument("--json", action="store_true", help="print everything measured as JSON")
    args = parser.parse_args(argv)
    if args.scale is not None and args.known_height is not None:
        parser.error("give --scale or --known-height, not both")
    for name in ("scale", "known_height"):
        value = getattr(args, name)
        if value is not None and not value > 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0, got {value}")
    try:
        result = propose(args.ply, args.scale, args.known_height)
    except FileNotFoundError:
        print(f"roqsim splat frame: no such file: {args.ply}", file=sys.stderr)
        return 2
    except SplatFileError as err:
        print(f"roqsim splat frame: {err}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    up = result["up"]
    print(
        f"{result['gaussians']:,} Gaussians; up is the capture's {up['capture_axis']} "
        f"(tilted {up['tilt_deg']} deg), floor at {result['floor_offset_units']} units along it; "
        f"{result['walls_found']} wall plane(s), {result['wall_yaw_deg']} deg off the capture's axes"
    )
    print(
        f"footprint {result['footprint_units'][0]} x {result['footprint_units'][1]} units, "
        f"{result['height_units']} units high"
        + (
            f", ceiling at {result['ceiling_units']}"
            if result["ceiling_units"]
            else ", no ceiling found"
        )
    )
    if "height_m" in result:
        fp = result["footprint_m"]
        print(f"at {result['scale']:.6g} m/unit: {fp[0]} x {fp[1]} m, {result['height_m']} m high")
    print()
    print(splat_block(result, args.file_ref or args.ply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
