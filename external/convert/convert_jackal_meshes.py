#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the Clearpath Jackal visual meshes into ``roqsim_mobile`` and verify them.

Source: Clearpath Robotics ``jackal_description`` (BSD-3), ``meshes/*.stl``.
        https://github.com/jackal/jackal @ 4ddf9b578bb7abce1115c8dc59d8b7f86aa9268c

Unlike the Husky (Collada, needing a real conversion), the Jackal ships **binary STL in metres**,
which MuJoCo loads directly. So this script does not convert — it *checks*. That is the point of it:
a silent unit error or a re-authored upstream mesh would shift the robot's hull without anything
failing, and `clearpath_jackal.xml` hard-codes collision primitives derived from these exact bounds.
Every mesh is therefore verified against the extents measured at port time and the script exits
non-zero on any mismatch, rather than copying whatever it finds.

The expected bounds below are in each STL's OWN frame, before the URDF link transform that
`clearpath_jackal.xml` reproduces. They come from the URDF's declared dimensions, independently:

  jackal-base.stl    chassis shell   -> 0.4943 x 0.3376 x 0.1840 m  (URDF chassis box 0.420 x 0.310
                                        x 0.184; the shell is wider than its own collision box)
  jackal-fender.stl  one fender      -> 0.2553 x 0.4300 x 0.0522 m  (fender pair spans the datasheet
                                        hull: 2 x 0.2553 = 0.5106 long, 0.430 wide)
  jackal-wheel.stl   one wheel       -> 0.2000 x 0.1996 x 0.0500 m  (URDF wheel radius 0.098,
                                        width 0.040; the tyre mesh is slightly proud of both)

Usage::

    python external/convert/convert_jackal_meshes.py --src <jackal_description>/meshes
    python external/convert/convert_jackal_meshes.py --src <...>/meshes --check-only
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

# external/ is a sibling of the family packages, so anchor back through parents[2].
DST_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "roqsim_mobile/src/roqsim_mobile/models/clearpath_jackal/meshes"
)

# name -> (expected extents in metres, absolute tolerance in metres)
EXPECTED: dict[str, tuple[tuple[float, float, float], float]] = {
    "jackal-base.stl": ((0.3376, 0.1840, 0.4943), 1e-3),
    "jackal-fender.stl": ((0.2553, 0.4300, 0.0522), 1e-3),
    "jackal-wheel.stl": ((0.2000, 0.1996, 0.0500), 1e-3),
}


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (min, max, n_triangles) of a *binary* STL, in the file's own units."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: too short to be an STL")
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + n * 50 != len(data):
        # An ASCII STL would land here. jackal_description ships binary; refuse rather than guess.
        raise ValueError(
            f"{path}: not a binary STL (header claims {n} triangles, file is {len(data)} bytes)"
        )
    rec = np.dtype([("normal", "<3f4"), ("verts", "<9f4"), ("attr", "<u2")])
    verts = np.frombuffer(data[84:], dtype=rec)["verts"].reshape(-1, 3)
    return verts.min(axis=0), verts.max(axis=0), n


def check(path: Path) -> tuple[float, float, float]:
    lo, hi, n = stl_bounds(path)
    extents = tuple(float(v) for v in (hi - lo))
    want, tol = EXPECTED[path.name]
    bad = [i for i in range(3) if abs(extents[i] - want[i]) > tol]
    axis = "xyz"
    if bad:
        detail = ", ".join(
            f"{axis[i]}: got {extents[i]:.4f} want {want[i]:.4f}" for i in bad
        )
        raise SystemExit(
            f"MESH CHECK FAILED  {path.name}: {detail}\n"
            f"  The model's collision primitives are derived from these bounds. Either the upstream\n"
            f"  mesh changed, or the units are not metres. Do not ship this port until it is resolved."
        )
    print(f"  ok  {path.name:20s} {n:5d} tris  extents {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f} m")
    return extents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="jackal_description/meshes directory")
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT, help=f"destination (default: {DST_DEFAULT})")
    ap.add_argument("--check-only", action="store_true", help="verify without installing")
    args = ap.parse_args()

    missing = [n for n in EXPECTED if not (args.src / n).is_file()]
    if missing:
        raise SystemExit(f"missing in {args.src}: {', '.join(missing)}")

    print(f"checking {len(EXPECTED)} meshes in {args.src}")
    for name in EXPECTED:
        check(args.src / name)

    if args.check_only:
        print("check-only: nothing installed")
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        shutil.copy2(args.src / name, args.dst / name)
    print(f"installed {len(EXPECTED)} meshes -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
