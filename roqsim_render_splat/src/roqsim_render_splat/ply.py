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

"""Read and write a 3D Gaussian splat as the standard 3DGS ``.ply`` stores it.

One ``vertex`` element per Gaussian, binary little-endian float32 properties, read **by name**:
exporters differ in the order and in whether they write the unused normals ``nx, ny, nz``.

- ``x, y, z``: the centre;
- ``rot_0..3``: the orientation, a quaternion ``w, x, y, z``, not normalised;
- ``scale_0..2``: the extent along the Gaussian's axes, as ``log`` of the standard deviation;
- ``opacity``: as a logit;
- ``f_dc_0..2``: the colour's spherical-harmonics coefficient of degree 0, per channel;
- ``f_rest_*``: the higher degrees, channel-major (all of red's, then green's, then blue's), 15 per
  channel for degree 3, 8 for 2, 3 for 1, none for 0.

Only the header is read to check a file (:func:`read_header`); :func:`read_splat` reads the arrays
through a memory map. Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The properties every 3DGS splat has.
REQUIRED = (
    ("x", "y", "z"),
    ("f_dc_0", "f_dc_1", "f_dc_2"),
    ("opacity",),
    ("scale_0", "scale_1", "scale_2"),
    ("rot_0", "rot_1", "rot_2", "rot_3"),
)

#: Spherical-harmonics degree by the number of ``f_rest_*`` properties.
REST_TO_DEGREE = {0: 0, 9: 1, 24: 2, 45: 3}

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}

#: The longest header read before a file is called not a PLY.
_MAX_HEADER = 1 << 20


class SplatFileError(ValueError):
    """A file that is not a 3DGS splat, said in terms of what it is and what it lacks."""


@dataclass(frozen=True)
class Header:
    """What a splat's header says: its Gaussians and their properties, in file order."""

    path: Path
    count: int
    properties: tuple[tuple[str, str], ...]
    header_bytes: int
    sh_degree: int

    @property
    def dtype(self) -> np.dtype:
        return np.dtype([(name, _PLY_TYPES[kind]) for name, kind in self.properties])


def read_header(path: str | Path) -> Header:
    """The header of the 3DGS splat at *path*; :class:`SplatFileError` naming what is wrong."""
    path = Path(path)
    with path.open("rb") as f:
        if f.read(4) != b"ply\n":
            raise SplatFileError(f"{path} is not a PLY file (it does not start with 'ply')")
        lines, size = [], 4
        while True:
            line = f.readline()
            size += len(line)
            if not line or size > _MAX_HEADER:
                raise SplatFileError(f"{path}: its PLY header never ends ('end_header' missing)")
            text = line.decode("ascii", "replace").strip()
            if text == "end_header":
                break
            lines.append(text)
    fmt = next((ln.split()[1] for ln in lines if ln.startswith("format ")), None)
    if fmt != "binary_little_endian":
        raise SplatFileError(
            f"{path}: a {fmt or 'format-less'} PLY; a 3DGS splat is binary_little_endian"
        )
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    for ln in lines:
        parts = ln.split()
        if parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append((parts[-1], "list"))
            else:
                elements[-1][2].append((parts[2], parts[1]))
    names = [e[0] for e in elements]
    if names != ["vertex"]:
        what = " and ".join(names) if names else "no elements"
        raise SplatFileError(
            f"{path} holds {what}; a 3DGS splat holds one 'vertex' element, one per Gaussian"
            + (" (this looks like a mesh)" if "face" in names else "")
        )
    _, count, props = elements[0]
    have = {name for name, _ in props}
    lacking = [name for group in REQUIRED for name in group if name not in have]
    if lacking:
        raise SplatFileError(
            f"{path} is not a 3DGS splat: its vertices lack {', '.join(lacking)} "
            f"(it has {', '.join(sorted(have)) or 'no properties'})"
        )
    odd = [f"{name} ({kind})" for name, kind in props if kind not in _PLY_TYPES]
    if odd:
        raise SplatFileError(f"{path}: properties of a type a splat does not use: {', '.join(odd)}")
    rest = sum(1 for name in have if name.startswith("f_rest_"))
    if rest not in REST_TO_DEGREE:
        raise SplatFileError(
            f"{path}: {rest} f_rest_* properties; spherical harmonics of degree 1, 2 or 3 have "
            "9, 24 or 45 (none for degree 0)"
        )
    return Header(path, count, tuple(props), size, REST_TO_DEGREE[rest])


@dataclass
class Splat:
    """A splat's Gaussians as stored (log-scales, logit opacities, unnormalised quaternions)."""

    means: np.ndarray  # (N, 3) float32
    quats: np.ndarray  # (N, 4) float32, w x y z
    log_scales: np.ndarray  # (N, 3) float32
    opacity_logits: np.ndarray  # (N,) float32
    sh: np.ndarray  # (N, K, 3) float32, K = (degree + 1) ** 2
    sh_degree: int

    def __len__(self) -> int:
        return len(self.means)


def read_splat(path: str | Path) -> Splat:
    """Every Gaussian of the splat at *path*.

    Where every property is float32 -- every splat exporter writes it so -- the records are read
    once as one float matrix and a property is a column of it; otherwise property by property.
    """
    header = read_header(path)
    names = [name for name, _ in header.properties]
    if all(_PLY_TYPES[kind] == "<f4" for _, kind in header.properties):
        matrix = np.fromfile(
            header.path, dtype="<f4", offset=header.header_bytes, count=header.count * len(names)
        ).reshape(header.count, len(names))

        def columns(want) -> np.ndarray:
            index = [names.index(n) for n in want]
            if index == list(
                range(index[0], index[0] + len(index))
            ):  # a run: a slice, not a gather
                return np.ascontiguousarray(matrix[:, index[0] : index[0] + len(index)])
            return np.ascontiguousarray(matrix[:, index])
    else:
        records = np.memmap(
            header.path,
            dtype=header.dtype,
            mode="r",
            offset=header.header_bytes,
            shape=(header.count,),
        )

        def columns(want) -> np.ndarray:
            return np.stack([np.asarray(records[n], dtype=np.float32) for n in want], axis=1)

    rest = 3 * ((header.sh_degree + 1) ** 2 - 1)
    sh = np.empty((header.count, (header.sh_degree + 1) ** 2, 3), dtype=np.float32)
    sh[:, 0, :] = columns(("f_dc_0", "f_dc_1", "f_dc_2"))
    if rest:
        flat = columns([f"f_rest_{i}" for i in range(rest)])
        sh[:, 1:, :] = flat.reshape(header.count, 3, rest // 3).transpose(0, 2, 1)
    return Splat(
        means=columns(("x", "y", "z")),
        quats=columns(("rot_0", "rot_1", "rot_2", "rot_3")),
        log_scales=columns(("scale_0", "scale_1", "scale_2")),
        opacity_logits=columns(("opacity",))[:, 0].copy(),
        sh=sh,
        sh_degree=header.sh_degree,
    )


def write_splat(path: str | Path, splat: Splat) -> Path:
    """Write *splat* as a standard 3DGS ``.ply`` (the Inria order, normals included as zeros)."""
    path = Path(path)
    n = len(splat)
    rest = 3 * ((splat.sh_degree + 1) ** 2 - 1)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    out = np.zeros(n, dtype=[(name, "<f4") for name in names])
    for i, axis in enumerate("xyz"):
        out[axis] = splat.means[:, i]
        out[f"scale_{i}"] = splat.log_scales[:, i]
    for i in range(3):
        out[f"f_dc_{i}"] = splat.sh[:, 0, i]
    if rest:
        flat = splat.sh[:, 1:, :].transpose(0, 2, 1).reshape(n, rest)
        for i in range(rest):
            out[f"f_rest_{i}"] = flat[:, i]
    out["opacity"] = splat.opacity_logits
    for i in range(4):
        out[f"rot_{i}"] = splat.quats[:, i]
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        + [f"property float {name}" for name in names]
        + ["end_header", ""]
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.tobytes())
    return path
