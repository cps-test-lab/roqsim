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

"""World plugin: outdoor ground -- a height field, from a DEM or generated.

Everything this substrate could stand a robot on was flat. The quadruped, the Husky, the Jackal and
the Warthog are all outdoor platforms whose papers are about what happens on ground that is not, and
the experiment those papers describe could not be expressed here at all: not "poorly approximated",
not "expressible with effort" -- there was no terrain.

MuJoCo has height fields natively, so this is the plumbing rather than the physics: elevation samples
in, an ``hfield`` geom out, with the two things a world actually needs around it -- it provides the
ground itself (``provides_world``), and it lights it, because a world definition it replaced would
have done both.

**Three sources, one shape.** Elevation arrives as a normalised grid, whatever it came from:

* ``source: generated`` (the default) -- fractal value noise, so a rough world costs no assets and
  no download. ``seed`` makes it reproducible, and ``roughness``/``octaves`` shape it.
* ``source: <path.npy>`` -- a numpy array, which is what a GIS pipeline can always write.
* ``source: <path.png|.tif|.tiff>`` -- a greyscale image, the interchange format every DEM tool
  exports. 16-bit is read at 16-bit: an 8-bit export of a 200 m hill quantises it to 80 cm steps,
  which a legged controller feels as a staircase.

A GeoTIFF is not read here -- geospatial reprojection is a library's job, not a simulator's -- but
the conversion is one command with the tools that own it (``gdal_translate -ot UInt16 -scale dem.tif
dem.png``), and what comes out is the second bullet.

**The vertical scale is stated in metres, not inherited from the file.** ``height:`` is the elevation
of the highest sample above the lowest, and the grid is normalised into it. A DEM carries real
metres, and a world that wants them says so; a world that wants "the same hills, half as steep" -- an
ordinary campaign factor -- changes one number rather than re-exporting a raster. What is refused is
guessing: an image has no unit, and inventing one would put a made-up gradient under every result.

Config::

    heightfield:
      source: generated      # 'generated', or a path to .npy / .png / .tif (relative to the world)
      size: [20.0, 20.0]     # ground extent in metres (x, y)
      height: 1.5            # metres from the lowest sample to the highest
      base: 1.0              # metres of solid below the lowest sample -- a wall, not a shell
      resolution: 96         # samples per side, when generated (a DEM keeps its own)
      seed: 0                # generated terrain is reproducible from this
      roughness: 0.55        # 0..1: how much each finer octave contributes
      octaves: 4             # how many scales of detail
      friction: [1.0, 0.005, 0.0001]   # ground friction, as MuJoCo's three coefficients
      rgba: [0.42, 0.38, 0.30, 1.0]
      light: true            # add a ceiling-height light (the world definition it replaces had one)

**Why it provides the world.** ``sim.world`` builds a floor; a terrain that let one be built would
put a plane through its own hills. Declaring ``provides_world`` is how the ``floorplan`` plugin
already says the same thing, and the engine then skips the world definition and warns if one was also
asked for.

**Contact against a height field is against its triangles**, not a smoothed surface, so the sample
spacing is the resolution of every wheel and foot interaction: 96 samples over 20 m is a 21 cm grid,
which a 10 cm wheel rides as facets. Raise ``resolution`` for a small rough patch rather than for a
large smooth one -- the cost is quadratic and it buys nothing where the ground is flat.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin

_log = logging.getLogger(__name__)

#: Suffixes read as a greyscale elevation image (Pillow), and as a raw array.
_IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".pgm")
_ARRAY_SUFFIXES = (".npy",)


class HeightfieldPlugin(Plugin):
    """See the module docstring."""

    #: It IS the ground, so the engine must not also build one under it.
    provides_world = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.source = str(self.config.get("source", "generated"))
        # Read defensively: a plugin is CONSTRUCTED before its config is validated, so a malformed
        # value must reach `validate_config` as an error message rather than raising here, where the
        # traceback names a line of this file instead of the key the world got wrong.
        size = self.config.get("size") or [20.0, 20.0]
        size = list(size) if len(list(size)) == 2 else [20.0, 20.0]
        self.size_x, self.size_y = float(size[0]), float(size[1])
        self.height = float(self.config.get("height", 1.5))
        self.base = float(self.config.get("base", 1.0))
        self.resolution = int(self.config.get("resolution", 96))
        self.seed = int(self.config.get("seed", 0))
        self.roughness = float(self.config.get("roughness", 0.55))
        self.octaves = int(self.config.get("octaves", 4))
        self.friction = [float(v) for v in self.config.get("friction", [1.0, 0.005, 0.0001])]
        self.rgba = [float(v) for v in self.config.get("rgba", [0.42, 0.38, 0.30, 1.0])]
        self.light = bool(self.config.get("light", True))
        #: The normalised grid, kept after build so a test (or a metric) can ask what the ground is.
        self.elevation: np.ndarray | None = None

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if "size" in config and len(config["size"]) != 2:
            errors.append("'size' must be [x, y] in metres")
        for key, minimum in (("height", 0.0), ("base", 0.0)):
            if key in config and float(config[key]) < minimum:
                errors.append(f"'{key}' must be >= {minimum}")
        if int(config.get("resolution", 96)) < 2:
            errors.append("'resolution' must be >= 2 (a height field needs a grid)")
        if int(config.get("octaves", 4)) < 1:
            errors.append("'octaves' must be >= 1")
        if not 0.0 <= float(config.get("roughness", 0.55)) <= 1.0:
            errors.append("'roughness' must be in [0, 1]")
        if "friction" in config and len(config["friction"]) != 3:
            errors.append("'friction' must be MuJoCo's three coefficients [slide, spin, roll]")
        if "rgba" in config and len(config["rgba"]) != 4:
            errors.append("'rgba' must be [r, g, b, a]")
        source = str(config.get("source", "generated"))
        if source != "generated":
            suffix = Path(source).suffix.lower()
            if suffix not in _IMAGE_SUFFIXES + _ARRAY_SUFFIXES:
                errors.append(
                    f"'source' must be 'generated' or a path ending in "
                    f"{', '.join(_IMAGE_SUFFIXES + _ARRAY_SUFFIXES)}; got {source!r}. A GeoTIFF is "
                    f"converted by the tools that own reprojection, e.g. "
                    f"`gdal_translate -ot UInt16 -scale dem.tif dem.png`."
                )
        return errors

    # -- the grid -----------------------------------------------------------------------------

    def _load(self) -> np.ndarray:
        """The elevation grid, normalised to [0, 1] -- the only shape the rest of this knows."""
        if self.source == "generated":
            grid = _fractal_noise(self.resolution, self.octaves, self.roughness, self.seed)
        else:
            path = Path(self.source)
            if not path.is_absolute():
                # Relative to the world file, like every other path a world names.
                path = Path(self.base_dir or ".") / path
            if not path.is_file():
                raise RuntimeError(f"heightfield: elevation source {str(path)!r} not found")
            grid = self._read_file(path)
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim != 2 or min(grid.shape) < 2:
            raise RuntimeError(
                f"heightfield: elevation must be a 2-D grid of at least 2x2, got shape {grid.shape}"
            )
        low, high = float(grid.min()), float(grid.max())
        if high - low < 1e-12:
            # Flat data with a `height` set is a silent no-op: the world looks like a plane and the
            # experiment believes it ran on terrain.
            raise RuntimeError(
                "heightfield: the elevation source is perfectly flat, so this would build a plane "
                "-- check the file, or use sim.world for flat ground on purpose"
            )
        return (grid - low) / (high - low)

    @staticmethod
    def _read_file(path: Path) -> np.ndarray:
        if path.suffix.lower() in _ARRAY_SUFFIXES:
            return np.load(path)
        from PIL import Image

        with Image.open(path) as image:
            # "I;16"/"I" keep 16- and 32-bit samples; converting to "L" first would quantise a DEM
            # to 256 levels, which a 200 m hill turns into 80 cm steps a legged controller feels.
            if image.mode not in ("I", "I;16", "I;16B", "F", "L"):
                image = image.convert("I")
            return np.asarray(image, dtype=np.float64)

    # -- build ---------------------------------------------------------------------------------

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        grid = self._load()
        self.elevation = grid
        nrow, ncol = grid.shape

        hfield = spec.add_hfield()
        hfield.name = f"{self.label}_terrain"
        hfield.nrow = nrow
        hfield.ncol = ncol
        # MuJoCo's size is (radius_x, radius_y, elevation, base): the first two are HALF extents, so
        # a 20 m world is 10 here. Getting that wrong doubles the terrain and is invisible until
        # something drives off the edge of a world twice the size it asked for.
        hfield.size = [self.size_x / 2.0, self.size_y / 2.0, max(self.height, 1e-6), self.base]
        # Row-major, and MuJoCo reads row 0 as -y: an image's first row is its top, so it is flipped
        # here rather than leaving every DEM mirrored about the x axis.
        hfield.userdata = np.flipud(grid).reshape(-1).tolist()

        geom = spec.worldbody.add_geom()
        geom.name = f"{self.label}_ground"
        geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
        geom.hfieldname = hfield.name
        geom.rgba = self.rgba
        geom.friction = self.friction
        # Group 0, like any floor: it is collidable, visible, and seen by every raycast sensor.
        if self.light:
            # The same shape the built-in world definition uses (roqsim/world.py): a top-down light
            # with an ambient term, because MuJoCo's default has ambient 0 and every north-facing
            # slope of a hill then renders near-black.
            light = spec.worldbody.add_light()
            light.pos = [0.0, 0.0, max(self.size_x, self.size_y)]
            light.dir = [0.0, 0.0, -1.0]
            light.ambient = [0.35, 0.35, 0.35]
            light.diffuse = [0.8, 0.8, 0.8]

        _log.info(
            "heightfield[%s]: %dx%d samples over %.1fx%.1f m, %.2f m of relief (%s)",
            self.label,
            nrow,
            ncol,
            self.size_x,
            self.size_y,
            self.height,
            self.source,
        )

    def height_at(self, x: float, y: float) -> float:
        """Ground elevation at a world (x, y), in metres -- nearest sample, no interpolation.

        Public because "how high is the ground here" is what a spawn pose, a goal check and a
        clearance metric all need, and computing it from ``hfield_data`` at every call site is how
        three of them end up disagreeing about the flip and the half-extent.
        """
        if self.elevation is None:
            raise RuntimeError("heightfield: no elevation grid yet (called before build)")
        nrow, ncol = self.elevation.shape
        col = int(round((x / self.size_x + 0.5) * (ncol - 1)))
        row = int(round((y / self.size_y + 0.5) * (nrow - 1)))
        col = min(max(col, 0), ncol - 1)
        row = min(max(row, 0), nrow - 1)
        return float(self.elevation[row, col] * self.height)


def _fractal_noise(size: int, octaves: int, roughness: float, seed: int) -> np.ndarray:
    """Fractal value noise on a ``size x size`` grid, in [0, 1] and reproducible from ``seed``.

    Value noise rather than Perlin: the gradient field Perlin needs buys smoother slopes than ground
    made of MuJoCo triangles can express anyway, and this is fifteen lines of numpy with no table to
    ship. Each octave is a coarse random lattice sampled up to the full grid with bilinear weights,
    so the sum has features at every scale -- which is what makes generated ground look like ground
    rather than like sandpaper.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), dtype=np.float64)
    amplitude, total = 1.0, 0.0
    lattice = 2
    for _ in range(octaves):
        coarse = rng.random((lattice + 1, lattice + 1))
        # Bilinear upsample to the full grid.
        rows = np.linspace(0, lattice, size)
        cols = np.linspace(0, lattice, size)
        r0 = np.floor(rows).astype(int).clip(0, lattice - 1)
        c0 = np.floor(cols).astype(int).clip(0, lattice - 1)
        fr = (rows - r0)[:, None]
        fc = (cols - c0)[None, :]
        top = coarse[r0][:, c0] * (1 - fc) + coarse[r0][:, c0 + 1] * fc
        bottom = coarse[r0 + 1][:, c0] * (1 - fc) + coarse[r0 + 1][:, c0 + 1] * fc
        out += amplitude * (top * (1 - fr) + bottom * fr)
        total += amplitude
        amplitude *= roughness
        lattice *= 2
    out /= total
    return (out - out.min()) / max(out.max() - out.min(), 1e-12)
