"""Texture discovery: resolve a plugin's ``texture:`` string to a PNG file.

Kept deliberately simple and **explicit** -- two forms, no cross-package name search (which would let
two packages ship the same name and collide):

1. **Package-qualified ``<package>:<name>``** (``roqsim_assets:Concrete030``) -> the texture
   ``<name>`` inside ``<package>``'s ``ASSETS_DIR`` (``assets/<name>/`` holding a single Color PNG).
   ``<package>`` is any importable module exposing an ``ASSETS_DIR`` attribute -- typically the shared
   :mod:`roqsim_assets`, but a package can point at its own module just as well.
2. **Filesystem path** (absolute, relative to ``base_dir``, or a path-like ending in ``.png``).

A bare name with no ``:`` and no path separator is rejected with a hint to qualify it -- so which
package a texture comes from is always spelled out in the world YAML.
"""

from __future__ import annotations

import os
import tempfile
from importlib import import_module
from pathlib import Path

import yaml


class TextureError(Exception):
    """Raised when a ``texture:`` reference cannot be resolved to a PNG file."""


class UVScaler:
    """Produces UV-scaled OBJ copies so a texture tiles at a chosen real-world scale on a **mesh**.

    MuJoCo maps a mesh that has **baked UV coordinates** by those coordinates and *ignores* the
    material's ``texrepeat``/``texuniform`` -- so for such a mesh the only way to change tile scale is
    to scale the UVs. This writes a copy of the OBJ with every ``vt`` multiplied by ``scale`` into a
    temp dir kept alive for the caller's lifetime (MuJoCo reads mesh files at ``spec.compile()`` time).

    A mesh **without** baked UVs (an ``.stl``, or an ``.obj`` with no ``vt`` lines) is left untouched --
    MuJoCo auto-projects it and the material's ``texuniform``+``texrepeat`` set the scale -- so
    :meth:`scaled` returns the original path. Hold one instance on the plugin across its ``build``.
    """

    def __init__(self, prefix: str = "roqsim_uv_") -> None:
        self._prefix = prefix
        self._tmp: tempfile.TemporaryDirectory | None = None

    def scaled(self, mesh_path: str | Path, scale: float, name: str) -> str:
        """Path to a copy of ``mesh_path`` with UVs scaled by ``scale`` (the original if not applicable)."""
        mesh_path = str(mesh_path)
        if scale == 1.0 or not mesh_path.lower().endswith(".obj"):
            return mesh_path
        with open(mesh_path) as fh:
            lines = fh.readlines()
        if not any(line.startswith("vt ") for line in lines):
            return mesh_path  # UV-less OBJ: auto-projected, texrepeat handles the scale
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix=self._prefix)
        out = os.path.join(self._tmp.name, f"{name}.obj")
        with open(out, "w") as dst:
            for line in lines:
                if line.startswith("vt "):
                    p = line.split()
                    dst.write(f"vt {float(p[1]) * scale:.6f} {float(p[2]) * scale:.6f}\n")
                else:
                    dst.write(line)
        return out


def _png_in(folder: Path) -> Path | None:
    """The single Color PNG inside a texture folder, or None (absent / not exactly one)."""
    if folder.is_dir():
        pngs = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".png")
        if len(pngs) == 1:
            return pngs[0]
    return None


def resolve_texture(texture: str, base_dir: Path | None = None) -> Path:
    """Resolve a ``texture:`` reference to an absolute PNG path. See the module docstring for forms."""
    # 1) filesystem path: absolute, relative to base_dir, or a path-like ending in .png that exists.
    p = Path(texture)
    if p.is_absolute():
        if p.exists():
            return p
        raise TextureError(f"texture path {texture!r} does not exist")
    if base_dir is not None:
        rel = Path(base_dir) / texture
        if rel.exists():
            return rel
    if texture.lower().endswith(".png") or "/" in texture:
        if p.exists():
            return p
        raise TextureError(f"texture path {texture!r} does not exist")

    # 2) package-qualified "<package>:<name>": <name> in <package>'s ASSETS_DIR.
    if ":" in texture:
        package, _, name = texture.partition(":")
        try:
            module = import_module(package)
        except ImportError as exc:
            raise TextureError(
                f"texture provider {package!r} (from {texture!r}) is not an importable package: {exc}"
            ) from exc
        assets_dir = getattr(module, "ASSETS_DIR", None)
        if not assets_dir:
            raise TextureError(f"texture provider {package!r} exposes no ASSETS_DIR")
        png = _png_in(Path(assets_dir) / name)
        if png is not None:
            return png
        raise TextureError(f"texture {name!r} not found in provider {package!r} ({assets_dir})")

    raise TextureError(
        f"texture {texture!r} must be package-qualified ('<package>:{texture}', e.g. "
        f"'roqsim_assets:{texture}') or a PNG path"
    )


def texture_manifest(texture_png: Path | str) -> dict:
    """Load the optional ``manifest.yaml`` next to a texture PNG ({} if absent or unreadable)."""
    path = Path(texture_png).parent / "manifest.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
