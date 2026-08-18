# SPDX-License-Identifier: Apache-2.0
"""What a tool's ``target`` may name, and the one check worth making before spending a subprocess.

Every tool here takes the same positional argument the ``roqsim`` CLI takes -- a world YAML, a baked
MJCF, a raw mesh, or a ``<pkg>:<name>`` model reference -- and hands it on unresolved. Resolution
belongs to :func:`roqsim.config_for_input`, which is the single dispatch; all this module does is notice
that a *path-shaped* target names a file that is not there, so the error says so up front instead of
arriving as a MuJoCo compile failure from inside a subprocess.

The suffix list mirrors the shapes :func:`roqsim.runner.is_model_ref` recognises as "not a model ref"
(``.yaml``/``.yml``/``.xml`` plus its ``_MESH_EXT``). It is copied rather than imported because roqsim
keeps that set private, inline in the predicate; if roqsim ever publishes it, delete this tuple and use
it. A model reference has no suffix and is deliberately *not* checked here -- only the resolver knows
whether a package provides it.
"""

from __future__ import annotations

from pathlib import Path

#: Suffixes that make a target a path we can cheaply confirm exists.
PATH_SUFFIXES = (".yaml", ".yml", ".xml", ".obj", ".stl", ".glb", ".gltf", ".fbx", ".dae", ".ply")


def is_path_target(target: str) -> bool:
    """True when ``target`` is path-shaped, i.e. its existence is ours to check."""
    return Path(target).suffix.lower() in PATH_SUFFIXES


def require_existing(target: str, label: str = "target") -> None:
    """Fail loudly when a path-shaped ``target`` names no file. Model references are left alone."""
    if is_path_target(target) and not Path(target).is_file():
        raise FileNotFoundError(f"{label}: no such file: {target}")
