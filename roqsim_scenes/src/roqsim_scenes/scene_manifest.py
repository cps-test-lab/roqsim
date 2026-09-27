"""The scene manifest's stamp: what a ``scene.json`` from an importer says it is.

Every stage-1 importer writes a ``scene.json`` (a name, bounds and the world-space meshes) with
:func:`stamp`, and every reader checks it with :func:`check`. The stamp names the format as well as
its version because the web scene descriptor ``roqsim export web`` writes is also a ``scene.json``.
An unstamped manifest is version 1.
"""

from __future__ import annotations

from roqsim.document import check_version

FORMAT = "roqsim_scenes.scene_manifest"
FORMAT_VERSION = 1


def stamp(manifest: dict) -> dict:
    """*manifest* with ``format`` and ``version`` first, as every writer emits it."""
    rest = {k: v for k, v in manifest.items() if k not in ("format", "version")}
    return {"format": FORMAT, "version": FORMAT_VERSION, **rest}


def check(manifest: dict, where: str) -> int:
    """Return the manifest's version, refusing another format or a newer version by name."""
    fmt = manifest.get("format")
    if fmt is not None and fmt != FORMAT:
        raise ValueError(
            f"{where} is a {fmt!r} document, not a scene manifest ({FORMAT!r}). A scene directory "
            f"holds the scene.json an importer wrote; the web scene descriptor shares the file "
            f"name and is not one."
        )
    return check_version(
        manifest, "version", reads=FORMAT_VERSION, document="scene manifest", where=str(where)
    )
