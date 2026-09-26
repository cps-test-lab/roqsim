"""The scene manifest's format stamp: what a ``scene.json`` from an importer says it is.

Every stage-1 importer (``floorplan-to-world``, ``sdf-to-scene``, ``jsonld-to-scene``,
``usd-to-scene``) writes a ``scene.json`` -- a name, bounds and the list of world-space meshes -- and
the bake (``scene-to-mjcf``) reads it. :func:`stamp` is how a writer marks one and :func:`check` is
how a reader refuses one it cannot read.

The stamp names the format as well as its version because another ``scene.json`` exists: the web
scene descriptor ``roqsim export web`` writes, which is ``roqsim.web_scene``. A reader handed the
wrong one is told so by name rather than failing on a missing key deep in a bake.

An unstamped manifest is version 1, the layout every manifest had before the stamp existed. The
version is bumped when a key changes meaning, not when one is added.
"""

from __future__ import annotations

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
    version = manifest.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"{where}: scene manifest version {version!r} is not a positive integer")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{where} is scene manifest version {version}; this roqsim_scenes reads up to "
            f"{FORMAT_VERSION}. Bake it with the version that wrote it, or re-import the scene."
        )
    return version
