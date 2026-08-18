# SPDX-License-Identifier: Apache-2.0
"""Recover the floorplan JSON from a scene authored by floorplan_to_world.py -- the reverse step.

``floorplan_to_world.py`` writes the authored floorplan (``{comment, description, rooms, lines, doors,
markers}``) to ``<scene_dir>/floorplan.json`` and stores only a *reference* to it in ``scene.json``
under ``"floorplan"`` (the relative filename). Recovering the floorplan is therefore following that
reference, not a geometry reconstruction -- the baked walls/OBJs alone cannot give the lines/rooms/doors
back (door cuts split one drawn line into several boxes; rooms and door semantics live only in the
floorplan), which is exactly why the file is kept.

Usage::

    roqsim scenes scene-to-floorplan --scene <scene_dir>            # print floorplan JSON to stdout
    roqsim scenes scene-to-floorplan --scene <scene_dir> --out floorplan.json

Fails loudly if the scene was not authored from a floorplan -- there is no lossy fallback that would
silently hand back an approximate, room-less floorplan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def floorplan_of(scene_dir: Path) -> dict:
    """The floorplan JSON for ``scene_dir``: follow ``scene.json``'s ``"floorplan"`` reference.

    ``scene.json["floorplan"]`` is the relative path of the authored floorplan file (written beside
    the manifest). Raises if the manifest is absent, carries no ``floorplan`` reference (the scene was
    not authored from a floorplan), or the referenced file is missing -- per repo policy, fail loud
    rather than return an approximate floorplan.
    """
    manifest_path = scene_dir / "scene.json"
    if not manifest_path.exists():
        raise ValueError(f"{scene_dir} has no scene.json; it is not a roqsim scene directory.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref = manifest.get("floorplan")
    if not isinstance(ref, str):
        raise ValueError(
            f"{manifest_path} has no 'floorplan' reference; the scene was not authored from a "
            f"floorplan, so it cannot be round-tripped to a floorplan JSON."
        )

    floorplan_path = scene_dir / ref
    if not floorplan_path.exists():
        raise ValueError(
            f"{manifest_path} references floorplan '{ref}', but {floorplan_path} does not exist."
        )
    return json.loads(floorplan_path.read_text(encoding="utf-8"))


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Recover the floorplan JSON from a sketch-authored scene."
    )
    ap.add_argument("--scene", required=True, type=Path, help="scene dir holding scene.json")
    ap.add_argument("--out", type=Path, help="write the floorplan JSON here (default: stdout)")
    args = ap.parse_args(argv)

    floorplan = floorplan_of(args.scene)
    text = json.dumps(floorplan, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote floorplan {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
