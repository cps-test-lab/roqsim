#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Import the ROS 2 ``turtlebot3_world`` into ``roqsim_scenes`` and bake it to MJCF.

Source: ROBOTIS ``turtlebot3_simulations`` (Apache-2.0), ``turtlebot3_gazebo/worlds`` +
        ``turtlebot3_gazebo/models/turtlebot3_world``.
        https://github.com/ROBOTIS-GIT/turtlebot3_simulations @ the commit pinned below.

The hexagonal room with nine pillars that TurtleBot 3's and nav2's own tutorials navigate. This
script is the scene's re-bake recipe: it fetches the pinned upstream tree, checks the four files it
reads byte for byte, and runs the two import stages with the arguments the port needs. Everything it
produces is regenerable from the pin, which is why only the derived world is committed.

Two of those arguments are decisions, not defaults, and both are argued in
``roqsim_scenes/src/roqsim_scenes/scenes/tb3_world/port_log.md``:

``--ignore-up-axis``
    ``wall.dae`` and ``hexagon.dae`` declare ``<up_axis>Y_UP</up_axis>`` over Z-up geometry. Honouring
    the declaration stands the room's fence up on edge -- 6.6 m tall, 1.27 m thick -- which neither
    the world it is used in nor the occupancy grid ROBOTIS publishes beside it agrees with.

the wall's collision
    a closed hexagonal ring, whose convex hull is the filled room. It survives both graph cuts in the
    importer and is cut instead by its own footprint into 20 convex prisms
    (``scene_mesh_io.split_extruded_shell``), automatically -- there is no flag for it here.

Usage::

    python external/convert/build_tb3_world_scene.py
    python external/convert/build_tb3_world_scene.py --check-only   # verify the pin, import nothing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sources import resolve_source  # noqa: E402  (path set above)

SOURCE_NAME = "turtlebot3_simulations"
SOURCE_REPO = "https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git"
SOURCE_COMMIT = "9be186fb03d84ed4f293e5c0db71d8c05bbc91f3"
SOURCE_SUBDIR = "turtlebot3_gazebo"

#: Every upstream file the import reads, by sha256. A re-authored mesh would move a wall or a pillar
#: without failing anything downstream -- the MJCF would still load, still step, and still look right
#: -- so the pin is checked rather than trusted, exactly as the mesh converters here do.
EXPECTED = {
    "worlds/turtlebot3_world.world": "c9366524b786938956219153997e8569835c8a7777ec5bf219e318fa137021e5",
    "models/turtlebot3_world/model.sdf": "80d2cd15882457b35ed9bbe7fcf70153158bdae4260916c8227e2fe5c624239a",
    "models/turtlebot3_world/meshes/wall.dae": "da086329366d78b9ee73a8dc210232683bdfee93db66bf8ddd14b1bd545cf089",
    "models/turtlebot3_world/meshes/hexagon.dae": "16b1756c50e90c2161f89e2ada1047b69d495889c53522197520116de23a4cda",
}

ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "roqsim_scenes/src/roqsim_scenes/scenes/tb3_world"
WORLD_XML = ROOT / "roqsim_scenes/src/roqsim_scenes/worlds/tb3_world/tb3_world.xml"


def _verify(gz_dir: Path) -> None:
    bad = []
    for rel, want in EXPECTED.items():
        f = gz_dir / rel
        if not f.is_file():
            bad.append(f"{rel}: missing")
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            bad.append(f"{rel}:\n    expected {want}\n    found    {got}")
    if bad:
        raise SystemExit(
            f"{SOURCE_NAME} @ {SOURCE_COMMIT} does not match the pin:\n  "
            + "\n  ".join(bad)
            + "\nThe upstream file changed under a commit that cannot change, or the checkout is "
            "dirty. Do not update the hash without re-reading the port log's measurements: they are "
            "what says this world is the world."
        )
    print(f"pin OK: {len(EXPECTED)} upstream file(s) match {SOURCE_COMMIT[:12]}")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="verify the pin and stop")
    args = ap.parse_args(argv)

    gz = resolve_source(SOURCE_NAME, SOURCE_REPO, SOURCE_COMMIT, subdir=SOURCE_SUBDIR)
    _verify(gz)
    if args.check_only:
        return 0

    from roqsim_scenes.cli import scene_to_mjcf, sdf_to_scene

    # Relative to the repository root, so `scene.json` and `assets.lock.json` record a path that
    # means the same thing in every checkout.
    world = (gz / "worlds/turtlebot3_world.world").relative_to(ROOT)
    rc = sdf_to_scene.main(
        [
            "--world",
            str(world),
            "--out-dir",
            str(SCENE_DIR.relative_to(ROOT)),
            "--scene-name",
            "tb3_world",
            "--lock",
            str((SCENE_DIR / "assets.lock.json").relative_to(ROOT)),
            "--model-path",
            str((gz / "models").relative_to(ROOT)),
            "--ignore-up-axis",
            "wall.dae",
            "--ignore-up-axis",
            "hexagon.dae",
        ]
    )
    if rc:
        return rc
    return scene_to_mjcf.main(
        [
            "--scene",
            str((SCENE_DIR / "scene.json").relative_to(ROOT)),
            "--out",
            str(WORLD_XML.relative_to(ROOT)),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
