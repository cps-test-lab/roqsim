# Third-party assets & provenance

This package vendors two imported environments: the **Depot** warehouse, brought in from Gazebo Fuel,
and the ROS 2 **TurtleBot3 World**, brought in from ROBOTIS `turtlebot3_simulations`.

## Depot warehouse (committed, CC-BY 4.0)

- **Upstream:** Gazebo Fuel — the pinned model version is recorded in
  `scenes/depot/assets.lock.json`, and the URL, author and licence verbatim in
  `scenes/depot/CREDITS.txt`, which is the attribution of record.
- **Licence:** CC-BY 4.0 — **attribution is a condition of redistribution.** Keep that `CREDITS.txt`
  (or an equivalent notice) with the assets or anything built from them.

| Vendored file | Source |
| --- | --- |
| `worlds/depot/depot.xml` + `assets/` | The **baked world**: the Fuel model converted to plain MJCF, one geom per object, with its own textures under `assets/` |
| `scenes/depot/scene.json`, `depot.sdf`, `assets.lock.json` | The port's provenance and re-bake recipe. The tessellated source meshes and textures (~43 MB) are **not** committed -- they are regenerable from the pinned Fuel model, and are git-ignored |

## TurtleBot3 World (committed, Apache-2.0)

- **Upstream:** ROBOTIS `turtlebot3_simulations` on GitHub, pinned by commit and by the sha256 of
  every file the import reads — both in `scenes/tb3_world/CREDITS.txt`, which is the attribution of
  record, and checked by `external/convert/build_tb3_world_scene.py` before it imports anything. The
  world's two Fuel `<include>`s (ground plane, sun; CC0) are pinned in
  `scenes/tb3_world/assets.lock.json`.
- **Licence:** Apache-2.0 — permissive, but it **requires the notice to travel**: keep that
  `CREDITS.txt` with the assets or anything built from them.

| Vendored file | Source |
| --- | --- |
| `worlds/tb3_world/tb3_world.xml` + `assets/` | The **baked world**: the Gazebo model converted to plain MJCF, one geom per object |
| `scenes/tb3_world/scene.json`, `assets.lock.json`, `port_log.md` | The port's provenance, decisions and re-bake recipe. The tessellated source meshes are **not** committed — they regenerate from the pin, and are git-ignored |

## Textures

The floor/wall textures used by generated floorplans are not vendored here -- they come from the
shared `roqsim_assets` package (`roqsim_assets:Concrete030` / `Concrete046` / `PlasteredWall04`,
ambientCG CC0). See that package's `THIRD_PARTY.md`.

## What is this package's own (Apache-2.0)

The importers and bakers (`usd-to-scene`, `sdf-to-scene`, `scene-to-mjcf`, `mjcf-to-world`,
`floorplan-to-world`, `floorplan-to-png`), the floorplan geometry and passability analysis, and the
shared bake look in `cli/floorplan.scene.yaml`.

## Adding a scene

Keep the `CREDITS.txt` the importer writes beside the assets and honour whatever it states. A scene
whose licence is unrecorded, or whose terms do not permit redistribution, does not get committed --
including a converted model of a real building, where permission from whoever owns the building is a
separate question from the licence on the file.
