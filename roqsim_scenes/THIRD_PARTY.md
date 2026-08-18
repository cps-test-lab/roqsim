# Third-party assets & provenance

This package vendors one imported environment: the **Depot** warehouse, brought in from Gazebo Fuel.

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
