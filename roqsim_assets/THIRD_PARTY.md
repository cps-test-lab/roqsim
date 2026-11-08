# Third-party assets & provenance

Every asset here carries **its own licence + attribution in its folder** — a `CREDITS.txt` next to the
files (source, licence, and any attribution requirement). Look there for the terms of a specific asset;
this file names every third-party asset folder, its source and its licence, below.

Only assets whose licence permits redistribution are committed: **CC0** (public domain) or **CC-BY /
CC-BY-SA** (redistributable *with* the attribution recorded in that folder's `CREDITS.txt`). Assets
under non-commercial (CC-*-NC) or no-derivatives (CC-*-ND) terms are **not** added to this repo.

- **Textures** (`assets/<Name>/`) — a single 1K Color/albedo PNG + a `manifest.yaml` (reflectance /
  physical_size) + `CREDITS.txt`. Bundled today are ambientCG and Poly Haven textures (all CC0).
- **Imported models** (`models/<Name>/`) — meshes brought in via `sketchfab_helper download`,
  which writes the `CREDITS.txt` automatically from the source's licence metadata. Under CC BY, the
  licence asks a copy to indicate that it was modified: `import` adds what it changed once it has,
  a model changed further says so by hand, and `inspect-prop` fails one that does not.

When adding an asset, keep its `CREDITS.txt` alongside it, honour whatever that file states, and add
its row below. `tests/test_credits.py` fails for a CC-BY folder this file does not name.

## Models (CC-BY 4.0, attribution required)

Each folder's `CREDITS.txt` carries the attribution and says what roqsim changed.

| Folder | Source | Author | Licence |
| --- | --- | --- | --- |
| `models/door/`, `models/door_frame/`, `models/door_glass/` (its handle) | [Sketchfab: *White and black wooden door low poly*](https://sketchfab.com/3d-models/white-and-black-wooden-door-low-poly-f7e91629cc574a42a60e6c78e6b05e6a) | oobexr | CC-BY 4.0 |
| `models/office_table/` | [Sketchfab: *Office_Table*](https://sketchfab.com/3d-models/office-table-c5e2ec3f92aa4e9fbc1826f5ef0abf28) | Art_bravado | CC-BY 4.0 |
| `models/trolley_wheel/` (and the casters `models/humanoid_gantry/` instances from it) | [Sketchfab: *Hard Surface Study (Trolley Wheel)*](https://sketchfab.com/3d-models/hard-surface-study-trolley-wheel-e9d511f7e4bd407d90c5a40ffe243170) | WizBeatrix | CC-BY 4.0 |

## Textures (CC0 1.0, no attribution required)

| Folder | Source |
| --- | --- |
| `assets/Concrete030/` | [ambientCG: Concrete030](https://ambientcg.com/view?id=Concrete030) |
| `assets/Concrete046/` | [ambientCG: Concrete046](https://ambientcg.com/view?id=Concrete046) |
| `assets/Fabric028/` | [ambientCG: Fabric028](https://ambientcg.com/view?id=Fabric028) |
| `assets/PlasteredWall04/` | [Poly Haven: Plastered Wall 04](https://polyhaven.com/a/plastered_wall_04) |
| `assets/WoodFloor051/` | [ambientCG: WoodFloor051](https://ambientcg.com/view?id=WoodFloor051) |

## What is this package's own (Apache-2.0)

`models/conveyor/`, `models/industrial_table/`, `models/graspable_box/`, `models/graspable_carton/`
and the frame of `models/humanoid_gantry/`: geometry authored for roqsim, each saying so in its
`CREDITS.txt`.
