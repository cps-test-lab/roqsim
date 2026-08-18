# Third-party assets & provenance

Every asset here carries **its own licence + attribution in its folder** — a `CREDITS.txt` next to the
files (source, licence, and any attribution requirement). Look there for the terms of a specific asset;
this file does not duplicate them.

Only assets whose licence permits redistribution are committed: **CC0** (public domain) or **CC-BY /
CC-BY-SA** (redistributable *with* the attribution recorded in that folder's `CREDITS.txt`). Assets
under non-commercial (CC-*-NC) or no-derivatives (CC-*-ND) terms are **not** added to this repo.

- **Textures** (`assets/<Name>/`) — a single 1K Color/albedo PNG + a `manifest.yaml` (reflectance /
  physical_size) + `CREDITS.txt`. Bundled today are ambientCG and Poly Haven textures (all CC0).
- **Imported models** (`models/<Name>/`) — meshes brought in via `sketchfab_helper download`,
  which writes the `CREDITS.txt` automatically from the source's licence metadata.

When adding an asset, keep its `CREDITS.txt` alongside it and honour whatever that file states.
