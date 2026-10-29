# external/convert — per-model conversion & build scripts

One-off, per-model mesh conversion / model-build scripts live **here**, never inside a
`roqsim_*/tools/` directory. The robot-family and sensor packages ship only the model, meshes,
licence, and runtime policy — they stay clean of rebuild tooling, which is a `external/` concern
(these run alongside the fetched-asset converters driven by `external/external_resources.py` and
`external/external_assets.yaml`).

- **New per-model converter → write it here** and reference it from docs and model comments as
  `external/convert/<script>.py` (paths are relative to the roqsim dir).
- A script that needs to anchor back into a package must go up through `external/`'s parent (a
  sibling of the packages), e.g. `Path(__file__).resolve().parents[2] / "<pkg>/src/..."` — not
  `parent.parent`.
- **Exception — reusable pipeline tool *suites* stay in-package:** `roqsim_assets/tools/` (prop
  import), `roqsim_scenes/tools/` (floorplan to world), `roqsim_walker/tools/import_actor.py`. Only
  single-model, one-off converters live here.

Blender/DAE conversion traps (winding flips, vestigial `<up_axis>`, pycollada as the reliable DAE→OBJ
path) are documented in `dae2obj.py`.
