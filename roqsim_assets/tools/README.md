# Prop import tools

**These files are wrappers.** The tools are subcommands — `roqsim assets --help` lists them, and
`roqsim assets <tool> --help` gives one tool's options, so nothing here needs a path. Each `<name>.py`
beside this README is three lines onto `roqsim_assets.cli.<name>`, kept so a tool still runs when you are
standing in this folder.

They **only produce files** — adding anything to git (and honouring the model's licence) is up to you.

Pipeline (each step is independent; run what you need):

```
roqsim assets sketchfab-helper download  →  roqsim assets reduce-mesh  →  roqsim render  →  roqsim assets inspect-prop
        download + licence                 decimate / join         look at it        sanity + fix origin
```

`roqsim render prop.obj --out prop.png` is the eyeball at any stage — it compiles the mesh into a lit,
grounded preview scene (with its own baseColor texture, if the `.mtl` names one) and writes a PNG. To see
a *finalized* model the way a scene places it — on the floor of a lit room, so an off-origin prop sits
visibly off-centre — render it by name: `roqsim render roqsim_assets:<name> --out check.png`.

## `sketchfab_helper` — search, preview, licence-check, download

All Sketchfab interaction lives in one command (`search` / `thumbs` / `info` / `download`). Its logic
is `roqsim_assets.sketchfab`; unlike the other tools it is installed as a **console command**
(`make venv` / `pip install -e .`), so it runs from the venv PATH with no path prefix.
`sketchfab_helper.py` here is a run-from-the-folder wrapper onto the same entry point, as is
`roqsim assets sketchfab-helper`. Pure stdlib;
`search` / `thumbs` / `info` need no token, `download` does (see below).

```bash
# search — results are ALREADY filtered to redistributable licences (CC0/CC-BY/CC-BY-SA) + downloadable
roqsim assets sketchfab-helper search "office chair" [--count 12]

# thumbnails — up to 12 uids/URLs; downloads previews to a temp dir, prints its path on the last line
roqsim assets sketchfab-helper thumbs <uid1> <uid2> ...

# info — licence verdict + face count, no token, writes nothing (add --out DIR to also drop CREDITS.txt)
roqsim assets sketchfab-helper info <model-url-or-uid>

# download — the glTF, licence-gated; needs a token (see below)
roqsim assets sketchfab-helper download <model-url-or-uid> --out downloads/my_prop

# equivalently, the bare console command installed by `make venv`:
sketchfab_helper search "office chair"
```

`search` prints uid, authoritative licence slug, face count, name and author (best-liked first); only
models importable into this repo are surfaced. `info` prints name / author / licence and a **verdict**:
only **CC0** and **CC-BY / CC-BY-SA** are safe to add to this (Apache-2.0) repo — CC-BY* require
crediting the author (captured in `CREDITS.txt`). CC-*-NC (non-commercial) and CC-*-ND (no-derivatives —
decimation would violate it) are flagged, and `download` refuses them unless you pass `--force` (for
local-only use). Feed a chosen uid to `roqsim assets sketchfab-helper info`, then `roqsim assets sketchfab-helper import` below.

The Sketchfab API token (needed only for `download` / `import`, not search/thumbs/info) is read
automatically from a `SKETCHFAB_API_TOKEN=…` line in the repo's git-ignored `.env` — no `--token` flag
needed. An environment variable or an explicit `--token` still overrides the file. Get a token at
https://sketchfab.com/settings/password.

## One-shot: `roqsim assets sketchfab-helper import`

Runs the whole pipeline with an interactive review loop, staging everything **outside** the asset
library so you judge the real geometry before committing to an import:

```bash
# evaluate quality only — stage + reduce + view, import nothing:
roqsim assets sketchfab-helper import <model-url-or-uid> --blender ~/blender/blender --preview

# evaluate, then get asked whether to import (into the roqsim_assets library by default):
roqsim assets sketchfab-helper import <model-url-or-uid> --blender ~/blender/blender \
    [--name my_prop] [--target-faces 20000] [--scale 1.0]

# import into a different package's models/ dir instead of the shared library:
roqsim assets sketchfab-helper import <uid> --blender ~/blender/blender --models-dir path/to/<pkg>/src/<pkg>/models
```

It **checks the licence permits committing _before_ downloading** (CC0 / CC-BY / CC-BY-SA; otherwise it
stops, unless `--force` for local-only use), writes `CREDITS.txt`, downloads the glTF into a **staging
dir** (a temp dir, or `--stage DIR` to keep it), transcodes textures to PNG, reduces, and renders the
result through `roqsim render --show`, which opens the picture in your image viewer. Answer whether the
reduction is fine (**default yes**); if not, pick a new `target-faces` and it re-reduces + re-renders. It then
finalizes the staged prop (`<name>.xml` + PNG textures) and — unless `--preview` — asks whether to
**import** it. Only on **yes** does the finished prop get copied into `--models-dir` as
`<models-dir>/<name>/` (registering it as a `roqsim.models` provider). A declined or `--preview` run
leaves the staged prop in the scratch dir and touches no package. Needs an image viewer for the review step;
committing the imported files is up to you. The individual steps below are what it calls.

### 1. `roqsim assets sketchfab-helper info` / `download` — licence + download
The download + licence step is the `sketchfab_helper` command documented above (`info` for the
preflight, `download` for the raw glTF). `roqsim assets sketchfab-helper import` does this internally.

### 2. `roqsim assets reduce-mesh` — decimate + join (Blender)
```bash
roqsim assets reduce-mesh --blender ~/blender/blender in.gltf out.obj --target-faces 20000
```
Imports glTF/GLB/OBJ/FBX/STL, **joins all parts into one mesh** (MuJoCo loads a single mesh per OBJ),
collapses to a triangle budget (UVs/materials kept), and writes a triangulated Z-up `.obj`. Already
within budget → **not decimated** (a low-poly model passes through untouched). Blender isn't on `PATH`
as `python`, so its path is the `--blender` argument (the script re-invokes Blender on itself); use
`--scale` if the source isn't in metres.

### 3. `roqsim render` — look at it
```bash
MUJOCO_GL=egl roqsim render out.obj --out preview.png
```
Compiles the mesh into a minimal lit scene and renders a 3/4 view, so you can see whether the decimation
held up and whether the texture is wired. It reports no geometry *facts*: measure those with
`inspect-prop` below, which reads the raw OBJ rather than a compiled model.

### 4. `roqsim assets inspect-prop` — sanity check + fix origin
```bash
roqsim assets inspect-prop models/<name>/              # report (exit non-zero on any FAIL)
roqsim assets inspect-prop models/<name>/ --fix-origin # centre footprint on (0,0), base to z=0
```
The deterministic sanity gate the `model-import` skill drives. Checks origin (footprint centred on the
floor), scale plausibility, up-axis, single-mesh, textured MJCF, single material, licence, and leftover
intermediates. **This is the only place geometry facts come from.** MuJoCo recentres a mesh to its centre
of mass at compile time, so anything measured off a *compiled* model reports a nicely centred box even for
a prop metres off origin; `inspect-prop` reads the raw OBJ (world truth = `raw × scale + geom pos`).
`--fix-origin` bakes the correction into the OBJ vertices. FAIL = must fix; WARN = judge it (see the
skill).

## Notes
- **Textures:** these tools give you clean, correctly-scaled *geometry*, and the preview shows the
  `.mtl`'s `map_Kd` if it names one — a prop that previews flat grey has no albedo bound, which is a
  finding, not a limitation of the preview. Wiring a *surface* texture is a separate step (drop the Color PNG into
  `roqsim_assets/assets/<Name>/` and reference it as `roqsim_assets:<Name>` from a scene) — a
  multi-material atlas would need per-part meshes/materials.
- **Licence hygiene:** keep the `CREDITS.txt` next to any model you add, and record the
  attribution in the relevant package's `THIRD_PARTY.md`.
