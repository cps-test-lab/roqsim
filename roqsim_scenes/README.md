# roqsim_scenes

MuJoCo scene worlds imported from CAD, USD or Gazebo SDF and baked into a **plain MJCF** you can load
with roqsim *or* any MuJoCo tool. Ships the **Depot** warehouse (from Gazebo Fuel, CC-BY-4.0).

There is **no runtime plugin**. An offline pipeline turns a source scene into a committed `.xml` world;
roqsim loads it via `sim.world` (which now accepts an MJCF file path), and so does
`python -m mujoco.viewer --mjcf …`.

A world can be baked from an **already-materialled MJCF building** — an export whose semantic
materials (glass, timber frames, concrete, terrazzo) and lighting are baked into
Every stage named below is a subcommand of `roqsim`: the module `foo_bar.py` runs as `roqsim scenes foo-bar`,
from any directory, and `roqsim scenes --help` lists all of them with one line each.

`mjcf_to_world.py` keeps that look verbatim and only adds what a runnable world needs:

```
path/to/building.xml                                 ← source building shell (an MJCF export)
mjcf_to_world.py  (+ ground plane, props, assets/)   ← venv (has mujoco)
worlds/<name>/<name>.xml + assets/                   ← the world (a normal MuJoCo model)
```

The `scene.json` route below is the **general import path** (USD/SDF → per-object OBJ + `scene.json` →
`scene_to_mjcf.py`), and is how the committed `depot` world was produced:

```
usd_to_scene.py   (USD → per-object OBJ + scene.json)      ← Blender (has USD import)
sdf_to_scene.py   (Gazebo/Ignition SDF → the same)         ← venv (fetches + pins Fuel models)
scene_to_mjcf.py  (scene.json + scene.yaml → <name>.xml)   ← venv (has mujoco)
```

A world that is **generated rather than authored** takes a different route. Its input is a 2D
occupancy grid — no source file, no meshes to convert, pin or hull — and there are two ways down from
it, because "occupancy grid" covers two different things:

```
gridmap_to_world.py      (grid → world YAML, one prop per cell, + map.pgm/map.yaml)   ← venv
gridmap_to_floorplan.py  (grid → floorplan.json → floorplan_to_world.py)              ← venv
```

Take the **world** route when the occupied cells are *obstacles*: a procedural field, cellular
automata, a random fill, scattered posts. Each cell becomes one parametric `roqsim_assets` prop
(`cylinder` or `box`) declared in the world YAML, so the obstacles stay out of the baked scene and a
campaign can vary the field as a factor without re-baking anything. It writes the Nav2 grid itself
rather than handing off to `scene_to_map.py`: there the map is *projected* out of a baked scene at
some scan height, whereas here the grid **is** the primary artifact, so world and map are
co-registered by construction with nothing to slice. `--merge` (box only) covers the cells with as
few rectangles as possible — exact, so the map is unaffected; it is refused for cylinders, whose gaps
between tangent posts are the point.

Take the **floorplan** route when the occupied cells are *architecture*: a downsampled building map,
a maze. One prop per cell is then the wrong representation however well it merges — a 40×40 building
map is 199 occupied cells and **7 wall lines** — and the floorplan gives you what the cell field
cannot express: walls with real thickness, door openings with lintels, an optional ceiling, and a
scene that round-trips back through `scene_to_floorplan.py`. The tool refuses grids that are not
walls rather than inventing them: it reports how much of the emitted wall stands over cells that were
*free*, which stays near zero on real walls and dominates on a scattered field (measured: 0% for a
building map, 9% for a maze, 42–51% for random fills).

A world that was **published only as a picture** — a paper's RViz occupancy grid or top view, with no
SDF and no `.pgm` — has one more front-end ahead of the human sketch route:

```
mapimage_to_floorplan.py  (map screenshot → floorplan.json)   ← venv (Pillow)
floorplan_to_world.py     (floorplan.json → world + scene)    ← venv
```

It emits exactly the `floorplan.json` the scene-builder's sketch window returns, so the generator
downstream cannot tell a traced plan from a drawn one. Everything it produces is a *measurement of a
figure*: `--scale` (pixels per metre) is required because there is no way to guess it, only
axis-aligned walls come out, and the debug overlay it writes has to be checked against the source
before anything is built on the result. It shares its segmentation with `gridmap_to_floorplan.py` —
`grid_to_floorplan.py` holds everything between a boolean grid and the floorplan schema, so the image
tool is only the part that gets a picture as far as a grid.

Why per-object (vs. one merged mesh): MuJoCo collides a mesh by its **convex hull**, so a single
merged building mesh hulls into a solid block filling the interior. Keeping the source's separate
objects means each wall/column/desk becomes its own **convex collider** — the scene is directly
collidable, no external collider file.

USD hands that split over for free (one prim per object). **SDF does not**, and `sdf_to_scene.py`
exists mostly to recover it. Three rules there are load-bearing — each was a bug that reached a
running world, and all three look identical from the outside ("the robot sinks into the floor"),
because the MJCF loads and steps happily in every case:

- **Split connected components, then cut at reflex edges.** Components alone are not enough: a
  building's walls and roof are usually *welded into one component*, whose hull is the solid brick
  above. The reflex-edge cut (`scene_mesh_io.split_convex_parts`) recovers the individual slabs;
  genuinely convex geometry passes through as one piece.
- **Take the ground height from the source's `<plane>`, not from the scene bounds.** A world that puts
  its ground at z=0 and drops the building to z=-0.1 (Gazebo's Warehouse) leaves an outdoor apron as
  the lowest geometry. Guessing from the bounds sinks the collision floor below the visible one. The
  importer records it as `ground_z` in `scene.json`; `scene.yaml` still overrides.
- **Honour `<visual>` vs `<collision>`, and `<submesh>`.** Collision geometry is collided but not
  rendered (group 3) when a distinct visual exists — it is a stand-in shape, not something to look at.
  Ignoring `<submesh>` re-loads a whole shared mesh file where the SDF asked for one named node.

Meshes keep the source's **UVs and diffuse textures** (one submesh per material, since a MuJoCo geom
has exactly one material). Source JPEGs are re-encoded to PNG on the way in — MuJoCo reads PNG only.

## What's here

- `scenes/depot/` — the port's provenance and re-bake recipe: `scene.json`, `depot.sdf`,
  `assets.lock.json` (the pinned Fuel model) and `CREDITS.txt`. The tessellated source meshes and
  textures (~43 MB) are regenerable from that pin and are git-ignored.
- `worlds/depot/depot.xml` + `assets/` — the **baked world**: one geom per object (rendered from true
  triangles, collided by its convex hull), with the scene's own textures. Self-contained (meshes
  copied into `assets/`).
- `worlds/depot.yaml` — a roqsim world whose `sim.world` points at `depot/depot.xml`.
- `src/roqsim_scenes/cli/` — the pipeline, reached as `roqsim scenes <tool>` (`roqsim scenes --help` lists all
  of them): `mjcf-to-world` packages an already-materialled source MJCF into a world, while
  `usd-to-scene` / `sdf-to-scene` / `scene-to-mjcf` are the general import path. `tools/` holds a
  run-from-the-folder wrapper per tool and nothing else.

## Run it

```
# via roqsim (add spawn_robot/spawn_arm plugins to populate the scene)
roqsim sim roqsim_scenes:depot

# or as a plain MuJoCo model, no roqsim
python -m mujoco.viewer --mjcf roqsim_scenes/src/roqsim_scenes/worlds/depot/depot.xml
```

`sim.world` accepts either a built-in name (`empty_room`) or a **path to an MJCF file** (resolved
relative to the world YAML), so the baked scene is loaded as the base world and plugins attach onto it.

## Regenerating the world

A world can be baked from an already-materialled source MJCF with `mjcf_to_world.py` (fast, no
Blender). The look (materials, lights) stays in that source file; the baker only layers on the ground
plane and props:

```
roqsim scenes mjcf-to-world \
    --mjcf path/to/building.xml \
    --prop path/to/prop.obj,12.9,10.4 \
    --out src/roqsim_scenes/worlds/mylab/mylab.xml
```

It adds an invisible collidable ground plane (`--ground-z` to pin the height, `--no-ground-plane` to
skip) sized to the scene, a small ambient headlight fill (`--headlight-ambient`, the source shell ships
no `<visual>`), and any `--prop`. The source's own bodies, materials, lights and `<custom>` metadata
pass through untouched.

### General import path (other scenes)

For a scene imported via `scene.json`, edit the look/collision/lighting in its `scene.yaml` (materials
by name glob, `physical_size` tile scale, `rgba` tint — RGB > 1 brightens, collision, light), then
re-bake with `scene_to_mjcf.py`:

```
roqsim scenes scene-to-mjcf --scene <name> \
    --prop path/to/prop.obj,12.9,10.4
```

`--prop PATH,X,Y[,YAW]` drops a mesh in (footprint centred at X,Y, base on the floor). Textures resolve
via `roqsim.textures` (`roqsim_assets:<Name>` or a PNG path); the meshes carry metre-scale UVs, so
`physical_size` sets true tile scale (the baker scales the UVs, since MuJoCo ignores `texrepeat` on a
UV'd mesh).

**Collision** is convex per object by default (`collision: convex`); walls/columns/doors are near-convex
so their hulls are exact, a concave prop collides as its filled hull. `collision: none` makes the scene
visual/lidar-only (only the ground plane is solid); `collide: false` on an object in `scene.json` opts
one prim out.

## Importing another USD

Stage 1 runs inside a **Blender 4.x** that ships a USD importer. The command finds Blender on `PATH`
and runs the tool inside it, so the incantation is not yours to remember:

```
roqsim scenes usd-to-scene <input.usd> src/roqsim_scenes/scenes/<name> <name> [--unit-scale 0.01]
```

IsaacSim authors in centimetres, so `--unit-scale 0.01` converts to metres. Then bake with
`roqsim scenes scene-to-mjcf --scene <name>`. See `THIRD_PARTY.md` for the depot import's provenance.

### The cluttered variant

A furnished USD variant may reference its furniture as **external
NVIDIA Omniverse cloud payloads** plus a few missing local paths, none bundled — without that asset
library the props don't import (you get the shell at a large import cost), so only the self-contained
shell is importable. To add furniture with a redistributable licence, import CC0/CC-BY props
with `sketchfab_helper import` and drop them in with `--prop`.

## From a 2D CAD drawing (DXF)

### Look at it first: `roqsim-cad-to-png`

A hand-drawn sketch you can convert straight away. A **building's** floorplan cannot: it arrives as a
40-layer architectural drawing where walls, insulation, furniture, dimension chains, axis grids and the
title block are all mixed together, and you have to know which layers carry the walls before converting
anything. `roqsim-cad-to-png` renders such a drawing to a PNG, layer by layer if you like:

```
roqsim-cad-to-png --in plan.dwg --list-layers                     # names + entity counts
roqsim-cad-to-png --in plan.dxf --out plan.png                    # everything, as drawn
roqsim-cad-to-png --in plan.dxf --out walls.png \
    --layers 'A_Beton,A_MW,A_Trockenbau' --no-text --no-hatch  # just the wall candidates
```

`--layers` / `--exclude-layers` take comma-separated shell globs (case-insensitive) against the layer
name; the framing follows the *selected* entities, so dropping the axis grid also crops the image to the
building. Layers the CAD file marks off are skipped (`--include-off-layers` to keep them). Other knobs:
`--width-px` (real output width; height follows the drawing's aspect), `--dpi` (line/text raster
quality), `--colors mono`, `--bg`, `--layout <name>` for a paper-space sheet. A filter that selects
nothing is an error, not a blank image.

Rendering needs the optional extra — `pip install 'roqsim_scenes[preview]'` (ezdxf + matplotlib). That is
the opposite trade-off from the converter below: a preview may approximate curves because it is thrown
away after you look at it, a floorplan may not.

**DWG input** needs an external DWG→DXF converter on `PATH`, since DWG is a proprietary binary format
no Python library reads: LibreDWG's `dwg2dxf` (free; note its AutoCAD 2018/AC1032 support is incomplete
and it rejects some files outright) or the ODA File Converter (proprietary, free download; a Qt app —
run it under `xvfb-run` on a headless machine). With neither installed the tool fails and names both,
rather than reaching for a same-named DXF sitting next to the DWG — that is a *different export*,
possibly of a different revision. Prefer asking for a DXF export; `--keep-dxf PATH` saves the
intermediate one for `roqsim-dxf-to-floorplan`.

### Convert it: `roqsim-dxf-to-floorplan`

Draw the walls in any CAD tool (Fusion, AutoCAD, QCAD…) and export the **sketch as DXF** — not STEP.
STEP exports solid/surface *bodies*, so a sketch-only design comes out empty; DXF keeps the 2D line
geometry. Then `roqsim-dxf-to-floorplan` turns that DXF into a **floorplan JSON** — the same
`{comment, rooms, lines, doors, markers}` structure the scene-builder's 2D window returns and
`floorplan_to_world.py` consumes:

```
roqsim-dxf-to-floorplan --dxf drawing.dxf --out scenes/<name>/floorplan.json
```

It reads the `LINE` and (straight) `LWPOLYLINE` segments, scales to metres from the DXF's `$INSUNITS`
header (override with `--scale <metres-per-unit>`), moves the drawing's bounding box to the origin
(`--no-recenter` to keep source coordinates), and welds near-coincident corners / drops duplicate
segments (`--snap-tol`, `--no-dedup`). It **fails loudly** on curved geometry (arcs, splines, polyline
bulges) rather than straightening a wall behind your back — straighten it in the CAD tool, or extend
the reader onto `ezdxf`.

Rooms and doors are left empty on purpose: open the result in the scene-builder's 2D window
(`sketch_floorplan_by_human`, `initial=…`) to name rooms, add door openings and tweak walls, then run
`floorplan_to_world.py` on the finished `floorplan.json` to bake the world (see the `scene-update`
skill for the full human-in-the-loop flow):

```
roqsim scenes floorplan-to-world \
    --floorplan scenes/<name>/floorplan.json --out-dir scenes/<name> \
    --scene-name <name> --world-out worlds/<name>.yaml \
    --markers-map scenes/<name>/markers.json   # required only if the floorplan has markers (props)
```

### Roofing it: `--ceiling`

By default a generated building has no ceiling — an open plan reads better from above, and that is
what most worlds want. `--ceiling` adds a concrete slab over the whole plan with its **soffit at
`--ceiling-h`** (`meshes/Ceiling.obj`, a `Ceiling` object in `scene.json`) and pulls the bake's
overhead light under it, since a light left above a solid slab leaves the room black. Take the roof
back off at run time with the core `ceiling` plugin, which deletes by height rather than by name:

```yaml
- ceiling: {keep: true, above_z: 2.6}   # --set components.ceiling.keep=false opens the roof
```

List that plugin **last**: it can only remove what the plugins before it built, and what hangs under
a soffit (`ceiling_panels`, `duct`, `strip_light` in `roqsim_assets`) is built by them.

### The look: `--bake-config`

The bake's materials/collision/lighting come from `--bake-config`, else the **scene dir's own
`scene.yaml`**, else the shared `src/roqsim_scenes/cli/floorplan.scene.yaml` (the reference
look, resolved beside the code that reads it so it ships with an installed copy). Give a
building its own `scene.yaml` beside its `floorplan.json` when its surfaces differ from the reference
— carpet instead of plaster underfoot, a raw concrete soffit — instead of repainting the look every
other generated room inherits. It is authored, so a rebake reads it and never overwrites it. A
`materials` entry takes `texture` / `rgba` / `physical_size` / `reflectance` / `emission` (the last is
what keeps a soffit, which faces away from every lamp below it, from rendering near-black).

Each marker becomes a `spawn_model` in the world YAML. `--markers-map` maps a marker id to the model
to place — either a bare name (`"single_bed"`) or `{"model": "single_bed", "yaw_deg": 180}` to also
set the prop's heading (about +Z, 0 = +x, CCW → `spawn_model`'s `rpy`). A heading the human dragged in
the 2D window (the marker's own `yaw_deg`) is honoured too; a `--markers-map` `yaw_deg` overrides it.
An unmapped marker is a hard error — props are never silently dropped.

## Look at a floorplan: `roqsim-floorplan-to-png`

A floorplan is 40 wall segments as numbers; nobody — human or agent — reads a layout out of that.
`roqsim-floorplan-to-png` draws it as a top-down architectural plan with a **metre scale bar**: walls with
their openings cut out, rooms filled, named and measured, markers, optionally the wall-line/door ids you
edit `floorplan.json` by. It works for any building; nothing about a particular one is in it.

The input is whatever you have at hand — the floorplan JSON, the scene dir that references one, a world
YAML, or a `<package>:<world>` ref:

```
roqsim-floorplan-to-png path/to/world.yaml -o plan.png    # a world (follows `extends:`)
roqsim-floorplan-to-png scenes/<name>                                  # a scene dir
roqsim-floorplan-to-png scenes/<name>/floorplan.json --ids             # the JSON, with the ids to edit by
roqsim-floorplan-to-png <src> --scale-bar bottom-left --scale-bar-length 5
roqsim-floorplan-to-png <src> --scale-bar 17,1.0        # the bar at an explicit metre position
roqsim-floorplan-to-png <src> --grid 1 --axes           # metre grid + x/y axes, to read positions off
roqsim-floorplan-to-png <src> --font-scale 2.6 --width-px 2200   # for a slide
roqsim-floorplan-to-png <src> --highlight "hall,workshop" --area-only "4-9" --area-decimals 0 --no-legend
roqsim-floorplan-to-png <src> --room-color 'meeting room=#d8e8d0' --room-color '7=#f6d6d0'
```

The scale bar is the point of the metre annotation — a PNG has no units, so without it you cannot tell
whether a 0.9 m door or a 2 m robot fits. It is positionable (`--scale-bar` takes a corner, `X,Y` in
world metres, or `none`) because the empty part of a plan differs per building. A corner placement sits
**flush with the plan's own edge** — a bottom corner ends its block (bar + labels) exactly on the
drawing's lowest edge, so the bar belongs to the plan instead of floating in the canvas under it; pass an
explicit `X,Y` when a building fills that corner. The bar stays a slim band as the type grows (only its
labels need the font's space), and `--scale-bar-height M` sets its thickness outright. Other knobs:
`--legend` (`below` by default, or a corner / `none`; `--no-legend` says the same), `--no-rooms` /
`--no-areas` / `--no-markers`, `--area-decimals N` (`0` for whole square metres), `--title TEXT` /
`--no-title`, `--wall-thickness`, `--width-px` (honoured exactly) and `--dpi`. Needs the same `preview`
extra (matplotlib).

**Picking rooms out** — for a slide that argues about one area, or to separate zones:
`--highlight "hall,workshop"` fills those rooms with the accent colour, `--room-color
'ROOM[,ROOM]=COLOUR'` (repeatable) gives each its own, `--area-only "4-9"` labels rooms with their m²
alone (the name goes, the number stays — a room whose identity does not matter to the plan still
contributes its floor area to it), and `--no-label "4-9"` drops their text entirely. A room
is named by its `name` as it reads in the plan (case-insensitively), by its numeric `id`, or by an id
range like `4-9`; a key that matches no room is a hard error listing the ones that exist, because a plan
with nothing highlighted — or everything still labelled — looks just as finished as a correct one. Room
*names* live in `floorplan.json` and can be edited there with no re-bake.

Room labels fit themselves to their room: at slide size a long name is **wrapped** over two or three
lines before the type is shrunk, and only a room too narrow to wrap into (a corridor) gets its label
turned upright, the way a drafted plan sets a corridor's name along it.

**For a presentation:** `--font-scale 2..3` (or `--font-size PT`, the room label's size in points) makes
everything readable from the back of a room. It scales the *point-width strokes* with the text — jambs,
glazing, room outlines, the bar frame — because the walls are metres wide and grow with the plan while
those do not, so text alone at 3x would sit among hairlines. Room labels then shrink to fit their own
room (never below 45 % of the plan's size) and stand upright in a room too narrow to hold them across,
the way a plan sets a corridor's name along it — a 9 m² phone cabin cannot carry the same type as a
165 m² hall.

What fills an opening is **not** in the floorplan — that is the generator's `--doors-map` — so the tool
reads `doors_map.json` beside the floorplan when there is one (`--doors-map` / `--no-doors-map` to
choose) and draws doors, cased doorways and windows as what they will become. Without that map an
opening carrying its own `height_m` is drawn as a neutral *opening*: calling it a window would be a
guess, and a wrong door in a plan misreads as a wrong world. The wall/opening geometry itself comes from
`roqsim_scenes.floorplan_geometry`, the same module `floorplan_to_world.py` bakes from, so the plan cannot
show a building the bake does not build.

**Doors** are drawn the way a technical drawing draws them: the leaf swung open with its quarter-circle
swing arc, at 90° by convention (`--door-swing DEG`, `0` for a closed leaf and no arc). The hinge side
and the direction of swing are not invented — they are the ones the world hangs the door with
(`floorplan_to_world.py`'s `door_placements` → the `door` plugin, whose defaults are `hinge_side: left`,
`swing: 1`), so the arc shows the floor a door really needs and the side it really opens to. A
full-height opening gets no leaf, matching the bare gate the generator leaves in it (`--ceiling-h` says
what counts as full height; default 2.5 m, the generator's default).

A world with a hand-written MJCF (no floorplan behind it) has no plan to draw and says so — it is not
approximated from the baked walls.
