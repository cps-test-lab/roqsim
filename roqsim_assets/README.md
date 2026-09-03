# roqsim_assets

The scenery half of the substrate: one home for what a world is *made of*, so packages share a single
copy instead of each vendoring their own. Three kinds of thing, and a tool group to add more:

- **surface textures** (`assets/<Name>/`) — referenced as `roqsim_assets:<Name>` by any scene or floor
  plugin (the mobile `floorplan`, `roqsim_scenes`'s `static_scene`, …);
- **placeable props** (`models/<name>/`) — resolved by short name and placed from a world YAML via the
  core `spawn_model` plugin;
- **scene plugins** (`box`, `moving_box`, `conveyor`, `door`, … — see [Plugins](#plugins)) — the props
  that are parametric or move, registered under `roqsim.plugins` so any world loads them by name;
- **`roqsim assets`** — the import pipeline that brings an outside model in and makes it scene-ready
  (`roqsim assets --help`; `tools/README.md` walks it step by step).

Robots, sensors and pedestrians are *not* here — they are robot-family packages (`roqsim_mobile`,
`roqsim_sensors`, `roqsim_walker`). What lives here is the environment they act in.

## Layout

Textures live one-per-folder under `assets/<Name>/`:

```
assets/Concrete030/Concrete030_1K-PNG_Color.png   + manifest.yaml
assets/Concrete046/…
assets/WoodFloor051/…
assets/Fabric028/…                                office carpet (ambientCG CC0)
```

Each folder holds a single Color PNG (MuJoCo loads PNG only) and an optional `manifest.yaml` with
surface properties used when a world doesn't set them explicitly:

```yaml
reflectance: 0.10     # MuJoCo material reflectance (0..1)
physical_size: 2.4    # real-world size (m) one image tile spans, so it tiles at true scale
```

## How plugins find these

The package exposes `ASSETS_DIR`. A plugin resolves a `texture:` value through
`roqsim.textures.resolve_texture`, which accepts two **explicit** forms (no cross-package name
search, so two packages can never collide on a shared name):

- **package-qualified** — `roqsim_assets:Concrete030` — the texture inside this package's
  `ASSETS_DIR`;
- **path** — an absolute / relative PNG path.

Add a texture by dropping `assets/<Name>/<Name>_Color.png` (+ optional `manifest.yaml`); it's then
referenceable as `roqsim_assets:<Name>`. Any other package can serve textures the same way — expose
an `ASSETS_DIR` and reference it as `<that_package>:<Name>`.

## Props: reusing another prop's mesh (hierarchical props)

Placeable props live one-per-folder under `models/<name>/<name>.xml` (see `models/__init__.py` and
`tools/README.md` for the import pipeline). A prop that wants another prop's geometry does **not**
copy the mesh — it references it by **relative sibling path**:

```xml
<compiler meshdir="." texturedir="."/>
<asset>
  <mesh name="wheel" file="../trolley_wheel/trolley_wheel_wheel.obj"/>
  <texture name="wheel_color" type="2d" file="../trolley_wheel/textures/wheel_baseColor.png"/>
</asset>
```

This is the sanctioned intra-package composition pattern (worked example: `humanoid_gantry` instances
the `trolley_wheel` caster four times). It resolves in **both** load paths: a standalone compile
(MuJoCo resolves `file=` against `meshdir="."`, the prop's own folder) and the `spawn_model` loader
(`roqsim.models.apply_assets` searches the model's own dir, where the `../` path also resolves).
The donor prop stays a normal standalone asset; keep its geometry facts (dims, origin, orientation)
documented in its MJCF header comment so reusers don't re-measure. Record the provenance link in the
reusing prop's `CREDITS.txt` (point at the donor's `CREDITS.txt`; don't duplicate it).

For borrowing meshes across *packages*, the core offers an `assets:` key in a model's
`.manifest.yaml` (see `roqsim/models.py` docstring) — but note it only resolves through the
spawn/loader path, **not** in a standalone compile (thumbnails, opening the XML directly), so prefer
relative sibling refs whenever donor and reuser live in the same package.

## Plugins

Besides assets, the package ships a few reusable scene plugins (registered in the
`roqsim.plugins` group, so any world loads them by name):

- **`box`** / **`cylinder`** — the two **anonymous** obstacles: a rectangular box (`size`, full
  extents) and a post (`radius` + full `height`). Every other prop here is a specific object; these
  two are shape and position and nothing else, which is what a navigation experiment's scenery
  usually is. Both take `pos` as `[x, y]` to sit on the floor or `[x, y, z]` for an explicit centre,
  plus `color` / `collide` / `friction` / `prefix`; `box` additionally takes `yaw` (a cylinder is
  rotationally symmetric, so it has none), and `cylinder` additionally takes `mass` (kg — unset means
  MuJoCo's default 1000 kg/m³ density, several times too heavy for anything hollow). Declared in the world YAML rather than baked into a scene,
  deliberately: a scene is what an occupancy grid gets generated *from*, so a baked obstacle lands in
  the map — and an experiment about *unknown* obstacles then has none. Round vs square is not
  cosmetic where clearance is the subject: two diagonally adjacent boxes of the cell pitch seal the
  diagonal, two tangent cylinders leave a gap. `box` also takes `free: true`, which adds a free joint
  and registers it as the entity's `base_joint` — the prerequisite for `simulation_interfaces`'
  `SetEntityState`. That is how an obstacle *appears* mid-trial: roqsim never recompiles the model at
  runtime, so a box that must show up on cue is compiled in up front, parked out of the way, and
  teleported into place by the scenario (`on_reset` re-seats it between repetitions).
- **`boxes`** — the same geometry as a **list**, one entry per obstacle
  (`instances: [{pos, size, …}, …]`, each accepting every `box` key). It exists so *how many* is the
  length of a config value rather than the number of plugin entries, which is what makes an obstacle
  population a campaign factor at all: `apply_overrides` resolves a plugin by name and deep-merges,
  and refuses an override matching no plugin — so a campaign can replace `boxes.instances` wholesale
  but could never have appended a fourth `box:` entry. A list rather than a `count:` because real
  populations are heterogeneous: a generator scales the count by path length and gives each obstacle
  its own pose and size, and it — not the substrate — knows the map and the clearance rule. Names are
  `<name>_<index>` unless an entry names itself, so `SetEntityState` can still address one of them.
- **`cylinders`** — the round counterpart of `boxes`, entry for entry (`instances: [{pos, radius,
  height, …}, …]`, each accepting every `cylinder` key). It exists for the same override reason, and
  for one that is specific to round objects: a population of graspable cylinders differs in
  *diameter* at a shared height, and a modelled asset scaled uniformly cannot change one without the
  other — so a family of radii is only expressible as a list of parametric entries.
- **`moving_box`** — the same anonymous box, but **moving**: a mocap body the plugin drives at a
  constant `speed`, either along `waypoints` (fixed route, `loop` / `ping_pong`) or as a seeded
  `random_walk` that ray-casts the compiled model ahead of itself and picks a new heading before it
  hits anything — no map file, no wall list, works in whatever geometry the world contains. Mocap
  rather than free-jointed on purpose: physics must not shove aside the obstacle that exists to
  obstruct, and a commanded motion should define the experiment rather than fight the solver. The
  `random_walk` seed is **required** (an unseeded obstacle is not reproducible) and `on_reset`
  re-seats the box *and* re-seeds it, so repetition N of a campaign cell never inherits N-1's state.
  Use `walker` (in `roqsim_walker`) when the mover should be a pedestrian; use this when the paper says
  "a box crosses the corridor at v m/s".
- **`prop_trajectory`** — an XY **stage** that carries a prop along a path read from a 2-column CSV
  (`path`, `units`, `speed`, `origin`, `start_index` for a phase offset). Where `moving_box` is an
  obstacle that moves, this is a *conveyance*: a carrier plate on two force-driven slide joints, so
  anything resting on it is taken along **by friction** and never teleported — the contact between
  object, plate and a closing gripper stays physical. That distinction is the whole reason it is not a
  `moving_box` mode: **a mocap body cannot carry anything.** MuJoCo integrates a mocap pose
  kinematically and gives it no velocity, so friction against it sees zero relative slip and transfers
  no tangential force — measured, a mocap plate slides 45 mm clean out from under a resting cube while
  touching it the whole way (pinned in `tests/test_prop_trajectory.py`). Mocap *blocks*; a driven joint
  *carries*. Use `conveyor` for a one-axis belt an object rides, and this when the path is 2-D and
  given to you (a dynamic-grasping benchmark's object motion).
- **`conveyor`** — a velocity-driven belt an object rides via a friction pair; live speed over ROS.
  It is a **benchtop** unit (feet at z=0.76): put a table under it with `spawn_model` — the
  `industrial_table` prop is the one the model used to bundle (spawn it at the belt's local
  `[-0.13, 0.6, 0]`), and any other ~0.76 m top (e.g. `desk_diy`, 0.758) does just as well.
- **`shelf`** — a parametric chipboard shelf built from boxes (`layers`/`width`/`depth`/`height`).
- **`palm_tree`** — a parametric artificial palm-like tree: trunk on a pot, a two-tier crown of
  arching fronds, and fruit-bunch clusters with a site at each bunch centre. The library's first
  foliage, and not decoration: a palm crown is a *thin, radially arranged, self-occluding* obstacle
  set with narrow passages between the fronds and the target tucked underneath — a different planning
  problem from boxes and shelves, and the reason the harvesting literature treats palm-like trees
  separately. Frond count/length/width/pitch/droop, the tier offset and the bunch list are all
  config, so a campaign can sweep crown density. Everything collides; nothing here is trim.
- **`door`** — a hinged swing-door leaf (box or the `door` / `door_glass` mesh model) inside a static
  `door_frame` casing, with a force-limited position actuator, placed by opening centre + `hinge_side`.
  Passive (holds its
  `open` angle) or, when `controllable`, an **automatic door**: `std_msgs/Float64` `cmd`/`state`
  topics + a `control_msgs/GripperCommand` action, and it pushes only gently — stalling on an
  obstacle, then giving up. `color` / `frame_color` repaint the leaf and its casing, so a world can put
  its doors in its own trim colour without a recoloured copy of the model — only the leaf's *colliding*
  geometry is painted, leaving non-colliding decoration (the chrome handle) its own finish, and a glazed
  leaf should therefore set `frame_color` alone. `leaf: false` makes it a **cased opening** — a
  *Türblatt*-less door: the casing is welded, nothing hangs in it, and no hinge/actuator/ROS surface is
  created, so a wall gap reads as a doorway instead of a hole while staying a `door` in the floorplan (the
  room loops it belongs to are unchanged) and still registering as a `door` entity. The floorplan generator
  auto-places one per door opening (`floorplan_to_world.py --doors-map`, which carries these colours,
  forwards `leaf`, and takes `skip: true` for an opening something else fills — a **window**).
- **`window`** — a **parametric** fixed window (glazed pane in a slim frame, boxes not meshes) for an
  opening that is glazed rather than hung: `width` / `height` / `depth` / `frame` / colours are config,
  placed by opening centre + wall yaw like `door`. Its defaults line a window up with this library's
  door unit (`height: 2.06` is `door_frame`'s outer casing height); openings are cut floor-up, so it is
  a floor-to-head unit with no sill.
- **`workbench`** — a **parametric** ESD assembly workbench (a MiniTec "Tisch Elektrisch 300": 2.00 ×
  0.70 m melamine worktop, drawer cabinet, perforated tool panel, monitor arm, overhead light frame).
  Parametric above all for its **electric lift column**: `height` is the worktop height, restricted to
  the column's real 0.695 … 0.995 m stroke, so the two catalogue positions are `height: 0.695` and
  `height: 0.995` and a campaign sweeps between them with a plain `ParameterVariationList` — no
  variation plugin and no second model. `width` / `depth` size the top, `cabinet: left|right|none`
  picks the drawer side, `superstructure: false` leaves a bare table. The uprights and light frame are
  floor-referenced; the tool panel and monitor mount ride with the worktop (0.20 m / 0.40 m above it,
  the catalogue's mounting positions), so the reach ergonomics hold at either column setting.
- **`ceiling_panels`** / **`duct`** / **`strip_light`** — what hangs under a soffit: a field of white
  acoustic panels over a rectangle (`area` + `panel` + `pitch`, only panels fully inside the rectangle
  are emitted), a round ventilation run between two points with tee drops into diffusers
  (`start`/`end`/`branches`), and a linear LED batten that can carry a **real light** (`emit: true`),
  so a room is lit from where the light visibly comes from. All three are visual-only and sit above
  head height, which is what lets the core `ceiling` plugin (`roqsim`) delete slab, panels, battens and
  ducts in one go for a top-down view — list `ceiling` *after* them, build order is list order. The
  panels and the deck carry a little `emission`: a ceiling faces away from every lamp in the room, so
  white panels without it render as dark grey slabs.

## Provenance & licences

Each asset folder carries its **own** `CREDITS.txt` (source + licence). Only CC0 / CC-BY(-SA) assets
are committed here; see [THIRD_PARTY.md](THIRD_PARTY.md) for the convention. The bundled textures are
ambientCG / Poly Haven (CC0).
