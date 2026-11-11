Baked lighting
==============

MuJoCo's classic renderer lights a world with a few lamps and hard shadows: a room comes out flat, its
walls evenly bright, nothing darkens under a table, and the default lamp's shadows fall where no real
light would put them. For a world that should look real -- photoreal-ish colour cameras, believable
shadows, soft contact shadows, light bounced off the walls (global illumination), ambient occlusion --
roqsim bakes the light once with Blender's path tracer (Cycles) into light maps, keeps them with the
world, and draws them through the classic renderer. What moves (a robot, a door's leaf) is lit live and
throws its shadow onto the baked surfaces.

Turning it on
-------------

One component, in the world YAML:

.. code-block:: yaml

   components:
     - baked_lighting: {}

then, once, by hand:

.. code-block:: bash

   roqsim bake worlds/my_world.yaml          # on the CPU; --device optix|cuda|hip|metal for a GPU

The bake lands beside the world's MJCF (``<mjcf stem>.bake/``, a few MB) and is committed with the world,
so every checkout and every container loads it as it is: loading a world never bakes. ``roqsim bake
--dry-run`` says what it would bake. It needs Blender 4.2 or later on ``PATH`` (or ``--blender PATH``);
the world itself loads without Blender.

**A missing or stale bake does not stop the world.** It loads unbaked, drawn as without the component,
with a warning that says which -- missing, or stale and what changed (a geom moved, the lights or the
settings differ) -- and gives the exact ``roqsim bake`` command. The warning is a world condition
(``SimContext.conditions``, :doc:`architecture` §3): it is in a recording's provenance, in ``roqsim
render``'s JSON record (``conditions``), in ``roqsim check``'s warnings and in
``roqsim_conditions.json`` in the run's output directory, so an image made unbaked can be told from a
baked one afterwards. A baked load records ``bake_used`` with the bake's digest. ``require: true`` (or
``--set components.baked_lighting.require=true`` for one run) stops the world instead.

What it costs, measured on a furnished room
-------------------------------------------

A room of about 45 m² with 30 props: 32 light maps.

- **Bake**, ``default`` preset, on a 20-core desktop CPU (no GPU): about 4.5 minutes from scratch, less
  when only the lights or the settings changed (the UV layout is reused). An 8-core CPU takes roughly
  twice that (estimated). ``quality: high`` is the full-quality bake, meant for a GPU.
- **On disk**: about 4 MB (3.8). The default cap is 5 MB per world (``max_mb``): every rebake adds its size to
  the history of the repository the world lives in.
- **Load**: five to ten seconds more, composing the textures in memory (about 150 MiB of textures on
  the GPU, against 16 unbaked).
- **Drawing**: on a GPU, the colour camera runs at between half and two thirds of the unbaked world's
  frame rate; most of that is the key light's shadow pass. In software GL (no GPU) set ``moving_shadows: false``
  and the baked world draws at the unbaked world's rate; only what moves loses its shadow.
- **Quality** against the full-quality bake: a mean error of about two 8-bit levels. What the default
  gives up: shadow edges about four times softer on walls and floor, small parts of props less exact,
  bounced colour coarse. Against the photos of the real room it models, it reproduces what the full-quality bake did: the
  walls' brightness against height and the floor's contact shadows come much closer than unbaked.

The keys
--------

``roqsim plugins describe baked_lighting`` lists them from the schema:

.. code-block:: yaml

   - baked_lighting:
       path: ""            # the bake's directory; default <mjcf stem>.bake beside the world's MJCF
                           #   (a world with no MJCF names one)
       require: false      # a missing or stale bake stops the world
       quality: default    # default (CPU-sized) | high (full quality, a GPU, more than 5 MB)
       texels_per_m: {}    # over the preset: {shell: 50, floor: 108, props: 400}
       samples: 0          # Cycles samples per texel over the preset's (default 64, high 512)
       denoise: true       # Open Image Denoise (Blender's own)
       lights: auto        # or a list of lights, below
       emitters: []        # [{match: "*/lamp_diffuser", strength_w_m2: 20.0, colour: [...], display_rgba: [...]}]
       exclude: []         # static geoms (glob on the name) left unbaked, lit live
       occluders: [...]    # the moving geoms (globs) whose shadow is baked; see below
       moving_shadows: true
       max_mb: 5.0

Everything that changes the result is a key, so the world says how it was baked and the bake's
fingerprint sees it; ``roqsim bake``'s flags (``--device``, ``--jobs``, ``--blender``) change none.

What is baked, what moves
-------------------------

- **Baked**: every drawn geom (group below 3) welded to the world, not mocap, not excluded. The shell's
  largest up-facing surface is the floor and gets the floor's resolution; a spawned prop (``A1/...``)
  is an entity with a light map of its own.
- **Lit live**: everything else. What hangs on a joint to the world (a door's leaf, a drawer, an arm)
  is an *occluder* in the bake at its reset pose; what floats free (a robot, a crate) and mocap bodies
  are not, so adding a robot to a world keeps its bake fresh. ``occluders:`` names them instead. A
  ``<light>`` on what moves (a robot's headlamp) is lit live too: it is not baked and stays when the
  world's lights are replaced by the run-time rig.
- **Glass** (a static geom with alpha below 1) is neither: light passes it, and it is drawn as it is.
- A geom the world no longer has (a ceiling opened by ``ceiling``, ``roqsim render --no-ceiling``) does
  not make the bake stale: the rest is drawn baked, recorded as ``bake_partial``.

Contacts and rays are those of the unbaked world: each baked geom keeps colliding and every ray still
meets it (it moves to geom group 3), and its bake is drawn by a visual mesh in the window group
(``presence.WINDOW_GEOM_GROUP``), which rays pass. The visual mesh is recorded as drawing its geom
(``roqsim.bake.compose.baked_sources``), so a segmentation camera's ``geoms:`` class that names the geom
labels what the camera sees of it. A camera that draws only chosen groups
(``geomgroup:``) sees the bake only if group 5 is among them.

Where the light comes from
--------------------------

``lights: auto`` takes the first that applies, and the bake records which:

1. **Emitting fixtures** -- static geoms whose material's emission is 0.5 or more (a ceiling panel): an
   emissive surface of 83 W/m² in the bake, glowing at run time. The world's MJCF lights are then their
   run-time stand-ins and are not baked again.
2. **The MJCF lights** welded to the world -- point, spot (a ``cutoff`` of 90° is a point), directional
   as a sun.
3. **The default** -- one area light the size of the static footprint, 10 cm under its top, facing
   down, and a dim sky: an overcast skylight.

``emitters`` adds lamps no material declares (a photographer's LED panel) and sets a fixture's
strength. ``lights`` as a list (``type: area|point|spot|sun`` with ``pos``, ``size``, ``dir``,
``power_w``, ``spot_deg`` or ``strength``) replaces rules 2 and 3. Only the ratio between sources
matters: one exposure per world puts the shell's lit walls at a fixed level.

The bake, renderer-neutral
--------------------------

The bake is the light alone, not the light times the surface colour: per entity a light map (direct
and bounced light, luminance plus coarse chroma), the indirect light and ambient occlusion at a quarter
of its resolution, its lightmap UVs, and ``bake.json`` (fingerprint, lights, settings, exposure,
times). The surface colour is the model's own textures and materials. MuJoCo's classic renderer is one
consumer (``roqsim.bake.compose``: light map times the model's colour, divided by the run-time rig's
light for each texel's normal, composed in memory). A physically based renderer takes the indirect
light and the occlusion as its emission and occlusion inputs and does the direct light itself -- for
MuJoCo's Filament renderer, a material's ``emissive`` and ``occlusion`` textures over the same
lightmap UVs; as a mesh carries one UV set, the adapter also resamples the surface colour into that
layout, as the classic renderer's composition does, multiplies the indirect light by it and takes its
colour from the chroma map, and lights directly with the world's lights where they stand for the baked
sources -- and a glTF export can carry the same files: each is an adapter over this format, not a
rebake. The format is documented in ``python -m pydoc roqsim.bake.store``.

Only the classic renderer (``mujoco``) draws a baked world today. A camera whose renderer does not
declare it draws the bake (``baked_lighting`` in ``roqsim renderers describe``) is refused when the
world loads, as its images would light the bake a second time: choose ``renderer: mujoco`` for it, or
load the world unbaked with ``disable: [baked_lighting]``.

The run-time rig
----------------

A baked world is drawn under a fixed rig, which replaces the world's lights: no headlight; a key light
straight down that throws the shadow of what moves; three shadowless fills. The baked textures are
divided by exactly that rig's light, so MuJoCo draws the bake and not the bake lit again, and what moves
is lit the way the room was: brighter on top than at its sides. The baked surfaces throw no shadow
(their geoms' ``castshadow`` is off), so the key darkens the floor only under what moves.

Limits
------

- A texture finer than about 500 texels per metre is drawn at that resolution in a baked world (thin
  lines of a floor pattern fade at a distance).
- A metal underside does not mirror the floor: the bake holds diffuse light only.
- Two baked geoms without names in bodies without names cannot be told apart by the bake, and a world
  with baked lighting refuses them, saying to name them.
- Moving a static prop makes the bake stale; the world then loads unbaked until it is baked again (the
  UV layout of everything else is reused).
