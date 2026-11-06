Future work
===========

A running list of design topics deferred from the current work. Recorded so the plugin lifecycle
(:doc:`architecture`), the single-writer threading model (§7) and the render service (§8) are
honoured when the work lands.

The 3D human-review addon has its own roadmap (a sketch-authored door hinge side, play/settle
controls); it lives with that component under :doc:`scene_builder` › Roadmap.

Rendering beyond the render service
-----------------------------------

**Context.** Every image is drawn through the render service (§8): one context and one upload per
renderer instance, one scene build per tick, an atlas read back once, and an ``async`` mode that
draws off the physics thread from a snapshot. What it leaves:

- *Post-processing on the physics thread.* A camera's own work on its frame -- ``mono8``
  conversion, the depth clip, the lens map, noise -- and a bridge's encoding (JPEG, PNG, RVL) still
  run in ``post_step``. For a depth camera that is most of its cost once the draw is shared; moving
  it to the worker that drew the frame is the next step, and needs the camera's processing to be a
  function of the frame and the seed alone, which it already is.
- *Depth through the atlas.* Colour is read back once per batch; depth and segmentation are still
  read per camera. A depth atlas needs the depth attachment's format matched and the reversed-Z
  read reproduced exactly.
- *A shadow map shared across one tick's cameras.* MuJoCo redraws it inside every ``mjr_render``,
  although it depends only on the lights, and the shadow pass is a large share of a draw. Sharing it
  needs owning MuJoCo's draw loop.
- *Deferred read-back.* A pixel-buffer object would let the atlas read of one batch overlap the next
  batch's draws, at the price of every frame arriving a tick later; with one read per batch that
  price buys little, so it is not built.
- *More than one GPU.* ``sim.render.workers`` deals cameras to several instances, each with a
  context of its own, all on the device the GL backend chose. Placing each worker's context on a
  device of its own (an EGL device per worker) is what a second GPU needs.
- *A batch renderer.* MuJoCo Warp's ray-traced batch renderer takes exactly what the contract gives
  (the model once, a state per tick, a batch of heterogeneous cameras) and suits worlds with very
  many cameras that accept its look; it would stand behind the interface as a renderer of its own.

Per-tick memoisation of lazy endpoint reads
-------------------------------------------

**Context.** Producers should compute an ``out`` endpoint's payload **on demand in its**
``read()`` **callback**, not eagerly in ``post_step``: ``BridgeBase.post_step`` only calls
``read()`` when the endpoint's ``_RateGate`` is due (``roqsim/bridge.py``), so the work then
happens at the endpoint rate (e.g. 50 Hz odom) instead of every 500 Hz physics step. The locomotion
and arm plugins follow this ("compute-on-read"); it avoids a per-robot per-step cost that otherwise
multiplies as robots are added.

**Gap.** ``read()`` is called **once per consumer per due-tick**. If two transports read the same
endpoint (e.g. a second bridge, or a bridge plus an in-process ``RobotHandle`` consumer) in the same
step, the payload is computed **twice**. Today this never happens -- each endpoint is ``owner``-scoped
to exactly one domain bridge -- so compute-on-read is strictly cheaper than an eager cache. But the
pattern quietly assumes a single reader.

**Proposed fix.** Memoise ``Endpoint.read`` per sim-time: cache ``(sim_time, value)`` and return the
cached value when read again at the same ``sim_time``, recomputing only when the tick advances. This
makes lazy reads inherently single-compute for **any** multi-consumer topology, keeps the rate-gating
in one place (the bridge), and lets every producer keep a plain ``read()`` with no caching logic of
its own. Small change local to ``context.Endpoint`` / ``BridgeBase``; the alternative (each producer
re-adding its own cache) is exactly the eager-``post_step`` coupling this pattern avoids.

Plugin-declared viewer keys
---------------------------

**Context.** The keys roqsim adds to the viewer window are declared as :class:`roqsim.keys.KeyBinding`
records, and a handler says which it owns in a ``key_bindings`` attribute. ``keys.merge()`` reads that
attribute off *anything* with ``getattr``, so a handler, its class and a plugin are already sources on
the same footing, and the F1 overlay renders whatever it is handed. A plugin that wanted a key --
drop a waypoint, arm a trigger, mark the interesting moment of a long run -- is one attribute away
from having one listed and conflict-checked.

**Gap: dispatch, not declaration.** The key callback runs on MuJoCo's UI thread, while a plugin's
state change must happen on the physics thread (§7, single-writer). Every core handler already
resolves that the same way -- ``key_callback`` debounces and counts, ``take_pending`` is read by the
driver -- so the plugin base wants that split *offered* rather than reimplemented per plugin, or the
first plugin to take a key will write ``model`` from the UI thread and mostly get away with it.

**Also open.**

- The sources would be ``Engine.plugins``, which the viewer layer cannot see: ``SimContext`` carries
  no plugin list and nothing viewer-related. They are also only known after ``engine.setup()``, which
  is *after* the loading window has opened -- so either the list is rebuilt when the world is adopted,
  or the window is opened later than it is now (it is deliberately early, to cover a slow compile).
- Whether two plugins claiming one key refuses the load or refuses the second key. ``merge`` raises
  today, which is right for a fixed core set and may be too blunt for a world someone assembled.
- Whether a plugin may claim a key Simulate owns. It cannot suppress one, so at best it shares --
  which is exactly what F1 does deliberately, and what nothing else should do by accident.

Exporting a model as CAD geometry (STEP)
----------------------------------------

**Context.** ``roqsim export mesh`` covers the consumers that want triangles: a pose estimator matches
against them, and a CAD tool imports them as a mesh body. What it cannot give a CAD tool is a *solid*
with analytic faces. A tessellated wheel arrives as a few hundred planar facets, so it cannot be
dimensioned, offset or mated against; and because shipped visual meshes are not always watertight, the
``mesh -> solid`` conversion may need a repair pass before it even gets that far. ``--groups 3`` (the
collision envelope, which *is* primitives) is today's answer and is a good one for designing a mount,
but it is the simplified shape rather than the real one.

**Gap.** ISO 10303 (STEP) is the interchange format that carries exact geometry and an assembly tree
with names and colours. Nothing here can write it. The one Open CASCADE touchpoint in the tree
(``external/convert``) reads STEP and tessellates it -- the opposite direction.

**Two routes, both real.** A hand-written part-21 writer needs no dependency and is well-defined work
(faceted shells from mesh geoms with shared vertex/edge topology, plus exact ``CYLINDRICAL_SURFACE`` /
``PLANE`` / ``SPHERICAL_SURFACE`` solids for the primitives, and an assembly node per body so
repeated geometry is instanced rather than copied) but it is on the order of a thousand lines, and the
degenerate cases -- a sphere's poles, a cylinder's seam edge -- are where third-party importers
disagree. Alternatively an optional extra on an OpenCascade binding buys exact primitives, sewing,
assembly/colour support and a reader to verify the output against, at the cost of a ~68 MB wheel that
must stay out of the container image.

**Either way the geom walk, the frame composition and the primitive tessellation in
``roqsim/export_mesh.py`` are the input**, so this is an added writer rather than a second exporter.

Deformable bodies: conveniences over ``<flexcomp>``
---------------------------------------------------

**Context.** A flex is written as MuJoCo's own ``<flexcomp>`` wherever a model lives, and roqsim
adds what it has to around it: the integrator and the refusals, a flex as a prop or a tool, its
material as a world key (``flex_material``), flex-aware contacts, the ``roqsim check`` report and
the web replay (:ref:`deformable-bodies` in :doc:`plugins`). Everything an experiment on one needs
can be stated with that, in the experiment's own models and world, with no code of its own.

**Gap.** Six things are left to MuJoCo's own vocabulary or to the world author, each of which a
further experiment could turn into a repeated hand-written step. None is built, because each is
cheaper to design against the second experiment that needs it than to guess from the first; each
item names the evidence that would justify it.

**Proposed fixes, each with its trigger.**

1. **A** ``soft_body`` **plugin** that generates the ``<flexcomp>`` from a few world keys -- shape,
   resolution, material, parent frame, a pinned face. MuJoCo bakes a flexcomp's lattice at
   compile, so today a body's size or discretisation is fixed by its MJCF, and sweeping either
   needs one asset per level. Built from keys, both become campaign factors like the material.
   *Trigger:* an experiment that sweeps a soft body's size or its discretisation.
2. **Material presets**: named rows with physical values and their sources, listed by
   ``roqsim catalog materials`` and an MCP ``list_materials``, and accepted by ``flex_material``.
   *Trigger:* choosing a material by kind ("silicone, Shore 00-30") rather than by the values a
   paper states.
3. **Named vertex groups**: selectors over a flex's rest positions (``face``, ``plane``, ``box``,
   ``near``, and their combinations) that name a set of vertices once, for an observation to read
   and for ``<pin>`` to hold. Today a pinned set is MuJoCo's grid ranges or vertex ids, and a
   measured part of a flex is an index list in the experiment's own code. *Trigger:* a second
   experiment that measures part of a flex -- a face's displacement, a region's contact.
4. **A** ``damping_ratio`` **key** on ``flex_material``, converted to ``<elasticity damping>`` so
   that the first mode rings at the stated ratio: ``damping = 2 * zeta / omega_1 - timestep``, with
   ``omega_1`` from :func:`roqsim.flex_modes.first_modes` and the timestep term removing the
   ``discrete`` integrator's numerical share. It inherits that rule's limit: it holds while the
   timestep resolves the mode (``omega_1 * timestep`` at most 0.3, where ``roqsim check`` starts
   warning ``flex-timestep``), and the key has to refuse, or warn, beyond it. *Trigger:* a second
   paper that states a damping ratio rather than a damping time.
5. **A** ``flex_monitor`` **observation plugin** publishing a flex's state as endpoints: its
   vertices, the centroids of named groups, the largest vertex speed, edge strain, whether every
   vertex is finite, and contact aggregates against named geoms or entities -- with "vertices in
   contact" taken from surface vertices near an element contact, since a contact with a solid names
   an element rather than a vertex. *Trigger:* a second experiment that reads a flex's state or its
   contacts during a run.
6. **A reduced-dof default.** A measured study of ``dof`` in ``full``, ``quadratic`` and
   ``trilinear``: first frequencies, static deflection, contact penetration, the outcome metric of
   an example task, and wall time; where ``quadratic`` holds within a stated tolerance,
   ``roqsim check`` recommends it for a flex above some size. A ``full`` grid has three degrees of
   freedom per vertex where the reduced modes have 24 or 81 in all, and they require
   ``selfcollide="none"``. *Trigger:* a flex sweep whose cost bounds how many cells it can run.

**Consequences / open questions.**

- *One source of a flex.* A ``soft_body`` plugin would be a fifth place a flex can come from, and
  ``flex_material``, ``roqsim check`` and the replay would have to see it exactly as they see a
  ``<flexcomp>`` -- which they do if the plugin writes a ``<flexcomp>`` into the spec rather than
  building a flex of its own.
- *Presets are data with provenance.* A material row is only as good as its source, and a preset
  that silently replaced a paper's stated values would move an experiment's result; presets
  would have to be something a world names, never a default.
- *Groups before monitors.* Items 3 and 5 share their selectors, and a monitor reading groups by
  index would have to be migrated once groups exist, so the groups come first.
- *A recommendation needs the study.* Item 6 changes what ``roqsim check`` advises, which is only
  as sound as the measurement behind it, and a reduced mode that holds for a block may not for a
  sheet or a cable.
