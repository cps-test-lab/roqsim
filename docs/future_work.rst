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
renderer. What it leaves:

- *One scene build and one read-back per tick.* Each camera builds the scene and reads its image
  back on its own. The scene could be built once per batch, each camera then setting only its view,
  its lights and its planes, and every colour image copied into one buffer read back once.

- *Off the physics thread.* The tick waits for its frames. A render worker that owns the context
  and draws from the snapshot the physics thread hands it, newest snapshot first and nothing
  queued, each frame stamped with its snapshot's sim time and its noise drawn there, would let
  physics step on; lockstep (§10) would keep waiting. The renderer contract already hands a
  renderer a snapshot and never the live ``MjData``, so it is a change to the service alone.
- *Post-processing on the physics thread.* A camera's own work on its frame -- ``mono8``
  conversion, the depth clip, the lens map, noise -- and a bridge's encoding (JPEG, PNG, RVL) still
  run in ``post_step``. For a depth camera that is most of its cost once the draw is shared.
- *A shadow map shared across one tick's cameras.* MuJoCo redraws it inside every ``mjr_render``,
  although it depends only on the lights, and the shadow pass is a large share of a draw. Sharing it
  needs owning MuJoCo's draw loop.
- *More than one GPU.* The service would deal a renderer's cameras to several instances, each with
  a context of its own, on a device of its own (an EGL device per instance).
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
