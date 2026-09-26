Future work
===========

A running list of design topics deferred from the current work. Recorded so the seams built now --
the plugin lifecycle (:doc:`architecture`), the ``RenderService`` (§8), and the single-writer
threading model (§7) -- are honoured when the work lands.

The 3D human-review addon has its own roadmap (world-space dots via ray-cast, object picking,
play/settle controls); it lives with that component under :doc:`scene_builder` › Roadmap.

Off-thread (asynchronous) camera rendering
------------------------------------------

**Problem.** Camera plugins render **inline on the physics thread**: ``CameraPlugin.post_step``
(``roqsim_sensors/plugins/camera_common.py``) drives a private ``mujoco.Renderer`` synchronously,
so every captured frame *blocks* a physics step. Rendering is far more expensive than a physics step
(GL scene update + rasterisation, and it grows with scene/mesh complexity -- e.g. the Unitree G1's
multi-MB link meshes), so a subscribed camera at meaningful resolution/rate pulls the real-time
factor below 1: the whole sim -- including the robot's motion -- runs in slow motion. This is the
sensor analogue of the interactive-viewer slowdown the runner avoids by rendering at a fixed display
cadence, decoupled from the 500 Hz step.

The plugin is ``parallel_safe = False`` and holds its own renderer, so it cannot be moved to the
(planned) parallel ``post_step`` executor as-is.

**What exists today (cheap mitigations, already in place).** ``camera_common.py`` already:

- **throttles** capture to ``rate_hz`` (renders at e.g. 15--30 Hz, not every step), and
- **skips** rendering entirely when the ``image`` endpoint reports no subscribers
  (``Endpoint.has_subscribers``; see :doc:`interfaces`).

These bound *how often* we pay the cost, but each render still blocks the physics thread, so an
actively-consumed camera still stalls the loop.

**Proposed fix: a dedicated render thread owned by the** ``RenderService`` **(§8).** Rendering moves
off the physics thread entirely:

- The ``RenderService`` (§8 -- "owns all GL/EGL contexts and camera renderers, created lazily and
  shared") gains a **render worker thread** with its **own** GL/EGL context (a GL context is
  thread-affine; it must be created and used on that thread only).
- The worker renders from an **immutable state snapshot**, never live ``MjData`` -- honouring the
  single-writer rule (§7: only the physics thread touches ``model``/``data``). The physics thread
  hands off a cheap snapshot per due frame (``mj_getState`` into a reusable buffer, or a
  double-buffered ``MjData`` copy); the worker calls ``mjv_updateScene`` + ``render`` on its copy.
  This extends the existing ``publish_snapshot``/``read_snapshot`` mechanism (§7) from scalar state
  to render state.
- Frames are produced at the worker's **own cadence** and published to the ``image`` endpoint
  asynchronously. Physics keeps stepping at real-time; camera frames simply **lag slightly**.

**Consequences / open questions.**

- *Latency & timestamps.* A frame reflects the snapshot it was rendered from, not "now". The
  frame's stamp must carry the **snapshot's sim-time**, not wall-clock, so downstream (tf, nav2,
  perception) stays consistent. Bound the max lag (drop stale snapshots rather than queue them).
- *Determinism / sync mode.* Under the planned synchronous/lockstep mode (§10), a camera is a
  *producer gate* -- the tick must be able to wait for "this tick's frame". Async rendering needs a
  path to run **synchronously** (block the gate until the worker returns the frame for the due tick)
  when lockstep is enabled, and async otherwise. Keep both behind the same ``RenderService`` API.
- *Back-pressure.* One worker shared across N cameras vs one worker per camera (context/VRAM cost);
  a bounded frame queue with newest-wins drop policy.
- *GL context lifetime.* Create/destroy the context on the worker thread; clean shutdown ordering
  with the physics loop (``shutdown`` runs on the physics thread today).

**Why not just render more/faster inline?** Lowering resolution/rate and decimating meshes (the fat
G1 STLs) reduces per-frame cost but does not remove the coupling -- the render still blocks the step.
Off-thread rendering is the structural fix; mesh/resolution work is complementary (raises the
achievable frame rate once decoupled).

**Reference point.** Isaac Sim solves this natively with a GPU render pipeline that runs concurrently
with GPU physics; MuJoCo's ``mujoco.Renderer`` is single-context/thread-affine, so concurrency here
is an explicit worker-thread + snapshot design rather than a built-in. For massively parallel RL
(many envs) the separate answer is batched GPU rendering (MJX / Madrona), which is out of scope for
this single-env, real-time, ROS-facing use.

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
re-adding its own cache) is exactly the eager-``post_step`` coupling this pattern removed.
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
