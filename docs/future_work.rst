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
sensor analogue of the interactive-viewer slowdown already fixed in the runner (rendering was
decoupled from the 500 Hz step to a fixed display cadence).

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
and arm plugins follow this ("compute-on-read"); it removed a per-robot per-step cost that otherwise
multiplies as robots are added.

**Gap.** ``read()`` is called **once per consumer per due-tick**. If two transports read the same
endpoint (e.g. a second bridge, or a bridge plus an in-process ``RobotHandle`` consumer) in the same
step, the payload is computed **twice**. Today this never happens -- each endpoint is ``owner``-scoped
to exactly one domain bridge -- so compute-on-read is strictly cheaper than the old cache. But the
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
