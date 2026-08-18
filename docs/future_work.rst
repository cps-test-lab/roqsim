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