Profiling & performance debugging
=================================

How to find where sim time goes. Three different questions, three different tools:

* **"Is a plugin slowing the step?"** — per-plugin, per-hook wall-time. Use the built-in
  ``--profile`` (§ :ref:`11-performance--benchmarking-impl` in :doc:`architecture`).
* **"Why does the world take so long to load?"** — one-shot load-phase totals (plugin resolution,
  world parse, per-plugin build, compile). Also ``--profile``: the runner prints a load table right
  after setup.
* **"Why is the real-time factor (RTF) below 1?"** — where the *physics thread's* wall time goes
  (physics vs. sensors vs. publishing vs. waiting), and whether any second thread actually overlaps.
  Use ``py-spy``.

The single-writer model (§7) means there is effectively **one hot thread** — the one calling
``engine.step()``. Everything a plugin does in ``pre_step``/``post_step``, plus the ROS bridge's
publishing, runs on it, serially, **500 times per sim-second**. That thread is what to profile.

Built-in per-hook profiler
--------------------------

With ``--profile`` the engine times every overridden hook and the standalone runner prints the
table at exit (timing is only *collected* with the flag — an unprofiled run does no timing work
at all):

.. code-block:: bash

   roqsim sim world.yaml --headless --pacing asap --seconds 40 --profile

Notes:

* Use ``--pacing asap`` to measure **raw throughput**. Under ``realtime`` pacing the pacer sleeps to
  cap at 1.0, so the table reflects sleeping, not compute.
* ``format_timing`` reports **average µs per call per hook**. A hook that self-decimates (lidar,
  walker, cameras, the bridge's rate-gated reads) shows its *amortized* cost — a large full-tick cost
  spread over the cheap early-return steps.
* ``mj_step`` itself is **not** in the table (it is not a plugin hook). To get the per-step wall cost
  including physics, diff the wall time of two ``--seconds`` runs (startup cancels out):
  ``RTF_intrinsic = Δsim / Δwall``. Example: 20 s → 26.3 s wall, 40 s → 36.1 s wall ⇒ 20 sim-s in
  9.8 s ⇒ ~2.0× intrinsic.

Load-phase profiling
--------------------

The same ``--profile`` flag also prints a load table to stderr immediately after ``setup()``, so a
slow-loading world can be attributed without waiting for the run to end:

.. code-block:: bash

   roqsim sim world.yaml --headless --pacing asap --steps 1 --profile

::

   load-phase timing (totals, ms):
     resolve_plugins              812.3
     world_load                     3.2
     setup_total                 9821.0
     compile                     7654.4
     make_data                     12.3
   per-plugin build/configure (count / total ms / max ms):
     SpawnModelPlugin         build        276     1234.5      45.6
     ShelfPlugin              build          5      123.4      80.1

Reading it:

* ``resolve_plugins`` — entry-point scanning, plugin class resolution, and every plugin's
  ``validate_config`` (which for spawn-style plugins includes model resolution).
* ``world_load`` — resolving ``sim.world`` and parsing the base MJCF.
* ``dedup_assets`` — the pre-compile pass that merges byte-identical file-backed meshes/materials
  from repeated ``spawn_model`` attaches (§ world definition in :doc:`architecture`). Its cost buys a
  much smaller ``compile``; ``sim.dedup_assets: false`` disables it.
* ``compile`` — ``MjSpec.compile()``, which includes all mesh/texture asset processing. With dedup
  off, worlds that attach many copies of the same model pay for each copy's assets here
  (``spec.attach`` duplicates the child's assets per instance).
* Unlike the per-hook table, this one reports **totals in ms**. Build/configure rows aggregate over
  instances that share a timing key: instances without an entry-level ``name:`` all key on their
  class name, so ``count`` is the instance count and ``max`` the worst single call.

Measuring end-to-end RTF (full ROS stack)
-----------------------------------------

The bridge publishes ``/clock``. Sample it twice, N wall-seconds apart, and divide:

.. code-block:: bash

   # inside a container on the sim's ROS_DOMAIN_ID
   ros2 topic echo /clock --once     # t0 -> sec+nanosec, note wall time
   sleep 20
   ros2 topic echo /clock --once     # t1
   # RTF = (sim_t1 - sim_t0) / wall_seconds

Thread/wall attribution with py-spy
-----------------------------------

``py-spy`` samples native + Python stacks of a **running** process without instrumenting it — the way
to see how the physics thread splits and whether a second thread overlaps.

It needs ``CAP_SYS_PTRACE``. Add it to the ``sim`` service temporarily (a throwaway compose override,
not the committed file)::

   # ptrace.override.yml
   services:
     sim:
       cap_add: [SYS_PTRACE]

.. code-block:: bash

   docker compose -f docker-compose.yml -f ptrace.override.yml up -d
   docker exec <sim-container> bash -lc '
     pip install --break-system-packages py-spy
     PID=$(pgrep -f run_bridge | head -1)
     py-spy record -f raw --nonblocking --pid $PID -d 15 -o /tmp/spy.raw'
   # hottest stacks (count is the last field of each folded line):
   awk "{c=\$NF; \$NF=\"\"; print c\"\t\"\$0}" spy.raw | sort -rn | head -25

Read the folded stacks by **leaf frame**: ``engine.py:step`` at the ``mj_step`` line is physics;
``clock.py:wait`` is the pacer sleeping; ``raycast.py:cast`` is the raycast, whichever sensor called
it (the frame above says which); frames under
``bridge.py post_step`` → ``_publish`` → ``rclpy .../publish`` are **publishing on the physics
thread**. If the ROS executor spin thread never appears, it is near-idle (the sim mostly publishes;
subscribers like nav2 rarely send back), so it is **not** competing for CPU.

The GIL, and a trap when testing threaded scaling
-------------------------------------------------

MuJoCo's Python bindings release the GIL around native ``mj_step`` (``mujoco`` 3.x, nanobind
``gil_scoped_release``), so physics C-work *can* overlap other threads' Python. **But** a naive test —
``mj_step(m, d)`` (``nstep=1``) in a Python loop on one thread, a busy loop on another — will look
**fully serial**. That is the *GIL convoy* effect, not a held GIL: each single step's release window
is sub-100 µs, and the stepping thread re-acquires the GIL before the waiter is scheduled. Verify GIL
release with a **batched** call instead, whose release window is long:

.. code-block:: python

   mujoco.mj_step(m, d, 100_000)   # one C call, GIL released for the whole batch -> a busy
                                   # thread on another core runs concurrently (measurably parallel)

Consequence for this engine: it steps with ``nstep=1`` and runs Python hooks between steps, so the
per-step GIL-free windows are exactly the ``mj_step`` (and pacer-sleep) intervals. A worker thread can
only overlap the physics thread *within those windows* — useful for moving publishing off the hot
path, but do not expect free parallelism from merely spawning threads.
