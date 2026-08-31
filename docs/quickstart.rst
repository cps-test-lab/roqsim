Quickstart
==========

This goes beyond the minimal :doc:`getting_started` manual and shows the main ways to run a world.

Standalone
----------

.. code-block:: bash

   # windowed, real time
   .venv/bin/roqsim sim <world.yaml>

   # headless, as fast as possible, stop after N steps, print timing
   .venv/bin/roqsim sim <world.yaml> --headless --pacing asap --steps 1000 --profile

   # scaled real time (4x)
   .venv/bin/roqsim sim <world.yaml> --pacing 4.0

   # reproducible noise, recorded, rendered to video afterwards
   .venv/bin/roqsim sim <world.yaml> --seed 7 --record run.npz --video run.webm

The target is a world YAML, an MJCF scene, or a ``<pkg>:<name>`` reference resolved from an installed
package (``roqsim sim roqsim_mobile:turtlebot4_demo``).

Pacing is ``realtime`` (default), a numeric factor, or ``asap``. Headless works anywhere, offscreen
sensors included: ``import roqsim`` picks a backend that exists on this machine (``egl`` where there
is a render device, ``osmesa`` where there is not), so ``MUJOCO_GL`` no longer has to be set by hand
— set it only to override that choice. To record a run, add ``--record`` — see
:ref:`recording-a-run` below.

.. note::

   **Windowed GL defaults.** On many Linux + GL-driver combinations the windowed launch aborts with
   ``gladLoadGL error``. When opening a window the runner applies two overridable defaults before any
   GL loads:

   - **Preloads ``libGLEW``** for the viewer. The passive viewer always opens its window via glfw,
     and that context is what needs libGLEW in the global symbol namespace. The runner locates it
     portably (via ``ldconfig`` — no arch-specific path) and re-exec's itself once with
     ``LD_PRELOAD`` set, so no manual ``export LD_PRELOAD=…`` is needed. No-op headless, with no
     ``DISPLAY``, when the environment already preloads GLEW, or when GLEW is absent. Opt out with
     ``ROQSIM_NO_GL_PRELOAD=1``.
   - **Defaults ``MUJOCO_GL=egl``** — but *only* the offscreen (camera) renderer. ``MUJOCO_GL`` does
     not affect the viewer window (always glfw); it selects the backend cameras render with. Forcing
     the cameras onto glfw too is what clashes with the window's glfw context and produces
     ``gladLoadGL error`` on camera worlds; giving cameras their own egl context avoids it. Override
     by setting ``MUJOCO_GL`` yourself (e.g. ``MUJOCO_GL=glfw``). In practice ``import roqsim`` has
     already chosen (see "Choosing a GL backend" in :doc:`interfaces`), and it has to: by the time the
     runner's ``main`` runs, mujoco is imported and the backend is bound. This default only still
     decides the windowed case under ``ROQSIM_NO_GL_SELECT``.

   Because the runner preloads libGLEW itself, do **not** also ``export LD_PRELOAD=…libGLEW`` by hand.
   A stale libGLEW preload lingering in the shell, combined with the ``egl`` default, drags GLX into
   the same process as MuJoCo's PyOpenGL EGL backend and crashes ``import mujoco`` with
   ``undefined symbol: eglQueryString``. If you see that, clear it with ``export -n LD_PRELOAD``.

Moving the camera
~~~~~~~~~~~~~~~~~

The window has two navigation modes and **F10** switches between them. The mode is named in the
lower-left corner for a couple of seconds whenever it changes, and once when the window opens.

**Mouse** (the mode a run opens in) is MuJoCo's viewer exactly as it ships: drag to orbit, right-drag
to pan, wheel to zoom. Nothing of roqsim's is in the way, so all three keep their full reach — which is
what an inspection driven by the mouse alone needs.

**Flight** is what F10 opts into, for getting *inside* a building-sized world rather than circling it
from outside. The **arrow keys fly** the free camera: ↑/↓ travel along the direction you are looking
(look down and you descend), ←/→ strafe level, **Page Up/Down** rise and drop, and holding **Shift**
flies three times faster. It applies to the free camera only — a ``track``\\ ed camera (below) owns
its own pose.

The drag stays MuJoCo's own orbit in flight too, but it turns about a point held ~1.5 m in front of
the eye rather than wherever the world framed its camera, so it reads as looking around rather than
circling the room. That is done by re-spelling the camera — a free camera's ``lookat`` and
``distance`` are just how its eye position is written down, so pulling them in renders the identical
image and only moves the pivot, which is not drawn. The cost is that MuJoCo's pan and zoom scale with
``distance`` and therefore cover much less ground per drag; in flight the arrow keys are the way to
travel, and F10 hands the reach back. Because the pivot is only a re-spelling, neither switch moves
the picture: you come out of flight looking at exactly what you were looking at.

Not WASD, which would be the obvious choice: MuJoCo's Simulate binds ``W``, ``A``, ``S``, ``D``,
``E`` and ``Q`` to visualization flags (wireframe, static bodies, select point, …), so those keys
would toggle rendering as they moved. Its arrow bindings only step the physics while it is *paused*,
which leaves them free during a run. (The scene-review window of ``roqsim_scene_builder`` is not
MuJoCo's window and does use WASD.)

Initial view
~~~~~~~~~~~~

A world can set how the interactive viewer's **camera** opens via an optional ``sim.view`` block
(windowed runs only; ignored when headless). Give any subset — omitted keys keep MuJoCo's
model-derived default:

.. code-block:: yaml

   sim:
     view:
       lookat: [0.0, 0.4, 0.95]   # point the camera orbits (world metres)
       distance: 3.4              # metres from lookat
       azimuth: 130               # degrees around +z
       elevation: -20             # degrees; negative looks down

A fixed manipulation cell, for instance, sets these for a 3/4 view of the arm at its working height.
``sim.view`` is the camera and nothing else: an unknown key there is an error, not a no-op.

Simulate's two side panels are hidden by default and are **not** world config — which camera a world
wants is part of the scene, whereas wanting the panels is a property of one interactive session, so
they are run-level flags:

.. code-block:: bash

   roqsim sim world.yaml --left-ui --right-ui

Either way Tab / Shift+Tab still toggle them at runtime.

Driving the robot by hand
~~~~~~~~~~~~~~~~~~~~~~~~~

The right panel's sliders write ``data.ctrl`` — but a world's controller plugin normally rewrites it
every tick, so a slider drag is undone before you see it. ``--manual-control`` stands every
controller down for the run and leaves ``ctrl`` to you:

.. code-block:: bash

   roqsim sim world.yaml --manual-control    # implies --right-ui, where the sliders are

It is a run-level switch, not world config: which controller a world wires up is part of the
experiment, whereas driving it by hand is a property of one interactive session. It applies to
*every* controller in the world (``diff_drive``, ``omni_drive``, ``arm_controller``,
``spot_locomotion``, ``g1_locomotion``) — they keep serving their endpoints and tracking odometry, they just stop
stamping ``ctrl``. Each still seeds ``ctrl`` once at reset, so the sliders open at the robot's home
pose rather than at zero.

What the sliders mean is the actuator's own units: wheel velocity for a ``diff_drive`` base, joint
position for an arm or Spot's legs, and **torque** for the G1's leg motors — a G1 left at zero torque
simply folds. Manual control needs the window, so it is refused under ``--headless``.

Following a mobile robot
~~~~~~~~~~~~~~~~~~~~~~~~

A static camera is fine for a fixed cell or a fixed arm, but a mobile robot drives out of frame. Add
``track`` to attach the camera to it — MuJoCo's tracking camera keeps the named body centred while
``distance`` / ``elevation`` / ``azimuth`` stay yours to orbit and zoom with the mouse:

.. code-block:: yaml

   sim:
     view:
       track: robot            # an entity name (as spawned) or a raw MJCF body name
       follow_heading: true    # chase cam: stay behind the robot as it turns
       azimuth: 180            # with follow_heading, an offset from the robot's yaw
       elevation: -20
       distance: 3.4

``track`` on its own keeps the azimuth world-fixed: the robot stays centred but spins in view as it
turns. ``follow_heading: true`` re-aims the camera at the robot's heading each frame, so it rides
behind the robot; ``azimuth`` is then an offset (``180`` = directly behind, ``0`` = head-on). Orbiting
with the mouse still works under ``follow_heading`` — the new angle is folded into the offset, so it
holds relative to the robot rather than being overwritten on the next frame. ``lookat`` is ignored
while tracking (MuJoCo drives it).

The name given to ``track`` is resolved against the entity registry first, so you write the robot's
name from its spawn plugin (``track: robot``) rather than its MJCF body prefix; any body name in the
compiled model also works. This works under both drivers — the standalone runner and the
scenario-execution adapter's viewer window.

Saving the view you framed
~~~~~~~~~~~~~~~~~~~~~~~~~~

Framing is done with the eye, not with numbers — so rather than transcribing a pose into ``sim.view``
by hand, fly the camera until the shot is right and press **F8**. A dialog shows the values and the
world YAML they would go into; confirm, and that world's ``sim.view`` block is rewritten:

.. code-block:: text

   Save the current camera view to

       /…/roqsim_scenes/src/roqsim_scenes/worlds/depot.yaml

       lookat:     [3.2, -1.05, 0.9]
       distance:   6.414
       azimuth:    142.5
       elevation:  -21.0

   This replaces the world's sim.view block.

Only that block is touched. The world is edited as *text*, not reparsed and dumped, so comments,
key order and one-line ``view: {…}`` styling all survive — a save shows up as a diff on the four value
lines and nothing else. The result is re-parsed and compared before it is written, so a bad edit is an
error rather than a mangled world.

A **tracking** view saves what it can mean: ``track`` and ``follow_heading`` are carried over,
``lookat`` is dropped (MuJoCo drives it), and under ``follow_heading`` the saved ``azimuth`` is the
*offset behind the robot* rather than the live world-frame angle — which is the half a world can state,
and the half that still means the same thing after the robot turns.

Two things to know. A saved free-camera ``distance`` is usually ~1.5 m, because the arrow-key walk
holds the mouse pivot that far in front of the eye (above): the pair ``lookat``/``distance`` still
names the same eye position and renders the identical image, it just no longer reads as "how far back
from the scene". And F8 needs a world YAML to write into — a run started from an MJCF scene or a bare
model reference has none, and says so rather than picking a file.

Checking a world before running it
----------------------------------

``roqsim check`` loads a world as far as it goes and reports **every** problem it found, in the order
the loader would hit them -- so a world with three bad keys reports three, and a mount that does not
exist is named rather than discovered by a run that dies quietly::

   roqsim check world.yaml
   roqsim check roqsim_mobile:husky_demo --json     # the same report, for a script

It runs five stages -- ``resolve``, ``inputs``, ``config``, ``build``, ``configure`` -- and says
which one it reached, because "the config is wrong" and "the config is fine and the model refused the
name a plugin asked for" send a reader to different files. It does not step the simulation: a world
that passes can still behave wrongly, but it cannot fail to *start*, which is the failure worth
catching before a campaign queues a thousand of them.

With no problems it prints the inventory instead: the entities that registered, the endpoints they
publish with their topics and types, and the model's totals. That is what the next thing gets written
against -- a scenario that drives ``robot``, a bridge that expects ``scan`` -- without opening the
world file and its manifests to work out what is in there.

.. _recording-a-run:

Recording a run
---------------

``--record`` samples MuJoCo state while the run proceeds, so a run can be looked at *afterwards* —
from any camera, at any resolution, as a still or a video — without re-running it:

.. code-block:: bash

   roqsim sim world.yaml --record run.npz
   roqsim sim world.yaml --record --capture-fps 10      # a slower rate, a smaller file

   roqsim render --state run.npz --out last.png        # where it ended up
   roqsim render --state run.npz --at 12.5 --out t.png # one moment
   roqsim render --state run.npz --out run.webm        # the whole run as video

Without ``--at`` you get the **last** sample, and the command says so on stderr along with the sample's
time — that is a choice you did not make, so it is not made silently. A video render reports progress the
same way: one line rewritten in place at a terminal, and a handful of lines when the output is a log.

``--capture-fps`` is **samples per simulated second**, so a recording plays back at 1× sim time
whatever pacing the run used. Samples can only be taken on a physics step, so the rate is snapped onto
the world's step grid and the command says so if that moved it noticeably; a fraction like ``500/17``
is accepted, which is what makes any rate it suggests something you can type back.

Recording costs the run almost nothing, which is why nothing is drawn while the loop runs. Measured over
30 s of sim with ``--pacing asap`` (so wall time *is* the loop cost): the default 25 fps is
indistinguishable from not recording; 500 fps — every single step on a ``0.002`` timestep — costs about
5.5%. A *rendered* frame, by contrast, is 2–6 ms depending on how much of the world is in shot (it is
scene-geometry bound, so resolution barely matters), against ~0.001 ms for a state sample: three orders
of magnitude, which is the whole reason the picture is made afterwards.

It costs the run almost no *memory* either, which is why an hour-long recording is a disk question and
not a "will this fit" one: each sample is written to a ``<recording>.npz.part`` stream as it is taken,
and ``close()`` packs that stream into the archive and removes it. Measured against holding the run in
memory, the loop cost is the same to within noise. The rate is otherwise a *disk* decision: at 25 fps a
mobile manipulator produces about 0.25 MB of samples per simulated minute, and the archive deflates that
by a factor that depends on the world — 70% of raw for a bare mobile robot, 11% for a world of
pedestrians — because a state vector repeats: most of a world stands still, so most of a record is the
record before it.

The file is a plain ``.npz``, so anything with numpy can read it — ``np.load("run.npz")`` gives a JSON
``meta`` member and a structured ``samples`` member with **both clocks** per sample: ``t`` is simulated
seconds and ``w`` is wall seconds *elapsed since the recording started* (``meta["wall_clock_origin"]``
names that zero). ``w`` is deliberately not a Unix timestamp — it is monotonic, so an NTP step mid-run
cannot make it go backwards, and it keeps nanosecond resolution that a float64 holding 1.7e9 does not.
Both are kept because neither is derivable from the other: their ratio is the run's real-time factor,
which is exactly what varies. Use ``t`` to ask what the physics did and ``w`` to ask what the run cost —
a stall on a slow sensor, or the gap where a viewer sat paused, exists only in ``w``. For arbitrary
computations over a run, use :mod:`roqsim.recording`:

.. code-block:: python

   from roqsim.recording import open_recording

   rec = open_recording("run.npz")
   rec.real_time_factor            # simulated seconds per real second, over the whole recording
   for sample in rec.range(8.0, 20.0):
       sample.sim_time, sample.wall_time, sample.index, sample.data
       ...          # any numpy/mujoco computation over a real restored state

``--record`` is for a run you launch yourself. A run launched *for* you — an orchestrator starting this
world through a ROS launch file, where the command line belongs to that file — asks for the same thing
through the environment, which both drivers honour:

.. code-block:: bash

   export ROQSIM_RECORD=run.npz              # same as --record, and an explicit flag still wins
   export ROQSIM_CAPTURE_FPS=25              # same as --capture-fps
   export ROQSIM_CAPTURE_EXPORT_DIR=capture  # also write a browser run capture (below)
   export ROQSIM_SIM_POSES=1                 # also stream sim_poses.csv (below)

.. _sim-poses:

The pose series
~~~~~~~~~~~~~~~

``ROQSIM_SIM_POSES`` streams a plain ``sim_poses.csv`` beside the recording: one row per sample per
free-standing body (those parented to the world — every robot, prop and walker, but not a wheel or an
arm link), with the world pose as a **quaternion** and the world **twist** read from the solver via
``mj_objectVelocity``:

.. code-block:: text

   timestamp,wall_time,frame,position.x/y/z,orientation.x/y/z/w,twist.linear.x/y/z,twist.angular.x/y/z

Two reasons it exists rather than leaving callers to difference the recording. A velocity obtained by
differencing positions is only ever as good as the interval it is divided by, and a consumer reading
poses off a *transport* is dividing by arrival times — quantised by whatever gates the clock, jittered
by delivery — so a constant speed comes out alternating. And a **stepped** run publishes nothing at
all, so this is the only pose series it has.

Beside it goes ``entities.json``, a one-line roster of what those rows *are* — each entity's name,
``kind``, ``body`` and whether it is currently ``present``, from the registry. The pose record cannot
carry ``kind``: a body name does not say whether it is a robot, a prop or a walker, and nothing
recovers that from the model. It is rewritten whenever the roster changes, so a trial that spawns an
obstacle mid-run is described by it rather than by the world it started in.

``timestamp`` is exact simulated seconds. One convention worth knowing: ``mj_step`` integrates ``qpos``
and leaves ``xpos`` holding the pose from *before* that integration, so a row is a coherent snapshot of
``timestamp - dt`` carrying the label ``timestamp`` — deliberately the same one-step lag the
``ground_truth_pose`` plugin publishes with, so the two describe the same instant. It cancels in every
derivative.

Like the clock map and unlike the ``.npz``, it is flushed per row, so a run killed outright still leaves
everything up to the last sample.

A relative path is anchored to ``RUN_OUTPUT_DIR`` (this run's own result directory) or, failing that,
``OUTPUT_DIR`` (the job's), so a run's artifacts land beside its other results instead of
wherever the launch left the working directory; otherwise it resolves against the working directory as
usual. Deliberately *not* ``SCENARIO_OUTPUT_DIR``: that is the root shared by every run of a batch, so
anchoring a per-run file there gives one path that each run of a sweep overwrites in turn. Recording stays a *session*
concern either way — the same footing as ``sim.headless``, which the world YAML rejects on purpose — so
there is no route to it through the world.

A recording also converts to a **browser run capture** — the motion half of replaying a run in a web
viewer, alongside the geometry ``roqsim export web`` emits:

.. code-block:: bash

   roqsim export capture --state run.npz --out capture/

It writes ``capture.json`` + ``capture.bin``: one track per joint value and one per body pose that
actually moved, each keyed by the name the scene descriptor uses, so the two artifacts address each other
without either knowing about MuJoCo. The format is the consumer's — whichever tool replays these is
where it is defined — and roqsim is one producer of it, the same relationship this package has with
URDF and SRDF.

A run stopped any of the normal ways writes its recording on the way out: closing the viewer window, one
Ctrl+C, or a **SIGTERM** — which is how a *supervised* run ends, whether that is ``docker stop``, a
container teardown, a scheduler eviction or a supervisor's timeout. Only ``SIGKILL`` loses it: an ``.npz``
writes its index at the end, so there is nothing to read. Such a run leaves its ``<recording>.npz.part``
sample stream behind, buffered up to whatever the kill interrupted; nothing reads it, and the archive's
*absence* remains the signal that the run did not end on purpose.

The recording is written before the capture is derived from it, so a problem with the browser artifact
costs you the numbers as well.

Exporting a model as one mesh
-----------------------------

``export web``, ``export urdf``, ``export srdf`` and ``export moveit`` all keep a model as a body tree,
because their consumers animate or plan it. A second class of consumer wants the opposite -- one rigid
mesh and nothing else. A model-based 6D pose estimator takes a single mesh and returns the pose *of that
mesh's frame*; a CAD tool imports one body and knows nothing about joints:

.. code-block:: bash

   roqsim export mesh --model turtlebot4 --out robot.obj --sidecar robot.json
   roqsim export mesh --model turtlebot4 --out robot.3mf --units mm     # for CAD
   roqsim export mesh --world w.yaml --prefix ur10e_ --out arm.ply      # one robot out of a world

The output's extension picks the format -- ``.stl``, ``.obj``, ``.ply``, ``.3mf`` -- and ``--format``
overrides it, which is also the only way to ask for ASCII STL, since both STL flavours share an
extension. An extension nothing recognises is a usage error rather than a guess.

**The frame is the contract.** Vertices come out in the frame of one body -- ``--frame``, defaulting to
the root of the selection, which for a mobile base is its own ``base_link`` because the wheels hang off
it -- so a pose estimated against the mesh IS the pose of that body, with nothing to compose onto it.
Every geom transform is composed back through the body tree from the compiled model rather than read
from the world, so exporting a robot from a bare model and from a world that spawns it somewhere odd
gives the same file.

Group 3 is excluded by default: the convention here is that group-3 geometry is collision-only, and a
chassis-swallowing collision cylinder is not a shape the robot has. ``--groups 3`` asks for exactly
that envelope instead, which is the small clean solid to design a mount against.

Formats differ in what they can carry, and it matters:

.. list-table::
   :header-rows: 1
   :widths: 12 20 20 48

   * - Format
     - Colour
     - Structure
     - Good for
   * - ``.stl``
     - none
     - one merged mesh
     - the universally readable lowest common denominator; states no unit anywhere
   * - ``.obj``
     - ``usemtl`` + ``.mtl``
     - one merged mesh
     - an estimator or renderer that matches on appearance (keep the ``.mtl`` beside it)
   * - ``.ply``
     - per-vertex RGB
     - one merged mesh
     - the same, in one self-contained file with no sidecar to lose
   * - ``.3mf``
     - base materials
     - **one named object per geom**
     - CAD: parts stay separable and named, and the unit is stated in the file

3MF is the one to hand a CAD tool. It is an OPC package (a zip of three XML members, written with the
standard library -- no dependency), and unlike the others it does not merge the selection: each geom
becomes a named, coloured ``<object>`` assembled into one build item, so ``left_wheel_cylinder`` is
something a designer can select and hide rather than anonymous triangles inside a lump. It also
declares ``unit="millimeter"``, which removes the guess that makes a 0.35 m robot import 0.35 mm tall.
One caveat: 3MF's specification wants manifold objects, and packaging does not make a non-watertight
source mesh manifold -- consumers repair it, a strict conformance checker will complain.

Units are metres by default, MuJoCo's own; ``--units mm`` is the CAD convention. Since an STL states no
unit anywhere, the choice is always recorded in the JSON summary (and in the file's header or unit
attribute where the format has one).

**Watertightness is reported, not claimed.** A merged robot mesh is usually not a closed solid: shipped
visual meshes may have boundary or non-manifold edges, and two geoms that touch do not fuse. That is
fine for rendering and for pose estimation, which need a silhouette and an appearance; a CAD tool may
need a repair pass before ``mesh -> solid``. So the summary counts boundary and non-manifold edges per
geom rather than asserting a solid, and ``--weld`` (default 1e-6 m) first merges vertices that are
merely duplicated, which is the common reason a mesh that looks closed is not.

Checking a run is healthy
-------------------------

Three things can go wrong with a run and raise no error anywhere: the simulation never starts
stepping, it steps far slower than realtime, or a robot stands still for the whole trial. The process
is up, the log is quiet and the exit status is 0, and the run produced nothing worth analysing.
``roqsim health`` is the check for exactly those three, and nothing else — everything else a run can
get wrong already reports itself.

.. code-block:: bash

   roqsim health <run-dir>                             # judge the record as it stands
   roqsim health <run-dir> --watch                     # follow a live run until something is wrong
   roqsim health <run-dir> --json                      # one document: findings, skips, and state

It is a **separate process that reads the two records above**, and it changes nothing about a run. A
check that lived inside the simulator would share the simulator's failure modes, and one that spoke
over the ROS bridge could not diagnose a broken bridge; this reads files, so it can also say something
true about a run that is wedged or already dead. The clock map answers checks 2 and 3, and
``sim_poses.csv`` answers check 1.

===  =========================================================  =======
#    check                                                      level
===  =========================================================  =======
1    every watched robot moves ≥ 1 cm per 60 s of sim time      warn
2    sim time starts advancing within 60 s                      error
3    sim time advances ≥ 5 s per 60 s of wall time (0.083x)     error
===  =========================================================  =======

Check 1 is only a warning because a robot standing still is often correct — waiting on a pedestrian, a
perception-only run, a manipulator-only phase — and a channel that interrupts healthy runs is one
nobody reads. Exit status is ``0`` when nothing is wrong (warnings are still printed), ``5`` on an
error-level finding, and ``2`` when the checks could not run at all. Exiting on a finding is the point:
a backgrounded command's output is invisible until it exits.

**Which of those bodies is a robot comes from the roster.** ``sim_poses.csv`` names every
free-standing body and cannot say which is which, so the recorder writes ``entities.json`` beside it
from the entity registry — name, ``kind``, ``body``, ``present`` — and check 1 watches the entities of
kind ``robot`` that are currently there. That is what makes one command correct for every world:
a check whose names have to be passed per world is a check that is absent from the run that needed
it. ``--robot`` remains as an **override**, for a run with no roster and for watching something the
registry does not call a robot.

An **absent** entity is deliberately not watched: its body stays in the compiled model, so the
recorder still writes rows for it, and a robot the trial has not brought in yet is standing still
entirely correctly. With no roster and no ``--robot``, check 1 reports itself *skipped* and says
which of the two reasons applies — and it warns about a named robot that never appears in the record,
rather than passing one it never looked at.

The same roster is what lets ``--json``'s ``state`` block carry a ``kind`` per entity, which the pose
record cannot: "the robot has not moved" and "the furniture has not moved" are different readings of
the same row.

**The precondition.** Both records exist only while a run is recording, so a run without
``ROQSIM_RECORD`` (and ``ROQSIM_SIM_POSES`` for check 1) cannot be checked. That is reported and exits
``2``; it never reads as health.

**Who runs it, when a supervisor does.** A run harness executes this command itself, in the running
container, on a bounded interval while somebody is watching the run — and reads ``--json``: the
``findings``, and ``level`` in particular, are the contract it acts on. An ``error``-level finding is
what such a supervisor ends a run on. So the exit code and the document are a public interface, and ``check`` slugs are names other
software matches on rather than prints. Nothing is pushed from inside the run and nothing is written
into a run's output by this: it is read on demand and answered.

**Silence means different things live and after the fact**, which is why the two modes differ. Both
records are sampled on *simulated*-time boundaries, so a frozen simulation writes nothing at all and
the only evidence of a stall is that nothing arrives. ``--watch`` is watching a run it expects to
continue, so that gap counts against it — and it stops without complaint when the ``.npz`` appears,
since that file is written by ``close()`` and so means the run *ended* rather than stopped. A one-shot
check has no such premise: it judges the span the record covers, because what happened after the last
row is not in the file. Without that split, every finished run would be reported as a stall a minute
after it ended.

Getting numbers out of a run
----------------------------

``roqsim state`` reads the same recording as numbers rather than pixels — poses, joint values, contacts,
sensors:

.. code-block:: bash

   roqsim state --state run.npz --check                       # what does this recording offer?
   roqsim state --state run.npz --at 12.5 --body base_link     # one moment  -> JSON
   roqsim state --state run.npz --body base_link --out b.csv   # whole run   -> CSV
   roqsim state --state run.npz --joint 'arm_*' --from 8 --to 20 --out arm.csv
   roqsim state --state run.npz --sensor front --out scan.npz  # the world's own lidar, re-run

``--at`` snaps to the nearest recorded sample and tells you which one it used, so you can see it landed a
few milliseconds off rather than assume it did not. Names accept globs, and a selector that matches
nothing is an error naming the near misses — never an empty column.

Every row leads with both of the recording's clocks — ``sim_time`` and ``wall_time`` (elapsed seconds
from the start of the recording, with its origin restated in the CSV header) — so "what was this run
doing when it slowed down?" is answerable from the same file as "where was the robot at t = 12.5".
An ``.npz`` of array series carries them as the ``times`` and ``wall_times`` members.

``--sensor`` re-runs a sensor **the world declares**, configured exactly as the world configured it, so
``--check`` is how you see what is on offer. Its output shape decides the file: a few values become CSV
columns, a scan becomes an ``.npz`` array, and an image is refused with a pointer to
``roqsim render --camera``. A re-run sensor is deterministic and gets the noise that moment would have had
(the recording carries the run's seed), but it is not bit-identical to what the live run published at
that timestamp — live, the sensor fires between recorded samples, so the value published then was
computed a moment earlier.

Rendering a picture
-------------------

``roqsim render`` writes an image of a world without opening a window, so it works over SSH, in a
container, and from a script. It takes the same targets ``roqsim sim`` does — a world YAML, a baked
``.xml`` scene, a model reference — plus a raw mesh:

.. code-block:: bash

   roqsim render world.yaml --out shot.png
   roqsim render roqsim_assets:industrial_table --out table.png    # one model, auto-framed
   roqsim render prop.obj --out prop.png                       # a mesh, before it is a model

The camera is the world's own ``sim.view`` (above). ``--view`` overrides it one key at a time, so you
can re-aim without editing the world; the keys are exactly the ones ``sim.view`` accepts:

.. code-block:: bash

   roqsim render world.yaml --out top.png --view elevation=-85 distance=45
   roqsim render world.yaml --out eye.png --view lookat=-3.2,-1.3,1.9 azimuth=200
   roqsim render world.yaml --out robot.png --focus robot      # find an unblocked angle for me
   roqsim render indoor.yaml --out room.png --no-ceiling       # see into a roofed world
   roqsim render world.yaml --out shot.png --size 1920x1080 --show

``--focus`` searches for a viewpoint with a clear line of sight, which is what you want indoors where
a wall is usually between the default camera and the thing you care about. ``--show`` opens the result.

Rendering needs an offscreen GL backend, which ``import roqsim`` already selected for this machine;
override it with ``MUJOCO_GL=egl`` (GPU) or ``MUJOCO_GL=osmesa`` (CPU). If mujoco was somehow
imported before roqsim — the one case that selection cannot reach — ``roqsim render`` says exactly
that rather than failing obscurely.

A vector value takes either spelling — ``lookat=-3.2,-1.3,1.9`` or ``lookat="-3.2 -1.3 1.9"``, MJCF's
— so a pose can be pasted back in whichever form you have it.

The command prints one line of JSON describing what it rendered, including the camera it used — in the
same vocabulary ``--view`` takes, so you can copy it back to reproduce a shot exactly.

From scenario-execution
-----------------------

roqsim implements scenario-execution's ``SimulationInterface``. Point the world file at it via an
environment variable and load the adapter by ``module:Class``:

.. code-block:: bash

   export ROQSIM_WORLD=roqsim_mobile/src/roqsim_mobile/worlds/turtlebot4_demo.yaml
   scenario_execution --simulation roqsim.scenario_adapter:MujocoSim <scenario.osc>

scenario-execution owns the loop and calls ``dt`` / ``setup`` / ``reset`` / ``step`` / ``shutdown``;
one behavior-tree tick advances one ``mj_step``.

By default the adapter runs headless. To show the interactive viewer window, pass a ``headless``
scenario parameter set to ``"False"``. This needs a display (the viewer window is always glfw,
independent of ``MUJOCO_GL``); with no ``DISPLAY`` set, ``reset()`` raises a clear ``DisplayError``
instead of crashing in the native viewer init. Unlike the standalone runner, the adapter does not
apply the runner's *windowed* GL defaults (the libGLEW preload and its re-exec); set them yourself if
a windowed adapter run hits ``gladLoadGL error`` on a camera world. The offscreen backend is not
among them — that one is chosen by ``import roqsim``, so the adapter gets it like everything else,
which is what makes a stepped run able to render at all.

The adapter can also leave a browser scene descriptor next to the run: set
``ROQSIM_SCENE_EXPORT_DIR`` and every (re)built world is exported as ``scene.json`` +
``scene.bin`` (+ textures) into that directory after ``reset()`` — the same format
``roqsim export web`` produces, but of the *exact* simulated world (``world_overrides``
included), captured at its true initial pose. A relative path resolves against the scenario's
``output_dir`` (which scenario-execution passes to ``setup()``; under a run harness that is the run's
result directory, so the descriptor ships as an ordinary run artifact for web viewers), falling back to
the process working directory.

It **records** on the same environment contract the standalone runner uses (``ROQSIM_RECORD``,
``ROQSIM_CAPTURE_FPS``, ``ROQSIM_CAPTURE_EXPORT_DIR`` — see :ref:`recording-a-run`), with a
relative path anchored to the scenario's ``output_dir`` here, since scenario-execution passes one. That
is what turns a run into something replayable: the descriptor above is the world's geometry, the
recording is what moved in it, and the run capture is that motion in the form a browser reads.

What a scenario can ask the simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scenario_execution_roqsim`` ships the substrate's own OSC actions — ``import osc.roqsim``:

.. code-block:: text

   entity_moved(entities: ['parcel'], threshold: 0.05, mode: displacement_mode!z, dwell: 8.0)
   entity_rotated(entities: ['crate'], angle: 0.5)
   set_model_override(instance: 'grip_fault')            # ...and `active: false` restores it

``entity_moved`` / ``entity_rotated`` succeed once the named entities have been displaced (or turned)
from where they were **when the action started** — net displacement, not path length, unlike
``osc.ros``'s ``odometry_distance_traveled``. ``set_model_override`` applies or restores a
``model_override`` fault (§9.2) and **fails the trial when the plugin reports the write changed
nothing**, so a run cannot record an unfaulted outcome under a faulted label.

All three work in a stepped run *and* in a ROS run, unedited: the transport is chosen from what the
runner offered. In-process they read ``MujocoSim.context`` (entity poses from ``data.xpos``, the fault
through the ``model_override:<name>`` blackboard handle, writes queued with ``ctx.post``); over ROS they
use ``simulation_interfaces/GetEntityState`` and ``<instance>/override``. Both are keyed on the same
**entity and instance names**, which is what makes one scenario serve both — see the package's README
for why TF is deliberately not the ROS pose source. None of them can run under ``remote()``: a remote
server is handed no simulation.

ROS 2 bridge
------------

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   source ros2_ws/install/setup.bash
   .venv/bin/python -m roqsim_ros_bridge.run_bridge \
       --world ros2_ws/src/roqsim_ros_bridge/worlds/turtlebot_ros2.yaml

   # in another sourced shell:
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.2}, angular: {z: 0.3}}'
   ros2 topic echo /odom
   ros2 topic echo /scan
   ros2 service call /get_simulator_features simulation_interfaces/srv/GetSimulatorFeatures

The bridge publishes ``/clock``; run other nodes with ``use_sim_time:=true``. See
:doc:`nav2_example` for a full navigation stack.
