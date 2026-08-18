.. _architecture:

Architecture & porting playbook
===============================

This is the source of truth for how roqsim is built and how to rework existing MuJoCo code into it. It documents both what exists today and designs that are **planned / not yet implemented** (so marked). A large external MuJoCo codebase is expected to be reworked into this structure — the **Porting playbook** (section 6) is the section to read for that.

   Status legend: **[impl]** implemented · **[planned]** designed, not yet built.

.. _1-overview--goals:

1. Overview & goals
-------------------

roqsim is a lightweight, plugin-driven MuJoCo simulator for mobile robots, robot arms, and mobile manipulators (plus extras like conveyor belts). A MuJoCo step loop plus **plugins** that hook into well-defined lifecycle points, all loaded and configured from a **single YAML file**.

Two ways to run, one engine:

-  **Standalone driver** (``runner.py``) — owns the loop; windowed by default, headless for Kubernetes; real-time / factor / as-fast-as-possible pacing; can record world state over time. **[impl]**
-  **scenario-execution driver** (``scenario_adapter.py``) — a ``SimulationInterface`` subclass; scenario-execution owns the loop and calls ``dt``/``setup``/``reset``/``step``/``shutdown``. It also publishes ``context``, the seam an in-process scenario action reads (§12, *Reaching a plugin from outside*). **[impl]**

Both wrap the same ``Engine`` **[impl]**, so behaviour is identical across the two.

Tick pipeline (one ``step()``):

::

    external threads ──post(cmd)──▶ [command queue]
                                         │ (drained on physics thread)
    physics thread:  drain ─▶ pre_step(all) ─▶ mj_step ─▶ post_step(all) ─▶ snapshot
                             (write ctrl)      (physics)   (read/publish)   (cross-thread reads)

Non-goals: roqsim is not a general game engine or a physics fork; it orchestrates MuJoCo. It does not recompile the model at runtime (see anti-patterns).

.. _2-lifecycle-reference:

2. Lifecycle reference
----------------------

A plugin (``roqsim.plugin.Plugin``) implements any subset of six hooks. The engine only calls hooks a plugin actually overrides (an un-overridden hook costs nothing and is absent from the timing table).

+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+
| Hook                 | When                                | May touch                                       | Typical use                                                                                    |
+======================+=====================================+=================================================+================================================================================================+
| ``build(spec, ctx)`` | once, pre-compile                   | ``spec`` only (``model``/``data`` are ``None``) | add bodies/geoms/sensors/assets                                                                |
+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``configure(ctx)``   | once, post-compile                  | ``model``, ``data``, ids                        | resolve body/site ids, open resources, advertise services, register ``RobotHandle``/``Entity`` |
+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``on_reset(ctx)``    | every reset, after ``mj_resetData`` | ``model``, ``data``                             | re-home arm, respawn objects                                                                   |
+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``pre_step(ctx)``    | each tick, before ``mj_step``       | ``data`` (write ``ctrl``, forces)               | apply commands                                                                                 |
+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``post_step(ctx)``   | each tick, after ``mj_step``        | ``data`` (read)                                 | publish, record                                                                                |
+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``shutdown(ctx)``    | once, teardown (reverse order)      | —                                               | close nodes, flush files                                                                       |
+----------------------+-------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------------------------+

Full-run sequence:

::

   setup():   build(p0)…build(pN)  →  spec.compile()  →  MjData  →  configure(p0)…configure(pN)
   reset():   drain →  mj_resetData → mj_forward → on_reset(p0)…on_reset(pN) → gates.reset()
   step():    drain → pre_step(p0…pN) → mj_step → post_step(p0…pN) → publish_snapshot
   shutdown(): shutdown(pN)…shutdown(p0)   (best-effort; a failure is logged, others still run)

Ordering rule: within a hook, plugins run in **YAML order**; ``shutdown`` runs in reverse. Cross-plugin dependencies are expressed by ordering + the blackboard, never by importing another plugin.

Reference implementation: ``roqsim/src/roqsim/engine.py``.

.. _3-api-contracts:

3. API contracts
----------------

.. _plugin-pluginpy-impl:

``Plugin`` (``plugin.py``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``__init__(self, config: dict | None, *, name: str | None)`` — receives its YAML ``config:`` section.
-  ``validate_config(self, config) -> list[str]`` — return error strings (empty = valid).
-  ``expand(cls, spec, world, base_dir) -> list[PluginSpec]`` *(classmethod, optional)* — extra specs to splice in right after this one at config load; used by spawn plugins to pull in a model's manifest (see §4). Default: none.
-  Hooks as in §2. ``parallel_safe: bool`` marks a read-only ``post_step`` for the future parallel executor. ``transport_only: bool`` marks a plugin that builds no geometry and holds no state (``BridgeBase`` and its subclasses), so the scene-only consumers — ``roqsim render``, the review window, the exporters — drop it and can therefore build a ``*_ros`` world without its middleware installed; ``roqsim sim`` keeps it unless asked for ``--no-communication``, which warns that the run then publishes and receives nothing (see :doc:`plugins` › Transport plugins).

.. _simcontext-contextpy-impl:

``SimContext`` (``context.py``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passed to every hook. Key members:

-  ``spec`` (build phase), ``model``, ``data``, ``dt``, ``sim_time``.
-  ``config`` — the full parsed YAML dict.
-  ``blackboard: Blackboard`` — ``set/get/require/__contains__``; typed cross-plugin store.
-  ``entities: EntityRegistry`` — ``add/remove/get/names/all`` of ``Entity(name, kind, body, meta)``; backs ``simulation_interfaces`` discovery.
-  ``render`` — a ``RenderService`` (lazily set; see §8). **[planned]**
-  ``post(cmd)`` / ``drain_commands()`` — the thread-safe command queue (§7).
-  ``publish_snapshot(d)`` / ``read_snapshot()`` — immutable snapshot for cross-thread readers.
-  ``register_gate(name, role)`` / ``gates()`` — step-gate API for synchronous mode (§10); **inert** now.

.. _robothandle-contextpy-impl:

``RobotHandle`` (``context.py``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RobotHandle(name, drive(vx,vy,w), read_odom()->(x,y,yaw,vx,vy,w))``. A controller plugin puts one on the blackboard (key convention ``robot:<name>``); a bridge looks it up. Both callables run on the physics thread.

.. _registry-registrypy-impl:

Registry (``registry.py``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``resolve_plugin(ref, base_dir) -> type[Plugin]``. See §4.

.. _config-configpy-impl:

Config (``config.py``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``load_config(path)`` / ``load_config_from_dict(raw, base_dir)`` → ``SimConfig``. ``instantiate_plugins(cfg) -> list[Plugin]`` resolves classes, expands manifests (``Plugin.expand``; see §4), constructs, and runs aggregated validation.

.. _4-config--registry:

4. Config & registry
--------------------

Single world YAML, two sections; ``plugins`` order = execution order:

.. code:: yaml

   sim:
     timestep: 0.004        # optional; else from the model
     pacing: realtime       # realtime | {factor: 4.0} | asap        [planned: honoured by runner]
     world: empty_room      # built-in name OR a path to an MJCF file; default empty_room (see below)
     integrator: implicitfast  # euler | rk4 | implicit | implicitfast
     noslip_iterations: 10  # solver effort; see "Solver options" below
     sync: {enabled: false} # foreseen lockstep mode (§10); inert

   plugins:
     - floorplan:                   # (1) entry-point short name -- the ref *is* the key
         mesh: envs/x.stl
         collision: true
       name: ground                 # optional instance name (reserved sibling key)
     - "my_pkg.mod:MyPlugin": { ... } # (2) importable module:Class (PYTHONPATH)
     - "./plugins/x.py:Foo": { ... }  # (3) file path:Class (relative to this YAML)

Each entry is a mapping with exactly one plugin-ref key (its value is the ``config`` map) plus an
optional reserved ``name:`` sibling. Three plugin-ref resolution forms (``resolve_plugin``):

1. **Short name** → ``roqsim.plugins`` entry-point group.
2. **``module.path:Class``** → ``importlib.import_module`` (any package on ``PYTHONPATH``).
3. **``path/to/file.py:Class``** → ``importlib.util.spec_from_file_location`` (relative to the config dir).

Forms 2 and 3 contain a colon, and the ref is the entry's *key*, so **quote it**
(``- "my_pkg.mod:MyPlugin": {...}``) — unquoted it parses only while no space follows the colon, so a
stray ``key: value`` space would silently truncate the ref. Short names have no colon and need no quotes.

Order: no ``:`` → must be an entry-point (else error). Has ``:`` → file if the left side ends in ``.py`` or exists on disk, else module. Every failure raises ``PluginError`` naming the attempted form.

Validation is **delegated to each plugin** (``validate_config``); ``instantiate_plugins`` aggregates all errors across all plugins and raises once, namespaced ``[name (ref)] message``.

World definition (``sim.world``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The static environment the robots/props stand in — ground + lighting — is a *world definition*, chosen with the ``sim.world`` key (``roqsim/world.py``). It is separate from robot-family scene plugins so a fixed cell (an arm, a conveyor) never has to depend on the mobile package for a floor. Semantics:

- ``sim.world`` is either a built-in world name or a **path to an MJCF file**. The only built-in is ``empty_room`` (a checker ground plane named ``floor``, a ceiling light, and four perimeter walls — a bounded, lit room); unset ⇒ ``empty_room``. A value that ends in ``.xml``/``.mjcf`` or contains a path separator is loaded as the base scene (``MjSpec.from_file``, resolved relative to the world YAML) — e.g. a baked scene like ``depot/depot.xml`` (see ``roqsim_scenes``). Anything that is neither the built-in nor a resolvable file is a fail-fast error.
- The engine builds/loads the world into the ``MjSpec`` **before** any plugin ``build``, so plugins attach onto it.
- A scene plugin that builds its **own** ground+lighting sets the class attribute ``provides_world = True`` (the mobile ``floorplan``, which also adds lidar walls). When such a plugin is present the engine **skips** the world definition; if ``sim.world`` was *also* set explicitly the engine logs a warning and lets the plugin win. So ``floorplan`` is the mobile scene, ``sim.world`` is the fixed-cell default, and they never double up the floor.

Policy specs (``roqsim.policy``) [impl]
'''''''''''''''''''''''''''''''''''''''

A policy-driven robot needs its observation assembled in exactly the layout its checkpoint was trained
on, and getting that wrong does not raise -- the robot twitches. Three plugins (``g1_locomotion``,
``oli_locomotion``, ``spot_locomotion``) each hand-assemble that vector, against two mutually
incompatible config schemas (``g1.yaml`` flat, ``oli/walk_param.yaml`` nested), so a fourth policy would
mean a third bespoke reader.

``roqsim.policy`` makes the layout data: a ``PolicySpec`` YAML beside the checkpoint lists the observation
terms in order, the actuated joints, the joints that are *observed but not commanded*, the control gains,
and the envelope the policy was trained for. ``PolicySpec.build_observation`` is then the only thing that
assembles an observation.

It sits in **core** rather than a robot-family package because every family needs it, and it costs core
nothing: it *describes* checkpoints and never loads them, so it imports only ``numpy`` and ``yaml``.
Checkpoint loading stays in the family plugins, which is where ``torch``/``onnxruntime`` belong.

Generality is measured, not asserted. The format is tested against the two policies that already ship --
the G1's 47-dim walk observation (with its gait phase) and Spot's 48-dim Isaac observation (with base
linear velocity, which the humanoids do not use) -- by rebuilding each from a spec and comparing
element-wise with the plugin's own builder. Neither plugin is migrated: they work and are covered, and
swapping a live observation builder would risk a silent regression for no present gain.

A useful thing that fell out of writing those tests: Unitree's ``get_gravity_orientation`` and Isaac
Lab's ``quat_rotate_inverse(q, [0,0,-1])`` are bit-identical (max difference 0 over 500 random
quaternions), so one ``projected_gravity`` term serves a humanoid and a quadruped alike.

Solver options (``sim.solver`` and friends) [impl]
''''''''''''''''''''''''''''''''''''''''''''''''''

``sim`` carries five optional passthroughs to MuJoCo's ``<option>``: ``solver`` (``newton``/``cg``/``pgs``), ``iterations``, ``ls_iterations``, ``noslip_iterations`` and ``impratio``. Each is left at MuJoCo's default unless a world sets it, because the right value is a property of the *experiment*, not of the framework: a navigation world wants the cheapest solve that keeps wheels stable, a manipulation world needs contacts that hold.

**A grasping world must set ``noslip_iterations``.** MuJoCo defaults it to ``0``, which leaves friction contacts a residual tangential drift. Measured on the G1/Dex1 pick: a 0.5 kg parcel gripped at 20 N between two pads crept out of the jaws at **0.119 m/s** and was dropped within two seconds; with ``noslip_iterations: 10`` the creep is **0.0009 m/s** and the lift holds indefinitely — a 137× reduction. ``iterations`` and ``ls_iterations`` alone changed nothing measurable, because this is the solver's dedicated slip-removal pass rather than general convergence. The failure mode is worth knowing because it presents as *insufficient friction* and is not: sweeping the sliding coefficient from 0.4 to 3.0 moved the creep rate by 17%, while halving the payload moved it by 80×.

Contact overrides (``sim.contact_override``, ``sim.cone``, ``sim.gravity``) [impl]
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Three more ``<option>`` passthroughs, added for contact-rich worlds where the *contact* is the
measurement rather than an incidental.

``sim.contact_override`` sets MuJoCo's global ``o_solref`` / ``o_solimp`` / ``o_friction``, which
replace every contact's own parameters::

   sim:
     cone: elliptic                # pyramidal (MuJoCo's default) | elliptic
     gravity: [0.0, 0.0, 0.0]
     contact_override:
       solref: [0.02, 1.0]                    # (timeconst, dampratio)
       solimp: [0.9, 0.95, 0.001, 0.5, 2.0]   # (dmin, dmax, width, midpoint, power)
       friction: [0.3, 0.005, 0.0001]         # (slide1, slide2, spin, roll1, roll2)

They are here for **fidelity** first. A published model that enables MuJoCo's global override has to
be reproducible as published, and these three are the values such a model states -- and often the ones
it randomizes, since the flag makes them the only contact parameters in play. One corpus
reconstruction turns on exactly that, and its spec records the flag as *required* for the three to
have any effect at all. That a sweep over them is then an ordinary campaign factor -- needing no
bespoke plugin and no hand-edited MJCF per cell, the same reason ``spawn_model``'s
``mass``/``friction`` exist -- is the second reason rather than the first.

**Global, and before compile; both halves are load-bearing.** Per-geom ``solref``/``solimp`` stay in
the model. A change aimed at *named* geoms, *during* a run, is the ``model_override`` plugin (§9.2):
this key cannot be aimed -- it replaces every contact's parameters, including the ones a model tuned
-- and cannot move once the model is built.

**Setting any of the three enables MuJoCo's** ``override`` **flag, and that coupling is why they are
one key rather than three.** Without the flag MuJoCo ignores all three silently: the model compiles,
runs, and quietly uses the untouched defaults, so a world that sets ``o_solref`` and observes no
effect reads as a solver that does not respond to tuning. Nothing in the resulting model shows the
missing flag.

A short vector is padded from MuJoCo's current value rather than zero-filled, so
``{solref: [0.05]}`` varies the contact time constant and leaves the damping ratio alone -- which is
what sweeping one element means. Unknown keys and over-long vectors are rejected at config load, not
at compile: a typo here is otherwise invisible.

``sim.cone`` selects the friction cone. Pyramidal is MuJoCo's default and cheaper; elliptic is the
physically correct one and matters when the *tangential* force is the measurement -- an insertion
benchmark reads F_x/F_y directly, and a pyramidal cone quantises their direction to the pyramid's
facets.

``sim.gravity`` is a world key rather than a model property because a fixed-base contact experiment
often runs at zero g deliberately: a simulated force-torque sensor is not tared, so with gravity on
every wrench sample carries a constant tool-weight bias that a force-energy metric is dominated by.

Ending a run from a plugin (``ctx.request_stop``) [impl]
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

A trial that knows it is finished -- the goal was reached, the episode failed, the recording is
complete -- can say so with ``ctx.request_stop(reason)``. The standalone driver polls
``ctx.stop_requested`` and leaves its loop cleanly, so ``shutdown`` still runs and files still flush.

Before this, a world was padded out to a wall-clock ``--seconds`` that had to be guessed high enough
for the slowest cell and was then wasted on every faster one. It is a *request*: the engine does not
act on it, so an embedding driver (scenario-execution, a test harness) may ignore it and keep
stepping. Physics-thread only, like every other write on ``SimContext``; the first reason wins.

Asset de-duplication (``sim.dedup_assets``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``spec.attach`` deep-copies a child model's meshes, textures, and materials under a unique prefix, with no sharing — so a world that spawns *N* copies of one prop carries *N* copies of its assets, and ``spec.compile()`` parses every OBJ, decodes every PNG, and builds a convex hull for each. Between the build loop and compile the engine runs a de-duplication pass (``roqsim/assets.py``, on by default; ``sim.dedup_assets: false`` opts out) that merges byte-identical **file-backed** meshes and materials and drops the textures left unreferenced. It keys on the resolved absolute file path (sound because ``apply_assets`` already absolutized every ref) plus the attributes that change compiled output, and retargets references onto the survivor. It rewrites only the name-string references (``geom.meshname``, ``.material`` on geoms/sites/meshes/skins) that survive attach — a material's texture-role vector is immutable once attached, so identical *materials* are merged rather than repointing textures directly. Builtin/procedural assets and non-2D textures (skyboxes, cube maps) are never touched. On the ``os`` world (240 identical trays) this takes ``compile`` from ~2.5 s to ~0.15 s and texture RAM from ~930 MB to ~60 MB; the ``--profile`` load report shows it as the ``dedup_assets`` phase.

Model plugin manifests (``expand``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model bundles the plugins intrinsic to it (a mobile base → ``diff_drive`` + ``lidar``; an arm → ``arm_controller``) in a ``<model>.manifest.yaml`` manifest next to its MJCF, so a world only spawns the model instead of re-declaring them. Mechanism:

- Before instantiation, ``_expand_plugins`` (``config.py``) resolves each plugin class and calls its ``expand(spec, world, base_dir)`` classmethod, **splicing the returned specs in immediately after** the plugin that produced them (so a spawn's controller/sensors land before a later ``ros2_bridge`` that reads their endpoints). ``world`` is the list of explicitly-declared specs; core does no dedupe of its own.
- Spawn plugins implement ``expand`` in one line via the shared ``roqsim.manifest.expand_manifest(spec, world, *, target_key, default_name, base_dir=None)``. It resolves the model (see *Model discovery* below), reads the ``<model>.manifest.yaml`` beside the resolved file, and wires each entry to this entity by setting ``config[target_key]`` to the spawn's ``name`` (``target_key="robot"`` for mobile, ``"arm"`` for manipulators). Distinct entities (two arms) never collide.
- When the world already declares the same ``(plugin ref, entity)``, the manifest default is **not injected** — the world's entry is the one that runs — but the manifest's config is **merged underneath it**: per key, the world's value wins and missing keys are filled from the manifest. This is what makes a *partial* override work (``diff_drive: {robot: robot, test_cmd: [...]}`` adds a scripted command and keeps the model's wheel geometry and actuator names). The merge is shallow on purpose: a nested value the world sets replaces the manifest's whole mapping rather than being deep-merged. The world's spec is mutated in place, which is safe because plugins are constructed only after expansion completes — so declaration order does not matter.

  .. note::
     This merge was previously a plain skip (the manifest entry was dropped whole), which meant a
     partial override silently fell back to the *plugin's* generic defaults for everything the world
     did not restate — e.g. a Husky's ``diff_drive`` inheriting TurtleBot wheel radius and actuator
     names, then failing to resolve them against its MJCF. ``husky_ros2.yaml`` crashed on exactly
     that, while ``plugins.rst`` documented the merge. If you rely on a world entry starting from the
     plugin's defaults rather than the model's, use ``default_plugins: false`` and declare it fully.
- Opt out per spawn with ``default_plugins: false``; a model with no manifest yields nothing.

Any new spawn plugin reuses this by calling ``expand_manifest`` with the config key its downstream plugins use to name the entity.

Model discovery (``roqsim.models``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A spawn plugin's ``model:`` string is resolved by ``roqsim.models.resolve_model``, which mirrors plugin resolution (``roqsim.registry``) so a model may live in **any** installed package — not just the spawn plugin's own:

- **Short name** (``ur10e``, ``turtlebot4``, ``conveyor``) — searched across every package that registers a provider in the ``roqsim.models`` entry-point group. The caller need not know which package ships it.
- **Package-qualified** ``<package>:<model>`` (``roqsim_manipulation_assets:ur10e``) — resolved within that provider; use it to disambiguate a name shipped by two packages, or to self-document a dependency. A dotted ``module:name`` exposing ``MODELS_DIR`` also works for models outside the group.
- **Filesystem path** (absolute, or relative to ``base_dir``) — loaded directly.

``resolve_model`` returns a ``ModelAsset(path, meshdirs, texturedirs)``. Crucially, **the asset dirs follow the model to its own package** (a provider exposes ``MODELS_DIR`` and optionally ``MESHES_DIR``/``TEXTUREDIR``): a spawn plugin calls ``apply_assets`` to rewrite the child spec's mesh/texture refs to absolute paths found across those dirs, so it no longer forces its own package's ``meshdir`` onto a foreign model — which is what made a cross-package model impossible before. A model that reuses other packages' meshes adds an ``assets:`` key to its ``<model>.manifest.yaml`` naming the provider(s) to borrow from — a single name or a **list**. Search order: the model's **own** declared ``meshdir`` always comes first — its assets can never be shadowed by same-named files elsewhere (Menagerie-style link names recur across robots, e.g. the Oli and the G1 both ship a ``left_hip_yaw_link.STL``) — then the borrowed providers in order, then the provider default and the model file's dir. Since MuJoCo allows only one ``meshdir`` per model, resolving to absolute paths is what lets meshes from *several* packages coexist. A package registers its models with::

    [project.entry-points."roqsim.models"]
    roqsim_assets = "roqsim_assets.models"   # a module exposing MODELS_DIR

Example: a downstream package can ship only a custom arm variant — say ``ur10e_custom.xml`` + ``ur10e_custom.manifest.yaml``, no mesh copies — whose manifest ``assets: [roqsim_manipulation_assets, roqsim_sensors]`` borrows the stock arm meshes from one package and a camera mesh from another. ``spawn_arm: {model: ur10e_custom}`` then places it even though ``spawn_arm`` lives in a third package, ``roqsim_manipulation`` — which ships no models at all.

.. _5-plugin-type-catalog:

5. Plugin-type catalog
----------------------

Types are *roles* (which hooks you implement), not separate base classes:

-  **Scene/init** (``build``): floorplan/environment, spawn robot/arm, pedestrians, conveyor geometry.
-  **Controller** (``pre_step``): diff-drive (``cmd_vel``), arm trajectory, conveyor velocity, ped ORCA.
-  **Sensor** (``post_step``, maybe ``render``): lidar (``mj_multiRay``, no GL), camera (RGB-D, GL), IMU.
-  **Transport/bridge** (``configure``\ +\ ``pre/post_step``\ +\ ``shutdown``): ROS 2 bridge, ``simulation_interfaces``.
-  **Recording/observability** (``post_step``): metrics/CSV (a task plugin's own series). Recording MuJoCo *state* is not a plugin — it is the driver's ``--record`` (see §8).
-  **Render** (future): viewer overlays, debug markers.
-  **Perturbation**: a sensor's own noise config (§9.1), or ``model_override`` changing a named
   model value mid-run on an external trigger (§9.2). There is no shared error-model framework; the
   one that existed was removed (§9.1).

``DummyPlugin`` (``plugins/dummy.py``) is a minimal end-to-end example implementing every hook.

.. _6-porting-playbook:

6. Porting playbook
-------------------

The incoming codebase is monolithic MuJoCo scripts. Rework them into plugins as follows.

.. _61-decision-tree--code-that-does-x--hook-y:

6.1 Decision tree — "code that does X → hook Y"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| The old code…                                                             | → hook                    | Notes                                |
+===========================================================================+===========================+======================================+
| builds/edits the model XML, adds bodies, meshes, sensors                  | ``build(spec)``           | use ``MjSpec``; runs before compile  |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| looks up ``model.body(...)``/site/actuator ids; opens files/sockets/nodes | ``configure``             | model exists here                    |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| sets ``data.ctrl``, ``data.qfrc_applied``, mocap poses each step          | ``pre_step``              | the only place to write actuation    |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| reads ``data.qpos``/sensordata, renders, publishes, logs                  | ``post_step``             | read-only w.r.t. the sim             |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| re-homes/reseeds state at episode start                                   | ``on_reset``              | after ``mj_resetData``               |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| closes resources                                                          | ``shutdown``              | reverse order                        |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+
| receives async input (ROS cb, socket, GUI)                                | wrap in ``ctx.post(cmd)`` | never touch ``data`` off-thread (§7) |
+---------------------------------------------------------------------------+---------------------------+--------------------------------------+

.. _62-recipes:

6.2 Recipes
~~~~~~~~~~~

-  **New robot / arm:** a scene plugin whose ``build`` attaches the robot MJCF into ``spec`` (``spec.attach`` / add body), registers an ``Entity(kind="robot")`` and a ``RobotHandle`` in ``configure``.
-  **New sensor:** a ``post_step`` plugin that reads ``data`` (or renders via ``ctx.render``), optionally adds its own noise (§9), and hands the reading off (blackboard/bridge). Register a producer gate (§10) if it should participate in sync mode.
-  **New controller:** a ``pre_step`` plugin that consumes a target (from blackboard / ``ctx.post``) and writes ``data.ctrl``. Expose a ``RobotHandle`` so a bridge can command it. It must honour ``ctx.manual_control`` (§7): when set, the human owns ``data.ctrl`` for the run — return from ``pre_step`` without writing it, so the viewer's control sliders drive the actuators. Seeding ``ctrl`` once in ``on_reset`` stays right (it opens the sliders at the home pose); the rule is about the per-tick write. This is what the runner's ``--manual-control`` switches, world-wide, for every controller at once — hence a run-level flag rather than per-plugin config. **[impl]**
-  **Environment/floorplan loader:** a ``build`` plugin that adds a mesh + collision geoms to ``spec``.
-  **External transport (ROS/other):** a transport plugin — ``configure`` spins the client thread, callbacks ``ctx.post(...)``, ``post_step`` publishes from ``data``, ``shutdown`` stops the client.
-  **Moving part (conveyor):** ``build`` adds an invisible belt body on a slide joint; ``pre_step`` forces its velocity and wraps position; a contact pair tunes belt↔object friction.
-  **Injected fault (model_override):** a plugin that in ``configure`` resolves a *named* selection of geoms/bodies/actuators, saves their current values and publishes a handle plus a ``std_srvs/SetBool`` service endpoint; ``set_active`` writes the target rows and ``on_reset`` writes them back. No ``pre_step`` at all -- the change rides on ``ctx.post`` from the service, and ``post_step`` runs only on the step after a change, to check the fault actually landed (§9.2).
-  **Articulated + commandable prop (door):** the ``door`` plugin (``roqsim_assets``) is both — ``build`` hangs a leaf on a hinge joint with a force-limited position actuator; ``pre_step`` drives it toward a target *openness* and, if the leaf stalls against an obstacle, backs off (a gentle automatic door); ``configure`` registers ``Entity(kind="door")``, a ``DoorHandle``, and — when ``controllable`` — ``std_msgs/Float64`` ``cmd``/``state`` endpoints plus a ``control_msgs/GripperCommand`` action (reusing the generic 1-DOF handler via its ``state_key`` hint). The natural home for a door in a floorplan world is the opening the generator already cut (``floorplan_to_world.py --doors-map``).

.. _63-decomposition-guidance:

6.3 Decomposition guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~

Split one big feature into cooperating plugins that talk through the **blackboard/entity registry**, not direct imports. Example: a monolithic "robot + lidar + ROS" script → ``spawn_robot`` (build) + ``diff_drive`` (control, publishes ``RobotHandle``) + ``lidar`` (sensor) + ``ros2_bridge`` (transport, looks up the handle). Each is independently testable and reorderable in YAML.

.. _64-worked-example-planned--filled-when-the-first-real-feature-is-ported:

6.4 Worked example [planned — filled when the first real feature is ported]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Will show a concrete before (monolithic script) → after (plugins + world YAML), step by step.

.. _7-concurrency--threading-model-impl-queuesnapshot-exercised-in-phase-5:

7. Concurrency & threading model [impl: queue/snapshot; exercised in Phase 5]
-----------------------------------------------------------------------------

**Single-writer rule:** only the physics thread (the one calling ``engine.step()``) ever touches ``model``/``data``. This is non-negotiable — ``MjData`` is not thread-safe.

External input (ROS callbacks, ``simulation_interfaces`` services, GUI) must **not** mutate ``data`` directly. It enqueues a callable via ``ctx.post(cmd)``; the engine **drains the queue at the start of ``pre_step``**, so every mutation happens on the physics thread, in FIFO order, deterministically. ``ctx.post`` is the substrate the ROS bridge (Phase 5) and synchronous mode (§10) build on.

For readers on other threads, the engine publishes an immutable ``snapshot`` after each step (``publish_snapshot``/``read_snapshot``). The default path, though, is to read in ``post_step`` on the physics thread — no snapshot needed.

``reset``/``shutdown`` drain/flush the queue so no command targets a half-rebuilt model.

Anti-patterns (do not do)
~~~~~~~~~~~~~~~~~~~~~~~~~

-  ❌ Mutating ``ctx.data`` from a ROS callback / worker thread. Use ``ctx.post``.
-  ❌ Recompiling the model mid-run. Modify ``spec`` only in ``build``; at runtime use mocap/qpos writes, :mod:`roqsim.presence` to make a compiled entity appear and disappear, or the ``model_override`` plugin to change a named model *value* (§9.2). Those three are not recompiles: they write fields the compiled model already has, and put back exactly what they found.
-  ❌ Blocking inside ``pre_step``/``post_step`` waiting on external I/O (except deliberately in sync mode).
-  ❌ Plugin importing another plugin. Cooperate via the blackboard/entity registry.

.. _8-rendering-planned:

8. Rendering [planned]
----------------------

A ``RenderService`` on the context owns all GL/EGL contexts and camera renderers, created lazily and shared so multiple sensor/render plugins never fight over contexts.

-  **Lidar** uses ``mujoco.mj_multiRay`` — **no GL**, works headless everywhere (M1 uses this; nav2 needs a ``LaserScan``).
-  **Camera / RGB-D** uses ``mujoco.Renderer`` (offscreen). Cameras render **only when consumed** (lazy).

-  **The offscreen backend is bound by ``import mujoco``, so it is chosen by ``import roqsim``.** ``MUJOCO_GL`` is read exactly once, inside ``mujoco/rendering/classic/gl_context.py``, while mujoco is being imported; it assigns ``GLContext`` there and then, and an *unset* value is not an error but a choice — it falls through to **glfw**. Everything that follows from that is the reason :func:`roqsim.gl.select_offscreen_gl` is called from the package ``__init__`` rather than from a driver's ``main``: a driver module's own imports reach mujoco before its ``main`` body runs, so a selection made there sets a variable nobody will read again. This is not hypothetical — the call lived in ``roqsim.runner.main`` and was inert for every headless run, invisibly, because a world with no camera never constructs a ``Renderer`` and therefore never instantiates the mis-bound backend. It surfaced the first time a camera world was dispatched to a cluster, as ``mujoco.FatalError: gladLoadGL error`` from inside ``MjrContext``. A node with a DRI render device gets ``egl``, one without gets ``osmesa``; an explicit ``MUJOCO_GL`` always wins and ``ROQSIM_NO_GL_SELECT`` opts out. Both backends are installed in the container images for the same reason the choice is deferred: which one works is a property of the node, not of the image. The one case the package ``__init__`` cannot cover — a consumer importing ``mujoco`` first — is caught by :func:`roqsim.rendering.check_gl_backend`, which every renderer in the tree funnels through and which names the cause and the fix rather than leaving MuJoCo's message to stand. **[impl]**
-  The interactive viewer (``mujoco.viewer.launch_passive``) is a *driver* concern, separate from the offscreen ``RenderService``, so windowed vs headless is a driver switch, not a plugin change. The standalone runner sets the viewer's initial free camera from the world's optional ``sim.view`` block (``lookat``/``distance``/``azimuth``/``elevation``; any subset — omitted keys keep MuJoCo's model-derived default). ``sim.view`` is the camera and *only* the camera: it is schema-checked on load, so an unknown key fails the run rather than being silently dropped. The two Simulate side panels are deliberately **not** expressible there — they are run-level flags (``--left-ui``/``--right-ui``, passed to ``launch_passive``'s ``show_left_ui``/``show_right_ui``, both hidden by default), on the same footing as ``--manual-control``: a world describes the experiment, whereas panels and hand-driving describe one interactive session. All of it is windowed-only (ignored under ``headless``) and is not a plugin concern. **[impl]**
-  **Rendering an image is a driver concern too, and it is a separate process.** ``roqsim render`` (:mod:`roqsim.render`, the tool; :mod:`roqsim.rendering` is the library it drives) compiles a target through the *same* dispatch as ``roqsim sim`` (:func:`roqsim.runner.config_for_input`) and writes one frame offscreen, so a world renders exactly as it simulates. It runs in its own process with its own GL context and touches no running simulation, which is why the picture path has no performance question attached to it. Three deliberate choices: the camera comes from the world's ``sim.view``, layered by ``--view`` through the *same* override path ``--set`` uses (so ``sim.view`` keeps one validator and one frozen key set — there is no second camera grammar); the headless camera is built by handing a shim to :func:`roqsim.viewer.setup_camera`, so ``track``/``follow_heading``/preview framing are inherited rather than reimplemented; and a model's ``home`` keyframe is preferred over ``qpos0``, shared with ``roqsim assets render-thumbnails`` so a model's thumbnail and its ``roqsim render`` output are the same picture. Raw meshes are accepted here (wrapped in the preview scene :mod:`roqsim.mesh_preview` owns -- in the core, not beside the prop pipeline that motivated it, because a capability ``roqsim render --help`` advertises cannot depend on an optional sibling being installed) even though ``roqsim sim`` refuses them: loose geometry cannot be meaningfully *simulated*, but rendering it is both harmless and useful. Stdout is exactly one line of JSON, a machine contract rather than a convenience — it reports the camera in ``--view``'s own vocabulary, so a shot can be reproduced by copying it back. **[impl]**

-  **Closing the window is a wait, not a request.** MuJoCo runs the window on threads it owns and ``Handle.close()`` only sets ``exitrequest``, so the window, its GL context and its X drawable are destroyed *after* the call returns. A process that closed and then exited promptly (a world that fails to load, ``--steps 1``, a short ``--seconds``) raced its own teardown: Python's ``atexit`` ran ``glfw.terminate`` under the render thread, whose in-flight ``glXSwapBuffers`` then hit a destroyed drawable, and Xlib's default handler called ``exit()`` from that thread while the main thread was already finalizing — the process hung after its last line of output (or segfaulted). Both drivers therefore open with :func:`roqsim.viewer.launch_viewer` and close with :func:`roqsim.viewer.close_viewer`, which joins those threads (~10 ms) so the teardown is ordered; never ``Handle.close()`` or ``with handle:`` directly. Relatedly, the cosmetic X11 retitle (:mod:`roqsim.window_title`) polls other clients' windows, where a window closing mid-pass is normal, so it installs an ignoring X error handler for the duration — Xlib's default one would kill the process over a window title. **[impl]**
-  **Windowed run + cameras — two GL contexts, two backends.** The viewer window is *always* glfw: ``mujoco.viewer`` imports and initialises glfw directly, independent of ``MUJOCO_GL``, which selects only the *offscreen* ``Renderer`` backend. So a windowed run of a camera world holds a glfw window context **and** an offscreen render context at once, and they must not both be glfw — two glfw contexts in one process collide and MuJoCo aborts with ``gladLoadGL error``. The runner (see :func:`roqsim.viewer.prepare_viewer_gl`) resolves this before any GL loads, for windowed launches only, with two overridable defaults: it preloads the system **libGLEW** (the glfw window context needs it in the global symbol namespace on many Linux/GL-driver combinations) via an ``LD_PRELOAD`` re-exec — ``LD_PRELOAD`` is read only at process startup — and defaults ``MUJOCO_GL=egl`` so the offscreen cameras get their own context. Override with ``MUJOCO_GL`` / ``ROQSIM_NO_GL_PRELOAD``. Caveat: a hand-exported ``LD_PRELOAD=…libGLEW`` left in the shell drags GLX into the process alongside MuJoCo's PyOpenGL EGL backend and crashes ``import mujoco`` with ``undefined symbol: eglQueryString`` — roqsim preloads libGLEW itself, so do not also export it. **[impl]**

.. _9-sensor-noise-impl:

9. Sensor noise [impl]
----------------------

.. _91-sensor-noise-impl:

9.1 Sensor noise
~~~~~~~~~~~~~~~~

**Design note:** an earlier generic, composable error-model framework (a ``roqsim.error_models`` registry wired into sensors via an ``error_model:`` config key) was **removed**. In practice noise is sensor-specific — its parameters, where it applies, and how it degrades only make sense in the context of one sensor — so a generic cross-sensor abstraction added indirection without real reuse.

Instead, each sensor owns its noise as plain config:

-  **Lidar** (``roqsim_sensors``): ``range_stddev`` adds zero-mean Gaussian noise to finite ranges (0 = off); ``dropout_percent`` randomly drops that percentage of points per scan to a "no return" (``inf``). Determinism: seeded from the run's ``--seed`` through :meth:`roqsim.context.SimContext.rng_for`, which returns a **counter-based** (Philox) generator keyed on ``(seed, sim_time, sensor name)``. Counter-based rather than stateful for a specific reason: a shared stateful generator's position depends on how many draws happened before it — sensor rates, step count, and for cameras whether anyone was subscribed — so it is not a function of the world at all, and a value drawn at t = 12.5 could not be reproduced without replaying the whole run. Keying on *simulated time* rather than a step counter is what lets a sensor be re-run from a **recording** and produce the same noise the live run published, because a restored state carries its ``sim_time`` and nothing carries a step count. Before this, ``ctx.rng`` was read by the sensors but never set by anything, so noisy runs were not reproducible at all. A sensor re-run from a recording (``roqsim state --sensor``) is therefore deterministic and noise-correct for the restored state, but **not** bit-identical to what the live run published at that timestamp: live, ``post_step`` runs at the physics rate, so a sensor's own ``rate_hz`` gate fires between recorded samples and the endpoint holds a scan computed slightly earlier than the sample that recorded it. Measured, about a quarter coincide exactly. Recovering the rest would mean recording every firing — bagging the topic, at ~26x the size of the state recording — which is the trade this design refuses.
-  Ground-truth physics stays clean **for sensor noise**: only the reported value is perturbed. A fault that is *physical* -- a grasp that slips, a wheel that loses traction -- is the opposite case, and is §9.2 rather than this.

When a future sensor needs a different noise shape, add it to that sensor's config, not to a shared framework. Reference: ``roqsim_sensors/src/roqsim_sensors/plugins/lidar.py``.

The test for which of the two a perturbation is, is **where the failure lives**. A lidar that mis-measures a wall it can see is noise. A gripper that stops holding is not a measurement at all, and no perturbation of a report can produce it.

.. _92-physical-faults-impl:

9.2 Physical faults (``model_override``) [impl]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runtime counterpart to ``--set``: name a model field, name the objects, name a target value, and let an **external trigger** switch between nominal and target. It is what makes "the robot loses the object it is carrying, at this instant" a property of the world rather than something scripted into whatever drives the robot. Save/restore is exact, from the values read at ``configure``, the way :mod:`roqsim.presence` restores what it hid.

**Not every model field can be written at runtime, and the ones that cannot fail silently** -- so the plugin ships a curated allowlist, one row per field with its object namespace, its write class and its caveat, and refuses everything else by name with the reason. Two write classes, both measured against MuJoCo 3.11.0:

-  ``live`` -- in effect on the next step: ``geom_friction``, ``geom_contype``/``geom_conaffinity``, ``actuator_forcerange``.
-  ``needs_setconst`` -- ``body_mass``, whose write the dynamics **ignore** until ``mj_setConst`` recomputes the derived inertia (measured: the mass matrix does not move, and ``body_invweight0``/``body_subtreemass`` keep their compile-time values). The plugin calls it.

``geom_size`` is refused: ``geom_rbound`` is cached at compile, so a box grown 0.05 → 0.4 m kept ``rbound`` at 0.0866 and, while overlapping the floor, produced ``ncon = 0`` -- geometry that renders big and collides as if small. Three further refusals are *decisions* rather than safety, and the plugin says so: ``geom_priority`` (it also governs ``condim``/``solref``/``solimp``, so it would swap the contact's stiffness model at the instant of the fault), MuJoCo's own ``sensor_noise`` (§9.1 gave that to per-sensor config), and every ``opt.*`` global (``sim.contact_override`` and the ``sim:`` block own those, *before compile*, so they are in the compiled model and in the run's provenance -- a runtime write would make the recorded value differ from the one that ran).

**The plugin never decides when to fire.** A fault's timing is the experiment's independent variable, and severity is the configured target value, so both stay sweepable: what crosses the wire is one bit. The inbound endpoint is therefore a ``std_srvs/SetBool`` **service** rather than a topic -- a command whose outcome the caller needs -- and its reply carries the verdict, so a scenario's ``service_call()`` can fail a trial that failed to inject its fault.

**An override that lands in the model and changes nothing** is the failure that would otherwise produce plausible wrong data. MuJoCo takes a contact's friction from the geom with the higher ``priority``, and at *equal* priority the element-wise **maximum** of the two, so lowering one side cannot bring a pair below the other side's value. ``configure`` raises for what is certain (an unknown name, an empty selection, an explicit ``<pair>`` covering a selected geom, a geom :mod:`roqsim.presence` has made absent, an all-visual-only selection, a ``geom_contype`` override without its ``geom_conaffinity`` partner -- measured to be a no-op alone). The rest is caught at runtime by reading the *applied* contact rather than predicting it: one step after a change, ``verified`` reads ``landed``, ``no_effect`` (with a warning, and a failed service reply) or ``untested``. Deliberately *not* done: a configure-time analysis of which geom governs which pair, which would re-derive MuJoCo's mixing rule inside a generic field writer and could only ever warn.

The allowlist is also the plugin's documentation and its **discovery surface**: ``roqsim scenes describe`` reports it as ``overridable.fields`` (world-independent, so it costs no model build), and ``--overridable GLOB`` adds ``overridable.targets`` -- the names an override can select and their current values -- for a caller that is not a roqsim process.

.. _10-synchronous--lockstep-mode-planned--seams-exist-behaviour-inert:

10. Synchronous / lockstep mode [planned — seams exist, behaviour inert]
------------------------------------------------------------------------

Later, as in CARLA's synchronous mode (ref: ``github.com/carla-simulator/ros-bridge``), the tick must be able to *block* until external consumers react: don't advance until ROS 2 returns a control command computed from **this tick's** sensor data, and/or until **every sensor due this tick** has published. This makes the loop a deterministic ``sensor-in → control-out`` lockstep, independent of wall-clock pacing.

Design:

-  **Gates** (``ctx.register_gate(name, role)``), two roles:

   -  **Producer** (sensor): calls ``gate.satisfy()`` once it has published this tick; the engine computes the *due* set from sensor rates and, in sync mode, waits for all due producers.
   -  **Consumer** (controller/bridge): gate stays pending until the expected input arrives via ``ctx.post``.

-  **Blocking step:** in sync mode ``step()`` publishes (``post_step``), then blocks on the barrier (Event/Condition) until all gates clear or a **timeout** fires; the queued control is applied in the next ``pre_step``. Composes with both drivers (scenario-execution's ``step()`` just takes longer; pacer bypassed).
-  **Config:** ``sim.sync = {enabled, timeout_s, wait_for: [sensors, control]}``; default off. ``GetSimulatorFeatures`` reports whether sync stepping is active.
-  **Safety:** timeout + a deadlock diagnostic (which gate never fired) prevents a missing/late node from hanging the sim.

The command-queue model (§7) is the substrate: sync mode adds a *wait on named gates*, not a new concurrency scheme. Today ``register_gate``/``gates`` exist and are reset each ``reset()``, but nothing waits on them.

.. _11-performance--benchmarking-impl:

11. Performance & benchmarking [impl]
-------------------------------------

With ``Engine(profile=True)`` (the runner's ``--profile``) the engine times every overridden hook and accumulates per-plugin, per-hook wall-time; profiling off means no timing work at all. ``Engine.timing_report()`` → ``{plugin: {hook: avg_µs}}``; ``Engine.format_timing()`` prints a table. Use it to answer "is a plugin slowing the sim?" — total plugin overhead should be a small fraction of ``mj_step``. The same flag records one-shot load phases (plugin resolution, world parse, compile, data creation): ``Engine.load_report()``/``format_load_report()``, printed by the runner right after setup — this answers "why does the world load slowly?". For the practical workflow — measuring RTF, ``py-spy`` thread attribution, and the GIL caveat — see :doc:`profiling`.

**Parallel post-step [planned]:** consistent with the single-writer rule, only ``post_step`` is safe to parallelize (read-only). Plugins opt in with ``parallel_safe = True``; a future thread-pool executor runs the safe ones concurrently while ``pre_step`` (which drains the queue and writes ``data``) stays sequential. M1 runs everything sequentially; the only genuine second thread is the ROS executor, mediated by the command queue.

.. _12-conventions:

12. Conventions
---------------

-  **Layout:** the framework core is its own pip package ``roqsim/`` (engine, plugin API, drivers, the generic ``dummy`` plugin, and the driver-level capture/render modules). Generic, robot-family-agnostic sensors live in ``roqsim_sensors/`` (``lidar``, ``oakd_camera``, ``realsense_d435``). Robot-family plugins + assets live in further sibling packages — wheeled bases in ``roqsim_mobile/`` (floorplan/spawn_robot/diff_drive/omni_drive, models, worlds); arms and manipulators are further siblings, and a robot belonging to two families goes in a package depending on both (``roqsim_mobile_manipulation/``) rather than widening one family's dependencies. The ROS bridge and the nav2 example are colcon packages under ``ros2_ws/src/`` (``roqsim_ros_bridge``, ``roqsim_nav2_example``). Keep the core ROS-free.
-  **Naming:** plugin classes ``PascalCase``, entry-point names ``snake_case``; blackboard handles under ``<kind>:<name>`` (``robot:<name>``, ``metrics:<name>``, ``model_override:<name>``, ``contact_monitor:<name>``, ``door:<name>:state``).
-  **Reaching a plugin from outside:** a plugin an out-of-process *or* out-of-package driver must reach **publishes a blackboard handle** in ``configure`` — a small callable or dataclass under ``<kind>:<name>``, where ``<name>`` is the instance's world-YAML ``name:``. Consumers resolve that key; they never iterate ``engine.plugins`` and never match on a class name, which breaks silently on a rename and cannot distinguish two instances of one plugin. Publish a *callable* when the value is replaced each step (``contact_monitor.read_state``) rather than the object it returns. The in-process seam a driver starts from is ``MujocoSim.context`` (a :class:`~roqsim.context.SimContext`, i.e. exactly the rights a plugin has, single-writer rule included) — not the ``Engine``, so ``plugins`` and ``config`` stay the engine's own. Over a transport the same capability is an ``Endpoint`` (§13); the two are the same declaration read two ways, which is what lets one scenario action serve a stepped run and a ROS run.
-  **Assets:** record upstream license for any vendored MJCF/mesh next to it (see ``roqsim_mobile/.../husky_a200/husky_a200_LICENSE`` for the pattern).
-  **Model layout:** a provider ships its models either *one folder per model* — ``models/<name>/<name>.xml`` with the manifest, licence, port log and thumbnail beside it and its meshes in ``models/<name>/meshes/``, referenced bare via ``<compiler meshdir="meshes">`` (``roqsim_manipulation_assets``, ``roqsim_assets``, ``roqsim_sensors``, ``roqsim_mobile``) — or *flat*, ``models/<name>.xml`` with meshes namespaced under ``models/meshes/<name>/`` (``roqsim_humanoid``, ``roqsim_quadruped``). ``resolve_model`` accepts both; follow whichever the package you are adding to already uses. Either way the meshes are per-model, never pooled in one dir, because Menagerie link names recur across robots. One consequence to know when *borrowing*: a folder-per-model provider sets ``MESHES_DIR`` to its models root, so a foreign model borrowing it via ``assets:`` names the mesh ``<model>/meshes/<file>`` (Frankie does this for the Panda's meshes), whereas a flat provider's is ``<model>/<file>``.
-  **New plugin checklist:** subclass ``Plugin``; implement only the hooks you need; add ``validate_config``; register via the ``roqsim.plugins`` entry-point (built-ins) or reference by ``module:Class``/``file.py:Class``; add a test.

.. _13-robot-interface--transport-bridges:

13. Robot interface & transport bridges [impl]
----------------------------------------------

A robot's I/O is **self-describing** and transport-neutral, so it is not duplicated in a per-backend
bridge and can be wired to any transport. It has three layers:

**1. Endpoints (neutral, in the robot packages).** In ``configure`` a plugin registers
``Endpoint``\ s on ``ctx.interface`` (see :class:`roqsim.context.Endpoint`): a ``name``,
``direction`` (``"out"`` → provide ``read``; ``"in"`` → provide ``write``), an ``owner`` (the entity),
a ``namespace`` (a plain scope string each backend attaches to the endpoint's topics/frames/actions),
an optional ``rate_hz``, and a ``backend`` dict of inert per-backend hints keyed by backend name. The
``read``/``write`` callables traffic in **neutral payloads** (numpy arrays, tuples, small dataclasses)
and run on the physics thread. Crucially the robot packages import nothing transport-specific — the
message *type* is named as a **string** (``"sensor_msgs.msg.LaserScan"``) under the backend hint.

**2. ``BridgeBase`` (backend-agnostic, in ``roqsim/bridge.py``).** Shared machinery for every
transport: it iterates ``ctx.interface``, applies the optional owner filter, rate-gates each ``out``
endpoint, runs the per-tick publish loop on the physics thread, and marshals inbound data onto the
physics thread via ``ctx.post`` (single-writer rule intact, §7). A backend implements a few hooks:
``_setup`` / ``_make_output`` / ``_make_input`` / ``_publish`` / ``_now`` / ``_tick`` / ``_teardown``.

**3. A concrete backend (transport-aware, in its own package).** ``roqsim_ros_bridge`` provides
``Ros2Bridge(BridgeBase)`` plus a registry (``roqsim_ros_bridge/registry.py``): ``resolve_type`` turns the
type string into a class via ``importlib`` (cached); converters keyed by that string fill an outbound
message in place, decoders turn an inbound message into a neutral payload, and a reflective path
(``msg.data = payload``) covers primitive ``std_msgs`` with no registered converter. Actions follow
the same pattern one level up: an ``in`` endpoint whose ros2 hint block carries ``action`` (an action
type string, e.g. ``control_msgs.action.FollowJointTrajectory``) is served as an ``ActionServer``
whose goal-execution policy comes from a handler registry (``roqsim_ros_bridge/actions.py``) — so even
MoveIt2 trajectory execution needs no dedicated bridge and no ROS import on the producer.

**Injection, not authoring.** A world does not declare its transport. ``with_transport`` appends the
bridge at load time (``roqsim sim --ros``; ``ROQSIM_ROS`` for the scenario-execution adapter), which is
the exact inverse of ``drop_transport_plugins`` and is what keeps a checked-in world **ROS-free** and
therefore runnable in a pip-only environment where the bridge is not registered at all. Transport
describes how a run is *deployed*; the world describes the experiment. Idempotent, so a world that
does declare one is left alone.

**Namespacing.** One bridge serves the whole world. Each endpoint carries the ``namespace`` its
producer declared (usually the spawn plugin's ``namespace:`` config, inherited via the entity meta),
and the backend attaches it to that endpoint's topic (``/robot1/odom``), TF frames
(``robot1/base_link``), and action name. The bridge-level ``namespace`` config is an optional global
outer prefix for the whole sim; the ``owner`` filter remains for deliberately splitting endpoints
across several transports.

**Hardwired topics.** A producer can pin an endpoint's topic to an *absolute* name that ignores the
namespace, so the sim matches an external / hardware topic layout exactly. Any endpoint-producing
plugin accepts a ``topics:`` map keyed by endpoint role name — ``topics: {image:
/camera/color/image_raw, joint_states: /joint_states}`` — read via ``Plugin.topic_override(name)``
(``roqsim/plugin.py``) when it fills the backend ``topic``. An absolute topic (leading ``/``) is
published verbatim by the ROS backend (``_resolve_topic`` in ``ros2_bridge.py``), bypassing
``ep.namespace`` (and the node namespace); a relative topic is scoped as usual. Only the topic is
overridden — TF frames stay namespaced. For example, a robot model can hardwire ``/joint_states``
and ``/camera/color/image_raw`` in its manifest so a sim world is a drop-in for the matching real
robot + its operator UI (at the cost of being single-arm; see the manifest note).

**Zero-copy / FPS.** Message objects are preallocated once per endpoint and refilled each tick
(``reuse_messages``, safe for inter-process subscribers); numeric arrays are handed to the message as
matching-dtype numpy buffers (one C-level copy) instead of per-element Python loops. Adding a new
backend (zenoh, zmq) is a new ``BridgeBase`` subclass + its own registry — robots and worlds are
unchanged.

.. _13-glossary--faq:

14. Glossary & FAQ
------------------

-  **spec vs model vs data:** ``MjSpec`` is the editable scene description (mutate in ``build``); ``MjModel`` is the compiled, immutable model; ``MjData`` is the mutable simulation state.
-  **build phase:** everything before ``spec.compile()`` — the only time to change scene structure.
-  **driver:** the code that owns the loop and calls the engine (standalone runner or the scenario-execution adapter).
-  **gate:** a named barrier for synchronous mode (§10).
-  **Why a command queue instead of a lock?** So plugin authors never reason about locks and mutation order is deterministic; the physics thread is the sole writer.
-  **Can I spawn arbitrary bodies at runtime?** Not by recompiling. A world declares everything a
   trial may bring in, and ``SpawnEntity`` / ``DeleteEntity`` *activate* one of those rather than
   creating it: see :mod:`roqsim.presence`. An absent entity keeps its pose and is excluded from
   raycasts, rendering and contacts, and from what ``GetEntities`` lists. Parking it out of sight
   instead is the obvious alternative and is worse -- a free body accelerates under gravity for as
   long as it is away, so it returns with whatever velocity it accumulated.
