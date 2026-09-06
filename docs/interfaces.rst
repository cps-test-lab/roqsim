Interfaces
==========

The public surfaces you use to configure and extend roqsim.

The world YAML
--------------

A single file defines the world *and* the plugin pipeline. Plugin order is execution order.

.. code-block:: yaml

   sim:
     timestep: 0.004        # optional; else taken from the model
     pacing: realtime       # realtime | asap | {factor: 4.0}   (standalone only)
     integrator: implicitfast   # euler | rk4 | implicit | implicitfast (default)
     dedup_assets: true     # default; merge identical attached prop assets before compile

   components:
     - floorplan:                       # (1) entry-point short name -- the ref *is* the key
         size: 3.0
       name: ground                     # optional instance name (reserved sibling key)
     - "my_pkg.mod:MyPlugin": { ... }   # (2) importable module:Class (PYTHONPATH)
     - "./plugins/x.py:Foo": { ... }    # (3) path to a .py file (relative to this YAML)

Each entry is a mapping with exactly one plugin-ref key (whose value is the plugin's ``config`` map)
plus an optional reserved ``name:`` sibling, which defaults to the ref. Writing ``name:`` *inside* the
config map is refused with the corrected spelling: no plugin reads it there, and a document that
placed it there would load with every entry answering to the plugin's ref. The plugin ref has three
resolution forms:

#. **short name** — a registered ``roqsim.plugins`` entry-point.
#. **module.path:Class** — imported off ``PYTHONPATH``.
#. **path/to/file.py:Class** — loaded directly from a file (relative to the YAML).

Because forms 2 and 3 contain a colon and the ref is now the entry's *key*, **quote it**
(``- "my_pkg.mod:MyPlugin": {...}``): unquoted it parses only while no space follows the colon, so a
stray ``key: value`` space would silently truncate the ref. Short names have no colon and need no
quotes.

Each plugin validates its own ``config:`` section; the engine aggregates all errors and fails fast
before the scene is built.

Overriding the world
--------------------

A caller can override any part of the world *before* it is built, by passing a nested dict that
mirrors the YAML. Plugins are addressed **by name** (their ``name:``, else their plugin ref), so
callers never depend on list indices:

.. code-block:: python

   from roqsim import load_config

   cfg = load_config("world.yaml", {"sim": {"headless": False},
                                    "plugins": {"floorplan": {"size": 4.0}}})

Values deep-merge (scalars and lists replace). Overrides must be applied at load time -- the scene is
compiled when the engine is built, so mutating a built ``SimConfig`` has no effect.

The standalone runner exposes the same thing on the command line, where
``roqsim.overrides_from_dotlist`` parses the ``path=value`` form::

   roqsim sim world.yaml --set components.floorplan.size=4.0

...and in a **file**, which is the same nested mapping kept somewhere a command line cannot keep
it::

   roqsim sim world.yaml --override debug.yaml

Reach for the file whenever the value is structured -- a list of obstacle instances, a nested
plugin config -- because flattening one onto argv loses it to quoting and word splitting. It is
also how a saved override set is reused, and how a run's own settings are replayed afterwards: an
embedding driver that varies the world per run writes exactly this document beside the results.
Both flags are repeatable and compose, later winning, so a saved set plus one ad-hoc tweak is
``--override debug.yaml --set sim.pacing=asap``.

Driven from scenario-execution, ``MujocoSim`` takes the nested dict as a ``world_overrides``
parameter -- naturally an OSC struct, which the framework passes as a nested dict -- and (re)builds
the world with it applied. That shape has no command line, so a *deployment* supplies the same
document through ``ROQSIM_WORLD_OVERRIDES`` (a path, alongside ``ROQSIM_WORLD``); a scenario
passing ``world_overrides`` itself wins, being the experiment speaking rather than the deployment.
See :doc:`quickstart`.

Selecting the world itself
--------------------------

A world is normally chosen once, by whoever starts the run: the positional argument to
``roqsim sim``, or ``ROQSIM_WORLD`` for the scenario-execution adapter (which is constructed with no
arguments). A *scenario* never has to know that this simulator has a thing called a world -- and
neither does it need to when a run harness sweeps the world across configurations, since that
too is the deployment choosing, one process at a time.

The exception is a study whose **experiment is the world** -- one problem instance per configuration.
Such a scenario declares a ``world`` parameter and ``MujocoSim.reset()`` takes it, rebuilding when it
changes, which makes that study one scenario instead of N. The world is part of the rebuild key, not
just the overrides: reusing the previous model for a new world would silently run the wrong one.

Adding ROS transport
--------------------

A checked-in world is **ROS-free**, and deliberately so: a world that declares ``ros2_bridge`` cannot
be loaded where the bridge is not registered -- it is a colcon package, absent from a pip-only
install -- so keeping it out of the file is what makes the world standalone-runnable. Transport is a
property of *how a run is deployed*, not of the experiment.

So it is appended at load time instead, by :func:`roqsim.config.with_transport` -- the exact inverse of
``drop_transport_plugins``, and idempotent, so a caller never has to know whether the author put one
there::

   roqsim sim world.yaml --ros                       # append ros2_bridge
   roqsim sim world.yaml --ros --tf-namespace robot  # publish /robot/tf, frames unchanged
   roqsim sim world.yaml --sim-control               # also serve simulation_interfaces

``--tf-namespace`` and ``--sim-control`` imply ``--ros``. Going the other way -- running a world whose
author *did* put the bridge in the file, on a machine without the middleware -- is
``--no-communication``, which drops it and warns that the run now communicates with nothing (see
:ref:`transport-only`). The two are refused together. The adapter reads the same three from the
environment (``ROQSIM_ROS``, ``ROQSIM_TF_NAMESPACE``, ``ROQSIM_SIM_CONTROL``), because there is
no command line to put them on -- and because they must not become scenario parameters, for the same
reason the world is not one.

This replaces a rewrite that copied the world to a temporary file and appended the plugin there --
implemented twice in one experiment, in bash and in Python. Nothing is copied now, so the scene's
relative path needs no fixing up either.

Choosing a GL backend
---------------------

``MUJOCO_GL`` selects the offscreen renderer, and which value works is a property of the **machine**:
``egl`` needs a render device, and a CPU-only node needs ``osmesa``. Left unset, MuJoCo's own default
fails on such a node with a message about a broken GL install -- the wrong diagnosis.

**The variable is read once, while ``import mujoco`` runs, and binds ``GLContext`` there and then.**
Setting it afterwards moves the variable and nothing else. Unset is not an error either: an empty
value falls through to ``glfw``, which on a headless node opens no display and aborts inside the
first ``mujoco.Renderer`` with ``mujoco.FatalError: gladLoadGL error``.

So ``import roqsim`` picks one when ``MUJOCO_GL`` is unset (see
:func:`roqsim.gl.select_offscreen_gl`): a render node means ``egl``, its absence means ``osmesa``.
An explicit ``MUJOCO_GL`` is always honoured, and ``ROQSIM_NO_GL_SELECT=1`` opts out. This has to
happen where the simulator runs -- a run configured on one machine and dispatched to another
cannot know.

The **package** ``__init__`` is the call site, not a driver's ``main``, and that is load-bearing
rather than tidy. A driver's own module-level imports reach mujoco long before its ``main`` body
executes, so a selection made there is made too late to bind anything -- which is exactly what
happened: the choice lived in ``roqsim.runner.main``, was silently ineffective for every headless
run, and stayed invisible because a world with no camera never constructs a ``Renderer``. The first
camera world dispatched to a cluster is what finally instantiated the mis-bound backend.

Because the residual case -- a consumer that imports ``mujoco`` before ``roqsim`` -- cannot be
reached from here, :func:`roqsim.rendering.check_gl_backend` guards every renderer in the tree and
names both the cause and the fix instead of letting ``gladLoadGL error`` stand.

``DISPLAY`` is deliberately not consulted, unlike the shell script this replaces. That script set
``MUJOCO_GL`` for the whole process; this picks only the *offscreen* renderer, and a window comes
from ``mujoco.viewer``'s own glfw context regardless. A base image may well set ``DISPLAY=:0``
unconditionally -- ours does -- so trusting it would make a headless run choose ``glfw`` and fail
against an X server that was never started.

Listing what a world is made of
-------------------------------

A world is never one file: it is the YAML, whatever it ``extends``, the MJCF that chain settles on,
and the meshes and textures that MJCF names -- all referenced by paths relative to each other::

   roqsim scenes inputs worlds/depot.yaml
   {"world": "/abs/worlds/depot.yaml", "packaged": false, "inputs": ["/abs/...", ...]}

One line of JSON, the same machine contract ``roqsim render`` uses. It exists as a *command* because the
caller is often not a roqsim process: something staging a world into a container has no other
reason to have roqsim installed, so it asks the image that does. ``packaged: true`` means the files
arrive with an installed package and nothing has to travel.

Listing what a world provides
------------------------------

The other half of the same question, for a caller holding an *override* rather than staging files::

   roqsim scenes describe worlds/turtlebot_nav2.yaml
   {"world": "...", "packaged": false, "inputs": [...],
    "plugins": [{"address": "robot", "ref": "spawn_robot", "name": "robot",
                 "entity": null, "enabled": true, "origin": "document",
                 "paths": ["components.robot.model", "components.robot.pos"]},
                {"address": "robot.lidar", "ref": "lidar", "name": null,
                 "entity": "robot", "enabled": true, "origin": "manifest",
                 "paths": ["components.robot.lidar.rays", "components.robot.lidar.max_range"]}],
    "addresses": ["robot", "robot.diff_drive", "robot.lidar", "robot.oakd_camera"],
    "entities": null}

``plugins`` reports every component that will **run** -- the document's own entries and everything its
models' manifests contribute -- under the ``address`` an override names it by, with the dotted paths
into its config that already exist. ``origin`` says which of the two a component came from.

``addresses`` is that set on its own, and **it is exactly what resolution accepts**: a caller checks a
sweep key against it before spending an image pull. Note the world above declares one entry and gets
three more from the turtlebot4's manifest -- those three are the ones a sweep is most likely to
want, and they used not to appear here at all.

A path not listed is not necessarily wrong (a plugin may accept a key its world leaves at the
default), so a caller reports an unlisted *path* as unverifiable. What the list does settle is the
expensive mistake: an *address* matching nothing, refused at load -- inside the container, after the
image pull.

``entities`` is ``null`` unless ``--entities`` is passed, because naming them means compiling the
model. There is no cheaper way to ask: which entities exist is settled at compile time, since roqsim
never recompiles mid-run and ``simulation_interfaces`` serves no ``SpawnEntity``. A caller checking
that a scenario only drives entities the world has pays for it; one resolving paths does not.

``overridable`` answers the same question one layer down, for the model values a run can change while
it is in progress (the ``model_override`` plugin, :ref:`architecture <92-physical-faults-impl>` §9.2)::

   roqsim scenes describe tiago_pick:tiago_pick --overridable 'gripper_right*'
   {..., "overridable": {
      "fields": [{"field": "geom_friction", "namespace": "geom", "write": "live",
                  "does": "...", "caveats": "...", "measured": "..."}, ...],
      "targets": {"geom": [{"name": "gripper_right_left_pad1",
                            "body": "gripper_right_fingertip_left_link",
                            "geom_priority": 1, "geom_friction": [0.7, 0.02, 0.001],
                            "geom_contype": 2, "geom_conaffinity": 1}, ...],
                  "actuator": [{"name": "gripper_right_finger_pos",
                                "actuator_forcerange": [-10.0, 10.0]}]}}}

``fields`` is always present and costs nothing: ``mjModel``'s field set is a property of MuJoCo rather
than of this world, so the allowlist needs no model built. Each row carries what the field *does* and
how it can silently do nothing, because that is what a caller choosing an override needs and cannot
infer from a field name.

``targets`` is the world-specific half -- the names an override can ``select`` and their current values
-- and is ``null`` unless ``--overridable GLOB`` is passed, for the same reason ``entities`` is. The
glob is not a convenience: a mobile-manipulator world has hundreds of geoms, and unnamed ones are
omitted because nothing can address them. A geom also reports its ``geom_priority``, which is what
decides whether overriding *this* side of a contact does anything at all.

Both halves come from one build when both flags are given: compiling the world is the expensive part.

``--override FILE`` applies an override tree first, the same file ``roqsim sim --override`` takes.
It is what makes the build-fed halves answer about the world a *run* would load rather than the one
the file declares: which entities a world compiles depends on its plugins' config, so a caller whose
obstacles come from its own overrides sees none of them without it::

   roqsim scenes describe world/secorolab_nav2.yaml --entities --override run.overrides.yaml
   {..., "entities": ["obstacle_0", "robot"], "errors": null}

An address the world does not have is still refused, exactly as it is refused when
a run loads -- which is the expensive mistake this command exists to catch first.

That build has **no transport in it**, and ``dropped_transport`` names what went::

   roqsim scenes describe worlds/depot_ros.yaml --entities
   describing the scene without transport: dropped ros2_bridge, sim_interfaces   # on stderr
   {..., "entities": ["obstacle_0", ...], "dropped_transport": ["ros2_bridge", "sim_interfaces"]}

A describe publishes nothing, so a world's bridge is dead weight here exactly as it is for ``roqsim
render`` and the exporters -- and since the ROS bridge ships in a colcon package, a pip-only
environment cannot resolve it at all, so requiring it would fail a describe over plugins that
contribute no geometry. Only *identified* transport goes (:func:`roqsim.config.drop_transport`, never the lenient
``drop_transport_plugins``): a misspelt geometry plugin has to stay fatal, because dropping it would
leave an entity missing from a list a caller reads as complete. The bridge is still in ``plugins``,
so ``plugins.ros2_bridge.*`` remains a checkable override.

When only the build fails, the reply is still printed -- with ``errors.build`` set and the build-fed
keys left ``null``::

   {"plugins": [...], "entities": null, "dropped_transport": [],
    "errors": {"build": "mesh not found: ..."}}

A caller keeps the half that cost nothing (which plugin keys exist) instead of losing the lot. **The
exit code is still non-zero**: ``0`` goes on meaning "fully answered", so a caller reading only the
status is never told a partial reply was a complete one. A world that cannot *load* has no half to
hand back and prints nothing.

Extending another world
------------------------

Overrides *modify* an existing world; ``extends`` *inherits* one. A world YAML may name a parent to
inherit its ``sim`` block and ``plugins`` list, then add, remove, or modify elements:

.. code-block:: yaml

   extends: roqsim_scenes:depot # a parent world YAML: "<package>:<world>" ref or a path
   sim:
     timestep: 0.001              # deep-merged over the parent's sim (child wins per key)
   disable:                       # OPTIONAL: drop inherited plugins by name (needs ``extends``)
     - graspable_box
   components:                    # child entries are APPENDED after the (kept) parent entries
     - spawn_robot: {model: oli, name: oli, prefix: oli_, pos: [13.2, 2.6]}

The ``extends`` value resolves like ``sim.world`` -- a ``<package>:<world>`` ref against a registered
``roqsim.worlds`` provider (to that provider's ``<world>.yaml``), or a path relative to the child
YAML's dir. The parent's relative ``sim.world`` is absolutized so it keeps resolving from the child's
location, and parent worlds may themselves ``extends`` (cycles are rejected).

``disable`` selectors match a plugin's reserved ``name:`` **or** its config ``name`` field (e.g.
``spawn_model: {name: graspable_box, ...}``); a selector that matches nothing is an error, not a
silent no-op. There is no separate "modify" key -- to change an inherited plugin, ``disable`` it and
re-add a tweaked copy in the child's ``plugins``.

The plugin lifecycle
--------------------

A plugin subclasses ``roqsim.plugin.Plugin`` and implements any subset of these hooks (the engine
only calls the ones you override):

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Hook
     - When
     - Use
   * - ``build(spec, ctx)``
     - once, pre-compile
     - mutate the ``MjSpec`` (add bodies/geoms/sensors/assets)
   * - ``configure(ctx)``
     - once, post-compile
     - resolve ids/handles, open resources, advertise services, register entities
   * - ``on_reset(ctx)``
     - each reset
     - restore initial state
   * - ``pre_step(ctx)``
     - each tick, before ``mj_step``
     - write controls/actuators
   * - ``post_step(ctx)``
     - each tick, after ``mj_step``
     - read state, publish, record
   * - ``shutdown(ctx)``
     - teardown (reverse order)
     - release resources

``validate_config(config) -> list[str]`` returns error strings (empty = valid).

SimContext
----------

Every hook receives a ``SimContext`` with:

* ``model`` / ``data`` / ``dt`` / ``sim_time`` — the MuJoCo handles (``spec`` during ``build``).
* ``config`` — the parsed world dict.
* ``blackboard`` — a typed key/value store for cross-plugin cooperation (``set`` / ``get`` /
  ``require``). E.g. a controller registers a ``RobotHandle`` under ``robot:<name>`` for in-process
  consumers (teleop, standalone drivers).
* ``interface`` — the interface registry (``add`` / ``all`` / ``by_direction``): where a plugin
  declares its ``Endpoint``\ s (see below). A transport/bridge plugin reads it to wire the robot up.
* ``entities`` — the entity registry (``add`` / ``get`` / ``names``); backs ``simulation_interfaces``.
* ``post(cmd)`` — the thread-safe command queue. External threads (ROS callbacks) **must** use this;
  the engine drains it on the physics thread at the start of ``pre_step``. Never touch ``data`` from
  another thread.
* ``control`` — run-control (play / pause / step / reset) consulted by the standalone driver.

Declaring a robot interface (endpoints)
---------------------------------------

A robot describes its own I/O so a bridge can wire it to *any* transport (ROS 2, and later zenoh /
zmq) without the robot package importing that transport. In ``configure`` a plugin registers
``Endpoint``\ s on ``ctx.interface``:

.. code-block:: python

   from roqsim.context import Endpoint

   ctx.interface.add(Endpoint(
       name="scan", direction="out", owner=self.robot,
       read=lambda: self._scan,          # returns a neutral payload (numpy/tuple/dataclass)
       rate_hz=10.0,
       backend={"ros2": {"type": "sensor_msgs.msg.LaserScan", "topic": "scan",
                         "frame_id": self.frame_id}},   # the site the rays are cast from
   ))

* ``direction`` — ``"out"`` (sim → world; provide ``read``) or ``"in"`` (world → sim; provide
  ``write``). ``read``/``write`` traffic in **neutral payloads**, never wire messages — this is what
  keeps the robot package backend-independent.

  An ``in`` endpoint also says what *kind* of interaction it is, through its backend hints, and the
  choice is about the interaction rather than about taste:

  .. list-table::
     :header-rows: 1
     :widths: 14 30 56

     * - Hint
       - Served as
       - Use when
     * - ``type``
       - a subscription
       - the input is a **stream** and the sender needs no answer (``cmd_vel``).
     * - ``service``
       - a service
       - the input is a **command whose outcome the caller needs**, so it can fail on it
         (``model_override``'s ``std_srvs/SetBool``: the reply says whether the fault landed).
     * - ``action``
       - an action server
       - the input is a **goal that takes time**, with feedback and cancellation
         (``FollowJointTrajectory`` for MoveIt 2).

  ``write`` returns ``None`` in all three cases. A reply is assembled by the backend's handler from
  the producer's published state — named by a ``state_key`` hint — rather than returned from the
  plugin, which is what keeps ``Endpoint`` free of any backend's reply types. Both the service and
  action handlers come from per-type registries in ``roqsim_ros_bridge`` (``services.py`` /
  ``actions.py``), so a new srv or action type is a handler there and no change here.
* ``owner`` — the entity the port belongs to, so a bridge can serve one robot in a many-robot world.
* ``backend`` — inert per-backend hints keyed by backend name. Naming the message *type as a string*
  (resolved by the bridge) means the robot package imports nothing transport-specific. Anything
  transport-specific a robot needs lives here. One hint is worth knowing about because it saves
  writing a converter: ``field`` names a member of a structured payload to publish as a *primitive*
  message, so a producer keeps its rich payload for in-process consumers and still gets a
  single-field topic. ``contact_monitor`` reads its ``ContactReport`` in-process and publishes

  .. code-block:: python

     backend={"ros2": {"type": "std_msgs.msg.Bool", "field": "in_contact", "topic": "collision"}}

  Without it the bridge's reflective fallback would assign the whole dataclass to ``Bool.data``. The
  alternative — a ``std_msgs.msg.Bool`` converter in the bridge registry — would put one plugin's
  attribute names in shared code, and the next primitive endpoint would add another.
  ``model_override`` is the second user and publishes *two* members of one report this way (a
  ``Bool`` of ``active`` and a ``String`` of ``verified``), which is the shape to copy when a producer
  has several primitives each worth a topic.
* ``has_subscribers`` — optional performance hint a bridge sets after wiring the endpoint (e.g. from
  a ROS 2 publisher's subscription count). A producer whose ``read`` is expensive to *produce* (a
  rendered camera frame) may check it in ``post_step`` and skip the work when it's ``False``; ``None``
  (no bridge loaded, or one that doesn't report this) means "assume yes". See ``roqsim_sensors``'s
  camera plugins.
* ``lazy`` — opt this endpoint out of *publishing* while ``has_subscribers`` reports nobody listening,
  so its payload is never even read. Distinct from the check above, and both are needed: the producer's
  own check is an OR over every endpoint one render feeds (one depth subscriber justifies the whole GL
  pass), while ``lazy`` is per-endpoint (a raw-image subscriber must not make the camera pay for a JPEG
  nobody wants). Default ``False``, because a publish can carry more than its message — a bridge
  deriving TF from an odometry payload would stop broadcasting the transform whenever nothing
  subscribed to ``/odom`` — and because it buys nothing for a cheap payload. Set on the camera plugins'
  ``image``, ``image_compressed``, ``depth``, ``depth_compressed`` and ``points``.

Running the ROS 2 bridge then needs no per-topic config — add ``ros2_bridge`` to the world. For a
second robot add another with ``namespace: robot2``: it serves that robot's endpoints and prefixes
its topics/frames (``/robot2/...``). See :doc:`architecture` for how the bridge machinery works.

``RobotHandle(name, drive(vx, vy, w), read_odom() -> (x, y, yaw, vx, vy, w))`` remains the uniform way
a controller exposes a robot to *in-process* consumers (teleop, the standalone driver).

simulation_interfaces services (ROS 2)
--------------------------------------

The ``sim_interfaces`` plugin (in ``roqsim_ros_bridge``) exposes a subset of
`ros-simulation/simulation_interfaces <https://github.com/ros-simulation/simulation_interfaces>`_:

* ``GetSimulatorFeatures`` — advertised capabilities.
* ``GetEntities``, ``GetEntityState``, ``SetEntityState`` — list and read/teleport entities.
* ``SpawnEntity``, ``DeleteEntity`` — make an entity appear or disappear (see below).
* ``GetSimulationState``, ``SetSimulationState`` — play / pause / stop.
* ``StepSimulation`` — step N times while paused.
* ``ResetSimulation`` — reset the world.

Spawning is activation, not creation
`````````````````````````````````````

roqsim never recompiles the model at runtime, so there is no body to add. A world declares
everything a trial may bring in, and ``SpawnEntity`` selects one of those; a name the world does
not carry is refused rather than approximated, because the alternative is a trial that believes
it spawned something. ``spawn_formats`` is therefore **empty** — offering ``mjcf`` would invite
a caller to send geometry that nothing can load.

``DeleteEntity`` makes an entity absent: excluded from raycasts, from rendering, from contacts,
and from what ``GetEntities`` lists. Its pose does not move, which is the point — parking it out
of sight leaves a free body accelerating under gravity for as long as it is away, so it comes
back with whatever velocity it accumulated. See :mod:`roqsim.presence` for the three model fields
this flips and why the geom *group* is the one that matters: ``mj_multiRay`` ignores
``contype``/``conaffinity`` and tests the real triangles, so disabling contact alone would leave
an absent obstacle a perfectly good lidar return.

A world can declare an entity absent from the start, with ``present: false`` on the entry that
registers it::

    - spawn_model: {model: pallet, pos: [4.0, 1.0], free: true, present: false}
      name: obstacle

That is what gives a trial something to spawn. The declared value is restored on every reset, so a
spare brought in during one repetition is a spare again in the next. Do not confuse it with
``enabled: false``, which removes the entry entirely -- no body is built, and there is nothing left
to spawn.
