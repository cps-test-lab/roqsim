Available plugins
=================

Built-in plugins by package. Reference each by its short name in a world file (or by
``module:Class`` / ``file.py:Class`` for your own — see :doc:`interfaces`).

The static environment (ground + light) is **not** a plugin but a *world definition* selected with
``sim.world`` (default ``empty_room``; see :doc:`architecture`). A fixed cell needs no scene plugin;
the mobile ``floorplan`` is the exception — it provides its own ground and overrides ``sim.world``.

Any endpoint-producing plugin below accepts an optional ``topics:`` map to **hardwire** an endpoint's
ROS topic to an absolute name, overriding the namespace+default — e.g. ``topics: {image:
/camera/color/image_raw}``. See :doc:`architecture` › Hardwired topics.

.. The catalog below is generated at build time from the roqsim.plugins entry points (see
   docs/_ext/plugin_docs.py), so it always matches the installed plugins. This note is a source
   comment and is intentionally not rendered.

.. roqsim-plugins::

ROS 2 bridge (``roqsim_ros_bridge``)
---------------------------------------

The bridge plugins live in the ROS 2 workspace (``ros2_ws/``), which is built with ``colcon`` rather
than pip-installed into the docs venv, so they are listed here by hand; they appear in the generated
catalog above once ROS is sourced and the workspace is on the path.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Plugin
     - Purpose and key config
   * - ``ros2_bridge``
     - Generic bridge: wires every endpoint declared on ``ctx.interface`` to ROS 2 (type resolved
       from the endpoint's type string; ``action``-hinted endpoints become action servers, e.g.
       ``FollowJointTrajectory`` for MoveIt2), plus ``/clock`` and ``tf``. No per-topic config —
       one bridge serves the whole world; each endpoint's own ``namespace`` (from its producer's
       config) scopes its topics/frames/actions. ``namespace`` (optional global outer prefix),
       ``tf_namespace`` (publish TF on ``/<ns>/tf`` + ``/<ns>/tf_static`` instead of the global
       ``/tf``; **topics only** — frame ids are unchanged, use ``frame_prefix`` for those),
       ``clock_rate_hz`` (default ``step`` — one ``/clock`` per physics step; a **rate** must
       divide every gated publish period or that publisher's stamps alias, and ``configure()``
       warns when one does not), ``reuse_messages``, ``rates`` (per-endpoint overrides), ``owner``
       (optional endpoint filter for multi-transport splits).
   * - ``sim_interfaces``
     - ``simulation_interfaces`` control plane (features / entities / state / step / reset). No
       required config; reuses the bridge's node when co-loaded.

.. note::

   **When to set** ``tf_namespace``. By default the bridge publishes TF on the global ``/tf``, which is
   right for a single robot owning the tree. But a namespaced Nav2 bringup follows the multi-robot
   convention of remapping ``/tf -> tf``, so *its* TF lives on ``/<robot>/tf`` — and tooling that
   assumes that convention (notably ``scenario_execution``'s ``NamespacedTransformListener``, which
   subscribes ``<namespace>/tf``) then never sees the bridge's ``odom -> base_link``. The symptom is a
   stack that looks healthy while a transform lookup hangs forever, e.g. scenario_execution's
   ``init_nav2`` stuck on *"Waiting for transform map -> base_link"*.

   Set ``tf_namespace`` to the robot's namespace so the bridge joins that tree::

       components:
         - ros2_bridge: {tf_namespace: a200_0000}   # -> /a200_0000/tf, /a200_0000/tf_static

   It must match the namespace the consuming stack uses, and it is **all or nothing**: every publisher
   of a link in the chain has to agree on one topic. Note this scopes the TF *topics* only — frame ids
   (``map``, ``odom``, ``base_link``) are untouched; namespace those with ``frame_prefix`` if needed.
   Setting the bridge's ``namespace`` does **not** do this: tf2_ros's broadcasters hardwire the
   absolute ``/tf``, which is exactly why this option exists.

.. _transport-only:

Transport plugins and the scene-only consumers
----------------------------------------------

A bridge publishes what the other plugins built; it adds nothing to the model. Such a plugin sets the
class attribute ``transport_only = True`` — ``BridgeBase`` does, so every transport inherits it,
including one you write — and the tools that want the *scene* rather than a running simulation drop it
before building: ``roqsim render``, the scene-review window (``roqsim_scene_builder``), and ``roqsim export
web/urdf/srdf/moveit``.

That is what makes a ``*_ros`` world **renderable without ROS**. Those same tools also skip a plugin
whose ref does not resolve at all (the bridge is registered by a colcon package, so it is absent from a
pip-only environment) and say so on stderr, naming it — geometry is unaffected either way, but the
other way to get there is a misspelt ref.

``roqsim sim`` keeps both by default: a simulation genuinely needs its transport, so an unresolvable
``ros2_bridge`` there stays the loud failure it should be (source ``ros2_ws/install/setup.bash``).

``--no-communication`` is the deliberate exception, for opening a ``*_ros`` world in the viewer on a
machine with no middleware installed:

.. code-block:: console

   roqsim sim depot_nav2.yaml --no-communication

It is not the render path's rule with a flag on it. Two differences, both because this one is a
*simulation*:

- Only a plugin that can be **identified** as transport goes — its class declares ``transport_only``,
  or its ref is one of the bridges ``roqsim`` names for ``--ros``. An unresolvable ref that is neither
  stays in the world and still fails the build, because there it is a misspelt plugin that would have
  changed the run.
- The run then **communicates with nothing**, and says so on every start: nothing published (no
  ``/clock``, no TF, no sensor topics, no odometry), nothing received (no ``/cmd_vel``, no goals, no
  services). An external stack sees a simulator that was never started, and a robot it would have
  driven stands still. Use it to look at a world, never to run its experiment.

``--no-communication`` and ``--ros``/``--tf-namespace``/``--sim-control`` are refused together rather
than silently ordered.

Nothing is required to know the flag exists, either. When a world's *only* unresolvable plugins are
its bridges, the failure says so and names both ways out — the bare "unknown plugin ``ros2_bridge``"
used to send the reader hunting for a typo that was never there.

Model plugin manifests
----------------------

A robot's controller and sensors are intrinsic to the *model*, not the *world*, so they ship with
the model in a ``<model>.manifest.yaml`` manifest next to its MJCF. A spawn plugin pulls them in
automatically, so a world just spawns the robot -- and the same applies to a *device* with more than
one sensor in it: the bundled ``d435`` is a D435i, so its manifest carries the ``imu`` component with
the inertial module's own extrinsic, and ``spawn_sensor: {model: d435}`` yields both ``camera/imu``
and the colour stream. A world that models the IMU-less D435 sets ``enabled: false`` on that
component (which is also how "does this device have an IMU" becomes a campaign factor):

.. code:: yaml

   components:
     - spawn_robot:
         model: turtlebot4
         pos: [0, 0]                # diff_drive + lidar + oakd_camera come with it
     - ros2_bridge: {}

An arm's manifest can also carry an **eye-in-hand sensor**: a camera among the arm's components rides
its flange, exactly as one among a mobile base's components rides the base -- there is no per-family
mount key to choose between. The
``open_manipulator_x`` model ships the MJCF ``d435_color`` camera at ROBOTIS's own RealSense mount pose
but deliberately leaves ``realsense_d435`` OUT of its manifest -- the arm provides the mount, the world
decides whether anything renders from it, at what rate, and whether it reprojects to a point cloud.

The same applies to ``spawn_arm`` (``roqsim_manipulation``): ``{model: ur10e}`` pulls in that
arm's ``arm_controller``; and to ``spawn_sensor`` (``roqsim_sensors``): ``{model: d435}`` pulls
in its ``realsense_d435`` capture plugin.

- **Override** a default: declare the same plugin inside that robot's/arm's ``components:`` block —
  your entry wins (e.g. add ``test_cmd``/``test_target``, or change ``lidar`` ``rays``). Nothing is
  duplicated. Matching is on the **label**: the entry's ``name:``, else its plugin ref, among that
  owner's components. There is no entity key to name — an entry belongs to the entity whose block it
  sits in — so the mistake this used to invite is not expressible: omitting ``robot:`` gave a *second*
  controller running alongside the manifest default rather than replacing it, two controllers fighting
  over the same actuators, and a config that silently had no effect.
  The override is **partial**: keys you do not mention keep the model's manifest values, so adding a
  ``test_cmd`` does not cost you the model's wheel geometry or actuator names. Per key, what the
  world says wins; nested values (e.g. ``topics:``) replace the manifest's mapping outright rather
  than being deep-merged. To start from the *plugin's* own defaults instead of the model's, opt out
  with ``default_plugins: false`` and declare the plugin fully.
- **Switch one off** with ``enabled: false`` -- as a sibling in the document, or from an override::

     roqsim sim world.yaml --set components.robot.oakd_camera.enabled=false

  The component is not deleted: it stays addressable, stays in the run's record saying it was turned
  off, and a later override can turn it back on. Disabling an entry disables everything it owns.
- **Opt out** entirely: set ``default_plugins: false`` on the ``spawn_*`` config.
- **Derive one manifest from another** with ``extends:``. ``unitree_g1_dex1`` is a ``unitree_g1``
  plus hands, and used to say so by repeating the base's locomotion and lidar blocks verbatim --
  two copies that then had to be kept in step by hand. It now inherits them::

     extends: unitree_g1        # a roqsim.models ref, or a path beside this manifest
     components:
       - arm_controller: {...}
         name: left_arm_controller

  What is inherited is **components, not geometry**: a derived model keeps its own MJCF. The base's
  components come first, so a derived entry with the same label is the one that runs. Cycles raise.
  A manifest may not carry ``sim:`` at all -- and neither may anything it extends -- because a model
  is a component of a world, and the run's seed, pacing and contact overrides belong to the world
  being run rather than to something included in it.

- **Add a manifest** for your own model: drop a ``<model>.manifest.yaml`` beside the MJCF listing the
  plugins (same shape as a world's ``components:``); the entity name is filled in for you, and each
  injected plugin also inherits the spawn's ``prefix`` — so a build-time plugin that welds geometry
  onto a spawned body (e.g. ``fiducial_marker`` with ``attach_to: wrist_3_link``) resolves the
  prefixed body name without the world having to know it. ``ur10e_custom.manifest.yaml`` ships an
  ``arm_controller``, an eye-in-hand ``realsense_d415``, and a ``fiducial_marker`` on the wrist.

A ``model:`` name is resolved across **all** installed packages, so a world can spawn a model that
lives in a different package from the spawn plugin. Register a package's models once::

   # pyproject.toml
   [project.entry-points."roqsim.models"]
   roqsim_assets = "roqsim_assets.models"   # module exposing MODELS_DIR

Then ``{model: conveyor}`` finds it by bare name (or ``{model: roqsim_assets:conveyor}``
to be explicit). A model that reuses another package's meshes need not copy them — add an
``assets: <provider>`` key to its ``<model>.manifest.yaml`` to borrow that provider's mesh/texture
dirs (e.g. ``assets: roqsim_manipulation_assets`` for a custom arm variant that keeps the stock meshes).

.. code:: yaml

   # turtlebot4.manifest.yaml — shipped next to turtlebot4.xml
   components:
     - diff_drive: {}
     - lidar:
         site: lidar
         rays: 360
         max_range: 12.0
     - oakd_camera:                   # renders: needs a GL backend (roqsim selects one on import)
         camera: oakd_rgb

Selecting a policy
------------------

A policy-driven controller can be pointed at a different checkpoint with ``policy:``::

   - spawn_robot: {model: unitree_g1}
     name: robot
     components:
       - g1_locomotion: {policy: g1_stand, station_keeping: true}

``policy:`` names a directory under the family's ``policy/`` holding ``<name>.spec.yaml`` and its
checkpoint (an absolute or relative path to a spec also works, for an out-of-tree policy). The spec
declares the observation layout, the joints the policy commands, the joints it merely *observes*, the
control gains, and the envelope it was trained for -- so adding a policy is dropping in a directory, not
editing Python. Omit the key and the controller uses its bundled default, unchanged.

Note the envelope is recorded, not enforced: outside it a policy does not fail, it balances worse. Check
it against what a world actually contains (``spec.envelope.check_payload(mass)``) rather than trusting a
port log to be read.

Navigation: detecting a collision
---------------------------------

A navigation trial usually fails by touching something, and ``contact_monitor`` is the observable
for that. It watches an entity's whole kinematic subtree (chassis *and* wheels) and reports any
contact against a geom outside its ``ignore`` list::

   - spawn_robot: {model: turtlebot4}
     name: robot
     components:
       - contact_monitor: {ignore: [floor], min_force: 1.0}

Two things are worth knowing before reaching for a proximity check instead:

* **Define the exception, not the rule.** A wheeled robot is in permanent, intended contact with the
  ground, so the plugin's contract is "everything counts except what you list". Listing what a robot
  may touch is short and stable; listing what it may not is neither. An ``ignore`` entry that matches
  no geom logs a warning rather than passing silently — that mismatch is exactly how a ground plane
  starts being scored as a collision.
* **A latched report is a failed trial.** With ``latch: true`` (the default) the report stays true
  after the robot bounces off, because a trial that hit something does not become clean again. Set
  ``latch: false`` for a live "am I touching anything right now" signal.

**How close did it come?** ``clearance_monitor`` is the companion, and deliberately its opposite
number. Contact is a bit: every configuration that does not touch scores identically, which is the
right failure criterion and a poor thing to optimise. Distance is the same question asked
continuously, so the two together give a verdict and a gradient over one geometry::

   - spawn_robot: {model: turtlebot4}
     name: robot
     components:
       - contact_monitor:   {ignore: [floor], min_force: 1.0}   # did it touch?
       - clearance_monitor: {ignore: [floor], distmax: 3.0}     # how close did it come?

Both are **components of the robot** — nested under the entry that spawns it, so ownership is the
shape of the document and there is no entity name to spell wrong. Their ``ignore`` lists should
agree: if one excludes the floor and the other does not, clearance reads zero forever while contact
reads clean, and the two describe different worlds.

Three things about it:

* **It measures real geometry.** ``mj_geomDistance`` against the actual shapes, so no footprint
  radius sits between the simulator and the number — the same objection this section raises against
  proximity checks applies to a circle drawn round a base. An articulated obstacle's nearest part is
  a limb, and that is what it finds and names.
* **Collision masks are respected.** ``mj_geomDistance`` itself ignores them, so a render-only geom
  would otherwise be reported as clearance to something the robot passes straight through — a
  pedestrian model carries six of those beside fifteen solid ones. Candidates and watched geoms are
  filtered by MuJoCo's own pairing rule, on both sides.
* **It never ends a trial.** ``contact_monitor`` latches and is what a scenario fails on; this
  observes and does not. Two plugins reporting one failure by different rules is how a trial starts
  disagreeing with itself, and a clearance threshold is precisely the tunable number the contact
  oracle exists to avoid. A scenario that *wants* to stop on a near-miss reads the endpoint and
  decides — with the threshold then stated in the experiment, where it belongs.

``compute_rate_hz`` (default 200) is separate from the publish ``rate_hz`` because measuring is a
distance query per geom pair: every physics step it cost about a fifth of the step budget on a nav
world, against a budget the simulator may already be over, while 200 Hz resolves ~1.5 mm at walking
pace — finer than anything downstream consumes. Beyond ``distmax`` the report reads that cutoff with
``saturated`` set, which says "at least this far" rather than offering a number that looks measured.

The endpoint payload carries the geom pair and the time of the *first* qualifying contact, so a
failure is attributable rather than merely flagged.

Over ROS 2 it is a ``std_msgs/Bool`` on ``collision`` — the same topic and type a Gazebo stack's
contact-sensor aggregator publishes, so a scenario's failure check ports between simulators
unchanged. The topic is *relative*, so it is scoped by the entity's namespace: two robots spawned
with ``namespace: a`` / ``namespace: b`` get ``/a/collision`` and ``/b/collision``, each attributable
to its own robot. Un-namespaced, two monitors publish on one ``/collision``, which is still a usable
"did anything hit something" signal but tells you nothing about which robot — and one monitor's
``False`` interleaves with the other's ``True``, so only a test *for* ``True`` is meaningful there.

.. note::

   Check the **value**, not the arrival of a message. The monitor publishes at ``rate_hz`` from the
   first step, including ``False``, so a scenario action that merely waits for data on ``/collision``
   fires immediately and fails every trial. A Gazebo aggregator that only starts publishing after a
   contact makes "wait for any message" look correct; it is not portable.

What a robot carries
--------------------

``payload`` adds a carried mass to one body of an entity -- a load is a property of the trial, not
of a robot family, so it is stated in the world and swept like any other factor::

   - spawn_robot: {model: turtlebot4}
     name: robot
     components:
       - payload: {mass: 2.5}      # kg, on the entity's root body

It is a **point mass at the body's centre of mass**: mass adds, and a point mass contributes no
inertia about its own centre. An ``offset`` is refused rather than approximated -- an offset payload
shifts the centre of mass and adds a parallel-axis term, which is a different body, not a heavier
one. ``mass: 0`` leaves the model untouched, so the unloaded cell of a sweep is identical to a world
that never declared a payload. Load a body other than the root with ``body:`` (the entity's spawn
prefix is applied for you), and a robot other than the owning entity with ``robot:``.

Where thrust is bounded this is the flight envelope rather than a detail: see
``roqsim_aerial/README.md``, which measures a quadrotor's hover collapsing at a thrust-to-weight
ratio of 1.

Injecting a physical fault
--------------------------

Some trials need something to *go wrong* at a chosen instant: a gripper that stops holding halfway
through a carry, a wheel that loses traction, a payload that changes under load. ``model_override``
makes that a property of the world rather than something scripted into whatever drives the robot ---
name a model field, name the objects, name the target value, and let the scenario switch it on::

   - model_override:
       overrides:
         - field: geom_friction
           select: [gripper_right_left_pad1, gripper_right_left_pad2,
                    gripper_right_right_pad1, gripper_right_right_pad2]
           to: 0.0
     name: grip_fault

and, for a wheel that keeps its grip only until it does not::

   - model_override:
       overrides: [{field: geom_friction, select: [wheel_left_tyre, wheel_right_tyre], to: 0.02}]
     name: traction_fault

The plugin is inert until fired, so adding it changes nothing about a nominal run.

**Firing it is a service call, and the reply is the point.** Over ROS 2 the inbound endpoint is a
``std_srvs/SetBool``, so a scenario does::

   service_call(service_name: '/grip_fault/override', service_type: 'std_srvs.srv.SetBool',
                data: '{\"data\": true}', response_variable: 'fault')

and can *fail the trial* when the reply says the fault did not land — a topic publish could only be
followed by hoping. In a ROS-free stepped run the same switch is
``ctx.blackboard.require("model_override:grip_fault").set_active(True)``, which is what an ``.osc``
action calls. Severity is not on the wire: it is the configured ``to:`` value, so sweeping how
slippery the pads get is an ordinary experiment factor rather than a runtime message.

Three things to know before writing one:

* **Which side of a contact you select decides whether anything happens.** MuJoCo takes a contact's
  friction from the geom with the higher ``priority``, and at *equal* priority the element-wise
  **maximum** of the two. So a default-priority wheel written to ``0.0`` against a floor declaring
  ``1.0`` changes nothing, and overriding a carton held by pads that carry ``priority="1"`` changes
  nothing either — select the geoms that *own* the contact, or select both sides. Use
  ``roqsim scenes describe <world> --overridable 'gripper_right*'`` to see the names, their current
  friction and their priority, rather than guessing.
* **A fault that did nothing says so.** One step after the change the plugin compares the *applied*
  contact against what it asked for and reports ``landed``, ``no_effect`` (a warning, and a failed
  service reply) or ``untested`` — the last meaning nothing was touching the selected geoms, which is
  not a failure. It is published as ``override_verified`` too, because a service call leaves no trace
  in a rosbag and ``mjModel`` is in neither the bag nor the state recording.
* **A reset returns the world to the configured state**, exactly, from the values read at startup.
  Without that, repetition 2 of a sweep cell would start already faulted and report a plausible
  wrong number — ``Engine.reset`` resets ``MjData`` and never touches ``MjModel``.

Not every model value can be written at runtime; ``geom_size`` and the ``opt.*`` globals are refused
by name, with the reason and with what to use instead (for the globals, ``sim.contact_override``,
which is *global* and applies *before compile* — a different tool for a different job). The full
allowlist, with what each field does and how it can silently do nothing, is in the plugin's own
``Config::`` block above and in ``roqsim scenes describe``'s ``overridable.fields``. Details and the
measurements behind each row: :ref:`architecture <92-physical-faults-impl>` §9.2.

Perception ground truth
-----------------------

Two plugins answer two different questions about the same objects, and an experiment usually wants
one of them, not both. ``object_detector`` reports an object's POSE in the robot's frame -- what a
manipulation stack consumes, and what a real pose estimator would output. ``segmentation_camera``
reports which PIXELS an object covers -- what an IoU, a mask AP or a training set is computed from::

   components:
     - spawn_robot: {model: turtlebot4}
       name: robot
       components:
         - segmentation_camera:
             camera: oakd_rgb
             classes:
               - {class_id: 1, name: parcel, bodies: ["graspable_*"]}
               - {class_id: 2, name: person, entities: [walker_1]}
             instances: true

That publishes a ``mono8`` class image, a ``16UC1`` instance image and a
``vision_msgs/Detection2DArray`` of tight boxes, all off one render through the named MJCF camera.

Three properties are worth knowing before a metric is built on it. Boxes measure the **visible**
extent, because that is the only extent derivable from a mask and the only one a detector could have
produced -- an occluded object shrinks and, below ``min_pixels``, is not reported at all. Instance ids
are **body ids**, so they are stable across frames and runs rather than depending on the order things
were seen in; they are correspondingly not contiguous, which is why the detections name them.
And class id **0 is background**: a declared class with id 0 would be indistinguishable from an
unlabelled geom, so it is refused at load.

What a run cost
---------------

``energy_monitor`` is the third observation plugin, beside the two that watch geometry: it meters the
actuators that move a robot and integrates their mechanical power, so "energy per metre", "how far on
a charge" and "which planner is cheaper" become numbers a run produces rather than numbers an
analysis fits::

   components:
     - spawn_robot: {model: turtlebot4}
       name: robot
       components:
         - energy_monitor: {efficiency: 0.72, idle_w: 8.0, capacity_wh: 26.0, voltage: 14.4}

The split between measurement and assumption is explicit, and the defaults assume nothing.
``force * velocity`` per actuator is measured, every step, at the physics rate -- reconstructed from
a recording afterwards it would be sampled at the recording's rate and need a drivetrain model to
turn poses back into effort, which is a fitted constant between the simulator and the result.
``efficiency``, ``idle_w`` and ``regenerative`` are the platform's own numbers; unset, the plugin
reports mechanical work and nothing else. A state of charge exists only where a ``capacity_wh`` was
given -- without one the fraction is reported as *unknown* rather than as a full battery.

Which actuators count is derived, not configured: every actuator driving a body of the robot's
kinematic subtree, so a world's other machines are not on this robot's bill and a model that gains a
joint does not need the world edited. An entity with no actuators is an error, because a meter
reading zero forever looks exactly like a robot that costs nothing to drive.

**It reports; it does not intervene.** A depleted battery latches and is published; the robot keeps
driving. Ending a trial is the experiment's decision, the same line ``contact_monitor`` draws about a
collision -- a scenario reads the endpoint and stops the run itself.

Degrading a sensor mid-run
--------------------------

``model_override`` changes the *physics*. Its counterpart changes what a sensor **reports** — a lidar
that starts dropping returns halfway down a corridor, a scanner whose noise triples when it fogs up.
That perturbation belongs in the sensor's own config (§9.1), not in a model field, so it is written
there and needs no plugin of its own::

   components:
     - spawn_robot: {model: turtlebot4}
       name: robot
       components:
         - lidar:
             range_stddev: 0.01
             dropout_percent: 2.0
             fault: {dropout_percent: 60.0, range_stddev: 0.35}   # held while active

The sensor is nominal until the fault is switched on, so adding a ``fault:`` block changes nothing
about a run that never fires it. A scenario switches it by the sensor's **address**::

   set_sensor_override(instance: 'robot.lidar', active: true)
   wait elapsed(8s)
   set_sensor_override(instance: 'robot.lidar', active: false)

and over ROS 2 the same switch is a ``std_srvs/SetBool`` at ``robot/lidar/override``, with
``robot/lidar/override_state`` and ``.../override_verified`` reporting back. The address is the dotted
path of labels with dots as slashes, because a dot is not legal in a ROS name; a bare ``lidar`` would
name neither of a robot's two lidars.

It mirrors ``model_override`` in the three ways that matter, rather than re-deciding them:

* **Severity is configured, not sent.** The ``fault:`` values are ordinary config, so sweeping how bad
  the fault gets is ``components.robot.lidar.fault.dropout_percent`` — an experiment factor,
  deterministic per cell and in the run's provenance. One bit crosses the wire.
* **The world never decides when.** No time trigger, no condition trigger; a fault's timing is the
  experiment's independent variable.
* **A fault that changed nothing is reported as such.** Applying a block whose values already equal
  the nominal reports ``no_effect``, and ``set_sensor_override``'s ``require_landed`` fails the trial
  on it — an unfaulted outcome wearing a faulted label is worse than a failed run. A *restore* has
  nothing to verify and reports ``untested``.

**Only keys the sensor reads per frame may be written.** Each sensor declares its own allowlist; on
the ray-casting sensors that is ``range_stddev``, ``dropout_percent``, ``max_range``, ``range_min``
and ``rate_hz``, and on the ``imu`` it is the noise, the biases and ``orientation`` -- so a trial can
drop the attitude channel or triple the rate noise partway through, which is what an IMU failure
looks like to a localisation filter. Everything else is refused **at load**, by name, with the reason — ``rays``,
``angle_min`` and ``angle_max`` because they change a ``LaserScan``'s length or the bearing its
indices mean, and ``site``/``frame_id``/``exclude_body`` because they are consumed once at
``configure``. This is the ``geom_size`` lesson from the physics channel: a value that writes fine,
takes effect nowhere, and reads back as though it had is worse than one that is refused.

A fault does not survive ``reset``: one process serves several trials, and a fault leaking into the
next would quietly turn a nominal control cell into a degraded one.

Manipulation: an arm on a linear axis
-------------------------------------

A gantry, a ceiling track or a seventh-axis floor rail is a prismatic joint *carrying* the arm base.
``spawn_arm``'s ``rail:`` expresses it, and it is the one thing ``mount:`` cannot: ``mount`` welds the
arm to a body that already exists, while a rail has to introduce the moving carriage itself::

   components:
     - spawn_arm:
         model: ur10e
         name: ur10e
         prefix: "ur10e_"
         pos: [0.0, 0.0, 2.6]              # where the axis sits
         rpy: [3.14159265, 0.0, 0.0]       # rolled 180 deg: the arm hangs from the ceiling
         rail: {axis: [1, 0, 0], range: [-2.0, 2.0], home: 0.0}

What this buys is **kinematic redundancy**: a 6-DOF arm on a rail is a 7-DOF system, so a task pose
has a one-parameter family of solutions and a planner can trade base travel against arm posture.

Three facts other code depends on:

* **The rail is joint 0.** Its MJCF joint and actuator are declared before the arm's, so
  ``arm_controller`` publishes and commands ``[rail_joint, <arm joints...>]`` — matching a URDF with
  the prismatic joint at the root of the chain, which is what MoveIt plans against.
* **``home`` stays the arm's joint vector**; the carriage's start is ``rail.home``. Folding the rail
  into ``home`` would invalidate every per-model default (a 6-value ``ur10e`` home would land on
  ``[rail, j1..j5]`` and leave ``wrist_3`` unset).
* **The carriage and track geoms are visual only.** A ceiling track that collides traps the arm
  against its own support from the first step, and the collision model a planner reasons about comes
  from the URDF/planning scene, not from these geoms. Model real structure as scene geometry.

``roqsim export urdf`` handles such a robot: a jointed root is emitted below a synthetic fixed ``world``
link (``--world-link``), which is the standard URDF spelling for a rail and keeps the fixed root MoveIt
requires. A root body with a **free** joint still becomes a fixed root — a floating base belongs in TF
(``odom -> base_link``), not in the description.

The SRDF is where that base reappears, and ``roqsim export srdf`` reads it off the model rather than
assuming it: a base riding a MuJoCo free joint gets a ``planar`` ``virtual_joint`` (``--base-joint
floating`` for a full 6-DOF one), while a base welded to the world — an arm on a pedestal, or a rail
whose DOF is already a URDF joint — gets none, which is how MoveIt spells a bolted-down robot. Declaring
a virtual joint nothing publishes does not fail loudly; move_group logs ``Missing virtual_joint`` and
never assembles a complete robot state, so ``--base-joint planar`` on a welded model is refused.

Manipulation: the whole MoveIt configuration
--------------------------------------------

``move_group`` needs six files, and the two above are the two a human would call the robot
description. ``roqsim export moveit`` writes all six:

.. code-block:: bash

   roqsim export moveit --world cell.yaml --out cfg/ --tip-site pinch --check

.. code-block:: text

   cfg/
     <arm>.urdf   meshes/     the kinematics, FK-checked against the MJCF
     <arm>.srdf               groups, named states, a sampled collision matrix
     kinematics.yaml          a solver over the chain the SRDF names
     joint_limits.yaml        the kinematic limits that time a geometric path
     moveit_controllers.yaml  which action a trajectory is executed against
     ompl_planning.yaml       the planner -- a starting point, meant to be overridden

The invariant is the same one the URDF export exists for, extended to the rest: **the configuration
MoveIt plans against is derived from the model the simulator loads**, and ``--check`` fails the export
when the two disagree by more than a micrometre.

That matters most for the file that looks least interesting. ``moveit_controllers.yaml`` maps MoveIt's
controller names onto the actions this substrate's *bridge* serves and onto the joint list
``arm_controller`` publishes — so it is read from the ``Endpoint`` objects the controller declared,
not restated. A name written by hand there can be right on the day and wrong after a world renames a
controller, and the failure is that a trajectory is executed against nothing.

Four more answers come off the model rather than from flags, each because getting it wrong is quiet:

* **the joint list and its order** — the ``ArmHandle`` the controller published, i.e. the names that
  reach ``/joint_states``. ``robot_state_publisher`` matches by name, so a list that is right about the
  robot and wrong about the order leaves MoveIt planning from a pose the arm is not in.
* **the home posture** — ``data.qpos`` after setup, not the world's ``home:`` key. The controller
  applies that key itself and a ``rest`` stance may overlay it, while MoveIt's start-state bounds check
  runs against the posture the arm is really in.
* **the collapse root** — the lowest common ancestor of every body an ``equality`` constraint touches.
  A closed linkage is exactly what URDF cannot express, and MuJoCo says where one is; collapse it and
  the loop is gone, miss it and the URDF keeps revolute DOFs nothing publishes.
* **``start_state_max_bounds_error``** — emitted only for an arm that has a *continuous* joint. MoveIt
  maps such a joint onto [-pi, pi] and ``CheckStartStateBounds`` then refuses to plan from a start
  state that has drifted a hair outside it, which surfaces as a phase failing instantly with
  ``START_STATE_INVALID`` right after a phase that succeeded — at a different phase each run. A
  range-limited arm has no such problem and gets no such setting.

**Pass ``--tip-site``.** Without it the arm chain ends at the tool flange, and a goal for the
fingertips has to be written as an offset from there — which multiplies every orientation tolerance by
that lever arm, so 0.15 rad of permitted tilt becomes ±33 mm at the fingers. One cell measured 61 mm of
lateral error against 12.2 mm of jaw clearance: MoveIt had satisfied the goal exactly, and the goal was
about the wrong point. ``--tip-site pinch`` emits a frame link at the gripper's own grasp site (through
a collapsed parent, where such a site usually sits), so a 3 mm position tolerance means 3 mm at the pads.

What it does **not** write is a ``planning.yaml``. The planning frame, the group name and the gripper's
units belong to whatever node drives the trial, and that is the experiment's file, not the substrate's.

Manipulation: what a contact task needs
---------------------------------------

``contact_monitor`` (above) treats contact as the failure, and ``model_override`` (above) can take a
contact away on command. A contact-rich manipulation task inverts both: contact *is* the task, and the
measurement is the wrench, not the trajectory. Four plugins make
that chain, and they are listed in a world in this order because each needs the previous one's
blackboard handle::

   - spawn_arm:            {model: ur5e, name: ur5e, prefix: "ur5e_"}
   - force_torque:         {name: ft, arm: ur5e, site: fts_site, frame: world}
   - peg_in_hole.py:PegInHolePlugin: {arm: ur5e, clearance: 0.001, hole_pos: [-0.49, -0.13, 0.0]}
   - cartesian_admittance: {arm: ur5e, ft: ft, law: admittance, site: tool_site}
   - insertion_task.py:InsertionTaskPlugin: {arm: ur5e, ft: ft, law: admittance, target_pos: [...]}

**Read the refs, not the order.** Three are named — ``spawn_arm`` and ``cartesian_admittance`` from
``roqsim_manipulation``, ``force_torque`` from ``roqsim_sensors`` — and resolve through entry points because
they are substrate: an arm, a sensor, a control law. Two are paths, and ship with the experiment: the
bored block whose clearance is the experimental variable, and the trial protocol built around it.

That division is the general one. The substrate owes a cell the *mechanism* — mount an arm, measure a
wrench, close a Cartesian loop. What is being inserted into what, and what counts as having inserted
it, is the experiment's to state.

Three things decide whether such a world measures anything at all:

* **Where the sensor cuts.** A site force sensor reports the wrench transmitted *through* that site
  from the body's children, so the tool must hang **below** it. A peg attached above the measurement
  site produces a wrench that is identically zero — which looks like a well-behaved controller, not
  like a broken world. The ``ur5e`` model ships ``fts_site`` (the cut) and ``tool_site`` (the attach
  point, further out) so the two cannot be confused.
* **Gravity and tool mass.** A real FT sensor is tared against the tool's weight before a
  measurement; a simulated one is not. With gravity on and a realistically-massed tool, any metric
  that integrates force is dominated by a static offset. Either set ``sim.gravity: [0, 0, 0]`` or give
  the tool a near-zero mass.
* **The controller's plant, not its gains.** ``cartesian_admittance`` closes a loop around
  ``arm_controller``'s position servo, which is stiff. Admittance gains taken from a system with a
  soft joint controller will oscillate and diverge on contact. Tune against a stability criterion
  fixed in advance, and record the result as a calibration — the ``ur5e`` model is the worked
  example, including the sweeps.

A trial plugin of this shape — approach → act → succeed/timeout/abort → write — calls
``ctx.request_stop()`` when it resolves, so a run ends when the trial does instead of being
padded to a guessed ``--seconds``. Two rules are worth copying from a trial-protocol plugin: give it
an explicit failure condition as well as a success one (a trial that can only succeed cannot produce a
success *rate*, it can only hang), and write the raw observable rather than the metric, because a
force-energy definition belongs to the analysis where it can still be argued with.

Manipulation: what a grasping world needs
-----------------------------------------

Four things have to line up before an object can be picked up, and three of them are opt-in because
they cost something a navigation world should not pay:

1. **A movable object.** Every prop in ``roqsim_assets`` is welded scenery by default. ``spawn_model``'s
   ``free: true`` adds a ``<freejoint/>``, registers the joint as the entity's ``base_joint`` (which is
   what ``simulation_interfaces``' ``SetEntityState`` requires to re-seat it), and re-seats it on
   ``reset`` so repetitions of a trial really are repetitions. Pair it with ``publish_tf: dynamic`` —
   nothing else publishes a free body's pose. ``graspable_box`` is the reference prop, sized and
   contact-tuned for a parallel gripper.
2. **Solver effort.** ``sim: {noslip_iterations: 10}``. Without it a firmly held object creeps out of
   the jaws; see "Solver options" in ``architecture.rst`` for the measurements.
3. **Scoped actuator ownership**, if the arm shares its entity with anything else. ``arm_controller``
   claims every joint actuator matching the entity prefix by default, which is right for a standalone
   arm and wrong for a humanoid or a mobile manipulator — it then also claims the legs or the wheels and
   fights their owner, writing position targets into what may be torque actuators. Give it ``joints:``
   (and ``gripper_actuator:``, which cannot be inferred once one entity carries two grippers), and each
   controller also reports only its own joints, so several can share one ``/joint_states`` topic.
4. **``mass`` / ``friction``** on the spawn, if either is a factor you want to vary — they are ordinary
   world-YAML keys, so an ordinary parameter sweep varies them and needs no new
   variation plugin.

``unitree_g1_dex1``'s manifest is a worked example of (3): two ``arm_controller`` instances on one
entity, each owning its seven arm joints and its own Dex1 gripper, alongside ``g1_locomotion`` on the
twelve leg motors.

Two more if the target **moves** (dynamic grasping):

5. **A conveyance, not a mover.** ``prop_trajectory`` (in ``roqsim_assets``) carries the object along a
   prescribed 2-D path by friction, on force-driven slide joints. Reach for it rather than ``moving_box``
   or ``walker``: both are **mocap**, and a mocap body has no velocity in MuJoCo's dynamics, so friction
   against it transfers no tangential force — it will slide out from under the object it is supposed to
   be carrying while staying in contact. Mocap blocks; a driven joint carries.
6. **A velocity command path**, if the controller is reactive. Resolved-rate and QP whole-body
   controllers emit joint *velocities*, and this plugin's actuators are position servos, so
   ``arm_controller``'s ``velocity_commands: true`` integrates ``target += qd·dt`` into the held target
   (clamped to the joint range, with a ``velocity_timeout_s`` watchdog so a dropped stream cannot leave
   the arm drifting). Position servos are kept deliberately — a MuJoCo ``<velocity>`` actuator sags under
   gravity at zero command. Be aware of what that costs a *metric*: the achieved profile carries the
   servo's own dynamics, so where end-effector acceleration is the measured quantity, check the tracking
   error and report the gains as part of the setup.

An arm carried by ``spawn_robot`` also needs ``arm_controller``'s ``rest`` stance: a robot spawn sets
the base pose and no joint stance, so the arm falls back to ``qpos0``. For the Panda that is not neutral
but an actively bad pose — its ``link5`` and ``hand`` collision geoms overlap by 0.030 m at all-zeros.
``rest`` seeds the spawn ``qpos`` *and* the held target by joint name, and re-seats on reset so repeated
trials start identically. ``frankie``'s manifest is the worked example of (5), (6) and ``rest``.

Scoring the trial, not self-reporting it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trial's **verdict belongs to the experiment**, not to the substrate — deciding what counts as
success is the thing the experiment is for. ``tiago_pick``'s ``pick_place_metrics`` is the
reference implementation, and the pattern generalises even though the plugin is not shipped here:

.. code:: yaml

   - pick_place_metrics:
       target: parcel
       grasp_links: [gripper_fingertip_left_link, gripper_fingertip_right_link]
       container: dropbox          # omit to score a pick alone
       lift_check: 0.05            # m the object must rise ...
       hold_s: 5.0                 # ... and stay held, both pads on it
       metrics_out: metrics.csv

It reports ``picked`` (lifted and *stayed* held through a dwell, with a
``contact_debounce_s`` tolerance so a momentary loss of one pad is not a slip), ``placed`` (came to
rest inside the container and supported by it), and ``success``. Nothing in it knows what arm,
gripper or base it is watching — the gripping links are named in config — so the same rule scores
different platforms.

What matters structurally is not which *package* holds it but which *side* it sits on: the verdict
must be on the SIM side, out of reach of whatever is driving. Two reasons:

* **Comparability.** When an in-process phase machine and an external MoveIt node solve the same
  trial, each scoring itself makes the comparison partly a comparison of two verdicts.
* **Observability.** Contact state is not on the ROS graph, and neither is "resting inside the box":
  a client can see the object's TF, but a parcel balanced on the rim looks identical to one on the
  floor of the container.

A driver that wants to react to a slip reads ``held()`` and decides for itself — the observer
reports and never terminates. Keep the rule platform-agnostic anyway (name the gripping links in
config rather than hard-coding them): that is what would let it move up into the substrate the day a
second experiment scores a pick.

Manipulation: where a workpiece lives
-------------------------------------

The substrate ships **robots**, not the things they work on. A bored block whose clearance is swept
at 0.1 mm, or an intersecting-pipe weldment whose dimensions were chosen because the paper never
stated its own, is one experiment's geometry: reused by nothing, and installed by everyone if it sits
in a family package. So a workpiece ships with the experiment that defines it, and there is no
special mechanism for that — it uses the two doors any downstream package uses:

* an ``roqsim.models`` entry point naming a module with ``MODELS_DIR``, after which
  ``spawn_model: {model: my_workpiece}`` resolves by bare name;
* a **path-loaded plugin** (``my_workpiece.py:MyWorkpiecePlugin``, relative to the world file), for a
  workpiece whose geometry does not exist until the world configures it — a fit expressed as a
  number, or goal poses derived from the same definition as the shape they lie on. No entry point,
  no install, no wheel: the world and the plugin travel together.

What the substrate owes such a cell is the arm, the sensing, the control law and the trial
machinery — all of which are addressed by name and none of which know what is being welded or
inserted.

Writing your own
----------------

Subclass ``roqsim.plugin.Plugin``, implement the hooks you need, add ``validate_config``, and
either register a ``roqsim.plugins`` entry-point or reference the class directly from the world
YAML. Referenced directly, the ref (``my_pkg.mod:MyPlugin`` or ``./plugins/x.py:Foo``) contains a
colon and is the plugins-list entry's *key*, so **quote it** — ``- "my_pkg.mod:MyPlugin": {...}`` —
to keep the colon from splitting the key.

**If your config names a file, implement ``sources``.** ``roqsim.config.world_sources`` — what
``roqsim scenes inputs`` and the exporters' ``--manifest`` report, and what a run harness
stages a world by — walks the YAML's ``extends`` chain and the MJCF's assets. It cannot see
into a plugin's config, so a mesh or a CSV named there is invisible to every caller asking
"what does this world need?" unless the plugin says so. Return absolute paths; entries that do
not exist are dropped, and the hook must never raise (callers treat it as best-effort). It is
the same rule as ``transport_only``: a capability is declared by the plugin, never listed in
the core.

**And resolve it against ``self.base_dir``, not the CWD.** That attribute is the directory of the
world document the entry was declared in, so a path written beside the world resolves the same
wherever the world is opened from. That matters more than it sounds: the working directory is not
the document's directory in general, and need not be the same twice. A world may be opened by
absolute path from somewhere unrelated, or copied under a different root by a tool that stages its
inputs — and a CWD-relative path then names a different file, or none. ``resolve_model`` and
``load_manifest`` both take a ``base_dir`` for this; pass ``self.base_dir``. A bundled model *name*
is a provider lookup and is unaffected.

A *spawn* plugin can bundle a model's default plugins by implementing ``expand`` via
``roqsim.manifest.expand_manifest`` (see the manifest mechanism in the :doc:`developer_guide` /
architecture reference). See the porting playbook in the :doc:`developer_guide`.
