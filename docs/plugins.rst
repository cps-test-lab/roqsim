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
web/urdf/srdf``.

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
automatically, so a world just spawns the robot:

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
slippery the pads get is an ordinary campaign factor rather than a runtime message.

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
  Without that, repetition 2 of a campaign cell would start already faulted and report a plausible
  wrong number — ``Engine.reset`` resets ``MjData`` and never touches ``MjModel``.

Not every model value can be written at runtime; ``geom_size`` and the ``opt.*`` globals are refused
by name, with the reason and with what to use instead (for the globals, ``sim.contact_override``,
which is *global* and applies *before compile* — a different tool for a different job). The full
allowlist, with what each field does and how it can silently do nothing, is in the plugin's own
``Config::`` block above and in ``roqsim scenes describe``'s ``overridable.fields``. Details and the
measurements behind each row: :ref:`architecture <92-physical-faults-impl>` §9.2.

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
``ctx.request_stop()`` when it resolves, so a campaign cell ends when the trial does instead of being
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
   world-YAML keys, so a campaign sweeps them with ``ParameterVariationList`` and needs no new
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
``roqsim scenes inputs`` and the exporters' ``--manifest`` report, and what a campaign runner
stages a world by — walks the YAML's ``extends`` chain and the MJCF's assets. It cannot see
into a plugin's config, so a mesh or a CSV named there is invisible to every caller asking
"what does this world need?" unless the plugin says so. Return absolute paths; entries that do
not exist are dropped, and the hook must never raise (callers treat it as best-effort). It is
the same rule as ``transport_only``: a capability is declared by the plugin, never listed in
the core.

A *spawn* plugin can bundle a model's default plugins by implementing ``expand`` via
``roqsim.manifest.expand_manifest`` (see the manifest mechanism in the :doc:`developer_guide` /
architecture reference). See the porting playbook in the :doc:`developer_guide`.
