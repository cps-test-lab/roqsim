Sensor coverage
===============

Estimate how well a set of sensors observes a **fixed** world — *how much of the room and which
objects are seen, and by how many sensors (0..N)* — and search for a layout (how many, which type,
placed where) that reaches a target coverage. Implemented in ``roqsim_sensors.coverage``; the
2D heatmap needs the optional ``coverage`` extra (``pip install 'roqsim_sensors[coverage]'``, already
installed by ``make venv``), the 3D render needs only MuJoCo. Rendering works headless without any
setup — ``import roqsim`` selects an offscreen backend for the machine (see
:func:`roqsim.gl.select_offscreen_gl`); set ``MUJOCO_GL`` only to override it.

There are three front doors onto the same core:

* the **plugin** ``sensor_coverage_probe`` — a world-YAML toggle that reports the coverage of the
  sensors already in a world, computed **once**, at ``configure``, for the mounts as placed;
* the **plugin** ``swept_coverage_monitor`` — the same geometry evaluated **repeatedly while the
  world runs**, accumulating the union of everything a *moving* sensor has covered over a run;
* the **CLI** ``roqsim sensors coverage`` — a placement-search workbench that evaluates *hypothetical*
  candidate mounts and iterates toward a target.

The first and third ask where to put a sensor; the second asks what a sensor that moves ended up
seeing. They are the same range → FOV → line-of-sight computation, so a number from one is
comparable with a number from another only when both sampled the same way — which is why all three
build their points with one sample-set builder.

Concepts
--------

* **Field of view.** Every sensor reduces to one ``SensorFov`` — a posed angular sector: a camera is a
  rectangular ``FRUSTUM``, a lidar a ``CONE_BAND`` (azimuth × elevation band). It is extracted per
  sensor type by an adapter (see :doc:`developer_guide` › Sensor coverage (analysis layer)), so a
  camera's FOV comes from its MJCF ``fovy``/resolution and a lidar's from its plugin defaults — never
  re-typed.
* **Coverage count.** For each sample point, the number of sensors that see it, gated by range → angular
  FOV → line of sight (occlusion by walls/furniture, but **not** by the sensor's own mount — see
  below). ``k=1`` means "seen by ≥1 sensor"; ``k=2`` is
  redundant coverage.
* **Sample targets.** ``volume`` is a 3D grid of free interior points (room coverage); ``objects`` are
  points on object surfaces, labelled by geom name (are these objects seen?).
* **Orientation matters.** With ``rpy = 0`` a sensor points along **+x** (world), up = +z. To look
  **down**, a camera needs ``rpy: [0, 1.5708, 0]``; an upright Livox (vertical band −7°..+52°) must be
  **inverted** (``rpy: [3.14159, 0, 0]``) or it sees almost nothing below it. Near-zero coverage is
  usually an orientation mistake.
* **Camera range is an assumption.** A camera has no physical far range; the ``far`` you give it is a
  *detection range* and too generous a value inflates coverage (depth cameras default it to their
  ``clip_far``).

The static plugin (world-YAML toggle)
-------------------------------------

List ``sensor_coverage_probe`` in a world's ``components:`` to compute coverage once (at ``configure``) and
write ``report.json`` plus a render; omit it for none. ``sensors: auto`` evaluates every MuJoCo camera
in the world but a device's depth camera, which is another stream of the device beside its colour
camera (``camera_common.DEPTH_CAMERA_SUFFIX``); give an explicit list for lidars/Livox or hypothetical placements.

.. code:: yaml

   components:
     - sensor_coverage_probe:
         sensors:                   # or `auto`: every MuJoCo camera in the world
           - {type: livox_mid360, pos: [3, 1, 2.4], rpy: [3.14159, 0, 0]}   # {type, pos, rpy, config}
         target: {k: 1, frac: 0.95} # judged into report.json's target_met
         sample: {volume: true, objects: true, resolution: 0.25, heights: [0.3, 1.0, 1.7]}
         out: coverage              # writes report.json + render(s) here
         render: both               # 3d | 2d | both | none
         palette: coverage          # 'coverage' (red 0->green many) | 'density' (light 0->dark many)

.. code:: bash

   roqsim sim world.yaml --headless --steps 1

The swept plugin (a sensor that moves)
--------------------------------------

``sensor_coverage_probe`` answers "what do these mounts see from where they are". It cannot answer
"what did this sensor see over the whole run", because coverage accrues *between* pose samples and
because an occluder may itself move. List ``swept_coverage_monitor`` instead to evaluate the same
geometry at ``compute_rate_hz`` while the world runs and accumulate the union of everything covered.

The mount is ``frame:``, a frame path (:ref:`paths`) whose pose is re-read from the model on every
evaluation rather than captured once: an entity's root, body, site, camera or declared frame, or a
body, site or camera of the world's own MJCF by its MuJoCo name -- a sensor on a gantry in the world
file is ``frame: gantry`` or ``frame: gantry_cam``. Nested under an entity the path is relative to it
(``.`` is the entity) and a leading ``/`` starts at the top of the world. ``pose`` offsets the sensor
in the frame's coordinates, a ``geometry_msgs/Pose`` whose omitted components are 0.

On a **camera** frame the field of view is that camera's: its intrinsics come from the model, and its
axes are MuJoCo's camera frame (looking along -z, +y up), so ``pose: {position: {z: -0.3}}`` moves it
0.3 m along the view. ``config`` may override ``fovy``/``width``/``height`` and sets the detection
range. On any **other** frame ``config`` states the field of view in full, looking along the frame's
+x as the adapter's placement form does.

.. code:: yaml

   components:
     - spawn_robot: {model: turtlebot4}
       name: robot
       components:
         - swept_coverage_monitor:
             type: oakd_camera        # which adapter builds the field of view
             frame: oakd/oakd_rgb     # relative to `robot`: its OAK-D's camera, with its intrinsics
             config: {far: 5.0}       # the detection range, an assumption for a camera
             sample: {volume: true, resolution: 0.25, heights: [0.5]}
             compute_rate_hz: 5.0     # how often the union is updated
             rate_hz: 2.0             # how often the fraction is published
             out: coverage            # optional report.json at shutdown

It publishes a ``coverage`` endpoint carrying the covered fraction, and puts a reader on the
blackboard under ``swept_coverage:<address>`` that hands out the sample points and each point's visit
count — so a scenario can react to coverage live, and a metric can be recomputed afterwards from the
points rather than from a summary.

Two properties are worth relying on: the union is **monotonic** (a cell once covered never becomes
uncovered, so holding still raises the visit counts and leaves the union untouched), and an occluded
cell stays at **exactly zero** visits rather than near zero. Both are asserted in the package's
tests, as is the thing the plugin exists for: on a corridor fixture, the same sensor driven along it
covers at least twice what it covers standing still.

Unlike the static probe this plugin does no rendering, which is what lets it run in parallel with
other simulations; a coverage figure and a picture of it are separate jobs here.

The CLI (placement search)
--------------------------

.. code:: bash

   # sensor types, default FOV, cost, and mount constraints
   roqsim sensors coverage catalog

   # evaluate a placement set -> report.json + render
   roqsim sensors coverage estimate \
       --world <mjcf-or-world-yaml> --placements p.json --target k=1,frac=0.95 --render both --out run/

   # deterministic max-coverage baseline over auto-generated ceiling mounts
   roqsim sensors coverage greedy \
       --world <w> --target k=1,frac=0.9 --types livox_mid360,oakd_camera --mount-z 3.0 --out run/

   # target specific rooms only (per-region report; restrict the search to those rooms)
   roqsim sensors coverage greedy \
       --world <w> --regions scene/floorplan.json --region-names "room 1,room 2" --restrict \
       --types livox_mid360 --mount-z 3.3 --target k=1,frac=0.95 --out run/

``estimate`` and ``greedy`` take ``roqsim sim``'s ``--set`` and ``--override``, so the coverage measured
is that of the world a run with those overrides builds; they apply to a world YAML or ref, and are
refused for an MJCF ``--world``.

``placements.json`` is a list of ``{type, pos, rpy, config?}`` using catalog types. The
``estimate`` report carries ``achieved`` (coverage fractions), ``uncovered_regions`` (where to add a
sensor), ``per_sensor_contribution`` (redundant sensors have ``unique_points: 0``), and ``per_object``.
To refine a layout, evaluate, read the gaps, adjust ``placements.json`` and evaluate again.

**Per-region coverage.** ``--regions`` restricts the *question* to named areas without touching the
sampler: it takes a JSON of ``{name, polygon|bbox, z_min?, z_max?}`` regions **or** a scene's
``floorplan.json`` (rooms are reconstructed from the wall segments), and adds a ``per_region`` block
(``fraction_covered_k1/k2`` per room) to ``report.json`` so a whole-building fraction cannot dilute the
signal for the rooms you care about. ``--region-names "a,b"`` subsets them; ``--restrict`` confines the
sample points — and, for ``greedy``, the objective and the candidate mounts — to the region union, so
"cover *these* rooms with the fewest sensors" is a single command. Regions are world-agnostic
(:mod:`roqsim_sensors.coverage.regions`): the same ``Region`` drives both the report and the search.

The renders (both the 3D marker view and the 2D top-down heatmap) encode the per-area sensor **count**.
``palette`` (CLI ``--palette``, plugin ``palette:``) picks how: ``coverage`` (default) is a red→green
hue ramp reading "is it covered" (0 sensors red, many green); ``density`` is a light→dark ramp reading
"how densely" — 0 sensors lightest, each additional overlapping sensor darker — so redundantly-covered
regions stand out as the darkest areas. Both read the same ``counts``; only the colour encoding differs.

Adding a sensor the tool doesn't know
-------------------------------------

If ``build_fov`` raises ``no coverage FOV adapter for sensor type '<type>'``, register one in
``roqsim_sensors/coverage/adapters.py`` (``@register_adapter("<type>")`` returning a ``SensorFov``)
and add a ``CATALOG`` entry in ``catalog.py``. The sensor plugin itself is never modified. See
:doc:`developer_guide` › Sensor coverage (analysis layer) for the design.

**A catalog entry states policy, not optics.** Write ``cost``, ``mount`` and ``description`` -- how
this device may realistically be deployed -- and name the bundled ``model`` it describes. The optics
are read from that model: ``fovy`` and ``resolution`` off the MJCF camera its manifest wires to the
capture plugin, ``near``/``far`` off the manifest's ``fov:`` block. So a camera's field of view is
stated once, in the model, and the same numbers drive ``spawn_sensor: {show_fov: true}`` and a
coverage study. A lidar entry names no model and needs no optics at all -- its adapter instantiates
the plugin and reads the defaults it resolved.

Restating those numbers in the catalog is what an entry must not do -- a copy cannot be kept true by
attention, and a catalog that disagrees with the model it names is worse than one that says nothing
-- so there is deliberately no field to write one in. ``fov_overrides`` exists for the genuine exception -- a study pinning a camera's ``far``, which is
an analysis assumption rather than a property of the device.

Visualising a sensor's FOV directly
------------------------------------

Independently of coverage, ``spawn_sensor: {show_fov: true}`` draws a sensor's field of view in the
viewer/renders. Three paths, tried in this order:

* a **camera** mount (the RealSense/Zivid models) synthesises a translucent view **frustum** from each
  camera but a depth camera beside a colour one, from its ``fovy``/aspect spanning ``fov_near``..``fov_range``, **always clipped against world geometry**
  into a visibility volume that stops at walls and objects (see below);
* a **camera-less** model that ships a bundled ``_fov`` mesh reveals it (none of the current
  models: the Zivid ships one but has a camera, so it takes the frustum path);
* otherwise a camera-less **lidar** whose manifest's ``fov:`` block declares an angular band
  (``h_min``/``h_max``/``v_min``/``v_max``) synthesises a translucent angular **sector** shell from
  those datasheet angles, using the very ray convention the capture plugin casts with -- a full 360deg
  dome for the Mid-360, a bounded forward 120deg x 70deg wedge for the Robin W1G, a flat fan for each
  2D scanner.

``fov_near``/``fov_range`` default to the **sensor model's own** ``fov: {near, far}`` block in its
``<model>.manifest.yaml`` (device knowledge lives with the device -- Zivid 1.3..5 m, D435 0.28..6 m,
Mid-360 0.1..40 m, Robin W1G 0.1..70 m), so ``show_fov: true`` alone draws the correct band; a world may
override either per placement. ``fov_near`` sets the near cap of the drawn volume, so its shape *is* the
valid detection band. A model that has a camera always synthesises its frustum from that camera, even the
Zivid (which also ships a bundled ``_fov`` envelope) -- the baked envelope stays hidden. The
``worlds/all_sensors_demo.yaml`` world shows every sensor's FOV and runs a coverage probe.

A synthesised camera frustum or lidar sector is **always** clipped into a **visibility volume** that
stops at walls and objects instead of passing through them (a ``fov_rays`` grid, default ``[32, 24]``,
is cast from the camera, and a sector's own azimuth x elevation grid from the scan site) -- occlusion
is unconditional, not a per-placement opt-in. Only a bundled ``_fov`` envelope is drawn un-clipped: it
is a baked mesh. The clip is a static build-time snapshot of the world *built so far*, so list
scene/floorplan plugins before the sensors; dynamic bodies occlude at their spawn pose and the volume
does not update at runtime. This is a per-sensor visual (what one sensor can see); for the quantitative per-area overlap count across all
sensors use the coverage probe with ``palette: density``.

A sensor never occludes itself
------------------------------

A real device's lens sits on the *outside* of its housing, but a MuJoCo ``<camera>`` sits at the pose
the datasheet gives — millimetres **behind** the geom that models that face. A visibility ray
therefore leaves the origin already inside the sensor's own body, and without care the sensor's own
housing is the first thing it hits. So every FOV carries the body it is mounted on
(``SensorFov.body_exclude``, from ``cam_bodyid``/``site_bodyid``) and the ray cast excludes it — the
same ``bodyexclude`` mechanism the ``lidar`` plugin's ``exclude_body`` uses for a scanner's own housing.

Worth stating because the symptom does not look like occlusion. The D435 mount's ``d435_front`` sits
4.3 mm ahead of its camera, which unexcluded blocks the whole central cone while wide-angle fringe
rays still escape — so a mounted camera reports a *plausible but low* number rather than an obvious
zero, and a narrow long-range sensor (the Zivid, near 1.3 m) reports exactly **0** coverage in a room
it sees perfectly well. A study that lets a mount occlude its own camera therefore under-reports every
``spawn_sensor``-mounted sensor. A *hypothetical* placement (``pos``/``rpy``, not
spawned) is unaffected: it has no body, because nothing of it exists to get in the way.
