Sensor coverage
===============

Estimate how well a set of sensors observes a **fixed** world — *how much of the room and which
objects are seen, and by how many sensors (0..N)* — and search for a layout (how many, which type,
placed where) that reaches a target coverage. Implemented in ``roqsim_sensors.coverage``; the
2D heatmap needs the optional ``coverage`` extra (``pip install 'roqsim_sensors[coverage]'``, already
installed by ``make venv``), the 3D render needs only MuJoCo. Rendering works headless without any
setup — ``import roqsim`` selects an offscreen backend for the machine (see
:func:`roqsim.gl.select_offscreen_gl`); set ``MUJOCO_GL`` only to override it.

There are two front doors onto the same core:

* the **plugin** ``sensor_coverage_probe`` — a world-YAML toggle that reports the coverage of the
  sensors already in a world;
* the **CLI** ``roqsim sensors coverage`` — a placement-search workbench that evaluates *hypothetical*
  candidate mounts and iterates toward a target.

Concepts
--------

* **Field of view.** Every sensor reduces to one ``SensorFov`` — a posed angular sector: a camera is a
  rectangular ``FRUSTUM``, a lidar a ``CONE_BAND`` (azimuth × elevation band). It is extracted per
  sensor type by an adapter (see :doc:`architecture` › Sensor coverage), so a camera's FOV comes from
  its MJCF ``fovy``/resolution and a lidar's from its plugin defaults — never re-typed.
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

The plugin (world-YAML toggle)
------------------------------

List ``sensor_coverage_probe`` in a world's ``components:`` to compute coverage once (at ``configure``) and
write ``report.json`` plus a render; omit it for none. ``sensors: auto`` evaluates every MuJoCo camera
in the world; give an explicit list for lidars/Livox or hypothetical placements.

.. code:: yaml

   components:
     - spawn_sensor: {model: mid360, name: cam_a, pos: [3, 1, 2.4], rpy: [3.14159, 0, 0]}
     - sensor_coverage_probe:
         sensors: auto              # or a list of {type, pos, rpy, config}
         target: {k: 1, frac: 0.95}
         sample: {volume: true, objects: true, resolution: 0.25, heights: [0.3, 1.0, 1.7]}
         out: coverage              # writes report.json + render(s) here
         render: both               # 3d | 2d | both | none
         palette: coverage          # 'coverage' (red 0->green many) | 'density' (light 0->dark many)

.. code:: bash

   roqsim sim world.yaml --headless --steps 1

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

``placements.json`` is a list of ``{type, pos, rpy, config?}`` using catalog types. The
``estimate`` report carries ``achieved`` (coverage fractions), ``uncovered_regions`` (where to add a
sensor), ``per_sensor_contribution`` (redundant sensors have ``unique_points: 0``), and ``per_object``.
The refine loop — evaluate, read the gaps, adjust ``placements.json``, re-evaluate — is what the
``sensor-coverage`` skill drives.

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

If ``build_fov`` raises ``no adapter for '<type>'``, register one in
``roqsim_sensors/coverage/adapters.py`` (``@register_adapter("<type>")`` returning a ``SensorFov``)
and add a ``CATALOG`` entry in ``catalog.py``. The sensor plugin itself is never modified. See
:doc:`architecture` › Sensor coverage for the design.

Visualising a sensor's FOV directly
------------------------------------

Independently of coverage, ``spawn_sensor: {show_fov: true}`` draws a sensor's field of view in the
viewer/renders. Three paths, by what the model provides:

* a **camera** mount (the RealSense/Zivid models) synthesises a translucent view **frustum** from the
  camera's ``fovy``/aspect spanning ``fov_near``..``fov_range``, **always clipped against world geometry**
  into a visibility volume that stops at walls and objects (see below);
* a camera-less **lidar** (Mid-360, Robin W1G) synthesises a translucent angular **sector** shell from
  the datasheet angles (``h_min``/``h_max``/``v_min``/``v_max``) in its manifest's ``fov:`` block, using
  the very ray convention the capture plugin casts with -- a full 360deg dome for the Mid-360, a bounded
  forward 120deg x 70deg wedge for the Robin W1G;
* a **camera-less** model that ships a bundled ``_fov`` mesh reveals it, if neither of the above applies
  (none of the current models: the Zivid ships one but has a camera, so it takes the frustum path).

``fov_near``/``fov_range`` default to the **sensor model's own** ``fov: {near, far}`` block in its
``<model>.manifest.yaml`` (device knowledge lives with the device -- Zivid 1.3..5 m, D435 0.2..6 m,
Mid-360 0.1..40 m, Robin W1G 0.1..70 m), so ``show_fov: true`` alone draws the correct band; a world may
override either per placement. ``fov_near`` sets the near cap of the drawn volume, so its shape *is* the
valid detection band. A model that has a camera always synthesises its frustum from that camera, even the
Zivid (which also ships a bundled ``_fov`` envelope) -- the baked envelope stays hidden. The
``worlds/all_sensors_demo.yaml`` world shows every sensor's FOV and runs a coverage probe.

A synthesised camera frustum is **always** clipped into a **visibility volume** that stops at walls and
objects instead of passing through them (a ``fov_rays`` grid, default ``[32, 24]``, is cast from the
camera against the world) -- occlusion is unconditional, not a per-placement opt-in. It applies only to
synthesised camera frustums; a bundled envelope and a lidar sector are drawn un-clipped. The clip is a
static build-time snapshot of the world *built so far*, so list scene/floorplan plugins before the
sensors; dynamic bodies occlude at their spawn pose and the volume does not update at runtime. This is a
per-sensor visual (what one sensor can see); for the quantitative per-area overlap count across all
sensors use the coverage probe with ``palette: density``.

A sensor never occludes itself
------------------------------

A real device's lens sits on the *outside* of its housing, but a MuJoCo ``<camera>`` sits at the pose
the datasheet gives — millimetres **behind** the geom that models that face. A visibility ray
therefore leaves the origin already inside the sensor's own body, and without care the sensor's own
housing is the first thing it hits. So every FOV carries the body it is mounted on
(``SensorFov.body_exclude``, from ``cam_bodyid``/``site_bodyid``) and the ray cast excludes it — the
same ``bodyexclude`` mechanism the ``lidar`` plugin's ``exclude_body`` uses for a robot's chassis.

Worth stating because the symptom did not look like occlusion. The D435 mount's ``d435_front`` sits
4.3 mm ahead of its camera, which blocked the whole central cone while wide-angle fringe rays still
escaped — so a mounted camera reported a *plausible but low* number rather than an obvious zero, and
a narrow long-range sensor (the Zivid, near 1.3 m) reported exactly **0** coverage in a room it saw
perfectly well. Numbers from before this fix under-report every ``spawn_sensor``-mounted sensor and
are not comparable with numbers from after it. A *hypothetical* placement (``pos``/``rpy``, not
spawned) is unaffected: it has no body, because nothing of it exists to get in the way.
