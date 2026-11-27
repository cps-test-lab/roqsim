Choosing a renderer
===================

Every image roqsim draws -- a camera's colour, depth and labels, ``roqsim render`` and its videos,
the thumbnails, the coverage pictures -- comes through one render service, and the renderer behind
it is a choice. MuJoCo's own renderer is the default; another is an installed package or a class
you name. How it works inside is :doc:`architecture` §8.

The lines a world writes
------------------------

.. code-block:: yaml

   sim:
     renderer: mujoco        # which renderer draws every camera; MuJoCo's when omitted.
                             # Any name `roqsim renderers list` shows, or module:Class / file.py:Class.
     render:                 # the render service's own settings
       mode: sync            # sync (default): the tick waits for its due frames, deterministic
                             # async: frames come from a render thread, stamped with their own sim time
       workers: 1            # instances of each renderer its cameras are dealt to (a context each);
                             # one camera's views (colour, its own depth optics) go to one instance
       timeout_s: 60         # how long a worker may take to open or to draw a batch waited for
     renderers:              # one settings block per renderer, validated by that renderer
       mujoco: {atlas: true} # `roqsim renderers describe mujoco` lists its keys

   components:
     - spawn_sensor:
         model: realsense_d435
       components:
         - realsense_d435: {renderer: mujoco}   # this camera only: overrides sim.renderer

A renderer another package adds is chosen the same way: ``sim.renderer: splat`` draws every
camera's view of a room captured as a Gaussian splat (:doc:`splat_rendering`).

A world that says nothing draws through MuJoCo's renderer, synchronously. From the command line
the same choice is one flag, which is ``--set sim.renderer=NAME`` applied last::

   roqsim sim world.yaml --renderer mujoco
   roqsim render world.yaml --out shot.png --renderer mujoco

MuJoCo's physically based renderer: ``filament``
-------------------------------------------------

``sim.renderer: filament`` draws through MuJoCo's Filament renderer: per-pixel light from any number
of lights, soft shadows, image-based light, metallic-roughness materials, ambient occlusion and colour
grading. It reads all of these from the model, so a world turns them on in its MJCF:

.. code-block:: xml

   <asset>
     <texture name="sky" type="skybox" file="room_sky.png"/>        <!-- the light around the room -->
     <material name="oak" texture="oak" roughness="0.6" metallic="0"/>
   </asset>
   <custom>
     <text name="filament.shadows.type" data="pcss"/>     <!-- soft shadows -->
     <numeric name="filament.cg.exposure" data="-0.5"/>   <!-- colour grading, in stops -->
   </custom>
   <worldbody>
     <light type="image" texture="sky" intensity="20000"/>
     <light pos="0 0 2.9" dir="0 0 -1" intensity="900" bulbradius="0.3"/>  <!-- candela -->
   </worldbody>

MuJoCo's rendering documentation lists every ``filament.*`` flag. **One light with a positive
``intensity`` switches the whole model to photometric light**: every light then gives exactly its own
intensity (lights left at zero go dark), the headlight is off and nothing lights the scene but the
lights and an ``image`` light. A model with none is drawn the way MuJoCo draws a model authored for
the classic renderer: a default environment light, the headlight (``visual/headlight``: on unless
``active="0"``; MuJoCo 4.0's ``auto`` is on) and the model's lights sharing a fixed intensity.

.. code-block:: yaml

   sim:
     renderer: filament
     renderers:
       filament:
         shadows: true          # draw shadows (their kind: the model's filament.shadows.type)
         post_processing: true  # tone mapping, grading, ambient occlusion, anti-aliasing, bloom
         reflections: true
         colour_textures: srgb  # a colour texture is read as sRGB, as the classic renderer
                                # shows it; `model`: by its colorspace
         software: false        # allow a CPU rasteriser where a GPU was given

It gives colour, depth and segmentation through pinhole, orthographic and free cameras, an off-centre
principal point, non-square pixels and a camera's own geom groups. It does not draw MuJoCo
decorations or the moving bodies alone, and a camera asking for either is refused at load.

- **Colour textures look as they do under the classic renderer.** The classic renderer shows a
  texture's bytes as display colours, and MuJoCo compiles a texture file left at
  ``colorspace="auto"`` to linear unless the file says otherwise; Filament honours the colourspace,
  and would draw such a texture lighter and paler than the same colour given as ``rgba``. So a
  material's colour texture is read as sRGB; a world whose textures are authored linear (for a
  physically based renderer) sets ``colour_textures: model``. Data textures (normal, roughness, ...)
  are always read by their colourspace.
- **It needs Vulkan.** It runs headless through Vulkan, never through EGL or a display: in a container
  the GPU's Vulkan driver must be mounted (for NVIDIA the ``graphics`` driver capability). Where no
  Vulkan context opens, the world fails to load with Filament's own words. A process given an NVIDIA
  GPU that would draw on a CPU rasteriser (lavapipe) is refused unless ``software: true``.
- **A frame is a function of the state only without post-processing.** Filament's dithering and
  ambient occlusion change from frame to frame by a few levels; ``post_processing: false`` draws
  identical frames for identical states, without ambient occlusion and tone mapping.
- **Its contexts live as long as the process.** Each open renderer holds a Vulkan context of its
  own, given back for the next world on its thread when it closes, and never destroyed: destroying
  one ends MuJoCo's own renderer in the same process (:doc:`architecture` §8).

What is installed, and what each takes
--------------------------------------

::

   roqsim renderers list               # every renderer, its package, one line about it
   roqsim renderers describe mujoco    # what it renders, and every setting with default and meaning

A renderer **declares** what it can give: which images (``rgb``, ``depth``, ``segmentation``),
which camera models (``pinhole``, ``orthographic``, ``free``), whether it is deterministic, whether
it runs in roqsim's process, whether it draws only the geom groups a camera names, whether it can
draw the moving bodies alone (robots and moved props, with nothing static: what a renderer that
brings its own static world, such as a captured splat, composites into it). When the world loads, every camera is checked against its renderer; a
camera asking for what its renderer cannot give fails the load, naming the camera, the renderer,
what was asked and what it gives. Nothing is drawn by MuJoCo's renderer instead behind your back.
An unknown renderer names the installed ones and the package that would add it, and a mistyped
settings key names the key, the block and the valid keys.

The two modes
-------------

``sync`` draws the cameras due in a tick before the plugins read them, and the tick waits: every
frame is the frame of the state it is published with, and a run is reproducible frame for frame.
It is what lockstep needs, so ``sim.sync.enabled`` refuses ``async``.

``async`` draws on a thread of its own from a copy of the state, while physics steps on. A frame
then shows a state a tick or more before the one it is published in, and carries that state's sim
time as its stamp, so tf and a perception stack stay consistent; a renderer slower than the ticks
skips stale states rather than falling behind. Use it when the real-time factor matters more than
having a frame for every tick. Every frame ``async`` publishes is the frame ``sync`` draws for the
same stamp, noise included.

On some GPUs MuJoCo's default 4x MSAA is not bit-reproducible from run to run (a handful of values,
one level); a world that needs identical bytes sets ``<visual><quality offsamples="0"/></visual>``
in its model.

.. _views-file:

Drawing a list of views
-----------------------

A picture drawn where a real camera stood -- its pose, its focal lengths, a principal point off the
image centre, its roll -- comes from a *views file*: a list of cameras, each a name, a size, ``fx fy
cx cy`` in pixels and a pose. Every renderer that declares ``posed`` draws it.

.. code-block:: yaml

   convention: colmap        # a preset, or {axes: ..., rotation: ..., pixel_origin: ...}
   near: 0.05                # optional, m; default 0.01
   far: 30.0                 # optional, m; default the model's own far plane
   views:
     - name: "2_1276"        # unique; the file stem of its images
       width: 353
       height: 628
       fx: 623.79
       fy: 626.54
       cx: 176.5
       cy: 314.0
       R: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # world to camera: looking along +y, level
       position: [0.462, -2.131, 0.543]

The file is JSON or YAML, read by its extension, and it names its convention: a preset, or all three
keys. A file that names neither is refused, and so is an unknown preset, naming the presets, the
keys and the command below; nothing is read with a default.

``axes``
   ``opencv``: the camera frame is x right, y down, z forward into the scene (also ROS's camera
   optical frame). ``mujoco``: x right, y up, looking down -z (MuJoCo, OpenGL).
``rotation``
   ``world_to_camera``: ``x_cam = R (x_world - position)``; ``t`` may be given in place of
   ``position``, ``x_cam = R x_world + t``. ``camera_to_world``: ``x_world = R x_cam + position``,
   the columns of ``R`` being the camera's axes in the world.
``pixel_origin``
   ``corner``: ``cx, cy`` count from the image's top-left corner, so the top-left pixel's centre is
   ``(0.5, 0.5)``. ``centre``: from the top-left pixel's centre, which is ``(0, 0)``.

The presets name the sets the data comes in, after where it comes from:

.. roqsim-view-presets::

A calibration's ``rvec`` and ``tvec`` are the ``opencv`` preset with ``R = cv2.Rodrigues(rvec)[0]``
and ``t = tvec``. Anything else is written as the keys: a camera-to-world pose in OpenGL's axes with
its principal point counted from the pixel corner is ``{axes: mujoco, rotation: camera_to_world,
pixel_origin: corner}``.

Under every convention: positions are metres in the world frame of the target as built (after its
components spawn); ``fx, fy, cx, cy`` are pixels of the image written, at its ``width`` and
``height``; x counts right from the image's left edge and y down from its top edge. The rotation is
exactly one of ``R`` (3x3 rows, orthonormal with determinant +1 to 1e-3, drawn as the rotation
nearest it, so a matrix written to a few decimals reads) or ``quat``: a list is
``w, x, y, z``, a mapping ``{x, y, z, w}`` names its parts as a ROS message does, and under ``ros``
only the mapping is admitted, since a ROS list is ``x, y, z, w``. The camera's centre is
``position``, a list or ``{x, y, z}``.

Drawing it
~~~~~~~~~~

.. code-block:: bash

   roqsim render <target> --views FILE --out DIR [--images rgb,depth,segmentation]
                 [--renderer NAME] [--state run.mcap [--at T]] [--set k=v] [--override f.yaml]
                 [--geomgroup N,...] [--check]

The target is anything ``roqsim render`` takes; with ``--state`` the recording names its world and
``--at`` picks the moment. Every view of the file is drawn by one renderer, opened once, in batches
from one state, so a file of thousands of views of different sizes is one process. ``--out`` is a
directory, and per view it holds each image ``--images`` asks for (colour alone by default, the same
for every renderer):

* ``<name>.png``: colour, 8-bit RGB as the renderer gives it;
* ``<name>.depth.npy``: float32 metres along the view axis (z-depth, not the distance along the
  pixel's ray), ``+inf`` where nothing was drawn;
* ``<name>.seg.png``: a 16-bit greyscale PNG, per pixel an index into the manifest's ``legend``
  (0: background; each other names the object's type, id, name, body and entity).
  :func:`roqsim.views.read_segmentation` gives it as uint16; an array made from the image as Pillow
  opens it is uint16 or int32, depending on the Pillow version, with the same indices.

``manifest.json`` lists what drew them: roqsim's version, the target (and the recording and its
moment), the renderer with its settings and capabilities, the world's conditions (``bake_used``),
the views file with its sha256 and its convention, the images, the depth convention, the legend,
and per view its size, its intrinsics and pose in roqsim's form, its clip planes and each file with
its sha256. A renderer that does not give an image asked for, or a posed camera, is refused with the
reason, before the world compiles where ``--renderer`` names it; nothing is drawn another way.
``--size``, ``--view``, ``--focus``, ``--camera``, ``--camera-path``, ``--overlay`` and
``--from``/``--to`` are refused beside ``--views``: the file states every camera, at one moment.

``--check`` draws nothing. It says per view, on stderr, how it read the file -- where the camera
stands, which way it looks and what is up in the world frame, and where the principal point lies in
the picture -- so that a wrong preset shows before a picture is drawn::

   view_0001 (colmap): stands at x -2.000, y 0.000, z 1.000 m; looks along +1.00 +0.00 +0.00,
   toward +x, level; up is +0.00 +0.00 +1.00, toward +z, rolled 0°; principal point 0.0 px right
   of and 0.0 px above the image centre (640x480)

It warns, without refusing, where a wrong convention typically shows: a picture whose up points
down the world's z ("upside down"), a principal point outside the image, or one more than a quarter
of the image from its centre. The JSON line carries the same per view under ``reading``.

From Python, :func:`roqsim.views.render_views` is the same, for a target or for a compiled model and
its data, and :func:`roqsim.views.write_views` writes what the command writes:

.. code-block:: python

   from roqsim.views import load_views, render_views, write_views

   views = load_views("views.yaml")
   for view, frame in render_views("roqsim_scenes:depot", views, images=("rgb", "depth")):
       frame.rgb, frame.depth          # render_api.Frame
   for view, frame in render_views((model, data), views):   # a model and its state, as they are
       ...
   write_views("renders/", render_views((model, data), views), {"target": "mine"})

Finding the conventions without the docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``roqsim views describe`` prints the keys, the presets and the units; with a preset it also prints
a ready example file in that preset's form, every key commented, which is a valid views file as it
stands. ``--json`` gives the same as one document.

.. code-block:: bash

   roqsim views describe                   # the conventions, as comments
   roqsim views describe colmap > v.yaml   # ...and an example file to fill in
   roqsim views describe ros --json

A renderer of your own
----------------------

Subclass :class:`roqsim.render_api.Renderer`, declare its ``capabilities`` and ``SETTINGS``, check
its block in ``validate_config(settings, base_dir)`` -- ``base_dir`` is the directory of the world being loaded, so a
file the settings name can be found missing, or of the wrong kind, when the world loads -- and
register it in your package. A relative path in renderer settings resolves against the world that
is loaded, also for settings it inherits through ``extends``; a world meant to be extended names its
files as ``<package>:<path>``, which resolve wherever the package is installed::

   [project.entry-points."roqsim.renderers"]
   myrenderer = "my_package.renderer:MyRenderer"

It is given the scene once (``open``), then per tick a snapshot of the state and a batch of camera
requests (``render``), and returns one frame per request. Each camera comes with its pose and its
intrinsics, ``(fx, fy, cx, cy)`` in pixels as MuJoCo projects it (a model camera's ``sensorsize``
and ``focal``/``principal``, or the symmetric pinhole of its ``fovy``; none for an orthographic camera or a ``fovy`` of 0, which
project through ``fovy``); a renderer declares whether it
projects an off-centre principal point (``principal_point``) and non-square pixels
(``non_square_pixels``), and whether it draws a *posed* camera (``posed``): one with no camera of
the model behind it, drawn at the pose and intrinsics its request carries, as a views file states
them. A camera that needs what its renderer does not project or draw is refused at load, or when a
tool draws through it, rather than drawn wrong.

A renderer that brings its own static world (a captured splat) and leaves the static geometry out
(``draws_static=False``) can still give the robots MuJoCo's own pictures: it opens a
:class:`~roqsim.rendering.MujocoRenderer` of its own with the same scene, its cameras copied with
``bodies="moving"``, and composites what that draws -- colour, and depth that is ``inf`` wherever no
moving body is -- by depth. Any renderer may honour ``bodies`` and declares so with ``moving_only``.

A renderer in another process stands behind the ``socket`` renderer (:mod:`roqsim.render_socket`):
its server speaks a small framed protocol over a stream socket, needing only the standard library
and numpy, and ``command`` in the renderer's settings starts it with the run.

A package for one server subclasses :class:`~roqsim.render_socket.SocketRenderer`: it declares the
server's capabilities, overrides ``scene_blobs`` to send the scene in the form its server reads
(written from the same walk ``roqsim export gltf`` uses), and states with ``open_timeout_s`` how long
its server may take to start -- the service waits for that, where it is longer than
``sim.render.timeout_s``, before it names the renderer as hung.

Choosing a ray caster
---------------------

The lidars do not render: they cast rays, through :mod:`roqsim.raycast`. Which backend casts them is
chosen the way a renderer is, for the world and per sensor. A world that says nothing needs nothing:
``auto`` puts each lidar on the backend that is faster for it, here::

   sim:
     raycaster: auto         # auto (the default) | cpu | mjwarp: every ray-cast sensor's backend
     raycasters:             # one settings block per backend
       mjwarp: {device: "cuda:0", auto_min_rays: 2000}

   components:
     - spawn_sensor: {model: mid360, frame_id: livox_frame}
       components:
         - livox_mid360: {raycaster: cpu}   # this lidar only: overrides sim.raycaster

- **auto** casts a lidar on the GPU (``mjwarp``) where all of these hold, and on the CPU otherwise:
  ``roqsim_raycast_mjwarp`` is installed, a CUDA device and a Warp build the host's driver runs are
  there, MuJoCo Warp takes the model (``mjw.put_model``), and the lidar casts at least
  ``auto_min_rays`` rays per sweep (2,000 by default: a 3D lidar, not a 2D scanner, whose few hundred
  rays are cheaper on the CPU in a small room).
- **cpu** forces MuJoCo's ``mj_multiRay`` on the CPU.
- **mjwarp** forces the GPU, and **never falls back**: without MuJoCo Warp, without a CUDA device,
  with a Warp build the host's driver cannot run (PyPI's Warp is the CUDA 13 build; Warp's release
  page has a CUDA 12 build of each release), or with a model MuJoCo Warp refuses, the world fails to
  load, naming the lidar, the backend and the reason.

**auto is never silent.** Each lidar's choice is logged when the world loads and raised as an
``info`` condition with the code ``raycaster`` (:mod:`roqsim.conditions`): its details hold what was
asked for, the backend, the ray count and the reason -- the threshold, or why the GPU could not run.
A recording's provenance and the run's ``roqsim_conditions.json``, which a campaign collects, carry it.

``sim.raycaster`` reaches every sensor built on the ray-cast base (``lidar``, ``livox_mid360``,
``seyond_robin_w1g``); the other callers of :mod:`roqsim.raycast` (a moving box's look-ahead, the
coverage tool, the navigator's caution check) cast on the CPU.

``mjwarp`` casts on a CUDA GPU through MuJoCo Warp (``pip install 'roqsim_raycast_mjwarp[mjwarp]'``;
its README). It builds a device copy of the scene once, when the first lidar that takes it is
configured, and every lidar on it shares it: the geom poses go to the device and its BVH is
refitted once per tick, whichever lidar casts first. It honours what the CPU cast honours -- the
visible geom groups, an absent entity, the excluded body -- and the lidars add their range window and
noise after the cast, as on the CPU. Its hits are the CPU's but for float32's few rays: a ray
grazing an edge, a tie between two geoms at one distance (its parity test bounds each). The noise is
drawn per ray, so such a ray changes its own point and no other.

It pays for a 3D lidar, and for any lidar in a mesh-heavy world: each sweep costs a fixed hand-over
to the device, about the cost of a 2D sweep on the CPU in a small room. In a mesh-heavy world even a
2D scanner is cheaper on the GPU; ``auto_min_rays`` lowers the threshold for such a world, or a
lidar names ``mjwarp``.

A backend of your own subclasses :class:`roqsim.raycast.Raycaster` (``open``, ``cast``,
``cast_many``, ``close``, its ``SETTINGS``) and registers under the ``roqsim.raycasters`` entry
point, or is named as ``module.path:Class``. One that declares ``auto = True`` and
``auto_min_rays(settings)`` is a candidate for ``auto``.
