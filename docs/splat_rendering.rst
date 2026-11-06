A captured room as the world your cameras see
=============================================

A room captured as a 3D Gaussian splat looks real in a way no mesh of it does. With the splat
renderer (the package ``roqsim_render_splat``), every camera of a world sees the capture, and the
robots MuJoCo simulates are drawn into it by depth: a table in front of the robot hides it, the
robot in front of the table hides the table. It is one renderer behind roqsim's render service
(:doc:`rendering`), chosen with one line, and it draws on a CUDA GPU.

From a ``.ply`` to a picture
----------------------------

::

   pip install roqsim_render_splat                    # with torch (a CUDA build) and gsplat
   roqsim splat frame my_room.ply --known-height 2.6  # where the capture sits; prints a block
   roqsim render my_room.yaml --out shot.png          # the world you pasted the block into

Start from the commented example world, ``roqsim_render_splat:splat_room_example``
(``roqsim_render_splat/src/roqsim_render_splat/worlds/splat_room_example.yaml``), which runs as
installed on a small synthetic room. A world states three things:

.. code-block:: yaml

   sim:
     world: my_room/collision.xml       # what the robot collides with (see below)
     renderer: splat                    # every camera draws through the splat renderer
     renderers:
       splat:
         splat:
           file: my_room/room.ply       # a standard 3DGS .ply, beside the world or <package>:<path>
           pose:                        # where the capture sits, as any pose in a world is written
             position: {x: 0.0, y: 0.0, z: 1.0}
             orientation: {roll: -1.5708, pitch: 0.0, yaw: 0.35}
           scale: 0.5                   # metres per unit of the capture

Every other setting has a default; ``roqsim renderers describe splat`` lists them with their
meanings: the spherical-harmonics degree drawn, the background behind the capture, the opacity
below which Gaussians are dropped at load (fewer to draw, a changed picture), the coverage below
which a pixel reads depth ``inf``, the CUDA device, and the GPU memory one pass may take.

Where the capture sits
----------------------

A capture comes in a frame of its own: its up axis is whatever its software chose (often ``-y``),
its floor anywhere, its walls at any angle and its units of no size. ``roqsim splat frame`` reads
the opaque Gaussians and reports the floor (the largest plane with the room on one side), the up
axis and its tilt, the walls' yaw, the footprint and the height, and prints the ``splat:`` block
that puts the floor at z = 0, the walls along x and y and the footprint's centre at the origin.

It does not guess the scale. Pass ``--scale`` when you know it (the capture's vendor may), or
``--known-height`` with a measured floor-to-ceiling height; without either the block says
``scale: null``, which the world refuses until you set it. Treat what it prints as a proposal: a
bowl-shaped floor, a missing ceiling or furniture larger than the floor are why the numbers it
measured are printed beside it.

What the robot collides with
----------------------------

A splat has no collision geometry. The world's own MJCF (``sim.world``) is what the robot meets,
and the splat renderer does not draw it. ``roqsim splat collision`` fits it to the capture: a floor
plane, four walls around the footprint, and boxes where the capture has Gaussians in the band of
heights the robot meets (``--band``, by default 0.08-0.40 m: above a wheeled robot's ground
clearance, below a desk top it passes under)::

   roqsim splat collision my_room.yaml --out my_room/collision.xml

To see whether the two agree, draw the same view through both renderers::

   roqsim render my_room.yaml --out room.png
   roqsim render my_room.yaml --out collision.png --renderer mujoco

What it gives, and what it does not
-----------------------------------

- **rgb**, as captured, with view-dependent colour.
- **depth**, the capture's expected depth along the view axis in metres; ``inf`` where the capture
  covers a pixel less than ``min_alpha`` (a ceiling that was never captured, for one).
- The **moving bodies** -- robots, props a plugin moves -- drawn by MuJoCo's renderer and put in
  by depth. They are lit by the world's lights, not by the light baked into the capture, and they
  cast no shadow onto it.
- No **segmentation**: a capture has no objects, and a camera asking for it fails when the world
  loads.
- No **orthographic** camera.
- No GPU, no picture: without a CUDA device it refuses to open and says so. It never draws the
  room through MuJoCo's renderer instead.

Both of the service's modes work (``sim.render.mode``): ``sync`` waits for every camera due in a
tick, and ``async`` draws on the service's worker while physics steps on; a frame ``async``
publishes is the one ``sync`` draws for the same stamp. A camera costs one to a few milliseconds
on one GPU, depending on how many Gaussians a capture holds and how close to them a camera stands;
``min_opacity`` and ``sh_degree`` trade picture for time.

Licences
--------

The renderer draws through gsplat (Apache-2.0) and PyTorch (BSD-3-Clause). The reference 3DGS
rasteriser is licensed for non-commercial research only, so it is neither a dependency nor used. A
capture carries its owner's licence, and possibly the terms of the code that trained it.
