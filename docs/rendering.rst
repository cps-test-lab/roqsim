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
(``non_square_pixels``), and a camera that needs what its renderer does not project is refused at
load, or when a tool draws through it, rather than drawn wrong.

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
