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

A world that says nothing draws through MuJoCo's renderer, synchronously. From the command line
the same choice is one flag, which is ``--set sim.renderer=NAME`` applied last::

   roqsim sim world.yaml --renderer mujoco
   roqsim render world.yaml --out shot.png --renderer mujoco

What is installed, and what each takes
--------------------------------------

::

   roqsim renderers list               # every renderer, its package, one line about it
   roqsim renderers describe mujoco    # what it renders, and every setting with default and meaning

A renderer **declares** what it can give: which images (``rgb``, ``depth``, ``segmentation``),
which camera models (``pinhole``, ``orthographic``, ``free``), whether it is deterministic, whether
it runs in roqsim's process, whether it draws only the geom groups a camera names. When the world loads, every camera is checked against its renderer; a
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

Subclass :class:`roqsim.render_api.Renderer`, declare its ``capabilities`` and ``SETTINGS``, and
register it in your package::

   [project.entry-points."roqsim.renderers"]
   myrenderer = "my_package.renderer:MyRenderer"

It is given the scene once (``open``), then per tick a snapshot of the state and a batch of camera
requests (``render``), and returns one frame per request. A renderer in another process stands
behind the ``socket`` renderer (:mod:`roqsim.render_socket`): its server speaks a small framed
protocol over a stream socket, needing only the standard library and numpy, and ``command`` in the
renderer's settings starts it with the run.
