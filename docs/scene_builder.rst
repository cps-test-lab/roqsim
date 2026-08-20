Scene builder (human 2D floorplan + 3D review)
==============================================

``roqsim_scene_builder`` is the **scene-building addon**: eyes on a scene, an agent's and a human's. Its
MCP server exposes two native-window tools a *human* answers in —

* ``review_scene_by_human`` — a **3D** window showing whatever ``roqsim`` can load; the human
  walks/looks/pans through it and drops numbered comment **dots**, and it **blocks** until Pass or
  Fail. Use it when a single rendered image cannot convey a 3D layout — a ported scene, a robot
  placement, a world under review.
* ``sketch_floorplan_by_human`` — a **2D** top-view window for authoring a floorplan's walls; it
  returns a finished **structured sketch** (rooms + lines + doors) that the deterministic generator
  ``roqsim scenes floorplan-to-world`` turns into a world. The floorplan is the single
  source of truth: the generator writes it to the scene's ``floorplan.json`` and the generated
  ``scene.json`` only **references** it (its ``floorplan`` field is the relative path
  ``floorplan.json``, never an embedded copy), so ``roqsim scenes scene-to-floorplan``
  round-trips a scene back to its floorplan JSON by following that reference. The floorplan-level and
  per-room ``description``\ s ride in ``floorplan.json`` and can be edited there later with no re-bake.
  See :ref:`sketch-floorplan-tool` below; the full authoring loop is the ``scene-update`` skill.

— and one tool that needs neither a window nor a person:

* ``render_scene`` — a PNG of a world, a model, a mesh, or a moment from a run recorded with
  ``roqsim sim --record``, rendered headless. It is how an *agent* looks at a scene, where the two
  tools above are how it asks a *human* to. See :ref:`render-scene-tool` below.

The sketch window is not the only source of that JSON. Two tools in ``roqsim_scenes`` produce the same
schema from something that already exists, leaving the generator downstream unchanged:

* ``roqsim scenes mapimage-to-floorplan`` — when the layout exists only as a *picture* (an
  occupancy-grid screenshot, a published top view). It measures a figure rather than reading a world,
  so it needs the scale in pixels per metre, and the human's job shrinks to checking the trace
  against the source in the overlay it writes.
* ``roqsim scenes gridmap-to-floorplan`` — when an occupancy grid is already in hand. Its sibling
  ``roqsim scenes gridmap-to-world`` places one prop per cell instead; the choice between them is what
  the cells *are*, obstacles or architecture, and the floorplan tool refuses grids that turn out to
  be the former rather than inventing walls through them.
* ``roqsim scenes dxf-to-floorplan`` — when the layout exists as a CAD drawing.

Both emit axis-aligned walls only. A hand-drawn plan, or a world with diagonal walls, still belongs
in the sketch window — the generator itself places a wall at any angle.

To *look* at a floorplan without opening a window — a review of what a sketch produced, or the layout an
agent needs before placing anything — ``roqsim-floorplan-to-png`` (in ``roqsim_scenes``) renders it as a
top-down plan PNG with a positionable metre scale bar, taking the floorplan JSON, a scene dir, a world
YAML or a ``<package>:<world>`` ref. It draws from the same geometry the generator bakes; see the
``roqsim_scenes`` README.

It reuses the engine and MuJoCo directly rather than reimplementing them: the same target dispatch as
the CLI (:func:`roqsim.config_for_input`), the shared offscreen renderer
(:class:`roqsim.FrameRenderer`), and roqsim's shared camera navigation (first-person look/walk/fly, plus
MuJoCo's own ``mjv_moveCamera`` for the pan). The window is plain **tkinter** + Pillow (no Qt),
dark-themed.

(The generic *image* review tool that used to live here now ships as its own standalone project,
``mcp-media-review`` — see below.)

Install and register
--------------------

``make venv`` installs the package and pulls its dependencies (``roqsim``, ``mujoco``, ``pillow``;
``fastmcp``/``click``). Register the MCP server with your client as a stdio server named
``scene-builder``, running ``roqsim-scene-builder serve``; the client then sees all three tools.

Installing the package also contributes a ``builder`` group to the ``roqsim`` command tree, so the same
three commands are reachable either way — ``roqsim builder serve`` is ``roqsim-scene-builder serve``. The
standalone script is what an MCP client is pointed at (it needs one executable, not a subcommand
path); ``roqsim builder`` is what a human already living in the ``roqsim`` tree types.

Inputs — the same as ``roqsim``
----------------------------------

``target`` accepts exactly what the ``roqsim`` CLI accepts, via the same
:func:`roqsim.config_for_input` dispatch:

* a world ``*.yaml`` / ``*.yml``,
* a baked MJCF ``*.xml`` scene,
* a model/robot reference (``<pkg>:<name>``, a bundled model name, or a path) shown alone in the
  default ``empty_room``.

The ``review_scene_by_human`` tool
----------------------------------

.. code-block:: text

    review_scene_by_human(target: str, message: str = "", settle_steps: int = 0,
                          timeout_s: float | None = None, title: str = "",
                          focus_object: str = "") -> dict

* **target** — world / MJCF / model reference (above). A path to a missing file fails loudly.
* **title** — a short heading atop the panel in a larger font (the *what* under review). Optional.
* **message** — the question shown beside the scene ("Is the shelving reachable, not clipping the
  wall?"). Optional but recommended.
* **settle_steps** — advance physics this many steps before showing the scene (let a dropped scene
  come to rest). Default 0.
* **focus_object** — name of a scene object (a world's ``spawn_model`` ``name:``, as returned under
  ``moves``) to point the initial camera at: the window opens zoomed to fill the view with that
  object, from a viewing angle picked to have a **clear line of sight** to it (it looks over/around
  walls so the object is not hidden behind one). Empty (default) keeps the automatic camera — a
  model preview for a bare model ref, otherwise the world's ``sim.view`` or MuJoCo's default. An
  unknown name is not an error: it warns and falls back to the automatic camera.
* **timeout_s** — seconds to wait for a verdict before ``TimeoutError`` (default 600).

Returns::

    {"verdict": "pass" | "fail", "comment": str,
     "annotations": [{"id": 1, "world": [1.2, 0.3, 0.8],
                      "target": {"geom": "shelf_top", "body": "shelf"}, "comment": "…",
                      "yaw_deg": 90}],   # yaw_deg only present when a heading was dragged
     "moves": [{"entity": "industrial_table_1", "model": "industrial_table",
                "pos": [10.4, 1.15, 0.0], "yaw_deg": 90.0}]}   # props dragged in Move-Objects mode; [] if none

Each annotation is a dot the human dropped on a surface: ``world`` is the 3D hit point and
``target`` names the geom/body it landed on (``null`` if none). ``yaw_deg`` (heading about +Z,
0 = +x, CCW) is present only when the human dragged a direction while placing the dot — take it as
the yaw of a prop the dot marks.

Each **move** is a ``spawn_model`` prop the human repositioned with **Move Objects** mode on:
``entity`` is the world YAML ``spawn_model`` ``name``, and ``pos`` (x, y, and z when raised) /
``yaw_deg`` are its new pose. The window does not touch the world itself — it reports the move as intent, exactly like a
comment dot; the caller (the ``scene-update`` skill) writes the pose back into the sketch/markers-map
and regenerates. ``moves`` is empty when nothing was dragged. Why a rebuild rather than nudging the
live body: a ``spawn_model`` prop is welded (no free joint), and roqsim treats the compiled
``MjModel`` as immutable at runtime — so on release the prop's ``spawn_model`` pose is edited and the
engine is recompiled (sanctioned between edits), never a ``model.body_pos`` write.

CLI (debugging)
---------------

``roqsim-scene-builder review-scene`` opens the same window with no MCP client in the loop::

    roqsim-scene-builder review-scene roqsim_scenes:depot -m "Layout ok?"
    roqsim-scene-builder review-scene roqsim_assets:industrial_table -m "Right scale?"
    roqsim-scene-builder review-scene scene.xml --settle-steps 200 --size 1280x800

It prints the verdict JSON and exits **0** (pass), **1** (fail), **2** (no display / load error),
**3** (window closed without a verdict).

The window navigates like a first-person game: **left-drag looks** (the camera turns about the eye,
not around a pivot in front of it), **WASD walks** — or the **arrow keys**, whichever hand is free —
``Q``/``E`` (or Page Up/Down) drop and rise, **Shift** is faster and **Ctrl** slower, the **wheel
flies** forward/back along the view, and right-drag still pans. That is
how a building-sized world gets inspected from the inside instead of only circled from outside.
``W``/``S`` follow where you look (look down, fly down) while ``A``/``D`` stay level. The keys act
while the pointer is over the 3D view — typing in the comment box stays text.

**Double-click a surface** to drop a numbered comment dot: the click is ray-picked, so the dot names
the geom/body it hit and is drawn as a colour-coded marker **sphere in the 3D scene** — it tracks the
camera (moves, zooms, occludes) rather than floating as a 2D overlay. **Hold that second click and drag** to give the dot a heading
(``yaw_deg``), drawn as an arrow on the ground plane; a plain double-click leaves it headingless.
Each dot gets a comment field and a ✕ to remove it (the rest renumber). Type an overall comment,
then Pass or Fail.

Toggle the **Move Objects** button (its hover tooltip carries the hint) to reposition props instead of
annotating: left-press a ``spawn_model`` prop and drag it — the prop's own mesh follows the cursor on
its floor plane (redrawn live, no recompile) — then release to commit (the engine rebuilds at the new
pose). **Shift-drag** rotates it about its centre; **Ctrl-drag** raises or lowers it in place (the
cursor's height on a vertical plane through the prop, clamped at the floor). Grabbing a wall, the floor, or the robot is simply
a no-op (they are baked geometry with no editable pose), so the press aims the camera as usual. Each
moved prop is listed below the comment dots with an accent swatch; its ✕ resets that prop, and the
``↶``/``↷`` buttons undo/redo the last move. Every move is reported under ``moves``; camera
navigation and dot-dropping are unchanged when the mode is off.

.. _sketch-floorplan-tool:

The ``sketch_floorplan_by_human`` tool
--------------------------------------

.. code-block:: text

    sketch_floorplan_by_human(message: str = "", initial: dict | None = None,
                              timeout_s: float | None = None, title: str = "") -> dict

A 2D top-view window with five modes — **draw** a wall (either **drag** freehand, straightened into
lines the instant the pencil lifts, or **click** a start point then **click** the end for one
straight wall), **move** a point, place a **door** opening (openings cannot overlap on the same
wall), **mark** a prop (drop a point and name it; hold and drag a direction to also set its heading),
**delete** a wall/door/marker — plus per-room name
boxes. The canvas is an unbounded
plane (wheel zooms, right-drag pans, a scale-bar legend shows the current scale; the view auto-zooms
to a seeded floorplan) — there is no overall room size. It returns a **finished, structured** sketch
(no raw-stroke step to post-process)::

    {"comment": str,
     "rooms":  [{"id": 1, "name": "room 1", "line_ids": [1, 2, 3, 4]}],
     "lines":  [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 8.0, "y1_m": 0.0}],
     "doors":  [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}],
     "markers": [{"id": 1, "x_m": 2.5, "y_m": 3.0, "comment": "office table", "in_room": 1,
                  "yaw_deg": 90}]}   # yaw_deg only present when a heading was dragged

* **lines** — independent wall segments, each with its own two endpoints and a **stable, monotonic
  id** (a split keeps the first half's id and mints a new one), so a reference like "line 3" survives
  edits. Coordinates are metres, y measured from the bottom, 2 decimals.
* **rooms** — the closed loops the walls enclose, auto-detected (planar faces), listed biggest-first
  and **nameable** (default ``room N``). Metadata for naming/lookup; the bake builds geometry from
  ``lines``.
* **doors** — standard-width **openings** attached to a wall by ``line_id`` + fraction ``t`` (so a
  door rides along its wall and is removed with it); the generator cuts a **2 m-high** hole out of the
  wall and leaves a solid **lintel** above it up to the ceiling. Height is the generator's
  ``--opening-h`` (default 2 m), overridable per door with an optional ``height_m``. Two openings may
  abut but not overlap on the same wall (the hover preview turns red where it would). Each opening is
  then fitted with a **swing-door leaf** (the ``door`` plugin — a hinged leaf with a position
  actuator, ROS-controllable as an automatic door); ``roqsim scenes floorplan-to-world``'s ``--doors-map``
  overrides per door which side is fixed, which way it swings, how open it starts, the leaf model
  (``door`` / ``door_glass``), its ``color`` / ``frame_color``, and whether it is ``controllable``. A
  **full-height** opening (``height_m`` ≥ the ceiling) stays a bare doorway — a swing leaf does not fit
  it. Two doors-map keys say the opening is not a swinging door, and **both keep it a floorplan
  ``door``** — so the room loops it belongs to never change, only what fills it: ``leaf: false`` forwards
  to the plugin for a **cased opening** (a *Türblatt*-less doorway — casing welded, no leaf, no
  hinge/actuator/ROS), and ``skip: true`` emits no door plugin at all, for an opening a ``window``
  plugin fills or a deliberately bare gap. The parametric ``window`` plugin is sized for exactly that
  (windows sit beside doors in the same wall). Openings all start at the **floor**, so a window
  with a sill is not expressible this way; a sill needs a change to how openings are cut.
* **markers** — prop points whose ``comment`` names the model to place; dropped in **mark** mode
  and/or added from 3D-review comment dots, and carried through a wall-editing round. ``in_room`` is
  the id of the room containing the marker (computed; ``null`` if outside every room). A marker also
  carries ``yaw_deg`` (heading about +Z, 0 = +x, CCW → the prop's ``spawn_model`` ``rpy``) **only
  when the human dragged a direction** out of the point in mark mode; a plain click leaves it
  headingless (the prop is placed axis-aligned). Orientation the agent decides for a 3D-review prop
  goes in the generator's ``--markers-map`` instead (``roqsim scenes floorplan-to-world``), whose ``yaw_deg``
  overrides a sketch heading.

``title`` (larger font) and ``message`` head the right-hand panel, both wrapping to the comment
box's width — use ``title`` for the *what* ("Apartment — 3 rooms") and ``message`` for the
*instruction*.

``initial`` pre-seeds the window with a sketch (same schema): ``lines`` keep their ids, ``rooms``
restore names, ``markers`` ride along — used for *describe-and-review* (the agent drafts a candidate
sketch from a text description; the view auto-zooms to it) and for iterating. In the window: the
wheel zooms, the right button pans, and ↶/↷ undo/redo the last edit.

CLI: ``roqsim-scene-builder sketch-floorplan -m "Draw the walls"`` prints the sketch JSON and exits
0 (sent) / 2 (no display) / 3 (closed without sending).

.. _render-scene-tool:

The ``render_scene`` tool
-------------------------

.. code-block:: text

    render_scene(target: str = "", state: str = "", at: float | None = None, out: str = "",
                 size: str = "960x540", view: list[str] | None = None, focus: str = "",
                 camera: str = "", no_ceiling: bool = False, inline: bool = False) -> dict

The one tool here with **no window and no human**: it renders and returns where the picture is. Use it
for "does this world look right", "where did the robot end up", "what did the run look like at
t = 12.5"; ``review_scene_by_human`` is for when a *person* must judge.

* **target** — the same shapes as above (world / MJCF / model ref), plus a raw mesh. Optional when
  ``state`` is given, because a recording names the world it came from.
* **state** / **at** — render a moment from a run recorded with ``roqsim sim --record``, at ``at``
  *simulated* seconds. It snaps to the nearest recorded sample and reports which one it used, so a
  caller sees it landed a few milliseconds off rather than assuming it did not. Omit ``at`` for the
  last sample.
* **view** / **focus** / **camera** — ``KEY=VALUE`` overrides in the world's own ``sim.view``
  vocabulary (a vector value is comma- or space-separated: ``["lookat=-3.2 -1.3 1.9"]``); an entity to
  frame on, searching for a clear line of sight (what you want indoors); or a fixed MJCF ``<camera>``
  to look through. ``camera`` owns its pose, so it excludes the other two.
* **no_ceiling** — drop a roofed world's ceiling to look into it from above.
* **out** / **size** / **inline** — where to write the PNG (default a temp file), ``WxH``, and
  whether to return the image itself.

Returns ``{"path", "width", "height", "camera", "nbody", "ngeom"}``, plus ``{"sim_time",
"sample_index", "requested_at", "at_error"}`` when rendering from a recording.

**It returns a path, not an image, by default** — and that is the point. An agent reads the returned
path with its own file-reading tool, so the image's tokens (roughly ``w*h/750``: ~700 for 960x540,
~310 for 640x360) are paid only if it actually looks. Pass ``inline=True`` when the picture should
appear in the conversation itself -- it comes back as an image content block beside the same record,
which stays the result's structured content; the default costs about forty tokens.

CLI: this is ``roqsim render`` — the tool shells out to it rather than importing it (see *Internals*),
so every flag above is that command's own.

Internals
---------

* **Out-of-process everything.** Both windows run as a subprocess (``roqsim-scene-builder
  review-scene …`` / ``sketch-floorplan … --json-out <tmp>``, via the shared ``window_runner``) and
  block on it, because tkinter — like any GUI toolkit — owns the main thread and cannot run inside
  the FastMCP worker thread. The subprocess writes the result JSON, which the tool reads back. A
  rendering crash therefore can never take the MCP server down. Timeout kills the subprocess; a
  window closed without a result is a loud ``RuntimeError``. ``render_scene`` shells out to ``roqsim
  render`` for the same reason and three more: this server is long-lived and must not acquire or leak
  a GL context per call, every render gets a fresh offscreen one, and the CLI and the tool cannot
  drift because there is only one implementation of the rendering itself. ``MUJOCO_GL`` is
  ``setdefault``-ed to ``egl`` for the child, so a GPU-less host can still set ``osmesa`` and be
  honoured.
* **3D review — single-threaded rendering.** The review window loads the world (``config_for_input``
  → ``Engine`` → ``setup``/``reset``), then renders **on demand** — only when the camera moves —
  through a :class:`roqsim.FrameRenderer` on the tkinter main thread. No physics loop and one
  thread, so the single-writer rule (``docs/architecture.rst`` §7) holds trivially and no snapshot
  machinery is needed. Mouse drags map to ``mujoco.mjv_moveCamera`` (the exact routine the native
  passive viewer uses); dots are ray-picked and drawn as marker spheres in the scene. The renderer
  is shared with the thumbnail tool and the RGB(-D) camera sensor (see :doc:`architecture`).
* **2D floorplan — no MuJoCo.** The sketch window is pure tkinter vector drawing on a metric,
  letterboxed canvas; its non-GUI core (the wall/room/door model, the metre↔pixel mapping, snapping,
  splitting, room detection, result assembly) is kept free of tkinter and unit-tested headless.
* **Addon.** No core roqsim package depends on this one; the suite installs and runs without it.

Roadmap
-------

* **Sketch-authored hinge side.** A door opening now gets a real swing leaf (the ``door`` plugin),
  but which edge is hinged / which way it swings still comes from the generator's ``--doors-map`` (or
  its defaults), not the 2D sketch. Capturing hinge side + swing direction as the human draws the
  opening is future work.
* **Play/settle controls** in the 3D review — a play/pause to watch dynamics, not only a static view.

The generic image tool (``mcp-media-review``)
---------------------------------------------

The browser-based *image* review tool (``review_by_human``: show an image, block for a pass/fail or
OK verdict) that previously lived in this package now ships as a **standalone, dependency-light
project**, ``mcp-media-review`` — it is generic (nothing roqsim-specific) and intended to be
reusable and open-sourced on its own (``github.com/fred-labs/mcp-media-review``). It lives in its own
git repo, so check it out beside this one, install it with ``pip install -e mcp-media-review``, and
register it with your MCP client as the ``media-review`` server. It is the still-image counterpart to
this package's 3D review.
