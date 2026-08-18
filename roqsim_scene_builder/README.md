# roqsim_scene_builder

Eyes on an roqsim scene, an agent's and a human's. Its MCP server exposes two native-window tools a
**human** answers in:

- **`review_scene_by_human`** — opens a native MuJoCo **3D** window showing whatever `roqsim` can
  load (a world YAML, a baked MJCF `.xml`, or a model/robot reference `roqsim_assets:<name>`),
  lets a human navigate it first-person — **left-drag looks, WASD or the arrow keys walk** (Q/E or
  PgUp/PgDn down/up, Shift fast), wheel flies, right-drag pans — drop numbered **comment dots**,
  and — in **Move Objects** mode — grab
  a `spawn_model` prop and drag it across the floor (Shift-drag to rotate, Ctrl-drag to raise/lower). **Blocks** until Pass or
  Fail; repositioned props come back under `moves`.
- **`sketch_floorplan_by_human`** — opens a **2D** top-view window where a human draws the walls of a
  floorplan (freehand drag straightened into lines immediately, or click-start/click-end for a
  straight wall), places non-overlapping door openings, and names rooms, and returns a finished
  **structured sketch** (rooms + lines + doors) that feeds the deterministic world generator
  `roqsim scenes floorplan-to-world` (from `roqsim_scenes`).

…and one tool that needs neither a window nor a person:

- **`render_scene`** — renders a world, model, mesh, or a moment from a run recorded with `roqsim sim
  --record` to a PNG, headless, and returns **the path** (not the image), so the picture's tokens are
  paid only if you look; `inline=True` returns the image itself. This is how an *agent* looks at a
  scene. It shells out to `roqsim render`, so it never holds a GL context in the long-lived server.

Both windows are plain **tkinter** + Pillow (no Qt), dark-themed. The 3D tool reuses the roqsim
engine and MuJoCo directly (`roqsim.config_for_input`, `roqsim.FrameRenderer`,
`mjv_moveCamera`); the 2D tool needs no MuJoCo.

## Quick start

```bash
make venv    # installs this package (pulls roqsim, mujoco, pillow)

# Debug the windows directly, no MCP client:
roqsim-scene-builder review-scene roqsim_scenes:depot -m "Is the layout right?"
roqsim-scene-builder sketch-floorplan -m "Draw the walls"

# Run as an MCP server; point your MCP client at this as a stdio server named `scene-builder`:
roqsim-scene-builder serve --transport stdio
```

Installing the package also adds a `builder` group to the `roqsim` command tree, so `roqsim builder serve`
/ `roqsim builder review-scene` / `roqsim builder sketch-floorplan` are the same three commands.

Each CLI prints its result JSON and exits 0 (pass/sent), 1 (fail), 2 (no display / load error),
3 (closed without a result).

## The tools

```
review_scene_by_human(target, message="", settle_steps=0, timeout_s=None, title="", focus_object="") -> dict
    # focus_object: name of a scene object to open the camera on (zoomed, clear line of sight); "" = automatic camera
    -> {"verdict": "pass"|"fail", "comment": str,
        "annotations": [{"id", "world": [x,y,z], "target": {"geom","body"} | null, "comment",
                         "yaw_deg"?}],  # yaw_deg only if a heading was dragged (double-click + hold-drag)
        "moves": [{"entity", "model", "pos": [x,y,z], "yaw_deg"}]}  # props dragged in Move-Objects mode; [] if none

sketch_floorplan_by_human(message="", initial=None, timeout_s=None, title="") -> dict
    -> {"comment": str,                                            # unbounded canvas -> no dimensions
        "rooms":  [{"id", "name", "line_ids"}],                    # closed loops, default "room N"
        "lines":  [{"id", "x0_m","y0_m","x1_m","y1_m"}],           # independent wall segments, stable ids
        "doors":  [{"id", "line_id", "t", "width_m"}],             # openings attached to a wall
        "markers": [{"id", "x_m","y_m", "comment", "in_room", "yaw_deg"?}]}  # props; in_room computed,
                                                                   # yaw_deg only if a heading was dragged

render_scene(target="", state="", at=None, out="", size="960x540", view=None, focus="",
             camera="", no_ceiling=False, inline=False) -> dict
    # state/at: a moment from an `roqsim sim --record` recording, in simulated seconds (nearest sample)
    -> {"path", "width", "height", "camera", "nbody", "ngeom"}
       # + {"sim_time", "sample_index", "requested_at", "at_error"} when rendering from a recording
       # + {"image"} with inline=True
```

Each `*_by_human` tool runs its window as a **subprocess** (tkinter owns the main thread), so it never
blocks or crashes the MCP server; the subprocess writes the result JSON which the tool reads back.
Both need a graphical session. `render_scene` is out-of-process for a different reason — no GL
context is held in the long-lived server — and needs no display at all. Offscreen frames use MuJoCo's
`egl` backend by default (override with `MUJOCO_GL`, e.g. `osmesa` on a GPU-less host).

The floorplan authoring loop (2D sketch → generate → 3D review, until pass) is the `scene-update`
skill. See `roqsim/docs/scene_builder.rst` for the full contract and internals.
