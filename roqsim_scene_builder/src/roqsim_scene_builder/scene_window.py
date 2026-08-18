# SPDX-License-Identifier: Apache-2.0
"""The native scene-review window (tkinter + MuJoCo offscreen frames).

Runs as its own process (spawned by :mod:`roqsim_scene_builder.scene_review`, or via
``roqsim-scene-builder review-scene``) because tkinter, like any GUI toolkit, owns the main thread.
It loads whatever ``roqsim`` would load (world YAML / MJCF / model ref -- via
:func:`roqsim.config_for_input`), renders the static scene with a shared
:class:`roqsim.FrameRenderer`, and lets the human navigate the scene and drop numbered comment *dots*.

Navigation is first-person rather than MuJoCo's orbit: left-drag aims the camera about the *eye*
(:func:`roqsim.rendering.look_in_place`), ``WASD``/``QE`` walk it (:func:`roqsim.walk_delta`, shared with
the roqsim viewer window), and the wheel flies along the view (:func:`roqsim.rendering.dolly`); right-drag
still pans through :func:`mujoco.mjv_moveCamera`. Orbiting can only circle a building-sized world
from outside, which is the one thing a scene review needs to do from within. The walk keys are bound
on the 3D canvas rather than the toplevel, so the same letters typed into the comment box stay text.

A dot is placed by double-clicking a visible surface: the click is ray-picked
(:meth:`roqsim.FrameRenderer.select`) to identify the geom/body and its 3D world point, and the
dot is then drawn as a small **marker sphere injected into the scene**, so it tracks the camera
(moves, zooms, and occludes) like any other object. Holding the second click of the double-click and
**dragging a direction** gives the dot a **heading** (``yaw_deg``), drawn as an arrow on the ground
plane; a plain double-click leaves it headingless.

With **Move Objects** mode on, a ``spawn_model`` prop can be grabbed and slid across the floor
(left-drag), turned (Shift-drag), or raised/lowered (Ctrl-drag). We never mutate the compiled
``MjModel`` -- that is the path the
roqsim docs forbid at runtime. Instead the drag redraws the prop's own mesh at the target pose by
transforming its geoms in the *render scene* (the transient ``MjvScene``, not ``model``/``data``); on
release the prop's pose is written into its ``spawn_model`` config entry and the engine is **rebuilt**
(recompiling between edits is sanctioned; mutating a live model is not), making the same move
permanent. The final poses come back under ``moves``; the caller (the ``scene-update`` skill) writes
them into the world YAML. Only props move -- walls and floor are baked into their meshes and have no
editable pose.

The non-GUI parts -- loading, the dot bookkeeping (:class:`DotModel`), and the pose helpers
(:func:`apply_prop_pose`, :func:`move_record`) -- are kept free of tkinter so they can be tested
headless; only :func:`run_window` needs a display.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from roqsim import (
    WALK_KEYS,
    walk_delta,
)  # the camera-walk vocabulary is shared with the roqsim viewer
from roqsim_scene_builder.annotate_ui import (  # theme + shared widgets live in one module
    BG,
    FAIL_BG,
    FG,
    MUTED,
    PANEL,
    PASS_BG,
    SEND_BG,
    add_tooltip,
    build_button_row,
    build_comment_box,
    build_point_rows,
    color_for,
    enable_edit_shortcuts,
    renumber,
    rgba_hex,
)

# color_for / rgba_hex are re-exported here so callers (and tests) can import the whole annotation
# vocabulary from the 3D window module as before the theme moved to annotate_ui.
__all__ = [
    "Dot",
    "DotModel",
    "Move",
    "MoveModel",
    "apply_prop_pose",
    "apply_view",
    "color_for",
    "load_engine",
    "move_record",
    "rgba_hex",
    "run_window",
    "walk_delta",
    "write_result",
]

_DEFAULT_SIZE = (960, 720)
_MARKER_RADIUS_M = 0.05  # comment-dot sphere: ~10 cm across, at real scene scale
_YAW_ARROW_LEN_M = 0.6  # heading-arrow length drawn from a dot when it has a yaw
_YAW_ARROW_W = 0.03  # heading-arrow shaft width
_YAW_MIN_DRAG_M = 0.05  # a drag shorter than this (world XY) sets no heading (a plain double-click)
_SHIFT_MASK = 0x0001  # tkinter event.state bit for Shift (rotate instead of translate)
_CTRL_MASK = 0x0004  # tkinter event.state bit for Control (raise/lower instead of translate)
_MOVE_SWATCH_RGBA = (0.20, 0.60, 0.86, 1.0)  # accent for moved-prop rows (no numbered 3D marker)
#: Arrow keys / Page Up-Down as aliases of the WASD directions, so either hand can fly. (They are
#: what the roqsim viewer window has to use, MuJoCo's UI having claimed the letters; offering both here
#: means one set of fingers works in both windows.)
_ARROW_ALIASES = {
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    "prior": "e",  # Page Up
    "next": "q",  # Page Down
}
_SPRINT_KEYS = ("shift_l", "shift_r")
_CRAWL_KEYS = ("control_l", "control_r")
_WALK_SPEED_MS = 2.5  # WASD travel speed (m/s at real scene scale -- roughly a brisk walk)
_WALK_SPRINT = 3.0  # Shift multiplier
_WALK_CRAWL = 0.25  # Ctrl multiplier
_WHEEL_STEP_M = 0.4  # one wheel notch flies this far along the view direction
_WALK_TICK_MS = 33  # ~30 Hz; each tick re-renders, so the step is scaled by the measured dt
_WALK_MAX_DT_S = 0.2  # cap the per-tick step when a slow render (or a stall) stretches the interval
_KEY_REPEAT_GRACE_MS = 60  # X11 auto-repeat sends release+press pairs; hold the key this long


# --- non-GUI core (testable without a display) ---------------------------------------------------


def _walk_key(keysym: str) -> str:
    """The walk direction a tkinter ``keysym`` names: WASD/QE as themselves, arrows as their alias."""
    key = (keysym or "").lower()
    return _ARROW_ALIASES.get(key, key)


def apply_view(cam: mujoco.MjvCamera, view: dict | None) -> None:
    """Point a free camera per a world's ``sim.view`` block (any subset of keys)."""
    if not view:
        return
    if "lookat" in view:
        cam.lookat[:] = [float(v) for v in view["lookat"]]
    if "distance" in view:
        cam.distance = float(view["distance"])
    if "azimuth" in view:
        cam.azimuth = float(view["azimuth"])
    if "elevation" in view:
        cam.elevation = float(view["elevation"])


def load_engine(target: str, settle_steps: int = 0, skip_transport: bool = True):
    """Load ``target`` the way ``roqsim`` does and return ``(engine, view)``.

    ``settle_steps`` optionally advances physics so a dropped scene comes to rest before review.

    A review is about geometry, so transport plugins are dropped (``skip_transport``, see
    :func:`roqsim.config.drop_transport_plugins`) -- which is what lets a ``*_ros`` world be reviewed in
    an environment without its middleware. Unloadable plugins are named on stderr, since the other way
    to get one is a misspelt ref.
    """
    import sys

    from roqsim import Engine, config_for_input, drop_transport_plugins

    cfg = config_for_input(target)
    if skip_transport:
        transport, unavailable = drop_transport_plugins(cfg)
        for label, why in [
            (transport, "not needed for a review"),
            (unavailable, "cannot be loaded here"),
        ]:
            if label:
                print(
                    f"roqsim-scene-builder: skipping {', '.join(label)} -- {why}.",
                    file=sys.stderr,
                    flush=True,
                )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(max(0, settle_steps)):
        engine.step()
    return engine, getattr(cfg, "view", None)


@dataclass
class Dot:
    """A comment marker anchored to a 3D world point on a picked geom.

    ``yaw_deg`` is an optional heading about +Z (degrees, 0 = +x, CCW) drawn by holding the second
    click of the double-click and dragging a direction; ``None`` means no heading was set.
    """

    id: int
    world: list[float]  # 3D anchor in world coordinates
    target: dict | None = None  # {"geom": <name|None>, "body": <name|None>} of the picked surface
    comment: str = ""
    yaw_deg: float | None = None

    @property
    def label(self) -> str:
        """Human-readable name of what this dot marks."""
        base = (
            "(point)"
            if not self.target
            else (self.target.get("body") or self.target.get("geom") or "(unnamed)")
        )
        return f"{base} ∠{self.yaw_deg:.0f}°" if self.yaw_deg is not None else base


@dataclass
class DotModel:
    """Ordered, 1-numbered comment dots. Pure bookkeeping -- no tkinter."""

    dots: list[Dot] = field(default_factory=list)

    def add(self, world, target: dict | None = None) -> Dot:
        dot = Dot(id=len(self.dots) + 1, world=[float(v) for v in world], target=target)
        self.dots.append(dot)
        return dot

    def delete(self, dot_id: int) -> None:
        self.dots = [d for d in self.dots if d.id != dot_id]
        renumber(self.dots)  # keep ids 1..N contiguous

    def to_annotations(self) -> list[dict]:
        return [
            {
                "id": d.id,
                "world": [round(v, 4) for v in d.world],
                "target": d.target,
                "comment": d.comment,
                # only present when a heading was dragged -- keeps headingless dots unchanged
                **({"yaw_deg": round(d.yaw_deg, 1)} if d.yaw_deg is not None else {}),
            }
            for d in self.dots
        ]


def apply_prop_pose(config: dict, pos, yaw_deg: float) -> dict:
    """Write a prop's new floor pose into its ``spawn_model`` config dict (pure, in place).

    Sets ``pos`` ([x, y, z], rounded) and folds ``yaw_deg`` into ``rpy[2]`` (radians) while keeping any
    existing roll/pitch. Leaves ``rpy`` off entirely when the whole orientation is zero, so an unrotated
    prop's entry stays as terse as the author wrote it. Returns the same dict for convenience.
    """
    config["pos"] = [round(float(v), 3) for v in pos]
    rpy = list(config.get("rpy", [0.0, 0.0, 0.0]))
    if len(rpy) < 3:
        rpy = [0.0, 0.0, 0.0]
    rpy[2] = round(math.radians(float(yaw_deg)), 5)
    if any(abs(v) > 1e-9 for v in rpy):
        config["rpy"] = [round(rpy[0], 5), round(rpy[1], 5), rpy[2]]
    return config


def move_record(entity_name: str, model: str, pos, yaw_deg: float) -> dict:
    """One ``moves`` payload entry: which prop moved, and to what floor pose (pure)."""
    return {
        "entity": entity_name,
        "model": model,
        "pos": [round(float(v), 3) for v in pos],
        "yaw_deg": round(float(yaw_deg), 1),
    }


@dataclass
class Move:
    """A prop the human repositioned in Move-Objects mode. ``spec`` is the prop's ``spawn_model`` config
    entry and ``orig_config`` a snapshot of it before the *first* move, so a reset restores the prop
    exactly. ``comment`` is unused -- it only satisfies the row-builder's item protocol."""

    id: int
    entity: str
    model: str
    pos: list[float]
    yaw_deg: float
    spec: object = None
    orig_config: dict | None = None
    comment: str = ""

    @property
    def label(self) -> str:
        """``entity → x, y`` (plus a heading when rotated and a height when raised) -- what the panel
        row shows."""
        base = f"{self.entity} → {self.pos[0]:.2f}, {self.pos[1]:.2f}"
        if len(self.pos) >= 3 and abs(self.pos[2]) > 1e-6:
            base += f" ↑{self.pos[2]:.2f}"
        return f"{base} ∠{self.yaw_deg:.0f}°" if abs(self.yaw_deg) > 1e-6 else base


@dataclass
class MoveModel:
    """The moved props, one row per prop (last move per prop wins). Mirrors :class:`DotModel` so it
    drives the same panel row-builder; pure bookkeeping, no tkinter."""

    moves: list[Move] = field(default_factory=list)

    def set(self, entity, model, pos, yaw_deg, spec=None, orig_config=None) -> Move:
        """Record (or update) the move for ``entity``. A prop already in the list keeps its id and its
        first ``orig_config`` -- only its target pose changes -- so re-dragging it does not stack rows
        or lose the true original."""
        for m in self.moves:
            if m.entity == entity:
                m.model, m.pos, m.yaw_deg = model, [float(v) for v in pos], float(yaw_deg)
                return m
        m = Move(
            len(self.moves) + 1,
            entity,
            model,
            [float(v) for v in pos],
            float(yaw_deg),
            spec,
            dict(orig_config) if orig_config is not None else None,
        )
        self.moves.append(m)
        return m

    def get(self, move_id: int) -> Move | None:
        return next((m for m in self.moves if m.id == move_id), None)

    def delete(self, move_id: int) -> None:
        self.moves = [m for m in self.moves if m.id != move_id]
        renumber(self.moves)  # keep ids 1..N contiguous

    def to_payload(self) -> list[dict]:
        return [move_record(m.entity, m.model, m.pos, m.yaw_deg) for m in self.moves]


def write_result(
    json_out: str | None, verdict: str, comment: str, dots: DotModel, moves: list | None = None
) -> dict:
    """Assemble the result dict and, if ``json_out`` is given, write it there.

    ``moves`` is the list of props the human repositioned in Move-Objects mode (empty when none moved);
    each entry is a :func:`move_record`.
    """
    result = {
        "verdict": verdict,
        "comment": comment,
        "annotations": dots.to_annotations(),
        "moves": moves or [],
    }
    if json_out:
        Path(json_out).write_text(json.dumps(result), encoding="utf-8")
    return result


# --- GUI ------------------------------------------------------------------------------------------


def run_window(
    target: str,
    message: str = "",
    settle_steps: int = 0,
    json_out: str | None = None,
    size: tuple[int, int] = _DEFAULT_SIZE,
    title: str = "",
    focus_object: str = "",
) -> int:
    """Open the review window for ``target`` and block until the human submits or closes it.

    Returns an exit code: 0 (pass, or a neutral Enter-submitted comment), 1 (fail), 2 (no display /
    load error), 3 (closed without a verdict). On any submit the result JSON is written to
    ``json_out`` (when given).
    """
    import os
    import sys

    from roqsim.viewer import has_display

    if not has_display():
        print(
            "roqsim-scene-builder: no DISPLAY -- the scene-review window needs a graphical session.",
            flush=True,
        )
        return 2
    # Offscreen frames use MuJoCo's egl backend by default (same as roqsim); the tkinter window
    # itself is just blitted images, no GL. Override by setting MUJOCO_GL yourself.
    os.environ.setdefault("MUJOCO_GL", "egl")

    import tkinter as tk  # imported here so the module stays importable headless

    from roqsim import FrameRenderer
    from roqsim.rendering import focus_camera, preview_camera
    from roqsim.runner import is_model_ref

    engine, view = load_engine(target, settle_steps)
    width, height = size
    fr = FrameRenderer(engine.ctx.model, width, height)
    ctx = engine.ctx
    entity = ctx.entities.get(focus_object) if focus_object else None
    if focus_object and entity is None:
        print(
            f"roqsim-scene-builder: no object {focus_object!r} in scene; "
            f"available: {', '.join(ctx.entities.names()) or '(none)'}. Using default camera.",
            file=sys.stderr,
            flush=True,
        )
    if entity is not None and entity.body:
        # Open looking at the requested object, from an angle with a clear line of sight to it.
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        fr.camera = focus_camera(ctx.model, ctx.data, [bid], aspect=width / height)
    elif is_model_ref(target) and not view:
        # A single model/robot shown by itself: zoom onto it, not the empty room it stands in.
        fr.camera = preview_camera(ctx.model, ctx.data, ctx.entities.all(), aspect=width / height)
    else:
        apply_view(fr.camera, view)

    app = _ReviewApp(tk, engine, fr, message, json_out, width, height, title, settle_steps)
    try:
        app.root.mainloop()
    finally:
        # A prop move rebuilds the engine/renderer, so close whatever the app now holds (which may no
        # longer be the originals created above) rather than the stale locals.
        app.fr.close()
        app.engine.shutdown()
    return app.exit_code


class _ReviewApp:
    """The tkinter window: 3D canvas on the left, verdict/comment/dots panel on the right."""

    def __init__(self, tk, engine, fr, message, json_out, width, height, title="", settle_steps=0):
        self.tk = tk
        self.engine = engine
        self.fr = fr
        self.json_out = json_out
        self.width = width
        self.height = height
        self.settle_steps = settle_steps  # re-applied when a prop move rebuilds the engine
        self.dots = DotModel()
        self.exit_code = 3  # closed-without-verdict unless a button sets it
        # Front/back Tk photos (see _rerender) -- also the only reference to them, without which Tk
        # garbage-collects the image out from under the canvas.
        self._photos: list = []
        self._image_item = None
        self._drag = None  # (x, y, button) during a camera drag
        self._yaw_dot = (
            None  # the just-placed Dot whose heading a held double-click drag is setting
        )
        self._marker_r = _MARKER_RADIUS_M
        # Move-Objects state: the movable-prop map, the currently grabbed prop, its drag ghost, and the
        # committed moves (one row per prop, listed in the panel like the comment dots).
        self._movable: dict[str, object] = {}
        self._sel: dict | None = None
        self._ghost: dict | None = None
        self.moves = MoveModel()
        self._undo: list = []  # snapshots for ↶ (taken before each move/reset)
        self._redo: list = []  # snapshots for ↷
        # WASD walk state: the held keys, a press counter per key (to tell an X11 auto-repeat
        # release from a real one), the pending after() job, and the last tick's clock reading.
        self._keys: set[str] = set()
        self._key_press_id: dict[str, int] = {}
        self._walk_job = None
        self._walk_t0 = 0.0
        self._sprint = False  # Shift held: travel faster
        self._crawl = False  # Ctrl held: travel slower (fine positioning)

        root = tk.Tk()
        root.title("roqsim scene review")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root = root
        self.edit_var = tk.BooleanVar(master=root, value=False)
        enable_edit_shortcuts(root)  # Ctrl+A select-all (Ctrl+C/V/X are Tk defaults) in text fields
        self._build(title, message)
        self._build_movable()
        self._rerender()

    # -- layout --
    def _build(self, title: str, message: str) -> None:
        tk = self.tk

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            body, width=self.width, height=self.height, bg="#141414", highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_press)
        self.canvas.bind("<B3-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)  # add a comment dot
        self.canvas.bind("<MouseWheel>", self._on_wheel)  # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_wheel)  # Linux scroll down
        # WASD walk. Bound on the canvas (not the toplevel) so the same letters typed in the comment
        # box stay text; the canvas takes focus when the pointer enters it or it is clicked.
        self.canvas.bind("<KeyPress>", self._on_key_press)
        self.canvas.bind("<KeyRelease>", self._on_key_release)
        self.canvas.bind("<Enter>", self._on_canvas_enter)
        self.canvas.bind("<FocusOut>", lambda _e: self._clear_keys())  # never miss a release
        self.canvas.focus_set()  # walkable right away, without a click first

        tk.Label(
            body,
            text=(
                "drag look · WASD or arrows walk · Q/E or PgUp/PgDn down/up · "
                "Shift fast · wheel fly · right-drag pan"
            ),
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=("TkDefaultFont", 9),
        ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

        panel = tk.Frame(body, bg=PANEL, width=320)
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        panel.grid_propagate(False)

        # title (larger font) + message head the panel; both wrap at 296 px = the 320-wide panel's
        # comment-box width (padx 12 each side), so they line up with the Comment box below.
        if title:
            tk.Label(
                panel,
                text=title,
                bg=PANEL,
                fg=FG,
                wraplength=296,
                justify="left",
                anchor="w",
                font=("TkDefaultFont", 15, "bold"),
            ).pack(fill="x", padx=12, pady=(12, 2 if message else 6))
        if message:
            tk.Label(
                panel,
                text=message,
                bg=PANEL,
                fg=FG,
                wraplength=296,
                justify="left",
                anchor="w",
                font=("TkDefaultFont", 11),
            ).pack(fill="x", padx=12, pady=((2 if title else 12), 6))

        self._build_edit_row(panel)

        self.dot_frame = tk.Frame(panel, bg=PANEL)
        self.dot_frame.pack(fill="x", padx=12, pady=4)

        # Moved props are listed here like the comment dots -- rebuilt (and its header hidden when
        # empty) by _rebuild_move_rows; each row's ✕ resets that prop to where it started. The frame
        # keeps its slot above the comment box; only the header is packed/unpacked.
        self.move_header = tk.Label(
            panel, text="Moved Objects", bg=PANEL, fg=MUTED, anchor="w", font=("TkDefaultFont", 9)
        )
        self.move_frame = tk.Frame(panel, bg=PANEL)
        self.move_frame.pack(fill="x", padx=12)

        self.comment = build_comment_box(tk, panel)
        # Enter (no Shift) with a non-empty comment submits a neutral "comment" verdict and closes,
        # like the media-review windows; Shift+Enter keeps the textarea's newline. Pass/Fail stay on
        # their buttons.
        self.comment.bind("<Return>", self._on_comment_return)

        build_button_row(
            tk,
            panel,
            [
                ("✗ Fail", lambda: self._submit("fail"), FAIL_BG),
                ("✓ Pass", lambda: self._submit("pass"), PASS_BG),
            ],
        )

    def _build_edit_row(self, panel) -> None:
        """The Move-Objects toggle and the ↶/↷ undo-redo buttons in one narrow row -- same button style
        and layout as the floorplan window's tool row."""
        tk = self.tk
        row = tk.Frame(panel, bg=PANEL)
        row.pack(fill="x", padx=8, pady=(8, 0))
        toggle = tk.Checkbutton(
            row,
            text="Move Objects",
            variable=self.edit_var,
            command=self._toggle_edit,
            indicatoron=False,
            bg="#2a2a2a",
            fg=FG,
            selectcolor=SEND_BG,
            activebackground="#333333",
            activeforeground=FG,
            relief="flat",
            font=("TkDefaultFont", 9),
            padx=0,
            bd=0,
        )
        toggle.pack(side="left", expand=True, fill="x", padx=1, ipady=4)
        add_tooltip(
            tk,
            toggle,
            "Drag a prop to move it; Shift-drag to rotate, Ctrl-drag to raise/lower."
            "\nWalls and floor are baked and cannot move.",
        )
        for icon, cmd in (("↶", self._undo_edit), ("↷", self._redo_edit)):
            tk.Button(
                row,
                text=icon,
                command=cmd,
                bg="#2a2a2a",
                fg=FG,
                width=2,
                activebackground="#333333",
                activeforeground=FG,
                relief="flat",
                font=("TkDefaultFont", 9),
                padx=0,
                pady=0,
                bd=0,
            ).pack(side="left", fill="x", padx=1, ipady=4)

    # -- rendering --
    def _decorate(self, scene) -> None:
        """Inject one marker sphere per dot (and a heading arrow for dots with a yaw) into the
        freshly-updated scene, before rasterisation."""
        for d in self.dots.dots:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([self._marker_r] * 3),
                np.array(d.world, dtype=float),
                np.eye(3).flatten(),
                np.array(color_for(d.id), dtype=np.float32),
            )
            g.label = str(d.id).encode()
            scene.ngeom += 1
            if d.yaw_deg is None or scene.ngeom >= scene.maxgeom:
                continue
            # heading arrow on the ground plane: local +z (the arrow axis) -> (cos, sin, 0) world.
            c, s = math.cos(math.radians(d.yaw_deg)), math.sin(math.radians(d.yaw_deg))
            mat = np.array([-s, 0.0, c, c, 0.0, s, 0.0, 1.0, 0.0])
            a = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                a,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array([_YAW_ARROW_W, _YAW_ARROW_W, _YAW_ARROW_LEN_M]),
                np.array(d.world, dtype=float),
                mat,
                np.array(color_for(d.id), dtype=np.float32),
            )
            scene.ngeom += 1
        self._decorate_selection(scene)

    def _decorate_selection(self, scene) -> None:
        """While a prop is dragged, redraw *its own mesh* at the target pose by rigidly transforming
        its geoms in the render scene -- the same MjvScene-decoration hook the dots use, so it touches
        neither ``model`` nor ``data``. The move maps the prop's current pose to the ghost's (a floor
        translation and/or a yaw about its centre); the on-release rebuild makes the identical move
        permanent."""
        sel, ghost = self._sel, self._ghost
        if sel is None or ghost is None:
            return
        dyaw = math.radians((ghost["yaw"] if ghost["yaw"] is not None else sel["yaw"]) - sel["yaw"])
        c, s = math.cos(dyaw), math.sin(dyaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        center_old = np.array([sel["pos"][0], sel["pos"][1], sel["pos"][2]])
        center_new = np.array([ghost["pos"][0], ghost["pos"][1], ghost["pos"][2]])
        geom_ids = sel["geom_ids"]
        geom_obj = int(mujoco.mjtObj.mjOBJ_GEOM)
        for i in range(scene.ngeom):
            g = scene.geoms[i]
            if int(g.objtype) != geom_obj or int(g.objid) not in geom_ids:
                continue
            g.pos[:] = center_new + rot @ (np.array(g.pos, dtype=float) - center_old)
            g.mat[:] = rot @ np.array(g.mat, dtype=float)  # MjvGeom.mat is a 3x3

    def _rerender(self) -> None:
        """Draw the current camera's frame into the canvas, double-buffered.

        Two failure modes, both of which show up as flicker once frames come continuously (flying)
        rather than one at a time (a drag): clearing the canvas first (``delete("all")``) leaves it
        empty until the new item is drawn, and pasting into the photo Tk is *currently displaying*
        lets a repaint catch it half-written. So the frame goes into the off-screen photo of a pair
        and the canvas item is then pointed at it -- the visible photo is never the one being written.
        """
        from PIL import Image, ImageTk

        frame = self.fr.render(self.engine.ctx.data, decorate=self._decorate)
        image = Image.fromarray(np.ascontiguousarray(frame))
        if self._photos and self._photos[0].width() == image.width:
            self._photos.reverse()  # the front photo becomes the back one, and vice versa
            self._photos[0].paste(image)
            self.canvas.itemconfigure(self._image_item, image=self._photos[0])
            return
        self._photos = [ImageTk.PhotoImage(image) for _ in range(2)]
        self.canvas.delete("all")
        self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photos[0])

    # -- camera interaction --
    def _on_press(self, event) -> None:
        self.canvas.focus_set()  # clicking the scene takes keyboard focus back from the comment box
        # In Move-Objects mode a left-press that lands on a movable prop grabs it (and does not orbit);
        # anything else (empty space, or a wall/robot) falls through to a normal camera drag.
        if self.edit_var.get() and event.num == 1 and self._begin_edit(event.x, event.y):
            return
        self._drag = (event.x, event.y, event.num)

    def _on_double_click(self, event) -> None:
        if self.edit_var.get():  # in Move-Objects mode double-click grabs, it does not drop dots
            return
        # Cancel the drag the first press of the pair started so the camera doesn't jump.
        self._drag = None
        target, world = self._pick(event.x, event.y)
        if world is None:  # clicked empty space -- no surface to anchor a dot to
            return
        # Keep the (still-held) dot so a drag now sets its heading; a plain double-click leaves it none.
        self._yaw_dot = self.dots.add(world, target=target)
        self._rebuild_dot_rows()
        self._rerender()

    def _pick(self, x: int, y: int):
        """Identify the geom/body under a click; return ``(target|None, world|None)``."""
        # mjv_select wants rely from the bottom; the scene is current from the last render.
        geom_id, body_id, world = self.fr.select(
            self.engine.ctx.data, x / self.width, 1.0 - y / self.height
        )
        if geom_id < 0:
            return None, None
        model = self.engine.ctx.model
        geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) if body_id >= 0 else None
        return {"geom": geom, "body": body}, world

    def _on_drag(self, event) -> None:
        if (
            self._sel is not None
        ):  # grabbed prop: translate, or Shift = rotate, or Ctrl = raise/lower
            self._drag_prop(event)
            return
        if self._yaw_dot is not None:  # dragging a heading out of the just-placed dot
            self._set_dot_yaw(self._yaw_dot, event.x, event.y)
            return
        if self._drag is None:
            return
        x0, y0, button = self._drag
        dx, dy = (event.x - x0) / self.height, (event.y - y0) / self.height
        if button == 3:
            self.fr.move_camera(mujoco.mjtMouse.mjMOUSE_MOVE_H, dx, dy)
        else:
            # Left-drag aims about the eye, not about the orbit point: with WASD doing the travelling,
            # an orbit would swing the camera around a pivot floating in front of it.
            self.fr.look_camera(dx, dy)
        self._drag = (event.x, event.y, button)
        self._rerender()

    def _set_dot_yaw(self, dot, x: int, y: int) -> None:
        """Set ``dot``'s heading from a drag: unproject the cursor onto the horizontal plane through
        the dot and take the angle from the dot to that point. Unprojecting to a fixed plane (rather
        than ray-picking geometry) lets the arrow sweep freely instead of snapping onto walls/props.
        Too short a drag leaves the heading unset."""
        ground = self._ground_point(x, y, dot.world[2])
        if ground is None:
            return
        dx, dy = ground[0] - dot.world[0], ground[1] - dot.world[1]
        if math.hypot(dx, dy) < _YAW_MIN_DRAG_M:
            return
        dot.yaw_deg = round(math.degrees(math.atan2(dy, dx)), 1)
        self._rebuild_dot_rows()
        self._rerender()

    def _pixel_ray(self, x: int, y: int):
        """The eye point and unit direction of the ray through pixel ``(x, y)``.

        Built from the last-rendered scene's perspective camera (its frustum + eye), so it needs a
        prior :meth:`_rerender`. Returns ``(pos, dir)`` as numpy arrays.
        """
        scene = self.fr.scene
        c0, c1 = scene.camera[0], scene.camera[1]
        pos = (np.array(c0.pos) + np.array(c1.pos)) / 2.0
        fwd = np.array(c0.forward, dtype=float)
        fwd /= np.linalg.norm(fwd)
        up = np.array(c0.up, dtype=float)
        up /= np.linalg.norm(up)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        bottom, top, near = c0.frustum_bottom, c0.frustum_top, c0.frustum_near
        half_h = (top - bottom) / 2.0
        half_w = half_h * (self.width / self.height)
        relx, rely = x / self.width, 1.0 - y / self.height  # rely from the bottom (GL convention)
        u = c0.frustum_center + (relx - 0.5) * 2.0 * half_w
        v = (bottom + top) / 2.0 + (rely - 0.5) * (top - bottom)
        d = fwd * near + right * u + up * v
        d /= np.linalg.norm(d)
        return pos, d

    def _ground_point(self, x: int, y: int, z0: float):
        """The world point where the ray through pixel ``(x, y)`` meets the horizontal plane ``z=z0``.

        Returns ``None`` if the ray is parallel to the plane.
        """
        pos, d = self._pixel_ray(x, y)
        if abs(d[2]) < 1e-9:
            return None
        return pos + (z0 - pos[2]) / d[2] * d

    def _height_point(self, x: int, y: int, cx: float, cy: float):
        """The world point where the ray through pixel ``(x, y)`` meets the vertical plane that passes
        through ``(cx, cy)`` and faces the camera (normal = the camera's horizontal look direction).

        Its ``z`` is the height the cursor points at while a prop stays fixed in plan -- what Ctrl-drag
        reads to raise/lower the prop. Returns ``None`` if the ray runs along the plane.
        """
        pos, d = self._pixel_ray(x, y)
        n = np.array(
            [d[0], d[1], 0.0]
        )  # horizontal look direction -> a vertical plane facing the eye
        nn = np.linalg.norm(n)
        if nn < 1e-9:  # looking straight down: no vertical plane to read a height from
            return None
        n /= nn
        denom = float(n @ d)
        if abs(denom) < 1e-9:
            return None
        t = float(n @ (np.array([cx, cy, 0.0]) - pos)) / denom
        return pos + t * d

    def _on_release(self, event) -> None:
        if self._sel is not None:  # finished (or skipped) a prop move
            self._commit_move()
            return
        if self._yaw_dot is not None:  # finished (or skipped) the heading drag
            self._yaw_dot = None
            self._rebuild_dot_rows()
            self._rerender()
            return
        self._drag = None

    def _on_wheel(self, event) -> None:
        # Normalise wheel direction across platforms: <Button-4/5> on Linux, event.delta elsewhere.
        up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        # The wheel flies forward/back rather than shrinking the orbit distance, so it composes with
        # WASD instead of collapsing the camera onto its own pivot.
        self.fr.dolly_camera(_WHEEL_STEP_M if up else -_WHEEL_STEP_M)
        self._rerender()

    # -- keyboard walk (WASD/QE translate the camera; the mouse still aims it) --
    def _on_canvas_enter(self, _event) -> None:
        """Take keyboard focus when the pointer enters the scene -- unless the human is mid-sentence in
        the comment box, whose text must not turn into camera moves."""
        if self.root.focus_get() is not self.comment:
            self.canvas.focus_set()

    def _on_key_press(self, event) -> None:
        key = _walk_key(event.keysym)
        if key in _SPRINT_KEYS:
            self._sprint = True
        elif key in _CRAWL_KEYS:
            self._crawl = True
        if key not in WALK_KEYS:
            return
        # A walk key's own event carries the modifier state, which is more reliable than tracking the
        # Shift/Ctrl keys alone (their press event reports the state *before* the modifier applied).
        self._sprint = bool(event.state & _SHIFT_MASK)
        self._crawl = bool(event.state & _CTRL_MASK)
        self._key_press_id[key] = self._key_press_id.get(key, 0) + 1
        if key not in self._keys:
            self._keys.add(key)
            self._start_walk()

    def _on_key_release(self, event) -> None:
        """Release a walk key -- but only after a grace period.

        X11 auto-repeat delivers a KeyRelease immediately followed by a KeyPress for a *held* key, so
        acting on the release directly would stall the camera between repeats. The key is dropped only
        if no fresh press bumped its counter meanwhile.
        """
        key = _walk_key(event.keysym)
        if key in _SPRINT_KEYS:
            self._sprint = False
        elif key in _CRAWL_KEYS:
            self._crawl = False
        if key not in self._keys:
            return
        press_id = self._key_press_id.get(key, 0)
        self.root.after(_KEY_REPEAT_GRACE_MS, lambda: self._release_key(key, press_id))

    def _release_key(self, key: str, press_id: int) -> None:
        if self._key_press_id.get(key, 0) != press_id:  # auto-repeat: the key is still down
            return
        self._keys.discard(key)

    def _clear_keys(self) -> None:
        """Drop every held key (pointer left the scene, or the window lost it) so the camera stops
        rather than drifting on a release we never see."""
        self._keys.clear()
        self._sprint = self._crawl = False

    def _start_walk(self) -> None:
        if self._walk_job is not None:
            return
        self._walk_t0 = time.monotonic()
        self._walk_job = self.root.after(_WALK_TICK_MS, self._walk_tick)

    def _stop_walk(self) -> None:
        """Cancel a pending tick so nothing renders into a window that is being torn down."""
        self._clear_keys()
        if self._walk_job is not None:
            self.root.after_cancel(self._walk_job)
            self._walk_job = None

    def _walk_tick(self) -> None:
        """Advance the camera by one frame's worth of travel and re-render, until no key is held.

        The step is measured against the wall clock rather than assumed from the tick interval, so the
        speed stays the same whether a frame renders in 5 ms or 50.
        """
        self._walk_job = None
        now = time.monotonic()
        dt = min(now - self._walk_t0, _WALK_MAX_DT_S)
        self._walk_t0 = now
        if not self._keys:
            return
        speed = _WALK_SPEED_MS * (_WALK_SPRINT if self._sprint else 1.0)
        if self._crawl:
            speed *= _WALK_CRAWL
        self.fr.walk_camera(self._keys, speed * dt)
        self._rerender()
        self._walk_job = self.root.after(_WALK_TICK_MS, self._walk_tick)

    # -- edit props --
    def _toggle_edit(self) -> None:
        if not self.edit_var.get():  # leaving edit mode drops any half-done grab
            self._sel = self._ghost = None
            self._rerender()

    def _build_movable(self) -> None:
        """Map each ``spawn_model`` prop's root-body name to its Entity. Rebuilt after every engine
        rebuild, since a fresh Engine re-registers the entities."""
        self._movable = {
            e.body: e for e in self.engine.ctx.entities.all() if e.kind == "prop" and e.body
        }

    def _movable_at(self, body_id: int):
        """The prop Entity a picked body belongs to (walking up to the prop's root), or ``None``.

        A prop can be several bodies; ``mjv_select`` may return a child link, so climb ``body_parentid``
        until a body is a registered prop root."""
        model = self.engine.ctx.model
        b = int(body_id)
        while b > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name in self._movable:
                return self._movable[name]
            b = int(model.body_parentid[b])
        return None

    def _prop_geom_ids(self, root_id: int) -> set[int]:
        """Every geom id under a prop's root body (its whole subtree), so its mesh can be redrawn at a
        dragged pose in the render scene without recompiling the model."""
        model = self.engine.ctx.model
        bodies = set()
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == root_id:
                    bodies.add(b)
                    break
                p = int(model.body_parentid[p])
        return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}

    def _spec_for(self, entity):
        """The ``spawn_model`` config entry that placed ``entity`` (matched by prefix + model), so its
        pose can be edited and the engine rebuilt. ``None`` if no entry matches."""
        prefix = entity.meta.get("prefix", "")
        model_ref = entity.meta.get("model")
        for spec in self.engine.config.plugins:
            if (
                spec.ref == "spawn_model"
                and spec.config.get("prefix", "") == prefix
                and spec.config.get("model") == model_ref
            ):
                return spec
        return None

    def _begin_edit(self, x: int, y: int) -> bool:
        """Grab the prop under the cursor. Returns True (press consumed) only when a movable prop was
        selected; False lets the press become a camera drag -- empty space, or baked geometry
        (walls/floor/robot) that has no editable pose and so simply is not grabbable."""
        geom_id, body_id, _world = self.fr.select(
            self.engine.ctx.data, x / self.width, 1.0 - y / self.height
        )
        if geom_id < 0:
            return False  # empty space -- orbit as usual
        entity = self._movable_at(body_id)
        if entity is None:
            return False  # baked geometry -- not a movable prop; orbit as usual
        spec = self._spec_for(entity)
        if spec is None:  # a prop with no editable spawn_model entry (e.g. baked into the MJCF)
            return False
        pos = list(spec.config.get("pos", [0.0, 0.0, 0.0]))
        while len(pos) < 3:
            pos.append(0.0)
        rpy = list(spec.config.get("rpy", [0.0, 0.0, 0.0]))
        yaw = math.degrees(rpy[2]) if len(rpy) >= 3 else 0.0
        root_id = mujoco.mj_name2id(self.engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        self._sel = {
            "entity": entity,
            "spec": spec,
            "pos": [float(v) for v in pos[:3]],
            "yaw": yaw,
            "geom_ids": self._prop_geom_ids(root_id),
            # snapshot of the entry before this grab; only kept for the prop's FIRST move (see
            # MoveModel.set), so a reset restores exactly what the author wrote.
            "orig_config": dict(spec.config),
        }
        self._ghost = None
        return True

    def _drag_prop(self, event) -> None:
        """Track the grabbed prop's ghost: Shift sweeps a heading about its centre, Ctrl raises/lowers
        it in place, otherwise the ghost slides to the cursor's point on the prop's floor plane."""
        sel = self._sel
        cx, cy, z0 = sel["pos"][0], sel["pos"][1], sel["pos"][2]
        if event.state & _SHIFT_MASK:  # rotate about the current centre
            ground = self._ground_point(event.x, event.y, z0)
            if ground is None:
                return
            dx, dy = ground[0] - cx, ground[1] - cy
            if math.hypot(dx, dy) < _YAW_MIN_DRAG_M:
                return
            self._ghost = {"pos": [cx, cy, z0], "yaw": round(math.degrees(math.atan2(dy, dx)), 1)}
        elif event.state & _CTRL_MASK:  # raise/lower in place, plan position unchanged
            point = self._height_point(event.x, event.y, cx, cy)
            if point is None:
                return
            z = round(max(0.0, float(point[2])), 3)  # clamp at the floor -- no burying props
            self._ghost = {"pos": [cx, cy, z], "yaw": None}
        else:  # translate on the floor (no arrow -- heading is unchanged)
            ground = self._ground_point(event.x, event.y, z0)
            if ground is None:
                return
            self._ghost = {"pos": [float(ground[0]), float(ground[1]), z0], "yaw": None}
        self._rerender()

    def _commit_move(self) -> None:
        """Write the dragged pose into the prop's ``spawn_model`` entry, record it in the moved-props
        list, and rebuild the engine so the real mesh moves. A grab with no drag (no ghost) just clears
        the selection."""
        sel, ghost = self._sel, self._ghost
        self._sel = None
        self._ghost = None
        if ghost is None:  # clicked a prop but never dragged -- nothing to commit
            self._rerender()
            return
        self._push_undo()  # snapshot the pre-move state for ↶
        final_yaw = ghost["yaw"] if ghost["yaw"] is not None else sel["yaw"]
        new_pos = [ghost["pos"][0], ghost["pos"][1], ghost["pos"][2]]
        apply_prop_pose(sel["spec"].config, new_pos, final_yaw)
        e = sel["entity"]
        self.moves.set(
            e.name, e.meta.get("model", ""), new_pos, final_yaw, sel["spec"], sel["orig_config"]
        )
        self._rebuild_engine()
        self._rebuild_move_rows()

    def _rebuild_engine(self) -> None:
        """Recompile the world from the edited config (the sanctioned way to realise a pose change --
        never a live ``model.body_pos`` write) and re-point the renderer, keeping the camera."""
        from roqsim import Engine, FrameRenderer

        cfg = self.engine.config
        cam = self.fr.camera
        new = Engine(cfg)
        new.setup()
        new.reset()
        for _ in range(max(0, self.settle_steps)):
            new.step()
        self.fr.close()
        self.fr = FrameRenderer(new.ctx.model, self.width, self.height, camera=cam)
        self.engine.shutdown()
        self.engine = new
        self._build_movable()
        self._rerender()

    # -- dot rows --
    def _rebuild_dot_rows(self) -> None:
        build_point_rows(self.tk, self.dot_frame, self.dots.dots, self._delete_dot)

    def _delete_dot(self, dot_id: int) -> None:
        self.dots.delete(dot_id)
        self._rebuild_dot_rows()
        self._rerender()

    # -- moved-prop rows --
    def _rebuild_move_rows(self) -> None:
        """(Re)draw the moved-props list, one row per prop (no per-row comment); show the header only
        when something has moved. Uses a single accent swatch, since these rows have no numbered 3D
        marker -- the moved mesh itself is the marker."""
        build_point_rows(
            self.tk,
            self.move_frame,
            self.moves.moves,
            self._reset_move,
            color=lambda _id: _MOVE_SWATCH_RGBA,
            with_comment=False,
        )
        if self.moves.moves:
            self.move_header.pack(fill="x", padx=12, pady=(6, 0), before=self.move_frame)
        else:
            self.move_header.pack_forget()

    def _reset_move(self, move_id: int) -> None:
        """The ✕ on a row: restore that prop's original ``spawn_model`` config and rebuild, then drop
        the row. Undoable. The stored ``spec`` is stable across rebuilds (Engine reuses the config
        objects)."""
        m = self.moves.get(move_id)
        if m is None:
            return
        self._push_undo()
        if m.spec is not None and m.orig_config is not None:
            m.spec.config.clear()
            m.spec.config.update(m.orig_config)
        self.moves.delete(move_id)
        self._rebuild_engine()
        self._rebuild_move_rows()

    # -- undo / redo (snapshots of the moved-prop state: the rows + the spawn_model poses) --
    def _spawn_specs(self) -> list:
        return [s for s in self.engine.config.plugins if s.ref == "spawn_model"]

    def _snapshot(self):
        """Capture what an edit changes: the moved-prop rows and every ``spawn_model`` pose. Moves are
        copied field-wise (keeping the live ``spec`` reference so a later reset still works); configs
        are copied by value against their spec."""
        moves = [
            Move(
                m.id,
                m.entity,
                m.model,
                list(m.pos),
                m.yaw_deg,
                m.spec,
                dict(m.orig_config) if m.orig_config is not None else None,
            )
            for m in self.moves.moves
        ]
        specs = [(s, copy.deepcopy(s.config)) for s in self._spawn_specs()]
        return (moves, specs)

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()  # a fresh edit invalidates the redo branch

    def _restore(self, snap) -> None:
        moves, specs = snap
        for spec, config in specs:
            spec.config.clear()
            spec.config.update(copy.deepcopy(config))
        self.moves.moves = [
            Move(
                m.id,
                m.entity,
                m.model,
                list(m.pos),
                m.yaw_deg,
                m.spec,
                dict(m.orig_config) if m.orig_config is not None else None,
            )
            for m in moves
        ]
        self._rebuild_engine()
        self._rebuild_move_rows()

    def _undo_edit(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())

    def _redo_edit(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())

    # -- submit / close --
    def _on_comment_return(self, event):
        """Enter in the comment box submits a neutral ``comment`` verdict and closes; Shift+Enter keeps
        the newline, and an empty comment does nothing (an accidental Enter, not a note)."""
        if event.state & _SHIFT_MASK:
            return None  # let the Text widget insert a newline
        if self.comment.get("1.0", "end-1c").strip():
            self._submit("comment")
        return "break"  # swallow the Enter either way, so no stray newline is left behind

    def _submit(self, verdict: str) -> None:
        self._stop_walk()
        write_result(
            self.json_out,
            verdict,
            self.comment.get("1.0", "end-1c"),
            self.dots,
            self.moves.to_payload(),
        )
        self.exit_code = 1 if verdict == "fail" else 0  # pass and comment are both a clean return
        self.root.destroy()

    def _on_close(self) -> None:
        self._stop_walk()
        self.exit_code = 3
        self.root.destroy()
