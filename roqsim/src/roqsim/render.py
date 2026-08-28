"""Render a world, scene, model or mesh to an image, headless.

This is the *tool*; the rendering library it drives is :mod:`roqsim.rendering` (``FrameRenderer`` and the
camera maths). Same relationship ``roqsim.export_web`` has with ``roqsim export web``.

    roqsim render world.yaml --out top.png --view elevation=-85 distance=45
    roqsim render roqsim_assets:industrial_table --out table.png        # one model, auto-framed
    roqsim render roqsim_scenes:depot --no-ceiling --out room.png
    roqsim render tiago_pick:tiago_pick --focus parcel --out parcel.png
    roqsim render prop.obj --out prop.png                        # a raw mesh, pre-finalization

The positional argument takes the *same* shapes as ``roqsim sim`` (see :func:`roqsim.runner.config_for_input`)
plus one more: a raw mesh. ``roqsim sim`` refuses meshes on purpose -- loose geometry is not something you
can meaningfully simulate -- but rendering one is both harmless and useful, so it is accepted here and
wrapped in a preview scene (floor, light, its own baseColor texture).

Output type follows ``--out``'s extension, so stills and video need no mutually exclusive flags:
``.png``/``.jpg`` is one frame, ``.webm``/``.mp4``/``.mkv`` is video. Video needs a recording to animate
(``--state``), because a world on its own has exactly one frame.

With ``--state`` the world target becomes **optional** -- the recording's provenance names it, so a
caller need not repeat what the file already knows.

**Stdout is exactly one line of JSON** and nothing else, so a caller parses rather than scrapes. Progress
and diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import logging_setup
from .capture import CaptureError
from .config import _VIEW_KEYS, PluginError, overrides_from_dotlist
from .models import ModelError
from .recording import RecordingError
from .rendering import FrameRenderer, GLBackendError, check_gl_backend, focus_camera
from .viewer import GL_HELP, DisplayError

log = logging.getLogger(__name__)

#: Extensions this tool writes a single frame to. A video extension needs ``--state``: a world on its
#: own has exactly one frame, so asking for a video of it is a mistake worth naming.
_IMAGE_EXT = (".png", ".jpg", ".jpeg")
_VIDEO_EXT = (".webm", ".mp4", ".mkv")

_DEFAULT_OUT = "render.png"
_DEFAULT_SIZE = "960x540"

#: Container -> ffmpeg encoder arguments. Measured on 250 frames at 960x540 of rendered-scene-like
#: content, and the result contradicts the usual assumption: **VP9 at ``-deadline realtime -cpu-used 8``
#: costs 3.2x less CPU than VP8** (1.34 s vs 4.26 s) because libvpx's VP8 realtime path parallelises
#: poorly, while VP8 actually compresses slightly *better* at those settings (0.31 vs 0.44 MB). VP9's
#: compression edge only appears at ``-deadline good``, which costs 4.4x the CPU -- the wrong trade.
#: x264 is both fastest and smallest (0.36 s, 0.09 MB) and gets a fragmented MP4 so a killed encode
#: still leaves a playable prefix.
_ENCODERS = {
    ".webm": [
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "34",
        "-b:v",
        "0",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
    ],
    ".mp4": [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+frag_keyframe+empty_moov",
    ],
    ".mkv": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"],
}


class RenderError(RuntimeError):
    """``roqsim render`` cannot produce the image it was asked for (see the message)."""


# Exit codes, distinct so an orchestrator can react without matching on stderr text.
EXIT_BAD_ARGS = 2
EXIT_NO_GL = 3
EXIT_PROVENANCE = 4


def parse_size(text: str) -> tuple[int, int]:
    """``"960x540"`` -> ``(960, 540)``. Even in both dimensions, because ``yuv420p`` needs it.

    Rounding up rather than refusing an odd size: the caller asked for a picture, not for a lecture
    about chroma subsampling, and a one-pixel difference is not what they were expressing.
    """
    try:
        w_text, h_text = text.lower().split("x")
        width, height = int(w_text), int(h_text)
    except ValueError as err:
        raise RenderError(f"--size {text!r}: expected WxH, e.g. 960x540") from err
    if width < 16 or height < 16:
        raise RenderError(f"--size {text!r}: both dimensions must be at least 16")
    return width + (width & 1), height + (height & 1)


def _is_number(token: str) -> bool:
    """True when ``token`` is a bare number, which in a ``--view`` list can only be a vector element."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _rejoin_split_vectors(pairs: list[str]) -> list[str]:
    """Fold a bare number back onto the pair before it: ``["lookat=-3.2", "-1.3", "1.9"]`` -> one pair.

    ``--view lookat=-3.2 -1.3 1.9`` is a natural way to write a three-vector, and the shell hands it to
    argparse as three tokens (the negative ones arrive at all only because argparse lets number-shaped
    tokens be values rather than flags). A bare *non*-number token is left alone, so it still reaches the
    KEY=VALUE error below -- that one is the greedy-nargs-ate-the-target case, a different mistake with a
    different fix.
    """
    joined: list[str] = []
    for token in pairs:
        if joined and "=" not in token and _is_number(token):
            joined[-1] += f" {token}"
        else:
            joined.append(token)
    return joined


def view_overrides(pairs: list[str] | None) -> dict:
    """Turn ``["azimuth=90", "lookat=1,2,0"]`` into ``{"sim": {"view": {...}}}``.

    A vector value may be written comma-separated (``lookat=1,2,0``) or space-separated (MJCF's
    spelling, ``lookat="1 2 0"``, split by the shell or not); both end up as three numbers.

    Deliberately sugar over the same override path ``--set`` uses, so ``sim.view`` has exactly one
    validator and one vocabulary: unknown keys are rejected by :func:`roqsim.config.load_config` with the
    identical message a world YAML gets, and there is no second camera grammar to keep in step. Keys are
    checked here too, before any file is opened, so a typo fails in a millisecond rather than after a
    world compiles.
    """
    if not pairs:
        return {}
    pairs = _rejoin_split_vectors(pairs)
    unknown = sorted({p.split("=", 1)[0].strip() for p in pairs if "=" in p} - set(_VIEW_KEYS))
    if unknown:
        raise RenderError(
            f"--view: unknown key(s) {', '.join(unknown)}; --view sets the camera only "
            f"({', '.join(sorted(_VIEW_KEYS))}), the same keys a world's sim.view accepts."
        )
    bad = [p for p in pairs if "=" not in p]
    if bad:
        # The likely cause is a greedy --view eating the positional target, so name that rather than
        # only restating the grammar.
        raise RenderError(
            f"--view {bad[0]!r}: expected KEY=VALUE, e.g. --view azimuth=90. If that was meant to be "
            "the thing to render, put it before the flag: roqsim render <target> --view ..."
        )
    view = overrides_from_dotlist(pairs)
    if isinstance(lookat := view.get("lookat"), str):
        # A three-vector arrives written either way: `lookat=1,2,0` is how it goes on a command line and
        # what the world YAML's own `[0, 0.4, 0.9]` suggests, while `lookat="-3.2 -1.3 1.9"` is MJCF's
        # spelling and the one a caller reaches for after reading a pose off a previous render. YAML
        # reads both as a single scalar string, so both are re-read as numbers here, in the sugar layer,
        # and the value arriving at sim.view is the list it was meant to be. Anything still malformed is
        # left alone for the single sim.view validator to reject, which quotes the text as it was typed.
        with contextlib.suppress(ValueError):
            view["lookat"] = [float(n) for n in lookat.strip(" []").replace(",", " ").split()]
    return {"sim": {"view": view}}


def is_mesh(target: str) -> bool:
    """True when ``target`` names a raw mesh file, which needs the preview-scene branch."""
    from .runner import _MESH_EXT

    return Path(target).suffix.lower() in _MESH_EXT


def mesh_scene(mesh_path: str) -> tuple[mujoco.MjModel, mujoco.MjvCamera]:
    """Compile a raw mesh into a lit, grounded preview scene and frame a 3/4 camera on it.

    Delegates the scene build to :func:`roqsim.mesh_preview.build_mesh_scene`, which owns the parts that
    are not obvious: shell inertia (a decimated import can have flipped faces, giving a near-zero
    volume that fails the default volume-based inertia) and wiring the sibling ``.mtl``'s ``map_Kd``
    baseColor PNG, since MuJoCo never reads textures from an OBJ/MTL itself.
    """
    from .mesh_preview import build_mesh_scene

    model = build_mesh_scene(mesh_path)
    mid = model.mesh("prop").id
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[adr : adr + num]
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = ((lo + hi) / 2).tolist()
    cam.distance = float(np.linalg.norm(hi - lo)) * 1.6 + 0.5
    cam.azimuth, cam.elevation = 45.0, -20.0
    return model, cam


def tilt_preview_light(model, data) -> int:
    """Tilt a straight-down light off vertical for a MODEL PREVIEW. Returns how many were moved.

    A model shown by itself is welded into ``empty_room``, whose single ceiling light points straight
    down -- which is the worst case for shadow-mapped rendering of a robot, because a robot's outer
    walls are vertical and therefore parallel to the light. MuJoCo's shadow bias is fixed, and a thin
    closed visual shell (a bumper, a body cover) is thinner than one shadow texel once the shadow map
    is stretched over a room, so the shell's far face shadows its own near face: the surface comes out
    combed with dark streaks that read as a defect in the mesh. It cost one investigation already.

    Measured, so the cheaper-looking knobs are not tried again: ``shadowclip`` changes nothing (it
    clips depth, not the footprint) and resolution barely helps -- at ``shadowsize`` 16384, a 1 GB
    depth texture, the streaks are still there. Tilting keeps the contact shadow that tells a viewer
    whether a part is floating, and lights form better than a top-down lamp does.

    It is HALF the fix, not all of it: the tilt removes the streaks that run down a shell's flank, but
    the shadow map still combs the terminator itself, where the surface turns away from the light and
    no fixed bias can cover a texel's depth spread. :func:`fill_preview_self_shadows` handles that
    half, and is why this one no longer claims to remove the artefact outright.

    PREVIEWS ONLY. In a world, that light is the world's own lighting and a campaign's images depend
    on it; this is why the fix lives here and not in the world definition.
    """
    tilted = 0
    for i in range(model.nlight):
        direction = np.asarray(model.light_dir[i], dtype=float)
        norm = float(np.linalg.norm(direction))
        # Within ~8 degrees of straight down. A light the world aimed deliberately is left alone.
        if norm == 0.0 or -direction[2] / norm < 0.99:
            continue
        height = float(model.light_pos[i][2])
        offset = 0.8 * height  # ~40 deg above the horizon, still inside the room's walls
        model.light_pos[i] = [offset, -offset, height]
        aim = -np.asarray(model.light_pos[i], dtype=float)
        model.light_dir[i] = aim / np.linalg.norm(aim)
        tilted += 1
    if tilted:
        mujoco.mj_forward(model, data)  # the renderer reads data.light_xpos/xdir, not the model's
    return tilted


#: Preview lighting split (see :func:`fill_preview_self_shadows`). The shadow-casting room light keeps
#: enough diffuse to cast a readable contact shadow; the headlight carries the modelling light. Public
#: because the thumbnail renderer builds its own preview scene and must light it the same way -- a
#: thumbnail and `roqsim render` are supposed to be the same picture.
PREVIEW_HEADLIGHT_DIFFUSE = 0.85
PREVIEW_HEADLIGHT_AMBIENT = 0.20
PREVIEW_LIGHT_DIFFUSE = 0.35
PREVIEW_LIGHT_AMBIENT = 0.30


def fill_preview_self_shadows(model) -> None:
    """Move a MODEL PREVIEW's modelling light into the headlight, leaving the room light to cast.

    The second half of the preview shadow fix (:func:`tilt_preview_light` is the first). Where a robot's
    surface turns away from the tilted room light, the shadow map combs the terminator into a ragged
    white/grey border that reads as two meshes intersecting. It is not the meshes: hiding a robot's
    visual geometry and rendering only its primitive collision capsules reproduces the same border, and
    the border disappears entirely with ``castshadow`` off.

    MuJoCo 3.11 exposes no shadow bias, and the knobs it does expose do not reach this: ``shadowclip``
    is inert, ``shadowsize`` 16384 only thins the comb, and ``shadowscale``, the spot cutoff and the
    light distance each trade the comb for a blockier or dimmer shadow. What works is denying the
    shadow map the contrast it needs to show: the headlight is attached to the camera and casts no
    shadow, so lighting the model with it and leaving the room light only strong enough to darken the
    floor puts the acne band on an already-lit surface. Measured on xarm7, terminator contrast 0.72 ->
    0.28, with the contact shadow intact.

    Worst on a matte pure-white robot (xarm7, m1013), where the lit side saturates and the shadowed
    side has nothing but ambient to fall back on.

    PREVIEWS ONLY, and only for lights :func:`tilt_preview_light` recognised as the built-in room's --
    a world that aimed its own lights owns its own look.
    """
    model.vis.headlight.diffuse = [PREVIEW_HEADLIGHT_DIFFUSE] * 3
    model.vis.headlight.ambient = [PREVIEW_HEADLIGHT_AMBIENT] * 3
    for i in range(model.nlight):
        if not model.light_castshadow[i]:
            continue
        model.light_diffuse[i] = [PREVIEW_LIGHT_DIFFUSE] * 3
        model.light_ambient[i] = [PREVIEW_LIGHT_AMBIENT] * 3


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Pose ``data`` at the model's ``home`` keyframe when it has one, else ``qpos0``; then forward.

    For an articulated robot the two are very different and ``qpos0`` is often not a pose anyone would
    recognise -- the TIAGo Pro's arms stick straight out in front of it at ``qpos0``. Rendering what the
    model actually spawns as is what a person expects, and it is what ``make thumbnails`` already does,
    so sharing this keeps a model's thumbnail and its ``roqsim render`` output identical.
    """
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)


def _entity_bodies(model: mujoco.MjModel, ctx, names: list[str]) -> list[int]:
    """Body ids for ``--focus`` names, resolved as an entity first and then as a plain body name."""
    ids: list[int] = []
    missing: list[str] = []
    for name in names:
        entity = ctx.entities.get(name) if ctx is not None else None
        body = getattr(entity, "body", None) or name
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            missing.append(name)
        else:
            ids.append(bid)
    if missing:
        known = sorted(ctx.entities.names()) if ctx is not None else []
        hint = f" Known entities: {', '.join(known)}." if known else ""
        raise RenderError(f"--focus: no entity or body named {', '.join(missing)}.{hint}")
    return ids


def resolve_camera(
    model, data, ctx, view: dict | None, focus: list[str] | None, aspect: float, *, preview: bool
):
    """The camera for one frame: ``--focus`` search, else whatever the viewer would have used.

    The no-focus branch hands a :class:`_CamShim` to :func:`roqsim.viewer.setup_camera`, so this inherits
    the *whole* ``sim.view`` surface -- a ``track`` target, ``follow_heading``, and the single-model
    preview framing -- rather than reimplementing any of it. A second implementation of ``sim.view`` is
    the drift the frozen key set exists to prevent.

    With ``--focus``, the occlusion search picks the base camera and the world's stated keys are then
    applied on top, which is what makes ``--view`` win per key with no precedence logic of its own.
    """
    from .viewer import apply_view, setup_camera

    shim = _CamShim(_default_free(model))
    if focus:
        shim.cam = focus_camera(model, data, _entity_bodies(model, ctx, focus), aspect=aspect)
        apply_view(shim, view)
        return shim.cam
    if ctx is None:  # a mesh preview has no engine context, so no sim.view to honour
        apply_view(shim, view)
        return shim.cam
    setup_camera(shim, view, ctx, preview=preview)
    return shim.cam


def _default_free(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    return cam


class _CamShim:
    """The three members :func:`roqsim.viewer.apply_view` touches, backed by a plain ``MjvCamera``.

    Lets the headless path reuse the viewer's own ``sim.view`` handling -- including ``track``, which
    ``mjv_updateScene`` honours on the struct -- instead of reimplementing it here. A second
    implementation of ``sim.view`` is exactly the kind of drift the frozen key set exists to prevent.
    """

    def __init__(self, cam: mujoco.MjvCamera) -> None:
        self.cam = cam

    def lock(self):
        import contextlib

        return contextlib.nullcontext()

    def sync(self) -> None:
        pass


def _disable_ceiling(cfg) -> bool:
    """Turn off the world's ``ceiling`` plugin if it has one. Returns whether anything changed.

    Applied to the *parsed* config rather than as a ``--set`` override, because an override that
    matches no component is refused -- correct for ``--set`` (a typo there would otherwise be a
    silent no-op) but wrong for a convenience flag whose intent a ceiling-less world already
    satisfies. Failing there also produced a genuinely hostile message: the refusal lists what the
    document has, which for a populated scene is several hundred names.

    Sets the plugin's own ``keep``, not the reserved ``enabled:`` sibling: this plugin opens the roof
    by *removing* geometry, so turning the component off would leave the ceiling standing.
    """
    changed = False
    for spec in cfg.plugins:
        if spec.ref == "ceiling" or spec.name == "ceiling":
            spec.config["keep"] = False
            changed = True
    return changed


def _preflight(out: Path, size: tuple[int, int], *, check: bool = False) -> None:
    """Refuse now, before any world compiles, what would otherwise fail minutes later.

    ``check`` keeps this side-effect-free: a dry run must not create the output directory it was only
    asked to validate.
    """
    suffix = out.suffix.lower()
    if suffix not in _IMAGE_EXT + _VIDEO_EXT:
        raise RenderError(
            f"--out {out.name}: unknown extension {suffix or '(none)'}. Images: "
            f"{', '.join(_IMAGE_EXT)}; video: {', '.join(_VIDEO_EXT)}."
        )
    parent = out.parent if str(out.parent) else Path()
    if not check:
        parent.mkdir(parents=True, exist_ok=True)
    elif not parent.exists():
        return  # nothing more to check about a directory a real run would create
    if not os.access(parent, os.W_OK):
        raise RenderError(f"--out {out}: {parent} is not writable")
    if suffix in _VIDEO_EXT and shutil.which("ffmpeg") is None:
        raise RenderError(
            f"--out {out.name} is a video and ffmpeg is not on PATH. Install it "
            "(apt install ffmpeg), or render a still to a .png."
        )
    # Asked of the backend mujoco actually bound, not of DISPLAY. The two are unrelated -- a
    # container image can export DISPLAY=:0 with no X server behind it, which is exactly how the
    # old `not has_display()` spelling let a doomed render through -- and MUJOCO_GL cannot be
    # trusted either, since it is read once during `import mujoco` and may have been set after.
    try:
        check_gl_backend()
    except GLBackendError as err:
        raise RenderError(
            f"{err}\n  (Set MUJOCO_GL=egl for a GPU, or MUJOCO_GL=osmesa for CPU-only.)"
        ) from err
    if size[0] < 16 or size[1] < 16:  # pragma: no cover - parse_size already refuses this
        raise RenderError(f"--size {size[0]}x{size[1]}: too small to render")


def build_target(
    target: str,
    overrides: dict | None,
    *,
    no_ceiling: bool = False,
    skip_transport: bool = True,
    world_model: dict | None = None,
):
    """Compile ``target`` and return ``(model, data, ctx, view, camera_or_None)``.

    A mesh short-circuits to the preview scene (no engine, no plugins, no ``sim.view``); everything else
    goes through the same dispatch and the same plugin build as ``roqsim sim``, minus the plugins that
    contribute nothing to the scene, so a world renders exactly as it simulates.

    That subtraction (``skip_transport``, see :func:`roqsim.config.drop_transport_plugins`) is what lets a
    ``*_ros`` world be rendered without ROS installed. Pass ``skip_transport=False`` to demand the
    simulator's own strict build.

    ``world_model`` is a recording's resolved component tree (:meth:`roqsim.config.SimConfig.as_record`).
    When given, the config is READ from it rather than resolved from ``target`` and ``overrides`` --
    so a recording renders the components that ran, whatever has happened to the override grammar
    since. ``target`` is still used for the assets those components name.
    """
    if is_mesh(target):
        model, cam = mesh_scene(target)
        data = mujoco.MjData(model)
        reset_to_home(model, data)
        return model, data, None, None, cam

    from .config import drop_transport_plugins
    from .engine import Engine
    from .runner import config_for_input

    if world_model is not None:
        from .config import SimConfig

        cfg = SimConfig.from_record(world_model)
    else:
        cfg = config_for_input(target, overrides)
    if skip_transport:
        transport, unavailable = drop_transport_plugins(cfg)
        if transport:
            log.info("not needed for a render, skipping: %s", ", ".join(transport))
        if unavailable:
            log.warning(
                "skipping plugin(s) this environment cannot load: %s. The geometry is unaffected "
                "(a transport plugin builds none) -- but check the spelling if you expected one.",
                ", ".join(unavailable),
            )
    if no_ceiling:
        _disable_ceiling(cfg)
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    reset_to_home(engine.ctx.model, engine.ctx.data)
    from .runner import is_model_ref

    if is_model_ref(target) and tilt_preview_light(engine.ctx.model, engine.ctx.data):
        # Only when the tilt recognised the built-in room's light: a world that aimed its own keeps them.
        fill_preview_self_shadows(engine.ctx.model)
    # Keep the engine reachable: a sensor replay has to run the plugins' post_step, and rebuilding the
    # world a second time to get at them would be both slow and a chance for the two to diverge.
    engine.ctx.engine = engine
    return engine.ctx.model, engine.ctx.data, engine.ctx, cfg.view, None


def render_target(
    target: str | None = None,
    out: str | Path = _DEFAULT_OUT,
    *,
    size: str | tuple[int, int] = _DEFAULT_SIZE,
    view: list[str] | None = None,
    focus: list[str] | None = None,
    camera: str | None = None,
    no_ceiling: bool = False,
    overrides: dict | None = None,
    check: bool = False,
    state: str | Path | None = None,
    at: float | None = None,
    start: float | None = None,
    stop: float | None = None,
    fps: str | None = None,
    speed: float | None = None,
) -> dict:
    """Render ``target`` (or a recording) to ``out`` and return the result record the CLI prints.

    The one entry point every caller shares -- the CLI, the MCP tool (via a subprocess) and
    ``render_thumbnails`` -- so a model looks the same however it was asked for.
    """
    out = Path(out)
    width, height = parse_size(size) if isinstance(size, str) else size
    _preflight(out, (width, height), check=check)
    video = out.suffix.lower() in _VIDEO_EXT
    if video and not state:
        raise RenderError(
            f"--out {out.name} is a video, which needs a recording to animate: pass "
            "--state run.npz. A world on its own has exactly one frame."
        )
    if not state and not target:
        raise RenderError(
            "nothing to render: give a world/scene/model/mesh, or --state a recording."
        )
    if not state and (at is not None or start is not None or stop is not None):
        raise RenderError("--at/--from/--to select a moment in a recording, so they need --state.")
    if at is not None and (start is not None or stop is not None):
        raise RenderError(
            "--at picks one moment and --from/--to pick a range; use one or the other."
        )
    if camera and (view or focus):
        raise RenderError(
            "--camera renders through a fixed MJCF camera, which owns its own pose, so it cannot be "
            "combined with --view or --focus."
        )

    merged = dict(overrides or {})
    for key, value in view_overrides(view).items():
        merged[key] = {**merged.get(key, {}), **value} if isinstance(value, dict) else value

    if state:
        return _render_recording(
            state,
            target,
            out,
            (width, height),
            merged,
            view,
            focus,
            camera,
            no_ceiling,
            check,
            at,
            start,
            stop,
            fps,
            speed,
            video,
        )

    # --no-ceiling is applied to the parsed config, not merged in here; see _disable_ceiling.
    model, data, ctx, world_view, fixed_cam = build_target(
        target, merged or None, no_ceiling=no_ceiling
    )
    cam = _pick_camera(
        model, data, ctx, world_view, focus, camera, fixed_cam, target, width, height
    )

    record = _base_record(out, width, height, model, cam)
    if check:
        record["rendered"] = False
        return record
    _render_one(model, data, cam, width, height, out)
    record["rendered"] = True
    return record


def _pick_camera(model, data, ctx, world_view, focus, camera, fixed_cam, target, width, height):
    if camera:
        return _fixed_camera(model, camera)
    if fixed_cam is not None:
        return fixed_cam
    from .runner import is_model_ref

    return resolve_camera(
        model,
        data,
        ctx,
        world_view,
        focus,
        width / height,
        preview=bool(target) and is_model_ref(target),
    )


def _base_record(out: Path, width: int, height: int, model, cam) -> dict:
    return {
        "path": str(out.resolve()),
        "width": width,
        "height": height,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "camera": _camera_record(cam),
    }


def _render_one(model, data, cam, width: int, height: int, out: Path) -> None:
    try:
        frame = FrameRenderer(model, width, height, camera=cam)
    except Exception as err:  # noqa: BLE001 - any GL init failure maps to the same guidance
        raise RenderError(GL_HELP.format(err=err)) from err
    try:
        from PIL import Image

        Image.fromarray(frame.render(data)).save(out)
    finally:
        frame.close()


def _render_recording(
    state,
    target,
    out,
    size,
    merged,
    view,
    focus,
    camera,
    no_ceiling,
    check,
    at,
    start,
    stop,
    fps,
    speed,
    video,
):
    """Render one moment, or a range, from a recording. The world comes from its provenance."""
    from .recording import RecordingError, open_recording

    width, height = size
    rec = open_recording(state)
    try:
        model, ctx = rec.build(target, no_ceiling=no_ceiling)
    except RecordingError:
        raise
    # The world's own sim.view is the baseline, so a replay of world X looks like a render of world X.
    # A recorded camera track (below) wins over it, and --view/--focus/--camera win over both.
    world_view = rec.view

    if check:
        record = rec.describe()
        record.update(
            {"out": str(out.resolve()), "width": width, "height": height, "rendered": False}
        )
        return record

    if video:
        return _render_video(
            rec, model, ctx, out, size, view, focus, camera, start, stop, fps, speed, world_view
        )

    sample = rec.at(at)
    if at is None:
        # Say which moment this is: "the last sample" is a choice the caller did not make, and they would
        # otherwise have to infer it from the JSON. Stderr, so stdout stays a clean machine contract.
        _t0, t1 = rec.span
        log.info(
            "rendering the LAST of %d samples (t=%.3f s of %.3f); use --at T for another moment",
            len(rec),
            sample.sim_time,
            t1,
        )
    cam = _recorded_camera(model, sample, ctx, world_view, view, focus, camera, width, height)
    record = _base_record(out, width, height, model, cam)
    record.update(rec.at_record(at, sample))
    _render_one(model, sample.data, cam, width, height, out)
    record["rendered"] = True
    return record


def _recorded_camera(model, sample, ctx, world_view, view, focus, camera, width, height):
    """The recorded session camera by default; ``--camera``/``--focus``/``--view`` override it.

    Following the recorded camera is what makes a replay show *what the person was looking at*, mouse
    drags and arrow-key flight included -- something no live encoder could offer, since it could only
    ever have captured one camera.
    """
    if camera:
        return _fixed_camera(model, camera)
    if focus or view:
        return resolve_camera(
            model, sample.data, ctx, world_view, focus, width / height, preview=False
        )
    if sample.camera is not None:
        return sample.camera
    return resolve_camera(model, sample.data, ctx, world_view, None, width / height, preview=False)


def _render_video(
    rec, model, ctx, out, size, view, focus, camera, start, stop, fps, speed, world_view=None
):
    """Render every sample in range and pipe it to ffmpeg. One sample is always exactly one frame.

    ``--fps``/``--speed`` change only the *declared* rate, never which samples are used, so no frame is
    ever duplicated, decimated or interpolated: every frame in the file is a state the simulation
    actually had.
    """
    from fractions import Fraction

    width, height = size
    rate = rec.fps
    if fps is not None:
        from .capture import parse_fps

        rate = parse_fps(fps)
    elif speed is not None:
        rate = rec.fps * Fraction(str(speed))
    declared = f"{rate.numerator}/{rate.denominator}"

    renderer = None
    count = 0
    progress = _Progress(_frames_in_range(rec, start, stop), f"rendering {out.name}", log)

    def frames():
        nonlocal renderer, count
        for sample in rec.range(start, stop):
            cam = _recorded_camera(
                model, sample, ctx, world_view, view, focus, camera, width, height
            )
            if renderer is None:
                try:
                    renderer = FrameRenderer(model, width, height, camera=cam)
                except Exception as err:  # noqa: BLE001
                    raise RenderError(GL_HELP.format(err=err)) from err
            else:
                renderer.camera = cam
            count += 1
            progress.tick()
            yield renderer.render(sample.data)

    try:
        encode_frames(frames(), out, declared, size)
    finally:
        progress.done()
        if renderer is not None:
            renderer.close()

    record = rec.describe()
    record.update(
        {
            "path": str(out.resolve()),
            "width": width,
            "height": height,
            "frames": count,
            "declared_fps": float(rate),
            "speed": round(float(rate / rec.fps), 4),
            "rendered": True,
        }
    )
    log.info(
        "video: %d frames at %s fps (%.2fx sim time) -> %s",
        count,
        float(rate),
        float(rate / rec.fps),
        out,
    )
    return record


def _fixed_camera(model, name: str):
    """Resolve ``--camera NAME`` to a fixed MJCF ``<camera>``, or say what the world does offer.

    ``FrameRenderer`` accepts a camera name directly, so this only has to validate -- but validating is
    the point: an unknown name would otherwise render through MuJoCo's default camera and look like a
    working answer to the wrong question.
    """
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
        known = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        have = ", ".join(n for n in known if n) or "none"
        raise RenderError(f"--camera {name!r}: this world declares no such camera. It has: {have}.")
    return name


class _Progress:
    """Report a long render's progress without filling a log with it.

    Two audiences that want opposite things. A person at a terminal wants to watch it move: one line,
    rewritten in place with ``\r``, costing no scrollback. A log file wants a handful of durable lines,
    not a thousand -- so when stderr is not a TTY this emits at most one line per :data:`_LOG_EVERY` of
    the work. That keeps a campaign's captured output readable while still showing something is happening.
    """

    #: Fraction of the work between log lines when nobody is watching a terminal.
    _LOG_EVERY = 0.25

    def __init__(self, total: int | None, label: str, logger: logging.Logger) -> None:
        self.total = total
        self.label = label
        self.log = logger
        self.count = 0
        self._next_log = self._LOG_EVERY
        self._tty = sys.stderr.isatty()

    def tick(self) -> None:
        self.count += 1
        if self._tty:
            if self.total:
                pct = 100.0 * self.count / self.total
                sys.stderr.write(f"\r{self.label}: {self.count}/{self.total} frames ({pct:.0f}%)")
            else:
                sys.stderr.write(f"\r{self.label}: {self.count} frames")
            sys.stderr.flush()
        elif self.total and self.count / self.total >= self._next_log:
            self.log.info("%s: %d/%d frames", self.label, self.count, self.total)
            self._next_log += self._LOG_EVERY

    def done(self) -> None:
        if self._tty:
            # Clear the in-place line so it cannot collide with whatever is printed next.
            sys.stderr.write("\r" + " " * 70 + "\r")
            sys.stderr.flush()


def _frames_in_range(rec, start, stop) -> int | None:
    """How many samples a range covers, so progress can show a percentage rather than a bare count."""
    try:
        lo = 0 if start is None else rec.index_at(start)
        hi = len(rec) - 1 if stop is None else rec.index_at(stop)
        return abs(hi - lo) + 1
    except Exception:  # noqa: BLE001 - a count is a nicety; a render must not fail for want of one
        return None


def encode_frames(frames, out: Path, fps: str, size: tuple[int, int]) -> None:
    """Pipe raw RGB frames to ffmpeg. ``fps`` is an exact rational string, e.g. ``500/17``.

    A rational rather than a rounded decimal because the samples are spaced ``1/(k*dt)`` apart: declaring
    ``-r 29.41`` on a stream whose real spacing is ``500/17`` drifts, which is the whole point of the
    capture-rate machinery.

    Raw ``rgb24`` in, not JPEG: ``FrameRenderer.render`` already returns a C-contiguous uint8 array, so
    there is nothing to gain by encoding twice.
    """
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        fps,
        "-i",
        "pipe:0",
        "-an",
        *_ENCODERS[out.suffix.lower()],
        str(out),
    ]
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    written = 0
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
            written += 1
    except BrokenPipeError as err:
        tail = (proc.stderr.read() or b"").decode(errors="replace").strip()[-2000:]
        raise RenderError(f"ffmpeg stopped after {written} frames: {tail or err}") from err
    finally:
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
    code = proc.wait()
    tail = (proc.stderr.read() or b"").decode(errors="replace").strip()[-2000:]
    if code != 0:
        raise RenderError(f"ffmpeg failed writing {out} (exit {code}): {tail}")
    if not written:
        raise RenderError(
            f"no frames were rendered, so {out} is not a video. Check --from/--to against the "
            "recording's span."
        )


def _camera_record(cam) -> dict:
    """The camera as plain JSON, in the same vocabulary ``--view`` accepts, so it can be fed back."""
    if not isinstance(cam, mujoco.MjvCamera):  # a fixed MJCF camera, named or by id
        return {"fixed": cam if isinstance(cam, str) else int(cam)}
    return {
        "lookat": [round(float(v), 4) for v in cam.lookat],
        "distance": round(float(cam.distance), 4),
        "azimuth": round(float(cam.azimuth), 2),
        "elevation": round(float(cam.elevation), 2),
    }


def _open_in_viewer(path: Path) -> None:
    """Best-effort ``xdg-open``. A failure to *show* a file must not fail the render that produced it."""
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener is None:
        log.warning("--show: no xdg-open/open on PATH; the file is at %s", path)
        return
    try:
        subprocess.Popen(  # noqa: S603 - a fixed opener on a path we just wrote
            [opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError as err:  # pragma: no cover
        log.warning("--show: could not open %s: %s", path, err)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roqsim render", description=__doc__.split("\n")[0])
    parser.add_argument(
        "target",
        nargs="?",
        help="what to render: a world YAML, an MJCF scene (.xml), a model reference "
        "(<pkg>:<name> or a bundled model name), or a raw mesh (.obj/.stl). Optional with --state, "
        "which names its own world; giving one anyway overrides it and is still checked.",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT, help=f"output file (default: {_DEFAULT_OUT})"
    )
    parser.add_argument("--size", default=_DEFAULT_SIZE, help=f"WxH (default: {_DEFAULT_SIZE})")
    # nargs="+" so several keys can share one flag (`--view azimuth=90 distance=30`), which is how a
    # camera is naturally written; repeating the flag works too. A greedy nargs can swallow the
    # positional target, so view_overrides refuses a token with no "=" and says why.
    parser.add_argument(
        "--view",
        action="extend",
        nargs="+",
        metavar="KEY=VALUE",
        help="override the world's sim.view, key by key: lookat (three numbers, comma- or "
        "space-separated), distance, azimuth, elevation, track, follow_heading",
    )
    parser.add_argument(
        "--focus",
        action="extend",
        nargs="+",
        metavar="NAME",
        help="frame on an entity or body, searching for an angle with a clear line of sight; "
        "--view wins for any key it sets",
    )
    parser.add_argument(
        "--no-ceiling",
        action="store_true",
        help="shorthand for --set components.ceiling.keep=false, to look into a roofed world",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="PATH=VALUE",
        help="override a world value, e.g. --set plugins.floorplan.size=4.0 (repeatable)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and report what would be rendered, without rendering it",
    )
    parser.add_argument(
        "--camera",
        metavar="NAME",
        help="render through a fixed MJCF <camera> instead of a free camera; owns its own pose, so it "
        "cannot be combined with --view/--focus",
    )
    recording = parser.add_argument_group(
        "from a recording",
        "Render a run recorded with `roqsim sim --record`. A video extension needs one of these.",
    )
    recording.add_argument(
        "--state", metavar="PATH", help="a recording written by `roqsim sim --record`"
    )
    recording.add_argument(
        "--at",
        type=float,
        metavar="T",
        help="one moment, in simulated seconds; snaps to the nearest sample and reports which "
        "(default with --state: the last sample)",
    )
    recording.add_argument("--from", dest="start", type=float, metavar="T", help="range start (s)")
    recording.add_argument("--to", dest="stop", type=float, metavar="T", help="range end (s)")
    recording.add_argument(
        "--fps",
        metavar="N",
        help="the video's declared rate (default: the recording's own, i.e. 1x sim time). Changes only "
        "the declared rate -- one sample is always exactly one frame.",
    )
    recording.add_argument(
        "--speed", type=float, metavar="X", help="shorthand for --fps = X * the recording's rate"
    )
    parser.add_argument("--show", action="store_true", help="open the result when it is written")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(
        verbose=args.verbose, stream=sys.stderr
    )  # stdout is the JSON record and nothing else

    # `roqsim render --out x.png world.yaml` lets --out swallow nothing, but `roqsim render --view a=1 w.yaml`
    # cannot happen (--view takes exactly one value); the trap is a world passed *to* a value-taking
    # flag, which argparse reports as a missing target. Catch the common shape with a better message.
    if args.target and Path(args.target).suffix.lower() in (".png", ".jpg", ".jpeg", *_VIDEO_EXT):
        parser.error(
            f"{args.target!r} looks like an output file, not something to render; "
            "put the world/model first: roqsim render <target> --out <file>"
        )

    try:
        record = render_target(
            args.target,
            args.out,
            size=args.size,
            view=args.view,
            focus=args.focus,
            camera=args.camera,
            no_ceiling=args.no_ceiling,
            overrides=overrides_from_dotlist(args.overrides),
            check=args.check,
            state=args.state,
            at=args.at,
            start=args.start,
            stop=args.stop,
            fps=args.fps,
            speed=args.speed,
        )
    except RenderError as err:
        print(f"roqsim render: {err}", file=sys.stderr)
        return EXIT_NO_GL if "MUJOCO_GL" in str(err) else EXIT_BAD_ARGS
    except RecordingError as err:
        print(f"roqsim render: {err}", file=sys.stderr)
        return EXIT_PROVENANCE
    except (DisplayError, PluginError, ModelError, CaptureError) as err:
        print(f"roqsim render: {err}", file=sys.stderr)
        return EXIT_BAD_ARGS

    print(json.dumps(record))
    if args.show and record.get("rendered"):
        _open_in_viewer(Path(record["path"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
