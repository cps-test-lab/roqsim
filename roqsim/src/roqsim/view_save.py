"""Save the viewer's live camera back into the world YAML (F8).

Framing a world is done with the eye, not with numbers: you fly the camera (:class:`roqsim.viewer.WalkKeys`)
until the shot is right, and then have to translate that pose into a ``sim.view`` block by hand. This
closes that loop -- F8 asks, and writes what you are looking at into the world it came from.

Three things it deliberately does *not* do:

* **It never reformats the world.** ``yaml.safe_load`` + ``safe_dump`` would round-trip a world into
  canonical YAML and drop every comment in it -- and roqsim's worlds are commented documents (see
  ``roqsim_scenes/.../depot.yaml``). So :func:`replace_sim_view` is a surgical *text* edit of the
  ``sim.view`` block alone, keeping flow style flow and block style block, and
  :func:`_check_only_view_changed` re-parses the result and fails loudly if anything else moved.
* **It never writes without being asked.** :func:`confirm_save` puts the values and the target file in
  front of the person first, because the write replaces a block someone hand-tuned.
* **It never invents a target.** A run started from an MJCF scene or a bare model reference has no
  world YAML, and that is refused with a message rather than by writing a file somewhere plausible.

Threading follows :class:`roqsim.capture.RecordToggle` exactly, and for the same reason: :meth:`SaveViewKey.key_callback`
runs on MuJoCo's UI thread and only sets a flag, while the driver picks it up on the physics thread --
which is where the camera may be read under the viewer's lock and where a modal dialog can block
without stalling the render thread.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


class ViewSaveError(RuntimeError):
    """The world YAML could not be rewritten (no target, unparseable, or the edit changed too much)."""


#: GLFW keycode for F8. Simulate claims F1-F7 (help/info/profiler/sensor/fullscreen/frame/label) and
#: roqsim's recording toggle is on F9, which leaves F8 -- and *only* a function key will do. Every one of
#: the 26 letters is bound by Simulate to a visualization or rendering flag (``V`` is ``mjVIS_TENDON``;
#: see ``mjVISSTRING``/``mjRNDSTRING``), and the passive viewer's key callback runs *in addition to*
#: Simulate's own handling rather than instead of it, so a letter here would toggle rendering on every
#: save. Same constraint that put travel on the arrows in :mod:`roqsim.viewer`.
KEY_F8 = 297

#: Auto-repeat delivers several presses from one held key (~0.2 s apart); above that, below a
#: deliberate double-press. Matches :attr:`roqsim.capture.RecordToggle.DEBOUNCE_S`.
DEBOUNCE_S = 0.4

#: Decimals kept for positions and distances (mm) and for angles (a tenth of a degree). The camera
#: carries far more precision than a framing decision has, and a world YAML is read by people.
_LENGTH_DP = 3
_ANGLE_DP = 1

#: ``sim.view`` keys in the order they are written, so a saved block reads like a hand-authored one.
_VIEW_ORDER = ("track", "follow_heading", "lookat", "distance", "azimuth", "elevation")

_INDENT_STEP = 2


class SaveViewKey:
    """F8 in the viewer: request that the current camera be saved to the world YAML.

    UI thread sets a flag, physics thread acts on it -- see the module docstring. ``chain`` (another
    key callback) is invoked first with the raw keycode, so wiring this in never swallows anyone
    else's keys.
    """

    def __init__(self, chain=None) -> None:
        self._chain = chain
        self._pending = 0
        self._last_accepted = 0.0

    def key_callback(self, keycode: int) -> None:
        """UI thread. Debounce, count, return -- no camera read, no dialog, no file I/O."""
        if self._chain is not None:
            self._chain(keycode)
        if int(keycode) != KEY_F8:
            return
        import time as _time

        now = _time.monotonic()
        if now - self._last_accepted < DEBOUNCE_S:
            return
        self._last_accepted = now
        self._pending += 1

    def take_pending(self) -> bool:
        """Physics thread. Whether a save was asked for, collapsing repeats into one."""
        if not self._pending:
            return False
        self._pending = 0
        return True


def view_from_camera(
    cam, existing: dict | None = None, *, azimuth_offset: float | None = None
) -> dict:
    """The ``sim.view`` block that reproduces ``cam``, carrying the world's own tracking setup over.

    Two keys are not simply read off the camera, because under a tracking view they do not mean what
    they appear to mean (see :class:`roqsim.viewer.TrackingCamera`):

    * ``lookat`` is driven by MuJoCo every frame while tracking, so saving it would freeze a moving
      robot's position into the world. A tracking view keeps ``track`` and drops ``lookat`` -- which is
      also what the loader wants, since it warns that ``lookat`` is ignored beside ``track``.
    * ``azimuth`` under ``follow_heading`` is the robot's yaw plus the configured offset, recomputed
      each frame. The offset is the part a world can state, so it is what gets saved --
      ``azimuth_offset`` (:attr:`roqsim.viewer.TrackingCamera.azimuth_offset`) rather than the live value.
    """
    existing = dict(existing or {})
    tracking = "track" in existing
    view: dict = {}
    if tracking:
        view["track"] = existing["track"]
        if existing.get("follow_heading"):
            view["follow_heading"] = True
    else:
        view["lookat"] = [_round(v, _LENGTH_DP) for v in cam.lookat]
    view["distance"] = _round(cam.distance, _LENGTH_DP)
    live_azimuth = (
        azimuth_offset if view.get("follow_heading") and azimuth_offset is not None else cam.azimuth
    )
    view["azimuth"] = _round(live_azimuth, _ANGLE_DP)
    view["elevation"] = _round(cam.elevation, _ANGLE_DP)
    return view


def _round(value, dp: int) -> float:
    # ``or 0.0`` folds -0.0 (which a camera reaches routinely) onto 0.0, so the written text and the
    # dict compared against it in _check_only_view_changed cannot disagree over a sign nobody meant.
    return round(float(value), dp) or 0.0


def _num(value: float, dp: int) -> str:
    """A float as a world YAML writes it: fixed decimals, trailing zeros dropped, never bare ``4``."""
    fixed = _round(value, dp)  # ``_round`` also folds -0.0, which is not a framing anyone chose
    text = f"{fixed:.{dp}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _scalar(key: str, value) -> str:
    if key == "lookat":
        return "[" + ", ".join(_num(v, _LENGTH_DP) for v in value) + "]"
    if key in ("azimuth", "elevation"):
        return _num(value, _ANGLE_DP)
    if key == "distance":
        return _num(value, _LENGTH_DP)
    if key == "follow_heading":
        return "true" if value else "false"
    # ``track`` -- an entity or body name. Dumped rather than interpolated so a name needing quotes
    # gets them; ``yaml.safe_dump`` appends a document-end marker, hence the first line only.
    return yaml.safe_dump(value, default_flow_style=True).partition("\n")[0].strip()


def format_view(
    view: dict, *, indent: int, flow: bool, comment: str = "", comment_col: int | None = None
) -> list[str]:
    """Render a ``sim.view`` block as YAML lines, in the style it is replacing.

    ``flow`` keeps a one-line ``view: {…}`` on one line (some worlds write it that way on purpose);
    otherwise the keys go one per line. ``comment`` is whatever trailed the original ``view:`` line and
    is put back verbatim -- it usually explains the framing, which the new numbers do not -- at
    ``comment_col`` where it fits, because worlds align their trailing comments in a column and a save
    that broke the column would show up as a diff on lines nobody edited.
    """
    pad = " " * indent
    items = [(k, view[k]) for k in _VIEW_ORDER if k in view]
    if flow:
        body = ", ".join(f"{k}: {_scalar(k, v)}" for k, v in items)
        return [_with_comment(f"{pad}view: {{{body}}}", comment, comment_col)]
    inner = " " * (indent + _INDENT_STEP)
    head = _with_comment(f"{pad}view:", comment, comment_col)
    return [head] + [f"{inner}{k}: {_scalar(k, v)}" for k, v in items]


def _with_comment(line: str, comment: str, column: int | None) -> str:
    if not comment:
        return line
    return line.ljust(max(column or 0, len(line) + 2)) + comment


# -- the surgical edit -------------------------------------------------------------------------


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_content(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _find_key(lines: list[str], key: str, indent: int, start: int, stop: int) -> int | None:
    """Index of the ``key:`` line sitting at exactly ``indent`` within ``[start, stop)``, else None."""
    want = f"{key}:"
    for i in range(start, stop):
        line = lines[i]
        if (
            _is_content(line)
            and _indent_of(line) == indent
            and line.strip().split(" ", 1)[0] == want
        ):
            return i
    return None


def _block_extent(lines: list[str], key_line: int, indent: int) -> int:
    """One past the last line belonging to the block opened at ``key_line`` (whose key is at ``indent``).

    Trailing blanks and comments are left *outside* the block: a comment sitting between the end of
    ``sim:`` and ``plugins:`` introduces what follows it far more often than it closes what precedes,
    so swallowing it into the replaced range would move it.
    """
    end = key_line + 1
    for i in range(key_line + 1, len(lines)):
        if not _is_content(lines[i]):
            continue
        if _indent_of(lines[i]) <= indent:
            break
        end = i + 1
    return end


def _flow_extent(lines: list[str], key_line: int) -> tuple[int, str]:
    """End of a ``view: {…}`` flow mapping (brace-balanced, may wrap) and the comment trailing it."""
    depth = 0
    for i in range(key_line, len(lines)):
        text = lines[i]
        for pos, char in enumerate(text):
            if char == "#" and depth == 0:
                break
            depth += (char == "{") - (char == "}")
            if depth == 0 and char == "}":
                rest = text[pos + 1 :].strip()
                return i + 1, rest if rest.startswith("#") else ""
    raise ViewSaveError(f"sim.view opens a '{{' on line {key_line + 1} that is never closed")


def _flow_scalar(value) -> str:
    """Any value as it would be written inside a flow mapping (``4.0``, ``realtime``, ``[1, 2]``)."""
    return yaml.safe_dump(value, default_flow_style=True).partition("\n")[0].strip()


def _flow_value(lines: list[str], key_line: int, key: str):
    """Parse the flow mapping that is ``key``'s value, spanning from ``key_line``."""
    end, comment = _flow_extent(lines, key_line)
    text = "\n".join(lines[key_line:end])
    if comment:
        text = text[: text.rindex(comment)]
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get(key), dict):
        raise ViewSaveError(f"{key!r} on line {key_line + 1} is not a mapping")
    return parsed[key], end, comment


def _item_indent(lines: list[str], block_line: int, block_end: int, fallback: int) -> int:
    for i in range(block_line + 1, block_end):
        if _is_content(lines[i]):
            return _indent_of(lines[i])
    return fallback


def replace_sim_view(text: str, view: dict) -> str:
    """Return ``text`` with its ``sim.view`` block replaced by ``view``, and nothing else touched.

    Handles the four shapes a world can be in: a block-style ``view:``, a one-line ``view: {…}``, a
    ``sim:`` block with no ``view`` (appended), and no ``sim:`` block at all (one is created above
    ``plugins:``). The result is re-parsed and compared against the original before it is returned, so
    a mis-sliced range is an error rather than a quietly mangled world.
    """
    lines = text.splitlines()
    sim_line = _find_key(lines, "sim", 0, 0, len(lines))

    if sim_line is None:
        indent = _INDENT_STEP
        block = ["sim:", *format_view(view, indent=indent, flow=False), ""]
        at = _find_key(lines, "plugins", 0, 0, len(lines))
        at = len(lines) if at is None else at
        new_lines = lines[:at] + block + lines[at:]
    elif lines[sim_line].partition("sim:")[2].strip().startswith("{"):
        # ``sim: {pacing: asap}`` -- the whole block is one flow mapping, so there is no line range to
        # splice into. Its keys are re-emitted (they carry no comments of their own, being on one line)
        # with ``view`` set; the trailing comment, which does, is put back.
        sim, end, comment = _flow_value(lines, sim_line, "sim")
        sim["view"] = view
        body = ", ".join(
            format_view(view, indent=0, flow=True)[0] if k == "view" else f"{k}: {_flow_scalar(v)}"
            for k, v in sim.items()
        )
        tail = f"  {comment}" if comment else ""
        new_lines = lines[:sim_line] + [f"sim: {{{body}}}{tail}"] + lines[end:]
    else:
        sim_end = _block_extent(lines, sim_line, 0)
        indent = _item_indent(lines, sim_line, sim_end, _INDENT_STEP)
        view_line = _find_key(lines, "view", indent, sim_line + 1, sim_end)
        if view_line is None:
            block = format_view(view, indent=indent, flow=False)
            new_lines = lines[:sim_end] + block + lines[sim_end:]
        else:
            original = lines[view_line]
            head, _, rest = original.partition("view:")
            rest = rest.strip()
            if rest.startswith("{"):
                end, comment = _flow_extent(lines, view_line)
                flow = True
            elif rest.startswith("#") or not rest:
                end, comment = _block_extent(lines, view_line, indent), rest
                flow = False
            else:
                raise ViewSaveError(
                    f"sim.view on line {view_line + 1} is neither a block nor a flow mapping: {rest!r}"
                )
            # Where the trailing comment sat, so a re-emitted line keeps the file's comment column.
            column = original.rindex(comment) if comment and comment in original else None
            block = format_view(
                view, indent=len(head), flow=flow, comment=comment, comment_col=column
            )
            new_lines = lines[:view_line] + block + lines[end:]

    out = "\n".join(new_lines)
    if text.endswith("\n"):
        out += "\n"
    _check_only_view_changed(text, out, view)
    return out


def _check_only_view_changed(before: str, after: str, view: dict) -> None:
    """Fail loudly unless ``after`` parses to ``before`` with ``sim.view`` set to ``view``.

    A text edit that slices the wrong range still produces valid YAML surprisingly often -- it just
    silently loses a plugin or reparents a key. This is what makes that a raised error instead of a
    world that loads and runs the wrong experiment.
    """
    try:
        parsed = yaml.safe_load(after) or {}
    except yaml.YAMLError as err:
        raise ViewSaveError(f"the rewritten world is not valid YAML: {err}") from err
    original = yaml.safe_load(before) or {}
    if not isinstance(original, dict):
        raise ViewSaveError("a world YAML must be a mapping at the top level")
    expected = {**original, "sim": {**(original.get("sim") or {}), "view": view}}
    if parsed != expected:
        raise ViewSaveError(
            "refusing to write: the edit would have changed more than sim.view "
            "(this is a bug in replace_sim_view, not in the world)"
        )


def save_view(path: Path, view: dict) -> None:
    """Write ``view`` into ``path``'s ``sim.view``, atomically.

    Replace-by-rename rather than truncate-and-write, because the alternative failure mode is an
    emptied world file -- and the file being overwritten is hand-authored research input.
    """
    path = Path(path)
    text = path.read_text()
    out = replace_sim_view(text, view)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(out)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# -- the dialog --------------------------------------------------------------------------------


def describe(path: Path, view: dict) -> str:
    """The human-readable summary the dialog shows: what will be written, and where."""
    keys = [k for k in _VIEW_ORDER if k in view]
    width = max(len(k) for k in keys) + 1
    body = "\n".join(f"    {k + ':':<{width}}  {_scalar(k, view[k])}" for k in keys)
    return (
        f"Save the current camera view to\n\n    {path}\n\n{body}\n\n"
        "This replaces the world's sim.view block."
    )


def confirm_save(path: Path, view: dict) -> bool:
    """Ask, modally, whether to write ``view`` to ``path``. ``False`` means the person said no.

    tkinter is created and torn down per dialog, on the caller's thread. That thread is the physics
    thread, which in ``roqsim sim`` is the process's main thread -- the condition tkinter actually
    requires, and the reason ``roqsim_scene_builder`` has to use a subprocess (its caller's main thread
    belongs to an MCP server) while this does not. The simulation is frozen for as long as the dialog
    is up, which is correct: the pose being saved must not move under the question. MuJoCo's window
    keeps rendering meanwhile, on its own thread.
    """
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)  # else it can open behind the viewer window
        return bool(
            messagebox.askokcancel("roqsim — save camera view", describe(path, view), parent=root)
        )
    finally:
        root.destroy()


def report_error(message: str) -> None:
    """Put a failure in front of the person who pressed F8, falling back to the log."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        try:
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror("roqsim — save camera view", message, parent=root)
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001 — no display for the dialog is not a reason to lose the message
        log.error("%s", message)


def save_current_view(
    cam,
    world_yaml: Path | None,
    existing: dict | None = None,
    *,
    azimuth_offset: float | None = None,
    logger: logging.Logger | None = None,
    ask=confirm_save,
) -> bool:
    """The whole F8 action: derive the block, ask, write. ``True`` when a world was rewritten.

    Never raises -- a failed save must not take the run down with it. Every refusal is reported
    where the person is looking (:func:`report_error`) rather than only to the log, because F8 is
    pressed in a window and a message on a terminal behind it is a message nobody reads.
    """
    logger = logger or log
    if cam is None:
        report_error("The viewer window is closing; the camera view was not saved.")
        return False
    if world_yaml is None:
        report_error(
            "This run has no world YAML to save into.\n\n"
            "F8 writes the camera into the sim.view block of the world it is running, and this run "
            "was started from an MJCF scene or a model reference. Run a world YAML to use it."
        )
        return False
    view = view_from_camera(cam, existing, azimuth_offset=azimuth_offset)
    try:
        if not ask(Path(world_yaml), view):
            logger.info("view save: cancelled")
            return False
        save_view(Path(world_yaml), view)
    except (ViewSaveError, OSError) as err:
        report_error(f"The camera view was not saved.\n\n{err}")
        logger.error("view save failed: %s", err)
        return False
    logger.info("view save: sim.view written to %s", world_yaml)
    return True
