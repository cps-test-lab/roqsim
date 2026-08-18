# SPDX-License-Identifier: Apache-2.0
"""Render a floorplan to a PNG plan view, with a metre scale bar.

The *look at it* tool for the floorplan side of the scene pipeline, next to
:mod:`roqsim_scenes.cad_to_png` (which previews the CAD drawing a floorplan is converted *from*). It draws
what a floorplan JSON says -- walls with their door/window openings cut out, rooms filled and named,
markers -- as a top-down architectural plan an agent or a human can look at in one glance, instead of
reading 40 wall segments as numbers.

It is a *general* tool: the input can be the floorplan JSON, the scene directory that references one, a
world YAML, or a ``<package>:<world>`` ref, and the geometry comes from
:mod:`roqsim_scenes.floorplan_geometry` -- the same wall/opening arithmetic :mod:`roqsim_scenes.cli.floorplan_to_world`
bakes, so the drawing cannot disagree with the world it previews. Nothing about any particular building
is known here.

Usage::

    roqsim-floorplan-to-png floorplan.json -o plan.png            # a floorplan JSON
    roqsim-floorplan-to-png path/to/scenes/<name>                 # a scene dir authored from one
    roqsim-floorplan-to-png path/to/world.yaml                    # a world (follows `extends:`)
    roqsim-floorplan-to-png floorplan.json --scale-bar bottom-left --scale-bar-length 5
    roqsim-floorplan-to-png floorplan.json --scale-bar 12,3.5    # the bar at an explicit metre position
    roqsim-floorplan-to-png floorplan.json --ids --grid 1 --axes  # for editing/placement work
    roqsim-floorplan-to-png floorplan.json --font-scale 2.6       # for a slide
    roqsim-floorplan-to-png floorplan.json --highlight "hall" --area-decimals 0 --no-legend

**The scale bar is the point of the metre annotation**: a plan without one is unusable for judging
whether a 0.9 m door or a 2 m robot fits, and a PNG has no units. It is positionable (a corner, or an
explicit ``x,y`` in world metres) because the empty part of a plan differs per building; a corner
placement sits flush with the plan's own edge (see :func:`corner_anchor`) rather than floating in the
canvas beside the drawing.

``--font-scale`` (or ``--font-size``) is for a slide or a poster: it scales the labels *and* the
point-width strokes, since the walls grow with the plan and hairlines do not.

What the plan cannot invent: a floorplan opening does not say whether it is a door, a bare doorway or a
window -- that lives in the generator's ``--doors-map``. When that map is found (or given) the openings
are drawn as what they will become; without it, an opening with its own ``height_m`` is drawn as a
neutral "opening" rather than being guessed into a window. A door is drawn the way a technical drawing
draws it -- leaf plus quarter-circle swing arc -- hinged and swinging the way the world hangs it, never a
guessed side.

``--highlight`` / ``--room-color`` fill named rooms with an accent instead of the pale wash, while
``--area-only`` reduces a room's label to its m² and ``--no-label`` drops it entirely, for a plan that
argues about one area and says nothing else. A key
matching no room is a hard error: a plan with nothing highlighted looks just as finished as a correct one.
Labels fit themselves to their room -- wrapped over lines before being shrunk, and turned upright only in
a room too narrow to wrap into.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from roqsim_scenes.floorplan_geometry import (
    Opening,
    WallPlan,
    bounds,
    label_spot,
    plan_walls,
    room_polygons,
)

# Ink. A plan is read for its geometry, so the palette stays quiet: dark walls on paper, rooms a pale
# wash, and colour reserved for the things a reader has to tell apart (opening kinds, markers).
_INK = {
    "wall": "#33383d",
    "wall_edge": "#15181b",
    "room_fill": "#eef2f7",
    "highlight": "#fbe3bd",  # the default --highlight accent
    "room_edge": "#c8d3de",
    "room_text": "#2f3b47",
    "door": "#b5822f",
    "doorway": "#8a949d",
    "window": "#2f7d9e",
    "opening": "#6f7a84",
    "marker": "#b8382c",
    "grid": "#e6eaee",
    "ids": "#7b858e",
    "scale": "#15181b",
}

# Text sizes in points, and the point-width strokes that belong with them, at ``--font-scale 1`` (a
# plan read on a screen). Both scale together: the walls are metres wide and grow with the plan, but the
# annotation ink -- jambs, glazing, room outlines, the bar frame -- is in points, so text alone at 3x for
# a projector would sit next to hairlines and read as a broken drawing.
_FONT_PT = {
    "title": 10.0,
    "room": 7.5,
    "marker": 6.0,
    "ids": 5.5,
    "bar_tick": 7.5,
    "bar_label": 7.0,
    "legend": 6.5,
    "axis_label": 7.0,
    "axis_tick": 6.0,
}
_LW_PT = {
    "wall_edge": 0.4,
    "room_edge": 0.5,
    "jamb": 0.6,
    "door": 0.6,
    "doorway": 0.7,
    "opening": 0.7,
    "window": 1.1,
    "arc": 0.5,
    "grid": 0.4,
    "bar": 0.7,
    "legend_line": 1.4,
}
_MARKER_DOT_PT = 3.2

# Glyph metrics of the default sans face, in units of the font size: mean advance per character and
# line height. Everything that has to guess how much space a label needs measures with these.
_CHAR_EM = 0.55
_LINE_EM = 1.35

# Opening kinds, in the order they appear in the legend, with the label a reader sees.
_KIND_LABELS = {
    "door": "door (leaf + swing)",
    "doorway": "cased doorway",
    "window": "window / glazing",
    "opening": "opening (kind unknown)",
}

_CORNERS = {
    "bottom-left": ("left", "bottom"),
    "bottom-right": ("right", "bottom"),
    "top-left": ("left", "top"),
    "top-right": ("right", "top"),
}

# Legend placement -> matplotlib (loc, bbox_to_anchor in axes fraction). `below` sits OUTSIDE the plan
# and is the default: a legend inside a corner covers whatever the building has there, and unlike the
# scale bar its size is not known before it is drawn, so its space cannot be reserved up front.
_LEGEND_LOC = {
    "below": ("upper center", (0.5, -0.01)),
    "bottom-left": ("lower left", None),
    "bottom-right": ("lower right", None),
    "top-left": ("upper left", None),
    "top-right": ("upper right", None),
}

_NICE_SCALE_STEPS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)

_DOORS_MAP_NAME = "doors_map.json"  # the generator's map, conventionally beside floorplan.json


@dataclass(frozen=True)
class Style:
    """Text and stroke weights, all derived from one ``scale`` (1 = screen, ~2-3 = presentation)."""

    scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.1 <= self.scale <= 12.0:
            raise ValueError(f"font scale {self.scale:g} is outside the usable range 0.1 .. 12")

    def pt(self, key: str) -> float:
        return _FONT_PT[key] * self.scale

    def lw(self, key: str) -> float:
        return _LW_PT[key] * self.scale

    @classmethod
    def from_font_size(cls, room_label_pt: float) -> Style:
        """The style whose *room label* is ``room_label_pt`` points -- everything else keeps its ratio."""
        return cls(scale=room_label_pt / _FONT_PT["room"])


def _require_matplotlib():
    """Import matplotlib (Agg), turning the ImportError into an actionable message."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # a plan is written to a file; never touch a display
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon, Rectangle
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "rendering a floorplan needs the optional preview dependencies: "
            "pip install 'roqsim_scenes[preview]'  (matplotlib)"
        ) from exc
    return plt, Polygon, Rectangle, Line2D


# --------------------------------------------------------------------------- input


@dataclass
class FloorplanSource:
    """A loaded floorplan and where it came from -- so the PNG can name its own source."""

    floorplan: dict
    path: Path  # the floorplan JSON itself
    name: str  # scene / file name, used for the default title and output filename
    doors_map: dict = field(default_factory=dict)
    doors_map_path: Path | None = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_floorplan(obj) -> bool:
    return isinstance(obj, dict) and "lines" in obj and "rooms" in obj


def floorplan_of_scene(scene_dir: Path) -> Path:
    """The floorplan JSON path a scene directory references (``scene.json``'s ``floorplan`` field).

    Same reference-following contract as :mod:`roqsim_scenes.cli.scene_to_floorplan`: a scene that was not authored
    from a floorplan cannot be drawn as one, and that is an error rather than a reconstruction attempt
    (the baked walls alone have lost the rooms, the door semantics and which line was one wall).
    """
    manifest_path = scene_dir / "scene.json"
    if not manifest_path.exists():
        raise ValueError(f"{scene_dir} has no scene.json; it is not a roqsim scene directory.")
    ref = _read_json(manifest_path).get("floorplan")
    if not isinstance(ref, str):
        raise ValueError(
            f"{manifest_path} has no 'floorplan' reference; the scene was not authored from a "
            f"floorplan, so it has no plan to draw. Pass a floorplan JSON directly if you have one."
        )
    path = scene_dir / ref
    if not path.exists():
        raise ValueError(
            f"{manifest_path} references floorplan '{ref}', but {path} does not exist."
        )
    return path


def _scene_dirs_for_world(world_yaml: Path) -> list[Path]:
    """Candidate scene dirs for a world YAML: ``<pkg>/scenes/<name>/`` for each name the world implies.

    A world does not point at its scene directory (it points at the *baked* MJCF), so the link is the
    package layout every scene package shares: ``worlds/<name>.yaml`` beside ``scenes/<name>/``. Both
    the YAML stem and the directory of ``sim.world`` are tried, because a world may be named differently
    from the scene it bakes.
    """
    import yaml

    doc = yaml.safe_load(world_yaml.read_text(encoding="utf-8")) or {}
    names = [world_yaml.stem]
    world_ref = ((doc.get("sim") or {}).get("world")) if isinstance(doc, dict) else None
    if isinstance(world_ref, str) and "/" in world_ref:
        names.append(Path(world_ref).parent.name)
    scenes = world_yaml.parent.parent / "scenes"
    out, seen = [], set()
    for name in names:
        cand = scenes / name
        if name not in seen:
            seen.add(name)
            out.append(cand)
    return out


def _extends_ref(world_yaml: Path) -> str | None:
    import yaml

    doc = yaml.safe_load(world_yaml.read_text(encoding="utf-8")) or {}
    ref = doc.get("extends") if isinstance(doc, dict) else None
    return ref if isinstance(ref, str) else None


def _world_yaml_of_ref(ref: str) -> Path | None:
    """``<package>:<world>`` -> its world YAML, via roqsim's world providers; ``None`` if unresolvable."""
    try:
        from roqsim.world import resolve_world_yaml_ref
    except ImportError:  # pragma: no cover - roqsim is a hard dependency of this package
        return None
    resolved = resolve_world_yaml_ref(ref)
    return Path(resolved) if resolved else None


def _floorplan_from_world(world_yaml: Path, _depth: int = 0) -> Path:
    """The floorplan of the scene a world YAML builds on, following ``extends:`` if it has none itself."""
    tried: list[Path] = []
    for scene_dir in _scene_dirs_for_world(world_yaml):
        tried.append(scene_dir)
        if (scene_dir / "scene.json").exists():
            return floorplan_of_scene(scene_dir)
    # A populated world usually carries no scene of its own -- its geometry is the base world it extends.
    ref = _extends_ref(world_yaml) if _depth < 4 else None
    if ref:
        base = Path(ref) if Path(ref).exists() else _world_yaml_of_ref(ref)
        if base is None:
            raise ValueError(
                f"{world_yaml} has no scene of its own and its 'extends: {ref}' does not resolve to a "
                f"world YAML; pass the floorplan JSON or the scene dir directly."
            )
        return _floorplan_from_world(base, _depth + 1)
    raise ValueError(
        f"{world_yaml} is not floorplan-authored: no scene.json found in "
        f"{', '.join(str(p) for p in tried)} and no 'extends:' to follow. A hand-written MJCF world has "
        f"no floorplan to draw -- pass a floorplan JSON or a floorplan-authored scene dir."
    )


def load_source(source: str, doors_map: str | None = None, use_doors_map: bool = True):
    """Resolve ``source`` to a :class:`FloorplanSource`.

    ``source`` is a floorplan JSON, a scene directory (its ``scene.json`` references the floorplan), a
    world YAML, or a ``<package>:<world>`` ref. Every failure is
    loud and names what was tried -- silently drawing a *different* floorplan than the caller meant is
    the one outcome worse than no drawing.
    """
    path = Path(source)
    if path.is_dir():
        fp_path = path / "floorplan.json"
        if not fp_path.exists():
            fp_path = floorplan_of_scene(path)
    elif path.is_file():
        if path.suffix.lower() in (".yaml", ".yml"):
            fp_path = _floorplan_from_world(path)
        elif path.suffix.lower() == ".json":
            obj = _read_json(path)
            if _is_floorplan(obj):
                fp_path = path
            elif "floorplan" in obj or "objects" in obj:  # a scene.json
                fp_path = floorplan_of_scene(path.parent)
            else:
                raise ValueError(
                    f"{path} is neither a floorplan JSON (needs 'lines' and 'rooms') nor a scene.json"
                )
        else:
            raise ValueError(f"{path}: expected a floorplan .json, a scene dir, or a world .yaml")
    elif ":" in source and (world := _world_yaml_of_ref(source)) is not None:
        fp_path = _floorplan_from_world(world)
    else:
        raise ValueError(
            f"{source!r} is not a file, a directory, or a resolvable '<package>:<world>' ref"
        )

    floorplan = _read_json(fp_path)
    if not _is_floorplan(floorplan):
        raise ValueError(f"{fp_path} is not a floorplan JSON (needs 'lines' and 'rooms')")

    # The scene dir's name is the building's name; a loose floorplan.json falls back to its own stem.
    name = fp_path.parent.name if fp_path.name == "floorplan.json" else fp_path.stem

    dm_path: Path | None = None
    if doors_map:
        dm_path = Path(doors_map)
        if not dm_path.exists():
            raise ValueError(f"--doors-map {dm_path} does not exist")
    elif use_doors_map and (cand := fp_path.parent / _DOORS_MAP_NAME).exists():
        dm_path = cand
    dm = {str(k): v for k, v in (_read_json(dm_path).items() if dm_path else [])}
    return FloorplanSource(
        floorplan=floorplan, path=fp_path, name=name, doors_map=dm, doors_map_path=dm_path
    )


# --------------------------------------------------------------------------- classification


def opening_kind(op: Opening, doors_map: dict, opening_h: float, ceiling_h: float = 2.5) -> str:
    """What an opening will be filled with: ``door`` | ``doorway`` | ``window`` | ``opening``.

    The floorplan itself only cuts holes; what fills one is the generator's ``--doors-map``
    (``skip: true`` -> a window plugin or a bare gap fills it, ``leaf: false`` -> a cased doorway).
    Without that map, an opening that sets its own ``height_m`` is drawn as an unknown ``opening``:
    calling it a window would be a guess, and a wrong door in a plan misreads as a wrong world.

    A **full-height** opening never gets a leaf, whatever the map says -- ``door_placements`` reports it
    and leaves a bare gate -- so the plan must not swing a door through one either.
    """
    entry = doors_map.get(str(op.door_id)) if op.door_id is not None else None
    if isinstance(entry, dict) and entry.get("skip"):
        return "window"
    if op.height_m >= ceiling_h:
        return "opening"
    if isinstance(entry, dict):
        return "doorway" if entry.get("leaf") is False else "door"
    if not math.isclose(op.height_m, opening_h, rel_tol=1e-6):
        return "opening"
    return "door"


# --------------------------------------------------------------------------- scale bar


def nice_scale_length(span_m: float) -> float:
    """A round bar length (m) covering roughly a sixth of the plan's width -- 1, 2, 5, 10, 20 ..."""
    target = span_m / 6.0
    for step in _NICE_SCALE_STEPS:
        if step >= target:
            return float(step)
    return float(_NICE_SCALE_STEPS[-1])


def parse_scale_bar_pos(text: str) -> tuple[str, tuple[float, float] | None]:
    """``bottom-left`` / ``none`` / ``"12,3.5"`` -> ``(kind, xy)``; ``kind`` is a corner, ``xy`` or off."""
    key = text.strip().lower()
    if key in ("none", "off"):
        return ("none", None)
    if key in _CORNERS:
        return (key, None)
    parts = key.replace(";", ",").split(",")
    if len(parts) == 2:
        try:
            return ("xy", (float(parts[0]), float(parts[1])))
        except ValueError:
            pass
    raise ValueError(
        f"scale bar position {text!r}: expected one of {', '.join(_CORNERS)}, 'none', or 'X,Y' in metres"
    )


def corner_anchor(
    kind: str, box: tuple[float, float, float, float]
) -> tuple[tuple[float, float], str, str]:
    """Where a corner-placed scale bar sits: its block **flush with the plan's own edge**.

    ``box`` is the building's extent. A bottom corner puts the bottom of the bar's block (bar + labels)
    exactly on the plan's lowest edge, so the bar reads as belonging to the drawing rather than floating
    in the canvas below it -- and it lands in the empty corner a plan usually has there rather than under
    the whole figure. Use an explicit ``X,Y`` when a building fills that corner.
    """
    xmin, ymin, xmax, ymax = box
    ha, va = _CORNERS[kind]
    return (xmin if ha == "left" else xmax, ymin if va == "bottom" else ymax), ha, va


_BAR_HEIGHT_FRACTION = 0.006  # bar thickness, as a fraction of the plan's width
_BAR_LABEL_FRACTION = 0.014  # the band its labels need, same units, per unit of font scale


def _scale_bar_block(
    length_m: float, span_m: float, style: Style, height_m: float | None = None
) -> tuple[float, float]:
    """``(bar height, total block height)`` in metres for a ``length_m`` bar on a ``span_m`` wide plan.

    Two terms, because they scale differently: the *bar* is a metre ruler and stays a slim band (growing
    only with the square root of the font scale, so a slide-size plan does not get a fat black slab), while
    the space under it for "0 / 2.5 / 5 m" tracks the font size linearly -- otherwise a presentation-size
    label would not fit beneath the bar it belongs to. ``height_m`` overrides the bar's own thickness.
    """
    bar_h = (
        height_m if height_m else max(_BAR_HEIGHT_FRACTION * span_m, 0.04) * math.sqrt(style.scale)
    )
    labels_h = max(_BAR_LABEL_FRACTION * span_m, 0.07) * style.scale
    return bar_h, bar_h + labels_h


_TICK_GAP = 0.85  # neighbouring bar labels may fill this much of the space between their ticks


def _tick_fits(mid: str, end: str, length_m: float, style: Style, pt_per_m: float | None) -> bool:
    """Is there room for the bar's midpoint label between the two end labels?"""
    if not pt_per_m:
        return True  # no canvas scale given: keep the tick (the caller knows its own layout)

    def width_m(text: str) -> float:
        return len(text) * _CHAR_EM * style.pt("bar_tick") / pt_per_m

    return (width_m(mid) + width_m(end)) / 2 < (length_m / 2) * _TICK_GAP


def draw_scale_bar(
    ax,
    Rectangle,
    *,
    length_m,
    span_m,
    anchor,
    ha,
    va,
    style=Style(),
    height_m=None,
    segments=2,
    label=None,
    pt_per_m=None,
):
    """Draw an alternating-segment scale bar with its metre labels, anchored at ``anchor`` (data coords).

    ``ha``/``va`` say which corner of the bar block ``anchor`` is, so the same routine serves a corner
    placement and an explicit ``x,y``. ``pt_per_m`` (the plan's scale on the canvas) lets the midpoint
    tick drop out when the type has grown too big for it -- at slide size "2.5" and "5 m" would otherwise
    run into each other and read as one number.
    """
    bar_h, block_h = _scale_bar_block(length_m, span_m, style, height_m)
    x0 = anchor[0] - length_m if ha == "right" else anchor[0]
    y_bar = anchor[1] + (block_h - bar_h) if va == "bottom" else anchor[1] - bar_h
    pad = bar_h * 0.6

    ax.add_patch(
        Rectangle(
            (x0 - pad, y_bar - (block_h - bar_h) - pad * 0.4),
            length_m + 2 * pad,
            block_h + pad,
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            zorder=10,
        )
    )
    seg = length_m / segments
    for i in range(segments):
        ax.add_patch(
            Rectangle(
                (x0 + i * seg, y_bar),
                seg,
                bar_h,
                facecolor=_INK["scale"] if i % 2 == 0 else "white",
                edgecolor=_INK["scale"],
                linewidth=style.lw("bar"),
                zorder=11,
            )
        )
    ticks = [(0.0, "0"), (length_m, f"{length_m:g} m")]
    mid = f"{length_m / 2:g}"
    if segments > 1 and _tick_fits(mid, ticks[1][1], length_m, style, pt_per_m):
        ticks.insert(1, (length_m / 2, mid))
    for t, text in ticks:
        ax.text(
            x0 + t,
            y_bar - (block_h - bar_h) * 0.18,
            text,
            ha="center",
            va="top",
            fontsize=style.pt("bar_tick"),
            color=_INK["scale"],
            zorder=11,
        )
    if label:
        ax.text(
            x0 + length_m / 2,
            y_bar + bar_h * 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=style.pt("bar_label"),
            color=_INK["scale"],
            zorder=11,
        )


# --------------------------------------------------------------------------- drawing


def _wall_quad(wall: WallPlan, t0: float, t1: float, thickness: float):
    """The four corners of the slab spanning ``t0..t1`` along ``wall`` -- centred on the drawn line."""
    ux, uy = wall.direction
    nx, ny = -uy * thickness / 2, ux * thickness / 2
    a, b = wall.point_at(t0), wall.point_at(t1)
    return [
        (a[0] + nx, a[1] + ny),
        (b[0] + nx, b[1] + ny),
        (b[0] - nx, b[1] - ny),
        (a[0] - nx, a[1] - ny),
    ]


def _draw_walls(ax, Polygon, walls, thickness, ids: bool, style: Style):
    for wall in walls:
        for t0, t1 in wall.solid:
            ax.add_patch(
                Polygon(
                    _wall_quad(wall, t0, t1, thickness),
                    closed=True,
                    facecolor=_INK["wall"],
                    edgecolor=_INK["wall_edge"],
                    linewidth=style.lw("wall_edge"),
                    zorder=3,
                )
            )
        if ids:
            ux, uy = wall.direction
            mx, my = wall.point_at(wall.length / 2)
            off = thickness * 3.5
            ax.text(
                mx - uy * off,
                my + ux * off,
                str(wall.line_id),
                ha="center",
                va="center",
                fontsize=style.pt("ids"),
                color=_INK["ids"],
                zorder=7,
            )


_DOOR_SWING_DEG = 90.0  # a plan draws the leaf at 90 deg by convention
_LEAF_THICKNESS_FRACTION = 0.45  # leaf panel thickness, as a fraction of the wall's
_ARC_SEGMENTS = 24


def _leaf_geometry(wall: WallPlan, op: Opening, entry: dict, swing_deg: float):
    """``(hinge, leaf_tip, arc_points)`` for a door leaf swung ``swing_deg`` open, in world metres.

    Mirrors how the world actually hangs the leaf -- ``floorplan_to_world.door_placements`` places the
    ``door`` plugin at the opening centre with the wall direction as the leaf's local +x, and that pair's
    defaults are ``hinge_side: left`` (hinge at the opening's first end, leaf reaching along the wall) and
    ``swing: 1``. The leaf sweeps toward ``leaf_dir * swing`` along the wall normal, so the arc shows the
    side the door really opens to, not a guessed one. The *angle* is the drafting convention rather than
    the door's initial ``open`` fraction: a plan is read for which way a door opens and how much floor it
    needs, not for the state the simulation happens to start in.
    """
    ux, uy = wall.direction
    nx, ny = -uy, ux
    width = op.width_m
    leaf_dir = -1.0 if str(entry.get("hinge_side", "left")).lower() == "right" else 1.0
    swing = 1.0 if float(entry.get("swing", 1) or 1) >= 0 else -1.0
    open_side = leaf_dir * swing

    hinge = wall.point_at(op.t_m - leaf_dir * width / 2)
    start = math.atan2(leaf_dir * uy, leaf_dir * ux)  # the closed leaf, lying in its opening
    sweep = math.radians(swing_deg) * swing
    arc = [
        (
            hinge[0] + width * math.cos(start + sweep * k / _ARC_SEGMENTS),
            hinge[1] + width * math.sin(start + sweep * k / _ARC_SEGMENTS),
        )
        for k in range(_ARC_SEGMENTS + 1)
    ]
    if abs(swing_deg - 90.0) < 1e-9:  # the exact 90 deg case, free of trig drift
        tip = (hinge[0] + open_side * nx * width, hinge[1] + open_side * ny * width)
    else:
        tip = arc[-1]
    return hinge, tip, arc


def _draw_leaf(ax, Polygon, Line2D, wall, op, thickness, entry, swing_deg, style: Style):
    """A door as a technical drawing: the leaf panel plus its quarter-circle swing arc."""
    hinge, tip, arc = _leaf_geometry(wall, op, entry, swing_deg)
    dx, dy = tip[0] - hinge[0], tip[1] - hinge[1]
    span = math.hypot(dx, dy) or 1.0
    half = max(thickness * _LEAF_THICKNESS_FRACTION, 0.02) / 2
    px, py = -dy / span * half, dx / span * half  # leaf half-thickness, across the leaf
    ax.add_patch(
        Polygon(
            [
                (hinge[0] + px, hinge[1] + py),
                (tip[0] + px, tip[1] + py),
                (tip[0] - px, tip[1] - py),
                (hinge[0] - px, hinge[1] - py),
            ],
            closed=True,
            facecolor="white",
            edgecolor=_INK["door"],
            linewidth=style.lw("door"),
            zorder=5,
        )
    )
    if swing_deg:
        ax.add_line(
            Line2D(
                [p[0] for p in arc],
                [p[1] for p in arc],
                color=_INK["door"],
                linewidth=style.lw("arc"),
                zorder=4,
            )
        )


def _draw_openings(
    ax,
    Polygon,
    Line2D,
    walls,
    thickness,
    doors_map,
    opening_h,
    ids: bool,
    style: Style,
    ceiling_h: float = 2.5,
    swing_deg: float = _DOOR_SWING_DEG,
):
    """Draw each opening in the style of what fills it. Returns the kinds actually drawn."""
    kinds: list[str] = []
    for wall in walls:
        ux, uy = wall.direction
        nx, ny = -uy * thickness / 2, ux * thickness / 2
        for op in wall.openings:
            kind = opening_kind(op, doors_map, opening_h, ceiling_h)
            if kind not in kinds:
                kinds.append(kind)
            t0, t1 = op.span
            quad = _wall_quad(wall, t0, t1, thickness)
            if kind == "door":
                # The gap stays empty (as in a drafted plan) -- the leaf and its arc say it is a door.
                entry = doors_map.get(str(op.door_id)) if op.door_id is not None else None
                _draw_leaf(
                    ax,
                    Polygon,
                    Line2D,
                    wall,
                    op,
                    thickness,
                    entry if isinstance(entry, dict) else {},
                    swing_deg,
                    style,
                )
            elif kind == "window":
                for s in (-0.45, 0.45):  # a glazing line each side of the wall centre
                    a = wall.point_at(t0)
                    b = wall.point_at(t1)
                    ax.add_line(
                        Line2D(
                            [a[0] + nx * 2 * s, b[0] + nx * 2 * s],
                            [a[1] + ny * 2 * s, b[1] + ny * 2 * s],
                            color=_INK["window"],
                            linewidth=style.lw("window"),
                            zorder=4,
                        )
                    )
            else:  # doorway / unknown opening: an outline, nothing filled in
                ax.add_patch(
                    Polygon(
                        quad,
                        closed=True,
                        facecolor="none",
                        edgecolor=_INK[kind],
                        linewidth=style.lw(kind),
                        linestyle=(0, (2.5, 1.5)) if kind == "doorway" else (0, (1, 1.5)),
                        zorder=4,
                    )
                )
            for t in (t0, t1):  # jambs: the reveal on each side of the hole
                p = wall.point_at(t)
                ax.add_line(
                    Line2D(
                        [p[0] + nx, p[0] - nx],
                        [p[1] + ny, p[1] - ny],
                        color=_INK["wall_edge"],
                        linewidth=style.lw("jamb"),
                        zorder=5,
                    )
                )
            if ids and op.door_id is not None:
                c = wall.point_at(op.t_m)
                off = thickness * 4.5
                ax.text(
                    c[0] - uy * off,
                    c[1] + ux * off,
                    f"d{op.door_id}",
                    ha="center",
                    va="center",
                    fontsize=style.pt("ids"),
                    color=_INK["door"],
                    zorder=7,
                )
    return kinds


_LABEL_FILL = 0.85  # a label uses this much of the space it has; the rest keeps it off the walls
_MIN_LABEL_FRACTION = 0.45  # a label shrinks to at most this much of the plan's label size
_ROTATE_GAIN = 1.35  # turn a label upright only if that beats the best wrapping by this much


def _label_sizes(
    text: str, polygon: list[tuple[float, float]], clearance_m: float, pt_per_m: float
) -> tuple[float, float]:
    """``(lying along the room, standing upright)``: the point size this exact text can take in the room.

    Uncapped -- the caller decides how big a label may get and how small it may shrink. The fit is
    estimated from glyph advances (no render-measure-render loop) against the room's bounding box, capped
    by the clearance at the label point so an L-shaped room is not measured by its notch.
    """
    lines = text.splitlines() or [""]
    longest = max(max(len(line) for line in lines), 1)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    width, height = max(xs) - min(xs), max(ys) - min(ys)

    def fits(along_m: float, across_m: float) -> float:
        along = min(along_m, 5 * clearance_m) * _LABEL_FILL
        across = min(across_m, 2 * clearance_m) * _LABEL_FILL
        return min(along / (longest * _CHAR_EM), across / (len(lines) * _LINE_EM)) * pt_per_m

    return fits(width, height), fits(height, width)


def _wrap_candidates(name: str, max_lines: int = 3) -> list[str]:
    """``name`` broken over 1..``max_lines`` lines -- the shapes a room label may take.

    No ``max_lines=`` on the wrap call: it would truncate with a placeholder, and for a short name the
    placeholder is wider than the width we are aiming at (a hard error over a label like ``room 4``).
    Whatever the wrap gives back is just another candidate, judged on the type size it buys.
    """
    out = [name]
    for lines in range(2, min(max_lines, len(name.split())) + 1):
        wrapped = "\n".join(textwrap.wrap(name, width=max(len(name) // lines + 1, 1)))
        if "\n" in wrapped and wrapped not in out:
            out.append(wrapped)
    return out


def _fit_label(
    room,
    *,
    areas: bool,
    area_decimals: int,
    clearance_m: float,
    pt_per_m: float,
    style: Style,
    show_name: bool = True,
) -> tuple[str, float, int]:
    """``(text, fontsize, rotation)``: the biggest a room's label can be drawn inside its own room.

    Three things are traded off, in this order of preference: keep the plan's type size; break the name
    over lines (``Meeting Room`` on two lines fits a room that one line does not); and only then turn the
    label upright, the way a plan sets a corridor's name along it. Type is never grown past the plan's
    size and never shrunk past :data:`_MIN_LABEL_FRACTION` of it -- at slide scale a 10 m² server room
    cannot carry the same type as a 165 m² hall, and a name spilling over its walls misreads as belonging
    to the neighbour.
    """
    want = style.pt("room")
    flat: tuple[str, float] | None = None  # best wrapping lying along the room
    turned: tuple[str, float] | None = None  # best wrapping standing upright
    for name in _wrap_candidates(room.name) if show_name else [""]:
        text = room_label(room, areas, area_decimals, name=name)
        straight, upright = _label_sizes(text, room.polygon, clearance_m, pt_per_m)
        # A marginal gain is not worth an extra line break, so ties go to the fewest lines.
        if flat is None or straight > flat[1] * 1.02:
            flat = (text, straight)
        if turned is None or upright > turned[1] * 1.02:
            turned = (text, upright)

    if flat[1] >= want or turned[1] <= flat[1] * _ROTATE_GAIN:
        text, size, rotation = flat[0], flat[1], 0
    else:
        text, size, rotation = turned[0], turned[1], 90
    return text, max(min(want, size), want * _MIN_LABEL_FRACTION), rotation


def room_keys(text: str | None) -> list[str]:
    """``"hall, 4-6, Server room"`` -> ``["hall", "4", "5", "6", "server room"]``.

    How a caller names rooms on the command line: a room's ``name`` (case-insensitively, as it reads in
    the plan) or its numeric ``id``, with ``4-9`` expanding to that range of ids -- a plan with a strip of
    ``room 4`` .. ``room 9`` is exactly where naming them one by one gets tedious. Which rooms exist is
    only known once the floorplan is loaded, so an unmatched key is caught in :func:`render`, not here.
    """
    out: list[str] = []
    for raw in (part.strip() for part in (text or "").split(",")):
        if not raw:
            continue
        lo, sep, hi = raw.partition("-")
        if sep and lo.strip().isdigit() and hi.strip().isdigit():
            first, last = int(lo), int(hi)
            if last < first:
                raise ValueError(f"room range {raw!r} runs backwards")
            out.extend(str(n) for n in range(first, last + 1))
        else:
            out.append(raw.lower())
    return out


def parse_room_colors(specs: list[str] | None, highlight: str | None) -> dict[str, str]:
    """``["a,b=#ffd", ...]`` (+ a ``--highlight`` list) -> ``{room key: colour}``. Keys: :func:`room_keys`."""
    out: dict[str, str] = dict.fromkeys(room_keys(highlight), _INK["highlight"])
    for spec in specs or []:
        names, sep, colour = spec.rpartition("=")
        if not sep or not names.strip() or not colour.strip():
            raise ValueError(f"--room-color {spec!r}: expected 'ROOM[,ROOM...]=COLOUR'")
        for key in room_keys(names):
            out[key] = colour.strip()
    return out


def _names_room(room, keys: set[str]) -> str | None:
    """The key in ``keys`` that picks out ``room`` (its name or its id), or ``None``."""
    for key in (room.name.lower(), str(room.id).lower()):
        if key in keys:
            return key
    return None


def _room_fill(room, colours: dict[str, str]) -> tuple[str, str | None]:
    """``(facecolor, matched key)`` for a room: its accent colour if one names it, else the pale wash."""
    key = _names_room(room, set(colours))
    return (colours[key], key) if key else (_INK["room_fill"], None)


def room_label(room, areas: bool, area_decimals: int = 1, name: str | None = None) -> str:
    """A room's plan label: its name, and its floor area to ``area_decimals`` places under it.

    ``name=""`` drops the name and leaves the area standing alone -- what a room whose identity does not
    matter to the plan still contributes to it.
    """
    shown = room.name if name is None else name
    area = f"{room.area_m2:.{area_decimals}f} m²" if areas else ""
    return "\n".join(part for part in (shown, area) if part)


def _draw_rooms(
    ax,
    Polygon,
    rooms,
    *,
    labels: bool,
    areas: bool,
    style: Style,
    pt_per_m: float,
    colours: dict[str, str] | None = None,
    area_decimals: int = 1,
    unlabelled: set[str] | None = None,
    area_only: set[str] | None = None,
):
    for room in rooms:
        fill, _ = _room_fill(room, colours or {})
        ax.add_patch(
            Polygon(
                room.polygon,
                closed=True,
                facecolor=fill,
                edgecolor=_INK["room_edge"],
                linewidth=style.lw("room_edge"),
                zorder=1,
            )
        )
    if not labels:
        return
    for room in rooms:
        if _names_room(room, unlabelled or set()):
            continue  # deliberately anonymous: a plan that argues about two halls does not name the rest
        x, y, clearance = label_spot(room.polygon)
        text, size, rotation = _fit_label(
            room,
            areas=areas,
            area_decimals=area_decimals,
            clearance_m=clearance,
            pt_per_m=pt_per_m,
            style=style,
            show_name=not _names_room(room, area_only or set()),
        )
        if not text:
            continue
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=size,
            color=_INK["room_text"],
            linespacing=_LINE_EM,
            zorder=6,
        )


def _draw_markers(ax, markers, style: Style):
    for m in markers:
        x, y = float(m["x_m"]), float(m["y_m"])
        dot = _MARKER_DOT_PT * style.scale
        ax.plot([x], [y], marker="o", markersize=dot, color=_INK["marker"], zorder=8)
        label = str(m.get("comment") or m.get("id") or "")
        if len(label) > 28:  # a marker comment can be a sentence; a plan label cannot
            label = label[:27] + "…"
        if label:
            ax.text(
                x,
                y,
                f"  {label}",
                ha="left",
                va="center",
                fontsize=style.pt("marker"),
                color=_INK["marker"],
                zorder=8,
            )


def _draw_grid(ax, Line2D, extent, step: float, style: Style):
    x0, y0, x1, y1 = extent
    n = math.floor((x1 - x0) / step) + math.floor((y1 - y0) / step)
    if n > 4000:
        raise ValueError(
            f"--grid {step:g} m would draw {n} lines over this plan; use a coarser step"
        )
    v = math.ceil(x0 / step) * step
    while v <= x1:
        ax.add_line(
            Line2D([v, v], [y0, y1], color=_INK["grid"], linewidth=style.lw("grid"), zorder=0)
        )
        v += step
    v = math.ceil(y0 / step) * step
    while v <= y1:
        ax.add_line(
            Line2D([x0, x1], [v, v], color=_INK["grid"], linewidth=style.lw("grid"), zorder=0)
        )
        v += step


def _legend(ax, Polygon, Line2D, kinds: list[str], placement: str, style: Style):
    handles = [Polygon([(0, 0)], facecolor=_INK["wall"], edgecolor=_INK["wall_edge"], label="wall")]
    for kind in _KIND_LABELS:
        if kind not in kinds:
            continue
        if kind == "window":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=_INK["window"],
                    linewidth=style.lw("legend_line"),
                    label=_KIND_LABELS[kind],
                )
            )
        elif kind == "door":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=_INK["door"],
                    linewidth=style.lw("legend_line"),
                    label=_KIND_LABELS[kind],
                )
            )
        else:
            handles.append(
                Polygon(
                    [(0, 0)],
                    facecolor="none",
                    edgecolor=_INK[kind],
                    linestyle="--",
                    label=_KIND_LABELS[kind],
                )
            )
    loc, anchor = _LEGEND_LOC[placement]
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        ncols=len(handles) if placement == "below" else 1,
        fontsize=style.pt("legend"),
        framealpha=0.92,
        edgecolor=_INK["room_edge"],
        borderpad=0.6,
        handlelength=1.6,
    ).set_zorder(12)


@dataclass
class PlanStats:
    """What the rendered plan contains -- printed as the CLI's one-line receipt."""

    width_m: float
    height_m: float
    walls: int
    rooms: int
    kinds: dict[str, int]
    scale_bar_m: float | None


def render(
    src: FloorplanSource,
    out: Path,
    *,
    scale_bar: str = "bottom-left",
    scale_bar_length: float | None = None,
    scale_bar_height: float | None = None,
    legend: str = "below",
    grid: float = 0.0,
    ids: bool = False,
    rooms: bool = True,
    room_labels: bool = True,
    areas: bool = True,
    area_decimals: int = 1,
    room_colors: dict[str, str] | None = None,
    unlabelled_rooms: list[str] | None = None,
    area_only_rooms: list[str] | None = None,
    markers: bool = True,
    wall_thickness: float = 0.1,
    opening_h: float = 2.0,
    ceiling_h: float = 2.5,
    door_swing_deg: float = _DOOR_SWING_DEG,
    width_px: int = 1800,
    dpi: int = 150,
    title: str | None = None,
    axes: bool = False,
    font_scale: float = 1.0,
    font_size: float | None = None,
) -> PlanStats:
    """Draw ``src`` to ``out`` (PNG) and return what was drawn.

    ``font_scale`` (or ``font_size``, the room label's size in points) scales every label and the
    point-width strokes with it -- 2..3 for a slide or a poster.
    """
    plt, Polygon, Rectangle, Line2D = _require_matplotlib()
    style = Style.from_font_size(font_size) if font_size else Style(scale=font_scale)

    fp = src.floorplan
    lines = fp.get("lines") or []
    walls = plan_walls(lines, fp.get("doors") or [], opening_h)
    room_list = room_polygons(fp) if rooms else []
    xmin, ymin, xmax, ymax = bounds(lines)
    half = wall_thickness / 2
    xmin, ymin, xmax, ymax = xmin - half, ymin - half, xmax + half, ymax + half
    span = max(xmax - xmin, ymax - ymin)

    bar_kind, bar_xy = parse_scale_bar_pos(scale_bar)
    bar_len = None
    if bar_kind != "none":
        bar_len = scale_bar_length if scale_bar_length else nice_scale_length(xmax - xmin)
    margin = 0.03 * span
    x0, x1 = xmin - margin, xmax + margin
    y0, y1 = ymin - margin, ymax + margin

    # The figure is exactly `width_px` wide (no bbox="tight" crop, which would silently ignore it), and
    # as tall as the plan's aspect ratio needs; constrained layout fits the title and the below-the-plan
    # legend inside that canvas instead of growing it.
    fig_w = width_px / dpi
    fig_h = fig_w * (y1 - y0) / (x1 - x0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")

    if grid:
        _draw_grid(ax, Line2D, (x0, y0, x1, y1), grid, style)
    # Points per metre of plan: what a label's size has to be judged against (the axes is ~fig-wide).
    pt_per_m = fig_w * 72.0 / (x1 - x0)
    colours = dict(room_colors or {})
    unlabelled = set(unlabelled_rooms or ())
    area_only = set(area_only_rooms or ())
    # Fail loud on a room key nobody has: a typo would otherwise render a plan with nothing highlighted
    # (or everything still labelled) that looks just like a finished answer.
    for what, keys in (
        ("--room-color/--highlight", set(colours)),
        ("--no-label", unlabelled),
        ("--area-only", area_only),
    ):
        matched = {key for room in room_list if (key := _names_room(room, keys))}
        if missing := sorted(keys - matched):
            raise ValueError(
                f"{what} names {missing} match no room; this floorplan has "
                f"{sorted(r.name for r in room_list)} (ids {sorted(str(r.id) for r in room_list)})"
            )
    _draw_rooms(
        ax,
        Polygon,
        room_list,
        labels=room_labels,
        areas=areas,
        style=style,
        pt_per_m=pt_per_m,
        colours=colours,
        area_decimals=area_decimals,
        unlabelled=unlabelled,
        area_only=area_only,
    )
    _draw_walls(ax, Polygon, walls, wall_thickness, ids, style)
    kinds = _draw_openings(
        ax,
        Polygon,
        Line2D,
        walls,
        wall_thickness,
        src.doors_map,
        opening_h,
        ids,
        style,
        ceiling_h,
        door_swing_deg,
    )
    if markers:
        _draw_markers(ax, fp.get("markers") or [], style)

    if axes:
        ax.set_xlabel("x [m]", fontsize=style.pt("axis_label"))
        ax.set_ylabel("y [m]", fontsize=style.pt("axis_label"))
        ax.tick_params(labelsize=style.pt("axis_tick"))
        for side in ax.spines.values():
            side.set_color(_INK["room_edge"])
    else:
        ax.set_axis_off()

    if bar_len:
        if bar_kind == "xy":
            anchor, ha, va = bar_xy, "left", "bottom"
        else:
            anchor, ha, va = corner_anchor(bar_kind, (xmin, ymin, xmax, ymax))
        draw_scale_bar(
            ax,
            Rectangle,
            length_m=bar_len,
            span_m=xmax - xmin,
            anchor=anchor,
            ha=ha,
            va=va,
            style=style,
            height_m=scale_bar_height,
            pt_per_m=pt_per_m,
        )
    if legend != "none" and kinds:
        _legend(ax, Polygon, Line2D, kinds, legend, style)

    if title is None:
        title = f"{src.name} — {xmax - xmin:.1f} × {ymax - ymin:.1f} m"
    if title:
        ax.set_title(title, fontsize=style.pt("title"), color=_INK["room_text"])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)

    counts: dict[str, int] = {}
    for wall in walls:
        for op in wall.openings:
            kind = opening_kind(op, src.doors_map, opening_h, ceiling_h)
            counts[kind] = counts.get(kind, 0) + 1
    return PlanStats(
        width_m=xmax - xmin,
        height_m=ymax - ymin,
        walls=len(walls),
        rooms=len(room_list),
        kinds=counts,
        scale_bar_m=bar_len,
    )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render a floorplan (or the floorplan behind a roqsim scene/world) to a PNG plan "
        "view with a metre scale bar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "SOURCE may be a floorplan JSON, a scene dir holding scene.json, a world YAML, or a\n"
            "'<package>:<world>' ref, e.g.:\n"
            "  roqsim-floorplan-to-png floorplan.json -o plan.png\n"
            "  roqsim-floorplan-to-png scenes/<name> --scale-bar bottom-left --ids\n"
        ),
    )
    ap.add_argument("source", help="floorplan JSON | scene dir | world YAML | <package>:<world>")
    ap.add_argument("-o", "--out", help="output PNG (default: <name>_floorplan.png here)")
    ap.add_argument(
        "--scale-bar",
        default="bottom-left",
        metavar="POS",
        help=f"scale bar position: {', '.join(_CORNERS)}, 'none', or 'X,Y' in world metres "
        "(default: bottom-left)",
    )
    ap.add_argument(
        "--scale-bar-length",
        type=float,
        metavar="M",
        help="scale bar length in metres (default: a round ~1/6 of the plan width)",
    )
    ap.add_argument(
        "--scale-bar-height",
        type=float,
        metavar="M",
        help="scale bar thickness in metres (default: a slim band, ~0.6%% of the plan width)",
    )
    ap.add_argument(
        "--legend",
        default="below",
        choices=[*_LEGEND_LOC, "none"],
        help="legend placement: 'below' the plan (default, never covers geometry), a corner, or 'none'",
    )
    ap.add_argument(
        "--grid", type=float, default=0.0, metavar="M", help="draw a metre grid at this spacing"
    )
    ap.add_argument(
        "--axes", action="store_true", help="show x/y axes in metres (for reading coordinates off)"
    )
    ap.add_argument(
        "--ids",
        action="store_true",
        help="annotate wall line ids and door ids (the handles floorplan.json uses)",
    )
    ap.add_argument(
        "--no-legend", action="store_true", help="omit the legend (same as --legend none)"
    )
    ap.add_argument("--no-rooms", action="store_true", help="do not fill/label the rooms")
    ap.add_argument("--no-room-labels", action="store_true", help="fill rooms but do not name them")
    ap.add_argument("--no-areas", action="store_true", help="omit the m² in room labels")
    ap.add_argument(
        "--area-decimals",
        type=int,
        default=1,
        metavar="N",
        help="decimals in a room's m² (default 1; 0 for whole square metres)",
    )
    ap.add_argument(
        "--room-color",
        action="append",
        metavar="ROOM[,ROOM]=COLOUR",
        help="fill these rooms (by name or id) with this colour instead of the default wash; "
        "repeatable, e.g. --room-color 'meeting room=#d8e8d0'",
    )
    ap.add_argument(
        "--no-label",
        metavar="ROOM[,ROOM]",
        help="do not name these rooms (by name, id, or an id range like 4-9); their fill and area stay",
    )
    ap.add_argument(
        "--area-only",
        metavar="ROOM[,ROOM]",
        help="label these rooms with their m² alone, no name (by name, id, or an id range like 4-9)",
    )
    ap.add_argument(
        "--highlight",
        metavar="ROOM[,ROOM]",
        help="fill these rooms (by name or id) with the accent colour -- shorthand for --room-color",
    )
    ap.add_argument("--no-markers", action="store_true", help="do not draw the marker points")
    ap.add_argument(
        "--doors-map",
        metavar="JSON",
        help="the generator's --doors-map, saying what fills each opening "
        f"(default: {_DOORS_MAP_NAME} beside the floorplan, if present)",
    )
    ap.add_argument(
        "--no-doors-map",
        action="store_true",
        help="ignore a doors_map.json found beside the floorplan",
    )
    ap.add_argument(
        "--wall-thickness",
        type=float,
        default=0.1,
        metavar="M",
        help="drawn wall thickness in metres (default 0.1, the generator's default)",
    )
    ap.add_argument(
        "--door-swing",
        type=float,
        default=_DOOR_SWING_DEG,
        metavar="DEG",
        help="how far a door leaf is drawn open (default 90, the drafting convention); 0 draws it "
        "closed with no arc",
    )
    ap.add_argument(
        "--ceiling-h",
        type=float,
        default=2.5,
        metavar="M",
        help="wall height in metres (default 2.5, the generator's default); an opening this tall gets "
        "no leaf, matching the bare gate the generator leaves",
    )
    ap.add_argument(
        "--opening-h",
        type=float,
        default=2.0,
        metavar="M",
        help="default opening height in metres (default 2.0, the generator's default)",
    )
    ap.add_argument(
        "--width-px", type=int, default=1800, help="image width in pixels (default 1800)"
    )
    ap.add_argument("--dpi", type=int, default=150, help="dots per inch (default 150)")
    ap.add_argument(
        "--title", help="plan title (default: name + extents; pass an empty string for none)"
    )
    ap.add_argument("--no-title", action="store_true", help="draw no title")
    fonts = ap.add_mutually_exclusive_group()  # two ways to say the same thing
    fonts.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        metavar="F",
        help="scale every label (and the point-width strokes) by F; use 2-3 for a presentation",
    )
    fonts.add_argument(
        "--font-size",
        type=float,
        metavar="PT",
        help="room-label size in points (default 7.5); the other labels keep their ratio to it",
    )
    args = ap.parse_args(argv)

    try:
        src = load_source(
            args.source, doors_map=args.doors_map, use_doors_map=not args.no_doors_map
        )
        out = Path(args.out) if args.out else Path(f"{src.name}_floorplan.png")
        stats = render(
            src,
            out,
            scale_bar=args.scale_bar,
            scale_bar_length=args.scale_bar_length,
            scale_bar_height=args.scale_bar_height,
            legend="none" if args.no_legend else args.legend,
            grid=args.grid,
            ids=args.ids,
            rooms=not args.no_rooms,
            room_labels=not args.no_room_labels,
            areas=not args.no_areas,
            area_decimals=args.area_decimals,
            room_colors=parse_room_colors(args.room_color, args.highlight),
            unlabelled_rooms=room_keys(args.no_label),
            area_only_rooms=room_keys(args.area_only),
            markers=not args.no_markers,
            wall_thickness=args.wall_thickness,
            opening_h=args.opening_h,
            ceiling_h=args.ceiling_h,
            door_swing_deg=args.door_swing,
            width_px=args.width_px,
            dpi=args.dpi,
            title="" if args.no_title else args.title,
            axes=args.axes,
            font_scale=args.font_scale,
            font_size=args.font_size,
        )
    except (ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    kinds = ", ".join(f"{n} {_KIND_LABELS[k]}" for k, n in sorted(stats.kinds.items())) or "none"
    bar = (
        f"{stats.scale_bar_m:g} m bar at {args.scale_bar}" if stats.scale_bar_m else "no scale bar"
    )
    print(
        f"wrote {out.resolve()}\n"
        f"  from {src.path}"
        + (f" + {src.doors_map_path.name}" if src.doors_map_path else " (no doors map)")
        + f"\n  {stats.width_m:.2f} x {stats.height_m:.2f} m, {stats.rooms} rooms, "
        f"{stats.walls} wall lines, openings: {kinds}\n  {bar}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
