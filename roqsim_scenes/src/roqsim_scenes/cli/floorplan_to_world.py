# SPDX-License-Identifier: Apache-2.0
"""Generate a roqsim world from a floorplan -- deterministically, no LLM in the loop.

This is stage 1 of the scene pipeline for a *hand-drawn* source: it consumes the floorplan JSON that the
scene-builder's ``sketch_floorplan_by_human`` tool returns (``lines`` = independent wall segments in
metres, one wall box each; ``doors`` = 2 m-high openings with a solid lintel above them up to the
ceiling, each fitted with a swing-door leaf via the ``door`` plugin -- unless it is full-height, then
it stays a bare opening; markers = labelled interior features) and emits the same artifacts every other
importer does -- world-space OBJs + ``scene.json`` -- then bakes them with the shared stage 2
(``scene_to_mjcf.py``) using the shared generated-room look (``floorplan.scene.yaml``), and writes the world
YAML. The marker -> prop model choice is the one thing left to the caller: it is passed in as a
``--markers-map`` and a marker with no mapping is a hard error, never a silent placeholder.

``--ceiling`` roofs the building with a concrete slab whose soffit is the wall top (``--ceiling-h``),
and pulls the bake's overhead light under it so the room is not lit from above a solid slab. The roof
comes back off at run time through the core ``ceiling`` plugin, which deletes by height rather than by
name -- so a world screenshots from inside with the ceiling and from above without it.

The bake's look comes from ``--bake-config``, else the scene dir's own ``scene.yaml``, else the shared
``floorplan.scene.yaml``: a building with its own floor/wall/soffit surfaces carries that scene.yaml
beside its ``floorplan.json`` instead of repainting the look every other generated room inherits.

The floorplan is the single source of truth: ``generate`` writes it verbatim to
``<out_dir>/floorplan.json`` and ``scene.json`` only *references* it (``"floorplan": "floorplan.json"``),
so a scene round-trips back to a floorplan JSON via ``scene_to_floorplan.py``. The floorplan-level
``description`` and the per-room ``description`` (carried inside ``rooms``) are the object-placement
intent an agent reads when choosing props; edit them in ``floorplan.json`` -- they are not baked into
geometry, so a description-only change takes effect with no re-run.

Reuses ``scene_mesh_io`` (``box``/``transform``/``write_obj``) for the geometry and ``scene_to_mjcf.py``
for the bake, so the generated world sits on exactly the same collision/material/lighting path as the
imported scenes.

Usage::

    roqsim scenes floorplan-to-world --floorplan floorplan.json \\
        --out-dir  ../src/roqsim_scenes/scenes/myroom --scene-name myroom \\
        --world-out ../src/roqsim_scenes/worlds/myroom.yaml \\
        --markers-map markers.json          # {"1": "industrial_table",
                                            #  "2": {"model": "single_bed", "yaw_deg": 180}}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from roqsim_scenes import scene_mesh_io as mio

# The wall/opening arithmetic is shared with the plan-view renderer (roqsim_scenes.floorplan_to_png), so it
# lives in the installed package: a preview drawing different openings than this baker cuts would lie.
from roqsim_scenes.floorplan_geometry import (  # noqa: F401 - re-exported: the tool's public geometry
    assign_doors,
    cut_openings,
    line_segments,
)

from . import scene_to_mjcf

_FLOOR_THICKNESS = 0.05
_CEILING_THICKNESS = 0.25  # a slab of structural concrete, not a suspended tile grid
_FLOOR_MARGIN_M = 0.2  # the floor extends this far beyond the outermost walls
_LIGHT_UNDER_CEILING_M = 0.15  # how far below a ceiling the bake's overhead light is pulled
_SHARED_CONFIG = Path(__file__).resolve().parent / "floorplan.scene.yaml"
_SCENE_CONFIG_NAME = "scene.yaml"  # a scene dir may carry its own look, overriding the shared one
_FLOORPLAN_NAME = (
    "floorplan.json"  # the authored floorplan, written beside scene.json and referenced by it
)


# --- geometry (pure) -----------------------------------------------------------------------------


def _yaw_matrix(yaw: float, translate) -> np.ndarray:
    """4x4 = rotate ``yaw`` about +Z, then translate."""
    c, s = math.cos(yaw), math.sin(yaw)
    mat = np.eye(4)
    mat[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    mat[:3, 3] = translate
    return mat


def wall_pieces(lines: list[dict], doors: list[dict], ceiling_h: float, opening_h: float):
    """The wall boxes to build as ``(p0, p1, z0, z1)``, with each line's door openings cut out.

    A drawn line becomes: full-height solid pieces ``(0, ceiling_h)`` around its openings, plus one
    lintel per opening spanning ``(height, ceiling_h)`` over the opening width (skipped when the
    opening reaches the ceiling). So a door is a 2 m-high hole with a beam above it, not a full gap.
    """
    per_seg = assign_doors(lines, doors, opening_h)
    pieces = []
    for i, ((x0, y0), (x1, y1)) in enumerate(line_segments(lines)):
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        openings = per_seg.get(i, [])
        for t0, t1 in cut_openings(length, [(t, w) for t, w, _ in openings]):
            pieces.append(
                ((x0 + ux * t0, y0 + uy * t0), (x0 + ux * t1, y0 + uy * t1), 0.0, ceiling_h)
            )
        for t_m, width, height in openings:
            if height >= ceiling_h:
                continue  # a full-height opening keeps a true doorway -- no lintel above it
            g0, g1 = max(0.0, t_m - width / 2), min(length, t_m + width / 2)
            pieces.append(
                ((x0 + ux * g0, y0 + uy * g0), (x0 + ux * g1, y0 + uy * g1), height, ceiling_h)
            )
    return pieces


def wall_box(
    p0, p1, z0: float, z1: float, thickness: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space (verts, faces, uv) for a wall slab spanning ``p0``->``p1``, from z=``z0`` to z=``z1``.

    ``z0``/``z1`` let a piece be a floor-standing wall (``0``..ceiling) or a lintel above an opening
    (``opening_h``..ceiling). The UVs are metric and per face (:func:`scene_mesh_io.box_uv`), so a
    wall texture tiles at true scale instead of being smeared along one axis.
    """
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        raise ValueError(f"degenerate wall segment {p0}->{p1} (zero length)")
    height = z1 - z0
    if height <= 0:
        raise ValueError(f"wall piece {p0}->{p1} has non-positive height (z {z0}->{z1})")
    verts, faces, uv = mio.box_uv(f"{length} {thickness} {height}")
    mid = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    return mio.transform(verts, _yaw_matrix(math.atan2(y1 - y0, x1 - x0), mid)), faces, uv


def floor_box(x0: float, y0: float, x1: float, y1: float, thickness: float = _FLOOR_THICKNESS):
    """World-space (verts, faces, uv) for a floor slab covering [x0,x1]x[y0,y1], top at z=0."""
    w, h = x1 - x0, y1 - y0
    verts, faces, uv = mio.box_uv(f"{w} {h} {thickness}")
    return (
        mio.transform(verts, _yaw_matrix(0.0, ((x0 + x1) / 2, (y0 + y1) / 2, -thickness / 2))),
        faces,
        uv,
    )


def ceiling_box(
    x0: float, y0: float, x1: float, y1: float, z: float, thickness: float = _CEILING_THICKNESS
):
    """World-space (verts, faces, uv) for a ceiling slab over [x0,x1]x[y0,y1], **soffit at z=``z``**.

    The slab sits entirely above the walls' top edge, which is what lets the core ``ceiling`` plugin
    take the roof off by height (it deletes geoms whose whole AABB clears its ``above_z``) without a
    per-object name list.
    """
    w, h = x1 - x0, y1 - y0
    verts, faces, uv = mio.box_uv(f"{w} {h} {thickness}")
    return (
        mio.transform(verts, _yaw_matrix(0.0, ((x0 + x1) / 2, (y0 + y1) / 2, z + thickness / 2))),
        faces,
        uv,
    )


# --- manifests (pure) ----------------------------------------------------------------------------


def scene_manifest(
    name: str,
    bbox: tuple[float, float, float, float],
    ceiling_h: float,
    n_walls: int,
    floorplan_ref: str | None = None,
    ceiling: bool = False,
) -> dict:
    """The ``scene.json`` for the floor + N walls (object names match floorplan.scene.yaml globs).

    When ``floorplan_ref`` is given it is stored under ``"floorplan"`` as a *relative path* to the
    authored floorplan JSON (written beside this manifest, see :func:`generate`) -- not an embedded
    copy. That file is the single source of truth for the lines/rooms/doors/markers and the
    object-placement ``description``\\ s; ``scene_to_floorplan.py`` follows the reference to
    round-trip a scene back to its floorplan JSON (the baked geometry alone loses lines/rooms/doors).
    """
    x0, y0, x1, y1 = bbox
    objects = [{"name": "Floor", "mesh": "meshes/Floor.obj", "collide": False}]
    for i in range(1, n_walls + 1):
        objects.append(
            {"name": f"Wall_{i:02d}", "mesh": f"meshes/Wall_{i:02d}.obj", "collide": True}
        )
    if ceiling:
        # Visual only, like the Floor: a convex collider over the whole plan would be a solid block
        # filling the building, and nothing in a ground-robot world touches the soffit anyway.
        objects.append({"name": "Ceiling", "mesh": "meshes/Ceiling.obj", "collide": False})
    manifest = {
        "name": name,
        "source": "floorplan_sketch",
        "ground_z": 0.0,
        "bounds_min": [float(x0), float(y0), -_FLOOR_THICKNESS],
        "bounds_max": [
            float(x1),
            float(y1),
            float(ceiling_h) + (_CEILING_THICKNESS if ceiling else 0.0),
        ],
        "objects": objects,
    }
    if floorplan_ref is not None:
        manifest["floorplan"] = floorplan_ref
    return manifest


def _view(bbox: tuple[float, float, float, float]) -> dict:
    x0, y0, x1, y1 = bbox
    span = max(x1 - x0, y1 - y0)
    return {
        "lookat": [round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3), 1.0],
        "distance": round(span * 1.8, 3),
        "azimuth": 90.0,
        "elevation": -55.0,
    }


def _map_entry(entry) -> tuple[str, float | None]:
    """Normalise a ``--markers-map`` value into ``(model, yaw_deg)``.

    A value is either a bare model name (``"single_bed"``) or a dict carrying the model plus optional
    placement detail (``{"model": "single_bed", "yaw_deg": 180}``). Orientation is a *caller* decision,
    like the model choice, so it lives here in the map -- not in the human-drawn sketch. A marker whose
    heading came from the sketch UI is honoured too (see :func:`world_doc`); a map ``yaw_deg`` overrides.
    """
    if isinstance(entry, dict):
        model = entry.get("model")
        if not model:
            raise KeyError(f"markers-map entry {entry!r} has no 'model'")
        yaw = entry.get("yaw_deg")
        return str(model), (float(yaw) if yaw is not None else None)
    return str(entry), None


def door_placements(
    lines: list[dict], doors: list[dict], doors_map: dict, ceiling_h: float, opening_h: float
) -> list[dict]:
    """One ``{"door": {...}}`` plugin entry per swing-door opening.

    Reuses the same opening geometry as :func:`assign_doors` (line + fraction ``t`` + ``width_m``) to
    place the leaf by the opening's **centre** and the wall yaw; ``hinge_side`` / ``swing`` / ``open`` /
    ``model`` / ``controllable`` come from ``doors_map[id]`` (defaults: a passive, closed wooden door
    hinged on the ``left`` -- automatic doors are opt-in). A **full-height** opening
    (``height >= ceiling``) is a true doorway/gate, not a swing door: it gets no leaf (a candidate for
    a future sliding/double door), reported on stderr so the drop is never silent.

    Two doors-map keys say the opening is not a swinging door. Both keep it a floorplan ``door``, so the
    room loops it belongs to are untouched -- only what fills it changes:

    * ``{"leaf": false}`` -- a **cased opening** (a *Türblatt*-less door). Forwarded to the plugin, which
      welds the casing and hangs no leaf, so the gap still reads as a doorway.
    * ``{"skip": true}`` -- **no door plugin at all**, for an opening something else fills: a ``window``
      plugin entry at the same centre, or a deliberately bare gap. Without it a window opening would get
      a wooden door in it.
    """
    by_id = {int(x["id"]): i for i, x in enumerate(lines)}
    out: list[dict] = []
    for d in doors:
        did = str(d.get("id", "?"))
        line_id, t, width = int(d["line_id"]), float(d["t"]), float(d.get("width_m", 0.9))
        height = float(d.get("height_m", opening_h))
        if line_id not in by_id:
            raise ValueError(f"door {did} references line {line_id}, which does not exist")
        idx = by_id[line_id]
        (x0, y0), (x1, y1) = (
            (lines[idx]["x0_m"], lines[idx]["y0_m"]),
            (lines[idx]["x1_m"], lines[idx]["y1_m"]),
        )
        length = math.hypot(x1 - x0, y1 - y0)
        if length < width:
            raise ValueError(
                f"door {did} needs a {width:g} m wall but line {line_id} is {length:.2f} m"
            )
        if height >= ceiling_h:
            print(
                f"floorplan_to_world: door {did} is full-height ({height:g} m >= ceiling "
                f"{ceiling_h:g} m); left as a bare opening (no swing leaf)",
                file=sys.stderr,
            )
            continue
        entry = dict(doors_map.get(did, {}))
        if entry.pop("skip", False):
            continue  # filled by something else (a `window` plugin) or left a bare gap on purpose
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        t_m = min(
            max(t * length, width / 2), length - width / 2
        )  # keep the opening inside the wall
        cx, cy = x0 + ux * t_m, y0 + uy * t_m
        yaw = math.atan2(uy, ux)
        door = {
            "name": entry.get("name", f"door_{did}"),
            "prefix": entry.get("prefix", f"door_{did}_"),
            "pos": [round(cx, 3), round(cy, 3), 0.0],
            "rpy": [0.0, 0.0, round(yaw, 5)],
            "width": round(width, 3),
            "height": round(height, 3),
            "model": entry.get("model", "door"),
            "hinge_side": entry.get("hinge_side", "left"),
            "swing": int(entry.get("swing", 1)),
            "open": float(entry.get("open", 0.0)),
            "controllable": bool(entry.get("controllable", False)),
        }
        for key in (
            "namespace",
            "max_angle",
            "thickness",
            "kp",
            "kv",
            "color",
            "frame_color",
            "leaf",
        ):
            if key in entry:
                door[key] = entry[key]
        out.append({"door": door})
    return out


def world_doc(
    name: str,
    xml_relpath: str,
    bbox: tuple[float, float, float, float],
    markers: list[dict],
    markers_map: dict,
    doors: list[dict] | None = None,
) -> dict:
    """The world-YAML document: the baked scene as ``sim.world`` + one spawn_model per marker.

    Every marker id must be in ``markers_map``; a missing one raises (fail loud, no placeholder). A
    map value is a bare model name or ``{"model", "yaw_deg"}``. A prop is placed axis-aligned unless a
    heading is given: the map's ``yaw_deg`` wins, else the marker's own ``yaw_deg`` (set in the sketch
    UI's Mark mode); the chosen yaw is emitted as spawn_model's ``rpy`` (roll/pitch stay 0 -- a
    floor-standing prop only turns about +Z).
    """
    plugins = []
    for m in markers:
        mid = str(m["id"])
        if mid not in markers_map:
            raise KeyError(
                f"marker {mid} ('{m.get('comment', '')}') has no model in --markers-map; "
                f"resolve it to a roqsim model (import one if needed) before generating"
            )
        model, yaw_deg = _map_entry(markers_map[mid])
        if (
            yaw_deg is None
        ):  # caller gave no heading -> honour a heading the human drew in the sketch
            yaw_deg = m.get("yaw_deg")
        spawn = {
            "model": model,
            "name": f"marker_{mid}",
            "prefix": f"marker_{mid}_",
            "pos": [round(float(m["x_m"]), 3), round(float(m["y_m"]), 3), 0.0],
        }
        if yaw_deg:
            spawn["rpy"] = [0.0, 0.0, round(math.radians(float(yaw_deg)), 5)]
        plugins.append({"spawn_model": spawn})
    # Doors first (structural), then the marker props.
    plugins = list(doors or []) + plugins
    return {
        "sim": {"world": xml_relpath, "pacing": "realtime", "view": _view(bbox)},
        "components": plugins,
    }


# --- orchestration -------------------------------------------------------------------------------


def bbox_of(
    lines: list[dict], margin: float = _FLOOR_MARGIN_M
) -> tuple[float, float, float, float]:
    """The floorplan's extent (x0, y0, x1, y1) -- the walls' bounding box, padded by ``margin``."""
    xs = [c for ln in lines for c in (ln["x0_m"], ln["x1_m"])]
    ys = [c for ln in lines for c in (ln["y0_m"], ln["y1_m"])]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _write_geometry(
    out_dir: Path, bbox, thickness: float, pieces, ceiling_h: float | None = None
) -> None:
    meshes = out_dir / "meshes"
    fv, ff, fuv = floor_box(*bbox)
    mio.write_obj(meshes / "Floor.obj", fv, ff, fuv)
    for i, (p0, p1, z0, z1) in enumerate(pieces, start=1):
        wv, wf, wuv = wall_box(p0, p1, z0, z1, thickness)
        mio.write_obj(meshes / f"Wall_{i:02d}.obj", wv, wf, wuv)
    if ceiling_h is not None:
        cv, cf, cuv = ceiling_box(*bbox, ceiling_h)
        mio.write_obj(meshes / "Ceiling.obj", cv, cf, cuv)


def resolve_bake_config(out_dir: Path, explicit: Path | None) -> Path:
    """Which look the bake uses: ``--bake-config`` > the scene dir's own ``scene.yaml`` > the shared one.

    The middle rung is what lets a *building* differ from the shared reference look (carpet instead of
    plaster underfoot, an exposed concrete soffit) without every generated room inheriting that
    choice: drop a ``scene.yaml`` beside the scene's ``floorplan.json``. It is authored, never
    generated, so a rebake keeps it. Same filename/precedence as ``scene_to_mjcf.py``'s own default.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"--bake-config {explicit} does not exist")
        return explicit
    own = out_dir / _SCENE_CONFIG_NAME
    return own if own.is_file() else _SHARED_CONFIG


def light_under_ceiling(config_path: Path, ceiling_h: float, tmp_dir: Path) -> Path:
    """Return a bake config whose overhead light hangs *below* ``ceiling_h`` (copying only if needed).

    The bake puts one hemispherical light at ``light.height`` above the floor. With a ceiling slab in
    the world, a light at or above the soffit is trapped above it and the room renders black -- a
    failure that looks like a broken texture, not a misplaced light. So a ceiling always pulls the
    light under it; the config is copied to ``tmp_dir`` rather than edited in place, since it may be
    the shared one every other scene bakes against.
    """
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    height = float((config.get("light") or {}).get("height", 4.0))
    if height < ceiling_h:
        return config_path
    lowered = round(ceiling_h - _LIGHT_UNDER_CEILING_M, 3)
    config.setdefault("light", {})["height"] = lowered
    print(
        f"  ceiling {ceiling_h:g} m: overhead light lowered {height:g} -> {lowered:g} m "
        f"(a light above the soffit leaves the room unlit)"
    )
    out = tmp_dir / "bake.scene.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out


def generate(
    floorplan: dict,
    out_dir: Path,
    scene_name: str,
    world_out: Path,
    markers_map: dict[str, str],
    ceiling_h: float,
    wall_thickness: float,
    opening_h: float,
    doors_map: dict[str, dict] | None = None,
    ceiling: bool = False,
    bake_config: Path | None = None,
) -> Path:
    """Emit floorplan.json + scene.json + OBJs, bake the MJCF, write the world YAML. Returns it."""
    import tempfile

    import yaml

    lines = floorplan.get("lines") or []
    if len(lines) < 1:
        raise ValueError("floorplan has no wall lines to build")
    bbox = bbox_of(lines)  # floor + scene bounds come from the walls' extent
    markers = floorplan.get("markers") or []
    # Fail loud on an unmapped marker BEFORE doing any work (writing meshes, baking the MJCF).
    missing = [str(m["id"]) for m in markers if str(m["id"]) not in markers_map]
    if missing:
        raise KeyError(
            f"markers {missing} have no model in --markers-map; resolve each to a roqsim model "
            f"(import one if needed) before generating"
        )

    # Doors are openings attached to a line: cut each wall into floor pieces + a lintel above.
    pieces = wall_pieces(lines, floorplan.get("doors") or [], ceiling_h, opening_h)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_geometry(out_dir, bbox, wall_thickness, pieces, ceiling_h if ceiling else None)
    # The floorplan is the single source of truth: write it verbatim beside scene.json and reference
    # it (scene.json carries only the path, never a copy).
    (out_dir / _FLOORPLAN_NAME).write_text(json.dumps(floorplan, indent=2), encoding="utf-8")
    (out_dir / "scene.json").write_text(
        json.dumps(
            scene_manifest(
                scene_name, bbox, ceiling_h, len(pieces), _FLOORPLAN_NAME, ceiling=ceiling
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # Bake with the shared stage 2. The look is the scene's own scene.yaml if it has one, else the
    # shared generated-room look. Output beside the world YAML.
    config_path = resolve_bake_config(out_dir, bake_config)
    baked_xml = world_out.parent / scene_name / f"{scene_name}.xml"
    baked_xml.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if ceiling:
            config_path = light_under_ceiling(config_path, ceiling_h, Path(tmp))
        # Stage 2 is a module now, so call it directly: a subprocess here only bought a path to a
        # script file, and that path is exactly what stopped working when the tools were installed.
        if scene_to_mjcf.main(
            ["--scene", str(out_dir), "--config", str(config_path), "--out", str(baked_xml)]
        ):
            raise RuntimeError(f"baking {scene_name} failed; see the stage-2 output above")

    door_plugins = door_placements(
        lines, floorplan.get("doors") or [], doors_map or {}, ceiling_h, opening_h
    )
    doc = world_doc(
        scene_name, f"{scene_name}/{scene_name}.xml", bbox, markers, markers_map, doors=door_plugins
    )
    world_out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated from a floorplan by floorplan_to_world.py. Edit the scene's floorplan.json and\n"
        "# re-run to change walls; edit --markers-map to change props, --doors-map to change door\n"
        "# leaves. sim.world points at the baked MJCF.\n"
    )
    world_out.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return world_out


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate a roqsim world from a floorplan.")
    ap.add_argument(
        "--floorplan", required=True, help="floorplan JSON from sketch_floorplan_by_human"
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="scene dir to write (floorplan.json + scene.json + meshes/)",
    )
    ap.add_argument("--scene-name", required=True, help="scene/world name")
    ap.add_argument("--world-out", required=True, help="world YAML path to write")
    ap.add_argument(
        "--markers-map",
        help="JSON {marker_id: model}; required if the floorplan has markers. A value is a "
        'model name ("single_bed") or {"model": ..., "yaw_deg": ...} to also set '
        "the prop's heading (else the marker's own drawn yaw is used)",
    )
    ap.add_argument(
        "--doors-map",
        help="JSON {door_id: {...}} overriding per-door leaf options (optional; unmapped "
        "doors get a passive, closed wooden door hinged 'left'). Keys: model "
        "(door|door_glass), hinge_side (left|right), swing (+1/-1), open (0..1), "
        "controllable (bool -> automatic door), namespace, max_angle, leaf (false -> a "
        "cased opening: casing welded, no leaf hung), skip (true -> emit no door plugin at "
        "all, for an opening a `window` fills or a deliberately bare gap)",
    )
    ap.add_argument(
        "--ceiling-h", type=float, default=2.5, help="wall/ceiling height in metres (default 2.5)"
    )
    ap.add_argument(
        "--opening-h",
        type=float,
        default=2.0,
        help="door-opening height in metres; the wall above it becomes a lintel up to the "
        "ceiling (default 2.0; a door's own height_m overrides this)",
    )
    ap.add_argument(
        "--wall-thickness", type=float, default=0.1, help="wall thickness in metres (default 0.1)"
    )
    ap.add_argument(
        "--ceiling",
        action="store_true",
        help="roof the building: a concrete slab over the whole plan, soffit at --ceiling-h. Off by "
        "default (an open plan reads better from above); the core `ceiling` plugin takes it back "
        "off at run time (`- ceiling: {enabled: false, above_z: ...}`)",
    )
    ap.add_argument(
        "--bake-config",
        help="scene.yaml giving the bake its look/collision/lighting (default: the scene dir's own "
        "scene.yaml if it has one, else the shared floorplan.scene.yaml)",
    )
    args = ap.parse_args(argv)

    floorplan = json.loads(Path(args.floorplan).read_text(encoding="utf-8"))
    markers_map = (
        json.loads(Path(args.markers_map).read_text(encoding="utf-8")) if args.markers_map else {}
    )
    markers_map = {str(k): v for k, v in markers_map.items()}
    doors_map = (
        json.loads(Path(args.doors_map).read_text(encoding="utf-8")) if args.doors_map else {}
    )
    doors_map = {str(k): v for k, v in doors_map.items()}

    out = generate(
        floorplan,
        Path(args.out_dir),
        args.scene_name,
        Path(args.world_out),
        markers_map,
        args.ceiling_h,
        args.wall_thickness,
        args.opening_h,
        doors_map,
        ceiling=args.ceiling,
        bake_config=Path(args.bake_config) if args.bake_config else None,
    )
    print(f"wrote world {out}")


if __name__ == "__main__":
    main()
