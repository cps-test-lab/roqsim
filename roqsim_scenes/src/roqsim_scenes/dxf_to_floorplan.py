# SPDX-License-Identifier: Apache-2.0
"""Convert a 2D CAD floorplan drawing (DXF) into a floorplan JSON.

The output is the exact floorplan dict consumed by
``roqsim_scene_builder.sketch_floorplan_by_human`` (as ``initial=``) and by
``roqsim scenes floorplan-to-world`` (``--floorplan``): metres, y-up, 2 decimals,
``{comment, rooms, lines[{id, x0_m, y0_m, x1_m, y1_m}], doors, markers}``. Write it as the scene's
``floorplan.json``.

Only straight wall geometry is understood -- ``LINE`` and straight ``LWPOLYLINE``
segments. Anything that would silently lose a wall (an arc, spline, circle, or a
polyline bulge) is a hard error, not a silent drop: a floorplan with a quietly
missing wall is worse than a loud failure. Rooms and doors are intentionally left
empty here -- they are named/placed by the human in the 2D sketch window, which is
where that intent belongs.

This is a dependency-free reader: a DXF is a flat stream of (group-code, value)
line pairs, and the subset we need (LINE / LWPOLYLINE in the ENTITIES section,
``$INSUNITS`` in HEADER) is trivial to walk directly, so we avoid pulling in
``ezdxf``. If a real drawing ever needs arcs or splines, switch this reader to
``ezdxf`` rather than approximating them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

# $INSUNITS code -> metres-per-unit. The values we can convert unambiguously;
# 0 (unitless) and anything unlisted force an explicit --scale.
_INSUNITS_TO_M = {
    1: 0.0254,  # inches
    2: 0.3048,  # feet
    4: 0.001,  # millimetres
    5: 0.01,  # centimetres
    6: 1.0,  # metres
}

# Entity types we deliberately refuse rather than approximate -- each can carry a
# wall we would otherwise drop or straighten. POINT is the only silently-ignored
# type (it is the sketch origin marker, not geometry).
_CURVED_ENTITIES = {"ARC", "CIRCLE", "SPLINE", "ELLIPSE", "POLYLINE"}


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float


def _read_pairs(path: str) -> list[tuple[int, str]]:
    """Read a DXF as a list of (group_code, value) pairs.

    DXF is line-oriented: an integer group code on one line, its value on the
    next. We tolerate CRLF and stray blank lines.
    """
    pairs: list[tuple[int, str]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip("\r\n") for ln in fh]
    i = 0
    n = len(lines)
    while i + 1 < n:
        code_raw = lines[i].strip()
        value = lines[i + 1]
        i += 2
        if code_raw == "":
            i -= 1  # resync on a lone blank line
            continue
        try:
            code = int(code_raw)
        except ValueError as exc:
            raise ValueError(
                f"malformed DXF: expected an integer group code, got {code_raw!r}"
            ) from exc
        pairs.append((code, value))
    return pairs


def _read_insunits(pairs: list[tuple[int, str]]) -> int | None:
    """Return the $INSUNITS value from the HEADER section, or None if absent."""
    for idx, (code, value) in enumerate(pairs):
        if code == 9 and value.strip() == "$INSUNITS":
            # The variable's value follows on the next 70-code pair.
            for code2, value2 in pairs[idx + 1 : idx + 3]:
                if code2 == 70:
                    return int(float(value2))
    return None


def _entities_slice(pairs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Return the pairs inside the ENTITIES section (exclusive of its ENDSEC)."""
    out: list[tuple[int, str]] = []
    in_entities = False
    for i, (code, value) in enumerate(pairs):
        v = value.strip()
        if code == 2 and v == "ENTITIES" and i > 0 and pairs[i - 1] == (0, "SECTION"):
            in_entities = True
            continue
        if in_entities and code == 0 and v == "ENDSEC":
            break
        if in_entities:
            out.append((code, value))
    return out


def _parse_entities(ent: list[tuple[int, str]]) -> list[Segment]:
    """Turn the ENTITIES pairs into straight wall segments, failing loud on curves."""
    segments: list[Segment] = []

    # Split into per-entity blocks: a 0-code starts a new entity.
    blocks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for code, value in ent:
        if code == 0:
            if cur:
                blocks.append(cur)
            cur = [(code, value)]
        elif cur:
            cur.append((code, value))
    if cur:
        blocks.append(cur)

    for block in blocks:
        etype = block[0][1].strip()
        codes: dict[int, list[str]] = {}
        for code, value in block[1:]:
            codes.setdefault(code, []).append(value)

        if etype == "LINE":
            segments.append(
                Segment(
                    float(codes[10][0]),
                    float(codes[20][0]),
                    float(codes[11][0]),
                    float(codes[21][0]),
                )
            )
        elif etype == "LWPOLYLINE":
            if 42 in codes:
                raise ValueError(
                    "LWPOLYLINE has a bulge (group code 42): this is an arc segment "
                    "that would be silently flattened. Straighten it in the CAD tool "
                    "or re-implement this reader on top of ezdxf."
                )
            xs = [float(v) for v in codes.get(10, [])]
            ys = [float(v) for v in codes.get(20, [])]
            if len(xs) != len(ys):
                raise ValueError("LWPOLYLINE has mismatched 10/20 vertex coordinates")
            closed = bool(int(float(codes.get(70, ["0"])[0])) & 1)
            for j in range(len(xs) - 1):
                segments.append(Segment(xs[j], ys[j], xs[j + 1], ys[j + 1]))
            if closed and len(xs) >= 2:
                segments.append(Segment(xs[-1], ys[-1], xs[0], ys[0]))
        elif etype == "POINT":
            continue  # sketch origin marker, not a wall
        elif etype in _CURVED_ENTITIES:
            raise ValueError(
                f"DXF contains a {etype} entity -- curved/legacy geometry this reader "
                "does not straighten (a wall would be dropped or approximated). "
                "Straighten it in the CAD tool, or switch this reader to ezdxf."
            )
        # Unknown non-geometric entities (TEXT, DIMENSION, ...) are ignored.

    return segments


def _transform(
    segments: list[Segment],
    scale: float,
    recenter: bool,
    snap_tol: float,
    dedup: bool,
) -> list[Segment]:
    """Scale to metres, optionally recentre to (0,0), snap, drop degenerate, dedup."""
    scaled = [Segment(s.x0 * scale, s.y0 * scale, s.x1 * scale, s.y1 * scale) for s in segments]
    if not scaled:
        return scaled

    if recenter:
        min_x = min(min(s.x0, s.x1) for s in scaled)
        min_y = min(min(s.y0, s.y1) for s in scaled)
        scaled = [Segment(s.x0 - min_x, s.y0 - min_y, s.x1 - min_x, s.y1 - min_y) for s in scaled]

    def snap(v: float) -> float:
        if snap_tol > 0:
            v = round(v / snap_tol) * snap_tol
        return round(v, 2)

    out: list[Segment] = []
    seen: set[tuple[float, float, float, float]] = set()
    for s in scaled:
        seg = Segment(snap(s.x0), snap(s.y0), snap(s.x1), snap(s.y1))
        if seg.x0 == seg.x1 and seg.y0 == seg.y1:
            continue  # zero-length after snapping
        if dedup:
            a = (seg.x0, seg.y0)
            b = (seg.x1, seg.y1)
            key = (*a, *b) if a <= b else (*b, *a)  # undirected
            if key in seen:
                continue
            seen.add(key)
        out.append(seg)
    return out


def dxf_to_sketch(
    dxf_path: str,
    scale: float | None = None,
    recenter: bool = True,
    snap_tol: float = 0.02,
    dedup: bool = True,
) -> tuple[dict, dict]:
    """Convert a DXF file to a floorplan sketch dict.

    Returns ``(sketch, info)`` where ``info`` carries a human summary
    (entity counts, unit/scale used, bbox in metres).
    """
    pairs = _read_pairs(dxf_path)
    insunits = _read_insunits(pairs)

    if scale is None:
        if insunits in _INSUNITS_TO_M:
            scale = _INSUNITS_TO_M[insunits]
        else:
            raise ValueError(
                f"DXF $INSUNITS={insunits!r} is unitless/unknown -- pass --scale "
                "(metres per drawing unit) explicitly."
            )

    raw = _parse_entities(_entities_slice(pairs))
    segs = _transform(raw, scale, recenter, snap_tol, dedup)
    if not segs:
        raise ValueError(
            f"no wall segments found in {dxf_path} -- the DXF has no LINE/LWPOLYLINE "
            "geometry to convert."
        )

    lines = [
        {"id": i + 1, "x0_m": s.x0, "y0_m": s.y0, "x1_m": s.x1, "y1_m": s.y1}
        for i, s in enumerate(segs)
    ]
    sketch = {"comment": "", "rooms": [], "lines": lines, "doors": [], "markers": []}

    xs = [c for s in segs for c in (s.x0, s.x1)]
    ys = [c for s in segs for c in (s.y0, s.y1)]
    info = {
        "insunits": insunits,
        "scale_m_per_unit": scale,
        "raw_segments": len(raw),
        "final_lines": len(segs),
        "bbox_m": (
            round(min(xs), 2),
            round(min(ys), 2),
            round(max(xs), 2),
            round(max(ys), 2),
        ),
    }
    return sketch, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a 2D DXF floorplan drawing into a floorplan JSON "
        "(the input to the scene-builder sketch window and `roqsim scenes floorplan-to-world`).",
    )
    ap.add_argument("--dxf", required=True, help="input DXF file")
    ap.add_argument(
        "--out",
        required=True,
        help="output floorplan JSON path (e.g. scenes/<name>/floorplan.json)",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="metres per drawing unit; overrides $INSUNITS auto-detection",
    )
    ap.add_argument(
        "--no-recenter",
        dest="recenter",
        action="store_false",
        help="keep the drawing's own coordinates instead of moving the bbox min to (0,0)",
    )
    ap.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        help="keep coincident duplicate segments (e.g. a boundary LWPOLYLINE that "
        "overlaps individual LINEs)",
    )
    ap.add_argument(
        "--snap-tol",
        type=float,
        default=0.02,
        help="snap endpoints to this grid in metres to weld near-coincident corners "
        "(0 disables; default 0.02)",
    )
    args = ap.parse_args(argv)

    sketch, info = dxf_to_sketch(
        args.dxf,
        scale=args.scale,
        recenter=args.recenter,
        snap_tol=args.snap_tol,
        dedup=args.dedup,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(sketch, fh, indent=2)
        fh.write("\n")

    bx0, by0, bx1, by1 = info["bbox_m"]
    print(
        f"{args.dxf}: $INSUNITS={info['insunits']} -> "
        f"{info['scale_m_per_unit']} m/unit | "
        f"{info['raw_segments']} raw segments -> {info['final_lines']} lines | "
        f"bbox {bx1 - bx0:.2f} x {by1 - by0:.2f} m "
        f"[({bx0},{by0})..({bx1},{by1})]\n"
        f"wrote {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
