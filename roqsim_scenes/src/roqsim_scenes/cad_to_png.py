# SPDX-License-Identifier: Apache-2.0
"""Render a 2D CAD drawing (DWG or DXF) to a PNG preview.

This is the *look at it* half of the CAD import path, next to
:mod:`roqsim_scenes.dxf_to_floorplan` (the *convert it* half). A building's floorplan
arrives as a 40-layer architectural drawing full of dimensions, furniture blocks
and title-block text; before any of it can be turned into a floorplan sketch you
have to see which layers actually carry the walls. ``--list-layers`` names them
and ``--layers`` renders only those, so the same tool answers both "what is in
this drawing?" and "does my wall layer selection look right?".

Rendering is done by ``ezdxf``'s drawing add-on (matplotlib backend), which
understands the whole entity set a real drawing uses -- blocks, arcs, hatches,
dimensions, text. That is the opposite trade-off from ``dxf_to_floorplan``, which
stays dependency-free and refuses curves: a preview may approximate geometry (it
is thrown away after you look at it), a floorplan may not.

DWG is a proprietary binary format that ``ezdxf`` cannot read, so a DWG input is
first converted to DXF by an external tool -- LibreDWG's ``dwg2dxf`` or the ODA
File Converter. Neither is bundled. If no backend is installed we fail with the
install options rather than quietly rendering something else (a DXF of the same
name sitting next to the DWG is a *different export*, possibly of a different
revision, and using it silently would misreport what was drawn).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

# Colour policies exposed on the CLI. Kept to two on purpose: the CAD layer colours
# as drawn, or flat monochrome for a legible print of the geometry alone.
_COLOR_POLICIES = ("color", "mono")

_DWG_HINT = (
    "DWG is a proprietary binary format; rendering it needs an external DWG->DXF "
    "converter on PATH. Install one of:\n"
    "  * LibreDWG (free): https://github.com/LibreDWG/libredwg -- provides `dwg2dxf`\n"
    "  * ODA File Converter (proprietary, free download): "
    "https://www.opendesign.com/guestfiles/oda_file_converter -- provides "
    "`ODAFileConverter`\n"
    "Or export a DXF from the CAD tool and pass that instead."
)


def _require_ezdxf():
    """Import ezdxf, turning the ImportError into an actionable message."""
    try:
        import ezdxf  # noqa: F401
        from ezdxf import bbox, recover
        from ezdxf.addons.drawing import config as draw_config
        from ezdxf.addons.drawing import matplotlib as draw_matplotlib
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "rendering a CAD drawing needs the optional preview dependencies: "
            "pip install 'roqsim_scenes[preview]'  (ezdxf + matplotlib)"
        ) from exc
    return bbox, recover, draw_config, draw_matplotlib


# --------------------------------------------------------------------------- DWG


def find_dwg_backend() -> tuple[str, str] | None:
    """Return ``(kind, executable)`` of the first available DWG->DXF converter."""
    exe = shutil.which("dwg2dxf")
    if exe:
        return ("libredwg", exe)
    for name in ("ODAFileConverter", "ODAFileConverter.exe", "TeighaFileConverter"):
        exe = shutil.which(name)
        if exe:
            return ("oda", exe)
    return None


def dwg_to_dxf(dwg_path: str, out_dxf: str) -> str:
    """Convert a DWG to DXF using whichever external converter is installed.

    Returns the DXF path. Raises if no backend exists or the conversion produced
    nothing -- a zero-byte DXF renders as an empty image, which looks like a
    drawing with nothing on the selected layers.
    """
    backend = find_dwg_backend()
    if backend is None:
        raise RuntimeError(f"cannot read {dwg_path}: no DWG converter found.\n{_DWG_HINT}")
    kind, exe = backend

    if kind == "libredwg":
        cmd = [exe, "-o", out_dxf, dwg_path]
    else:
        # ODAFileConverter only does directory -> directory, and names the output
        # after the input stem, so it gets private in/out dirs we then rename from.
        in_dir = tempfile.mkdtemp(prefix="oda-in-")
        out_dir = tempfile.mkdtemp(prefix="oda-out-")
        staged = os.path.join(in_dir, os.path.basename(dwg_path))
        shutil.copy2(dwg_path, staged)
        # <in> <out> <output version> <output type> <recurse> <audit>
        cmd = [exe, in_dir, out_dir, "ACAD2018", "DXF", "0", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        produced = [f for f in os.listdir(out_dir) if f.lower().endswith(".dxf")]
        if not produced:
            raise RuntimeError(
                f"{os.path.basename(exe)} produced no DXF for {dwg_path} "
                f"(exit {proc.returncode}).\n{proc.stdout}\n{proc.stderr}\n"
                "Note: ODAFileConverter is a Qt GUI application -- on a headless "
                "machine run it under `xvfb-run`."
            )
        shutil.move(os.path.join(out_dir, produced[0]), out_dxf)
        return out_dxf

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_dxf) or os.path.getsize(out_dxf) == 0:
        raise RuntimeError(
            f"{os.path.basename(exe)} failed to convert {dwg_path} "
            f"(exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    return out_dxf


def to_dxf(path: str, workdir: str) -> str:
    """Return a DXF path for ``path``, converting from DWG into ``workdir`` if needed."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dxf":
        return path
    if ext == ".dwg":
        out = os.path.join(workdir, os.path.splitext(os.path.basename(path))[0] + ".dxf")
        return dwg_to_dxf(path, out)
    raise ValueError(f"unsupported CAD input {path!r}: expected a .dwg or .dxf file")


# --------------------------------------------------------------------------- read


def _read(dxf_path: str):
    """Load a DXF with ezdxf's recovery reader, failing on unrecoverable damage.

    Real CAD exports routinely need hundreds of harmless fixes (unused handles,
    invalid extrusions); those are reported, not fatal. Auditor *errors* are, since
    they mean entities were dropped.
    """
    _, recover, _, _ = _require_ezdxf()
    doc, auditor = recover.readfile(dxf_path)
    if auditor.has_errors:
        detail = "\n".join(f"  {e.code}: {e.message}" for e in auditor.errors[:10])
        raise RuntimeError(
            f"{dxf_path} has {len(auditor.errors)} unrecoverable DXF error(s); "
            f"entities may be missing from the render:\n{detail}"
        )
    return doc, auditor


def _layout(doc, name: str):
    """Return the named layout; 'model'/'modelspace' select the model space."""
    if name.lower() in ("model", "modelspace"):
        return doc.modelspace()
    available = list(doc.layout_names())
    if name not in available:
        raise ValueError(f"no layout {name!r} in this drawing; available: {available}")
    return doc.layout(name)


def layer_report(cad_path: str) -> list[tuple[str, int, bool]]:
    """Return ``(layer, entity_count_in_modelspace, is_off)`` sorted by count desc.

    Layers with no model-space entities are included with count 0 -- an empty
    wall layer is exactly the surprise worth seeing.
    """
    with tempfile.TemporaryDirectory(prefix="roqsim-cad-") as tmp:
        doc, _ = _read(to_dxf(cad_path, tmp))
        counts = Counter(e.dxf.layer for e in doc.modelspace())
        rows = [
            (layer.dxf.name, counts.get(layer.dxf.name, 0), layer.is_off()) for layer in doc.layers
        ]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


# ------------------------------------------------------------------------ render


def _make_filter(
    include: list[str] | None, exclude: list[str] | None, skip_off: bool, off_layers: set[str]
):
    """Build the ezdxf ``filter_func``: which entities take part in the render.

    Patterns are shell globs matched case-insensitively against the layer name,
    because CAD layer names are conventionally structured (``A-WALL-FULL``,
    ``A-WALL-PART``) and selecting a family is the common case.
    """

    def keep(entity) -> bool:
        layer = str(getattr(entity.dxf, "layer", ""))
        low = layer.lower()
        if skip_off and layer in off_layers:
            return False
        if include and not any(fnmatch.fnmatch(low, p.lower()) for p in include):
            return False
        if exclude and any(fnmatch.fnmatch(low, p.lower()) for p in exclude):
            return False
        return True

    return keep


def render(
    cad_path: str,
    out_png: str,
    layout: str = "model",
    layers: list[str] | None = None,
    exclude_layers: list[str] | None = None,
    width_px: int = 2000,
    dpi: int = 200,
    colors: str = "color",
    background: str = "white",
    text: bool = True,
    hatch: bool = True,
    include_off_layers: bool = False,
    keep_dxf: str | None = None,
) -> dict:
    """Render ``cad_path`` (DWG or DXF) to ``out_png``. Returns a summary dict."""
    if colors not in _COLOR_POLICIES:
        raise ValueError(f"--colors must be one of {_COLOR_POLICIES}, got {colors!r}")

    bbox, _, draw_config, draw_matplotlib = _require_ezdxf()

    with tempfile.TemporaryDirectory(prefix="roqsim-cad-") as tmp:
        dxf_path = to_dxf(cad_path, tmp)
        if keep_dxf and dxf_path != cad_path:
            shutil.copy2(dxf_path, keep_dxf)
        doc, auditor = _read(dxf_path)
        lay = _layout(doc, layout)

        off_layers = {layer.dxf.name for layer in doc.layers if layer.is_off()}
        keep = _make_filter(layers, exclude_layers, not include_off_layers, off_layers)

        selected = [e for e in lay if keep(e)]
        if not selected:
            raise RuntimeError(
                f"nothing to draw: no entities in layout {layout!r} passed the layer "
                f"filter (include={layers}, exclude={exclude_layers}). "
                "Run with --list-layers to see the layer names."
            )

        # Size the figure from the drawing's own aspect ratio, so --width-px is the
        # real pixel width and no axis gets letterboxed.
        extents = bbox.extents(selected, fast=True)
        if not extents.has_data or extents.size.x <= 0 or extents.size.y <= 0:
            raise RuntimeError(
                f"the {len(selected)} selected entities have no 2D extent; "
                "nothing could be framed for rendering."
            )
        aspect = extents.size.y / extents.size.x
        size_inches = (width_px / dpi, max(width_px * aspect, 1.0) / dpi)

        cfg = draw_config.Configuration(
            color_policy=(
                draw_config.ColorPolicy.COLOR
                if colors == "color"
                else draw_config.ColorPolicy.MONOCHROME
            ),
            text_policy=(draw_config.TextPolicy.FILLING if text else draw_config.TextPolicy.IGNORE),
            hatch_policy=(
                draw_config.HatchPolicy.NORMAL if hatch else draw_config.HatchPolicy.IGNORE
            ),
            background_policy=(
                draw_config.BackgroundPolicy.WHITE
                if background == "white"
                else draw_config.BackgroundPolicy.BLACK
                if background == "black"
                else draw_config.BackgroundPolicy.OFF
            ),
        )

        out_dir = os.path.dirname(os.path.abspath(out_png))
        os.makedirs(out_dir, exist_ok=True)
        draw_matplotlib.qsave(
            lay,
            out_png,
            dpi=dpi,
            config=cfg,
            filter_func=keep,
            size_inches=size_inches,
        )

        return {
            "source": cad_path,
            "converted_from_dwg": dxf_path != cad_path,
            "dxf_version": doc.acad_release,
            "insunits": doc.header.get("$INSUNITS"),
            "layout": layout,
            "entities_total": sum(1 for _ in lay),
            "entities_drawn": len(selected),
            "layers_off_skipped": sorted(off_layers) if not include_off_layers else [],
            "audit_fixes": len(auditor.fixes),
            "extent": (
                round(extents.size.x, 3),
                round(extents.size.y, 3),
            ),
            "size_px": (width_px, int(round(width_px * aspect))),
            "out": out_png,
        }


# --------------------------------------------------------------------------- CLI


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render a 2D CAD drawing (DWG or DXF) to a PNG preview, "
        "optionally restricted to selected layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        "  roqsim-cad-to-png --in plan.dwg --list-layers\n"
        "  roqsim-cad-to-png --in plan.dwg --out plan.png\n"
        "  roqsim-cad-to-png --in plan.dxf --out walls.png --layers 'a-wall*' --no-text\n",
    )
    ap.add_argument("--in", dest="input", required=True, help="input .dwg or .dxf file")
    ap.add_argument("--out", help="output PNG path (not needed with --list-layers)")
    ap.add_argument(
        "--list-layers",
        action="store_true",
        help="print the drawing's layers with their model-space entity counts and exit",
    )
    ap.add_argument("--layout", default="model", help="layout to render (default: model space)")
    ap.add_argument(
        "--layers",
        help="comma-separated layer name globs to include (default: all), e.g. 'a-wall*,a-door*'",
    )
    ap.add_argument("--exclude-layers", help="comma-separated layer name globs to drop")
    ap.add_argument(
        "--include-off-layers",
        action="store_true",
        help="also draw layers the CAD file marks as off (they are skipped by default)",
    )
    ap.add_argument(
        "--width-px", type=int, default=2000, help="output width in pixels (default 2000)"
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="raster resolution for line widths and text (default 200)",
    )
    ap.add_argument(
        "--colors",
        choices=_COLOR_POLICIES,
        default="color",
        help="'color' keeps the CAD layer colours, 'mono' draws everything in the "
        "foreground colour (default: color)",
    )
    ap.add_argument(
        "--bg",
        dest="background",
        choices=("white", "black", "transparent"),
        default="white",
        help="background (default white)",
    )
    ap.add_argument("--no-text", dest="text", action="store_false", help="omit TEXT/MTEXT")
    ap.add_argument("--no-hatch", dest="hatch", action="store_false", help="omit HATCH fills")
    ap.add_argument(
        "--keep-dxf",
        metavar="PATH",
        help="also write the intermediate DXF produced from a DWG input "
        "(the input to roqsim-dxf-to-floorplan)",
    )
    args = ap.parse_args(argv)

    if args.list_layers:
        for name, count, is_off in layer_report(args.input):
            print(f"{count:7d}  {'off ' if is_off else '    '}{name}")
        return 0

    if not args.out:
        ap.error("--out is required (or use --list-layers)")

    info = render(
        args.input,
        args.out,
        layout=args.layout,
        layers=_split_csv(args.layers),
        exclude_layers=_split_csv(args.exclude_layers),
        width_px=args.width_px,
        dpi=args.dpi,
        colors=args.colors,
        background=args.background,
        text=args.text,
        hatch=args.hatch,
        include_off_layers=args.include_off_layers,
        keep_dxf=args.keep_dxf,
    )
    ex, ey = info["extent"]
    px, py = info["size_px"]
    print(
        f"{info['source']}: {info['dxf_version']}"
        f"{' (via DWG->DXF)' if info['converted_from_dwg'] else ''} | "
        f"$INSUNITS={info['insunits']} | layout {info['layout']} | "
        f"{info['entities_drawn']}/{info['entities_total']} entities drawn | "
        f"extent {ex} x {ey} drawing units | {px}x{py} px\n"
        f"wrote {info['out']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
