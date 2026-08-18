"""Search a fixed world for sensor placements that reach a coverage target.

Reached as ``roqsim sensors coverage`` (the first line above is what ``roqsim sensors --help`` prints).

Subcommands:

* ``estimate`` -- compile a world, sample points, evaluate a set of placements, and write the report
  (JSON) + a render. This is the inner call of the propose -> evaluate -> refine loop.
* ``catalog`` -- print the sensor catalog (types, FOV, cost, mount constraints) as JSON.
* ``greedy`` -- a deterministic submodular max-coverage baseline over candidate mount poses; a non-LLM
  sanity check and warm-start.

Run with the root ``.venv`` and ``MUJOCO_GL=egl`` for headless rendering. Example::

    roqsim sensors coverage estimate \\
        --world .../depot.xml --placements p.json --target k=1,frac=0.9 --out run/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import catalog as catalog_mod
from .adapters import build_fov
from .engine import coverage
from .report import build_report

# -- world loading -----------------------------------------------------------------------------------


def load_world(world: str):
    """Compile ``world`` (an MJCF path, or an roqsim world YAML / package ref) -> (model, data)."""
    import mujoco

    p = Path(world)
    if p.suffix.lower() in (".xml", ".mjcf") and p.exists():
        model = mujoco.MjModel.from_xml_path(str(p))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    # A world YAML / package ref: build through the roqsim engine so plugins (floorplan, spawns)
    # contribute their geometry, then read the compiled model/data back off the context.
    try:
        from roqsim.config import load_config
        from roqsim.engine import Engine
        from roqsim.world import resolve_world_yaml_ref
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"cannot load world {world!r}: {exc}") from exc
    # A `<package>:<world>` ref (the same form `extends:` accepts) -> resolve to its YAML path first;
    # load_config only takes paths/built-in names, so without this a ref like `pkg:world` is mistaken
    # for a filename and errors.
    ref = resolve_world_yaml_ref(world) if (not p.exists() and ":" in world) else None
    cfg = load_config(ref or (str(p) if p.exists() else world))
    engine = Engine(cfg)
    engine.setup()
    ctx = engine.ctx
    mujoco.mj_forward(ctx.model, ctx.data)
    return ctx.model, ctx.data


# -- shared helpers ----------------------------------------------------------------------------------


def parse_target(text: str | None) -> dict:
    """Parse ``k=1,frac=0.95`` -> {'metric','k','value'}. Empty -> no target."""
    if not text:
        return {}
    parts = dict(kv.split("=", 1) for kv in text.split(",") if "=" in kv)
    return {
        "metric": "fraction_covered",
        "k": int(parts.get("k", 1)),
        "value": float(parts.get("frac", parts.get("value", 1.0))),
    }


def _read_placements(path: str) -> list[dict]:
    obj = json.loads(Path(path).read_text())
    placements = obj["placements"] if isinstance(obj, dict) else obj
    if not isinstance(placements, list):
        raise SystemExit(f"{path}: expected a list of placements or {{'placements': [...]}}")
    return placements


def _build_samples(model, data, args):
    from . import sampling

    points_list = []
    labels_list = []
    names: list[str] = []
    want = args.sample
    if want in ("volume", "both"):
        heights = tuple(float(h) for h in args.heights)
        vol = sampling.room_volume_points(model, data, resolution=args.resolution, heights=heights)
        if len(vol):
            points_list.append(vol)
            labels_list.append(np.full(len(vol), -1))
    if want in ("objects", "both"):
        surf, lab, names = sampling.object_surface_points(model, data, per_object=args.per_object)
        if len(surf):
            points_list.append(surf)
            labels_list.append(lab)
    if not points_list:
        raise SystemExit("no sample points produced -- check --sample / --heights / the world")
    points = np.vstack(points_list)
    labels = np.concatenate(labels_list)
    return points, labels, names


def _resolve_regions(args):
    """Load + subset the regions named on the CLI, or ``None`` when ``--regions`` was not given."""
    if not getattr(args, "regions", None):
        return None
    from . import regions as regionsmod

    regs = regionsmod.load_regions(args.regions)
    if getattr(args, "region_names", None):
        regs = regionsmod.select(regs, args.region_names)
    if not regs:
        raise SystemExit(f"--regions {args.regions!r} produced no regions")
    return regs


def _apply_restrict(points, labels, regions):
    """Filter sample points/labels to the union of ``regions`` (the ``--restrict`` footprint)."""
    from . import regions as regionsmod

    mask = regionsmod.union_mask(points, regions)
    if not mask.any():
        raise SystemExit("--restrict: no sample points fall inside the given regions")
    return points[mask], labels[mask]


def _per_region(result, regions):
    """Per-region coverage rows for the report, or ``None`` when no regions were given."""
    if not regions:
        return None
    from . import regions as regionsmod

    return regionsmod.per_region_coverage(result.points, result.counts, regions)


def _print_per_region(per_region) -> None:
    for row in per_region or []:
        print(
            f"  region {row['name']!r}: n={row['n_points']} "
            f"k1={row['fraction_covered_k1']:.3f} k2={row['fraction_covered_k2']:.3f}"
        )


def _build_fovs(model, data, placements):
    fovs = []
    for i, prop in enumerate(placements):
        placed = catalog_mod.placed_from_proposal(prop, index=i)
        fovs.append(build_fov(model, data, placed))
    return fovs


def _render_outputs(
    model, data, result, out_dir: Path, mode: str, palette: str = "coverage"
) -> list[str]:
    from . import viz

    written = []
    if mode in ("3d", "both"):
        p = str(out_dir / "coverage_3d.png")
        written.append(viz.render_coverage_3d(model, data, result, p, palette=palette))
    if mode in ("2d", "both"):
        p = str(out_dir / "heatmap_2d.png")
        written.append(viz.render_heatmap_2d(result, p, palette=palette))
    return written


# -- subcommands -------------------------------------------------------------------------------------


def cmd_catalog(args) -> int:
    print(json.dumps(catalog_mod.catalog_as_dict(), indent=2))
    return 0


def cmd_estimate(args) -> int:
    model, data = load_world(args.world)
    points, labels, names = _build_samples(model, data, args)
    regions = _resolve_regions(args)
    if regions and args.restrict:
        points, labels = _apply_restrict(points, labels, regions)
    placements = _read_placements(args.placements)
    fovs = _build_fovs(model, data, placements)
    result = coverage(model, data, fovs, points, labels=labels, label_names=names)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = parse_target(args.target)
    per_region = _per_region(result, regions)
    report = build_report(
        result,
        world=args.world,
        target=target,
        placements=placements,
        gap_resolution=args.resolution,
        per_region=per_region,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _print_per_region(per_region)

    rendered = []
    if args.render != "none":
        rendered = _render_outputs(model, data, result, out_dir, args.render, args.palette)

    a = report["achieved"]
    met = report["target_met"]
    print(
        f"COVERAGE_OK world={args.world} points={a['n_points']} sensors={len(fovs)} "
        f"k1={a['fraction_covered_k1']:.3f} k2={a['fraction_covered_k2']:.3f} "
        f"mean={a['mean_count']:.2f}"
        + (f" target_met={met}" if target else "")
        + f" -> {out_dir / 'report.json'}"
        + (f" + {', '.join(Path(r).name for r in rendered)}" if rendered else "")
    )
    return 0


def cmd_greedy(args) -> int:
    from .optimize import generate_candidates, greedy_baseline

    model, data = load_world(args.world)
    points, labels, names = _build_samples(model, data, args)
    regions = _resolve_regions(args)
    if regions and args.restrict:
        points, labels = _apply_restrict(points, labels, regions)
    if args.candidates:
        candidates = _read_placements(args.candidates)
    else:
        candidates = generate_candidates(
            model, data, types=args.types.split(","), spacing=args.spacing, z=args.mount_z
        )
        if regions:
            # Only propose mounts whose (x, y) sits above a target region -- a sensor over room 5 does
            # not help cover rooms 1 & 2. Footprint (xy) test: the region z-band gates sample points,
            # not the ceiling the mount hangs from.
            import numpy as _np

            from . import regions as regionsmod

            keep = regionsmod.footprint_mask(_np.array([c["pos"] for c in candidates]), regions)
            candidates = [c for c, k in zip(candidates, keep, strict=True) if k]
        print(f"generated {len(candidates)} candidate mounts", file=sys.stderr)
    if not candidates:
        raise SystemExit(
            "no candidate mounts to search (regions may not overlap any room footprint)"
        )

    target = parse_target(args.target)
    chosen, result = greedy_baseline(
        model,
        data,
        candidates,
        points,
        labels=labels,
        label_names=names,
        target_frac=float(target.get("value", 0.95)),
        max_sensors=args.max_sensors,
        k=int(target.get("k", 1)),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_region = _per_region(result, regions)
    report = build_report(
        result,
        world=args.world,
        target=target,
        placements=chosen,
        gap_resolution=args.resolution,
        per_region=per_region,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "placements.json").write_text(json.dumps({"placements": chosen}, indent=2))
    if args.render != "none":
        _render_outputs(model, data, result, out_dir, args.render, args.palette)
    a = report["achieved"]
    k = int(target.get("k", 1))
    frac = a.get(f"fraction_covered_k{k}", a["fraction_covered_k1"])
    print(
        f"GREEDY_OK chose={len(chosen)} sensors k{k}={frac:.3f} target_met={report['target_met']} "
        f"-> {out_dir / 'placements.json'}"
    )
    _print_per_region(per_region)
    return 0


def _add_common_sampling(sp):
    sp.add_argument("--world", required=True, help="MJCF path, world YAML, or roqsim world ref")
    sp.add_argument("--out", required=True, help="output directory")
    sp.add_argument("--sample", choices=("volume", "objects", "both"), default="both")
    sp.add_argument(
        "--resolution", type=float, default=0.25, help="volume grid / gap-cluster resolution [m]"
    )
    sp.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.3, 1.0, 1.7],
        help="volume sample heights [m]",
    )
    sp.add_argument("--per-object", type=int, default=64, help="max surface points per object")
    sp.add_argument("--target", default=None, help="e.g. 'k=1,frac=0.95'")
    sp.add_argument(
        "--regions",
        default=None,
        help="JSON of named regions (polygons/bboxes) or a scene floorplan.json -> per-region coverage",
    )
    sp.add_argument(
        "--region-names", default=None, help="comma-separated subset of --regions to keep"
    )
    sp.add_argument(
        "--restrict",
        action="store_true",
        help="restrict sampling (and the greedy objective) to the union of the regions",
    )
    sp.add_argument("--render", choices=("3d", "2d", "both", "none"), default="3d")
    sp.add_argument(
        "--palette",
        choices=("coverage", "density"),
        default="coverage",
        help="colour encoding: 'coverage' (red 0->green many) or 'density' (light 0->dark many)",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("catalog", help="print the sensor catalog as JSON")
    sp.set_defaults(func=cmd_catalog)

    sp = sub.add_parser(
        "estimate", help="evaluate placements against a world -> report.json + render"
    )
    _add_common_sampling(sp)
    sp.add_argument(
        "--placements", required=True, help="JSON: list of placements or {'placements': [...]}"
    )
    sp.set_defaults(func=cmd_estimate)

    sp = sub.add_parser("greedy", help="deterministic max-coverage baseline over candidate mounts")
    _add_common_sampling(sp)
    sp.add_argument(
        "--candidates", default=None, help="JSON candidate poses; omit to auto-generate"
    )
    sp.add_argument(
        "--types", default="livox_mid360,oakd_camera", help="auto-candidate sensor types"
    )
    sp.add_argument("--spacing", type=float, default=3.0, help="auto-candidate grid spacing [m]")
    sp.add_argument("--mount-z", type=float, default=3.0, help="auto-candidate mount height [m]")
    sp.add_argument("--max-sensors", type=int, default=10, help="stop after this many sensors")
    sp.set_defaults(func=cmd_greedy)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
