# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Bake a world's lighting with Blender's Cycles, once, into the world's own files.

    roqsim bake <world.yaml|pkg:world> [--device cpu|cuda|optix|hip|metal] [--jobs N]
                [--blender PATH] [--work DIR] [--dry-run] [--set key=value ...]

The world names how it is baked: its ``baked_lighting`` component (``quality``, ``texels_per_m``,
``samples``, ``denoise``, ``lights``, ``emitters``, ``exclude``, ``occluders``, ``max_mb``, ``path``).
This command loads the world exactly as ``roqsim sim`` does, takes from that component the world as it
loads (its export, lights and fingerprint, :mod:`roqsim.bake.analysis`), hands it to Blender
(:mod:`roqsim.bake.blender_bake`, ``--jobs`` processes side by side, largest first) and writes the bake
where the component reads it (:mod:`roqsim.bake.store`). What it changes is the bake only: the
device, the job count and Blender's path are not in it.

On the CPU (the default) a furnished room bakes in minutes with the ``default`` preset; ``quality:
high`` wants a GPU (``--device optix`` on NVIDIA RTX, ``cuda``, ``hip`` on AMD, ``metal`` on Apple). A
GPU type that is not there fails and names ``--device cpu``. A bake over the component's ``max_mb``
fails and names its largest maps: lower ``texels_per_m``, or raise ``max_mb`` on purpose. An entity
whose geometry and size are unchanged since the last bake reuses its lightmap UVs (the slowest step),
so a bake after a light or a setting changed path-traces again without laying out again.

Prints one line of JSON on stdout: the bake's path, its size, the times per step, the device.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from .. import exit_status
from . import fingerprint as fp
from .analysis import Analysis
from .compose import rasterize
from .settings import FORMAT_VERSION, SHELL_P95
from .store import (
    LUMA,
    EntityMaps,
    _down,
    bake_meta,
    encode_entity,
    read_entity,
    read_meta,
    size_on_disk,
)
from .store import write as write_bake

DEVICES = ("cpu", "cuda", "optix", "hip", "metal", "oneapi")
#: The Cycles threads per Blender process on the CPU (``--jobs`` defaults to the cores over this).
THREADS_PER_JOB = 4
#: Blender's major.minor the bake is written against.
BLENDER_MIN = (4, 2)


class BakeRunError(Exception):
    """A bake that cannot be made, with the next step in its message."""


# -- Blender ------------------------------------------------------------------------------------------


def find_blender(given: str | None = None) -> tuple[str, str]:
    """Blender's executable and version, or a :class:`BakeRunError` saying how to get one."""
    exe = given or os.environ.get("BLENDER") or shutil.which("blender")
    if not exe or not (Path(exe).is_file() or shutil.which(exe)):
        raise BakeRunError(
            "roqsim bake needs Blender 4.2 or later (Cycles bakes the light): install it from "
            "https://www.blender.org/download/ and put `blender` on PATH, or pass --blender PATH"
        )
    try:
        out = subprocess.run([exe, "-b", "--factory-startup", "--version"], capture_output=True, text=True,
                             timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired) as err:
        raise BakeRunError(f"{exe} does not run as Blender ({err}); pass --blender PATH") from None
    match = re.search(r"Blender (\d+)\.(\d+)(\.\d+)?", out)
    if not match:
        raise BakeRunError(f"{exe} did not report a Blender version; pass --blender PATH")
    version = tuple(int(v) for v in match.groups()[:2])
    if version < BLENDER_MIN:
        raise BakeRunError(f"{exe} is Blender {match.group(0)[8:]}; the bake needs 4.2 or later")
    return exe, match.group(0)[8:]


def blender_script() -> str:
    return str(resources.files("roqsim.bake").joinpath("blender_bake.py"))


# -- the work directory -----------------------------------------------------------------------------


def write_work(analysis: Analysis, work: Path) -> None:
    """What Blender reads (see :mod:`roqsim.bake.blender_bake`)."""
    export = analysis.export
    (work / "entities").mkdir(parents=True, exist_ok=True)
    (work / "textures").mkdir(exist_ok=True)
    materials = []
    for i, mat in enumerate(export.materials):
        entry = {"rgba": mat["rgba"]}
        if mat["texture"] >= 0:
            name = f"textures/tex{mat['texture']}.png"
            if not (work / name).exists():
                # MuJoCo's row 0 is v = 0, which Blender's image holds at the bottom: written as is, the
                # PNG's top row is v = 0, so the albedo UVs go to Blender with v turned (1 - v).
                Image.fromarray(export.textures[mat["texture"]]).save(work / name)
            entry["texture_file"] = name
        materials.append(entry)
    entities = {}
    for name, ent in export.entities.items():
        uv1 = ent.uv1.copy()
        uv1[..., 1] = 1.0 - uv1[..., 1]
        np.savez_compressed(work / "entities" / f"{name}.npz", pos=ent.pos.astype(np.float32),
                            nrm=ent.nrm.astype(np.float32), uv1=uv1.astype(np.float32),
                            mat=ent.mat, geom=ent.geom)
        entities[name] = {"size": ent.size, "class": ent.cls, "two_sided": ent.two_sided,
                          "triangles": int(len(ent.pos))}
    np.savez_compressed(work / "occluders.npz", pos=export.occluders.reshape(-1, 3).astype(np.float32))
    sources = analysis.light["emitters"]
    if export.emitters:
        pos, nrm, src = [], [], []
        index = {s["geom"]: i for i, s in enumerate(sources)}
        for g, (p, n) in export.emitters.items():
            rec = next(r for r in export.geoms if r.id == g)
            pos.append(p)
            nrm.append(n)
            src.append(np.full(len(p), index[rec.name]))
        np.savez_compressed(work / "emitters.npz", pos=np.concatenate(pos).astype(np.float32),
                            nrm=np.concatenate(nrm).astype(np.float32), source=np.concatenate(src))
    light = {"lights": analysis.light["lights"], "sky": analysis.light["sky"]}
    manifest = {"entities": entities, "materials": materials, "emitters": sources if export.emitters else [],
                "light": light, "settings": analysis.settings.as_record()}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1))


def plan_jobs(analysis: Analysis, jobs: int) -> list[list[str]]:
    """The entities split over *jobs* processes, largest first onto the least loaded (by an estimate
    of their cost: the passes grow with the texels and the sides baked, the UV layout with the
    triangles)."""
    cost = {n: (e.size / 512.0) ** 2 * (2 if e.two_sided else 1) + len(e.pos) / 20000.0
            for n, e in analysis.export.entities.items()}
    jobs = max(1, min(jobs, len(cost)))
    load = [0.0] * jobs
    parts: list[list[str]] = [[] for _ in range(jobs)]
    for name in sorted(cost, key=lambda n: -cost[n]):
        i = load.index(min(load))
        parts[i].append(name)
        load[i] += cost[name]
    return [p for p in parts if p]


def run_blender(exe: str, work: Path, parts: list[list[str]], device: str, threads: int,
                uv_from: Path | None, log) -> list[Path]:
    """Run the Blender processes side by side; their output directories."""
    script = blender_script()
    procs = []
    for i, part in enumerate(parts):
        out = work / f"out{i}"
        cmd = [exe, "-b", "--factory-startup", "--python", script, "--", "--work", str(work), "--out", str(out),
               "--entities", ",".join(part), "--device", device.upper(), "--threads", str(threads)]
        if uv_from is not None:
            cmd += ["--uv-from", str(uv_from)]
        logf = open(work / f"blender{i}.log", "w")  # noqa: SIM115 - closed below
        procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf, out, part))
    failed = []
    for proc, logf, out, part in procs:
        code = proc.wait()
        logf.close()
        if code != 0:
            failed.append((code, out, part))
    if failed:
        code, out, part = failed[0]
        text = (work / f"blender{procs.index(next(p for p in procs if p[2] == out))}.log").read_text(errors="replace")
        said = [ln[len("roqsim-bake: "):] for ln in text.splitlines() if ln.startswith("roqsim-bake: ")]
        if code == 3:
            raise BakeRunError(said[-1] if said else f"no {device} device; --device cpu bakes on the CPU")
        if code == 4:
            raise BakeRunError(said[-1] if said else "Open Image Denoise not found; `denoise: false`")
        tail = "\n".join(text.splitlines()[-15:])
        raise BakeRunError(f"Blender failed (exit {code}) baking {', '.join(part)}; its log ends:\n{tail}")
    return [p[2] for p in procs]


# -- from the passes to the bake -----------------------------------------------------------------------


def _load(dirs: list[Path], name: str, kind: str) -> np.ndarray | None:
    for d in dirs:
        p = d / f"{name}.{kind}.npy"
        if p.exists():
            return np.load(p).astype(np.float32)
    return None


def choose_sides(light, back, uv, size) -> tuple[np.ndarray, np.ndarray]:
    """Per triangle, whether its back sees more light (sampled at seven points inside it), and per
    texel, whether it takes the back's value."""
    w = np.random.default_rng(0).dirichlet([1, 1, 1], 7)
    q = np.einsum("sk,tkd->tsd", w, uv.astype(np.float64))
    ij = np.clip(np.floor(q * size).astype(int), 0, size - 1)
    front = (light[ij[..., 1], ij[..., 0]] @ LUMA).mean(1)
    rear = (back[ij[..., 1], ij[..., 0]] @ LUMA).mean(1)
    flip = rear > front
    ids = rasterize(uv, size)
    side = np.zeros((size, size), bool)
    side[ids >= 0] = flip[ids[ids >= 0]]
    return flip, side


def exposure_of(maps: dict[str, dict]) -> float:
    """The exposure that puts the 95th percentile of the shell's lit vertical texels at
    :data:`roqsim.bake.settings.SHELL_P95` (or of every entity's, in a world without a shell)."""
    values = []
    for name, m in maps.items():
        if m["class"] != "shell" and any(v["class"] == "shell" for v in maps.values()):
            continue
        n = m["normal"] * 2 - 1
        y = m["light"] @ LUMA
        vertical = (np.abs(n[..., 2]) < 0.3) & (np.linalg.norm(n, axis=-1) > 0.5) & (y > 0)
        values.append(y[vertical if vertical.any() else y > 0])
    v = np.concatenate(values) if values else np.zeros(0)
    if not len(v) or float(np.percentile(v, 95)) <= 0:
        raise BakeRunError("the bake is dark everywhere: the world's lights reach no baked surface")
    return SHELL_P95 / float(np.percentile(v, 95))


def finish(analysis: Analysis, dirs: list[Path]) -> tuple[dict[str, bytes], dict, dict]:
    """The bake's files, its ``bake.json`` entries per entity and the per-entity statistics."""
    export = analysis.export
    raw = {}
    for name, ent in export.entities.items():
        light = _load(dirs, name, "light")
        if light is None:
            raise BakeRunError(f"Blender wrote no light map for {name}")
        uv = _load(dirs, name, "uv2")
        m = {"class": ent.cls, "light": light, "direct": _load(dirs, name, "direct"),
             "ao": _load(dirs, name, "ao"), "normal": _load(dirs, name, "normal"), "uv": uv,
             "flip": np.zeros(len(uv), bool)}
        if ent.two_sided:
            back = _load(dirs, name, "light_back")
            flip, side = choose_sides(light, back, uv, light.shape[0])
            for k in ("light", "direct", "ao"):
                b = _load(dirs, name, f"{k}_back")
                m[k] = np.where(side[..., None], b, m[k])
            m["normal"] = np.where(side[..., None], 1.0 - m["normal"], m["normal"])
            m["flip"] = flip
        raw[name] = m
    exposure = exposure_of(raw)
    files, entries = {}, {}
    for name, m in raw.items():
        indirect = np.clip(m["light"] - m["direct"], 0, None) @ LUMA
        maps = EntityMaps(light=m["light"], indirect=_down(indirect, 2), ao=_down(m["ao"] @ LUMA, 2),
                          uv=m["uv"], geom=export.entities[name].geom, flip=m["flip"])
        f, entry = encode_entity(name, maps)
        entry["class"] = m["class"]
        entry["faces_turned"] = int(m["flip"].sum())
        files.update(f)
        entries[name] = entry
    return files, entries, {"exposure": exposure}


def reusable_uvs(analysis: Analysis, root: Path, into: Path) -> list[str]:
    """Write the old bake's lightmap UVs of every entity whose geometry and size are unchanged, as
    Blender's ``--uv-from`` reads them; the entities they are for."""
    meta = read_meta(root) if root.is_dir() else None
    if not meta or meta.get("format") != FORMAT_VERSION:
        return []
    old_geoms = meta.get("fingerprint", {}).get("geoms", {})
    cur_geoms = analysis.fingerprint["geoms"]
    reused = []
    into.mkdir(parents=True, exist_ok=True)
    for name, ent in analysis.export.entities.items():
        entry = meta.get("entities", {}).get(name)
        if not entry or entry.get("size") != ent.size:
            continue
        keys = {analysis.export.geoms[int(g)].key for g in np.unique(ent.geom)}
        if keys != {k for k, v in old_geoms.items() if name in v.get("entities", {})}:
            continue
        if not all(fp._close(cur_geoms[k]["entities"].get(name), old_geoms[k]["entities"].get(name)) for k in keys):
            continue
        try:
            maps = read_entity(root, name, entry)
        except Exception:  # noqa: BLE001 - an unreadable old bake is laid out afresh
            continue
        if len(maps.uv) != len(ent.pos):
            continue
        np.save(into / f"{name}.uv2.npy", maps.uv.astype(np.float32))
        reused.append(name)
    return reused


def bake(plugin, *, device: str = "cpu", jobs: int | None = None, blender: str | None = None,
         work: Path | None = None, keep_work: bool = False, log=None) -> dict:
    """Bake the world *plugin* (a set-up :class:`~roqsim.plugins.baked_lighting.BakedLightingPlugin`)
    was loaded into; returns the summary ``roqsim bake`` prints."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    analysis: Analysis = plugin.analysis
    root: Path = plugin.root
    if device not in DEVICES:
        raise BakeRunError(f"--device {device}: one of {', '.join(DEVICES)}")
    exe, version = find_blender(blender)
    times = {"analyse_s": round(getattr(plugin, "analysis_seconds", 0.0), 2)}
    if not analysis.export.entities:
        raise BakeRunError("the world has no static, drawn geometry to bake")
    # Cycles spreads a pass over a process's threads, and the passes are most of a bake: a few
    # processes of several threads each keep every core busy without one large light map waiting on
    # a single thread.
    jobs = jobs or max(1, (os.cpu_count() or 1) // THREADS_PER_JOB)
    if device != "cpu":
        jobs = 1  # one GPU, one process
    parts = plan_jobs(analysis, jobs)
    threads = max(1, (os.cpu_count() or 1) // len(parts))
    owned = work is None
    work = Path(tempfile.mkdtemp(prefix="roqsim-bake-")) if work is None else Path(work)
    work.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.perf_counter()
        write_work(analysis, work)
        reused = reusable_uvs(analysis, root, work / "uv_from")
        times["export_s"] = round(time.perf_counter() - t0, 2)
        log(f"roqsim bake: {len(analysis.export.entities)} light maps in {len(parts)} Blender process(es) "
            f"on the {device.upper()}, {analysis.settings.samples} samples"
            + (f"; reusing the UV layout of {len(reused)}" if reused else ""))
        t0 = time.perf_counter()
        dirs = run_blender(exe, work, parts, device, threads, work / "uv_from" if reused else None, log)
        times["blender_s"] = round(time.perf_counter() - t0, 2)
        per_entity = {}
        for d in dirs:
            for f in d.glob("times.*.json"):
                per_entity.update(json.loads(f.read_text())["entities"])
        t0 = time.perf_counter()
        files, entries, info = finish(analysis, dirs)
        times["encode_s"] = round(time.perf_counter() - t0, 2)
    finally:
        if owned and not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    total = sum(len(b) for b in files.values())
    cap = float(plugin.config.get("max_mb", 5.0)) * 1e6
    if total > cap:
        largest = sorted(((len(b), n) for n, b in files.items()), reverse=True)[:5]
        raise BakeRunError(
            f"the bake would take {total / 1e6:.1f} MB, over max_mb {cap / 1e6:g}: largest "
            + ", ".join(f"{n} {b / 1e6:.2f} MB" for b, n in largest)
            + ". Lower texels_per_m for those classes, or raise max_mb on purpose (every rebake adds "
            "its size to the history of the repository the world is committed in)"
        )
    for name, entry in entries.items():
        entry["times"] = per_entity.get(name, {})
    meta = bake_meta(analysis, entries, info["exposure"], {
        "baked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "blender": version,
        "jobs": len(parts),
        "times": times,
    })
    write_bake(root, meta, files)
    return {"path": str(root), "bytes": size_on_disk(root), "entities": len(entries), "device": device,
            "blender": version, "jobs": len(parts), "times": times, "uv_reused": reused,
            "digest": meta["digest"]}


def main(argv=None) -> int:
    from ..config import load_config
    from ..engine import Engine
    from ..override_options import add_override_options, overrides_from_options
    from ..plugins.baked_lighting import BAKING, BakedLightingPlugin
    from ..world import resolve_world_yaml_ref

    parser = argparse.ArgumentParser(
        prog="roqsim bake",
        description="Bake a world's lighting with Blender's Cycles into its baked_lighting component's "
        "bake (lighting, shadows, global illumination, ambient occlusion; once, by hand).",
        epilog=exit_status.epilog(exit_status.BAD_INPUT, note="1: the bake failed (no Blender, no such "
                                  "GPU, over max_mb, Blender's own error), with the next step."),
    )
    parser.add_argument("world", help="world YAML, or a package ref such as roqsim_scenes:depot")
    parser.add_argument("--device", default="cpu", choices=DEVICES, help="where Cycles runs (default: cpu)")
    parser.add_argument("--jobs", type=int, default=None, help="Blender processes on the CPU (default: the cores over 4)")
    parser.add_argument("--blender", default=None, help="Blender's executable (default: blender on PATH)")
    parser.add_argument("--work", type=Path, default=None, help="keep the intermediates here")
    parser.add_argument("--dry-run", action="store_true", help="say what would be baked, bake nothing")
    add_override_options(parser)
    args = parser.parse_args(argv)
    target = args.world
    if ":" in target and not Path(target).exists():
        target = resolve_world_yaml_ref(target) or target
    try:
        cfg = load_config(target, overrides_from_options(args))
    except Exception as err:  # noqa: BLE001 - the world's own error, said once
        return exit_status.fail("roqsim bake", err)
    engine = Engine(cfg, preview=True)
    engine.ctx.blackboard.set(BAKING, True)
    try:
        engine.setup()
        plugins = [p for p in engine.plugins if isinstance(p, BakedLightingPlugin)]
        if not plugins:
            raise BakeRunError(
                f"{target} has no baked_lighting component: add `- baked_lighting: {{}}` to its "
                "components, then bake it"
            )
        plugin = plugins[0]
        if args.dry_run:
            export = plugin.analysis.export
            print(json.dumps({
                "path": str(plugin.root),
                "entities": {n: {"class": e.cls, "size": e.size, "triangles": int(len(e.pos))}
                             for n, e in export.entities.items()},
                "lights": plugin.analysis.light["rule"],
                "emitters": [s["geom"] for s in plugin.analysis.light["emitters"]],
                "digest": fp.digest(plugin.analysis.fingerprint),
                "settings": plugin.analysis.settings.as_record(),
            }))
            return 0
        summary = bake(plugin, device=args.device, jobs=args.jobs, blender=args.blender, work=args.work,
                       keep_work=args.work is not None)
    except BakeRunError as err:
        print(f"roqsim bake: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # noqa: BLE001 - a plugin's or the world's error, said once
        return exit_status.fail("roqsim bake", err)
    finally:
        engine.shutdown()
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
