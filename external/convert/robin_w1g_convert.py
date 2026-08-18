"""Convert the Seyond Robin W1G CAD into MuJoCo meshes: cascadio (STEP->OBJ) + Blender processing.

The ``convert.script`` for the ``seyond_robin_w1g_meshes`` resource (see external/external_assets.yaml).
Runs in the external-resources tool venv (needs ``cascadio``) and shells out to Blender for the
reframe / decimation (needs Blender >= 4.0 on ``$BLENDER`` or ``PATH`` -- declared ``needs_blender``).

Contract (called by external/external_resources.py):
    --sources <robin-w1g.stp>
    --targets <body.obj>
    --blender <path>                # optional; else $BLENDER or 'blender'
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import which

HERE = Path(__file__).resolve().parent


def _resolve_blender(name: str) -> str:
    found = which(name) or (name if Path(name).exists() else None)
    if not found:
        sys.exit(f"Blender not found ({name!r}); install it or set BLENDER=/path/to/blender")
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    args = ap.parse_args()
    if len(args.sources) != 1 or len(args.targets) != 1:
        sys.exit("robin_w1g_convert: expected 1 source (stp) and 1 target (body)")

    step = str(Path(args.sources[0]).resolve())
    body_out = str(Path(args.targets[0]).resolve())
    Path(body_out).parent.mkdir(parents=True, exist_ok=True)

    import cascadio

    blender = _resolve_blender(args.blender)
    with tempfile.TemporaryDirectory() as td:
        robin_obj = str(Path(td) / "robin.obj")
        print(f"cascadio: {step} -> {robin_obj}")
        cascadio.step_to_obj(step, robin_obj, tol_linear=0.1, tol_angular=0.5, use_colors=True)
        cmd = [
            blender,
            "-b",
            "-P",
            str(HERE / "robin_w1g_blender.py"),
            "--",
            robin_obj,
            body_out,
        ]
        print("blender:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    _check_orientation(body_out)
    print(f"robin_w1g: wrote {body_out}")


def _check_orientation(body_obj: str) -> None:
    """Fail loudly if the exported housing is not in the expected sim frame (window +x, Z up).

    Blender's OBJ exporter can remap axes across versions/builds; a wrong export would load into
    MuJoCo rotated (window facing up or sideways) while the site/FOV (fixed in robin_w1g.xml) still
    point along +x -- the classic "mesh looks up, FOV to the side" bug. The reframed device is
    ~0.125 m deep (x, incl. connector), ~0.105 m wide (y), ~0.085 m tall (z), so a correct export is
    longest in x and shortest in z. Anything else means the exporter rotated the mesh; refuse it."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with open(body_obj) as fh:
        for line in fh:
            if line.startswith("v "):
                for i, tok in enumerate(line.split()[1:4]):
                    v = float(tok)
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
    ext = [hi[i] - lo[i] for i in range(3)]
    # Expect: x longest (~0.125, depth incl. connector), z shortest (~0.085, height), y ~0.105.
    order_ok = (
        ext[0] == max(ext)
        and ext[2] == min(ext)
        and 0.11 < ext[0] < 0.14
        and 0.075 < ext[2] < 0.095
        and abs(lo[2]) < 1e-3  # base dropped to z=0
    )
    if not order_ok:
        sys.exit(
            f"robin_w1g: exported housing has wrong orientation (extents x={ext[0]:.3f} "
            f"y={ext[1]:.3f} z={ext[2]:.3f} m, min_z={lo[2]:.3f}); expected x longest ~0.125, "
            f"z shortest ~0.085, base at z=0. The OBJ exporter in this Blender likely remapped the "
            f"up axis. Re-run with a Blender whose wm.obj_export honours up_axis='Z' (tested on "
            f"4.0/4.2), or the model will load rotated."
        )


if __name__ == "__main__":
    main()
