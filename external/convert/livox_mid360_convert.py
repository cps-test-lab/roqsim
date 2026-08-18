"""Convert Livox Mid-360 STEP CAD into MuJoCo meshes: cascadio (STEP->OBJ) + Blender processing.

The ``convert.script`` for the ``livox_mid360_meshes`` resource (see external/external_assets.yaml).
Runs in the external-resources tool venv (needs ``cascadio``) and shells out to Blender for the mesh
split / decimation (needs Blender >= 3.6 on ``$BLENDER`` or ``PATH`` -- declared ``needs_blender``).

Contract (called by external/external_resources.py):
    --sources <housing.stp>
    --targets <body.obj> <dome.obj>
    --blender <path>                            # optional; else $BLENDER or 'blender'
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
    if len(args.sources) != 1 or len(args.targets) != 2:
        sys.exit("livox_mid360_convert: expected 1 source (housing) and 2 targets (body, dome)")

    housing_step = str(Path(args.sources[0]).resolve())
    body_out, dome_out = (str(Path(t).resolve()) for t in args.targets)
    for t in (body_out, dome_out):
        Path(t).parent.mkdir(parents=True, exist_ok=True)

    import cascadio

    blender = _resolve_blender(args.blender)
    with tempfile.TemporaryDirectory() as td:
        housing_obj = str(Path(td) / "housing.obj")
        print(f"cascadio: {housing_step} -> {housing_obj}")
        cascadio.step_to_obj(
            housing_step, housing_obj, tol_linear=0.1, tol_angular=0.5, use_colors=True
        )
        cmd = [
            blender, "-b", "-P", str(HERE / "livox_mid360_blender.py"),
            "--", housing_obj, body_out, dome_out,
        ]
        print("blender:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"livox_mid360: wrote {body_out}, {dome_out}")


if __name__ == "__main__":
    main()
