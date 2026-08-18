"""Convert the Zivid 3 XL250 CAD into MuJoCo meshes: Blender STL processing.

The ``convert.script`` for the ``zivid_xl250_meshes`` resource (see external/external_assets.yaml).
Runs in the external-resources tool venv but does no Python-side geometry work -- it only shells out
to Blender (needs Blender >= 4.0 for ``wm.stl_import``, on ``$BLENDER`` or ``PATH`` -- declared
``needs_blender``). Zivid's STL is a single dark part, so there is no STEP->OBJ material split (unlike
the Livox pipeline) and hence no ``cascadio`` dependency.

Contract (called by external/external_resources.py):
    --sources <zivid.stl>
    --targets <body.obj> <fov.obj>
    --blender <path>                # optional; else $BLENDER or 'blender'
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
        sys.exit("zivid_convert: expected 1 source (stl) and 2 targets (body, fov)")

    stl = str(Path(args.sources[0]).resolve())
    body_out, fov_out = (str(Path(t).resolve()) for t in args.targets)
    for t in (body_out, fov_out):
        Path(t).parent.mkdir(parents=True, exist_ok=True)

    blender = _resolve_blender(args.blender)
    cmd = [blender, "-b", "-P", str(HERE / "zivid_blender.py"), "--", stl, body_out, fov_out]
    print("blender:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    _check_orientation(body_out)
    print(f"zivid: wrote {body_out}, {fov_out}")


def _check_orientation(body_obj: str) -> None:
    """Fail loudly if the exported housing is not Z-up (the frame zivid.xml is authored against).

    Blender's OBJ exporter can remap axes differently across versions/builds; a Y-up export would load
    into MuJoCo lying on its back with the optics pointing at the ceiling while the camera/FOV (fixed in
    zivid.xml) still point along +y -- exactly the 90-degrees-off "mesh looks up, FOV to the side" bug.
    The device is 336 (w, x) x 70 (d, y) x 133 (h, z) mm, so a correct export is widest in x and tallest
    in z. Anything else means the exporter rotated the mesh; refuse it rather than ship a wrong model."""
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
    # Expect: x widest (~0.336), z tallest (~0.133), y shallowest (~0.070).
    order_ok = (
        ext[0] == max(ext)
        and ext[2] > ext[1]
        and 0.30 < ext[0] < 0.37
        and 0.11 < ext[2] < 0.15
    )
    if not order_ok:
        sys.exit(
            f"zivid: exported housing has wrong orientation (extents x={ext[0]:.3f} y={ext[1]:.3f} "
            f"z={ext[2]:.3f} m); expected x widest ~0.336 and z tallest ~0.133. The OBJ exporter in "
            f"this Blender likely remapped the up axis (Y-up instead of Z-up). Re-run with a Blender "
            f"whose wm.obj_export honours up_axis='Z' (tested on 4.0/4.2), or the model will load "
            f"rotated (optics facing up)."
        )


if __name__ == "__main__":
    main()
