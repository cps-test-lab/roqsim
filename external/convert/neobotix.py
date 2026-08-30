"""Shared plumbing for the Neobotix ports, all four of which come from `neo_simulation2`.

Extracted at the third caller. The four platforms differ in drive — the MPO-700 steers, the MPO-500
runs omni wheels, the MP-400 and MP-500 are differential — so their MJCF bodies and actuators are
genuinely per-robot. What is *not* per-robot is everything below: one pinned source, one Collada
pipeline, one palette convention, and one set of MuJoCo quirks this vendor's exports trip over.

Deliberately not a single generator for all four. The bodies differ in tree shape (a steering layer
or not, four casters or one), and folding those into one template produced more branching than the
duplication it removed. The pipeline is the part that was actually the same three times.

Two MuJoCo facts this vendor's meshes force, worth stating once here rather than in each generator:

* **`inertia="shell"` on every mesh.** These are CAD surface exports split per material, so several
  sub-meshes are thin shells with no meaningful enclosed volume and MuJoCo refuses to integrate an
  inertia over them ("mesh volume is too small"). It never needed to — every Neobotix body carries
  the vendor's own explicit ``<inertial>``, so a mesh-derived inertia is discarded anyway.
* **Colour lives in the split, not the OBJ.** MuJoCo reads no OBJ material, so a part keeps its
  colour only if ``dae2obj`` gave each bound material its own file and the MJCF names one material
  per sub-geom. Skipping that renders the whole robot flat grey — which for these platforms loses the
  SICK scanners' signature yellow, the wheel accents and the status LEDs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf_source import mesh_scales

NEO_URL = "https://github.com/neobotix/neo_simulation2.git"
#: The `humble` branch. `rolling` and `jazzy-sync` ship flattened URDFs whose joints are all `fixed`;
#: `humble` keeps the xacro macros that describe the actual mechanisms.
NEO_COMMIT = "832041452c1a0199afea1e9b65adf37381e96214"
DEFAULT_FACES = 4000


def wrapper(model: str, joint_type: str = "continuous") -> str:
    """A top-level xacro that includes *model*'s body with a chosen ``ODM_joint_type``.

    Needed because the vendor's own top-levels declare that as a ``<xacro:property>`` rather than a
    ``<xacro:arg>``, so it cannot be overridden from outside — and the MPO-700's hardcodes ``fixed``,
    which would give a swerve robot welded wheels. Going through our own top-level also keeps the
    expansion to geometry, skipping their Gazebo xacro and its ros2_control block.
    """
    return f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="{model}">
  <xacro:arg name="use_docking_adapter" default="false"/>
  <xacro:property name="ODM_joint_type" value="{joint_type}"/>
  <xacro:property name="arm" value=""/>
  <xacro:property name="use_arm" value="false"/>
  <xacro:include filename="$(find neo_simulation2)/robots/{model}/urdf/{model}_body.urdf.xacro"/>
</robot>
"""


def convert_meshes(
    source: Path,
    urdf: ET.Element,
    package: Path,
    model: str,
    root: Path,
    budgets: dict[str, int] | None = None,
) -> dict[str, str]:
    """Collada -> per-material OBJ -> decimated OBJ. Returns ``{mesh stem: MJCF scale}``.

    Only the meshes the expanded tree actually references are converted, so an unused vendor asset
    never ships. ``budgets`` overrides the face budget per source stem, for the occasional mesh that
    is far heavier than the rest (the MPO-500's wheel Collada is 8.3 MB with every roller modelled,
    and its contact is a sphere, so the mesh is purely cosmetic).

    ``dae2obj``'s palette is written beside the meshes as ``<model>.materials.json`` so a
    generator's ``--check`` can rebuild the MJCF without Blender or pycollada.
    """
    scales = mesh_scales(urdf)
    (package / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (package / "meshes").glob("*.obj"):
        stale.unlink()
    wanted = {}
    for mesh in urdf.iter("mesh"):
        rel = mesh.get("filename").split("neo_simulation2/", 1)[1]
        wanted[Path(rel).stem] = source / rel
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "dae"
        staged.mkdir()
        for stem, path in wanted.items():
            shutil.copy2(path, staged / f"{stem}.dae")
        raw = Path(tmp) / "obj"
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "dae2obj.py"), str(staged), str(raw)],
            check=True, capture_output=True,
        )
        palette = json.loads((raw / "materials.json").read_text())
        (package / f"meshes/{model}.materials.json").write_text(
            json.dumps(palette, indent=2, sort_keys=True) + "\n")
        for stem, parts in palette.items():
            budget = (budgets or {}).get(stem, DEFAULT_FACES)
            for sub, _rgb in parts:
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(budget), "--no-materials",
                     str(raw / f"{sub}.obj"), str(package / "meshes" / f"{sub}.obj")],
                    check=True, cwd=root, capture_output=True,
                )
    return scales


def colours(package: Path, model: str) -> dict[str, str]:
    """``{sub-mesh: MJCF rgba}`` from the palette sidecar.

    One material per sub-mesh, named after it, rather than a hand-maintained colour -> name table:
    with ten colours across four meshes 1:1 is simpler, impossible to get wrong, and an upstream
    repaint only changes an rgba.
    """
    return {
        sub: " ".join(f"{float(c):g}" for c in (*rgb[:3], 1.0))
        for parts in json.loads((package / f"meshes/{model}.materials.json").read_text()).values()
        for sub, rgb in parts
    }


def asset_block(shipped: set[str], palette: dict[str, str], scales: dict[str, str]) -> str:
    """The MJCF ``<asset>`` body: one material per sub-mesh, then every mesh, all shell-inertia."""
    return "".join(
        f'    <material name="{sub}_mat" rgba="{rgba}"/>\n'
        for sub, rgba in sorted(palette.items()) if sub in shipped
    ) + "".join(
        f'    <mesh file="{sub}.obj" scale="{scales.get(sub.split("__")[0], "1 1 1")}"'
        f' inertia="shell"/>\n'
        for sub in sorted(shipped)
    )


def subs_for(stem: str, shipped: set[str]) -> list[str]:
    """The shipped sub-meshes of a URDF mesh reference, matched by prefix.

    ``dae2obj`` splits a multi-material Collada into ``<stem>__<material>`` while the URDF only ever
    names ``<stem>``, so every consumer needs this and none should re-derive it.
    """
    return sorted(s for s in shipped if s == stem or s.startswith(f"{stem}__"))
