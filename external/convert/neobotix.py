"""Shared plumbing for the Neobotix ports.

Extracted at the third caller. The four platforms differ in drive — the MPO-700 steers, the MPO-500
runs omni wheels, the MP-400 and MP-500 are differential — so their MJCF bodies and actuators are
genuinely per-robot. What is *not* per-robot is everything below: one pinned source, one Collada
pipeline, one palette convention, and one set of MuJoCo quirks this vendor's exports trip over.

Deliberately not a single generator for all four. The bodies differ in tree shape (a steering layer
or not, four casters or one), and folding those into one template yields more branching than the
duplication it removes. The pipeline is the part that is actually the same for each.

Two MuJoCo facts this vendor's meshes force, worth stating once here rather than in each generator:

* **`inertia="shell"` on every mesh.** These are CAD surface exports split per material, so several
  sub-meshes are thin shells with no meaningful enclosed volume and MuJoCo refuses to integrate an
  inertia over them ("mesh volume is too small"). It never needed to — every Neobotix body carries
  the vendor's own explicit ``<inertial>``, so a mesh-derived inertia is discarded anyway.
* **Colour lives in the split, not the OBJ.** MuJoCo reads no OBJ material, so a part keeps its
  colour only if ``dae2obj`` gave each bound material its own file and the MJCF names one material
  per sub-geom. Skipping that renders the whole robot flat grey — which for these platforms loses the
  body's signature yellow, the wheel accents and the status LEDs.

The SICK scanners are not part of any generated robot: they are device models in ``roqsim_sensors``
(``sick_s300``, ``sick_microscan3``) that each robot's manifest mounts at the vendor's joint origin, so
every generator removes the scanner links from the expanded tree first (:func:`drop_scanner_links`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from urdf_source import mesh_scales

NEO_URL = "https://github.com/neobotix/neo_simulation2.git"
#: The `humble` branch. `rolling` and `jazzy-sync` ship flattened URDFs whose joints are all `fixed`;
#: `humble` keeps the xacro macros that describe the actual mechanisms.
NEO_COMMIT = "832041452c1a0199afea1e9b65adf37381e96214"

ROX_URL = "https://github.com/neobotix/rox.git"
#: `jazzy`. The ROX is the one Neobotix platform NOT in `neo_simulation2`, at any branch -- it has its
#: own repository, and `rox_description` is the package that ships its geometry. `humble` predates the
#: Diff variant entirely (no diff_drive.xacro), and every file the Diff port reads is byte-identical
#: between `jazzy` and `rolling`.
ROX_COMMIT = "c865076d5412aca9c861e3b3326fb6a30cb393b1"
#: The ROS package inside that repository whose `package://` refs the meshes are named by.
ROX_PACKAGE = "rox_description"

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


def drop_scanner_links(urdf: ET.Element, names: tuple[str, ...]) -> None:
    """Remove the scanner links *names*, and the fixed joints that attach them, from *urdf* in place.

    The device model a manifest mounts carries the scanner's housing, mesh and mass, so the robot's
    MJCF carries none of them. Removing the links before anything reads the tree keeps all three out
    at once: :func:`convert_meshes` converts only the meshes the tree still references, the body loop
    never sees the links, and the mass audit sums what is left. A name the tree lacks, or a link that
    something else hangs from, is refused rather than skipped: either means the vendor description
    changed under the pin.
    """
    for name in names:
        link = next((lk for lk in urdf.findall("link") if lk.get("name") == name), None)
        if link is None:
            raise ValueError(f"{name!r} is not a link of the expanded description")
        hung = [j.get("name") for j in urdf.findall("joint") if j.find("parent").get("link") == name]
        if hung:
            raise ValueError(f"{name!r} carries {hung}; removing it would orphan them")
        urdf.remove(link)
        for joint in [j for j in urdf.findall("joint") if j.find("child").get("link") == name]:
            urdf.remove(joint)


def convert_meshes(
    source: Path,
    urdf: ET.Element,
    package: Path,
    model: str,
    root: Path,
    budgets: dict[str, int] | None = None,
    pkg: str = "neo_simulation2",
) -> dict[str, str]:
    """Collada -> per-material OBJ -> decimated OBJ. Returns ``{mesh stem: MJCF scale}``.

    Only the meshes the expanded tree actually references are converted, so an unused vendor asset
    never ships. ``budgets`` overrides the face budget per source stem, for the occasional mesh that
    is far heavier than the rest (the MPO-500's wheel Collada is 8.3 MB with every roller modelled,
    and its contact is a sphere, so the mesh is purely cosmetic).

    ``pkg`` is the ROS package the URDF's ``package://`` refs name, and *source* is that package's
    directory -- the two differ between the Neobotix repositories: in ``neo_simulation2`` one package
    holds every robot, while the ROX's geometry is its own ``rox_description``.

    ``dae2obj``'s palette is written beside the meshes as ``<model>.materials.json`` so a
    generator's ``--check`` can rebuild the MJCF without Blender or pycollada.
    """
    scales = mesh_scales(urdf)
    (package / "meshes").mkdir(parents=True, exist_ok=True)
    for stale in (package / "meshes").glob("*.obj"):
        stale.unlink()
    wanted = {}
    for mesh in urdf.iter("mesh"):
        rel = mesh.get("filename").split(f"{pkg}/", 1)[1]
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
                out = package / "meshes" / f"{sub}.obj"
                subprocess.run(
                    [sys.executable, "-m", "roqsim.commands", "assets", "reduce-mesh",
                     "--target-faces", str(budget), "--no-materials",
                     str(raw / f"{sub}.obj"), str(out)],
                    check=True, cwd=root, capture_output=True,
                )
                # MuJoCo refuses a mesh of fewer than 4 vertices, and these exports carry the
                # occasional stray triangle bound to its own material -- a degenerate scrap of the
                # CAD, not a part. Dropping it here keeps it out of the palette and the asset block
                # too, because both filter on what was actually shipped. Said out loud rather than
                # silently, so a sub-mesh that is genuinely missing cannot hide behind it.
                if _vertex_count(out) < 4:
                    print(f"  dropped {sub}: {_vertex_count(out)} vertices, below MuJoCo's minimum")
                    out.unlink()
    return scales


def _vertex_count(obj: Path) -> int:
    return sum(1 for line in obj.read_text().splitlines() if line.startswith("v "))


def hull_obj(package: Path, stems: list[str], name: str, scale: str = "1 1 1",
             clip: list[tuple[tuple[float, float, float], float]] | None = None) -> str:
    """Write the convex hull of *stems* as ``meshes/<name>.obj``; returns *name*.

    For a vendor link whose collision IS its full visual mesh. That mesh is split per material by
    `dae2obj`, and a collision geom can name only one of the pieces -- which silently under-fills the
    body, because the alphabetically first piece is not the largest. MuJoCo convex-hulls a collision
    mesh anyway, so the hull of every piece together is what the vendor's own collision compiles to,
    at a few dozen faces instead of several thousand.

    *scale* is the MJCF scale its source meshes carry, and it is baked into the hull's vertices --
    so the hull is written in metres and needs no scale of its own. Leaving it to be inferred is the
    trap: `asset_block` derives a scale from the sub-mesh naming this asset deliberately does not
    share, and would silently give a decimetre hull a scale of 1.

    *clip* is a list of ``((nx, ny, nz), d)`` half-spaces, each keeping the vertices with
    ``n . v <= d``. A hull is convex by construction, so a body whose real outline is NOT convex --
    a chassis with a chamfered corner, say -- comes out with the chamfer filled in, and anything the
    vendor put in that corner ends up inside the collision. Clipping to the planes the chamfer
    already follows keeps the hull convex and the corner open.

    Faces are wound outward, so MuJoCo compiles the result as a closed solid.
    """
    from scipy.spatial import ConvexHull

    verts = np.vstack([
        np.array([[float(x) for x in line.split()[1:4]]
                  for line in (package / f"meshes/{stem}.obj").read_text().splitlines()
                  if line.startswith("v ")])
        for stem in stems
    ])
    verts = verts * np.array([float(v) for v in scale.split()])
    for normal, offset in clip or []:
        verts = verts[verts @ np.array(normal) <= offset]
    hull = ConvexHull(verts)
    index = {int(v): i for i, v in enumerate(hull.vertices)}
    lines = [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in verts[hull.vertices]]
    for simplex, plane in zip(hull.simplices, hull.equations, strict=True):
        a, b, c = verts[simplex]
        if np.dot(np.cross(b - a, c - a), plane[:3]) < 0:
            simplex = simplex[[0, 2, 1]]
        lines.append("f " + " ".join(str(index[int(v)] + 1) for v in simplex))
    (package / f"meshes/{name}.obj").write_text("\n".join(lines) + "\n")
    return name


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
