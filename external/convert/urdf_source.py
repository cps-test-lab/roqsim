"""Shared plumbing for the generators that build a model from a ROS 2 xacro tree.

Five ports now start the same way -- ROSbot, Doosan, Panther, Raspberry Pi Mouse, Ridgeback -- and
each had grown its own copy of this. The copies were identical apart from which packages they
symlink and which arguments they pass, which is the signal to extract: the fifth port paid again for
plumbing the first four had already debugged.

What lives here is everything that is *not* about a particular robot:

* **Expanding the xacro at all.** A description's ``$(find some_package)`` needs an ament index, and
  a vendor tree routinely reaches for a sibling package for its Gazebo or ros2_control block -- the
  Doosan wants ``dsr_controller2``, the Panther ``husarion_ugv_controller``, the Ridgeback
  ``clearpath_control`` -- none of which contributes anything to the model. A private index over the
  pinned checkouts satisfies them without installing ROS packages.
* **ROS's ``PYTHONPATH``.** The ``xacro`` entry point imports its own package, which is on the path
  only once the ROS setup has been sourced. Sourcing it would leak the entire ROS environment into
  the build; adding the one ``site-packages`` it needs does not, and keeps the failure legible when
  ROS is absent.
* **Reading a URDF back out.** Inertials, poses and a link's visuals, in the forms MJCF wants.

``link_visuals`` emits **every** ``<visual>`` of a link, meshes and primitives alike, and that is
deliberate rather than tidy: hand-picking a link's mesh is what left the Raspberry Pi Mouse's LDS-01
floating three centimetres above its mount, because the scanner link also carries four leg cylinders.
No test noticed; a person opening the viewer did.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def expand_xacro(
    packages: dict[str, Path],
    entry: Path,
    work: Path,
    args: list[str] | None = None,
    wrapper: str | None = None,
) -> ET.Element:
    """Expand a xacro tree into a plain URDF element, with a private ament index.

    ``packages`` maps package name -> its checkout, and every one is put on the index; several are
    usually there only so ``$(find ...)`` resolves. ``entry`` is the xacro to expand, or the path a
    ``wrapper`` is written to when the vendor's own top-level file pulls in a repository we do not
    want (both the ROSbot and the Panther reach for a components repository that only adds whichever
    sensors a unit was ordered with -- a deployment choice, not the platform).

    Raises with what to install when ``xacro`` is missing, rather than guessing a tree: every mass,
    inertia and offset a port records comes out of this.
    """
    xacro = shutil.which("xacro") or "/opt/ros/jazzy/bin/xacro"
    if not Path(xacro).exists():
        raise RuntimeError(
            f"xacro is required to expand {entry.name} and was not found.\n"
            "Install ROS 2 (ros-jazzy-xacro) or `pip install xacro`, then re-run. Refusing to guess "
            "the expanded tree: every mass, inertia and offset comes from it."
        )

    share = work / "prefix/share"
    (share / "ament_index/resource_index/packages").mkdir(parents=True, exist_ok=True)
    for name, path in packages.items():
        (share / f"ament_index/resource_index/packages/{name}").write_text("")
        if not (share / name).exists():
            (share / name).symlink_to(path)

    if wrapper is not None:
        entry = work / "wrapper.urdf.xacro"
        entry.write_text(wrapper)

    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = f"{work / 'prefix'}:{env.get('AMENT_PREFIX_PATH', '')}"
    ros_site = sorted(Path(xacro).resolve().parents[1].glob("lib/python3*/site-packages"))
    if ros_site:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ros_site[-1]), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

    proc = subprocess.run(
        [xacro, str(entry), *(args or [])], capture_output=True, text=True, env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(f"xacro failed:\n{proc.stderr.strip()}")
    return ET.fromstring(proc.stdout)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> str:
    """URDF roll/pitch/yaw as an MJCF ``quat`` string (w x y z)."""
    import numpy as np

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    q = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
    return " ".join(f"{v:.7g}" for v in q)


def pose(element: ET.Element | None) -> tuple[str, str]:
    """An element's ``<origin>`` as ``(pos, quat_attr)``; the quat is empty when it is identity."""
    origin = element.find("origin") if element is not None else None
    xyz = ((origin.get("xyz") if origin is not None else None) or "0 0 0").replace(",", " ")
    rpy = [float(v) for v in
           (((origin.get("rpy") if origin is not None else None) or "0 0 0")
            .replace(",", " ")).split()]
    quat = rpy_to_quat(*rpy)
    return xyz, ("" if quat.startswith("1 ") else f' quat="{quat}"')


def inertial(link: ET.Element, full: bool = False) -> dict[str, str]:
    """A link's ``<inertial>`` as MJCF attributes.

    ``full`` returns the six-term ``fullinertia`` (needed where a vendor publishes off-diagonal
    terms, as Doosan does) rather than the three-term ``diaginertia``.
    """
    element = link.find("inertial")
    origin = element.find("origin")
    inertia = element.find("inertia")
    keys = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz") if full else ("ixx", "iyy", "izz")
    return {
        "pos": ((origin.get("xyz") if origin is not None else None) or "0 0 0").replace(",", " "),
        "mass": element.find("mass").get("value"),
        ("fullinertia" if full else "diaginertia"): " ".join(inertia.get(k) for k in keys),
    }


def link_visuals(
    link: ET.Element,
    indent: str,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    materials: dict[str, str] | None = None,
    default_material: str | None = None,
    default_rgba: str = "0.15 0.15 0.15 1",
) -> str:
    """Every ``<visual>`` of *link* as MJCF geoms -- meshes AND primitives.

    Emitting only the meshes is what left the Raspberry Pi Mouse's scanner floating above its mount;
    a link's cylinders and boxes are as much a part of it as its mesh.

    ``materials`` maps a mesh stem to an MJCF material name; a link's own ``<material name=...>`` is
    used when the map has nothing for it, and ``default_material`` when neither does.

    An **exactly** repeated visual -- same geometry, same pose, same material -- is emitted once. A
    vendor tree can carry one: the Warthog's ``diff_link`` lists ``susp-link.stl`` twice at an
    identical origin, and that mesh already spans both sides (y -0.400..0.400), so the second is a
    no-op that only makes the two coincident geoms z-fight. The test is byte equality of the emitted
    geom, which is why this cannot repeat the Raspberry Pi Mouse's dropped leg cylinders: a geom that
    differs in any attribute is a different geom and is still emitted. If upstream ever gives the
    duplicate the mirrored origin it was presumably meant to have, it stops matching and comes back.
    """
    out = ""
    emitted: set[str] = set()
    for visual in link.findall("visual"):
        xyz, quat = pose(visual)
        x, y, z = (float(v) for v in xyz.split())
        placed = f'pos="{x + offset[0]:g} {y + offset[1]:g} {z + offset[2]:g}"{quat}'
        geometry = visual.find("geometry")
        mesh = geometry.find("mesh")
        cylinder = geometry.find("cylinder")
        box = geometry.find("box")
        sphere = geometry.find("sphere")

        geom = ""
        if mesh is not None:
            stem = Path(mesh.get("filename")).stem
            named = visual.find("material")
            material = (materials or {}).get(stem) or (
                named.get("name") if named is not None else None
            ) or default_material
            attr = f' material="{material}"' if material else ""
            geom = f'{indent}<geom class="visual" mesh="{stem}"{attr} {placed}/>\n'
        elif cylinder is not None:
            half = float(cylinder.get("length")) / 2
            geom = (f'{indent}<geom class="visual" type="cylinder"'
                    f' size="{float(cylinder.get("radius")):g} {half:g}" {placed}'
                    f' rgba="{default_rgba}"/>\n')
        elif box is not None:
            half = " ".join(f"{float(v) / 2:g}" for v in box.get("size").replace(",", " ").split())
            geom = (f'{indent}<geom class="visual" type="box" size="{half}" {placed}'
                    f' rgba="{default_rgba}"/>\n')
        elif sphere is not None:
            geom = (f'{indent}<geom class="visual" type="sphere"'
                    f' size="{float(sphere.get("radius")):g}" {placed} rgba="{default_rgba}"/>\n')
        if geom and geom not in emitted:
            emitted.add(geom)
            out += geom
    return out

def link_primitives(
    link: ET.Element,
    tag: str,
    cls: str,
    indent: str,
    materials: dict[str, str] | None = None,
    name_prefix: str | None = None,
    mesh_stems: dict[str, str] | None = None,
) -> str:
    """A link's ``<visual>`` or ``<collision>`` entries as MJCF geoms, primitives included.

    :func:`link_visuals` covers the visual side for the mesh-based ports and emits exactly the
    attribute order their committed models already carry, so it is left alone. This is for the
    descriptions built out of *primitives*, whose collision geometry is
    boxes, cylinders and spheres worth reproducing one-for-one rather than hulling. Those need the
    collision side too, and named, so a contact-based sensor has something to reference.

    ``mesh_stems`` maps a mesh basename stem to the MJCF mesh asset name; a mesh visual is skipped
    when the map has nothing for it, which is how a description that references a mesh it does not
    install is handled without silently emitting a dangling asset.
    """
    out = ""
    for index, element in enumerate(link.findall(tag)):
        xyz, quat = pose(element)
        geometry = element.find("geometry")[0]
        named = element.find("material")
        material = (materials or {}).get(named.get("name")) if named is not None else None
        attr = f' material="{material}"' if material and tag == "visual" else ""
        # Named from the TAG, not the class: a geom's name must not change because it was given a
        # different contact class. Naming it from `cls` renamed the caster the moment it earned a
        # caster_collision class, and a test looking up the old name got mj_name2id's -1 back and
        # silently asserted against whichever geom sits last in the model.
        name = f' name="{name_prefix or link.get("name")}_{tag}{index}"' if tag == "collision" else ""
        if geometry.tag == "mesh":
            stem = (mesh_stems or {}).get(Path(geometry.get("filename")).stem)
            if stem is None:
                continue
            out += (f'{indent}<geom class="{cls}" type="mesh" mesh="{stem}"'
                    f'{name}{attr} pos="{xyz}"{quat}/>\n')
            continue
        if geometry.tag == "cylinder":
            size = f'{float(geometry.get("radius")):g} {float(geometry.get("length")) / 2:g}'
        elif geometry.tag == "sphere":
            size = f'{float(geometry.get("radius")):g}'
        else:
            size = " ".join(f"{float(v) / 2:g}"
                            for v in geometry.get("size").replace(",", " ").split())
        out += (f'{indent}<geom class="{cls}" type="{geometry.tag}" size="{size}"'
                f'{name}{attr} pos="{xyz}"{quat}/>\n')
    return out


def mesh_scales(urdf: ET.Element) -> dict[str, str]:
    """``{mesh stem: MJCF scale string}`` for every mesh the URDF references.

    URDF puts the scale on the *reference*, MJCF puts it on the *asset*, so it has to be collected
    before the assets are written. Ignoring it is the classic meters-versus-millimetres trap, and it
    is worse than it sounds because it can be **non-uniform** and it can hide completely: one
    description in this corpus scaled a head mesh by ~2e-4 per axis, so emitted at 1:1 the mesh came
    out 1000 units across on a 0.2 m robot -- and no physics check noticed, because that link's
    *collision* was a primitive and only its visual was the mesh. It took a render. Any generator
    that emits mesh assets should route them through here.

    Raises when one mesh is referenced at two different scales, rather than silently picking one.
    """
    scales: dict[str, str] = {}
    for mesh in urdf.iter("mesh"):
        stem = Path(mesh.get("filename")).stem
        raw = (mesh.get("scale") or "1 1 1").replace(",", " ").split()
        scale = " ".join(f"{float(v):g}" for v in raw)
        if scales.setdefault(stem, scale) != scale:
            raise RuntimeError(
                f"mesh {stem!r} is referenced at two scales ({scales[stem]!r} and {scale!r}); MJCF "
                f"puts the scale on the asset, so this needs one asset per scale."
            )
    return scales
