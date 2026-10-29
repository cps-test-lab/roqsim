#!/usr/bin/env python3
"""Build roqsim's FANUC M-20iD/25 and CRX-25iA MJCFs from FANUC's own description packages.

Source: ``FANUC-CORPORATION/fanuc_description`` (Apache-2.0, (c) FANUC America Corporation and FANUC
CORPORATION), pinned by commit. MuJoCo Menagerie carries no FANUC arm.

- ``m20id25``: ``fanuc_m20_description``, FANUC's URDF name ``m20_25_18d`` ("URDF for M-20/25-18D
  (M-20iD/25)").
- ``crx25ia``: there is no CRX-25iA description. FANUC lists the CRX-30iA row as "CRX-30iA
  (CRX-25iA)" on its supported-models page, a FANUC maintainer names the ``crx30ia`` files "the
  official ones for CRX-25iA" (fanuc_description issue #10), and the two datasheets give the same
  ranges, speeds, reaches and mass. So this builds the CRX-25iA from ``crx30ia``; only its payload
  rating (25 kg at 1889 mm) is the CRX-25iA's, and that lives in the manifest, not in the MJCF.

Passed through verbatim from FANUC's ``robot/<id>.urdf.xacro``: every link offset, mass, centre of
mass, full inertia tensor, joint axis and position limit, and the ``wbase``/``flange``/
``fanuc_flange`` frames (as sites). The wrapper's ``world -> base_link`` offset (0.75 m for the CRX)
is a mount, so it is left to the world, which places the arm.

Not from FANUC, and said so in the MJCF header and the LICENSE file:

- **Meshes.** COLLADA visuals converted per material through ``dae2obj.py`` (pycollada, node
  transforms applied) and decimated with ``reduce-mesh``. The forearm lettering ("FANUC Robot
  M-20" / "FANUC Robot CRX-30iA", a red word mark and dark text, each one flat material piece
  filling letter-shaped openings in the forearm) is **painted over**: merged into the forearm piece
  in its colour, so the surface stays closed and reads blank. Apache-2.0 grants no trademark right,
  and on the CRX-25iA the lettering would also name the wrong model.
- **Collision.** FANUC's convex STL per link, rewritten as OBJ in metres; where that one hull
  closes over a recess by more than 20 mm (``roqsim assets collision diff --tol 20``), a few convex
  pieces of the visual mesh instead (CoACD, ``pip install coacd``, seeded). Chain neighbours, and
  any other pair measured not to touch, are excluded.
- **Actuators.** URDF has none. Each joint gets an affine position servo whose gains follow from one
  formula (``servo_gains``); armature stands in for reflected rotor inertia, which FANUC does not
  publish. ``forcerange`` is the description's ``effort``.
- **Sites.** ``tool_site`` on ``fanuc_flange`` (z out of the flange), and on the CRX an ``fts_site``
  there for the embedded force sensing.

Needs ``xacro``, ``pycollada``, ``trimesh`` and ``coacd`` (pip) and Blender
(``reduce-mesh``) to rebuild; the committed model needs none of them.

Usage::

    python external/convert/build_fanuc_mjcf.py           # fetch, expand, convert, write both
    python external/convert/build_fanuc_mjcf.py --check   # rebuild the MJCF and diff it
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, rpy_to_quat, write_license  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models"

URL = "https://github.com/FANUC-CORPORATION/fanuc_description.git"
COMMIT = "fb40c9803a826ba68c7c8e28ba904a25efa7fcd2"  # tag v2.3.0
TAG = "v2.3.0"

JOINTS = [f"J{i}" for i in range(1, 7)]
LINKS = ["base_link"] + [f"J{i}_link" for i in range(1, 7)]
STEMS = ["base", "j1", "j2", "j3", "j4", "j5", "j6"]

#: Triangle budget per material piece. Keeps each arm under 60k triangles and 5 MB, the size of
#: the other vendor-converted arms (lbr_iiwa14 58k, m1013 43k).
TARGET_FACES = 4000

#: A lettering piece is flat: one of its extents is under this (metres).
DECAL_THICKNESS = 0.025

#: CoACD's concavity threshold for a split link, and its seed (the decomposition is randomised).
SPLIT_THRESHOLD = 0.05
SPLIT_SEED = 1

#: Servo bandwidth per joint (Hz) and damping ratio. The base axes move the arm, the wrist axes a
#: tool; a wrist servo as slow as a base servo lags every tool motion.
BANDWIDTH_HZ = [8.0, 8.0, 8.0, 12.0, 12.0, 12.0]
DAMPING_RATIO = 1.0
#: The rated payload may sag no joint by more than this (rad). A proportional servo holds a load
#: with an error; FANUC's controller (integral action) does not show one.
MAX_SAG = math.radians(1.0)
#: Where the payload's centre of mass is assumed, along the tool axis from the flange (m).
PAYLOAD_OFFSET = 0.1

VARIANTS = {
    "m20id25": {
        "title": "FANUC M-20iD/25",
        "fanuc_id": "m20_25_18d",
        "package": "fanuc_m20_description",
        # FANUC Europe MDS-03437-EN "M-20iD/25 (Hollow wrist)" / fanucamerica.com product page.
        "payload": 25.0,
        "reach": 1.831,
        # Reflected rotor inertia (kg*m^2): not published. Order-of-magnitude estimate, rotor
        # inertia of the axis' servo motor class times the square of a typical gear ratio for the
        # axis (J1-J3 ~1e-3 kg*m^2 x 140^2, J4-J6 ~1e-4 x 100^2). An assumption. servo_gains
        # re-derives kp and kv from the inertia it includes, so bandwidth and damping stay as set
        # and what the armature moves is the stiffness against an outside force: halving or
        # doubling it changes kp by -7/+15 % on J1-J2, -21/+43 % on J3, and about in proportion
        # on J4-J6, whose inertia it dominates.
        "armature": [20.0, 20.0, 10.0, 1.0, 1.0, 0.5],
        "lettering": {
            "j4": {
                "into": [0.898, 0.898, 0.0],
                "pieces": [[0.898, 0.0, 0.0], [0.149, 0.149, 0.149]],
            }
        },
        "fts": False,
        # Links whose single hull closes over a recess a neighbour or a workpiece enters, by more
        # than COLLISION_TOLERANCE (`roqsim assets collision diff`): decomposed into this many
        # convex pieces of the visual mesh. The rest keep FANUC's hull.
        "split": {"base_link": 4, "J1_link": 4, "J2_link": 4, "J3_link": 4},
        # Non-adjacent pairs that cannot touch, each with the measurement that says so.
        "exclude": [
            (
                "J4_link",
                "J6_link",
                "J6 turns inside J4's wrist fork across J5, its visual mesh a constant 16 mm clear\n"
                "         of the fork over J5's whole range; the fork's hull closes over it in every pose.",
            )
        ],
    },
    "crx25ia": {
        "title": "FANUC CRX-25iA",
        "fanuc_id": "crx30ia",
        "package": "fanuc_crx_description",
        # FANUC Europe MDS-04893-EN "CRX-25iA": 25 kg at 1889 mm.
        "payload": 25.0,
        "reach": 1.889,
        # As above, for the CRX's smaller motors and strain-wave reducers (J1-J3 ~2.5e-4 x 140^2,
        # J4-J6 ~5e-5 x 100^2). Halving or doubling it changes kp by -4/+7 % on J1-J2, -13/+26 %
        # on J3, and up to proportionally on J4-J6.
        "armature": [5.0, 5.0, 2.5, 0.5, 0.5, 0.3],
        "lettering": {
            "j4": {"into": [1.0, 1.0, 1.0], "pieces": [[1.0, 0.0, 0.0], [0.102, 0.102, 0.102]]}
        },
        "fts": True,
        "split": {"J2_link": 3},
        "exclude": [],
    },
}


def fmt(v: float) -> str:
    return f"{v:.6g}"


def vec(text: str | None) -> list[float]:
    return [float(v) for v in (text or "0 0 0").replace(",", " ").split()]


def spdx_lines(description: Path, variant: dict) -> list[str]:
    """FANUC's own SPDX header of the URDF macro, verbatim."""
    macro = description / f"urdf/{variant['fanuc_id']}_urdf_macro.xacro"
    lines = [ln.strip() for ln in macro.read_text().splitlines() if ln.strip().startswith("SPDX-")]
    if not any("License-Identifier: Apache-2.0" in ln for ln in lines):
        raise RuntimeError(f"{macro} no longer declares Apache-2.0: re-check the licence first")
    return lines


def expand(variant: dict, description: Path, work: Path) -> ET.Element:
    """FANUC's own robot wrapper, with only its ``$(find <pkg>)`` resolved (no ament index here)."""
    pkg = variant["package"]
    text = (description / f"robot/{variant['fanuc_id']}.urdf.xacro").read_text()
    return expand_xacro(
        {pkg: description}, None, work, wrapper=text.replace(f"$(find {pkg})", str(description))
    )


def obj_header(lines: list[str], spdx: list[str]) -> str:
    return "".join(f"# {ln}\n" for ln in [*lines, *spdx])


def _compact(path: Path) -> np.ndarray:
    """Rewrite an OBJ with only the vertices its faces use; return their extent.

    ``dae2obj.py`` writes each material's piece with the whole mesh's vertex buffer, so a piece
    carries every other piece's vertices unreferenced: dead weight on disk, and a bounding box
    that is the whole link's rather than the piece's.
    """
    lines = path.read_text().splitlines()
    verts = [ln for ln in lines if ln.startswith("v ")]
    faces = [[int(t.split("/")[0]) for t in ln.split()[1:]] for ln in lines if ln.startswith("f ")]
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used, start=1)}
    path.write_text(
        "".join(verts[i - 1] + "\n" for i in used)
        + "".join("f " + " ".join(str(remap[i]) for i in f) + "\n" for f in faces)
    )
    v = np.array([[float(x) for x in verts[i - 1].split()[1:4]] for i in used])
    return v.max(0) - v.min(0)


def _faces(path: Path) -> int:
    return sum(1 for ln in path.read_text().splitlines() if ln.startswith("f "))


def _append(dst: Path, src: Path) -> None:
    """Append one compacted OBJ's geometry to another's."""
    have = sum(1 for ln in dst.read_text().splitlines() if ln.startswith("v "))
    lines = src.read_text().splitlines()
    verts = [ln for ln in lines if ln.startswith("v ")]
    faces = [
        "f " + " ".join(str(int(t) + have) for t in ln.split()[1:])
        for ln in lines
        if ln.startswith("f ")
    ]
    text = dst.read_text().splitlines()
    dst.write_text(
        "\n".join([ln for ln in text if ln.startswith("v ")] + verts)
        + "\n"
        + "\n".join([ln for ln in text if ln.startswith("f ")] + faces)
        + "\n"
    )


def _strip_normals(path: Path) -> None:
    """Drop exported normals, so MuJoCo derives smooth ones from the shared vertices.

    Blender's OBJ export writes about one normal per face after a decimation, which renders every
    triangle flat-shaded.
    """
    out = []
    for ln in path.read_text().splitlines():
        if ln.startswith("vn "):
            continue
        if ln.startswith("f "):
            ln = "f " + " ".join(t.split("/")[0] for t in ln.split()[1:])
        out.append(ln)
    path.write_text("\n".join(out) + "\n")


def convert_meshes(
    name: str, description: Path, pkg: Path, spdx: list[str], urdf: ET.Element
) -> dict[str, list]:
    """Collada -> per-material OBJ -> decimated OBJ; STL hulls -> OBJ in metres.

    The forearm lettering is cut into the body mesh, its letters separate flat pieces filling the
    openings. Leaving them out would leave letter-shaped holes, so each is merged into the body
    piece it sits in and takes the body's colour: the surface stays closed and carries no lettering.
    """
    variant = VARIANTS[name]
    links_by_name = {link.get("name"): link for link in urdf.findall("link")}
    src = description / f"meshes/{variant['fanuc_id']}"
    meshes = pkg / "meshes"
    if meshes.exists():
        shutil.rmtree(meshes)
    meshes.mkdir(parents=True)
    materials: dict[str, list] = {}
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "dae2obj.py"),
                str(src / "visual"),
                str(raw),
            ],
            check=True,
            capture_output=True,
        )
        found = json.loads((raw / "materials.json").read_text())
        keep = Path(tempfile.mkdtemp())
        raw_whole = {}
        for stem in STEMS:
            raw_whole[stem] = keep / f"{stem}.obj"
            shutil.copyfile(raw / f"{stem}.obj", raw_whole[stem])
        for stem in STEMS:
            lettering = variant["lettering"].get(stem, {"into": None, "pieces": []})
            pieces = [list(c) for c in lettering["pieces"]]
            body = next((sub for sub, rgb in found[stem] if rgb == lettering["into"]), None)
            if pieces and body is None:
                raise RuntimeError(f"{name} {stem}: no piece coloured {lettering['into']}")
            merged = []
            for sub, rgb in found[stem]:
                extent = _compact(raw / f"{sub}.obj")
                if rgb in pieces:
                    if extent.min() > DECAL_THICKNESS:
                        raise RuntimeError(
                            f"{name} {sub} {rgb} is not flat ({extent}): not lettering"
                        )
                    merged.append((sub, rgb))
            if sorted(rgb for _, rgb in merged) != sorted(pieces):
                raise RuntimeError(f"{name} {stem}: expected lettering {pieces}, found {merged}")
            for sub, _ in merged:
                _append(raw / f"{body}.obj", raw / f"{sub}.obj")
            materials[stem] = []
            for sub, rgb in found[stem]:
                if rgb in pieces:
                    continue
                out = f"{stem}_visual_{len(materials[stem])}"
                target = meshes / f"{out}.obj"
                if _faces(raw / f"{sub}.obj") > TARGET_FACES:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "roqsim.commands",
                            "assets",
                            "reduce-mesh",
                            "--target-faces",
                            str(TARGET_FACES),
                            "--no-materials",
                            str(raw / f"{sub}.obj"),
                            str(target),
                        ],
                        check=True,
                        cwd=ROOT,
                        capture_output=True,
                    )
                    _strip_normals(target)
                    how = f"decimated to {TARGET_FACES} triangles"
                else:
                    shutil.copyfile(raw / f"{sub}.obj", target)
                    how = "not decimated"
                header = obj_header(
                    [
                        f"{out}.obj: {variant['title']} visual mesh, {stem}, colour {rgb}.",
                        f"Derived from FANUC-CORPORATION/fanuc_description @ {COMMIT},",
                        f"  {variant['package']}/meshes/{variant['fanuc_id']}/visual/{stem}.dae.",
                        f"Modified by the roqsim authors: converted from COLLADA, one material, {how}"
                        + (", the forearm lettering merged in" if sub == body and merged else ""),
                        "  (external/convert/build_fanuc_mjcf.py). Apache-2.0: see the LICENSE beside the model.",
                    ],
                    spdx,
                )
                target.write_text(header + target.read_text())
                materials[stem].append([out, rgb])

    import trimesh

    collision: dict[str, list] = {}
    for link, stem in zip(LINKS, STEMS, strict=True):
        visual_pos = _pure(links_by_name[link].find("visual"))
        if link in variant["split"]:
            import coacd

            coacd.set_log_level("error")
            whole = trimesh.load(raw_whole[stem], process=True)
            parts = coacd.run_coacd(
                coacd.Mesh(whole.vertices, whole.faces),
                threshold=SPLIT_THRESHOLD,
                max_convex_hull=variant["split"][link],
                seed=SPLIT_SEED,
                max_ch_vertex=64,
                preprocess_mode="auto",
            )
            collision[link] = []
            for k, (v, f) in enumerate(parts):
                hull = trimesh.Trimesh(v, f).convex_hull
                out = f"{link}_collision_{k}"
                header = obj_header(
                    [
                        f"{out}.obj: {variant['title']} collision piece {k}, {link}.",
                        f"Derived from FANUC-CORPORATION/fanuc_description @ {COMMIT},",
                        f"  {variant['package']}/meshes/{variant['fanuc_id']}/visual/{stem}.dae.",
                        "Modified by the roqsim authors: a convex piece of the visual mesh (CoACD "
                        f"{importlib.metadata.version('coacd')}, threshold {SPLIT_THRESHOLD}, seed {SPLIT_SEED}),",
                        "  in place of FANUC's one hull, which closes over the link's recesses",
                        "  (external/convert/build_fanuc_mjcf.py). Apache-2.0: see the LICENSE beside the model.",
                    ],
                    spdx,
                )
                body = "".join(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n" for p in hull.vertices)
                body += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in hull.faces)
                (meshes / f"{out}.obj").write_text(header + body)
                collision[link].append([out, visual_pos])
            continue
        stl = trimesh.load(src / f"collision/{stem}.stl", process=False)
        scale = 0.001 if stl.extents.max() > 20 else 1.0  # the CRX's STLs are in millimetres
        v = np.asarray(stl.vertices) * scale
        header = obj_header(
            [
                f"{link}_collision.obj: {variant['title']} collision hull, {link}.",
                f"Derived from FANUC-CORPORATION/fanuc_description @ {COMMIT},",
                f"  {variant['package']}/meshes/{variant['fanuc_id']}/collision/{stem}.stl.",
                "Modified by the roqsim authors: rewritten as OBJ"
                + (", scaled from millimetres to metres" if scale != 1.0 else "")
                + " (external/convert/build_fanuc_mjcf.py).",
                "Apache-2.0: see the LICENSE beside the model.",
            ],
            spdx,
        )
        body = "".join(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n" for p in v)
        body += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in stl.faces)
        (meshes / f"{link}_collision.obj").write_text(header + body)
        collision[link] = [[f"{link}_collision", _pure(links_by_name[link].find("collision"))]]
    shutil.rmtree(keep)
    # Beside the colours in the same sidecar: which collision pieces each link has, and where.
    materials["collision"] = collision
    return materials


def _origin(element: ET.Element | None) -> tuple[list[float], list[float]]:
    origin = element.find("origin") if element is not None else None
    if origin is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return vec(origin.get("xyz")), vec(origin.get("rpy"))


def _pure(element: ET.Element) -> str:
    xyz, rpy = _origin(element)
    if any(abs(a) > 1e-12 for a in rpy):
        raise ValueError(f"unexpected rotation {rpy} in {ET.tostring(element)[:120]!r}")
    return " ".join(fmt(v) for v in xyz)


def build(
    name: str,
    urdf: ET.Element,
    materials: dict[str, list],
    spdx: list[str],
    gains: list[tuple[float, float]] | None,
) -> str:
    variant = VARIANTS[name]
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    palette: dict[str, list[float]] = {}
    assets = []
    for stem in STEMS:
        for sub, rgb in materials[stem]:
            palette[f"mat_{sub}"] = rgb
            assets.append(f'    <mesh file="{sub}.obj" inertia="shell"/>\n')
    for link in LINKS:
        for mesh, _ in materials["collision"][link]:
            assets.append(f'    <mesh file="{mesh}.obj"/>\n')
    material_block = "".join(
        f'    <material name="{n}" rgba="{" ".join(fmt(c) for c in rgb)} 1"/>\n'
        for n, rgb in sorted(palette.items())
    )

    wbase = joints["base_link-wbase"]
    flange = joints["J6-flange"]
    fanuc_flange = joints["flange-fanuc_flange"]
    assert flange.find("parent").get("link") == "J6_link"
    ff_xyz, ff_rpy = _origin(fanuc_flange)
    if any(abs(v) > 1e-12 for v in ff_xyz):
        raise ValueError("fanuc_flange is offset from flange; carry the offset through")
    ff_quat = rpy_to_quat(*ff_rpy)
    flange_pos = _pure(flange)

    body = ""
    closing = []
    indent = ""
    for i, (link_name, stem) in enumerate(zip(LINKS, STEMS, strict=False)):
        link = links[link_name]
        indent = "    " + "  " * i
        if i == 0:
            body += f'{indent}<body name="{link_name}" childclass="{name}">\n'
        else:
            joint = joints[JOINTS[i - 1]]
            assert joint.find("child").get("link") == link_name
            assert joint.find("parent").get("link") == LINKS[i - 1]
            body += f'{indent}<body name="{link_name}" pos="{_pure(joint)}">\n'
        inert = inertial(link, full=True)
        body += (
            f'{indent}  <inertial pos="{inert["pos"].strip()}" mass="{inert["mass"]}"'
            f' fullinertia="{inert["fullinertia"]}"/>\n'
        )
        if i > 0:
            joint = joints[JOINTS[i - 1]]
            limit = joint.find("limit")
            axis = " ".join(fmt(v) for v in vec(joint.find("axis").get("xyz")))
            body += (
                f'{indent}  <joint name="{JOINTS[i - 1]}" axis="{axis}"'
                f' range="{fmt(float(limit.get("lower")))} {fmt(float(limit.get("upper")))}"'
                f' armature="{fmt(variant["armature"][i - 1])}"/>\n'
            )
        vpos = _pure(link.find("visual"))
        for sub, _ in materials[stem]:
            body += (
                f'{indent}  <geom class="visual" mesh="{sub}" pos="{vpos}" material="mat_{sub}"/>\n'
            )
        for mesh, pos in materials["collision"][link_name]:
            body += f'{indent}  <geom class="collision" mesh="{mesh}" pos="{pos}"/>\n'
        if i == 0:
            body += f'{indent}  <site name="wbase" pos="{_pure(wbase)}"/>\n'
        closing.append(indent)
    body += f'{indent}  <site name="flange" pos="{flange_pos}"/>\n'
    body += f'{indent}  <site name="fanuc_flange" pos="{flange_pos}" quat="{ff_quat}"/>\n'
    body += (
        f"{indent}  <!-- The mounting face: z out of the flange, the axis a tool mounts along. -->\n"
        f'{indent}  <site name="tool_site" pos="{flange_pos}" quat="{ff_quat}"/>\n'
    )
    if variant["fts"]:
        body += (
            f"{indent}  <!-- Where the embedded force sensing reports its wrench: FANUC's"
            f" driver frames it at fanuc_flange. -->\n"
            f'{indent}  <site name="fts_site" pos="{flange_pos}" quat="{ff_quat}"/>\n'
        )
    for ind in reversed(closing):
        body += f"{ind}</body>\n"

    actuators = ""
    for k, joint_name in enumerate(JOINTS):
        limit = joints[joint_name].find("limit")
        effort = float(limit.get("effort"))
        kp, kv = gains[k] if gains else (1.0, 0.0)
        actuators += (
            f'    <general name="{joint_name}" joint="{joint_name}" gaintype="fixed" biastype="affine"'
            f' gainprm="{fmt(kp)}" biasprm="0 -{fmt(kp)} -{fmt(kv)}"'
            f' ctrlrange="{fmt(float(limit.get("lower")))} {fmt(float(limit.get("upper")))}"'
            f' forcerange="-{fmt(effort)} {fmt(effort)}"/>\n'
        )
    excludes = "".join(
        f'    <exclude body1="{a}" body2="{b}"/>\n' for a, b in zip(LINKS, LINKS[1:], strict=False)
    )
    for a, b, why in variant["exclude"]:
        excludes += f'    <!-- {why} -->\n    <exclude body1="{a}" body2="{b}"/>\n'
    speeds = ", ".join(
        fmt(math.degrees(float(joints[j].find("limit").get("velocity")))) for j in JOINTS
    )
    gain_lines = "\n".join(
        f"      {j}: kp {fmt(kp)} N*m/rad, kv {fmt(kv)} N*m*s/rad, armature "
        f"{fmt(variant['armature'][k])} kg*m^2"
        for k, (j, (kp, kv)) in enumerate(zip(JOINTS, gains or [(0, 0)] * 6, strict=False))
    )
    spdx_block = "\n".join(f"    {ln}" for ln in spdx)
    home = " ".join(fmt(v) for v in HOME)
    return TEMPLATE.format(
        name=name,
        title=variant["title"],
        fanuc_id=variant["fanuc_id"],
        package=variant["package"],
        commit=COMMIT,
        tag=TAG,
        spdx=spdx_block,
        speeds=speeds,
        gain_lines=gain_lines,
        bandwidth=", ".join(fmt(b) for b in BANDWIDTH_HZ),
        sag=fmt(math.degrees(MAX_SAG)),
        payload=fmt(variant["payload"]),
        offset=fmt(PAYLOAD_OFFSET),
        materials=material_block,
        assets="".join(assets),
        body=body,
        actuators=actuators,
        excludes=excludes,
        home=home,
        crx_note=CRX_NOTE if name == "crx25ia" else "",
        split=", ".join(variant["split"]),
    )


#: FANUC's own ``default`` group state (fanuc_moveit_config/srdf/crx30ia.srdf): all zero but J5,
#: which points the flange down.
HOME = [0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]

CRX_NOTE = """
    CRX-25iA = CRX-30iA hardware: built from FANUC's crx30ia description, which FANUC names the
    official one for the CRX-25iA (fanuc_description issue #10; supported_models.md row
    "CRX-30iA (CRX-25iA)"). The CRX-25iA's 25 kg rating is in the manifest.
"""

TEMPLATE = """<mujoco model="{name}">
  <!--
    {title} for MuJoCo.

    Derived from FANUC's {package} ({fanuc_id}), FANUC-CORPORATION/fanuc_description
    @ {commit} ({tag}):
{spdx}
    Modified by the roqsim authors: GENERATED by external/convert/build_fanuc_mjcf.py. Do not
    hand-edit: re-run the generator. Apache-2.0, see the LICENSE file beside this model. "FANUC" and
    the model name say where the model comes from; this model is not made or endorsed by FANUC.
{crx_note}
    FANUC's own values: link offsets, masses, centres of mass, full inertia tensors, joint axes,
    position limits, forcerange (the description's `effort`) and the wbase / flange / fanuc_flange
    frames. Names are FANUC's (base_link, J1_link..J6_link, J1..J6); spawn_arm adds the world's prefix.
    Speeds ({speeds} deg/s, FANUC's description and datasheet alike) are enforced by the manifest's
    arm_controller `max_velocity`: MJCF has no joint velocity limit.

    Not FANUC's:
    - Visual meshes are FANUC's COLLADA split per material and decimated; the forearm lettering
      (word mark and model name) is painted over in the forearm's colour.
    - Collision is FANUC's convex hull per link, except where that hull closes over a recess by
      more than 20 mm ({split}): there, convex pieces of the visual mesh (CoACD).
    - Each joint is an affine position servo. With I the joint's largest diagonal mass-matrix entry
      over the home, zero and stretched poses (armature included) and w = 2*pi*f:
        kp = max(I * w^2, tau_payload / {sag} deg),  kv = 2 * zeta * sqrt(kp * I),  zeta = 1,
      f = {bandwidth} Hz, tau_payload the rated {payload} kg payload's gravity torque {offset} m out
      along the tool axis in the worst of those poses. Armature stands in for the reflected rotor
      inertia, which FANUC does not publish (an estimate; see the generator). Gravity is compensated
      by spawn_arm, not by the servo.
{gain_lines}

    No <option>: timestep, integrator, solver and contact overrides belong to the experiment.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="{name}">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <site size="0.005" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
  </default>

  <asset>
{materials}{assets}  </asset>

  <worldbody>
{body}  </worldbody>

  <contact>
    <!-- Chain neighbours. MuJoCo's filterparent is skipped when the parent is the world body, and
         base_link is welded to the world, so every pair would be measured; FANUC's hulls of
         neighbouring links overlap at the joint between them. -->
{excludes}  </contact>

  <actuator>
{actuators}  </actuator>

  <keyframe>
    <!-- FANUC's `default` group state from its MoveIt configuration: J5 points the flange down. -->
    <key name="home" qpos="{home}" ctrl="{home}"/>
  </keyframe>
</mujoco>
"""


def servo_gains(name: str, xml: str, pkg: Path) -> list[tuple[float, float]]:
    """kp, kv per joint, from the compiled model (the formula is in the MJCF header)."""
    import mujoco

    variant = VARIANTS[name]
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(pkg / "meshes")
    model = spec.compile()
    data = mujoco.MjData(model)
    tool = model.site("tool_site").id
    poses = {
        "home": HOME,
        "zero": [0.0] * 6,
        # Upper arm and forearm horizontal, forward: the largest inertia about J1 and J2.
        "stretched": [0.0, math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0],
        "stretched, tool down": [0.0, math.pi / 2, math.pi / 2, 0.0, -math.pi / 2, 0.0],
    }
    inertia = np.zeros(6)
    tau = np.zeros(6)
    full = np.zeros((model.nv, model.nv))
    jacp = np.zeros((3, model.nv))
    for q in poses.values():
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, full)
        inertia = np.maximum(inertia, np.diag(full))  # armature is part of qM's diagonal
        point = data.site_xpos[tool] + data.site_xmat[tool].reshape(3, 3)[:, 2] * PAYLOAD_OFFSET
        mujoco.mj_jac(model, data, jacp, None, point, model.site_bodyid[tool])
        force = np.array([0.0, 0.0, -9.81 * variant["payload"]])
        tau = np.maximum(tau, np.abs(jacp.T @ force))
    gains = []
    for k in range(6):
        w = 2 * math.pi * BANDWIDTH_HZ[k]
        kp = max(inertia[k] * w * w, tau[k] / MAX_SAG)
        kv = 2 * DAMPING_RATIO * math.sqrt(kp * inertia[k])
        gains.append((float(f"{kp:.3g}"), float(f"{kv:.3g}")))
    return gains


def license_header(name: str, spdx: list[str]) -> list[str]:
    variant = VARIANTS[name]
    return [
        f"{variant['title']} -- vendored geometry and description.",
        "",
        f"Upstream:   {URL.removesuffix('.git')}",
        f"            {variant['package']} ({variant['fanuc_id']})",
        f"Commit:     {COMMIT} (tag {TAG})",
        "Copyright:  FANUC America Corporation and FANUC CORPORATION. The URDF macro carries:",
        *[f"              {ln}" for ln in spdx],
        "License:    Apache License 2.0: the repository's README and LICENSES/Apache-2.0.txt,",
        f"            {variant['package']}/package.xml and the macro's SPDX header. The meshes",
        "            carry no terms of their own and there is no NOTICE file.",
        *(
            [
                "CRX-25iA:   built from FANUC's crx30ia description, which FANUC names the official one",
                "            for the CRX-25iA (fanuc_description issue #10).",
            ]
            if name == "crx25ia"
            else []
        ),
        "",
        "Changes by the roqsim authors (Apache-2.0 section 4(b)):",
        "  - the MJCF is generated from FANUC's URDF; actuators, servo gains, armature, the",
        "    tool_site/fts_site sites, the contact excludes and the home keyframe are not FANUC's;",
        "  - visual meshes converted from COLLADA to OBJ, split per material and decimated; the",
        "    forearm lettering is painted over (merged into the forearm in its colour);",
        "  - collision meshes rewritten as OBJ in metres; on the links whose one hull closes",
        "    over a recess, convex pieces of the visual mesh instead (CoACD).",
        "Every derived file names its source file and carries FANUC's SPDX lines.",
        "",
        'Trademarks: Apache-2.0 section 6 grants none. "FANUC" and the model names say only',
        "where this model comes from; it is not made or endorsed by FANUC, and no FANUC logo is",
        "shipped.",
        "",
        "Regenerate with: external/convert/build_fanuc_mjcf.py",
        "The full text of the grant follows.",
    ]


def credits(name: str) -> str:
    variant = VARIANTS[name]
    return (
        f"{variant['title']} ({name})\n"
        f"  Source:    {URL.removesuffix('.git')} @ {COMMIT} ({TAG}),\n"
        f"             {variant['package']}, URDF {variant['fanuc_id']}\n"
        "  Author:    FANUC America Corporation and FANUC CORPORATION\n"
        "  License:   Apache-2.0 (full text and changes: "
        f"{name.upper()}_LICENSE beside this file)\n"
        "  Converted: external/convert/build_fanuc_mjcf.py\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", choices=sorted(VARIANTS))
    args = parser.parse_args()

    repo = resolve_source("fanuc_description", URL, COMMIT)
    failed = False
    for name in [args.only] if args.only else sorted(VARIANTS):
        variant = VARIANTS[name]
        description = repo / variant["package"]
        spdx = spdx_lines(description, variant)
        pkg = MODELS / name
        pkg.mkdir(parents=True, exist_ok=True)
        target = pkg / f"{name}.xml"
        materials_file = pkg / "meshes" / f"{name}.materials.json"
        with tempfile.TemporaryDirectory() as tmp:
            urdf = expand(variant, description, Path(tmp))
        if args.check:
            materials = json.loads(materials_file.read_text())
        else:
            materials = convert_meshes(name, description, pkg, spdx, urdf)
            materials_file.write_text(json.dumps(materials, indent=2, sort_keys=True) + "\n")
        # Twice: the gains are computed from the compiled model, which the first pass provides.
        xml = build(
            name,
            urdf,
            materials,
            spdx,
            servo_gains(name, build(name, urdf, materials, spdx, None), pkg),
        )
        if args.check:
            if not target.exists() or target.read_text() != xml:
                print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
                failed = True
            else:
                print(f"{target}: up to date with {COMMIT[:12]}")
            continue
        write_license(
            repo / "LICENSES/Apache-2.0.txt",
            pkg / f"{name.upper()}_LICENSE",
            license_header(name, spdx),
        )
        (pkg / "CREDITS.txt").write_text(credits(name))
        target.write_text(xml)
        print(f"wrote {target} + meshes + {name.upper()}_LICENSE + CREDITS.txt")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
