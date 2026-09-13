#!/usr/bin/env python3
"""Build roqsim's Neobotix ROX-Diff MJCF from `neobotix/rox`.

The ROX is Neobotix's current platform line and the one that is **not** in `neo_simulation2` at any
branch: it has its own repository, whose `rox_description` ships four kinematics (Argo, Argo-Trio,
Diff, Trike) behind one xacro. This builds the **Diff**: two driven wheels on the centre line and
four passive casters, so ``diff_drive`` and no ``slip_factor``, the same line the MP-400 draws.

Unlike its three siblings this port needs **no wrapper xacro**. Their top-levels declare the joint
type as a `<xacro:property>` that cannot be overridden from outside, which is why `neobotix.wrapper`
exists; `rox.urdf.xacro` declares `rox_type`, `joint_type` and `scanner_type` as real `<xacro:arg>`s,
so the vendor's own top-level is expanded directly with the arguments below. Leaving `arm_type` and
`use_d435` at their defaults also keeps the expansion clear of `ur_description`,
`robotiq_description` and `realsense2_description`, which the manipulator variants would pull in.

**The wheel inertia is wrong, and the geometry proves it rather than merely suggesting it.**
`diff_wheel.xacro` declares `izz = 0.05625` at `mass = 5.0`, which is exactly one half m r^2 for
r = 0.15 m -- the wheel's DIAMETER, used where its radius belongs. Three independent things say the
radius is 0.075 m: the collision sphere, the joint's mounting height (a 0.15 m wheel would put the
axle below the floor), and the visual mesh, which measures 0.1499 m across. So every wheel's inertia
tensor is four times too large. It is shipped unchanged and pinned by a test: the audit exists to
check the vendor's numbers, and substituting a plausible one would defeat it.

The masses are the vendor's own placeholders -- every `<inertial>` in this description carries the
comment "These are not accurate value", and the data sheet publishes payload but no dead weight, so
there is nothing to calibrate the 140 kg body against. Use this model for navigation, not dynamics.

**The scanners are not in this model.** The manifest mounts the `sick_nanoscan3` device model at the
vendor's two lidar joints, and that device carries each housing, mesh and mass. This generator
removes both links from the expanded tree first (:func:`neobotix.drop_scanner_links`), so the MJCF's
mass sum is the description's minus their 0.001 kg each.

Usage::

    python external/convert/build_rox_diff_mjcf.py           # fetch, convert, write
    python external/convert/build_rox_diff_mjcf.py --check    # rebuild and diff
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neobotix import (  # noqa: E402
    ROX_COMMIT, ROX_PACKAGE, ROX_URL, asset_block, colours, convert_meshes, drop_scanner_links,
    hull_obj, subs_for,
)
from sources import resolve_source  # noqa: E402
from urdf_source import expand_xacro, inertial, mesh_scales, pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "roqsim_mobile/src/roqsim_mobile/models/rox_diff"

WHEELS = ("left", "right")
CASTERS = ("front_left", "front_right", "back_left", "back_right")
#: Scanner links removed from the expanded tree: the manifest mounts sick_nanoscan3 at both.
SENSOR_LINKS = ("lidar_1_link", "lidar_2_link")
#: The vendor's own arguments for this variant. `joint_type` drives the WHEELS; the Diff's casters are
#: hardcoded `fixed` in diff_drive.xacro, which is why they come out as jointless bodies below.
XACRO_ARGS = [
    "rox_type:=diff",
    "joint_type:=continuous",
    "scanner_type:=nanoscan",
    "use_gz:=false",
]
#: The caster's contact, replacing the vendor's collision sphere. Theirs is radius 0.124 centred on
#: the swivel lead, which reaches the floor from a link mounted 0.124 m up -- but it is therefore a
#: 248 mm ball that also bulges 124 mm ABOVE the link, and at the front-left corner that swallows the
#: robot's own nanoScan3: every ray of that scanner starts inside it and the scan reads its too-close
#: value in all 1651 directions.
#:
#: These are the wheel the vendor's own mesh draws. Sliced in the link frame, diff_caster.dae is a
#: circle 80 mm across centred at x +0.045 -- the same swivel lead they used -- spanning z -0.124 to
#: -0.054, so its centre is 89 mm below the link and its radius 35 mm. The contact point is thus
#: unchanged (bottom of the sphere on the floor) while the ball is the caster instead of the mount
#: height. `build` asserts the contact still lands on the floor.
CASTER_WHEEL_RADIUS = 0.035
CASTER_WHEEL_POS = (0.045, 0.0, -0.089)
#: short_frame.dae is 44954 vertices of chassis that collides as one mesh; the rest are modest.
BUDGETS = {"short_frame": 6000}
#: The chassis collision: one convex hull over every short_frame sub-mesh (see `geoms`).
BASE_HULL = "rox_diff_base_collision"


def base_hull_clip(urdf: ET.Element) -> list[tuple[tuple[float, float, float], float]]:
    """The two half-spaces that keep the scanner corners out of the chassis collision hull.

    This chassis is a frame, not a block: at the scanners' height its outline is a rectangle with
    both diagonal corners CHAMFERED, and a nanoScan3 sits in each chamfer looking out along the
    diagonal. A convex hull cannot represent that, so hulling the chassis fills both chamfers and
    buries the scanners inside their own robot -- every ray then starts in collision geometry and the
    scan reads its too-close value in every direction.

    The plane is taken from the vendor's own mount: each scanner's distance from the base origin
    along its diagonal, which is where the mesh's chamfer already lies. So the clip is read off the
    description rather than chosen, and the hull becomes the octagon the robot actually is.
    """
    planes = []
    for joint in urdf.findall("joint"):
        if joint.find("child").get("link") not in SENSOR_LINKS:
            continue
        x, y, _ = (float(v) for v in pose(joint)[0].split())
        radius = (x * x + y * y) ** 0.5
        planes.append(((x / radius, y / radius, 0.0), radius))
    if len(planes) != len(SENSOR_LINKS):
        raise ValueError(f"expected a joint per scanner link, found {len(planes)}")
    return planes


def build(urdf: ET.Element, shipped: set[str], scales: dict[str, str]) -> str:
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    palette = colours(PKG, "rox_diff")

    def geoms(link: ET.Element, indent: str, tyre_class: str = "wheel_collision") -> str:
        out = ""
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, quat = pose(visual)
            for sub in subs_for(Path(mesh.get("filename")).stem, shipped):
                mat = f' material="{sub}_mat"' if sub in palette else ""
                out += f'{indent}<geom class="visual" mesh="{sub}"{mat} pos="{xyz}"{quat}/>\n'
        for collision in link.findall("collision"):
            shape = collision.find("geometry")[0]
            xyz, quat = pose(collision)
            if shape.tag == "sphere":
                radius, xyz = float(shape.get("radius")), xyz
                if tyre_class == "caster_collision":
                    radius = CASTER_WHEEL_RADIUS
                    xyz = " ".join(f"{v:g}" for v in CASTER_WHEEL_POS)
                out += (f'{indent}<geom class="{tyre_class}" name="{link.get("name")}_tyre"'
                        f' size="{radius:g}" pos="{xyz}"/>\n')
            else:
                # The vendor collides this link with its FULL visual mesh, which dae2obj split per
                # material; BASE_HULL is the convex hull of every piece, so the collision is the whole
                # chassis rather than whichever piece sorts first. Naming one piece would under-fill
                # the body by 42 mm in length and 35 mm in height here.
                out += (f'{indent}<geom class="collision" mesh="{BASE_HULL}"'
                        f' name="{link.get("name")}_collision" pos="{xyz}"{quat}/>\n')
        return out

    # base_link is an EMPTY link in this description -- no inertial, no geometry -- and base_footprint
    # hangs off it through a fixed joint at the identity, carrying the whole body. The two are folded
    # into one MuJoCo root body here; the manifest's `frames:` re-declares base_footprint so the vendor
    # name stays addressable after the flattening.
    footprint_joint = joints["base_footprint"]
    if pose(footprint_joint) != ("0 0 0", ""):
        raise ValueError(
            f"base_footprint_joint is no longer the identity ({pose(footprint_joint)}); folding it "
            "into base_link would move the body. Re-read the description under the new pin."
        )
    body = links["base_footprint"]

    casters = ""
    for corner in CASTERS:
        name = f"caster_wheel_{corner}_link"
        pos, quat = pose(joints[name])
        casters += FIXED_BODY.format(
            name=name, body_pos=pos, body_quat=quat,
            geoms=geoms(links[name], "          ", tyre_class="caster_collision"),
            **inertial(links[name]),
        )
    wheels = ""
    for side in WHEELS:
        name = f"wheel_{side}_link"
        pos, quat = pose(joints[name])
        wheels += WHEEL_BODY.format(
            name=name, body_pos=pos, body_quat=quat, joint=joints[name].get("name"),
            geoms=geoms(links[name], "            "), **inertial(links[name]),
        )
    # The caster sphere is ours, not the vendor's, so its contact point is checked rather than
    # assumed: the bottom of the ball must still sit on the floor the link is mounted above.
    caster_z = float(pose(joints[f"caster_wheel_{CASTERS[0]}_link"])[0].split()[2])
    drop = caster_z + CASTER_WHEEL_POS[2] - CASTER_WHEEL_RADIUS
    if abs(drop) > 1e-9:
        raise ValueError(
            f"the caster sphere would sit {drop * 1000:+.3f} mm off the floor; re-derive "
            "CASTER_WHEEL_* from diff_caster.dae under the new pin"
        )
    wheel_z = float(pose(joints["wheel_left_link"])[0].split()[2])
    radius = float(links["wheel_left_link"].find("collision/geometry/sphere").get("radius"))
    excludes = "".join(
        f'    <exclude body1="base_link" body2="wheel_{s}_link"/>\n' for s in WHEELS
    ) + "".join(
        f'    <exclude body1="base_link" body2="caster_wheel_{c}_link"/>\n' for c in CASTERS
    )
    return TEMPLATE.format(
        commit=ROX_COMMIT, assets=asset_block(shipped, palette, scales), excludes=excludes,
        base_geoms=geoms(body, "        "), casters=casters, wheels=wheels,
        rest_height=f"{radius - wheel_z:g}",
        **{f"base_{k}": v for k, v in inertial(body).items()},
    )


FIXED_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
{geoms}        </body>
"""

WHEEL_BODY = """        <body name="{name}" pos="{body_pos}"{body_quat}>
          <inertial pos="{pos}" mass="{mass}" diaginertia="{diaginertia}"/>
          <joint name="{joint}" class="wheel"/>
{geoms}        </body>
"""

TEMPLATE = """<mujoco model="rox_diff">
  <!--
    Neobotix ROX-Diff - a DIFFERENTIAL-drive base: two driven wheels on the centre line, four passive
    casters. One of the four kinematics the ROX line ships; the others are not built here.

    GENERATED by external/convert/build_rox_diff_mjcf.py from neobotix/rox @ {commit}
    (BSD as declared in rox_description/package.xml - see rox_diff_LICENSE). Do not hand-edit:
    re-run the generator.

    Drive: a true two-wheel differential drive, so `diff_drive` and NO slip_factor - it does not turn
    by scrubbing. The wheel axis is the vendor's own +y.

    THE WHEEL INERTIA IS FOUR TIMES TOO LARGE, and provably so rather than suspiciously: the declared
    izz 0.05625 at 5 kg is 1/2 m r^2 for r = 0.15 m, the wheel's DIAMETER used where its radius
    belongs. The collision sphere, the 0.075 m axle height and the 0.1499 m visual mesh all agree the
    radius is 0.075 m. The vendor's value is kept so the mass audit checks the description; see the
    port log.

    THE MASSES ARE THE VENDOR'S PLACEHOLDERS. Every inertial in this description is commented "These
    are not accurate value", and the ROX data sheet publishes payload (300 kg) but no dead weight, so
    the 140 kg body has nothing to calibrate against. Use this model for navigation, not dynamics.

    base_link is EMPTY in the description and base_footprint carries the body through a fixed joint at
    the identity; the two are folded into this one root body, and the manifest's `frames:` re-declares
    base_footprint so the vendor frame name survives the flattening.

    The casters are FIXED in the description - passive spheres, not articulated wheels - so they slide
    rather than roll. They carry a low-friction contact class with `priority`, without which MuJoCo
    takes the MAXIMUM of the two contacting geoms' friction, the floor's value wins, and four loaded
    spheres fight every turn.

    The SICK nanoScan3s are not in this file: the manifest mounts the `sick_nanoscan3` device model at
    the vendor's two lidar joints, and that device carries each scanner's housing and mass.
  -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <default>
    <default class="rox_diff">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" rgba="0.6 0.1 0.1 0.35"/>
      </default>
      <default class="wheel_collision">
        <!-- Driven tyres: this base moves because these push on the ground. -->
        <geom type="sphere" group="3" rgba="0.05 0.05 0.05 0.4" friction="1.0 0.005 0.0001"/>
      </default>
      <default class="caster_collision">
        <!-- A passive caster swivels; a fixed sphere cannot, so it stands in for one with a low
             friction and `priority` to make that friction actually apply (without `priority` MuJoCo
             takes the MAXIMUM of the two contacting geoms' friction and the floor's value wins).

             0.01 is a swivel caster's rolling resistance on a hard floor, and on this robot the
             value is load-bearing rather than cosmetic: its six contacts are coplanar and rigid, so
             the vertical split between wheels and casters is indeterminate, and MuJoCo settles it
             with the great majority of the weight on the four casters rather than the drive tyres.
             At the 0.04 its siblings carry, caster drag then EXCEEDS the traction those tyres can
             generate and the robot under-rotates by 11% at 0.3 rad/s. See the port log: the load
             split is the finding, this is the coefficient that is honest about it.

             The sphere itself is the caster WHEEL from the vendor's mesh, not their collision
             primitive -- see CASTER_WHEEL_RADIUS in the generator for why theirs cannot be used. -->
        <geom type="sphere" group="3" rgba="0.6 0.1 0.1 0.35"
              friction="0.01 0.005 0.0001" priority="2"/>
      </default>
      <default class="wheel">
        <!-- The description's own axis, +y in the base frame: its joint carries no rotation. -->
        <joint axis="0 1 0" damping="0.5" armature="0.02" limited="false"/>
      </default>
    </default>
  </default>

  <asset>
{assets}  </asset>

  <contact>
    <!--
      The vendor's base collision IS the full body mesh, and MuJoCo convex-hulls a collision mesh, so
      the hull closes over the wheel and caster pockets and overlaps what sits inside them.
    -->
{excludes}  </contact>

  <worldbody>
    <body name="base_link" childclass="rox_diff">
      <freejoint name="base_free"/>
      <inertial pos="{base_pos}" mass="{base_mass}" diaginertia="{base_diaginertia}"/>
      <site name="base_imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
{base_geoms}{casters}{wheels}    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servos, one per driven wheel. ctrlrange is the vendor's own Nav2 profile
         (rox_navigation/configs/navigation_diff.yaml) 0.8 m/s over the 0.075 m wheel radius, with
         headroom; forcerange is sized to accelerate the declared 155.6 kg well past that file's
         0.25 m/s^2, so the servo tracks rather than saturates. -->
    <velocity name="wheel_left_motor" joint="wheel_left_joint" kv="220" ctrlrange="-14 14" forcerange="-120 120"/>
    <velocity name="wheel_right_motor" joint="wheel_right_joint" kv="220" ctrlrange="-14 14" forcerange="-120 120"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 {rest_height} 1 0 0 0  0 0"/>
  </keyframe>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = resolve_source("rox", ROX_URL, ROX_COMMIT) / ROX_PACKAGE
    target = PKG / "rox_diff.xml"
    with tempfile.TemporaryDirectory() as tmp:
        urdf = expand_xacro({ROX_PACKAGE: source}, source / "urdf/rox.urdf.xacro",
                            Path(tmp), args=XACRO_ARGS)
    clip = base_hull_clip(urdf)
    drop_scanner_links(urdf, SENSOR_LINKS)

    if args.check:
        shipped = {p.stem for p in (PKG / "meshes").glob("*.obj")}
        if not target.exists() or target.read_text() != build(urdf, shipped, mesh_scales(urdf)):
            print(f"{target} differs from a fresh build - was it hand-edited?", file=sys.stderr)
            return 1
        print(f"{target}: up to date with {ROX_COMMIT[:12]}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    scales = convert_meshes(source, urdf, PKG, "rox_diff", ROOT, BUDGETS, pkg=ROX_PACKAGE)
    hull_obj(PKG, subs_for("short_frame", {p.stem for p in (PKG / "meshes").glob("*.obj")}),
             BASE_HULL, scales["short_frame"], clip=clip)
    shipped = {p.stem for p in (PKG / "meshes").glob("*.obj")}
    write_licence()
    target.write_text(build(urdf, shipped, scales))
    print(f"wrote {target} + meshes + rox_diff_LICENSE")
    return 0


def write_licence() -> None:
    """`neobotix/rox` ships no LICENSE file, so there is nothing to copy -- see the text for why.

    The three `neo_simulation2` ports copy that repository's MIT LICENSE beside the model. This one
    cannot: the ROX repository carries no licence text at all, only a declaration in the package
    manifest. Writing the declaration and its provenance is what the sidecar can honestly be.
    """
    (PKG / "rox_diff_LICENSE").write_text(f"""\
The model and the meshes in meshes/ are converted from

    {ROX_URL.removesuffix(".git")}
    commit {ROX_COMMIT}
    files  {ROX_PACKAGE}/urdf/**, {ROX_PACKAGE}/meshes/{{short_frame,diff_wheel,diff_caster}}.dae

Copyright (c) Neobotix GmbH
Licence: BSD, as declared in the package manifest (no licence text ships upstream).
Converted to OBJ in metres and assembled into MJCF by external/convert/build_rox_diff_mjcf.py; every
mass, inertia, joint origin and collision primitive is read from the same repository.

--------------------------------------------------------------------------------

The repository ships no LICENSE file, and the meshes carry no per-file copyright header. The only
licence statement the upstream makes is the one in the package manifest that ships the description:

    {ROX_PACKAGE}/package.xml, at the commit above

      <maintainer email="ros@neobotix.de">Neobotix</maintainer>
      <license>BSD</license>
      <author email="padmanabhan@neobotix.de">Pradheep Padmanabhan</author>

"BSD" names a family, not a document: the 2-clause, 3-clause and original 4-clause texts differ in
what they require. No variant's text is reproduced here, because choosing one would put terms in the
licensor's mouth that the licensor did not write. What every variant does require -- that the
copyright notice and the licence statement travel with the copy -- is what this file is: it names the
copyright holder, the author, the exact upstream revision, and the exact files the model derives
from.

If a downstream use needs the variant pinned (an SPDX-clean SBOM, say), ask Neobotix GmbH to state
it, and replace this paragraph with their answer.
""")


if __name__ == "__main__":
    raise SystemExit(main())
