"""Generate a MoveIt SRDF for a robot exported to URDF.

Why this is generated too
========================
An SRDF is mostly *semantics* a human must choose — which chain is the arm, which joint is the gripper —
but the bulk of its content is a **collision matrix**, and that is a mechanical fact about the geometry.
MoveIt's Setup Assistant produces it by sampling random configurations and recording which link pairs
never touch, always touch, or are adjacent; those get `disable_collisions` entries. Hand-authoring it is
what the G1 config does (58 entries) and it is exactly the kind of list that goes stale silently: the
URDF here is regenerated whenever a mount pose or a gripper changes, and a stale matrix then either
blocks every plan (a pair that now always touches is still checked) or hides a real self-collision.

So the semantics come from arguments and the matrix is computed, from the **same MuJoCo model the URDF
was exported from**. The sampling is the Setup Assistant's algorithm; MuJoCo is just a faster collision
engine to run it in.

What the matrix means, and why "never" is safe to disable
--------------------------------------------------------
Three reasons a pair is disabled, matching the Setup Assistant's own vocabulary:

* ``Adjacent`` — the two links are joined by a joint. They almost always touch at their shared axis, and
  no planner can help that, so it must be disabled or every state is in collision.
* ``Always`` — they collide in every sampled configuration (a shroud overlapping its neighbour). Same
  consequence.
* ``Never`` — they collided in none of ``--samples`` configurations. Disabling these is what makes
  collision checking affordable; it is a sampling argument, so it is only as good as the sample count,
  which is why the count is recorded in a comment in the output.

Links whose geometry is visual-only carry no `<collision>` in the exported URDF at all, so they can
never collide and land in ``Never`` -- consistent, not a special case.

The base joint is computed too
------------------------------
Whether the robot's base FLOATS is the other mechanical fact in this file, and it used to be assumed:
a ``planar`` ``virtual_joint`` from ``odom`` was emitted unconditionally, which is right for the mobile
manipulators the substrate mostly serves and wrong for an arm bolted to a pedestal. It is readable from
the model -- a floating base rides a MuJoCo free joint, a bolted one is welded to the world -- so
``resolve_base_joint`` reads it instead, and ``--base-joint`` only overrides the verdict.

Getting it wrong is expensive and nearly silent, which is why it is computed and why the two
directions are treated differently. A virtual joint nothing publishes leaves move_group logging "The
complete state of the robot is not yet known. Missing virtual_joint" once a second, never assembling a
complete robot state; planning does not fail, it degrades. Measured on a fixed-base UR5e cell: 7/28
queries solved with the joint present against 28/28 without it. So asking for one on a welded model is
refused outright, while suppressing one on a floating model is only warned about -- that is a
legitimate choice for a parked base whose pose is not planned for.

Usage::

    roqsim export srdf --world w.yaml --strip ur5e_ --name husky_ur5e_2f85 \\
        --arm-base base --arm-tip wrist_3_link --gripper-joint robotiq_85_left_knuckle_joint \\
        --gripper-open 0.0 --gripper-close 0.8 --out cfg/husky_ur5e_2f85.srdf
"""

from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np

from . import logging_setup

logger = logging.getLogger(__name__)

#: What the exported robot's base is, in MoveIt's terms. ``fixed`` emits no ``virtual_joint`` at all,
#: which is how MoveIt spells a bolted-down robot ("No root/virtual joint specified in SRDF. Assuming
#: fixed joint"). ``auto`` reads the answer off the model; see ``resolve_base_joint``.
BASE_JOINTS = ("auto", "planar", "floating", "fixed")


def _body_of_geom(model, geom: int) -> int:
    return int(model.geom_bodyid[geom])


def exported_root_body(model: mujoco.MjModel, links: dict[int, str]) -> int:
    """The MuJoCo body that is the exported URDF's ROOT link.

    A URDF is a single tree, so exactly one mapped body has an unmapped parent. More than one means
    ``links`` reaches outside the robot -- a URDF link name matched a body in another subtree -- and
    every base verdict below it would be about the wrong kinematic chain.
    """
    roots = [b for b in links if int(model.body_parentid[b]) not in links]
    if len(roots) != 1:
        names = sorted(links[b] for b in roots)
        raise ValueError(
            f"expected exactly one root link in the exported robot, found {len(roots)}: {names}. "
            "A URDF has a single root, so this means a URDF link name matched a MuJoCo body outside "
            "the robot (check --strip and the model's name prefixes)."
        )
    return roots[0]


def base_free_joint(model: mujoco.MjModel, links: dict[int, str]) -> str | None:
    """Name of the free joint the exported robot's base rides, or ``None`` if it is welded down.

    The walk goes past the exported subtree up to the world body, because a mounted arm's own root is
    welded and the free joint belongs to the base it rides (``spawn_arm``'s ``mount:`` welds the arm
    onto a ``spawn_robot`` body, and that body carries the ``freejoint``). Only a FREE joint counts: a
    hinge or slide above the root -- a rail carriage, a gantry bridge -- is a real URDF joint that
    ``roqsim export urdf`` emits below a synthetic fixed ``world`` link, so MoveIt already has the fixed
    root it requires and needs no virtual joint for it.
    """
    body = exported_root_body(model, links)
    while body > 0:
        adr, num = int(model.body_jntadr[body]), int(model.body_jntnum[body])
        for j in range(adr, adr + num):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                return name or f"<unnamed joint {j}>"
        body = int(model.body_parentid[body])
    return None


def resolve_base_joint(
    model: mujoco.MjModel, links: dict[int, str], requested: str = "auto"
) -> tuple[str, str]:
    """``(base_joint, why)`` -- what to emit for the base, and the evidence for it.

    The reason travels with the verdict because it ends up in the generated file: a reader of the SRDF
    must be able to see WHY it has no virtual joint without re-deriving it from the MJCF.
    """
    if requested not in BASE_JOINTS:
        raise ValueError(f"base_joint must be one of {', '.join(BASE_JOINTS)}, not {requested!r}")

    free = base_free_joint(model, links)
    detected = f"free joint {free}" if free else "welded to the world, no free joint above the root"

    if requested == "auto":
        # A wheeled base is planar and a humanoid/legged one is floating, and BOTH are a MuJoCo free
        # joint -- the difference is a modelling choice about what may be planned for, not geometry. So
        # auto picks planar (the substrate's mobile bases) and a floating base must be asked for.
        return ("planar" if free else "fixed"), detected

    if requested in ("planar", "floating") and free is None:
        raise ValueError(
            f"--base-joint {requested} was requested, but this robot's base is {detected}. Nothing "
            f"would publish the {requested} transform below the planning frame, so move_group would log "
            "'The complete state of the robot is not yet known. Missing virtual_joint' forever and "
            "never assemble a complete robot state (measured on a fixed-base cell: 7/28 queries "
            "solved with the joint present, 28/28 without). Use --base-joint fixed, or auto."
        )
    if requested == "fixed" and free is not None:
        logger.warning(
            "--base-joint fixed on a robot whose base rides %s: MoveIt will plan as if the base sat "
            "at the planning frame's origin. Correct only if the base is parked and its pose is not "
            "planned for; otherwise poses given in world coordinates will be off by the base's pose.",
            detected,
        )
    return requested, detected


def collision_matrix(
    model: mujoco.MjModel, links: dict[int, str], samples: int = 10000, seed: int = 0
) -> list[tuple[str, str, str]]:
    """``(link_a, link_b, reason)`` for every pair MoveIt should stop checking.

    ``links`` maps MuJoCo body id -> URDF link name; only those bodies are considered, so the caller
    decides what "the robot" is (and a collapsed gripper's members map to one link, which is what makes
    the matrix agree with the URDF rather than with the MJCF).
    """
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)

    _require_self_collision(model, links)
    return _sample_matrix(model, data, links, rng, samples)


def unmask_self_collision(spec: mujoco.MjSpec) -> int:
    """Let the robot collide with itself in ``spec``, returning how many geoms were changed.

    Must be called BEFORE ``compile()``: MuJoCo decides at compile time which geoms can ever collide,
    so clearing the mask on an already-compiled ``MjModel`` has no effect (measured -- the contacts
    stay at zero).

    Why it is needed at all: contype/conaffinity is a SIMULATION choice, while the collision matrix
    asks a GEOMETRIC question. Robot models here routinely disable self-collision on purpose --
    ``contype=2 / conaffinity=1`` on every collision geom, because the vendor's collision primitives of
    neighbouring links overlap at rest and would fight (the TIAGo Pro's port log calls it A7; the G2
    port does the same). Sampling such a model finds no contacts at all, so every pair is marked
    ``Never`` and the SRDF switches MoveIt's self-collision checking off entirely.

    Visual-only geoms (both flags already 0) are left alone: they carry no ``<collision>`` in the
    exported URDF either, so a link made only of them genuinely cannot collide.
    """
    changed = 0
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 1, 1
            changed += 1
    return changed


def _require_self_collision(model: mujoco.MjModel, links: dict[int, str]) -> None:
    """Refuse to sample a model whose links are masked apart -- the matrix would be meaningless.

    Every pair would come back ``Never`` on no evidence, which is not a conservative default but the
    most dangerous possible SRDF: it disables self-collision checking for the whole robot, and nothing
    downstream says so. Measured on the TIAGo Pro before this check existed: 0 contacts over 200 random
    poses, 1711 pairs marked ``Never``, 0 marked ``Always``.
    """
    collidable = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in links
        and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
    ]
    for a, b in combinations(collidable, 2):
        if links[int(model.geom_bodyid[a])] == links[int(model.geom_bodyid[b])]:
            continue
        if (int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) or (
            int(model.geom_contype[b]) & int(model.geom_conaffinity[a])
        ):
            return
    raise ValueError(
        "no two links in this model can collide with each other: their geoms are masked apart by "
        "contype/conaffinity, so sampling would mark every pair 'Never' and produce an SRDF that "
        "disables MoveIt's self-collision checking entirely. Compile the sampling model with "
        "`unmask_self_collision(spec)` before `spec.compile()` -- clearing the mask after compiling "
        "does not work."
    )


def _sample_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    links: dict[int, str],
    rng: np.random.Generator,
    samples: int,
) -> list[tuple[str, str, str]]:
    """The sampling itself. Split out so ``collision_matrix`` can restore the contact mask on any exit."""

    # Everything below counts in LINK space, not body space. Several MuJoCo bodies can share one URDF
    # link (that is what `--collapse` does), and MoveIt checks links -- so counting per body pair would
    # emit a link against ITSELF, and emit the same link pair twice with contradictory reasons when two
    # members of a collapsed gripper disagree.
    def key(a: int, b: int) -> tuple[str, str] | None:
        la, lb = links[a], links[b]
        return None if la == lb else (min(la, lb), max(la, lb))

    # Joints to randomise: the hinges/slides that EXIST IN THE URDF, i.e. those of bodies that are a
    # link in their own right. A joint inside a collapsed subtree is not in the URDF and its geometry
    # was baked into the lumped link at export time, so moving it here would explore configurations
    # MoveIt cannot reach and misclassify pairs as "sometimes". A free base is left alone too -- the
    # base's pose is not part of self-collision.
    def is_link_root(body: int) -> bool:
        return links.get(int(model.body_parentid[body])) != links[body]

    movable = [
        j
        for j in range(model.njnt)
        # int(): mjt* enums compare False against a numpy scalar (`x in (...)` puts the enum on the
        # left of `==`); an empty `movable` would sample nothing and mark every pair Always.
        if int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        and int(model.jnt_bodyid[j]) in links
        and is_link_root(int(model.jnt_bodyid[j]))
    ]

    collided: dict[tuple[str, str], int] = {
        k: 0 for a, b in combinations(sorted(links), 2) if (k := key(a, b)) is not None
    }
    total_contacts = 0

    for _ in range(samples):
        for j in movable:
            lo, hi = (float(v) for v in model.jnt_range[j])
            if hi <= lo:  # unlimited joint
                lo, hi = -np.pi, np.pi
            data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
        # Positions + collision detection only; no dynamics, so this is cheap per sample.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        seen = set()
        for c in range(data.ncon):
            a = _body_of_geom(model, data.contact[c].geom1)
            b = _body_of_geom(model, data.contact[c].geom2)
            if a in links and b in links and (k := key(a, b)) is not None:
                seen.add(k)
        for k in seen:
            collided[k] += 1
        total_contacts += len(seen)

    # Adjacency in the MuJoCo body tree, mapped through `links` so a collapsed subtree's internal
    # adjacency disappears with it (both members are the same link, and `key` drops that).
    adjacent = set()
    for body in links:
        parent = int(model.body_parentid[body])
        while parent > 0 and parent not in links:
            parent = int(model.body_parentid[parent])
        if parent in links and (k := key(body, parent)) is not None:
            adjacent.add(k)

    if total_contacts == 0 and len(set(links.values())) > 1:
        # Every pair is about to be marked `Never` on the strength of no evidence at all, which yields an
        # SRDF that disables MoveIt's self-collision checking entirely. A real articulated robot put in
        # thousands of random configurations touches itself somewhere; zero means the geometry never got
        # a chance to collide.
        logger.warning(
            "no self-contacts in %d sampled configurations: every pair will be marked 'Never', which "
            "disables self-collision checking for the whole robot. Check that the model's collision "
            "geoms are not masked apart (contype/conaffinity) and that the sampled joints have ranges.",
            samples,
        )

    out: list[tuple[str, str, str]] = []
    for (a, b), count in sorted(collided.items()):
        if (a, b) in adjacent:
            reason = "Adjacent"
        elif count == 0:
            reason = "Never"
        elif count >= samples:
            reason = "Always"
        else:
            continue  # sometimes collides: this is exactly what MoveIt must keep checking
        out.append((a, b, reason))
    return out


def build_srdf(
    model: mujoco.MjModel,
    links: dict[int, str],
    *,
    name: str,
    arm_base: str,
    arm_tip: str,
    gripper_joint: str,
    gripper_open: float,
    gripper_close: float,
    home: dict[str, float],
    base_joint: str = "auto",
    base_link: str | None = None,
    parent_frame: str = "odom",
    samples: int = 10000,
) -> ET.ElementTree:
    base_joint, why = resolve_base_joint(model, links, base_joint)
    root_link = links[exported_root_body(model, links)]
    logger.info(
        "base joint: %s (%s); MoveIt's planning frame is %s",
        base_joint,
        why,
        root_link if base_joint == "fixed" else parent_frame,
    )

    robot = ET.Element("robot", name=name)
    robot.append(
        # NB: no double hyphen anywhere in this text. XML forbids `--` inside a comment; ElementTree
        # writes it without complaint and then neither it nor MoveIt can parse the file back. That
        # rules out naming the generator's own options here.
        ET.Comment(
            f" GENERATED by `roqsim export srdf`. Do not edit; regenerate with `make srdf`.\n"
            f"       Groups and named states come from the generator's arguments; every\n"
            f"       disable_collisions entry below was COMPUTED from the same MuJoCo model the URDF\n"
            f"       was exported from, by sampling {samples} random configurations (the algorithm\n"
            f"       MoveIt's Setup Assistant uses). A pair marked 'Never' collided in none of them.\n"
            f"       Base joint: {base_joint} ({why}), read from that model as well. "
        )
    )

    if base_joint == "fixed":
        # No virtual_joint AT ALL is how MoveIt spells a bolted-down robot; it then logs "No
        # root/virtual joint specified in SRDF. Assuming fixed joint" and the planning frame is the
        # root link itself. Declaring type="fixed" would be the other spelling, but it moves the
        # planning frame to a parent_frame that nothing in the URDF or TF tree provides.
        robot.append(ET.Comment(f" No virtual_joint: this robot is {why}. "))
    else:
        # The base floats: its pose comes from TF (parent_frame -> root link), published by the
        # bridge's odometry and corrected by AMCL. MoveIt therefore knows where the base is without
        # planning for it, and planning happens in the arm group only -- which is why the base is
        # allowed to drift.
        child_link = base_link or root_link
        if child_link != root_link:
            # srdfdom checks this and SKIPS the joint on mismatch, which degrades exactly like having
            # no virtual joint at all -- with an error line that is easy to miss in move_group's log.
            raise ValueError(
                f"virtual joint child_link {child_link!r} is not the exported URDF's root link "
                f"({root_link!r}). MoveIt requires the virtual joint's child to BE the model root and "
                f"skips it otherwise, leaving the robot state incomplete. Pass base_link={root_link!r} "
                "or leave it unset."
            )
        ET.SubElement(
            robot,
            "virtual_joint",
            name="virtual_joint",
            type=base_joint,
            parent_frame=parent_frame,
            child_link=child_link,
        )

    arm = ET.SubElement(robot, "group", name="arm")
    ET.SubElement(arm, "chain", base_link=arm_base, tip_link=arm_tip)

    gripper = ET.SubElement(robot, "group", name="gripper")
    ET.SubElement(gripper, "joint", name=gripper_joint)

    ET.SubElement(
        robot,
        "end_effector",
        name="hand",
        parent_link=arm_tip,
        group="gripper",
        parent_group="arm",
    )

    # Named states. `open`/`closed` are in the gripper JOINT's units, matching the values the gripper's
    # manifest gives arm_controller -- so a MoveIt named target and a GripperCommand agree.
    for state, value in (("open", gripper_open), ("closed", gripper_close)):
        grp = ET.SubElement(robot, "group_state", name=state, group="gripper")
        ET.SubElement(grp, "joint", name=gripper_joint, value=f"{value:.6g}")
    if home:
        grp = ET.SubElement(robot, "group_state", name="home", group="arm")
        for joint, value in home.items():
            ET.SubElement(grp, "joint", name=joint, value=f"{value:.6g}")

    matrix = collision_matrix(model, links, samples=samples)
    for a, b, reason in matrix:
        ET.SubElement(robot, "disable_collisions", link1=a, link2=b, reason=reason)
    logger.info(
        "collision matrix: %d disabled pairs (%d adjacent, %d always, %d never) from %d samples",
        len(matrix),
        sum(1 for _, _, r in matrix if r == "Adjacent"),
        sum(1 for _, _, r in matrix if r == "Always"),
        sum(1 for _, _, r in matrix if r == "Never"),
        samples,
    )
    return ET.ElementTree(robot)


def links_from_urdf(model: mujoco.MjModel, urdf: Path, strip: str) -> dict[int, str]:
    """Map MuJoCo body id -> URDF link name, for the links the exported URDF actually has.

    Read from the URDF rather than recomputed so the two files cannot disagree about what a link is --
    which matters most for a collapsed gripper, where several MuJoCo bodies became one URDF link.
    """
    # noqa S314: the URDF being read back is the one this exporter just wrote, not caller input.
    root = ET.parse(urdf).getroot()  # noqa: S314
    names = {link.get("name") for link in root.findall("link")}
    parent_of = {
        j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint")
    }

    def link_name_of(body: int) -> str | None:
        raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        stripped = raw[len(strip) :] if strip and raw.startswith(strip) else raw
        return next((c for c in (raw, stripped) if c in names), None)

    out: dict[int, str] = {}
    for body in range(1, model.nbody):
        # Walk up to the nearest ancestor that IS a URDF link: that is exactly where `--collapse` folded
        # this body's geometry. Without the walk, a collapsed subtree's members were dropped from the
        # matrix entirely -- and a dropped pair means "keep checking", so a finger that overlaps the
        # wrist in every reachable state would block every plan with nothing to explain why.
        #
        # The walk stops at the world body, so a body under no exported link (the mobile base the arm is
        # mounted on, a prop, a second robot) stays unmapped rather than having its contacts blamed on
        # the arm.
        cur = body
        while cur > 0:
            if (name := link_name_of(cur)) is not None:
                out[body] = name
                break
            cur = int(model.body_parentid[cur])

    # The ROOT link is matched by name like any other -- unless `roqsim export urdf --root-link` RENAMED
    # it, which is the normal case (a Menagerie arm's root body is `base`, MoveIt and nav2 want
    # `base_link`). Then no body carries the name, the root stays unmapped, and its geometry is absent
    # from the matrix: no `Adjacent` entry for root/first link, so MoveIt keeps checking a pair that
    # touches at its shared axis in every state. So follow the URDF's own parent chain up from the
    # topmost mapped link, stepping one body up in MuJoCo for each URDF link, and name what we pass.
    # The chain stops at the world body, which leaves a SYNTHETIC ancestor unmapped on purpose: a rail
    # carriage's fixed `world` link has no MuJoCo body to claim.
    if not out:  # nothing matched at all; the caller reports that, with the prefix it used
        return out
    body = exported_root_body(model, out)
    link = out[body]
    while (parent_link := parent_of.get(link)) is not None and parent_link not in out.values():
        body = int(model.body_parentid[body])
        if body == 0:
            break
        out[body] = parent_link
        link = parent_link
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export srdf",
        description="Generate a MoveIt SRDF (incl. a sampled collision matrix) for an roqsim robot.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--world", help="world YAML (compiled via the plugin pipeline)")
    source.add_argument("--mjcf", help="a bare MJCF file")
    parser.add_argument("--urdf", required=True, help="the URDF `roqsim export urdf` produced")
    parser.add_argument("--out", required=True, help="output .srdf path")
    parser.add_argument("--name", required=True, help="robot name (must match the URDF's)")
    parser.add_argument("--strip", default="", help="prefix stripped from link names in the URDF")
    parser.add_argument("--arm-base", required=True, help="first link of the arm chain")
    parser.add_argument(
        "--arm-tip", required=True, help="last link of the arm chain (the tool flange)"
    )
    parser.add_argument(
        "--base-joint",
        default="auto",
        choices=BASE_JOINTS,
        help="how the base attaches to the planning frame. Default `auto` READS IT OFF THE MODEL: a "
        "base riding a MuJoCo free joint gets a planar virtual_joint, a base welded to the world gets "
        "none (which is how MoveIt spells a bolted-down robot). `floating` for a base with full 6 DOF "
        "(humanoid, legged, aerial), `fixed` to force no virtual joint. Asking for planar/floating on "
        "a welded model is refused: nothing would publish the transform, and move_group then never "
        "assembles a complete robot state",
    )
    parser.add_argument(
        "--base-link",
        default=None,
        help="child link of the virtual joint. Defaults to the exported URDF's root link, which is "
        "the only value MoveIt accepts -- set it only to assert that root explicitly",
    )
    parser.add_argument(
        "--parent-frame",
        default="odom",
        help="parent frame of the virtual joint, i.e. MoveIt's PLANNING FRAME (ignored without a "
        "virtual joint, where the planning frame is the root link itself). Every pose handed to "
        "move_group (collision objects, goals) must be expressed in a frame reachable from this one, "
        "or the planning scene rejects it with `Unknown frame: <x>` and no plan can ever succeed. "
        "`odom` suits a base whose drift is not corrected; a mobile manipulator whose objects are "
        "given in world/map coordinates wants `map` (AMCL then supplies map->odom).",
    )
    parser.add_argument("--gripper-joint", required=True)
    parser.add_argument("--gripper-open", type=float, required=True)
    parser.add_argument("--gripper-close", type=float, required=True)
    parser.add_argument(
        "--home",
        default="",
        help="comma-separated joint=value pairs for the arm's `home` named state",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="random configurations sampled for the collision matrix (Setup Assistant default: 10000)",
    )
    parser.add_argument(
        "--skip-plugins",
        default="",
        help="extra plugin names/refs to drop before compiling; transport/bridge plugins are always "
        "dropped (they contribute no geometry)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_srdf")

    from .export_web import _compile_from_world

    skip = {s.strip() for s in args.skip_plugins.split(",") if s.strip()}
    if args.mjcf:
        # Compiled here rather than through `_compile_from_mjcf`, because the mask has to be cleared on
        # the SPEC: MuJoCo fixes what can collide at compile time, so a model whose robot geoms are
        # masked apart (the usual `contype=2 / conaffinity=1`) can never be made to self-collide
        # afterwards. This is a sampling model, not a simulation one.
        spec = mujoco.MjSpec.from_file(str(Path(args.mjcf)))
        changed = unmask_self_collision(spec)
        log.info("enabled self-collision on %d geoms for sampling", changed)
        model = spec.compile()
    else:
        # The world pipeline compiles the spec inside the engine and does not hand it back, so the mask
        # cannot be cleared on this path. `collision_matrix` refuses a masked model rather than
        # producing an SRDF that silently disables self-collision checking -- export a robot's
        # description from its MJCF (`--mjcf`) instead, which is where a robot description belongs.
        model, _d, _v = _compile_from_world(args.world, skip, {}, log)

    links = links_from_urdf(model, Path(args.urdf), args.strip)
    if not links:
        raise SystemExit(f"no URDF link matched a body in the model (strip={args.strip!r})")

    home = {}
    for item in (p for p in args.home.split(",") if p.strip()):
        joint, _, value = item.partition("=")
        home[joint.strip()] = float(value)

    # Every ValueError this module raises describes an SRDF that would be WRONG (a base joint nothing
    # publishes, a child link MoveIt would skip, a model masked out of self-collision). Those are
    # answers to the operator, not bugs, so they read as a message rather than as a traceback.
    try:
        tree = build_srdf(
            model,
            links,
            name=args.name,
            arm_base=args.arm_base,
            arm_tip=args.arm_tip,
            gripper_joint=args.gripper_joint,
            gripper_open=args.gripper_open,
            gripper_close=args.gripper_close,
            home=home,
            base_joint=args.base_joint,
            base_link=args.base_link,
            parent_frame=args.parent_frame,
            samples=args.samples,
        )
    except ValueError as err:
        raise SystemExit(f"roqsim export srdf: {err}") from err
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
