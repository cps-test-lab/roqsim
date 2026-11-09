# SPDX-License-Identifier: Apache-2.0
"""Retargeting tables: a body mapped onto the canonical humanoid, one table per body.

A new character or a new humanoid robot is one table, not new code. A robot's table is YAML beside
its model, named ``<model>.retarget.yaml`` (found through the ``roqsim.models`` providers as the
model is). A walker needs none: its skeleton is the canonical humanoid itself, with its blueprint's
bone lengths (:class:`~roqsim_motion.description.Description`)::

    humanoid: unitree_g1_dex1
    base: {body: pelvis, height: 0.79}       # the root body, and its height standing
    joints:                                  # canonical joint -> its degrees of freedom, in the
      spine: [z: waist_yaw_joint,            #   order the body's chain has them (parent first);
              x: waist_roll_joint,           #   `axis: joint` or `axis: {joint, sign, offset}`
              y: waist_pitch_joint]
      elbow_l: [y: left_elbow_joint]
    hands:
      l: {frame: left_grasp, rpy: [0, 0, 0],   # the frame site (or `body:` and `pos:`), and the turn
          grip: {actuator: left_dex1_gripper, open: 0.0, closed: 0.04}}   # onto the grip convention
    gaze: spine                              # the joint that turns to look (head by default)
    rest: {left_elbow_joint: 1.3}            # angles at rest, where the model's zero is not
    physical: {servos: own, kp: 300.0, kv: 10.0}

A joint's angle is its canonical local rotation decomposed in the order listed, each angle times
``sign`` plus ``offset``. Limits are read from the model, never restated here. A table naming a
joint, site or actuator the model lacks is refused by :meth:`Table.check`, naming the table, the key
and what the model has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from . import quat as Q
from .description import JOINTS


class TableError(ValueError):
    """A retargeting table that is malformed, or names what its model lacks."""


@dataclass(frozen=True)
class Dof:
    axis: str
    joint: str
    sign: float = 1.0
    offset: float = 0.0


@dataclass
class Table:
    name: str
    base_body: str
    base_height: float
    joints: dict  # canonical joint -> [Dof, ...] in chain order
    hands: dict = field(default_factory=dict)  # side -> {"frame": site, "grip": {...}}
    gaze: str = "head"
    physical: dict = field(default_factory=dict)
    source: str = ""
    rest: dict = field(default_factory=dict)  # model joint -> its angle standing at rest (rad)

    def angles(self, local: dict) -> dict:
        """Model joint name -> angle (rad) for the canonical local rotations ``local``."""
        out = {}
        for joint, dofs in self.joints.items():
            rot = local.get(joint)
            if rot is None:
                continue
            order = "".join(d.axis for d in dofs)
            if len(order) == 1:
                values = [_single_angle(rot, order)]
            elif len(order) == 2:
                values = Q.euler(rot, order + _third(order))[:2]
            else:
                values = Q.euler(rot, order)
            for d, a in zip(dofs, values, strict=True):
                out[d.joint] = d.sign * a + d.offset
        return out

    def model_joints(self) -> list[str]:
        return [d.joint for dofs in self.joints.values() for d in dofs]

    def check(self, model, prefix: str = "") -> None:
        """Refuse a table that names a body, joint, site or actuator ``model`` lacks."""

        def has(kind, name):
            return mujoco.mj_name2id(model, kind, prefix + name) >= 0

        missing = []
        if not has(mujoco.mjtObj.mjOBJ_BODY, self.base_body):
            missing.append(f"base.body {self.base_body!r}")
        for j in self.model_joints():
            if not has(mujoco.mjtObj.mjOBJ_JOINT, j):
                missing.append(f"joint {j!r}")
        for side, hand in self.hands.items():
            if "frame" in hand and not has(mujoco.mjtObj.mjOBJ_SITE, hand["frame"]):
                missing.append(f"hands.{side}.frame {hand['frame']!r} (a site)")
            if "body" in hand and not has(mujoco.mjtObj.mjOBJ_BODY, hand["body"]):
                missing.append(f"hands.{side}.body {hand['body']!r}")
            act = (hand.get("grip") or {}).get("actuator")
            if act and not has(mujoco.mjtObj.mjOBJ_ACTUATOR, act):
                missing.append(f"hands.{side}.grip.actuator {act!r}")
        if missing:
            have = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(model.njnt)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith(prefix)
            ]
            raise TableError(
                f"retargeting table {self.name!r} ({self.source or 'inline'}) names what the model "
                f"lacks under prefix {prefix!r}: {', '.join(missing)}. Its joints: {', '.join(have)}"
            )


def _third(order: str) -> str:
    return next(a for a in "xyz" if a not in order)


def _single_angle(rot, axis: str) -> float:
    """The angle of ``rot`` about ``axis``: its swing-twist twist part."""
    v = np.asarray(rot, float)
    i = "xyz".index(axis) + 1
    return 2.0 * math.atan2(v[i], v[0]) if abs(v[0]) + abs(v[i]) > 1e-12 else 0.0


def load_table(path) -> Table:
    """Read a table from YAML, refusing an unknown key or canonical joint by name."""
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise TableError(f"retargeting table {path}: {exc}") from exc
    return parse_table(doc, source=str(path))


def parse_table(doc: dict, source: str = "") -> Table:
    if not isinstance(doc, dict):
        raise TableError(f"retargeting table {source}: a mapping, not {type(doc).__name__}")
    known = {"humanoid", "base", "joints", "hands", "gaze", "physical", "rest"}
    extra = sorted(set(doc) - known)
    if extra:
        raise TableError(
            f"retargeting table {source}: unknown key(s) {extra}; the keys are {sorted(known)}"
        )
    try:
        name = str(doc["humanoid"])
        base = doc["base"]
        base_body, height = str(base["body"]), float(base["height"])
    except (KeyError, TypeError) as exc:
        raise TableError(
            f"retargeting table {source}: needs 'humanoid' and 'base: {{body, height}}'"
        ) from exc
    joints = {}
    for joint, dofs in (doc.get("joints") or {}).items():
        if joint not in JOINTS:
            raise TableError(
                f"retargeting table {source}: {joint!r} is no canonical joint ({', '.join(JOINTS)})"
            )
        if dofs in (None, "none"):
            continue
        out = []
        for entry in dofs:
            ((axis, value),) = entry.items()
            if axis not in ("x", "y", "z"):
                raise TableError(f"retargeting table {source}: {joint}: axis {axis!r} is x, y or z")
            if isinstance(value, dict):
                out.append(
                    Dof(
                        axis,
                        str(value["joint"]),
                        float(value.get("sign", 1.0)),
                        float(value.get("offset", 0.0)),
                    )
                )
            else:
                out.append(Dof(axis, str(value)))
        joints[joint] = out
    hands = {}
    for side, hand in (doc.get("hands") or {}).items():
        if side not in ("l", "r"):
            raise TableError(f"retargeting table {source}: hands are 'l' and 'r', not {side!r}")
        if ("frame" in hand) == ("body" in hand):
            raise TableError(
                f"retargeting table {source}: hands.{side} names a 'frame' (a site) or a 'body', one of them"
            )
        hands[side] = {"grip": dict(hand.get("grip") or {})}
        if "frame" in hand:
            hands[side]["frame"] = str(hand["frame"])
        else:
            hands[side]["body"] = str(hand["body"])
            hands[side]["pos"] = [float(v) for v in hand.get("pos", [0.0, 0.0, 0.0])]
        if "rpy" in hand:
            hands[side]["rpy"] = [float(v) for v in hand["rpy"]]
    rest = {str(k): float(v) for k, v in (doc.get("rest") or {}).items()}
    mapped = {d.joint for dofs in joints.values() for d in dofs}
    unmapped = sorted(set(rest) - mapped)
    if unmapped:
        raise TableError(
            f"retargeting table {source}: rest names {unmapped}, which no canonical joint maps"
        )
    return Table(
        name,
        base_body,
        height,
        joints,
        hands,
        str(doc.get("gaze", "head")),
        dict(doc.get("physical") or {}),
        source,
        rest,
    )


def find_table(model_name: str) -> Table:
    """The table beside model ``model_name`` (``<model>.retarget.yaml``), found as the model is."""
    from roqsim.models import resolve_model

    asset = resolve_model(model_name)
    path = Path(asset.path).with_name(f"{model_name}.retarget.yaml")
    if not path.is_file():
        raise TableError(
            f"model {model_name!r} has no retargeting table: expected {path} beside its model"
        )
    return load_table(path)
