"""Scene plugin: slots on a body -- where a worker stands to work at it, and where its hand goes.

A slot is sites on a body, by a naming convention (``docs/plugins.rst``, "Slots"): a **hand
target** and one or more **stances**.

* The hand target is a site named for the slot. ``kind: grip`` makes it ``<entry>/<slot>/grip``,
  in the grip-frame convention of ``roqsim_motion.hands`` (+x out of the palm toward what it takes,
  +z along the fingers, down), so ``grasp`` takes it. ``kind: place`` makes it ``<entry>/<slot>``,
  +z up, its yaw the heading what is set there is set down at, so ``place`` sets down on it.
  ``kind: press`` makes it ``<entry>/<slot>``, +x out of the face, so ``press`` pushes along -x.
* Each stance is a site ``<entry>/<slot>/stand`` (``/stand_<k>`` for several), +x the way to face.
  ``roqsim_motion``'s ``approach`` and ``carry`` to the slot, or to its grip, walk to it -- the
  nearest of several -- where they are given no ``standoff`` or ``facing``.

A prop whose plugin states its own slots needs none of this; this states them on anything else --
a box standing in for a rack, a mesh model, a body of a baked scene. The poses are in the body's
frame (a ``box``'s origin is its centre), so a slot on a drawer moves with the drawer. A stance is
put at the body's origin height: what it says is where it stands on the floor and its heading.

Nested under the entry it states slots on, whose name its sites carry::

    - box: {prefix: rack_, pose: {position: {x: 2.0, y: 0.0}}, size: [0.6, 1.2, 0.9], motion: static}
      name: rack
      components:
        - slots:
            body: rack_box      # the body the slots are on, as the model names it (required)
            slots:
              bin_0:
                kind: grip      # grip | place | press (default place)
                at: [0.1, 0.0, 0.5, 3.14159]    # x, y, z, yaw in the body's frame (m, rad)
                stand: [[0.6, 0.0, 3.14159]]    # x, y, yaw in the body's frame, on the floor

The sites are ``<entry>/<slot>/grip`` (or ``<entry>/<slot>``) and ``<entry>/<slot>/stand[_<k>]``,
named for the entry, not for the body, so a scenario names a slot as it names the entity. It adds
them in ``build``, after the entry it is nested under has built its body, and before a humanoid that
grasps one of its grips is built (a humanoid's holds are made in ``build_late``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin, PluginError
from roqsim.schema import Field

KINDS = ("grip", "place", "press")
#: What a slot states; anything else is a mistake that would leave it without a stance or target.
KEYS = ("kind", "at", "stand")


def _numbers(v) -> bool:
    """Whether ``v`` is a list of finite numbers."""
    return isinstance(v, (list, tuple)) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v
    )


def _yaw_quat(yaw: float) -> list:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _grip_quat(yaw: float) -> list:
    """The grip frame of a hand pushing along ``yaw``: +x that way, +z down, +y = z x x."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([c, s, 0.0, s, -c, 0.0, 0.0, 0.0, -1.0])  # row-major; columns x, y, z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot)
    return [float(v) for v in quat]


class SlotsPlugin(Plugin):
    """Adds a body's slots -- hand targets and stances -- as sites, at build."""

    #: States slots on the entry it is nested under: it belongs inside that entry's ``components:``.
    requires_owner = True

    CONFIG_SCHEMA = {
        "body": Field(str, required=True, doc="the body the slots are on, as the model names it"),
        "slots": Field(dict, required=True, doc="slot name -> {kind, at, stand}"),
    }

    @classmethod
    def validate_config(cls, config) -> list:
        errors = []
        slots = config.get("slots")
        if not isinstance(slots, dict) or not slots:
            return ["'slots' is a mapping of slot names to {kind, at, stand}"]
        for name, slot in slots.items():
            kind = slot.get("kind", "place") if isinstance(slot, dict) else "place"
            if not isinstance(name, str) or not name or "/" in name or name.startswith("stand"):
                errors.append(
                    f"slot {name!r}: a slot's name has no '/' and does not begin with 'stand', "
                    f"which names a stance"
                )
            elif name.startswith("grip") and kind != "grip":
                errors.append(
                    f"slot {name!r}: a {kind} slot's name does not begin with 'grip': its site "
                    f"would read as a grip"
                )
            if not isinstance(slot, dict):
                errors.append(f"slot {name!r}: a mapping of kind, at and stand")
                continue
            for key in slot:
                if key not in KEYS:
                    errors.append(
                        f"slot {name!r}: unknown key {key!r}; a slot states {', '.join(KEYS)}"
                    )
            kind = slot.get("kind", "place")
            if kind not in KINDS:
                errors.append(f"slot {name!r}: 'kind' is one of {', '.join(KINDS)}, not {kind!r}")
            at = slot.get("at")
            if at is not None and not (_numbers(at) and len(at) in (3, 4)):
                errors.append(f"slot {name!r}: 'at' is [x, y, z] or [x, y, z, yaw], not {at!r}")
            stand = slot.get("stand", [])
            if not isinstance(stand, (list, tuple)) or any(
                not (_numbers(p) and len(p) == 3) for p in stand
            ):
                errors.append(f"slot {name!r}: 'stand' is a list of [x, y, yaw], not {stand!r}")
            if at is None and not stand:
                errors.append(f"slot {name!r}: states neither a hand target ('at') nor a stance")
            if at is None and "kind" in slot:
                errors.append(f"slot {name!r}: 'kind' {kind!r} names a hand target, but no 'at'")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        on = str(self.config["body"])
        body = spec.body(on)
        if body is None:
            raise PluginError(
                f"slots {self.address}: no body {on!r} to state slots on -- the body of the entry "
                f"{self.entity!r} it is nested under, as the model names it"
            )
        owner = self.entity
        for name, slot in self.config["slots"].items():
            kind = slot.get("kind", "place")
            at = slot.get("at")
            if at is not None:
                x, y, z = (float(v) for v in at[:3])
                yaw = float(at[3]) if len(at) == 4 else 0.0
                if kind == "grip":
                    body.add_site(name=f"{owner}/{name}/grip", pos=[x, y, z], quat=_grip_quat(yaw))
                else:
                    body.add_site(name=f"{owner}/{name}", pos=[x, y, z], quat=_yaw_quat(yaw))
            stands = slot.get("stand", [])
            for k, (sx, sy, syaw) in enumerate(stands):
                suffix = "stand" if len(stands) == 1 else f"stand_{k}"
                body.add_site(
                    name=f"{owner}/{name}/{suffix}",
                    pos=[float(sx), float(sy), 0.0],
                    quat=_yaw_quat(float(syaw)),
                )
