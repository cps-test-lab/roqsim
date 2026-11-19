"""Scene plugin: slots on a body -- where a worker stands to work at it, and where its hand goes.

A slot is sites on a body, by the convention ``roqsim_motion``'s moves read (``docs/plugins.rst``,
"Slots"): a **hand target** and one or more **stances**.

* The hand target is a site named for the slot. ``kind: grip`` makes it ``<body>/<slot>/grip``,
  in the grip-frame convention of ``roqsim_motion.hands`` (+x out of the palm toward what it takes,
  +z along the fingers, down), so ``grasp`` takes it. ``kind: place`` makes it ``<body>/<slot>``,
  +z up, its yaw the heading what is set there is set down at, so ``place`` sets down on it.
  ``kind: press`` makes it ``<body>/<slot>``, +x out of the face, so ``press`` pushes along -x.
* Each stance is a site ``<body>/<slot>/stand`` (``/stand_<k>`` for several), on the floor, +x the
  way to face. ``approach`` and ``carry`` to the slot, or to its grip, walk to it -- the nearest of
  several -- where they are given no ``standoff`` or ``facing``.

A prop whose plugin states its own slots needs none of this; this states them on anything else --
a box standing in for a rack, a mesh model, a body of a baked scene. The poses are in the body's
frame (a ``box``'s origin is its centre), so a slot on a drawer moves with the drawer. A stance is
put at the body's origin height: what the moves read of it is where it is on the floor and its
heading.

Config::

    slots:
      on: rack_box          # the body the slots are on, by its name in the model (required)
      slots:
        bin_0:
          kind: grip        # grip | place | press (default place)
          at: [0.1, 0.0, 0.95, 3.14159]   # x, y, z, yaw in the body's frame (m, rad)
          stand: [[0.6, 0.0, 3.14159]]    # x, y, yaw in the body's frame, on the floor

It adds sites in ``build``, so it is listed after the plugin that builds the body, and before a
humanoid that grasps one of its grips is built (a humanoid's holds are made in ``build_late``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin, PluginError
from roqsim.schema import Field

KINDS = ("grip", "place", "press")


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

    CONFIG_SCHEMA = {
        "on": Field(str, required=True, doc="the body the slots are on, by its model name"),
        "slots": Field(dict, required=True, doc="slot name -> {kind, at, stand}"),
    }

    @classmethod
    def validate_config(cls, config) -> list:
        errors = []
        slots = config.get("slots")
        if not isinstance(slots, dict) or not slots:
            return ["'slots' is a mapping of slot names to {kind, at, stand}"]
        for name, slot in slots.items():
            if not isinstance(slot, dict):
                errors.append(f"slot {name!r}: a mapping of kind, at and stand")
                continue
            kind = slot.get("kind", "place")
            if kind not in KINDS:
                errors.append(f"slot {name!r}: 'kind' is one of {', '.join(KINDS)}, not {kind!r}")
            at = slot.get("at")
            if at is not None and not (isinstance(at, (list, tuple)) and len(at) in (3, 4)):
                errors.append(f"slot {name!r}: 'at' is [x, y, z] or [x, y, z, yaw], not {at!r}")
            stand = slot.get("stand", [])
            if not isinstance(stand, (list, tuple)) or any(
                not (isinstance(p, (list, tuple)) and len(p) == 3) for p in stand
            ):
                errors.append(f"slot {name!r}: 'stand' is a list of [x, y, yaw], not {stand!r}")
            if at is None and not stand:
                errors.append(f"slot {name!r}: states neither a hand target ('at') nor a stance")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        on = str(self.config["on"])
        body = spec.body(on)
        if body is None:
            raise PluginError(
                f"slots {self.address}: no body {on!r} to state slots on -- list this after the "
                "plugin that builds it"
            )
        for name, slot in self.config["slots"].items():
            kind = slot.get("kind", "place")
            at = slot.get("at")
            if at is not None:
                x, y, z = (float(v) for v in at[:3])
                yaw = float(at[3]) if len(at) == 4 else 0.0
                if kind == "grip":
                    body.add_site(name=f"{on}/{name}/grip", pos=[x, y, z], quat=_grip_quat(yaw))
                else:
                    body.add_site(name=f"{on}/{name}", pos=[x, y, z], quat=_yaw_quat(yaw))
            stands = slot.get("stand", [])
            for k, (sx, sy, syaw) in enumerate(stands):
                suffix = "stand" if len(stands) == 1 else f"stand_{k}"
                body.add_site(
                    name=f"{on}/{name}/{suffix}",
                    pos=[float(sx), float(sy), 0.0],
                    quat=_yaw_quat(float(syaw)),
                )
