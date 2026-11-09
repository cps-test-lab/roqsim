"""Scene plugin: a cabinet of drawers that slide when something pulls them.

A static carcass and ``drawers`` drawers stacked in it, each a tray on a slide joint along the
cabinet's +x (out of its front), with friction and an end stop, a handle bar on its front and a
grip site ``<name>/drawer_<i>/grip`` on the handle, in the grip-frame convention of
``roqsim_motion.hands`` (+x out of the palm toward the handle: the cabinet's -x; +z along the
fingers: down), so a hand that holds it pulls the drawer through the hold. A drawer's floor carries
what is put in it. Nothing drives a drawer: it moves because it is pulled or pushed.

Each drawer is an entity of its own (``<name>/drawer_<i>``), so a move names it as a target.
"""

from __future__ import annotations

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors
from roqsim.schema import Field

_GREY = [0.55, 0.57, 0.6, 1.0]
_DARK = [0.2, 0.2, 0.22, 1.0]


class DrawerCabinetPlugin(Plugin):
    provides_entity = True

    CONFIG_SCHEMA = {
        "pose": Field(
            dict,
            required=True,
            doc="the cabinet's front-bottom centre on the floor; its front faces +x",
        ),
        "size": Field(
            list,
            default=[0.6, 0.7, 0.9],
            length=3,
            unit="m",
            doc="the carcass: depth, width, height",
        ),
        "drawers": Field(int, default=2, minimum=1, maximum=6, doc="how many drawers, stacked"),
        "travel": Field(
            float, default=0.45, minimum=0.05, unit="m", doc="how far a drawer comes out"
        ),
        "friction": Field(
            float, default=8.0, minimum=0.0, unit="N", doc="a drawer's sliding friction"
        ),
        "wall": Field(
            float, default=0.12, minimum=0.02, unit="m", doc="the height of a drawer's walls"
        ),
    }

    def validate_config(self, config: dict) -> list[str]:
        errors = config_pose_errors(config, f"drawer_cabinet {self.address}")
        s = self.settings_for(config)
        if s.travel >= s.size[0]:
            errors.append(
                f"drawer_cabinet {self.address}: 'travel' must be under the depth {s.size[0]} m"
            )
        return errors

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.pos, self.quat = config_pose(self.config)

    def drawer(self, i: int) -> str:
        return f"{self.address}/drawer_{i}"

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        s = self.settings
        depth, width, height = s.size
        n = self.address
        pos = list(self.pos)
        pos[2] = pos[2] or 0.0
        carcass = spec.worldbody.add_body(name=f"{n}/carcass", pos=pos, quat=list(self.quat))
        wall = 0.02
        # back, sides, top, bottom: open at the front (+x) for the drawers
        for size, at in (
            ((wall / 2, width / 2, height / 2), (-depth + wall / 2, 0.0, height / 2)),
            ((depth / 2, wall / 2, height / 2), (-depth / 2, width / 2 - wall / 2, height / 2)),
            ((depth / 2, wall / 2, height / 2), (-depth / 2, -width / 2 + wall / 2, height / 2)),
            ((depth / 2, width / 2, wall / 2), (-depth / 2, 0.0, height - wall / 2)),
            ((depth / 2, width / 2, 0.04), (-depth / 2, 0.0, 0.04)),
        ):
            carcass.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(at), rgba=_GREY
            )
        slot = (height - 0.08 - wall) / s.drawers
        inner_w = width - 2 * wall - 0.01
        inner_d = depth - wall - 0.02
        for i in range(s.drawers):
            z0 = 0.08 + i * slot
            tray_h = min(slot - 0.02, s.wall)
            d = carcass.add_body(name=self.drawer(i), pos=[-inner_d / 2, 0.0, z0 + 0.005])
            j = d.add_joint(
                name=f"{self.drawer(i)}/slide",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[1.0, 0.0, 0.0],
            )
            j.range = [0.0, s.travel]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.frictionloss = s.friction
            j.damping = [5.0, 0.0, 0.0]
            t = 0.01
            for size, at in (
                ((inner_d / 2, inner_w / 2, t / 2), (0.0, 0.0, t / 2)),  # floor
                ((t / 2, inner_w / 2, tray_h / 2), (inner_d / 2 - t / 2, 0.0, tray_h / 2)),  # front
                ((t / 2, inner_w / 2, tray_h / 2), (-inner_d / 2 + t / 2, 0.0, tray_h / 2)),  # back
                ((inner_d / 2, t / 2, tray_h / 2), (0.0, inner_w / 2 - t / 2, tray_h / 2)),
                ((inner_d / 2, t / 2, tray_h / 2), (0.0, -inner_w / 2 + t / 2, tray_h / 2)),
            ):
                d.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=list(size),
                    pos=list(at),
                    rgba=_GREY,
                    mass=0.6,
                )
            hx = inner_d / 2 + 0.035
            hz = tray_h * 0.6
            d.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[hx, -0.12, hz, hx, 0.12, hz],
                size=[0.012, 0, 0],
                rgba=_DARK,
                mass=0.1,
            )
            for y in (-0.12, 0.12):
                d.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[inner_d / 2, y, hz, hx, y, hz],
                    size=[0.008, 0, 0],
                    rgba=_DARK,
                    mass=0.02,
                )
            # the grip frame on the bar's axis, where a hand holds it: +x toward the cabinet (the hand's
            # palm faces it), +z down along the fingers
            d.add_site(name=f"{self.drawer(i)}/grip", pos=[hx, 0.0, hz], quat=[0.0, 0.0, 1.0, 0.0])
            d.add_site(name=f"{self.drawer(i)}/inside", pos=[0.0, 0.0, t])

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(name=self.address, kind="prop", body=f"{self.address}/carcass", meta={})
        )
        for i in range(self.settings.drawers):
            ctx.entities.add(
                Entity(
                    name=self.drawer(i),
                    kind="prop",
                    body=self.drawer(i),
                    meta={"slide_joint": f"{self.drawer(i)}/slide"},
                )
            )
