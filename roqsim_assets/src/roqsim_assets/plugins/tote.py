"""Scene plugin: a tote -- an open box of five walls -- that hands carry by its two ends.

A free body, so physics carries it and a scenario can place it. Its two short sides (the tote's ±x)
are solid walls with no handle built into them; where a hand holds each is a grip site outside the
wall, a handle's radius (:data:`GRIP_OUT`) from it and 3 cm under the rim: ``<name>/grip_l`` (the +x
end, for a left hand when the tote is carried in front, its long side across the body) and
``<name>/grip_r`` (the -x end), in the grip-frame convention of ``roqsim_motion.hands``: +x out of
the palm toward the tote, +z along the fingers, down. ``look_full`` / ``look_empty`` bodies, contents shown and hidden, let a
``part_state`` give it those states.

Contact is tuned to be held and set down, as ``graspable_box``'s: box primitives, ``condim 4``.
"""

from __future__ import annotations

import mujoco

from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors
from roqsim.schema import Field

#: How far out from its wall a handle's grip frame is (m): a handle's radius, where a palm on the wall
#: holds it.
GRIP_OUT = 0.015


class TotePlugin(Plugin):
    provides_entity = True

    CONFIG_SCHEMA = {
        "pose": Field(dict, required=True, doc="where it stands; omit z to stand it on the floor"),
        "size": Field(list, default=[0.4, 0.3, 0.2], length=3, unit="m", doc="length (between the handles), width, height"),
        "mass": Field(float, default=1.5, minimum=0.05, unit="kg", doc="the empty tote's mass"),
        "color": Field(list, default=[0.15, 0.35, 0.7, 1.0], length=4, doc="its colour"),
        "contents": Field(bool, default=False, doc="add look_full / look_empty bodies for a part_state"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.pos, self.quat = config_pose(self.config)
        self._qpos = None

    def validate_config(self, config: dict) -> list[str]:
        return config_pose_errors(config, f"tote {self.address}")

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        s = self.settings
        L, W, H = s.size
        n = self.address
        t = 0.008
        pos = list(self.pos)
        if self.config["pose"].get("position", {}).get("z") is None:
            pos[2] = 0.0
        body = spec.worldbody.add_body(name=n, pos=pos, quat=list(self.quat))
        body.add_freejoint(name=f"{n}/free")
        m = s.mass / 5.0
        common = {"rgba": list(s.color), "mass": m, "condim": 4, "friction": [1.2, 0.005, 0.0001]}
        for size, at in (
            ((L / 2, W / 2, t / 2), (0.0, 0.0, t / 2)),
            ((t / 2, W / 2, H / 2), (L / 2 - t / 2, 0.0, H / 2)),
            ((t / 2, W / 2, H / 2), (-L / 2 + t / 2, 0.0, H / 2)),
            ((L / 2, t / 2, H / 2), (0.0, W / 2 - t / 2, H / 2)),
            ((L / 2, t / 2, H / 2), (0.0, -W / 2 + t / 2, H / 2)),
        ):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(at), **common)
        # Where a hand holds it: a grip frame outside each end wall, palm toward the tote, fingers
        # down; a handle's radius out from the wall, so the palm lies on it.
        hz = H - 0.03
        body.add_site(name=f"{n}/grip_l", pos=[L / 2 + GRIP_OUT, 0.0, hz], quat=[0.0, 0.0, 1.0, 0.0])
        body.add_site(name=f"{n}/grip_r", pos=[-L / 2 - GRIP_OUT, 0.0, hz], quat=[0.0, 1.0, 0.0, 0.0])
        body.add_site(name=f"{n}/bottom", pos=[0.0, 0.0, 0.0])
        if s.contents:
            full = body.add_body(name=f"{n}/look_full")
            full.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2 - 0.03, W / 2 - 0.03, H / 3],
                          pos=[0.0, 0.0, H / 3 + t], rgba=[0.75, 0.6, 0.3, 1.0], contype=0, conaffinity=0,
                          mass=0.0)
            empty = body.add_body(name=f"{n}/look_empty")
            empty.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.001], pos=[0.0, 0.0, t],
                           rgba=list(s.color), contype=0, conaffinity=0, mass=0.0)

    def configure(self, ctx: SimContext) -> None:
        m = ctx.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.address}/free")
        adr = int(m.jnt_qposadr[jid])
        self._qpos = (adr, m.qpos0[adr : adr + 7].copy(), int(m.jnt_dofadr[jid]))
        ctx.entities.add(Entity(name=self.address, kind="prop", body=self.address,
                                meta={"base_joint": f"{self.address}/free", "prefix": f"{self.address}/"}))

    def on_reset(self, ctx: SimContext) -> None:
        adr, q0, dof = self._qpos
        ctx.data.qpos[adr : adr + 7] = q0
        ctx.data.qvel[dof : dof + 6] = 0.0
