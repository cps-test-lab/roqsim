"""Scene plugin: a push button on a housing, pressed by whatever pushes it.

A sprung button on a slide joint in front of a static housing, facing the housing's +x. It moves
because something pushes it -- a hand's press proxy, a robot's finger -- never because a controller
sets it. It reports itself: pressed once it has travelled past ``threshold`` of its stroke and until
it comes back under ``release`` of it (a snap action, so a button ringing as it is let go is pressed
once), an event ``pressed`` and ``released`` at each change, the ``pressed`` endpoint (for an OpenSCENARIO
``entity_monitor``), and a :class:`ButtonHandle` under ``button:<name>``. It decides nothing: what a
press leads to -- a machine starting -- is a scenario's.

**A pressable's housing must not collide with its button.** A child body of a body welded to the
world is not filtered from that body's geoms, so a button sunk into its housing would be held by it;
the button and its housing are excluded from each other (``<contact><exclude>``), and anything else
modelled as a button needs the same.

The button's front face carries a site ``<name>/button``, +x out of the face: the target a hand
presses, its press axis the site's -x.
"""

from __future__ import annotations

import mujoco

from roqsim import endpoint
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors
from roqsim.schema import Field


class ButtonHandle:
    """Published under ``button:<name>``: whether the button is pressed and how far it travelled."""

    def __init__(self, plugin: PushButtonPlugin):
        self._p = plugin
        self.name = plugin.address

    @property
    def pressed(self) -> bool:
        return self._p.is_pressed

    @property
    def travel(self) -> float:
        """How far the button is pushed in (m)."""
        return self._p.travel_now

    @property
    def presses(self) -> int:
        """How many times it was pressed this episode."""
        return self._p.count


class PushButtonPlugin(Plugin):
    provides_entity = True

    CONFIG_SCHEMA = {
        "pose": Field(dict, required=True, doc="the housing's centre, as SpawnEntity states a pose; the button faces its +x"),
        "housing": Field(list, default=[0.06, 0.16, 0.16], length=3, unit="m", doc="the housing's full extents"),
        "radius": Field(float, default=0.02, minimum=0.005, unit="m", doc="the button's radius"),
        "stroke": Field(float, default=0.012, minimum=0.002, unit="m", doc="how far the button goes in"),
        "threshold": Field(float, default=0.6, minimum=0.05, maximum=1.0, doc="share of the stroke that is pressed"),
        "release": Field(float, default=0.4, minimum=0.0, maximum=1.0,
                         doc="share of the stroke it must come back under to be released (below threshold)"),
        "stiffness": Field(float, default=300.0, minimum=1.0, unit="N/m", doc="the spring that pushes it back"),
        "color": Field(list, default=[0.8, 0.1, 0.1, 1.0], length=4, doc="the button's colour"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.pos, self.quat = config_pose(self.config)
        self.is_pressed = False
        self.travel_now = 0.0
        self.count = 0
        self.handle = ButtonHandle(self)

    def validate_config(self, config: dict) -> list[str]:
        return config_pose_errors(config, f"push_button {self.address}")

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        s = self.settings
        n = self.address
        hx, hy, hz = (v / 2 for v in s.housing)
        housing = spec.worldbody.add_body(name=f"{n}/housing", pos=list(self.pos), quat=list(self.quat))
        g = housing.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], rgba=[0.75, 0.75, 0.72, 1.0])
        g.name = f"{n}/housing"
        depth = 0.012
        button = housing.add_body(name=f"{n}/button", pos=[hx + depth / 2, 0.0, 0.0])
        j = button.add_joint(name=f"{n}/stroke", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1.0, 0.0, 0.0])
        j.range = [0.0, s.stroke]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.stiffness = [s.stiffness, 0.0, 0.0]
        j.springref = 0.0
        j.damping = [2.0, 0.0, 0.0]
        bg = button.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[s.radius, depth / 2, 0.0],
            quat=[0.7071068, 0.0, 0.7071068, 0.0],
            rgba=list(s.color),
            mass=0.02,
        )
        bg.name = f"{n}/button"
        button.add_site(name=f"{n}/button", pos=[depth / 2, 0.0, 0.0])
        ex = spec.add_exclude()
        ex.bodyname1 = f"{n}/housing"
        ex.bodyname2 = f"{n}/button"

    def configure(self, ctx: SimContext) -> None:
        m = ctx.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.address}/stroke")
        self._qadr = int(m.jnt_qposadr[jid])
        ctx.entities.add(Entity(name=self.address, kind="prop", body=f"{self.address}/housing",
                                meta={"stroke_joint": f"{self.address}/stroke"}))
        ctx.blackboard.set(f"button:{self.address}", self.handle)
        self._ctx = ctx

    def on_reset(self, ctx: SimContext) -> None:
        self.is_pressed = False
        self.travel_now = 0.0
        self.count = 0

    def post_step(self, ctx: SimContext) -> None:
        self.travel_now = float(ctx.data.qpos[self._qadr])
        s = self.settings
        # a snap action: pressed past the threshold, released only back under the release point, so a
        # button that rings as the finger lets go is pressed once
        if self.is_pressed:
            pressed = self.travel_now >= min(s.release, s.threshold) * s.stroke
        else:
            pressed = self.travel_now >= s.threshold * s.stroke
        if pressed != self.is_pressed:
            self.is_pressed = pressed
            if pressed:
                self.count += 1
            ctx.events.emit(self.address, "pressed" if pressed else "released", travel=self.travel_now)

    @endpoint.out(rate=30.0, ros2=None)
    def pressed(self) -> bool:
        """Whether the button is pushed in past its threshold."""
        return self.is_pressed
