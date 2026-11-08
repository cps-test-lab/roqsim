"""Machine: a sliding door whose leaf runs sideways along its wall, with open and close programs.

One entity's own procedures (``docs/architecture.rst``, "Machines and scenarios"): the door opens and
closes, ramped at its speed, and gives up on an obstruction. A door that opens because a walker or a
robot comes near is a scenario reading a ``zone_trigger`` before it and starting the door.

The model is generated: a leaf (a box, colliding) on a horizontal slide joint driven by a
force-capped position servo, so an obstacle is pushed gently and never crushed, inside a visual-only
frame of two posts and a lintel. It is placed by its **opening centre** (``pose.position``) and the
wall direction (the pose's yaw), like ``door``; ``slide`` says which way the leaf runs to open
(``+1`` along the wall's +x, ``-1`` the other way), and the leaf hangs ``offset`` proud of the wall
so it runs past the wall beside the opening.

Programs (:class:`~roqsim.machine.MachineHandle` under ``machine:<name>``): ``open()`` and
``close()``, each ramped at ``speed`` and ending once the leaf is there; a leaf that stays blocked
for ``stall_timeout`` holds where it is and the program ends with the error ``stalled at 0.31 m``
(and a ``stalled`` event). ``value("leaf")`` is the leaf's target travel (m, 0 closed).

Config::

    sliding_door:
      prefix: sd_1_         # MJCF name prefix (distinct per door)
      pose:                 # opening CENTRE in the world (roqsim.pose); yaw aligns it with its wall
        position: {x: 0.0, y: 0.0}
        orientation: {yaw: 0.0}
      width: 1.2            # opening and leaf width (m)
      height: 2.1           # leaf height (m)
      thickness: 0.04       # leaf thickness (m)
      offset: 0.08          # how far proud of the wall the leaf hangs (m)
      slide: 1              # +1 / -1: which way along the wall the leaf runs to open
      speed: 0.5            # leaf speed of open and close (m/s)
      max_force: 150.0      # the servo's force cap (N): an obstacle is pushed, never crushed
      stall_timeout: 2.0    # s blocked before a program gives up
      stall_speed: 0.01     # m/s below which a leaf short of its target is blocked
      color: [0.75, 0.8, 0.85, 0.6]   # the leaf's rgba; a glazed leaf by default
"""

from __future__ import annotations

import mujoco

from roqsim.context import SimContext
from roqsim.machine import MachinePlugin, program
from roqsim.pose import config_pose, config_pose_errors

_FRAME_RGBA = [0.35, 0.37, 0.4, 1.0]
_DEFAULTS = {
    "width": 1.2,
    "height": 2.1,
    "thickness": 0.04,
    "offset": 0.08,
    "speed": 0.5,
    "max_force": 150.0,
    "stall_timeout": 2.0,
    "stall_speed": 0.01,
}


class SlidingDoorPlugin(MachinePlugin):
    ACTUATORS = ("leaf",)
    HOME = {"leaf": 0.0}
    ENTITY_KIND = "door"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.prefix = str(self.config.get("prefix", ""))
        self.pos, self.quat = config_pose(self.config)
        self.n = {}
        for key, default in _DEFAULTS.items():
            try:
                self.n[key] = float(self.config.get(key, default))
            except (TypeError, ValueError):
                self.n[key] = default
        self.slide = 1.0 if float(self.config.get("slide", 1) or 1) >= 0 else -1.0
        self.color = list(self.config.get("color") or [0.75, 0.8, 0.85, 0.6])

    def validate_config(self, config: dict) -> list[str]:
        errors = config_pose_errors(config, "sliding_door")
        for key in _DEFAULTS:
            if key not in config:
                continue
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"sliding_door: {key!r} is a positive number")
        if config.get("slide", 1) not in (1, -1):
            errors.append("sliding_door: 'slide' is +1 or -1")
        color = config.get("color")
        if color is not None and not (isinstance(color, list) and len(color) in (3, 4)):
            errors.append("sliding_door: 'color' is [r, g, b] or [r, g, b, a]")
        errors += super().validate_config(config)
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        n = self.n
        w, h, t = n["width"], n["height"], n["thickness"]
        child = mujoco.MjSpec()
        frame_body = child.worldbody.add_body(name="sliding_door")
        for name, half, pos in (
            ("post_a", [0.04, 0.04, h / 2 + 0.02], [-w / 2 - 0.04, 0, h / 2 + 0.02]),
            ("post_b", [0.04, 0.04, h / 2 + 0.02], [w / 2 + 0.04, 0, h / 2 + 0.02]),
            ("lintel", [w + 0.08, 0.05, 0.04], [self.slide * w / 2, n["offset"], h + 0.06]),
        ):
            geom = frame_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos)
            geom.name = name
            geom.rgba = _FRAME_RGBA
            geom.contype = geom.conaffinity = 0
        leaf = frame_body.add_body(name="leaf", pos=[0, n["offset"], 0.01 + h / 2])
        joint = leaf.add_joint(name="leaf_slide", type=mujoco.mjtJoint.mjJNT_SLIDE)
        joint.axis = [self.slide, 0, 0]
        joint.range = [0.0, w]
        geom = leaf.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[w / 2, t / 2, h / 2 - 0.01])
        geom.name = "leaf"
        geom.rgba = self.color if len(self.color) == 4 else [*self.color, 1.0]
        geom.mass = 40.0
        act = child.add_actuator()
        act.name = "leaf"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = "leaf_slide"
        act.set_to_position(4000.0, 400.0)
        act.ctrlrange = [0.0, w]
        act.ctrllimited = True
        act.forcerange = [-n["max_force"], n["max_force"]]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        self.register_entity(ctx, body=self.prefix + "sliding_door", width=self.n["width"])
        super().configure(ctx)

    @program
    def open(self):
        """Slide the leaf open."""
        yield from self._slide(self.n["width"])

    @program
    def close(self):
        """Slide the leaf shut."""
        yield from self._slide(0.0)

    def _slide(self, target: float):
        n = self.n
        self.ramp("leaf", target, n["speed"])
        blocked = 0.0
        dof = self._actuators["leaf"].dadr
        while not self.reached("leaf"):
            moving = abs(float(self._ctx.data.qvel[dof])) >= n["stall_speed"]
            blocked = 0.0 if moving else blocked + self._ctx.dt
            if blocked >= n["stall_timeout"]:
                where = min(max(self.position("leaf"), 0.0), n["width"])  # inside its range
                self.set("leaf", where)
                self.step("stalled", travel=where)
                self.fail(f"stalled at {where:.2f} m")
            yield
