"""Machine: a C-frame press whose ram strokes only behind its closed guard door.

One entity's own procedures (``docs/architecture.rst``, "Machines and scenarios"): the press closes
its guard, makes its stroke and opens the guard again, and changes the state of the part in its die
at the stroke's bottom through that part's handle (``part_state``). Loading the part, and starting
the press once the robot that loaded it has left, is a scenario's.

The model is generated, with no vendor's look: a base whose top is the press table, a column at the
back, a head above the table, a die plate on the table, a ram on a vertical slide under the head, and
a guard door that slides down in front of the opening. One neutral grey body, a blue accent on the
head, safety yellow on the door's edges. Its origin is the centre of its footprint on the floor, the
opening facing +x. Ram and door are position servos (``ram``, ``door``), gravity-compensated as a
cylinder holds its load; the ram is visual-only, so a stroke changes the part's state rather than
pushing it.

The die zone is a ``zone_trigger`` nested under the press (``<press>.die``), riding on the table: a
part counts as in the die when its centre of mass is over the die plate. A world may nest its own
``die`` entry to change the zone.

Programs (:class:`~roqsim.machine.MachineHandle` under ``machine:<name>``):

* ``stroke(show="", hide="")`` -- close the guard (``door_close``), wait until it has stopped (the
  interlock), a fast approach (``approach``), a slow power stroke (``power``), the part's states
  changed at the bottom (``bottom``), a fast return (``return``), wait until the ram has stopped, open
  the guard (``door_open``). With ``show``/``hide`` it needs exactly one part in the die and a
  ``part_state`` entry in the world; with an empty die it fails, naming what it holds.
* ``close_guard()`` -- the door alone, for a cell that loads behind it.
* ``open_guard()`` -- returns the ram home and waits until it is at rest there (after a cancelled
  stroke it may stand anywhere down its travel), then opens the door.

The press is ``ready`` with its ram up and its guard open.

Config::

    press:
      prefix: press_1_      # MJCF name prefix (distinct per press)
      pose:                 # the press in the world (roqsim.pose); its footprint centre on the floor
        position: {x: 0.0, y: 0.0, z: 0.0}
        orientation: {yaw: 0.0}
      table_height: 0.9     # height of the press table (m)
      door_speed: 0.25      # guard door speed (m/s): 2 s each way
      approach_speed: 0.2   # ram speed down to contact (m/s)
      power_speed: 0.03     # ram speed of the power stroke (m/s)
      return_speed: 0.2     # ram speed back up (m/s)
      ram_contact: 0.195    # ram travel where the power stroke begins (m, down from home)
      ram_bottom: 0.215     # ram travel at the stroke's bottom (m): the punch 5 mm above the die
"""

from __future__ import annotations

import mujoco

from roqsim.config import PluginSpec
from roqsim.context import SimContext
from roqsim.machine import MachineError, MachinePlugin, program
from roqsim.pose import config_pose, config_pose_errors

_GREY = [0.55, 0.57, 0.6, 1.0]
_DARK = [0.25, 0.26, 0.28, 1.0]
_ACCENT = [0.12, 0.35, 0.65, 1.0]
_YELLOW = [0.95, 0.75, 0.05, 1.0]
_STEEL = [0.7, 0.72, 0.75, 1.0]

#: The opening between the table and the head (m), and the guard door's travel across it.
_OPENING = 0.5
_DOOR_TRAVEL = 0.5
#: The ram's travel (m): its punch rests 0.22 m above the die at home.
_RAM_TRAVEL = 0.25
#: Half-extents of the die zone over the die plate (m).
_DIE_ZONE = [0.12, 0.12, 0.04]

_NUMBERS = {
    "table_height": 0.9,
    "door_speed": 0.25,
    "approach_speed": 0.2,
    "power_speed": 0.03,
    "return_speed": 0.2,
    "ram_contact": 0.195,
    "ram_bottom": 0.215,
}


class PressPlugin(MachinePlugin):
    ACTUATORS = ("door", "ram")
    HOME = {"door": 0.0, "ram": 0.0}
    NEEDS = {"die": "zone:{address}.die"}
    ENTITY_KIND = "press"
    expansion_keys = frozenset({"table_height"})

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.prefix = str(self.config.get("prefix", ""))
        self.pos, self.quat = config_pose(self.config)
        self.n = {key: self._number(key) for key in _NUMBERS}

    def _number(self, key: str) -> float:
        try:
            return float(self.config.get(key, _NUMBERS[key]))
        except (TypeError, ValueError):
            return _NUMBERS[key]

    @classmethod
    def expand(cls, spec: PluginSpec, world, base_dir) -> list[PluginSpec]:
        """The die zone, unless the world nests its own ``die``."""
        if any(child.label == "die" for child in spec.children):
            return []
        table = float(spec.config.get("table_height", _NUMBERS["table_height"]))
        zone = {
            "body": "press",
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": table + 0.03 + _DIE_ZONE[2]}},
            "size": [2 * v for v in _DIE_ZONE],  # a zone's size is its full extents
        }
        return [
            PluginSpec(
                ref="zone_trigger", name="die", config=zone, children=[], entity=spec.address
            )
        ]

    def validate_config(self, config: dict) -> list[str]:
        errors = config_pose_errors(config, "press")
        for key in _NUMBERS:
            if key not in config:
                continue
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"press: {key!r} is a positive number")
        contact = config.get("ram_contact", _NUMBERS["ram_contact"])
        bottom = config.get("ram_bottom", _NUMBERS["ram_bottom"])
        if (
            isinstance(contact, (int, float))
            and isinstance(bottom, (int, float))
            and not 0 < contact < bottom <= _RAM_TRAVEL
        ):
            errors.append(
                f"press: 0 < ram_contact < ram_bottom <= {_RAM_TRAVEL} (the ram's travel, m)"
            )
        errors += super().validate_config(config)
        return errors

    # -- model ---------------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        table = self.n["table_height"]
        head_bottom = table + 0.04 + _OPENING
        press = child.worldbody.add_body(name="press")

        def box(body, name, half, pos, rgba, collide=True):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, rgba=rgba)
            geom.name = name
            if not collide:
                geom.contype = geom.conaffinity = 0
            return geom

        box(press, "base", [0.35, 0.3, table / 2], [0, 0, table / 2], _GREY)
        box(press, "die", [0.12, 0.12, 0.02], [0, 0, table + 0.02], _STEEL)
        # Parts that meet keep a 1 mm step between their faces: two faces in one plane flicker in
        # every render as the camera moves (roqsim_assets.shared_faces).
        box(press, "column", [0.0795, 0.22, 0.45], [-0.2695, 0, head_bottom - 0.05], _GREY)
        box(press, "head", [0.33, 0.24, 0.12], [-0.02, 0, head_bottom + 0.12], _GREY)
        box(press, "accent", [0.005, 0.2, 0.03], [0.315, 0, head_bottom + 0.15], _ACCENT, False)
        box(press, "foot", [0.4, 0.34, 0.02], [0, 0, 0.02], _DARK)

        ram = press.add_body(name="ram", pos=[0, 0, head_bottom])
        ram.gravcomp = 1.0
        joint = ram.add_joint(name="ram_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1])
        joint.range = [0.0, _RAM_TRAVEL]
        box(ram, "ram", [0.05, 0.05, 0.12], [0, 0, -0.04], _STEEL, False).mass = 5.0
        box(ram, "punch", [0.015, 0.015, 0.06], [0, 0, -0.22], _DARK, False).mass = 0.2

        door = press.add_body(name="door", pos=[0.37, 0, table + 0.04 + _OPENING * 1.5])
        door.gravcomp = 1.0
        joint = door.add_joint(name="door_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1])
        joint.range = [0.0, _DOOR_TRAVEL]
        half = _OPENING / 2
        box(door, "door", [0.01, 0.25, half], [0, 0, 0], [0.75, 0.8, 0.85, 0.5]).mass = 3.0
        for side in (-1, 1):
            box(
                door,
                f"door_edge_{side + 1}",
                [0.012, 0.019, half - 0.002],
                [0, side * 0.23, 0],
                _YELLOW,
                False,
            )
        box(door, "door_edge_bottom", [0.011, 0.248, 0.015], [0, 0, -half + 0.016], _YELLOW, False)
        press.add_site(
            name="die_zone",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=_DIE_ZONE,
            pos=[0, 0, table + 0.03 + _DIE_ZONE[2]],
            group=4,
        )

        for name, target, kp, kv, travel in (
            ("door", "door_slide", 20000.0, 400.0, _DOOR_TRAVEL),
            ("ram", "ram_slide", 50000.0, 1000.0, _RAM_TRAVEL),
        ):
            act = child.add_actuator()
            act.name = name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.target = target
            act.set_to_position(kp, kv)
            act.ctrlrange = [0.0, travel]
            act.ctrllimited = True

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def configure(self, ctx: SimContext) -> None:
        self.register_entity(ctx, body=self.prefix + "press")
        super().configure(ctx)

    # -- programs ------------------------------------------------------------------------------
    @program
    def stroke(self, show: str = "", hide: str = ""):
        """Close the guard, make one three-phase stroke, open the guard.

        Args:
            show: a state shown on the part in the die at the stroke's bottom
            hide: a state hidden there
        """
        part = ""
        states = None
        if show or hide:
            states = self._ctx.blackboard.get("part_state")
            if states is None:
                raise MachineError(
                    f"machine {self.address}: a stroke that changes a part needs a part_state entry "
                    "in the world"
                )
            inside = self.die.inside()
            if len(inside) != 1:
                self.fail(
                    f"a stroke that changes a part needs one part in the die; it holds {inside or 'none'}"
                )
            part = inside[0]
        n = self.n
        yield from self.move("door", _DOOR_TRAVEL, speed=n["door_speed"], event="door_close")
        yield from self.until_stopped("door")
        yield from self.move("ram", n["ram_contact"], speed=n["approach_speed"], event="approach")
        yield from self.move("ram", n["ram_bottom"], speed=n["power_speed"], event="power")
        if part:
            if hide:
                states.hide(part, hide)
            if show:
                states.show(part, show)
        self.step("bottom", part=part)
        yield from self.move("ram", self.home["ram"], speed=n["return_speed"], event="return")
        yield from self.until_stopped("ram")
        yield from self.move("door", self.home["door"], speed=n["door_speed"], event="door_open")
        yield from self.until_stopped("door")

    @program
    def close_guard(self):
        """Close the guard door."""
        yield from self.move("door", _DOOR_TRAVEL, speed=self.n["door_speed"], event="door_close")
        yield from self.until_stopped("door")

    @program
    def open_guard(self):
        """Return the ram home, then open the guard door."""
        yield from self.move("ram", self.home["ram"], speed=self.n["return_speed"], event="return")
        yield from self.until_stopped("ram")
        yield from self.move(
            "door", self.home["door"], speed=self.n["door_speed"], event="door_open"
        )
        yield from self.until_stopped("door")

    def ready_when(self) -> bool:
        return self.at_home("ram", "door")
