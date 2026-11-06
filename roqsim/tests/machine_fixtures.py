# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Machines the machine tests run: generated models, no file from outside.

A module rather than a conftest, so a world addresses them as ``machine_fixtures:<Class>`` -- the
same reason :mod:`recording_plugin` is one.
"""

from __future__ import annotations

import mujoco

from roqsim.machine import MachinePlugin, program

#: A press: a guard door and a ram on slide joints, each a position servo, and a site at the die.
PRESS = """<mujoco model="press">
  <worldbody>
    <body name="press">
      <site name="die_zone" type="box" size="0.2 0.2 0.05" pos="0 0 0.05"/>
      <body name="door" pos="0.3 0 0.2">
        <joint name="door_joint" type="slide" axis="0 0 -1" range="0 0.4"/>
        <geom type="box" size="0.01 0.2 0.2" contype="0" conaffinity="0" mass="2"/>
      </body>
      <body name="ram" pos="0 0 0.5">
        <joint name="ram_joint" type="slide" axis="0 0 -1" range="0 0.1"/>
        <geom type="box" size="0.03 0.03 0.03" contype="0" conaffinity="0" mass="5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="door" joint="door_joint" kp="20000" kv="400" ctrlrange="0 0.4"/>
    <position name="ram" joint="ram_joint" kp="50000" kv="1000" ctrlrange="0 0.1"/>
  </actuator>
</mujoco>"""

DOOR_CLOSED, DOOR_SPEED = 0.4, 0.2  # 2 s each way
CONTACT, BOTTOM = 0.094, 0.1
APPROACH, POWER, RETURN = 0.2, 0.03, 0.2  # m/s


class Press(MachinePlugin):
    """Machine: a press whose ram strokes only while its guard door is closed.

    Config::

        prefix: p1_   # MJCF name prefix
        fail_at: ""   # a step name at which the stroke raises (a defect, for the tests)
    """

    ACTUATORS = ("door", "ram")

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        child = mujoco.MjSpec.from_string(PRESS)
        frame = spec.worldbody.add_frame()
        spec.attach(child, prefix=self.config.get("prefix", ""), frame=frame)

    def configure(self, ctx) -> None:
        self.register_entity(ctx, body=self.config.get("prefix", "") + "press", kind="press")
        super().configure(ctx)

    @program
    def stroke(self, depth: float = BOTTOM):
        """Close the guard, make one three-phase stroke, open the guard.

        Args:
            depth: the ram's bottom (m)
        """
        yield from self.move("door", DOOR_CLOSED, speed=DOOR_SPEED, event="door_close")
        yield from self.until_stopped("door")
        yield from self.move("ram", CONTACT, speed=APPROACH, event="approach")
        if self.config.get("fail_at") == "power":
            raise ValueError("a die that is not there")
        yield from self.move("ram", depth, speed=POWER, event="power")
        self.step("bottom", depth=depth)
        yield from self.move("ram", self.home["ram"], speed=RETURN, event="return")
        yield from self.until_stopped("ram")
        yield from self.move("door", self.home["door"], speed=DOOR_SPEED, event="door_open")
        yield from self.until_stopped("door")

    @program
    def jam(self):
        """Fail as a blocked machine does: its own outcome, not a defect."""
        yield from self.wait(0.1)
        self.fail("jammed")

    @program
    def early(self):
        """Raise before the first step: a defect where the start is applied."""
        raise ValueError("at once")
        yield

    @program
    def dwell(self, seconds: float):
        """Do nothing for a while."""
        yield from self.wait(seconds)

    def ready_when(self) -> bool:
        return self.at_home("ram", "door")


class Spinner(MachinePlugin):
    """Machine: a setpoint the plugin applies itself (a speed), with no actuator.

    Config::

        ramp: 0.0   # m/s^2; 0 sets the speed at once
    """

    SETPOINTS = {"speed": "m/s"}
    HOME = {"speed": 0.1}

    def build(self, spec, ctx) -> None:
        spec.worldbody.add_body(name=f"{self.address}_body")

    def configure(self, ctx) -> None:
        self.register_entity(ctx, body=f"{self.address}_body")
        super().configure(ctx)
        self.applied: list[float] = []

    def pre_step(self, ctx) -> None:
        super().pre_step(ctx)
        self.applied.append(self.value("speed"))

    @program
    def run(self, speed: float):
        """Run at a speed."""
        ramp = float(self.config.get("ramp", 0.0))
        if ramp > 0:
            self.ramp("speed", speed, ramp)
        else:
            self.set("speed", speed)
        yield from self.until_reached("speed")


#: One slide joint, x in [-1, 1], and a wall a block on it meets at x = 0.3 when ``obstacle`` is set.
AXIS = """<mujoco model="axis">
  <worldbody>
    <body name="axis">
      <body name="slider">
        <joint name="x_joint" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="block" type="box" size="0.05 0.05 0.05" mass="1"/>
      </body>
    </body>
    {obstacle}
  </worldbody>
  <actuator>
    {actuator}
  </actuator>
</mujoco>"""
#: A mocap body, so a test can take it away.
WALL = '<body name="wall" mocap="true" pos="0.45 0 0"><geom type="box" size="0.1 0.2 0.2"/></body>'
SERVO = (
    '<position name="x" joint="x_joint" kp="2000" kv="100" ctrlrange="-1 1" forcerange="-50 50"/>'
)


class Axis(MachinePlugin):
    """Machine: one slide axis whose actuator the world states, for the base's refusals and waits.

    Config::

        actuator: ""       # the MJCF <actuator> child driving x_joint, named x (default a servo)
        obstacle: false    # a wall stops the block at x = 0.3
        timeout: 0.0       # MOVE_TIMEOUT for this instance (s); 0 keeps the class's
    """

    ACTUATORS = ("x",)

    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        if float(self.config.get("timeout", 0.0)):
            self.MOVE_TIMEOUT = float(self.config["timeout"])

    def build(self, spec, ctx) -> None:
        xml = AXIS.format(
            actuator=self.config.get("actuator") or SERVO,
            obstacle=WALL if self.config.get("obstacle") else "",
        )
        frame = spec.worldbody.add_frame()
        frame.pos = [0, 0, 0.5]
        spec.attach(mujoco.MjSpec.from_string(xml), prefix="", frame=frame)

    def configure(self, ctx) -> None:
        self.register_entity(ctx, body="axis")
        super().configure(ctx)

    @program
    def go(self, x: float, speed: float = 0.0):
        """Move to x, at once or ramped."""
        yield from self.move("x", x, speed=speed or None, event="go")

    @program
    def nudge(self, x: float, speed: float | None = None):
        """Move to x, ramped when a speed is given."""
        yield from self.move("x", x, speed=speed)

    @program
    def go_then_fail(self, x: float, speed: float, after: float):
        """Ramp toward x and fail after a while, mid-ramp."""
        self.ramp("x", x, speed)
        yield from self.wait(after)
        self.fail("guard open")

    @program
    def go_then_raise(self, x: float, speed: float, after: float):
        """Ramp toward x and raise after a while, mid-ramp (a defect)."""
        self.ramp("x", x, speed)
        yield from self.wait(after)
        raise ValueError("broken")

    @program
    def go_then_linger(self, x: float, speed: float):
        """Ramp toward x, with a cleanup that yields, which a closed generator may not do."""
        self.ramp("x", x, speed)
        try:
            yield from self.wait(100.0)
        finally:
            yield

    @program
    def wait_for(self, seconds: float):
        """Wait on a condition that never holds, for at most ``seconds``."""
        yield from self.until(lambda: False, timeout=seconds, what="the feeder")


class PressController(MachinePlugin):
    """Machine: the press's controller with no model of its own, nested under the ``spawn_model``
    that placed the press's MJCF; it drives that entity's actuators and speaks for it."""

    ACTUATORS = ("door", "ram")
    registers_entity = False

    @program
    def close(self):
        """Close the guard."""
        yield from self.move("door", DOOR_CLOSED, speed=DOOR_SPEED)


class FedPress(Press):
    """Machine: the press, needing a feeder machine's handle."""

    NEEDS = {"feeder": "machine:feeder"}


#: A 5 kg carriage on a vertical slide, held by a position servo of kp 5000: under gravity it
#: stands load/kp = 9.81 mm below its target.
LIFT = """<mujoco model="lift">
  <worldbody>
    <body name="lift" pos="0 0 1">
      <body name="carriage">
        <joint name="z_joint" type="slide" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="box" size="0.05 0.05 0.05" mass="5" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="z" joint="z_joint" kp="5000" kv="500" ctrlrange="-0.5 0.5"/>
  </actuator>
</mujoco>"""


class Lift(MachinePlugin):
    """Machine: a loaded vertical axis, for holding under load."""

    ACTUATORS = ("z",)

    def build(self, spec, ctx) -> None:
        spec.attach(mujoco.MjSpec.from_string(LIFT), prefix="", frame=spec.worldbody.add_frame())

    def configure(self, ctx) -> None:
        self.register_entity(ctx, body="lift")
        super().configure(ctx)

    @program
    def dwell(self, seconds: float):
        """Stand for a while."""
        yield from self.wait(seconds)

    @program
    def go(self, z: float):
        """Move to z at once."""
        yield from self.move("z", z)


class HomeClash(Press):
    """Machine: a handle named like the machine's own ``home``."""

    NEEDS = {"home": "machine:feeder"}
