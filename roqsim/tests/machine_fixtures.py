# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Machines the machine, scenario and zone tests run: generated models, no file from outside.

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
