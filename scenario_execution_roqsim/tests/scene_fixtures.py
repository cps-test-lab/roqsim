"""One scene written both ways, and the machine and the mover it needs."""

from __future__ import annotations

import mujoco

from roqsim.machine import MachinePlugin, program
from roqsim.plugin import Plugin
from roqsim.scenario import ScenarioPlugin

MODEL = """<mujoco model="cell">
  <worldbody>
    <body name="cell"/>
  </worldbody>
</mujoco>"""

BOX = """<mujoco model="box">
  <worldbody>
    <body name="box" mocap="true" pos="-1 0 0.5">
      <geom size="0.05" contype="0" conaffinity="0" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""


class Press(MachinePlugin):
    """Machine: a press with no joint of its own; a stroke takes a stated time."""

    def build(self, spec, ctx) -> None:
        spec.attach(mujoco.MjSpec.from_string(MODEL), prefix="", frame=spec.worldbody.add_frame())

    def configure(self, ctx) -> None:
        self.register_entity(ctx, body="cell")
        super().configure(ctx)

    @program
    def stroke(self, seconds: float = 0.2):
        """Stroke for a while."""
        yield from self.wait(seconds)
        self.step("stroked")

    @program
    def jam(self):
        """Fail as a blocked machine does."""
        yield from self.wait(0.05)
        self.fail("jammed")


class Box(Plugin):
    """A body moving along x at 1 m/s through the world."""

    provides_entity = True

    def build(self, spec, ctx) -> None:
        spec.attach(mujoco.MjSpec.from_string(BOX), prefix="", frame=spec.worldbody.add_frame())

    def configure(self, ctx) -> None:
        self._mocap = int(ctx.model.body("box").mocapid[0])
        from roqsim.context import Entity

        ctx.entities.add(Entity(self.address, "object", body="box"))

    def on_reset(self, ctx) -> None:
        ctx.data.mocap_pos[self._mocap] = [-1.0, 0.0, 0.5]

    def pre_step(self, ctx) -> None:
        ctx.data.mocap_pos[self._mocap][0] += ctx.dt


class Scene(ScenarioPlugin):
    """Scenario: when the box enters the gate, stroke the press, then wait for the gate to empty."""

    NEEDS = {"press": "machine:press", "gate": "zone:gate"}

    def program(self):
        yield from self.entered(self.gate)
        yield from self.run(self.press, "stroke", seconds=0.1)
        yield from self.left(self.gate)


class Jams(ScenarioPlugin):
    """Scenario: a program that ends in the machine's error."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        yield from self.run(self.press, "jam")
