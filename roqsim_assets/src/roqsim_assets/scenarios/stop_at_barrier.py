"""Scenario: stop a belt while a part is in the light barrier at its end."""

from __future__ import annotations

from roqsim.scenario import ScenarioPlugin


class StopAtBarrier(ScenarioPlugin):
    """Scenario: stop the belt when a part enters the barrier, run it again once the part is gone.

    The belt runs and stops; the barrier reports presence. That a part in the barrier stops the belt
    is this scene's decision, not the belt's.
    """

    NEEDS = {"belt": "machine:belt", "barrier": "zone:belt_end"}

    def program(self):
        speed = self.belt.value("speed")
        while True:
            part = yield from self.entered(self.barrier)
            yield from self.run(self.belt, "stop")
            self.event("held", part=part)
            yield from self.left(self.barrier)
            yield from self.run(self.belt, "run", speed=speed)
            self.event("released", part=part)
