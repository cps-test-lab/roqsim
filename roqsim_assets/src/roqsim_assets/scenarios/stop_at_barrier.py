"""Scenario: stop a belt while a part is in the light barrier at its end."""

from __future__ import annotations

from roqsim.scenario import ScenarioPlugin


class StopAtBarrier(ScenarioPlugin):
    """Scenario: stop the belt when a part enters the barrier, run it again once the part is gone, at
    the speed the belt was set to run at when it stopped -- read at each stop, so a belt something
    else re-speeded comes back at its new speed, and one stopped while ramping comes back at the
    speed it was ramping to. A belt that stood still when the part arrived stays still.

    The belt runs and stops; the barrier reports presence. That a part in the barrier stops the belt
    is this scene's decision, not the belt's. With ``take_to``, the scene also plays the worker who
    takes the part away after ``hold`` seconds; without it, someone else does.

    Config::

        cycles: 1      # parts stopped for before the scene ends; 0 runs for ever
        hold: 1.0      # with take_to: seconds the part waits in the barrier before it is taken
        take_to: {}    # where a worker puts the part it takes away: a world pose; empty, nobody
    """

    NEEDS = {"belt": "machine:belt", "barrier": "zone:belt_end"}

    def program(self):
        cycles = int(self.config.get("cycles", 1))
        take_to = self.config.get("take_to") or {}
        done = 0
        while not cycles or done < cycles:
            part = yield from self.entered(self.barrier)
            speed = self.belt.goal("speed")  # what it was set to run at, whoever set it
            if self.belt.busy:
                # Something is bringing the belt to a new speed: a part in the barrier stops it
                # anyway, which is this scene's decision. Read the goal first; a cancel holds there.
                self.belt.cancel()
                yield from self.until(lambda: not self.belt.busy, what="the belt's run to end")
            yield from self.run(self.belt, "stop")
            self.event("held", part=part)
            if take_to:
                yield from self.wait(float(self.config.get("hold", 1.0)))
                yield from self.set_entity_state(part, take_to)
            yield from self.left(self.barrier)
            yield from self.run(self.belt, "run", speed=speed)
            self.event("released", part=part)
            done += 1
