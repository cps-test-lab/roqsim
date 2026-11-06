"""Scenario: open a door while someone is near it, and close it a while after they have gone."""

from __future__ import annotations

from roqsim.scenario import ScenarioPlugin


class DoorForWhoComes(ScenarioPlugin):
    """Scenario: open the door when its zone reports anyone, close it ``close_delay`` after the zone
    is empty -- unless someone came back meanwhile, when it waits for the zone to empty again.

    The door only opens and closes; the zone only reports who is there. Which entities count is the
    zone's ``entities`` filter (a walker, a robot), and the decision is this scene's.

    Config::

        close_delay: 2.0   # seconds the zone stays empty before the door closes
        cycles: 1          # open-and-close cycles before the scene ends; 0 runs for ever
    """

    NEEDS = {"door": "machine:door_1", "near": "zone:door_approach"}

    def program(self):
        delay = float(self.config.get("close_delay", 2.0))
        cycles = int(self.config.get("cycles", 1))
        done = 0
        while not cycles or done < cycles:
            who = yield from self.entered(self.near)
            self.event("someone_near", who=who)
            yield from self.run(self.door, "open", check=False)
            while True:
                yield from self.left(self.near)
                yield from self.wait(delay)
                if not self.near.inside():  # nobody came back meanwhile
                    break
                self.event("came_back", who=self.near.inside()[0])
            yield from self.run(self.door, "close", check=False)
            done += 1
