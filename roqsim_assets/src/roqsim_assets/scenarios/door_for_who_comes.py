"""Scenario: open a door while someone is near it, and close it a while after they have gone."""

from __future__ import annotations

from roqsim.scenario import ScenarioPlugin


class DoorForWhoComes(ScenarioPlugin):
    """Scenario: open the door when its zone reports anyone, close it ``close_delay`` after the zone
    is empty -- unless someone came back meanwhile, when it waits for the zone to empty again.

    While the door closes it watches the zone, as an automatic door does: someone entering it
    (``reopened`` with ``who``), or the close ending in an error such as a stall on something in the
    doorway (``reopened`` with ``error``), cancels the close and opens the door again; it closes
    once the zone has been empty for ``close_delay`` again. A cycle ends with a close that ended
    without an error, so a scene ends with the door shut. A door that keeps failing to close with
    nobody in the zone -- something the zone does not count stands in the doorway -- keeps reopening,
    as an automatic door does, and says ``blocked`` once, at the ``blocked_after``-th such close in a
    row, so a run that ends at its step budget shows why.

    The door only opens and closes; the zone only reports who is there. Which entities count is the
    zone's ``entities`` filter (a walker, a robot), and the decision is this scene's.

    Config::

        close_delay: 2.0   # seconds the zone stays empty before the door closes
        cycles: 1          # open-and-close cycles before the scene ends; 0 runs for ever
        blocked_after: 3   # failed closes in a row, nobody in the zone, before it says blocked
    """

    NEEDS = {"door": "machine:door_1", "near": "zone:door_approach"}

    def program(self):
        delay = float(self.config.get("close_delay", 2.0))
        cycles = int(self.config.get("cycles", 1))
        blocked_after = int(self.config.get("blocked_after", 3))
        done = 0
        while not cycles or done < cycles:
            who = yield from self.entered(self.near)
            self.event("someone_near", who=who)
            failed = 0  # closes in a row that failed with nobody in the zone
            while True:  # until a close ends with the door shut
                yield from self.run(self.door, "open", check=False)
                yield from self._until_gone(delay)
                ended = yield from self._close()
                if ended == "closed":
                    break
                failed = failed + 1 if ended == "failed" else 0
                if failed == blocked_after:
                    self.event("blocked", closes=failed)
            done += 1

    def _until_gone(self, delay: float):
        """Wait until the zone has been empty for ``delay``."""
        while True:
            yield from self.left(self.near)
            yield from self.wait(delay)
            if not self.near.inside():  # nobody came back meanwhile
                return
            self.event("came_back", who=self.near.inside()[0])

    def _close(self):
        """Close the door, watching the zone: ``closed``, ``someone`` (someone entered) or
        ``failed`` (the close ended in an error). Either of the last two says ``reopened``."""
        since = self.ctx.sim_time
        closing = self.start(self.door, "close", check=False)
        comer = self.spawn(self.entered(self.near, since=since), name="comer")
        ended = yield from self.first(closing, comer)
        if ended is comer:
            self.event("reopened", who=comer.result)
            return "someone"
        if closing.result.error:
            self.event("reopened", error=closing.result.error)
            return "failed"
        return "closed"
