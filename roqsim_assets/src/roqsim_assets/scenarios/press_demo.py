"""Scenario: place a strip in the press's die and punch it."""

from __future__ import annotations

from roqsim.scenario import ScenarioPlugin


class PressDemo(ScenarioPlugin):
    """Scenario: place a part in the press's die, then start the press and wait for its stroke.

    The press decides how a stroke runs -- its guard, its interlock, its three phases, the part's
    state at the bottom; this scene decides only that the part goes in and when the press starts.

    Config::

        part: strip_1        # the part's entity name
        die_pose:            # where the part goes: a world pose (roqsim.pose)
          position: {x: 0.0, y: 0.0, z: 0.945}
        show: hole           # the state the stroke shows on the part
        hide: ""             # the state it hides
    """

    NEEDS = {"press": "machine:press_1"}

    def program(self):
        part = self.config.get("part", "strip_1")
        die_pose = self.config.get("die_pose") or {"position": {"x": 0.0, "y": 0.0, "z": 0.945}}
        yield from self.set_entity_state(part, die_pose)
        yield from self.wait(0.2)  # let it settle on the die
        status = yield from self.run(
            self.press,
            "stroke",
            show=str(self.config.get("show", "hole")),
            hide=str(self.config.get("hide", "")),
        )
        self.event("punched", part=part, seq=status.seq)
