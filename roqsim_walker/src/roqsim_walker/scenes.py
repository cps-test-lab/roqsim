"""Example scenes of a worker with hands: the scenarios, written on ``roqsim.scenario``'s helper.

Each scene is a world in ``roqsim_walker/worlds/`` that runs the scenario below, and a twin world
(``*_osc.yaml``) that runs the same scene as OpenSCENARIO (``*.osc``, with ``osc.roqsim``'s
``humanoid_*`` actions). The two give one timeline. A scenario decides which moves the worker makes
and when; each move is the worker's own (``roqsim_motion.moves``), and each machine runs its own
programs (``docs/architecture.rst``, "Machines and scenarios").

**Scene 1's machine is a seam.** :class:`StartMachine` drives whatever the world places as
``press_1`` through the machine contract, and runs its program :attr:`StartMachine.PROGRAM` -- the
only name the scene knows of it. The world places ``roqsim_assets``' example press there; another
machine takes its place by swapping the world's ``press_1`` entry and setting ``PROGRAM`` (here, and
the one ``machine_run`` line of ``handling_start_machine.osc``) to the program that starts it.
"""

from __future__ import annotations

import math

from roqsim.scenario import ScenarioPlugin


class StartMachine(ScenarioPlugin):
    """Scenario 1: a worker walks to a machine, presses its start button, and the machine runs."""

    NEEDS = {"worker": "humanoid:worker", "press": "machine:press_1"}
    #: The program that starts the machine: the seam's one name (module docstring).
    PROGRAM = "stroke"

    def program(self):
        yield from self.run(self.worker, "approach", target="start/button", standoff=0.5)
        yield from self.run(self.worker, "press", hand="right", target="start/button")
        yield from self.run(self.press, self.PROGRAM)


class DrawerToBench(ScenarioPlugin):
    """Scenario 2: a worker opens a drawer, takes a tote out by its handles, carries it to a bench
    and sets it down."""

    NEEDS = {"worker": "humanoid:worker"}

    def program(self):
        w = self.worker
        yield from self.run(w, "approach", target="cabinet/drawer_1/grip", standoff=0.75)
        yield from self.run(w, "grasp", hand="right", target="cabinet/drawer_1/grip")
        yield from self.run(w, "pull", hand="right", distance=0.44)
        yield from self.run(w, "release", hand="right")
        yield from self.run(w, "approach", target="tote_1", standoff=0.45)
        yield from self.run(w, "grasp", hand="both", target="tote_1")
        yield from self.run(w, "carry", to="bench", standoff=0.9)
        yield from self.run(w, "approach", target="bench", standoff=0.55, facing=math.pi / 2)
        yield from self.run(w, "place", at="bench")


class BoxSwap(ScenarioPlugin):
    """Scenario 3: at a robot cell's drawer, a worker swaps the full tote for an empty one, its
    hands on the totes' handles, once the cell's tote shows full.

    The cell's side is what the cell reports: a ``zone_trigger`` on the drawer (a tote is in it) and
    the tote's ``part_state`` (``full``). The robot that fills it is not part of the scene."""

    NEEDS = {"worker": "humanoid:worker", "states": "part_state", "drawer": "zone:drawer_zone"}

    def program(self):
        w = self.worker
        yield from self.entered(self.drawer, "tote_*")
        yield from self.until(lambda: self.states.is_shown("tote_full", "full"), what="the cell's tote is full")
        yield from self.run(w, "approach", target="cell/drawer_1/grip", standoff=0.75)
        yield from self.run(w, "grasp", hand="right", target="cell/drawer_1/grip")
        yield from self.run(w, "pull", hand="right", distance=0.44)
        yield from self.run(w, "release", hand="right")
        yield from self.run(w, "approach", target="tote_full", standoff=0.45)
        yield from self.run(w, "grasp", hand="both", target="tote_full")
        yield from self.run(w, "carry", to="trolley", standoff=0.9)
        yield from self.run(w, "approach", target="trolley", standoff=0.55, facing=math.pi)
        yield from self.run(w, "place", at="trolley")
        yield from self.run(w, "approach", target="tote_empty", standoff=0.45)
        yield from self.run(w, "grasp", hand="both", target="tote_empty")
        yield from self.run(w, "carry", to="cell/drawer_1/inside", standoff=0.9)
        yield from self.run(w, "approach", target="cell/drawer_1/inside", standoff=0.5, facing=0.0)
        yield from self.run(w, "place", at="cell/drawer_1/inside")
        # A drawer pushed shut from where it was pulled open: in two strokes, stepping up between.
        for distance in (0.2, 0.22):
            yield from self.run(w, "approach", target="cell/drawer_1/grip", standoff=0.45)
            yield from self.run(w, "grasp", hand="right", target="cell/drawer_1/grip")
            yield from self.run(w, "push", hand="right", distance=distance)
            yield from self.run(w, "release", hand="right")
