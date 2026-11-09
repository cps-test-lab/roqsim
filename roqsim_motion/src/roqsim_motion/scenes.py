# SPDX-License-Identifier: Apache-2.0
"""Example scenes of a humanoid handling things: the scenarios, written on ``roqsim.scenario``'s
helper, for any humanoid body -- a walker with hands, or a humanoid robot.

Each scene is a world (``roqsim_walker/worlds/handling_*.yaml``; the robot's in ``roqsim_humanoid``)
that runs the scenario below, and a twin world
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
    hands on the totes' handles, once a tote is in the drawer.

    The cell's side is what the cell reports: a ``zone_trigger`` on the drawer (a tote is in it).
    The robot that fills the tote is not part of the scene, and the scene does not wait on what the
    tote shows; its OpenSCENARIO twin waits on the same. The same scene drives a humanoid robot
    (rename the handle with ``needs:``), with its own stand-offs.

    Config::

        standoffs:               # how far in front of what it works the humanoid stands (m)
          drawer: 0.75           #   to pull the drawer open
          push: 0.35             #   to push it shut
          tote: 0.45             #   to take a tote by its handles
          place: 0.55            #   to set a tote down on the trolley or in the drawer
        pull: 0.44               # how far the drawer is pulled out (m)
        strokes: 1               # in how many strokes it is pulled out; it is pushed shut in as many,
                                 #   and in two at least
    """

    NEEDS = {"worker": "humanoid:worker", "drawer": "zone:drawer_zone"}
    STANDOFFS = {"drawer": 0.75, "push": 0.35, "tote": 0.45, "place": 0.55}

    def program(self):
        w = self.worker
        st = {**self.STANDOFFS, **(self.config.get("standoffs") or {})}
        pull = float(self.config.get("pull", 0.44))
        yield from self.entered(self.drawer, "tote_*")
        strokes = int(self.config.get("strokes", 1))
        for _ in range(strokes):
            yield from self.run(w, "approach", target="cell/drawer_1/grip", standoff=st["drawer"])
            yield from self.run(w, "grasp", hand="right", target="cell/drawer_1/grip")
            yield from self.run(w, "pull", hand="right", distance=pull / strokes)
            yield from self.run(w, "release", hand="right")
        yield from self.run(w, "approach", target="tote_full", standoff=st["tote"])
        yield from self.run(w, "grasp", hand="both", target="tote_full")
        yield from self.run(w, "carry", to="trolley", standoff=0.9)
        yield from self.run(w, "approach", target="trolley", standoff=st["place"], facing=math.pi)
        yield from self.run(w, "place", at="trolley")
        yield from self.run(w, "approach", target="tote_empty", standoff=st["tote"])
        yield from self.run(w, "grasp", hand="both", target="tote_empty")
        yield from self.run(w, "carry", to="cell/drawer_1/inside", standoff=0.9)
        yield from self.run(w, "approach", target="cell/drawer_1/inside", standoff=st["place"] - 0.05, facing=0.0)
        yield from self.run(w, "place", at="cell/drawer_1/inside")
        # A drawer pushed shut from where it was pulled open: in strokes, stepping up between.
        pushes = max(2, strokes)
        for distance in [pull / pushes] * pushes:
            yield from self.run(w, "approach", target="cell/drawer_1/grip", standoff=st["push"])
            yield from self.run(w, "grasp", hand="right", target="cell/drawer_1/grip")
            yield from self.run(w, "push", hand="right", distance=distance)
            yield from self.run(w, "release", hand="right")
