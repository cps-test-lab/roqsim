# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Scenarios the scenario tests run, against the machines of :mod:`machine_fixtures`."""

from __future__ import annotations

from roqsim.plugin import Plugin
from roqsim.scenario import ScenarioError, ScenarioPlugin


class TwoStrokes(ScenarioPlugin):
    """Scenario: stroke the press twice, a pause between.

    Config::

        pause: 0.2   # seconds between the strokes
    """

    NEEDS = {"press": "machine:press"}

    def program(self):
        first = yield from self.run(self.press, "stroke")
        self.event("first", seq=first.seq)
        yield from self.wait(float(self.config.get("pause", 0.2)))
        yield from self.run(self.press, "stroke")
        self.event("second")


class Jammed(ScenarioPlugin):
    """Scenario: run a program that fails; ``check: false`` reads the status instead.

    Config::

        check: true   # whether the failure raises
    """

    NEEDS = {"press": "machine:press"}

    def program(self):
        status = yield from self.run(self.press, "jam", check=bool(self.config.get("check", True)))
        self.event("after", error=status.error)


class Concurrent(ScenarioPlugin):
    """Scenario: two tasks beside each other, joined, then a race the shorter wins."""

    NEEDS = {"press": "machine:press", "spin": "machine:spin"}

    def program(self):
        stroke = self.start(self.press, "dwell", seconds=0.3)
        spin = self.start(self.spin, "run", speed=0.4)
        yield from self.join(stroke, spin)
        self.event("joined")
        short = self.spawn(self._ticks("short", 0.1), name="short")
        long = self.spawn(self._ticks("long", 1.0), name="long")
        winner = yield from self.first(short, long)
        self.event("won", by=winner.name, long_cancelled=long.cancelled)

    def _ticks(self, name, seconds):
        yield from self.wait(seconds)
        self.event(f"{name}_done")


class Walk(ScenarioPlugin):
    """Scenario: send a walker along two points and wait for it."""

    NEEDS = {"walker": "walker:worker"}

    def program(self):
        status = yield from self.route(self.walker, [(1.0, 0.0), (1.0, 1.0)], check=False)
        self.event("arrived", error=status[-1])


class FakeWalker(Plugin):
    """A stand-in for a walker's handle: a route takes ``route_s`` and ends with ``error``.

    Config::

        route_s: 0.5   # how long a route takes
        error: ""      # what it ends with
    """

    def configure(self, ctx):
        self._ctx = ctx
        self.seq, self.finished, self.until = 0, False, None
        ctx.blackboard.set("walker:worker", self)
        self.name = "worker"

    def send_route(self, poses):
        self.seq += 1
        self.finished = False
        self.until = self._ctx.sim_time + float(self.config.get("route_s", 0.5))
        self.poses = poses
        return self.seq

    def status(self):
        error = self.config.get("error", "") if self.finished else ""
        return (self.seq, self.finished, 0, 0.0, error)

    def post_step(self, ctx):
        if self.until is not None and ctx.sim_time >= self.until:
            self.finished, self.until = True, None


class Place(ScenarioPlugin):
    """Scenario: place the block at the pose its config gives.

    Config::

        pose: {}   # the pose place() is given
    """

    def program(self):
        yield from self.place("block", self.config.get("pose"))
        self.event("placed")


class Impatient(ScenarioPlugin):
    """Scenario: wait for a condition that never holds, for a moment."""

    def program(self):
        yield from self.until(lambda: False, timeout=0.2, what="the feeder")


class Broken(ScenarioPlugin):
    """Scenario: a program with a defect."""

    def program(self):
        yield
        raise KeyError("no such part")


class Clashing(ScenarioPlugin):
    """Scenario: a handle named like the scenario's own config."""

    NEEDS = {"config": "machine:press"}

    def program(self):
        yield


class CatchAtJoin(ScenarioPlugin):
    """Scenario: two starts beside each other, one of which jams; the join catches its error."""

    NEEDS = {"press": "machine:press", "spin": "machine:spin"}

    def program(self):
        jam = self.start(self.press, "jam")
        spin = self.start(self.spin, "run", speed=0.4)
        try:
            yield from self.join(jam, spin)
        except ScenarioError as exc:
            self.event("caught", error=str(exc))
        yield from self.wait(0.05)
        self.event("went_on")


class Unjoined(ScenarioPlugin):
    """Scenario: a start that jams, never joined."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        self.start(self.press, "jam")
        yield from self.wait(0.5)


class ReturnsWhileAStartRuns(ScenarioPlugin):
    """Scenario: a start that jams later, the program returning at once."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        self.start(self.press, "jam")
        yield


class ReturnsBesideAGoodStart(ScenarioPlugin):
    """Scenario: a start that succeeds, the program returning before it ends."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        self.start(self.press, "stroke")
        yield


class Outlasts(ScenarioPlugin):
    """Scenario: a start that jams, the program outlasting any run."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        self.start(self.press, "jam")
        while True:
            yield


class TimesOut(ScenarioPlugin):
    """Scenario: a timeout race a timer wins over a machine's start."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        timer = self.spawn(self.wait(0.05), name="timer")
        yield from self.first(self.start(self.press, "stroke"), timer)
        yield from self.run(self.press, "stroke")


class FirstOfNothing(ScenarioPlugin):
    """Scenario: first() with no tasks."""

    def program(self):
        yield from self.first()


class WaitNaN(ScenarioPlugin):
    """Scenario: a wait that is no number of seconds."""

    def program(self):
        yield from self.wait(float("nan"))


class NoYield(ScenarioPlugin):
    """Scenario: a program with no yield."""

    def program(self):
        return None


class CancelsAFinishedStart(ScenarioPlugin):
    """Scenario: cancel a start that has ended, with the next start not yet applied."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        earlier = self.start(self.press, "dwell", seconds=0.05)
        yield from self.until(lambda: not self.press.busy)
        later = self.start(self.press, "dwell", seconds=0.1)
        earlier.cancel()
        yield from self.join(later)


class WatcherBeforeItsStart(ScenarioPlugin):
    """Scenario: a task created before the start it joins, the program returning at once."""

    NEEDS = {"press": "machine:press"}

    def program(self):
        made = []

        def watch():
            while not made:
                yield
            try:
                yield from self.join(made[0])
            except ScenarioError as exc:
                self.event("caught", error=str(exc))

        self.spawn(watch(), name="watch")
        made.append(self.start(self.press, "jam"))
        yield


class UntilNaN(ScenarioPlugin):
    """Scenario: a condition with a timeout that is no number."""

    def program(self):
        yield from self.until(lambda: False, timeout=float("nan"))


class Overtaken(ScenarioPlugin):
    """Scenario: send a route that another overtakes, without checking."""

    NEEDS = {"walker": "walker:worker"}

    def program(self):
        route = self.spawn(self.route(self.walker, [(1.0, 0.0)], check=False), name="route")
        yield  # the route task sends its route (seq 1) in the next step
        yield
        self.walker.send_route([(2.0, 0.0)])  # seq 2 overtakes it
        yield from self.join(route)
        self.event("overtaken", seq=route.result[0])
