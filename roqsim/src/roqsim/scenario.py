# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Python scenarios: the scene, written as a generator on simulated time.

A scenario says which entities act when, who waits for whom, what a presence or an event leads to,
and the order of the work. It drives entities only through their handles -- a machine's
(:mod:`roqsim.machine`), a walker's, a zone's -- and it owns no model, writes no actuator and
touches no ``data`` but to place an entity, so it cannot become a machine by accident. The line
between the two is ``docs/architecture.rst``, "Machines and scenarios". The same scene may be
written as OpenSCENARIO and run by scenario-execution instead.

A scenario is a :class:`ScenarioPlugin` whose :meth:`~ScenarioPlugin.program` is a generator; each
``yield`` is one physics step::

    class PressDemo(ScenarioPlugin):
        \"\"\"Scenario: place a strip in the press and punch it.\"\"\"
        NEEDS = {"press": "machine:press_1"}     # resolved at reset, refused when missing

        def program(self):
            yield from self.place("strip_1", {"position": {"x": 0.0, "y": 0.0, "z": 0.8}})
            yield from self.run(self.press, "stroke", show="hole", hide="plain")
            self.event("punched", part="strip_1")

The program advances in ``post_step``. A world declares a scenario after the machines it drives,
so within a step it sees their state; a start it makes is applied at the next step, whatever the
order. Tasks (:meth:`~ScenarioPlugin.spawn`, :meth:`~ScenarioPlugin.start`) advance once per step
in the order they were created, so a scenario is deterministic: no threads, no wall clock.

When ``program`` has returned and every start it did not join has ended, the scenario emits
``scenario_end`` and its ``done`` endpoint turns true,
for an OpenSCENARIO monitor to wait on; with ``stop_at_end: true`` it also ends a standalone run
(``ctx.request_stop``). A scenario-execution run is ended by its scenario, never by a plugin. A
program that raises stops the run naming the scenario; a machine's start that ends in an error
raises :class:`ScenarioError` in the program that waited for it, unless it asked for the status.
"""

from __future__ import annotations

import fnmatch
import inspect
import math
from collections.abc import Callable, Iterator

from . import endpoint
from .context import SimContext
from .plugin import Plugin


class ScenarioError(RuntimeError):
    """What a scenario waited for ended in an error, or a handle it needs is not there."""


class Task:
    """A generator a scenario runs beside its program: one step per physics step.

    ``done`` once it returned, failed or was cancelled; ``result`` is what it returned. A task that
    keeps its error (a :meth:`ScenarioPlugin.start` task) ends on it with ``error`` set, and the
    error is raised to whoever joins the task.
    """

    def __init__(self, gen: Iterator, name: str = "", *, keeps_error: bool = False):
        self._gen = gen
        self.name = name
        self.done = False
        self.cancelled = False
        self.result = None
        #: Whether an error ends the task for its joiner rather than stopping the run at once.
        self.keeps_error = keeps_error
        #: The error the task ended on, kept for whoever joins it.
        self.error: BaseException | None = None
        #: Whether ``error`` has reached a joiner.
        self.delivered = False
        #: Whether ``error`` has gone into the events.
        self.said = False
        #: Called when the task is cancelled, to stop what it started.
        self.on_cancel: Callable[[], None] | None = None

    def advance(self) -> None:
        if self.done:
            return
        try:
            next(self._gen)
        except StopIteration as stop:
            self.done = True
            self.result = stop.value
        except Exception as exc:
            if not self.keeps_error:
                raise
            self.done = True
            self.error = exc

    def take_error(self) -> BaseException | None:
        """The error the task ended on, once: the first caller receives it."""
        if self.error is None or self.delivered:
            return None
        self.delivered = True
        return self.error

    def cancel(self) -> None:
        """Stop the task at its current ``yield``, and what it started where it says how
        (:meth:`ScenarioPlugin.start` cancels its machine's program)."""
        if not self.done:
            self._gen.close()
            self.done = True
            self.cancelled = True
            if self.on_cancel is not None:
                self.on_cancel()


class ScenarioPlugin(Plugin):
    """Base of a Python scenario: the scene, driving entities through their handles.

    The scene, not a machine -- see ``docs/architecture.rst``, "Machines and scenarios", and
    :mod:`roqsim.scenario` for the shape. A subclass writes :meth:`program` and names the handles it
    needs in :attr:`NEEDS`.

    Config (every scenario; a subclass adds its own keys)::

        needs: {press: machine:press_2}   # rename a handle NEEDS names, so one scenario serves two cells
        stop_at_end: false                # end a standalone run when the scene ends
    """

    #: Blackboard handles the scenario drives, attribute -> key. Each is resolved at every reset,
    #: after every plugin configured, and set as an attribute; a missing one is refused by name.
    NEEDS: dict[str, str] = {}

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._ctx: SimContext | None = None
        self._tasks: list[Task] = []
        #: Every task that keeps its error, so one that nobody joins still stops the run.
        self._kept: list[Task] = []
        self._main: Task | None = None
        #: Whether the end was reached on this pass, leaving one more for a joiner to take an error.
        self._closing = False
        self._done = False
        #: Set when a task raised: the run stops, and the scene never reports done.
        self._failed = False

    # -- the scene ----------------------------------------------------------------------------
    def program(self) -> Iterator:
        """The scene: a generator, one ``yield`` per physics step. A subclass writes it."""
        raise NotImplementedError(f"scenario {self.address}: a scenario writes program()")

    # -- lifecycle ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = []
        needs = config.get("needs")
        if needs is not None:
            if not isinstance(needs, dict):
                errors.append(f"scenario {self.address}: 'needs' maps a handle's name to its key")
            else:
                unknown = sorted(set(needs) - set(self.NEEDS))
                if unknown:
                    errors.append(
                        f"scenario {self.address}: 'needs' names {unknown}, which it does not use "
                        f"(it needs {sorted(self.NEEDS) or 'nothing'})"
                    )
        if not isinstance(config.get("stop_at_end", False), bool):
            errors.append(f"scenario {self.address}: 'stop_at_end' is true or false")
        # A handle is set as an attribute of that name, so a name the scenario already has would
        # replace it -- its context, its config, a method of the helper.
        taken = sorted(n for n in self.NEEDS if hasattr(type(self), n) or n in vars(self))
        if taken:
            errors.append(
                f"scenario {self.address}: NEEDS names {taken}, which the scenario already has; "
                "name the handle otherwise"
            )
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx

    def on_reset(self, ctx: SimContext) -> None:
        """Resolve the handles and start the program from its beginning."""
        self._say_unjoined(ctx)
        for task in self._tasks:
            task.on_cancel = None  # a reset ends what it started as well
            task.cancel()
        needs = {**self.NEEDS, **(self.config.get("needs") or {})}
        for attr, key in needs.items():
            value = ctx.blackboard.get(key)
            if value is None:
                raise ScenarioError(
                    f"scenario {self.address}: needs {key!r}, which no entry of the world published"
                )
            setattr(self, attr, value)
        self._done = False
        self._failed = False
        self._closing = False
        self._tasks = []
        self._kept = []
        program = self.program()
        if not inspect.isgenerator(program):
            raise ScenarioError(
                f"scenario {self.address}: program() is a generator, with one yield per physics "
                "step; this one has no yield"
            )
        self._main = self.spawn(program, name="program")

    def post_step(self, ctx: SimContext) -> None:
        """Advance every task by one step, in the order they were created."""
        if self._main is None or self._failed:
            return
        for task in list(self._tasks):
            try:
                task.advance()
            except Exception as exc:
                self._fail(exc)
        self._tasks = [t for t in self._tasks if not t.done]
        for task in self._kept:
            if task.error is not None and not task.said:
                task.said = True
                ctx.events.emit(
                    self.address, "start_failed", start=task.name, error=str(task.error)
                )
        if self._main.done and not self._done:
            # A start nobody joined is not abandoned: the scene ends when it has, and its error
            # stops the run here.
            if not all(task.done for task in self._kept):
                self._closing = False
                return
            # One more pass for an error: a joiner created before the start has not yet seen it end.
            if not self._closing and any(
                t.error is not None and not t.delivered for t in self._kept
            ):
                self._closing = True
                return
            for task in self._kept:
                error = task.take_error()
                if error is not None:
                    self._fail(error)
            self._done = True
            for task in self._tasks:
                task.cancel()
            self._tasks = []
            ctx.events.emit(self.address, "scenario_end")
            if self.config.get("stop_at_end", False):
                ctx.request_stop(f"scenario {self.address} done")

    def shutdown(self, ctx: SimContext) -> None:
        self._say_unjoined(ctx)

    def _say_unjoined(self, ctx: SimContext) -> None:
        """Log a start's error that reached no one, for a trial a reset or the run's end cut short:
        neither can fail the run (:meth:`roqsim.engine.Engine.shutdown`). ``start_failed`` is in the
        events already, from the step the start failed in."""
        for task in self._kept:
            error = task.take_error()
            if error is not None:
                ctx.logger.error(
                    "scenario %s: a start's error reached no one: %s", self.address, error
                )

    def _fail(self, exc: BaseException):
        """Stop the scene on ``exc``: the run stops, and ``done`` never turns true."""
        self._failed = True
        if isinstance(exc, ScenarioError):
            raise ScenarioError(f"scenario {self.address}: {exc}") from exc
        # The program's own exception, of its own type, with the scenario named beside it.
        note = f"raised in scenario {self.address}'s program"
        if hasattr(exc, "add_note"):  # Python 3.11+
            exc.add_note(note)
        else:
            self._ctx.logger.error("%s: %s", note, exc)
        raise exc

    @endpoint.out(rate=10.0)
    def done(self) -> bool:
        """Whether the scene has ended: its program returned, and every start it did not join ended."""
        return self._done

    # -- the helper ---------------------------------------------------------------------------
    @property
    def ctx(self) -> SimContext:
        return self._ctx

    def spawn(self, gen: Iterator, name: str = "", *, keeps_error: bool = False) -> Task:
        """Run ``gen`` beside the program; it advances from the next step on. An error it raises
        stops the run, unless ``keeps_error``: then it ends the task and reaches whoever joins it."""
        task = Task(gen, name, keeps_error=keeps_error)
        self._tasks.append(task)
        if keeps_error:
            self._kept.append(task)
        return task

    def start(self, handle, program: str, *, check: bool = True, **args) -> Task:
        """Start ``program`` on a machine's handle and return a task that ends when it does; the
        task's ``result`` is the status. With ``check``, the machine's error ends the task as a
        :class:`ScenarioError` that :meth:`join` or :meth:`first` raises in whoever waits on it -- a
        ``try`` around the join catches it, as around :meth:`run`, and reaches the first to join it.
        The scene does not end before the start has; an error nobody joined then stops the run. A
        run that ends first logs that error. Cancelling the task cancels this start's program, so
        ``first(start(...), timer)`` leaves the machine free."""
        seq = handle.start(program, **args)
        task = self.spawn(
            self._await_start(handle, program, seq, check),
            name=f"{handle.name}.{program}",
            keeps_error=True,
        )

        def cancel_the_program() -> None:
            # A finished start has nothing to cancel, and a later one is not this task's.
            status = handle.status()
            if status.seq < seq or (status.seq == seq and not status.finished):
                handle.cancel()

        task.on_cancel = cancel_the_program
        return task

    def run(self, handle, program: str, *, check: bool = True, **args):
        """Start ``program`` on a machine's handle and wait until this start ended. Raises
        :class:`ScenarioError` with the machine's error, unless ``check=False``; returns the
        :class:`~roqsim.machine.MachineStatus`."""
        seq = handle.start(program, **args)
        return (yield from self._await_start(handle, program, seq, check))

    def _await_start(self, handle, program: str, seq: int, check: bool):
        while True:
            status = handle.status()
            if status.seq == seq and status.finished:
                break
            if status.seq > seq:
                if check:
                    raise ScenarioError(
                        f"{handle.name}.{program} (seq {seq}) was followed by seq {status.seq} "
                        "before it ended"
                    )
                return status
            yield
        if status.error and check:
            raise ScenarioError(f"{handle.name}.{program} (seq {seq}): {status.error}")
        return status

    def route(self, mover, poses, *, check: bool = True):
        """Send a walker (``send_route``) or a navigator (``send_goals``) along ``poses`` and wait
        until that route ended; a route that did not reach its goals raises :class:`ScenarioError`
        unless ``check=False``. A route another overtook raises the same way, and with ``check=False``
        returns the status, as :meth:`run` does. Returns the mover's status tuple."""
        send = getattr(mover, "send_route", None) or mover.send_goals
        seq = send([tuple(p) for p in poses])
        while True:
            status = mover.status()
            if status[0] == seq and status[1]:
                break
            if status[0] > seq:
                if check:
                    raise ScenarioError(f"{mover.name}: route {seq} was followed by {status[0]}")
                return status
            yield
        if status[-1] and check:
            raise ScenarioError(f"{mover.name}: route {seq}: {status[-1]}")
        return status

    def join(self, *tasks: Task):
        """Wait until every task is done; raise the first error one of them ended on. An error
        reaches one joiner: a second join of that task returns."""
        while not all(t.done for t in tasks):
            yield
        for task in tasks:
            error = task.take_error()
            if error is not None:
                raise error

    def first(self, *tasks: Task):
        """Wait until one task is done, cancel the others, and return the one that finished. With
        no task that can finish, it raises :class:`ScenarioError` rather than waiting forever."""
        while True:
            if all(task.cancelled for task in tasks):
                raise ScenarioError("first() was given no task that can finish")
            for task in tasks:
                if task.done and not task.cancelled:
                    for other in tasks:
                        if other is not task:
                            other.cancel()
                    error = task.take_error()
                    if error is not None:
                        raise error
                    return task
            yield

    def until(self, cond: Callable[[], bool], timeout: float | None = None, what: str = ""):
        """Yield until ``cond()`` holds; past ``timeout`` seconds raise :class:`ScenarioError`."""
        if timeout is not None and math.isnan(timeout):
            raise ScenarioError(f"until() takes a timeout in seconds, not {timeout!r}")
        start = self._ctx.sim_time
        while not cond():
            if timeout is not None and self._ctx.sim_time - start > timeout:
                raise ScenarioError(f"{what or 'the condition'} not true after {timeout:g} s")
            yield

    def wait(self, seconds: float):
        """Yield for ``seconds`` of simulated time."""
        if not math.isfinite(seconds):
            raise ScenarioError(f"wait() takes a number of seconds, not {seconds!r}")
        end = self._ctx.sim_time + float(seconds)
        while self._ctx.sim_time < end - 1e-9:
            yield

    def entered(
        self, zone, pattern: str = "*", timeout: float | None = None, since: float | None = None
    ):
        """Wait until an entity matching ``pattern`` is inside ``zone`` (a ``zone_trigger``'s handle)
        and return its name -- at once when one already is. With ``since`` (s, sim time), an entity
        that entered at or after it counts even if it has left again, so a scene that was busy
        cannot miss one passing through. Past ``timeout`` seconds raise :class:`ScenarioError`, as
        :meth:`until` does."""
        names = zone.inside if since is None else lambda: zone.entered_since(since)
        return (
            yield from self._first_of(
                names, pattern, timeout, f"an entity matching {pattern!r} in zone {zone.name}"
            )
        )

    def left(self, zone, pattern: str = "*", timeout: float | None = None):
        """Wait until no entity matching ``pattern`` is inside ``zone``; ``timeout`` as
        :meth:`entered`. It reads who is inside now: list a scenario after the zones it waits on, so
        that on the first step a zone has already said."""
        yield from self.until(
            lambda: not any(fnmatch.fnmatchcase(n, pattern) for n in zone.inside()),
            timeout,
            f"zone {zone.name} empty of {pattern!r}",
        )

    def dwelled(
        self, zone, pattern: str = "*", timeout: float | None = None, since: float | None = None
    ):
        """Wait until an entity matching ``pattern`` has been inside ``zone`` for its ``dwell``, and
        return its name; ``timeout`` and ``since`` as :meth:`entered`. A zone whose ``dwell`` is 0
        reports no dwelling at all, so waiting on one is refused rather than left to hang."""
        if not zone.dwell > 0:
            raise ScenarioError(
                f"zone {zone.name} has dwell 0, which reports no dwelling; give it a dwell to "
                "wait on one"
            )
        return (
            yield from self._first_of(
                zone.dwelling if since is None else lambda: zone.dwelled_since(since),
                pattern,
                timeout,
                f"an entity matching {pattern!r} dwelling in zone {zone.name}",
            )
        )

    def _first_of(self, names: Callable[[], list[str]], pattern: str, timeout, what: str):
        """Yield until one of ``names()`` matches ``pattern``, and return it; past ``timeout``
        seconds raise :class:`ScenarioError` saying ``what`` was waited for."""
        start = self._ctx.sim_time
        while True:
            for name in names():
                if fnmatch.fnmatchcase(name, pattern):
                    return name
            if timeout is not None and self._ctx.sim_time - start > timeout:
                raise ScenarioError(f"{what} not true after {timeout:g} s")
            yield

    def place(self, entity: str, pose: dict):
        """Put ``entity`` at ``pose``, at rest, as ``set_entity_state`` does; one step.

        ``pose`` is a world pose as a world file writes one (:func:`roqsim.pose.parse_pose`):
        ``position`` with ``x`` and ``y`` required and ``z`` optional, ``orientation`` as a
        quaternion or ``roll``/``pitch``/``yaw`` (identity when absent), bare or under a
        ``PoseStamped``'s ``header``/``pose`` in the world frame. A missing ``z`` is the resting
        height the world's ``spawn_model`` gives a pose without one: 0.0, the floor, where a model
        authored with its origin on the floor stands. Anything else -- not a mapping, a missing
        ``x`` or ``y``, a number that is not finite, an unknown key, a frame other than the
        world's -- raises :class:`roqsim.pose.PoseError` naming what is wrong. An entity the world
        does not have raises :class:`roqsim.entity_control.UnknownEntity`, and one that cannot be
        placed (welded scenery) :class:`roqsim.entity_control.EntityRefused`."""
        from .entity_control import set_state
        from .pose import parse_pose

        position, quat = parse_pose(pose)
        x, y, z = position
        set_state(self._ctx, entity, [x, y, 0.0 if z is None else z], quat)
        yield

    def event(self, name: str, **fields) -> dict:
        """Emit an event of the scene, with the scenario as its source."""
        return self._ctx.events.emit(self.address, name, **fields)
