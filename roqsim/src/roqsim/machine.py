# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Machines: one entity's own procedures, offered through one handle.

A machine is what one entity can do by itself -- move its actuators, run its programs, be in its
states. It acts only on itself and on what it physically works on (the part in its tool, through
that part's handle), and it decides nothing about other entities: who acts when, and what a presence
leads to, is a scenario's (:mod:`roqsim.scenario`). The line is drawn in ``docs/architecture.rst``,
"Machines and scenarios".

A machine package's plugin subclasses :class:`MachinePlugin` and writes its programs as generator
methods marked :func:`program`; each ``yield`` is one physics step on simulated time::

    class PressPlugin(MachinePlugin):
        ACTUATORS = ("door", "ram")          # position actuators, by name after the prefix
        HOME = {"door": 0.0, "ram": 0.0}     # where they stand after a reset; `home:` overrides

        @program
        def stroke(self, show: str = ""):
            \"\"\"Close the guard, make one stroke, open the guard.\"\"\"
            yield from self.move("door", 0.4, speed=0.2, event="door_close")
            yield from self.until_stopped("door")          # the interlock
            yield from self.move("ram", 0.1, speed=0.03, event="power")
            ...

The base gives every machine the same outside:

* a :class:`MachineHandle` on the blackboard under ``machine:<name>``: ``start(program, **args)``
  returning a sequence number, ``status()`` (a :class:`MachineStatus`), ``busy``, ``ready``,
  ``cancel()`` and ``value(name)``. A start while busy is refused at the call. It is the contract
  of the walker's ``send_route``/``status`` made general: a caller knows its own start ended when
  ``status().seq`` is the number it was given and ``finished`` is set;
* an event on ``ctx.events`` (:mod:`roqsim.events`) for each step a program names, between a
  ``program_start`` and a ``program_end``, so a recording carries the machine's timeline;
* the endpoints ``start``, ``status`` and ``cancel`` on the machine's entity, for the control socket
  (they are kept off ROS);
* actuators ramped at a stated speed, in the joint's unit per second, as a cylinder or a servo axis
  moves: a ramp moves the position actuator's target and the physics moves the joint. An actuator a
  machine drives is a plain position servo (fixed gain, affine bias whose stiffness matches the gain,
  unit gear), so its control is the joint's target; any other is refused. Plugin-owned
  :attr:`~MachinePlugin.SETPOINTS` (a belt's speed) ramp the same way and are applied by the plugin.

Fail loudly: an actuator the model lacks or one that is no plain position servo, a ``home`` the
machine does not drive or outside its range, a handle in :attr:`~MachinePlugin.NEEDS` that nothing
published, an unknown program or argument -- each is said naming the machine. A program fails, with
the error in ``status()`` and the run going on, when it calls :meth:`MachinePlugin.fail` (a door
stalled), when it sends an actuator outside its range, and when a joint has not arrived
:attr:`~MachinePlugin.MOVE_TIMEOUT` seconds after its target did (an obstruction). A program that raises anything else is a
defect: with ``on_error: raise`` (the default) the run stops naming the machine and the program;
with ``on_error: report`` the error goes into ``status()``. A program that ended in an error, as a
cancelled one, leaves the machine holding where it is.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import mujoco

from . import endpoint
from .context import Entity, SimContext
from .plugin import Plugin

#: How near its target a position actuator's joint is when it has reached it (the joint's unit).
REACHED = 0.002
#: How slowly a joint that has reached its target may still move to count as stopped (unit/s).
STOPPED_SPEED = 0.01
#: Seconds a joint may take to arrive after its target has (the end of any ramp) before a move fails.
MOVE_TIMEOUT = 5.0
ON_ERROR = ("raise", "report")

_PROGRAM_MARK = "__roqsim_program__"


class MachineError(RuntimeError):
    """A machine refused something: an unknown program or argument, a start while busy, a model
    without the actuator it drives."""


class ProgramFailed(Exception):
    """Raised by :meth:`MachinePlugin.fail`: the program's own outcome was a failure."""


def program(fn: Callable) -> Callable:
    """Mark a generator method of a :class:`MachinePlugin` as one of its programs.

    Its parameters are the program's arguments, its docstring the program's documentation.
    """
    if not inspect.isgeneratorfunction(fn):
        raise TypeError(f"@program {fn.__qualname__}: a program is a generator (it yields)")
    setattr(fn, _PROGRAM_MARK, True)
    return fn


@dataclass(frozen=True)
class MachineStatus:
    """Where a machine's last start stands.

    Attributes:
        seq: the start this is about; 0 before any start, and after a reset
        program: its program
        finished: whether it ended, done or not
        error: empty when it ended done; else why not ("cancelled", a failure, an exception)
    """

    seq: int = 0
    program: str = ""
    finished: bool = False
    error: str = ""


class MachineHandle:
    """Published by a :class:`MachinePlugin` under ``machine:<name>``.

    ``start``, ``cancel``, ``status``, ``busy`` and ``ready`` are safe from any thread: a start is
    checked and stamped at the call and applied on the physics thread at the next step. ``value``
    reads the physics thread's state and belongs there.
    """

    def __init__(self, machine: MachinePlugin):
        self._machine = machine
        #: The machine's entity name.
        self.name: str = machine.address
        #: Its programs, by name.
        self.programs: tuple[str, ...] = tuple(sorted(type(machine).programs()))

    def start(self, program: str, **args) -> int:
        """Start ``program`` with ``args``; its sequence number. Raises :class:`MachineError` for an
        unknown program or argument, and while the machine is busy."""
        return self._machine.request_start(program, args)

    def status(self) -> MachineStatus:
        """The last start's :class:`MachineStatus`."""
        return self._machine.status_now

    @property
    def busy(self) -> bool:
        """A start is pending or a program runs."""
        return self._machine.busy

    @property
    def ready(self) -> bool:
        """Not busy, and the machine's own precondition holds (:meth:`MachinePlugin.ready_when`)."""
        return not self._machine.busy and bool(self._machine.ready_when())

    def cancel(self) -> int:
        """End the running (or pending) program where it is; its sequence number, 0 if none."""
        return self._machine.request_cancel()

    def value(self, name: str) -> float:
        """The current target of an actuator or setpoint (the end of any ramp is not it)."""
        return self._machine.value(name)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MachineHandle({self.name!r}, programs={list(self.programs)})"


@dataclass
class _Actuator:
    name: str
    aid: int
    qadr: int
    dadr: int
    lo: float
    hi: float
    limited: bool


@dataclass
class _Run:
    seq: int
    program: str
    gen: Iterator


class MachinePlugin(Plugin):
    """Base of a machine's plugin: actuators, programs, waits, events and the standard handle.

    One entity's own procedures -- see ``docs/architecture.rst``, "Machines and scenarios", for the
    line between a machine and a scenario, and :mod:`roqsim.machine` for the contract.

    A subclass declares :attr:`ACTUATORS`, :attr:`SETPOINTS`, :attr:`HOME` and its programs; it calls
    ``super()`` in each hook it overrides (``validate_config``, ``configure``, ``on_reset``,
    ``pre_step``, ``post_step``). A subclass that builds its own model does so in ``build``; one that
    controls a model a ``spawn_model`` placed has no ``build`` and resolves its actuators under its
    owner's prefix (:attr:`machine_prefix`).

    Config (every machine; a subclass adds its own keys)::

        home: {ram: 0.054}   # where an actuator or setpoint stands at the start of an episode and
                             #   after a reset, over the class's HOME; inside the actuator's range
        on_error: raise      # a program that raises: raise (stop the run, naming it) | report
                             #   (its error in status(), the caller decides)
    """

    provides_entity = True

    #: The model's position actuators the programs drive, by name after :attr:`machine_prefix`.
    ACTUATORS: tuple[str, ...] = ()
    #: Plugin-owned quantities that ramp like actuators, by name -> unit; the plugin applies them.
    SETPOINTS: dict[str, str] = {}
    #: Where the actuators and setpoints stand at the start of an episode; a world's ``home:``
    #: overrides it per name. An actuator named in neither stands at 0, clamped into its range.
    HOME: dict[str, float] = {}
    #: Blackboard handles the machine needs, attribute -> key (``{address}`` is its own address);
    #: resolved at every reset, after every plugin configured, and refused when one is missing.
    NEEDS: dict[str, str] = {}
    #: The kind of the entity a machine registers when it registers one itself.
    ENTITY_KIND = "machine"
    #: How near and how still a joint is when it has reached and stopped on its target.
    REACHED: float = REACHED
    STOPPED_SPEED: float = STOPPED_SPEED
    #: Seconds a joint may take to arrive after its target has, before ``move``, ``until_reached``
    #: and ``until_stopped`` fail the program; each also takes its own ``timeout``.
    MOVE_TIMEOUT: float = MOVE_TIMEOUT

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        #: False for an instance whose config makes it no machine: no handle, no endpoints.
        self.is_machine = True
        self.home: dict[str, float] = {}
        self.on_error = str(self.config.get("on_error", "raise"))
        self._ctx: SimContext | None = None
        self._actuators: dict[str, _Actuator] = {}
        #: name -> current target of every actuator and setpoint
        self._cmd: dict[str, float] = {}
        #: name -> (goal, speed): a target ramping toward its goal
        self._ramps: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._pending: tuple[int, str, dict] | None = None
        self._cancelling = 0
        self._run: _Run | None = None
        self._status = MachineStatus()
        self._raise: BaseException | None = None
        self.handle: MachineHandle | None = None

    # -- declaration --------------------------------------------------------------------------
    @classmethod
    def programs(cls) -> dict[str, Callable]:
        """Every program of the class, by name."""
        found: dict[str, Callable] = {}
        for klass in reversed(cls.__mro__):
            for attr, value in vars(klass).items():
                if getattr(value, _PROGRAM_MARK, False):
                    found[attr] = value
                elif attr in found:
                    del found[attr]
        return found

    @classmethod
    def describe_programs(cls) -> list[dict]:
        """The programs as ``roqsim plugins describe`` publishes them: name, doc, parameters."""
        out = []
        for name, fn in sorted(cls.programs().items()):
            doc = inspect.getdoc(fn) or ""
            params = []
            for p in list(inspect.signature(fn).parameters.values())[1:]:
                row = {"name": p.name}
                if p.annotation is not inspect.Parameter.empty:
                    row["type"] = (
                        p.annotation
                        if isinstance(p.annotation, str)
                        else getattr(p.annotation, "__name__", str(p.annotation))
                    )
                if p.default is not inspect.Parameter.empty:
                    row["default"] = p.default
                params.append(row)
            out.append({"name": name, "doc": doc.split("\n")[0], "parameters": params})
        return out

    @property
    def machine_prefix(self) -> str:
        """The MJCF name prefix :attr:`ACTUATORS` are resolved under: the ``prefix`` key, else the
        owning entity's prefix for a controller a manifest put under a spawned model."""
        if "prefix" in self.config:
            return str(self.config["prefix"])
        if self.entity and self._ctx is not None:
            owner = self._ctx.entities.get(self.entity)
            if owner is not None and isinstance(owner.meta, dict):
                return str(owner.meta.get("prefix", ""))
        return ""

    @property
    def endpoint_owner(self) -> str:
        """The machine's entity: its own address, or the spawned entity it controls."""
        return self.machine_entity

    @property
    def machine_entity(self) -> str:
        """The entity whose machine this is: the address of the entity this plugin registers, or,
        for a controller with no model of its own, the entity it is nested under."""
        return self.address if self.registers_entity else (self.entity or self.address)

    #: Whether the subclass registers its own entity (a machine that builds its model); a controller
    #: under a spawned model sets this False and speaks for its owner.
    registers_entity = True

    # -- config ------------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        """The keys every machine takes: ``home`` names what it drives, ``on_error`` is known."""
        errors = []
        what = f"machine {self.address}"
        home = config.get("home")
        if home is not None:
            if not isinstance(home, dict):
                errors.append(f"{what}: 'home' maps an actuator or setpoint to its start value")
            else:
                driven = set(self.ACTUATORS) | set(self.SETPOINTS)
                for key, value in home.items():
                    if key not in driven:
                        errors.append(
                            f"{what}: home names {key!r}, which it does not drive (it drives "
                            f"{sorted(driven) or 'nothing'})"
                        )
                    elif isinstance(value, bool) or not isinstance(value, (int, float)):
                        errors.append(f"{what}: home {key!r} is a number, not {value!r}")
        if str(config.get("on_error", "raise")) not in ON_ERROR:
            errors.append(f"{what}: 'on_error' is one of {list(ON_ERROR)}")
        return errors

    # -- configure ----------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        """Resolve the actuators and ``home``, register the entity (if :attr:`registers_entity`)
        and publish the handle. A subclass that registers its entity with its own meta calls
        :meth:`register_entity` before ``super().configure``."""
        self._ctx = ctx
        if not self.is_machine:
            return
        if self.registers_entity and ctx.entities.get(self.address) is None:
            raise MachineError(
                f"machine {self.address}: register its entity before MachinePlugin.configure "
                "(self.register_entity)"
            )
        self._resolve_actuators(ctx)
        self._resolve_home(ctx)
        self.handle = MachineHandle(self)
        ctx.blackboard.set(f"machine:{self.machine_entity}", self.handle)

    def register_entity(
        self, ctx: SimContext, body: str, kind: str | None = None, **meta
    ) -> Entity:
        """Register the machine's entity, with its prefix in the meta as presence and looks expect."""
        meta.setdefault("prefix", self.config.get("prefix", ""))
        entity = Entity(name=self.address, kind=kind or self.ENTITY_KIND, body=body, meta=meta)
        ctx.entities.add(entity)
        return entity

    def _resolve_actuators(self, ctx: SimContext) -> None:
        m = ctx.model
        prefix = self.machine_prefix
        for name in self.ACTUATORS:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
            if aid < 0:
                own = [
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    for i in range(m.nu)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(
                        prefix
                    )
                ]
                raise MachineError(
                    f"machine {self.address}: it drives {name!r} ({prefix + name!r}), which the "
                    f"model lacks (its actuators: {', '.join(own) or 'none'})"
                )
            if int(m.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise MachineError(
                    f"machine {self.address}: actuator {prefix + name!r} does not drive a joint, so "
                    "a ramp or a wait cannot read where it is"
                )
            jid = int(m.actuator_trnid[aid, 0])
            if int(m.jnt_type[jid]) not in (
                int(mujoco.mjtJoint.mjJNT_SLIDE),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            ):
                raise MachineError(
                    f"machine {self.address}: actuator {prefix + name!r} drives a joint that is "
                    "neither a slide nor a hinge"
                )
            servo = _not_a_position_servo(m, aid)
            if servo:
                raise MachineError(
                    f"machine {self.address}: actuator {prefix + name!r} is no plain position "
                    f"servo ({servo}), so its control is not the joint's target"
                )
            lo, hi = (float(v) for v in m.actuator_ctrlrange[aid])
            self._actuators[name] = _Actuator(
                name=name,
                aid=aid,
                qadr=int(m.jnt_qposadr[jid]),
                dadr=int(m.jnt_dofadr[jid]),
                lo=lo,
                hi=hi,
                limited=bool(m.actuator_ctrllimited[aid]),
            )

    def _resolve_home(self, ctx: SimContext) -> None:
        home = {**self.HOME, **{k: float(v) for k, v in (self.config.get("home") or {}).items()}}
        for name, value in home.items():
            if name not in self._actuators and name not in self.SETPOINTS:
                raise MachineError(
                    f"machine {self.address}: home names {name!r}, which it does not drive (it "
                    f"drives {sorted([*self._actuators, *self.SETPOINTS]) or 'nothing'})"
                )
            act = self._actuators.get(name)
            if act is not None and act.limited and not act.lo <= value <= act.hi:
                raise MachineError(
                    f"machine {self.address}: home {name!r} is {value}, outside its range "
                    f"[{act.lo}, {act.hi}]"
                )
        for name, act in self._actuators.items():
            if name not in home:
                home[name] = min(max(0.0, act.lo), act.hi) if act.limited else 0.0
        for name in self.SETPOINTS:
            home.setdefault(name, 0.0)
        self.home = home

    # -- reset and steps ----------------------------------------------------------------------
    def on_reset(self, ctx: SimContext) -> None:
        """Drop any program, put every actuator and setpoint at ``home``, resolve :attr:`NEEDS`."""
        if not self.is_machine:
            return
        with self._lock:
            self._pending = None
            self._cancelling = 0
            self._run = None
            self._status = MachineStatus()
        self._raise = None
        self._ramps.clear()
        d = ctx.data
        for name, value in self.home.items():
            self._cmd[name] = value
            act = self._actuators.get(name)
            if act is not None:
                d.ctrl[act.aid] = value
                d.qpos[act.qadr] = value
                d.qvel[act.dadr] = 0.0
        for attr, key in self.NEEDS.items():
            setattr(self, attr, self.require_handle(key))

    def require_handle(self, key: str):
        """The blackboard entry ``key`` (``{address}`` filled in), refused naming the machine when
        nothing published it."""
        key = key.format(address=self.address, entity=self.machine_entity)
        value = self._ctx.blackboard.get(key)
        if value is None:
            raise MachineError(
                f"machine {self.address}: needs {key!r}, which no entry of the world published"
            )
        return value

    def pre_step(self, ctx: SimContext) -> None:
        """Advance the ramps and write the actuators' targets. A subclass applies its setpoints
        after ``super().pre_step(ctx)``."""
        if self._raise is not None:
            exc, self._raise = self._raise, None
            raise exc
        if not self.is_machine:
            return
        dt = float(ctx.model.opt.timestep)
        for name, (goal, speed) in list(self._ramps.items()):
            now = self._cmd[name]
            delta = goal - now
            if abs(delta) <= speed * dt:
                self._cmd[name] = goal
                del self._ramps[name]
            else:
                self._cmd[name] = now + (speed * dt if delta > 0 else -speed * dt)
        if ctx.manual_control:
            return
        for name, act in self._actuators.items():
            ctx.data.ctrl[act.aid] = self._cmd[name]

    def post_step(self, ctx: SimContext) -> None:
        """Advance the running program by one step."""
        run = self._run
        if run is not None and run.seq != self._cancelling:
            self._advance(run)

    # -- the program's vocabulary -------------------------------------------------------------
    def set(self, name: str, value: float) -> None:
        """Set ``name``'s target to ``value`` at once; any ramp on it ends."""
        self._check_driven(name)
        value = self._in_range(name, float(value))
        self._ramps.pop(name, None)
        self._cmd[name] = value

    def ramp(self, name: str, value: float, speed: float) -> None:
        """Move ``name``'s target toward ``value`` at ``speed`` (its unit per second)."""
        self._check_driven(name)
        if not speed > 0:
            raise ValueError(f"machine {self.address}: a ramp's speed is positive, not {speed!r}")
        goal = self._in_range(name, float(value))
        if goal == self._cmd[name]:
            self._ramps.pop(name, None)
        else:
            self._ramps[name] = (goal, float(speed))

    def move(
        self,
        name: str,
        value: float,
        speed: float | None = None,
        event: str | None = None,
        timeout: float | None = None,
        **fields,
    ):
        """``set`` or ``ramp``, then wait until ``name`` has reached ``value`` (``until_reached``,
        with its ``timeout``). With ``event``, emit that step's event as the move begins."""
        if speed is None:
            self.set(name, value)
        else:
            self.ramp(name, value, speed)
        if event:
            self.step(event, **fields)
        yield from self.until_reached(name, timeout=timeout)

    def until(self, cond: Callable[[], bool], timeout: float | None = None, what: str = ""):
        """Yield until ``cond()`` holds; past ``timeout`` seconds the program fails."""
        start = self._ctx.sim_time
        while not cond():
            if timeout is not None and self._ctx.sim_time - start > timeout:
                self.fail(
                    f"machine {self.address}: {what or 'the condition'} not true after {timeout:g} s"
                )
            yield

    def until_reached(self, *names: str, timeout: float | None = None):
        """Yield until each of ``names`` is at the end of its ramp (a joint within :attr:`REACHED`).

        A joint that has not arrived ``timeout`` seconds (default :attr:`MOVE_TIMEOUT`) after its
        target did -- an obstruction -- fails the program, naming the machine and the actuator."""
        yield from self._until_arrived(names, False, timeout)

    def until_stopped(self, *names: str, timeout: float | None = None):
        """Yield until each of ``names`` has reached its target and is at rest there; ``timeout`` as
        :meth:`until_reached`."""
        yield from self._until_arrived(names, True, timeout)

    def _until_arrived(self, names, still: bool, timeout: float | None):
        limit = self.MOVE_TIMEOUT if timeout is None else float(timeout)
        arrived_target: dict[str, float] = {}  # name -> when its target arrived (its ramp ended)
        while True:
            now = self._ctx.sim_time
            waiting = [n for n in names if not self.reached(n, still=still)]
            if not waiting:
                return
            for name in waiting:
                if name in self._ramps:
                    arrived_target.pop(name, None)
                    continue
                since = arrived_target.setdefault(name, now)
                if now - since > limit:
                    where = (
                        f" (it is at {self.position(name):g})" if name in self._actuators else ""
                    )
                    self.fail(
                        f"machine {self.address}: {name!r} did not "
                        f"{'stop on' if still else 'reach'} {self._cmd[name]:g} within {limit:g} s "
                        f"of its target{where}"
                    )
            yield

    def wait(self, seconds: float):
        """Yield for ``seconds`` of simulated time."""
        end = self._ctx.sim_time + float(seconds)
        while self._ctx.sim_time < end - 1e-9:
            yield

    def step(self, name: str, **fields) -> dict:
        """Emit the event of a named step of the running program."""
        run = self._run
        return self._ctx.events.emit(
            self.machine_entity,
            name,
            program=run.program if run else "",
            seq=run.seq if run else 0,
            **fields,
        )

    def fail(self, reason: str) -> None:
        """End the program here, with ``reason`` as its status error. Does not return."""
        raise ProgramFailed(reason)

    def value(self, name: str) -> float:
        """The current target of an actuator or setpoint."""
        self._check_driven(name)
        return float(self._cmd[name])

    def position(self, name: str) -> float:
        """Where an actuator's joint is now."""
        return float(self._ctx.data.qpos[self._actuators[name].qadr])

    def reached(self, name: str, still: bool = False) -> bool:
        """Whether ``name`` has arrived where it was sent (and, with ``still``, is at rest)."""
        self._check_driven(name)
        if name in self._ramps:
            return False
        act = self._actuators.get(name)
        if act is None:
            return True  # a setpoint is where its target is
        d = self._ctx.data
        near = abs(float(d.qpos[act.qadr]) - self._cmd[name]) <= self.REACHED
        return near and (not still or abs(float(d.qvel[act.dadr])) <= self.STOPPED_SPEED)

    def at_home(self, *names: str) -> bool:
        """Whether each of ``names`` (all, by default) stands at its home and is at rest."""
        for name in names or tuple(self.home):
            if abs(self._cmd[name] - self.home[name]) > 1e-9 or not self.reached(name, still=True):
                return False
        return True

    def ready_when(self) -> bool:
        """The machine's own precondition for a start, beyond not being busy. Default: none."""
        return True

    def _check_driven(self, name: str) -> None:
        if name not in self._actuators and name not in self.SETPOINTS:
            raise MachineError(
                f"machine {self.address}: it drives no {name!r} (it drives "
                f"{sorted([*self._actuators, *self.SETPOINTS]) or 'nothing'})"
            )

    def _in_range(self, name: str, value: float) -> float:
        """``value`` as a target of ``name``; outside the actuator's range the program fails."""
        act = self._actuators.get(name)
        if act is not None and act.limited and not act.lo <= value <= act.hi:
            self.fail(
                f"machine {self.address}: target {value:g} for {name!r} is outside its range "
                f"[{act.lo:g}, {act.hi:g}]"
            )
        return value

    # -- starts, cancels, status --------------------------------------------------------------
    @property
    def busy(self) -> bool:
        run = self._run
        return self._pending is not None or (run is not None and run.seq != self._cancelling)

    @property
    def status_now(self) -> MachineStatus:
        return self._status

    def bind(self, program_name: str, args: dict) -> Callable[[], Iterator]:
        """The program ``program_name`` bound to ``args``, refused naming what is wrong."""
        programs = type(self).programs()
        fn = programs.get(program_name)
        if fn is None:
            raise MachineError(
                f"machine {self.address}: no program {program_name!r}; it has "
                f"{', '.join(sorted(programs)) or 'none'}"
            )
        sig = inspect.signature(fn)
        try:
            bound = sig.bind(self, **args)
        except TypeError as exc:
            takes = [p for p in list(sig.parameters)[1:]]
            raise MachineError(
                f"machine {self.address}: program {program_name!r}: {exc} (it takes "
                f"{', '.join(takes) or 'no arguments'})"
            ) from None
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            annotation = sig.parameters[name].annotation
            if not _fits(annotation, value):
                raise MachineError(
                    f"machine {self.address}: program {program_name!r}: {name} is "
                    f"{annotation if isinstance(annotation, str) else annotation.__name__}, "
                    f"not {value!r}"
                )
        return lambda: fn(*bound.args, **bound.kwargs)

    def request_start(self, program_name: str, args: dict) -> int:
        """Check and stamp a start, and post it to the physics thread. Any thread."""
        if not self.is_machine:
            raise MachineError(f"machine {self.address}: this one has nothing to start")
        args = dict(args or {})
        self.bind(program_name, args)
        with self._lock:
            if self.busy:
                running = self._pending[1] if self._pending else self._run.program
                seq = self._pending[0] if self._pending else self._run.seq
                raise MachineError(
                    f"machine {self.address}: busy running {running!r} (seq {seq}); start "
                    f"{program_name!r} once it is ready"
                )
            self._seq += 1
            seq = self._seq
            self._pending = (seq, program_name, args)
        self._ctx.post(lambda _ctx, seq=seq: self._apply_start(seq))
        return seq

    def request_cancel(self) -> int:
        """Cancel the pending or running program; its sequence number, 0 if none. Any thread."""
        with self._lock:
            if self._pending is not None:
                seq, name, _args = self._pending
                self._pending = None
                self._status = MachineStatus(seq, name, True, "cancelled")
                return seq
            run = self._run
            if run is None or run.seq == self._cancelling:
                return 0
            self._cancelling = run.seq
        self._ctx.post(lambda _ctx, seq=run.seq: self._apply_cancel(seq))
        return run.seq

    def preempt(self, program_name: str, **args) -> int:
        """Cancel whatever runs and start ``program_name`` now. Physics thread only.

        For the machine's own transport endpoints, where a new command means "this is the target
        now" (a door's ``cmd``); the handle's ``start`` still refuses while busy."""
        self.bind(program_name, args)
        with self._lock:
            pending, self._pending = self._pending, None
            if pending is not None:
                self._status = MachineStatus(pending[0], pending[1], True, "cancelled")
            run = self._run
        if run is not None:
            self._finish(run, "cancelled")
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._pending = (seq, program_name, dict(args))
        self._apply_start(seq)
        return seq

    def _apply_start(self, seq: int) -> None:
        with self._lock:
            if self._pending is None or self._pending[0] != seq:
                return  # cancelled before it ran, or dropped by a reset
            _seq, name, args = self._pending
            self._pending = None
            run = _Run(seq=seq, program=name, gen=self.bind(name, args)())
            self._run = run
            self._status = MachineStatus(seq, name, False, "")
        self._ctx.events.emit(
            self.machine_entity, "program_start", program=name, seq=seq, args=args
        )
        try:
            self._advance(run)
        except Exception as exc:  # noqa: BLE001 - a posted command's error would only be logged
            self._raise = exc

    def _apply_cancel(self, seq: int) -> None:
        run = self._run
        if run is not None and run.seq == seq:
            self._finish(run, "cancelled")

    def _advance(self, run: _Run) -> None:
        try:
            next(run.gen)
        except StopIteration:
            self._finish(run, "")
        except ProgramFailed as exc:
            self._finish(run, str(exc) or "failed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._finish(run, error)
            if self.on_error == "raise":
                raise RuntimeError(
                    f"machine {self.address}: program {run.program!r} (seq {run.seq}) raised: {exc}"
                ) from exc

    def _finish(self, run: _Run, error: str) -> None:
        with self._lock:
            if self._run is run:
                self._run = None
            if self._cancelling == run.seq:
                self._cancelling = 0
            self._status = MachineStatus(run.seq, run.program, True, error)
        if error:
            # Cancelled or failed, the machine holds where it is: a ramp's target stops moving.
            run.gen.close()
            self._ramps.clear()
        self._ctx.events.emit(
            self.machine_entity, "program_end", program=run.program, seq=run.seq, error=error
        )

    # -- endpoints ----------------------------------------------------------------------------
    @endpoint.command(when="is_machine", ros2=None)
    def start(self, program: str, args: dict | None = None) -> int:
        """Start one of the machine's programs; its sequence number.

        Args:
            program: the program's name
            args: the program's arguments by name
        """
        return self.request_start(program, args or {})

    @endpoint.out(rate=10.0, when="is_machine", ros2=None)
    def status(self) -> MachineStatus:
        """Where the machine's last start stands."""
        return self._status

    @endpoint.command(when="is_machine", ros2=None)
    def cancel(self) -> int:
        """End the running program where it is; its sequence number, 0 if none."""
        return self.request_cancel()


def _not_a_position_servo(m: mujoco.MjModel, aid: int) -> str:
    """Why actuator ``aid`` is no plain position servo -- its control the joint's target -- or ``""``.

    A plain servo has a fixed gain, an affine bias whose position term is the gain's negative (so the
    joint settles where the control says) and a unit gear; a motor, a velocity servo or a geared
    servo would let a program read its control as a position it is not."""
    if int(m.actuator_gaintype[aid]) != int(mujoco.mjtGain.mjGAIN_FIXED):
        return "its gain is not fixed"
    if int(m.actuator_biastype[aid]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
        return "it has no affine bias: a motor or a velocity servo"
    kp = float(m.actuator_gainprm[aid, 0])
    if kp <= 0 or abs(float(m.actuator_biasprm[aid, 1]) + kp) > 1e-9 * max(1.0, kp):
        return "its bias stiffness does not match its gain: a velocity servo or a custom law"
    if abs(float(m.actuator_biasprm[aid, 0])) > 1e-12:
        return "its bias has an offset"
    gear = [float(g) for g in m.actuator_gear[aid]]
    if gear[0] != 1.0 or any(gear[1:]):
        return f"its gear is {gear[0]:g}, not 1"
    return ""


def _fits(annotation, value) -> bool:
    """Whether ``value`` passes for a program parameter's annotation (plain types only)."""
    if annotation is inspect.Parameter.empty:
        return True
    kinds = {"float": float, "int": int, "str": str, "bool": bool}
    if isinstance(annotation, str):
        annotation = kinds.get(annotation.split("|")[0].strip())
        if annotation is None:
            return True
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation in (str, bool):
        return isinstance(value, annotation)
    return True
