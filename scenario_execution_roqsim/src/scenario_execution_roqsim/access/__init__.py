# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""How an action reaches the world: one seam, two transports, the same names.

An roqsim simulation is driven two ways, and a scenario action must work in both:

* **stepped, in-process** -- scenario-execution's own runner owns the loop, the simulator shares its
  process, and the action is handed the adapter in ``setup(**kwargs)`` as ``simulation``.
* **over the control socket** -- the simulator is another process (``roqsim sim``, under the ROS
  runner or on its own), and every endpoint it has is reached over the socket it serves
  (:mod:`roqsim.control_client`).

Writing an action twice would be the obvious answer and the wrong one: the *semantics* are identical,
only the plumbing differs. So the plumbing is the abstraction, and the actions are written once
against it.

It works out cleanly because both transports speak the same vocabulary -- the world's endpoints
(architecture.rst §13), addressed by the entity that owns one and its name:

===========================  ==================================  ====================================
need                         in-process                          over the control socket
===========================  ==================================  ====================================
pose of entity ``X``         ``sim/entities/X/pose``, read        the same endpoint, over the socket
call command ``C`` of ``X``  ``ctx.interface``, its ``write``     ``call``, which waits for it
did it land                  its confirming endpoint, read       the same, in the reply
                             after the step that applied it
report ``R`` of ``X``        ``ctx.interface.find(X, R)``        ``read`` of the same endpoint,
                             ``.read()``, any field              any field
place / spawn / delete       :mod:`roqsim.entity_control`         ``sim/entities/set_state`` and
                                                                 ``set_presence``: the same code
drive ``X``'s navigator      its handle                          its route endpoints
time                         the runner's ``clock``               the runner's ``clock``
===========================  ==================================  ====================================

**One refusal, one text.** A refusal comes from the producer -- a plugin's command raising, a
placement :mod:`roqsim.entity_control` refuses -- or from the resolution in this module, which
builds its messages from the same list of endpoints on both routes. So a scenario reads the same
message whichever shape it ran in.

**Time is not asked of the transport.** ``Clock.now()`` is already the framework's abstraction:
``SimulationClock`` under the stepped runner, ``RosClock`` (i.e. ``/clock``) under the ROS one. An
action takes the clock it is handed and never knows which.

What DOES differ, and is stated rather than hidden: over the socket a read is a round-trip, so the
instant a threshold is crossed is resolved at the tick period rather than at the physics step. A
dwell shorter than one tick means "the first tick past the threshold" on both paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class AccessError(RuntimeError):
    """The world cannot answer: no transport, an unknown entity, a missing plugin instance.

    Always an AUTHORING error -- a name that does not exist, a world without the plugin the scenario
    fires -- so an action turns it into ``ActionError``. A runtime verdict (the fault did not land) is
    not one of these; that is a result, and results are returned, not raised.
    """


class EntityAbsent(AccessError):
    """The entity exists and is absent (deleted, or not spawned yet): it has no pose now, and may.

    A fact about the trial rather than about the scenario, so a condition waits for the entity to
    be present; any other caller lets it reach the scenario as an AccessError.
    """


@dataclass(frozen=True)
class Pose:
    """A body's world pose. ``quat`` is ``(w, x, y, z)``.

    MuJoCo's ``xquat`` order, which is also the order the bridge fills
    ``geometry_msgs/Quaternion`` in (``sim_interfaces._get_entity_state``), so the two transports
    hand back the same numbers in the same order and the caller never asks which it is talking to.
    ``movable`` is false for a body welded to the world, whose pose never changes.
    """

    pos: np.ndarray
    quat: np.ndarray
    movable: bool = True


@dataclass(frozen=True)
class CommandOutcome:
    """What became of a command.

    ``ok`` is about the TRANSPORT and the simulator: the command ran and returned ``result``, or
    ``detail`` says why not -- the producer's own refusal, or no outcome in time. ``confirmation``
    is the value of the endpoint that confirms the command, read after the step that applied it,
    and ``verified`` its verdict where it has one (``landed`` / ``no_effect`` / ``untested``);
    ``confirmed`` is false when a confirmation was due and could not be read (a paused run). They
    are separate because "the simulator never applied it" and "it applied and changed nothing" call
    for different messages, and only the caller knows whether either should fail the trial.
    """

    ok: bool
    detail: str = ""
    result: object = None
    confirmation: object = None
    verified: str = ""
    confirmed: bool = True


class PendingCall(ABC):
    """Shared by every in-flight call: why it is still waiting, when it can say.

    ``poll()`` returning ``None`` means "not yet", and "not yet" has two very different causes: the
    write is queued and will drain next step, or the thing that would answer does not exist. The
    second is indistinguishable from the first until a timeout fires, at which point the trial has
    spent its whole budget and reports only that it ran out -- which is what a world that served no
    control plane looked like.
    """

    @abstractmethod
    def poll(self):
        """The outcome, or ``None`` while it is not yet known."""

    def pending_reason(self) -> str | None:
        """A phrase naming what is missing, or ``None`` when waiting is simply progress."""
        return None


class CommandCall(PendingCall):
    """A command in flight. ``poll()`` returns ``None`` until the outcome is known.

    Two-phase on both transports: a command is a request with an outcome, and its confirmation is
    read after the step that applied it. Neither may block -- an action that blocks the tick either
    stalls the tree or, in the stepped shape, deadlocks the very step it is waiting for.
    """

    @abstractmethod
    def poll(self) -> CommandOutcome | None: ...


@dataclass(frozen=True)
class TeleportOutcome:
    """What became of a teleport. ``ok`` is false only for an authoring-adjacent runtime fact that
    is still a result rather than a raise -- the named entity has no free joint to place (e.g. a
    static prop), which :meth:`WorldAccess.set_entity_state` reports here rather than as
    :class:`AccessError`, because "this entity cannot be teleported" is a fact about the WORLD a
    campaign chose, not about the call being malformed.
    """

    ok: bool
    detail: str


class TeleportCall(PendingCall):
    """A pose write in flight. ``poll()`` returns ``None`` until the outcome is known.

    Two-phase for the same reason :class:`CommandCall` is: in-process the wait is for ``ctx.post``
    to be drained by the next ``pre_step``; over the socket it is for the reply.
    """

    @abstractmethod
    def poll(self) -> TeleportOutcome | None: ...


@dataclass(frozen=True)
class SpawnOutcome:
    """What became of a spawn. ``ok`` is false for a runtime fact rather than a raise, on the same
    terms as :class:`TeleportOutcome`: the world compiled no such entity to activate, or it has no
    free joint and the pose asked for is not the one it is welded at. Both are facts about the
    WORLD a campaign chose.
    """

    ok: bool
    detail: str


class SpawnCall(PendingCall):
    """A presence flip in flight. ``poll()`` returns ``None`` until the outcome is known."""

    @abstractmethod
    def poll(self) -> SpawnOutcome | None: ...


@dataclass(frozen=True)
class NavOutcome:
    """What became of a navigation route.

    ``ok`` false is a trial fact, not an authoring one: the route was preempted by a newer goal, or
    the navigator gave up on it. An entity that has no navigator at all raises instead -- that is a
    world that cannot answer, which is the distinction :class:`AccessError` exists to keep.
    """

    ok: bool
    detail: str = ""


class NavCall(PendingCall):
    """A route in flight. ``poll()`` returns ``None`` until the outcome is known.

    Keyed on the navigator's **sequence number**, not on a bare "finished" flag, and that is the
    whole reason this is a call object rather than a boolean read. A navigator that has completed
    whatever it was doing before is *already* finished when a new route is queued, so a caller
    watching the flag would report an arrival that had not happened. The sequence says whose arrival
    it is: equal and finished means yours; larger means something preempted you.
    """

    @abstractmethod
    def poll(self) -> NavOutcome | None: ...

    def cancel(self) -> None:
        """Stop the mover. Idempotent -- an action's ``request_cancel`` may fire more than once."""


@dataclass(frozen=True)
class ReportReading:
    """The value a plugin's report holds now, and which field of it that is.

    ``value`` is a single number, flag or string. ``field`` is the field that was read: the one the
    scenario named, or for a bare report the one its endpoint publishes (``""`` where the
    publication is the whole payload, a plain number or flag). ``source`` says where it was read
    from, for a message.
    """

    value: object
    field: str
    source: str


class ReportCall(PendingCall):
    """A report being watched. ``poll()`` returns the current :class:`ReportReading`, or ``None``
    while no value is known yet: the producer has none, the field holds none, or over the socket
    the reply has not arrived. Each poll after a reading starts the next one."""

    @abstractmethod
    def poll(self) -> ReportReading | None: ...


def plain(value):
    """A report value as a plain Python value: a NumPy scalar becomes its Python scalar and an
    array a list, so a scenario variable holds the same value whichever transport read it."""
    if isinstance(value, np.generic):
        return value.item()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def published_field(backend: dict) -> str:
    """The field of a report its ROS publication carries (the ``field`` hint), ``""`` for all of it.

    What a bare ``value: '<endpoint>'`` means on both transports, so the short form reads the
    same value whichever way the world is reached.
    """
    return str((backend.get("ros2") or {}).get("field") or "")


def parse_value(text: str):
    """A scenario's ``value`` string as a payload: JSON where it parses, else the string itself.

    ``'true'`` is ``True``, ``'{"vx": 0.5}'`` a mapping, ``'fast'`` the word.
    """
    import json

    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


# -- resolving what a scenario names ----------------------------------------------------------------
# Both transports resolve against the same rows -- one per endpoint: its path, owner, name and kind
# -- so an unknown name is refused with the same text whichever one a scenario runs over.


def find_endpoint(rows: list[dict], entity: str, endpoint: str, *, kind: str) -> dict:
    """The row *entity*'s *endpoint* names, of *kind* (``out``, or ``in`` for a command or stream).

    Addressed as the world names it: the entity that owns the endpoint and its name (``grip_fault``,
    ``override``), or a component address and a name (``robot.lidar``, ``override``) -- the path
    ``robot/lidar/override`` -- where one entity owns two endpoints of that name. Resolved by
    :func:`roqsim.paths.resolve`, the grammar frames share.
    """
    from roqsim.paths import Offer, PathError, address_path, resolve

    direction = "out" if kind == "out" else "in"
    offers = [
        Offer(
            component=row["path"][: -len(row["name"]) - 1] if row["path"] != row["name"] else "",
            name=row["name"],
            kind="out" if row["kind"] == "out" else "in",
            what=row["kind"],
            alias=f"{address_path(row['owner'])}/{row['name']}" if row["owner"] else "",
            target=row,
        )
        for row in rows
    ]
    try:
        return resolve(offers, f"{address_path(entity)}/{endpoint}", direction).target
    except PathError as err:
        unknown = err
        if err.reason == "ambiguous":
            paths = sorted(o.path for o in err.matches)
            raise AccessError(
                f"{entity!r} has {len(paths)} endpoints named {endpoint!r}: "
                f"{', '.join(paths)}. Name one by the address of the component that declares "
                f"it (entity 'robot.lidar' for robot/lidar/{endpoint})."
            ) from None
    what = "command or stream" if direction == "in" else "report"
    own = sorted(
        row["path"]
        for row in rows
        if (row["owner"] == entity or row["path"].startswith(address_path(entity) + "/"))
        and (row["kind"] == "out") == (direction == "out")
    )
    listed = ", ".join(own) if own else "(none)"
    plural = "commands and streams" if direction == "in" else "reports"
    near = f" Did you mean {unknown.suggestion!r}?" if unknown.suggestion else ""
    raise AccessError(
        f"no {what} {endpoint!r} of {entity!r}.{near} Its {plural}: {listed}. An endpoint is "
        "addressed by the entity that owns it (the world's `name:`) or the component that declares "
        "it, and its name."
    )


def no_report(rows: list[dict], entity: str, report: str, *, is_entity: bool) -> AccessError:
    """Why *entity* has no report *report*, listing the reports this world does publish."""
    offered: dict[str, list[str]] = {}
    for row in rows:
        if row["kind"] == "out":
            offered.setdefault(row["owner"], []).append(row["name"])
    if entity in offered:
        return AccessError(
            f"entity {entity!r} publishes no report {report!r}. It publishes: "
            f"{', '.join(sorted(offered[entity]))}."
        )
    what = (
        f"{entity!r} is an entity, but no plugin on it publishes a report"
        if is_entity
        else f"no entity {entity!r} publishes a report"
    )
    listed = "; ".join(
        f"{owner or '(no entity)'}: {', '.join(sorted(names))}"
        for owner, names in sorted(offered.items())
    )
    return AccessError(
        f"{what}. A report is an endpoint a plugin declares on the entity it watches, addressed by "
        f"that entity's `name:`. This world publishes: {listed or '(none)'}."
    )


#: What a report's value may be when neither the scenario nor the endpoint names a field.
_SCALARS = (bool, int, float, str, np.generic)


def _fields_of(payload) -> list[str]:
    """The named fields of a report, for a refusal that lists what can be asked for."""
    import dataclasses

    if isinstance(payload, dict):
        return [str(k) for k in payload]
    if dataclasses.is_dataclass(payload):
        return [f.name for f in dataclasses.fields(payload)]
    if hasattr(payload, "_fields"):  # a namedtuple
        return list(payload._fields)
    try:
        return sorted(k for k in vars(payload) if not k.startswith("_"))
    except TypeError:  # a tuple, an array, a number: no names to offer
        return []


def report_value(
    name: str, payload, field: str, published: str, source: str
) -> ReportReading | None:
    """Field *field* of a report (else the field its endpoint publishes), as a reading.

    One function for both transports: in-process *payload* is the producer's object, over the
    socket the mapping it arrived as, and a field is looked up the same way in either. A field
    holding ``None`` has no value yet (``None`` is returned); one holding a structure or a sequence
    is refused, naming it, since a condition compares one value.
    """
    field = field or published
    report = name.rpartition(".")[2]
    if not field:
        if isinstance(payload, _SCALARS):
            return ReportReading(plain(payload), "", source)
        fields = _fields_of(payload)
        raise AccessError(
            f"{name} publishes no single field (its endpoint names none for ROS), so name the "
            f"one to read: '{report}.<field>', with <field> one of: "
            f"{', '.join(fields) if fields else '(none -- a single value)'}."
        )
    if isinstance(payload, dict):
        found = field in payload
        value = payload.get(field)
    else:
        found = hasattr(payload, field)
        value = getattr(payload, field, None)
    if not found:
        fields = _fields_of(payload)
        raise AccessError(
            f"{name} has no field {field!r}. It has: "
            f"{', '.join(fields) if fields else '(no named fields)'}."
        )
    value = plain(value)
    if value is None:
        return None
    if not isinstance(value, (bool, int, float, str)):
        fields = _fields_of(payload)
        raise AccessError(
            f"{name}.{field} is not a single number, flag or string, so no condition can compare "
            f"it. Name a field of {name} that is one: "
            f"{', '.join(fields) if fields else '(no named fields)'}."
        )
    return ReportReading(value, field, source)


def no_navigator(name: str, offered: list[str]) -> AccessError:
    """Why *name* cannot be driven, naming what this world can drive."""
    return AccessError(
        f"entity {name!r} has no navigator, so nothing can drive it. A `navigator` component "
        f"must be nested under the entry that provides it (spawn_robot, spawn_model with "
        f"`motion: driven`, or walker). This world can navigate: "
        f"{', '.join(sorted(offered)) if offered else '(nothing)'}."
    )


#: Where the core serves an entity's ground-truth pose (:mod:`roqsim.entity_pose`).
POSE_PATH = "sim/entities/{name}/pose"


def posed_entities(rows: list[dict]) -> list[str]:
    """The entities the core serves a pose for, by their endpoint rows."""
    prefix, suffix = POSE_PATH.split("{name}")
    return sorted(
        row["path"][len(prefix) : -len(suffix)]
        for row in rows
        if row["path"].startswith(prefix) and row["path"].endswith(suffix)
    )


def no_entity(rows: list[dict], name: str) -> AccessError:
    """Why *name* has no pose: the closest entity names, or the ones there are."""
    import difflib

    known = posed_entities(rows)
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    hint = (
        f"Did you mean {', '.join(repr(c) for c in close)}?"
        if close
        else f"Known entities: {', '.join(known) or '(none)'}."
    )
    return AccessError(
        f"the simulator has no entity called {name!r} with a body, so it has no pose. The name is "
        f"the world's `name:` for that entity, not a body name and not a TF frame. {hint}"
    )


def pose_reading(name: str, payload) -> Pose:
    """A core pose endpoint's value as a :class:`Pose`; ``None`` is an absent entity."""
    if payload is None:
        raise EntityAbsent(
            f"entity {name!r} is absent (deleted, or not spawned yet): nothing can see or touch it, "
            "so it has no pose until it is spawned."
        )
    get = payload.get if isinstance(payload, dict) else lambda k: getattr(payload, k)
    return Pose(
        pos=np.asarray(get("position"), dtype=float),
        quat=np.asarray(get("orientation"), dtype=float),
        movable=bool(get("movable")),
    )


def immovable(name: str) -> AccessError:
    """Why a condition on *name*'s motion can never be met: it is welded to the world."""
    return AccessError(
        f"entity {name!r} is welded to the world: its pose never changes, so waiting for it to "
        "move or turn never ends. Give it a free joint (`motion: physics` on a spawn_model) or "
        "drive it (`motion: driven`), or name something that can move."
    )


class WorldAccess(ABC):
    """The seam. See the module docstring."""

    #: For messages, so a failure says which transport answered.
    transport: str = "unknown"

    @abstractmethod
    def ready(self) -> bool:
        """Can the world be asked anything yet?

        False in the stepped shape until the world is built -- the tree is set up before the first
        ``reset()``, and a caller must wait a tick rather than trigger a compile. False over the
        socket until the simulator has answered.
        """

    @abstractmethod
    def ground_truth_pose(self, name: str) -> Pose | None:
        """Entity *name*'s pose as the core's ``sim/entities/<name>/pose`` endpoint reads it.

        The same endpoint on both transports, so an entity welded to the world (a shelf) has a
        pose here too, and a name the core serves no pose for is refused with the same text
        (:func:`no_entity`) -- at once, since the world never had it. ``None`` while a reply is in
        flight; :class:`EntityAbsent` while the entity exists and is absent.
        """

    @abstractmethod
    def call_endpoint(self, entity: str, endpoint: str, value=None) -> CommandCall:
        """Write *value* to *entity*'s command or stream *endpoint*. Never blocks.

        Addressed by :func:`find_endpoint`. A command's outcome carries what it returned, and the
        value of the endpoint that confirms it where it names one; a stream's is known once the
        value is queued. *value* is the payload: for an endpoint declared with typed parameters, a
        mapping of their names.
        """

    @abstractmethod
    def navigate(self, name: str, goal_poses, *, wait: bool) -> NavCall:
        """Send ``name`` through ``goal_poses`` (world-frame ``(x, y, yaw)``). Never blocks.

        ``goal_poses`` must not be empty; running the route the entity was configured with is
        :meth:`start_route`, not a route with no poses.

        This drives the simulator's own mover. It is not ``osc.nav2``'s ``nav_to_pose``, which
        commands an external nav2 stack: that one is the subject of the experiment, this one is the
        apparatus around it.
        """

    @abstractmethod
    def start_route(self, name: str, *, wait: bool) -> NavCall:
        """Run the route ``name`` was configured with (``navigator: {goals: [...]}``). Never blocks.

        What lets a world own an opponent's trajectory -- identical in every repetition, and visible
        in a campaign's config diff -- while the scenario owns only its timing. An entity with no
        configured route raises :class:`AccessError`: there is nothing to run, and succeeding would
        read as an arrival.
        """

    @abstractmethod
    def set_entity_state(
        self, name: str, pos: np.ndarray, quat: np.ndarray, lin=None, ang=None
    ) -> TeleportCall:
        """Place a free-jointed entity at ``pos`` (metres) / ``quat`` (w, x, y, z), moving at
        ``lin``/``ang``. Never blocks.

        The state an entity is *in*, which is pose and velocity together -- the shape
        ``simulation_interfaces``' ``EntityState`` has, and the reason this is not called a
        teleport: the same call serves placing a robot at a per-configuration start pose (a
        per-RUN value a MuJoCo compile cannot vary) and handing a body a velocity it should be
        moving with.

        ``lin``/``ang`` default to **zero**, which is what placing something means: a body put
        somewhere is not still carrying the velocity it had. A caller that wants motion states it,
        rather than the state being half-settable. The placement is
        :func:`roqsim.entity_control.set_state` on both transports.
        """

    @abstractmethod
    def set_entity_presence(self, name: str, present: bool, pos=None, quat=None) -> SpawnCall:
        """Make an entity perceivable (or not), placing it as it appears. Never blocks.

        Presence is what a per-RUN start pose should go through, rather than a teleport: the pose is
        applied in the SAME transaction as the flip, so the entity is never perceivable at a pose
        nobody asked for. A teleport can only spawn-at-nominal-then-move, which is visible for a
        step and accelerates a free body under gravity in between.

        A pose is **required when making an entity present**, and refused when making it absent: a
        transport that quietly sent the origin would move the entity somewhere nobody named, and an
        absent entity keeps the pose it had, which is what lets it come back where it was.

        Making an entity present that already is (or absent that already is) is refused, as a
        result rather than a raise. The flip is :func:`roqsim.entity_control.set_presence` on both
        transports.
        """

    @abstractmethod
    def entity_report(self, entity: str, report: str, field: str = "") -> ReportCall:
        """Watch one value a plugin publishes: field *field* of *entity*'s endpoint *report*.

        Addressed as the world names it -- the entity that owns the endpoint and the endpoint's name
        (``'ur5e'``, ``'force_limit'``), never a topic. An empty *field* means the one the
        endpoint's ROS publication carries (``LimitReport.tripped`` for ``force_limit``), so the
        short form reads one value whichever way the world is reached. Every field of the report
        is readable on both transports.

        Raises :class:`AccessError`, from this call or from ``poll()``, for an entity, report or
        field that does not exist, listing what does, and for a field that is not a single number,
        flag or string.
        """

    def pending_reason(self) -> str | None:
        """Why :meth:`ready` is still false, when that is more than the world not being built yet."""
        return None

    def teardown(self) -> None:
        """Drop anything the transport allocated. Called from the action's ``shutdown``."""


def select(kwargs: dict, *, what: str) -> WorldAccess:
    """Pick the backend from the setup kwargs the RUNNER provided.

    Not from configuration: which transport is present is a property of how the scenario is being
    executed, and a scenario that had to declare it would have to be edited to move between the two.
    ``simulation`` is offered by the stepped runner and makes this in-process; otherwise the
    simulator is another process, reached over the control socket it serves -- found the way
    ``roqsim ls`` finds it (``ROQSIM_CONTROL``, or the only one running), once
    it answers, so a simulator that is still starting is waited for rather than refused.

    Imported lazily, per backend, so the in-process path never pays for MuJoCo at tree-build time
    and the socket path never imports ZeroMQ unless it is taken.
    """
    sim = kwargs.get("simulation")
    if sim is not None:
        from .in_process import InProcessAccess

        return InProcessAccess(sim)
    from .ipc import IpcAccess

    return IpcAccess()


def clock_of(kwargs: dict):
    """The runner's clock. ``clock`` on the stepped runner, ``sim_clock`` on the ROS one.

    Both are ``scenario_execution.simulation.Clock``, so ``now()`` is sim-time seconds on either --
    which is what makes a dwell mean the same thing as ``timeout()`` and as a recorded timestamp.
    Wall clock is never an option: under ``pacing: asap`` the two differ by orders of magnitude.
    """
    return kwargs.get("clock") or kwargs.get("sim_clock")
