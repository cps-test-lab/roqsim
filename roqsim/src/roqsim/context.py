"""Shared per-run state passed to every plugin hook.

:class:`SimContext` is the single object plugins use to cooperate. It exposes the MuJoCo model/data,
config, a typed :class:`Blackboard`, an :class:`EntityRegistry`, the thread-safe command queue
(``post``/``submit``/``drain_commands``) with its :class:`CommandFuture` and the latest-value
:class:`StreamSlot`, and the (currently inert) step-gate API used by the foreseen synchronous mode.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .seed import SeedError

if TYPE_CHECKING:
    import mujoco

    from .endpoint import Param, ValueType

_log = logging.getLogger(__name__)


class Blackboard:
    """A tiny typed key/value store for cross-plugin data (no direct plugin-to-plugin imports).

    Values are looked up by string key. Use :meth:`require` when a missing value is a hard error
    (e.g. a bridge that needs a robot handle registered by a controller plugin).
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"blackboard entry {key!r} is required but was never set")
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data


@dataclass
class RobotHandle:
    """A controller plugin publishes this so transport/bridge plugins can command the robot.

    ``drive`` takes body-frame velocities (vx forward, vy left, w yaw-rate). ``read_odom`` returns
    the latest ``(x, y, yaw, vx, vy, w)`` estimate. Both run on the physics thread.

    ``kinematics`` says which of those three components the base can actually realise, so a consumer
    that must *shape* a command -- a planner turning a direction into a twist, a teleop mapping a
    stick -- can do so without a list of robot names. Declared by the controller for the same reason
    ``transport_only`` is declared by the plugin: a name list in the core would silently serve only
    the drives we happen to ship, and an out-of-tree one would be shaped wrongly and in silence.

    * ``unicycle`` -- drives and turns, cannot strafe. Differential and skid-steer bases, and the
      legged platforms, whose locomotion controllers take the same twist.
    * ``holonomic`` -- any planar velocity, including sideways. Mecanum, omni-wheel and swerve.
    * ``ackermann`` -- cannot turn in place, and a twist states a *curvature*: the steering angle is
      derived from ``w / v``, so ``w`` with ``v == 0`` steers the wheels nowhere. A consumer that
      commands a stop-and-pivot leaves a car sitting still with its wheels straight.

    It defaults to ``unicycle`` because that is the largest family here and because a default lets
    every existing publisher stay as it is; a base that is not one declares it.
    """

    name: str
    drive: Callable[[float, float, float], None]
    read_odom: Callable[[], tuple[float, float, float, float, float, float]]
    kinematics: str = "unicycle"


@dataclass
class Entity:
    """A named thing in the world (robot, object, pedestrian) discoverable by simulation_interfaces."""

    name: str
    kind: str  # "robot" | "object" | "pedestrian" | ...
    body: str | None = None  # MuJoCo body name, when applicable
    meta: dict = field(default_factory=dict)
    #: Whether anything can perceive or touch it. Absent entities stay in the compiled model
    #: -- nothing can add a body to one at runtime -- but are excluded from raycasts, from
    #: rendering, from contacts, and from what the control plane lists. See :mod:`roqsim.presence`.
    present: bool = True


class EntityRegistry:
    """Registry of entities in the world. Backs simulation_interfaces spawn/delete/get-state."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}

    def add(self, entity: Entity) -> None:
        """Register *entity*. A name already taken is refused, never overwritten.

        Entities are named by the address of the entry that creates them, which is unique by
        construction -- so a collision here means a bug, not a world an author could write. A silent
        overwrite would leave one entity hiding another: every sensor and controller aimed at the
        lost one would resolve to the survivor, and report plausible numbers about the wrong thing.
        """
        existing = self._entities.get(entity.name)
        if existing is not None and existing is not entity:
            raise RuntimeError(
                f"entity {entity.name!r} is already registered (as kind {existing.kind!r}, "
                f"body {existing.body!r}). Entity names come from entry addresses and cannot "
                f"collide in a loaded document, so this is a plugin registering a name it built "
                f"itself rather than one derived from its address."
            )
        self._entities[entity.name] = entity

    def remove(self, name: str) -> None:
        self._entities.pop(name, None)

    def get(self, name: str) -> Entity | None:
        return self._entities.get(name)

    def names(self, present_only: bool = False) -> list[str]:
        """Entity names; with *present_only*, just the ones anything can currently perceive.

        The control plane lists the present ones, because an absent entity is one this world
        compiled but the trial has not brought in yet -- reporting it would make ``GetEntities``
        disagree with every sensor.
        """
        return [n for n, e in self._entities.items() if e.present or not present_only]

    def all(self, present_only: bool = False) -> list[Entity]:
        return [e for e in self._entities.values() if e.present or not present_only]


@dataclass
class Endpoint:
    """One backend-neutral I/O port of a robot's interface, declared by the owning plugin.

    A plugin that produces or consumes data (a controller, a sensor) registers its ports on
    ``ctx.interface`` in ``configure()``. A transport/bridge plugin (ROS 2, zenoh, zmq, ...) reads
    the registry and wires each port to its wire protocol -- so the robot and its bridge share no
    hand-maintained key contract.

    The robot package imports nothing backend-specific: ``read``/``write`` traffic in *neutral*
    payloads (numpy arrays, tuples, small dataclasses), never wire messages. Backend particulars
    (message type, topic, frame, QoS, ...) live as inert data in ``backend``, keyed by backend name,
    e.g. ``backend={"ros2": {"type": "sensor_msgs.msg.LaserScan", "topic": "scan"}}``. Naming a type
    as a *string* keeps the package free of backend *imports* while still letting a robot describe
    backend-specific details -- the bridge resolves the string (e.g. via ``importlib``).

    ``read`` (for ``direction == "out"``) returns the current neutral payload and runs on the physics
    thread. ``write`` (for ``direction == "in"``) receives a neutral payload. Unless ``marshalled``
    is set, it runs on the physics thread and the bridge marshals the call there via
    :meth:`SimContext.submit`, so plugins never touch ``data`` off-thread. A ``marshalled`` write is
    safe to call from any thread because it queues the work itself -- what the decorators of
    :mod:`roqsim.endpoint` produce: a command's write returns a :class:`CommandFuture`, a stream's
    stores the payload in a latest-value :class:`StreamSlot`. A bridge calls it directly.

    ``params``, ``result`` and ``payload_type`` are the endpoint's schema, as data any bridge can
    read (see :mod:`roqsim.endpoint`). An ``in`` endpoint with ``params`` takes a mapping of those
    names (``None`` for none) and checks it before queueing: a command's ``write`` returns a future
    that raises :class:`roqsim.endpoint.ParameterError` for a misfit, a stream's raises it to the
    caller. ``result`` types the ``out`` payload or the command's outcome, and ``payload_type`` is
    what a transport carries, which is how a bridge serves a decorated endpoint that names no wire
    type (``transport``). ``topic`` and ``qos`` are what the world set for it.

    An ``in`` endpoint says what *kind* of interaction it is through its backend hints, and the choice
    is about the interaction rather than about taste: a plain ``type`` is a stream with no answer, a
    ``service`` is a command whose outcome the caller needs (so it can fail on it), and an ``action``
    is a goal that takes time, reports feedback and can be cancelled. The bridge's inbound callback
    returns a :class:`CommandFuture` for the call, whose value is the producer's own and never a
    backend's reply type; a reply is assembled by the backend's handler from that outcome and from
    the producer's published state.
    ``rate_hz`` is the default publish rate (0 => every step / event-driven); a bridge may override it.

    ``namespace`` is a plain scope string declared by the producer (usually from its ``namespace:``
    config); each bridge attaches it however its transport scopes things -- topic prefix, TF frame
    prefix, action name. Empty means unscoped. This is what keeps several robots' identical ports
    (two arms' ``joint_states``) apart under a single bridge, with no bridge-specific config.

    ``has_subscribers`` is an optional performance hint: a transport/bridge may, after wiring the
    endpoint, set this to a zero-arg callable reporting whether anyone is currently listening (e.g.
    a ROS 2 publisher's subscription count). A producer whose ``read`` is expensive to *produce*
    (a rendered camera frame, not just a cheap ray cast) may check it in ``post_step`` and skip the
    work when it returns ``False``. Left as ``None`` (the default) when no transport is loaded, or
    when the active one doesn't support the introspection -- producers must treat that as "assume
    yes" so the endpoint stays live by default.

    ``lazy`` opts THIS endpoint out of publishing while ``has_subscribers`` reports nobody listening,
    so an expensive payload is never even read. It is per-endpoint on purpose, and distinct from the
    render-side check above: a producer whose one render feeds several endpoints must render when
    *any* of them has a consumer, but must only pay each endpoint's own serialisation cost (a JPEG
    encode, a megabyte of raw pixels) when *that* endpoint has one. Left ``False`` by default because
    it is wrong for anything whose publish has a side effect beyond the message -- a bridge deriving
    TF from an odometry endpoint would stop broadcasting the transform whenever nothing happened to
    subscribe to ``/odom`` -- and because it buys nothing for a cheap payload.
    """

    name: str
    direction: str  # "out" (sim -> world) | "in" (world -> sim)
    owner: str = ""  # entity name this port belongs to (identity; see ``namespace`` for scoping)
    namespace: str = ""  # transport scope prefix; a bridge attaches it to topics/frames/actions
    read: Callable[[], Any] | None = None
    write: Callable[[Any], None] | None = None
    rate_hz: float = 0.0
    backend: dict[str, dict | None] = field(default_factory=dict)
    has_subscribers: Callable[[], bool] | None = None
    lazy: bool = False
    marshalled: bool = (
        False  # ``write`` queues onto the physics thread itself; call it from anywhere
    )
    #: The named parameters ``write`` takes (:class:`roqsim.endpoint.Param`), for an ``in`` endpoint
    #: declared with :mod:`roqsim.endpoint`: its payload is a mapping of these names, checked before
    #: anything is queued. ``None``: an untyped write, handed its payload as the bridge built it.
    params: tuple[Param, ...] | None = None
    #: The type of what ``read`` returns (``out``) or what a command's future resolves to, as
    #: :class:`roqsim.endpoint.ValueType`; ``None`` when not declared.
    result: ValueType | None = None
    #: The type a transport carries (:class:`roqsim.endpoint.ValueType`): an ``out``'s result, or the
    #: dataclass an ``in`` endpoint takes -- its ``params`` are that type's fields, by name, or its one
    #: parameter is the whole value. ``None`` when not declared. A bridge maps it to its wire type.
    payload_type: ValueType | None = None
    #: The world's name for this endpoint on a transport (the producer's ``topics:`` config): absolute
    #: with a leading ``/``, else under ``namespace``. ``None``: the backend's hint, else ``name``.
    topic: str | None = None
    #: The world's quality of service for this endpoint (the producer's ``qos:`` config), as a full
    #: profile of :func:`roqsim.endpoint.qos_profile`; it wins over a backend hint's. ``None``: unset.
    qos: dict[str, Any] | None = None
    #: A bridge serves this endpoint without a hint block for its backend, from ``payload_type``'s
    #: default mapping; a hint block of ``None`` keeps it off that backend. Set for every decorated
    #: endpoint. ``False``: served only by a backend whose hint block it carries.
    transport: bool = False
    #: ``"out"``, ``"command"`` or ``"stream"``; empty on a hand-built endpoint, whose kind
    #: :func:`endpoint_kind` infers from its direction and hints.
    kind: str = ""
    #: The address of the plugin that registered it (``robot.lidar``), stamped by the registry.
    #: What a transport that addresses endpoints by path builds the path from.
    producer: str = ""
    #: For a command: the name of an ``out`` endpoint of the same producer whose value confirms it
    #: -- a verdict its ``post_step`` records after the command applied.
    confirm: str = ""
    #: A stream's latest-value slot, set by :mod:`roqsim.endpoint`; its ``write`` then takes the
    #: writing transport as ``source``, so two driving one stream are told apart.
    slot: StreamSlot | None = None
    #: What it is, for a reader outside the process: a decorated method's docstring.
    doc: str = ""
    #: Where a decorated endpoint's options come from, for a reader outside the process: ``rate``
    #: (``{"from": <attribute or config key>}``), ``when`` and ``family`` (the key named), each
    #: ``"computed"`` for a callable and absent where not given. Empty on a hand-built endpoint.
    options: dict[str, Any] = field(default_factory=dict)


def endpoint_kind(ep: Endpoint) -> str:
    """What an endpoint is to a caller: ``out``, ``command`` (an outcome to wait for) or ``stream``.

    A decorated endpoint says so itself. A hand-built ``in`` endpoint is a stream when its only
    hint is a topic ``type``, and a command otherwise -- a ``service`` or ``action`` hint, or none.
    """
    if ep.kind:
        return ep.kind
    if ep.direction == "out":
        return "out"
    hints = [h for h in ep.backend.values() if isinstance(h, dict)]
    if hints and all("type" in h and "service" not in h and "action" not in h for h in hints):
        return "stream"
    return "command"


_T = TypeVar("_T")


class CommandFuture(Generic[_T]):
    """The outcome of a command submitted to the physics thread (:meth:`SimContext.submit`).

    A caller on another thread waits for it with a timeout. :meth:`result` returns what the command
    returned or raises what it raised; :meth:`wait` only says whether it has run. A command that
    raises while nobody is blocked in :meth:`result` is also logged, so a failure whose caller gave
    up waiting, or never asked, is not lost.

    A command that answers only later returns one of its own, and declares what it resolves to:
    ``-> CommandFuture[RunState]`` is described as a ``RunState`` result.
    """

    __slots__ = ("_done", "_error", "_lock", "_value", "_waiters")

    def __init__(self) -> None:
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._value: Any = None
        self._error: BaseException | None = None
        self._waiters = 0

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the command has run (returned or raised). ``False`` on timeout."""
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> Any:
        """The command's return value; re-raises its exception. :class:`TimeoutError` if not run."""
        with self._lock:
            self._waiters += 1
        try:
            if not self._done.wait(timeout):
                raise TimeoutError(f"the physics thread did not run the command within {timeout} s")
        finally:
            with self._lock:
                self._waiters -= 1
        if self._error is not None:
            raise self._error
        return self._value

    def _resolve(self, value: Any = None, error: BaseException | None = None) -> bool:
        """Settle the future on the physics thread. ``True`` when a caller is waiting on it."""
        self._value, self._error = value, error
        with self._lock:
            waited = self._waiters > 0
            self._done.set()
        return waited


#: Two transports writing one stream this close together (seconds, wall clock) are both driving it.
TWO_WRITERS_WINDOW_S = 1.0


class StreamSlot:
    """The latest value an inbound stream delivered, applied once on the physics thread.

    :meth:`put` is safe from any thread and never blocks: a newer value replaces one not yet
    applied. :meth:`SimContext.drain_commands` hands a pending value to ``apply`` once and clears
    it, so a stream that delivers several values within one step applies only the last.
    """

    __slots__ = ("_pending", "_source", "_since", "_warned", "apply", "name")

    def __init__(self, name: str, apply: Callable[[Any], None]) -> None:
        self.name = name
        self.apply = apply
        # One slot of a bounded deque: append and pop are atomic, and append drops the older value.
        self._pending: deque = deque(maxlen=1)
        self._source: str | None = None
        self._since = 0.0
        self._warned = False

    def put(self, payload: Any, source: str | None = None) -> None:
        """Keep *payload* as the value to apply. *source* names the transport that wrote it.

        Two transports writing one stream overwrite each other value by value, which reads as a
        robot that jitters rather than as a conflict; the first time a second source writes within
        :data:`TWO_WRITERS_WINDOW_S` of the other, this logs one WARNING naming both.
        """
        self._pending.append(payload)
        if source is None:
            return
        now = time.monotonic()
        if (
            self._source is not None
            and source != self._source
            and now - self._since < TWO_WRITERS_WINDOW_S
            and not self._warned
        ):
            self._warned = True
            _log.warning(
                "stream %r is written by both %s and %s: the latest value wins, so each "
                "overwrites the other's commands",
                self.name,
                self._source,
                source,
            )
        self._source, self._since = source, now

    def _take(self) -> tuple[bool, Any]:
        try:
            return True, self._pending.pop()
        except IndexError:
            return False, None


class InterfaceRegistry:
    """Registry of the world's :class:`Endpoint`s. Read by transport/bridge plugins.

    A transport plugin binds this registry **once**, in its own ``configure()``, which is why the
    world YAML convention puts the bridge after its producers. :meth:`mark_bound` lets it record
    that, so a producer listed too late fails loudly instead of going silently unpublished -- the
    symptom is a missing topic or TF frame with nothing in the log, which is expensive to track down
    from the consumer end.
    """

    def __init__(self) -> None:
        self._endpoints: list[Endpoint] = []
        self._bound_by: str | None = None
        #: The address of the plugin being configured, set by the engine around ``configure``: an
        #: endpoint added without a ``producer`` is stamped with it.
        self.producer: str = ""
        #: The transport plugins that bound this registry, in binding order.
        self.bridges: list = []

    def add(self, endpoint: Endpoint, *, on_demand: bool = False) -> None:
        """Register *endpoint*.

        ``on_demand`` marks one that is only ever read when a consumer asks for it by name -- the
        core's entity poses (:mod:`roqsim.entity_pose`) -- so registering it after a bridge bound
        loses no publication, and is allowed.
        """
        if self._bound_by is not None and not on_demand:
            raise RuntimeError(
                f"endpoint {endpoint.name!r} (owner {endpoint.owner!r}) was registered after "
                f"{self._bound_by!r} already bound the interface, so nothing would publish it. "
                f"List the producing plugin BEFORE {self._bound_by!r} in the world YAML."
            )
        if not endpoint.producer:
            endpoint.producer = self.producer
        self._endpoints.append(endpoint)

    def mark_bound(self, by: str) -> None:
        """Record that *by* (a transport plugin) has bound the endpoint set."""
        self._bound_by = by

    def all(self) -> list[Endpoint]:
        return list(self._endpoints)

    def by_direction(self, direction: str) -> list[Endpoint]:
        return [e for e in self._endpoints if e.direction == direction]

    def find(self, owner: str, name: str) -> Endpoint | None:
        """The endpoint *owner* declared as *name*, or ``None`` if it declared none.

        ``(owner, name)`` is how a consumer outside the world addresses one endpoint -- a name alone
        repeats across entities (every monitored arm has a ``force_limit``). Where one entity
        declares a name more than once, as a robot with several arm controllers does with
        ``joint_states`` (each scoped by its own namespace), the pair cannot tell them apart, and
        this raises rather than returning whichever was registered first.
        """
        found = [e for e in self._endpoints if e.owner == owner and e.name == name]
        if len(found) > 1:
            scopes = ", ".join(repr(e.namespace) for e in found)
            raise LookupError(
                f"entity {owner!r} declares {len(found)} endpoints named {name!r} (namespaces "
                f"{scopes}), so (entity, name) does not identify one of them"
            )
        return found[0] if found else None


class Gate:
    """A named barrier condition used by the foreseen synchronous/lockstep mode.

    Producers (sensors) call :meth:`satisfy` once they have published for the current tick;
    consumers (controllers) leave it pending until their expected input arrives. In free-running
    mode (the M1 default) gates are recorded but never waited on.
    """

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role  # "producer" | "consumer"
        self._event = threading.Event()

    def satisfy(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    def is_satisfied(self) -> bool:
        return self._event.is_set()


class SimContext:
    """Everything a plugin needs, passed to every hook.

    During the build phase ``spec`` is set and ``model``/``data`` are ``None``; after compile they
    are populated and ``spec`` is left in place for reference (do not mutate it at runtime).
    """

    def __init__(self, config: dict, logger: logging.Logger | None = None):
        self.config: dict = config
        self.logger: logging.Logger = logger or logging.getLogger("roqsim")

        # MuJoCo handles (filled by the engine).
        self.spec: mujoco.MjSpec | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

        # Shared cooperation surfaces.
        self.blackboard = Blackboard()
        self.entities = EntityRegistry()
        self.interface = InterfaceRegistry()
        #: Entities whose core pose endpoint is registered (:mod:`roqsim.entity_pose`).
        self.entity_poses: set[str] = set()
        self.render = None  # lazily set to a RenderService when first needed

        #: What each spawned model's actuators ended up running under, keyed by entity: a list of
        #: :class:`roqsim.actuators.ResolvedActuator`, filled by the spawn plugins at ``configure``
        #: and written into the run's provenance. It carries EVERY actuator, not only the ones an
        #: ``actuators:`` block changed, because "what did this joint run under" is a question about
        #: the run rather than about the diff -- an answer listing only the changes would need the
        #: model opened to be understood. A world that overrides nothing still fills it.
        self.actuator_tables: dict[str, list] = {}

        #: What each published endpoint ended up going out at: one row per output a bridge bound,
        #: filled at ``configure`` and written into the run's provenance. A publish can only land on
        #: a physics step, so a requested rate that is not a whole number of steps is served at a
        #: neighbouring one -- the row carries both numbers, because a reader holding only the world
        #: document has the requested one and no way to learn the other.
        self.endpoint_rates: list[dict] = []

        # Manual control: when True the *human* owns ``data.ctrl`` this run, so every controller
        # plugin must leave it alone and let the viewer's control sliders drive the actuators. A
        # run-level switch (the runner's ``--manual-control``), not world config: which controller a
        # world wires up is a property of the experiment, whereas driving it by hand is a property of
        # one interactive session. Controllers still track state and serve their endpoints; they only
        # stop stamping ctrl. Seeding ctrl once in ``on_reset`` is fine (and wanted -- it puts the
        # sliders at the robot's home pose); the rule is about the per-tick write in ``pre_step``.
        self.manual_control: bool = False

        # Deterministic noise. `seed` is set by the driver (`roqsim.seed.resolve_seed`); `None` means
        # "draw one and record it", which is the driver's job, not this object's. `rng_for` raises on
        # `None` rather than standing in a default -- see its docstring for why.
        self.seed: int | None = None
        #: Which trial this is within the process, counted from 0 and advanced by
        #: :meth:`roqsim.engine.Engine.reset`. It is part of the noise key: a reset puts
        #: ``data.time`` back to zero, so without it every trial after the first would
        #: replay the *same* noise sequence -- repetitions that are duplicates wearing
        #: the clothes of samples. Keyed rather than re-seeded so the whole series stays
        #: reproducible from one base seed, and so trial *i* of two different runs draws
        #: the same noise when their seeds match (common random numbers).
        self.episode: int = 0

        # Run-control (play/pause/step/reset); consulted by the standalone driver.
        from .control import RunControl

        self.control = RunControl()

        # End-of-run request, for the standalone driver: a trial run by `roqsim sim` that knows it is
        # finished says so, rather than the world being padded out to a wall-clock `--seconds`
        # guessed high enough for the slowest cell. `roqsim sim` polls `stop_requested` and exits its
        # loop cleanly, so `shutdown` still runs and files still flush. Under scenario-execution the
        # scenario owns the end of the run and nothing reads this.
        self.stop_requested: bool = False
        self.stop_reason: str = ""

        # Thread-safe command queue: external threads post, the physics thread drains. A deque's
        # append and popleft are atomic, so neither side takes a lock; the one consumer is the
        # physics thread, so a non-empty check followed by popleft cannot race another reader.
        self._commands: deque[tuple[Callable[[SimContext], Any], CommandFuture | None]] = deque()
        # Latest-value slots of the inbound streams, applied once per drain.
        self._streams: list[StreamSlot] = []

        # Step gates (inert until synchronous mode is enabled).
        self._gates: dict[str, Gate] = {}
        self.sync_enabled: bool = False

        # Post-step immutable snapshot for cross-thread readers.
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict | None = None

    # -- deterministic randomness -------------------------------------------------------------

    def rng_for(self, name: str):  # -> numpy.random.Generator (imported lazily below)
        """A generator whose draws are a pure function of ``(seed, episode, sim_time, name)``.

        **Counter-based, not stateful**, and that is the whole point. A shared stateful generator's
        position depends on how many draws happened before it -- sensor rates, step count, and for
        cameras whether anyone was subscribed -- so it is not even a function of the world, and a value
        drawn at t = 12.5 cannot be reproduced without replaying the entire run. A counter-based
        generator (numpy's Philox) is randomly accessible: the same ``(key, counter)`` reproduces the
        same draws with no stream to replay.

        The counter is keyed on **simulated time**, not on a step counter, because that is what a
        recording carries: a restored state knows its ``sim_time`` but nothing knows how many steps
        preceded it. That is what lets a sensor be re-run from a recording and produce the *same* noise
        the live run published.

        Call this **once per (sensor, step)** and draw from the result -- not once per value. A generator
        costs ~8 us to construct, which is 11% of a 1080-beam lidar's own work at 30 Hz (0.02% of wall
        time) but would be absurd per beam. Noise draws are vectorised anyway, so the natural shape is
        already the right one.

        **An unset seed raises.** A seed is driver-owned, so a run without one is missing a required
        input, and standing in a default would be the worst possible failure here: every trial of
        every run draws the same numbers, each run still looks like its own, the recording still
        reports the seed as absent, and only reading the drawn values reveals it. A world run
        repeatedly to estimate a spread would estimate nothing, and say nothing. So it fails at
        the first draw, before there are results to mistake for samples.

        Raised here and not from ``setup()`` because this is the only place that knows a draw is
        happening: a geometry export or a ``scenes describe`` builds the same world and needs no
        seed at all.
        """
        import numpy as np

        if self.seed is None:
            raise SeedError(
                "no seed was resolved for this run, so there is nothing to draw the "
                f"randomness for {name!r} from. The seed is the DRIVER's to resolve, and which "
                "kind of driver this is decides how. A driver that RUNS the world calls "
                "`roqsim.seed.resolve_seed(explicit, logger, config_seed=cfg.seed)` and assigns "
                "the result to `ctx.seed` BEFORE `engine.setup()` (`configure` may read it, "
                "`pre_step` does) -- `roqsim sim` and the scenario adapter both do. A driver that "
                "only LOOKS at the world -- a render, an export, a map, a load check -- passes "
                "`Engine(cfg, preview=True)` instead, which pins the fixed preview seed, since "
                "there is no run to reproduce and a picture is not a measurement."
            )
        seed = int(self.seed)
        step = 0 if self.model is None or self.data is None else round(self.sim_time / self.dt)
        # A stable hash of the sensor name: Python's hash() is salted per process, which would make a
        # run irreproducible across processes -- exactly what this exists to prevent.
        import zlib

        stream = zlib.crc32(name.encode()) & 0xFFFFFFFF
        # The episode occupies a counter slot rather than perturbing the key, which keeps the key
        # meaning exactly "the run's seed" and leaves the generator randomly accessible: (seed,
        # episode, step) addresses a draw directly, with no stream to replay. A reset restarts
        # `sim_time`, so without this slot trial 2 would re-draw trial 1's noise step for step.
        return np.random.Generator(
            np.random.Philox(key=seed, counter=[step, stream, int(self.episode), 0])
        )

    # -- time ---------------------------------------------------------------------------------
    @property
    def dt(self) -> float:
        if self.model is None:
            raise RuntimeError("dt is unavailable before the model is compiled")
        return float(self.model.opt.timestep)

    @property
    def sim_time(self) -> float:
        return float(self.data.time) if self.data is not None else 0.0

    # -- command queue ------------------------------------------------------------------------
    def post(self, command: Callable[[SimContext], None]) -> None:
        """Enqueue a callable to run on the physics thread at the start of the next ``pre_step``.

        While the run is paused the driver runs it from its idle loop (:meth:`roqsim.engine.Engine.idle`).

        This is the ONLY safe way for a non-physics thread (e.g. a ROS executor) to cause a change
        to ``model``/``data``. The command receives this context when it runs. An exception it
        raises is logged; use :meth:`submit` when the caller needs the outcome.
        """
        self._commands.append((command, None))

    def submit(self, command: Callable[[SimContext], Any]) -> CommandFuture:
        """Like :meth:`post`, and return a :class:`CommandFuture` carrying the command's outcome."""
        future = CommandFuture()
        self._commands.append((command, future))
        return future

    def stream_slot(self, name: str, apply: Callable[[Any], None]) -> StreamSlot:
        """A latest-value slot for an inbound stream; ``apply`` runs on the physics thread."""
        slot = StreamSlot(name, apply)
        self._streams.append(slot)
        return slot

    def drain_commands(self) -> int:
        """Run queued commands in FIFO order, then apply each stream's latest value, on the
        calling (physics) thread. Returns the number of commands run.

        A command's exception goes to the caller waiting on its future; one that nobody waits on
        is logged. Neither stops the loop.
        """
        commands = self._commands
        n = 0
        while commands:
            command, future = commands.popleft()
            n += 1
            try:
                value = command(self)
            except Exception as exc:  # a bad command must not kill the loop
                if future is None or not future._resolve(error=exc):
                    self.logger.exception("posted command raised")
                continue
            if future is not None:
                future._resolve(value)
        for slot in self._streams:
            fresh, payload = slot._take()
            if not fresh:
                continue
            try:
                slot.apply(payload)
            except Exception:
                self.logger.exception("stream %r raised applying its latest value", slot.name)
        return n

    # -- snapshots ----------------------------------------------------------------------------
    def request_stop(self, reason: str = "") -> None:
        """Ask the standalone driver to end the run after this step. Idempotent; the first reason wins.

        `roqsim sim` honours it. Under scenario-execution the scenario owns when a run ends and the
        adapter does not read it, so a trial that must end a scenario run publishes its outcome for
        the scenario to condition on. The engine itself does not act on it -- a request, not a kill
        switch. Physics-thread only, like every other write on this object. A reset withdraws it.
        """
        if not self.stop_requested:
            self.stop_requested = True
            self.stop_reason = reason
            self.logger.info("stop requested: %s", reason or "(no reason given)")

    def publish_snapshot(self, snapshot: dict) -> None:
        with self._snapshot_lock:
            self._snapshot = snapshot

    def read_snapshot(self) -> dict | None:
        with self._snapshot_lock:
            return None if self._snapshot is None else dict(self._snapshot)

    # -- gates (foreseen synchronous mode; inert by default) ----------------------------------
    def register_gate(self, name: str, role: str) -> Gate:
        gate = Gate(name, role)
        self._gates[name] = gate
        return gate

    def gates(self) -> list[Gate]:
        return list(self._gates.values())
