"""Shared per-run state passed to every plugin hook.

:class:`SimContext` is the single object plugins use to cooperate. It exposes the MuJoCo model/data,
config, a typed :class:`Blackboard`, an :class:`EntityRegistry`, the thread-safe command queue
(``post``/``drain``), and the (currently inert) step-gate API used by the foreseen synchronous mode.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mujoco


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
    """

    name: str
    drive: Callable[[float, float, float], None]
    read_odom: Callable[[], tuple[float, float, float, float, float, float]]


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
    the registry and wires each port to its wire protocol -- so the robot and its bridge no longer
    duplicate a hand-maintained key contract.

    The robot package imports nothing backend-specific: ``read``/``write`` traffic in *neutral*
    payloads (numpy arrays, tuples, small dataclasses), never wire messages. Backend particulars
    (message type, topic, frame, QoS, ...) live as inert data in ``backend``, keyed by backend name,
    e.g. ``backend={"ros2": {"type": "sensor_msgs.msg.LaserScan", "topic": "scan"}}``. Naming a type
    as a *string* keeps the package free of backend *imports* while still letting a robot describe
    backend-specific details -- the bridge resolves the string (e.g. via ``importlib``).

    ``read`` (for ``direction == "out"``) returns the current neutral payload and runs on the physics
    thread. ``write`` (for ``direction == "in"``) receives a neutral payload; the bridge marshals it
    onto the physics thread via :meth:`SimContext.post`, so plugins never touch ``data`` off-thread.

    An ``in`` endpoint says what *kind* of interaction it is through its backend hints, and the choice
    is about the interaction rather than about taste: a plain ``type`` is a stream with no answer, a
    ``service`` is a command whose outcome the caller needs (so it can fail on it), and an ``action``
    is a goal that takes time, reports feedback and can be cancelled. ``write`` returns ``None`` in
    every case -- a reply is assembled by the backend's handler from the producer's published state,
    not returned from here, which is what keeps this dataclass free of any backend's reply types.
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
    backend: dict[str, dict] = field(default_factory=dict)
    has_subscribers: Callable[[], bool] | None = None
    lazy: bool = False


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

    def add(self, endpoint: Endpoint) -> None:
        if self._bound_by is not None:
            raise RuntimeError(
                f"endpoint {endpoint.name!r} (owner {endpoint.owner!r}) was registered after "
                f"{self._bound_by!r} already bound the interface, so nothing would publish it. "
                f"List the producing plugin BEFORE {self._bound_by!r} in the world YAML."
            )
        self._endpoints.append(endpoint)

    def mark_bound(self, by: str) -> None:
        """Record that *by* (a transport plugin) has bound the endpoint set."""
        self._bound_by = by

    def all(self) -> list[Endpoint]:
        return list(self._endpoints)

    def by_direction(self, direction: str) -> list[Endpoint]:
        return [e for e in self._endpoints if e.direction == direction]


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
        self.render = None  # lazily set to a RenderService when first needed

        # Manual control: when True the *human* owns ``data.ctrl`` this run, so every controller
        # plugin must leave it alone and let the viewer's control sliders drive the actuators. A
        # run-level switch (the runner's ``--manual-control``), not world config: which controller a
        # world wires up is a property of the experiment, whereas driving it by hand is a property of
        # one interactive session. Controllers still track state and serve their endpoints; they only
        # stop stamping ctrl. Seeding ctrl once in ``on_reset`` is fine (and wanted -- it puts the
        # sliders at the robot's home pose); the rule is about the per-tick write in ``pre_step``.
        self.manual_control: bool = False

        # Deterministic noise. `seed` is set by the driver (`roqsim sim --seed`); `None` means "draw one
        # and record it", which is the driver's job, not this object's. See `rng_for`.
        self.seed: int | None = None

        # Run-control (play/pause/step/reset); consulted by the standalone driver.
        from .control import RunControl

        self.control = RunControl()

        # End-of-run request. A trial that knows it is finished -- the goal was reached, the episode
        # failed, the recording is complete -- should be able to say so, rather than the world being
        # padded out to a wall-clock `--seconds` that has to be guessed high enough for the slowest
        # cell and is then wasted on every faster one. The driver polls `stop_requested` and exits
        # its loop cleanly, so `shutdown` still runs and files still flush.
        self.stop_requested: bool = False
        self.stop_reason: str = ""

        # Thread-safe command queue: external threads post, the physics thread drains.
        self._commands: queue.Queue[Callable[[SimContext], None]] = queue.Queue()

        # Step gates (inert until synchronous mode is enabled).
        self._gates: dict[str, Gate] = {}
        self.sync_enabled: bool = False

        # Post-step immutable snapshot for cross-thread readers.
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict | None = None

    # -- deterministic randomness -------------------------------------------------------------

    def rng_for(self, name: str):  # -> numpy.random.Generator (imported lazily below)
        """A generator whose draws are a pure function of ``(seed, sim_time, name)``.

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
        """
        import numpy as np

        seed = 0 if self.seed is None else int(self.seed)
        step = 0 if self.model is None or self.data is None else round(self.sim_time / self.dt)
        # A stable hash of the sensor name: Python's hash() is salted per process, which would make a
        # run irreproducible across processes -- exactly what this exists to prevent.
        import zlib

        stream = zlib.crc32(name.encode()) & 0xFFFFFFFF
        return np.random.Generator(np.random.Philox(key=seed, counter=[step, stream, 0, 0]))

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

        This is the ONLY safe way for a non-physics thread (e.g. a ROS executor) to cause a change
        to ``model``/``data``. The command receives this context when it runs.
        """
        self._commands.put(command)

    def drain_commands(self) -> int:
        """Run all queued commands on the calling (physics) thread. Returns the count executed."""
        n = 0
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            try:
                command(self)
            except Exception:  # a bad command must not kill the loop
                self.logger.exception("posted command raised")
            n += 1
        return n

    # -- snapshots ----------------------------------------------------------------------------
    def request_stop(self, reason: str = "") -> None:
        """Ask the driver to end the run after this step. Idempotent; the first reason wins.

        Physics-thread only, like every other write on this object. The engine itself does not act
        on it -- an embedding driver (scenario-execution, a test harness) is free to ignore it and
        keep stepping -- so it is a request, not a kill switch.
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
