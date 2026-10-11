"""Control-plane plugin: a subset of ros-simulation/simulation_interfaces.

Implements the interfaces most useful for scenario-driven testing of the M1 turtlebot world:
  * GetSimulatorFeatures   — advertise what is supported
  * GetEntities            — list entities, narrowed by a POSIX ERE name filter and a sphere
  * GetEntityState/SetEntityState — read/teleport an entity's free-joint or mocap body
  * GetSpawnables          — the absent entities SpawnEntity can select, by uri
  * SpawnEntity/DeleteEntity — make a compiled entity perceivable at initial_pose, or absent
  * GetSimulationState/SetSimulationState — play/pause/stop (standalone driver)
  * StepSimulation         — step N times while paused, answering once they ran
  * ResetSimulation        — reset the world, answering once it has

Concurrency: services run on the bridge/executor thread. State changes go through the shared
:class:`roqsim.control.RunControl`; anything touching ``data`` goes through
:func:`roqsim_ros_bridge.physics.run_on_physics`, which posts to the physics thread and waits (see
docs/architecture.rst §7). Every mutating service *waits*: answering ``RESULT_OK`` before the change
has run makes a stalled simulator indistinguishable from a working one.

Reuses the ``rclpy`` node created by :class:`~roqsim_ros_bridge.ros2_bridge.Ros2Bridge` when present
(looked up on the blackboard under ``ros2_node``); otherwise creates and spins its own.
"""

from __future__ import annotations

import math
import re
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from simulation_interfaces.msg import (
    Bounds,
    Result,
    SimulationState,
    SimulatorFeatures,
    Spawnable,
)
from simulation_interfaces.srv import (
    DeleteEntity,
    GetEntities,
    GetEntityState,
    GetSimulationState,
    GetSimulatorFeatures,
    GetSpawnables,
    ResetSimulation,
    SetEntityState,
    SetSimulationState,
    SpawnEntity,
    StepSimulation,
)

from roqsim import control as ctl
from roqsim.kinematics import body_twist
from roqsim.placement import PLACEABLE_MODES_HINT, carries_velocity, place_body
from roqsim.plugin import Plugin
from roqsim.presence import entity_geom_ids, set_present
from roqsim.schema import Field

from .physics import DEFAULT_TIMEOUT_S, run_on_physics

_STATE_TO_MSG = {
    ctl.STOPPED: SimulationState.STATE_STOPPED,
    ctl.PLAYING: SimulationState.STATE_PLAYING,
    ctl.PAUSED: SimulationState.STATE_PAUSED,
    ctl.QUITTING: SimulationState.STATE_QUITTING,
}
_MSG_TO_STATE = {v: k for k, v in _STATE_TO_MSG.items()}

#: How far a welded entity's pose may differ from a requested one and still count as satisfied.
#: Tight, because it exists to absorb float round-trips through the message, not to accept a pose
#: somewhere else.
_POSE_EPS = 1e-6

#: Frame names that mean the world frame. Empty is the service's own default for it; 'world' is
#: spelled out because that is the name the service description gives that frame.
_WORLD_FRAMES = ("", "world")


def _pose_from(msg):
    """``((pos, quat), None)`` for a ``geometry_msgs/Pose``, or ``(None, why)`` when it is no pose.

    A zero-norm quaternion is the one finite value that cannot be a rotation, and a NaN or infinite
    component is no position or rotation at all; the services that take a pose have a code saying
    so (``INVALID_POSE``), so either is reported rather than repaired -- a NaN written into the
    model would corrupt every later step. The quaternion is normalised because MuJoCo reads a free
    joint's quaternion as a unit one; a caller's near-unit value would otherwise scale the body's
    orientation.
    """
    p, o = msg.position, msg.orientation
    values = (p.x, p.y, p.z, o.w, o.x, o.y, o.z)
    if not all(math.isfinite(v) for v in values):
        return None, "the pose carries a NaN or infinite component."
    norm = math.sqrt(o.w * o.w + o.x * o.x + o.y * o.y + o.z * o.z)
    if norm < _POSE_EPS:
        return None, (
            "the pose carries a zero-length quaternion, which is not a rotation. Send a unit "
            "quaternion; the identity is w=1."
        )
    return ((p.x, p.y, p.z), (o.w / norm, o.x / norm, o.y / norm, o.z / norm)), None


def _foreign_frame(stamped) -> str:
    """The frame a stamped request states, when it is not the world frame this simulator knows."""
    frame_id = getattr(getattr(stamped, "header", None), "frame_id", "")
    return frame_id if frame_id and frame_id not in _WORLD_FRAMES else ""


#: POSIX character classes in the POSIX locale, as Python character-set contents.
_POSIX_CLASSES = {
    "alnum": "0-9A-Za-z",
    "alpha": "A-Za-z",
    "blank": " \\t",
    "cntrl": "\\x00-\\x1f\\x7f",
    "digit": "0-9",
    "graph": "\\x21-\\x7e",
    "lower": "a-z",
    "print": "\\x20-\\x7e",
    "punct": "!-/:-@\\[-`{-~",
    "space": " \\t\\n\\r\\f\\v",
    "upper": "A-Z",
    "xdigit": "0-9A-Fa-f",
}

#: Characters a backslash makes literal outside a bracket expression: the ones an ERE gives a
#: meaning, and the closing ``]`` and ``}``. Before any other character it is undefined.
_ERE_ESCAPABLE = set(".[\\()*+?{|^$]}")


class _PosixRegexError(ValueError):
    """A pattern that is not a POSIX Extended regular expression this module can translate."""


def _set_char(c: str) -> str:
    """One literal character inside a Python character set."""
    return "\\" + c if c in "\\]^-[&~|" else c


def _bracket(pattern: str, i: int) -> tuple[str, int]:
    """Translate the bracket expression opening at ``pattern[i] == '['``; ``(python, next_i)``.

    In a POSIX bracket expression a backslash is an ordinary character, ``]`` first is literal,
    ``[:name:]`` is a character class, and ``[.c.]`` / ``[=c=]`` name the single character ``c``
    (the POSIX locale has no multi-character collating elements). Python reads all of these
    differently, so each item is re-emitted as a Python set member.
    """
    n = len(pattern)
    j = i + 1
    out = "["
    if j < n and pattern[j] == "^":
        out += "^"
        j += 1
    items: list[tuple[str, str]] = []  # ("char", c) or ("class", python)
    first = True
    while True:
        if j >= n:
            raise _PosixRegexError("unterminated bracket expression '['")
        c = pattern[j]
        if c == "]" and not first:
            j += 1
            break
        first = False
        if c == "[" and j + 1 < n and pattern[j + 1] in ":.=":
            kind = pattern[j + 1]
            close = pattern.find(kind + "]", j + 2)
            if close < 0:
                raise _PosixRegexError(f"unterminated '[{kind}' in a bracket expression")
            name = pattern[j + 2 : close]
            j = close + 2
            if kind == ":":
                if name not in _POSIX_CLASSES:
                    raise _PosixRegexError(f"unknown character class '[:{name}:]'")
                items.append(("class", _POSIX_CLASSES[name]))
            elif len(name) == 1:
                items.append(("char", name))
            else:
                raise _PosixRegexError(
                    f"'[{kind}{name}{kind}]' names no single character in the POSIX locale"
                )
            continue
        items.append(("char", c))
        j += 1
    k = 0
    while k < len(items):
        kind, val = items[k]
        is_range = (
            kind == "char"
            and k + 2 < len(items)
            and items[k + 1] == ("char", "-")
            and items[k + 2][0] == "char"
        )
        if is_range:
            lo, hi = val, items[k + 2][1]
            if ord(lo) > ord(hi):
                raise _PosixRegexError(f"range '{lo}-{hi}' runs backwards")
            out += _set_char(lo) + "-" + _set_char(hi)
            k += 3
            continue
        out += val if kind == "class" else _set_char(val)
        k += 1
    return out + "]", j


def _posix_ere(pattern: str) -> re.Pattern:
    """Compile a POSIX Extended regular expression (``EntityFilters.filter``) for :meth:`search`.

    The service names POSIX ERE, and Python's ``re`` reads several ERE constructs otherwise, so the
    pattern is translated rather than handed over: bracket expressions (character classes such as
    ``[[:digit:]]``, collating symbols, equivalence classes, and a backslash that is literal inside
    brackets); ``$``, which Python also matches before a trailing newline; ``.``, which in POSIX
    also matches a newline; and a backslash outside brackets, which escapes a special character and
    is otherwise undefined in POSIX, so ``\\d`` or a back-reference is refused rather than given
    Python's meaning. A ``(?`` group is likewise not ERE. Matching is unanchored, as ``regexec``
    is. Raises :class:`_PosixRegexError` (or :class:`re.error`) for what is not a valid ERE.
    """
    out = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            if i + 1 >= n:
                raise _PosixRegexError("trailing backslash")
            nxt = pattern[i + 1]
            if nxt not in _ERE_ESCAPABLE:
                raise _PosixRegexError(
                    f"'\\{nxt}' is not defined in a POSIX Extended regular expression; use a "
                    "bracket expression such as '[[:digit:]]', or escape only a special character"
                )
            out.append(re.escape(nxt))
            i += 2
        elif c == "[":
            translated, i = _bracket(pattern, i)
            out.append(translated)
        elif c == "(" and pattern.startswith("(?", i):
            raise _PosixRegexError("'(?' is not a POSIX Extended regular expression construct")
        elif c == "$":
            out.append(r"\Z")
            i += 1
        else:
            out.append(c)
            i += 1
    return re.compile("".join(out), re.DOTALL)


def _unsupported_entity_filters(filters):
    """``(result_code, message)`` for a ``GetEntities`` filter this simulator cannot apply, else
    ``None``.

    Entities carry no category and no tags here, which ``get_simulator_features`` says by
    advertising neither ENTITY_CATEGORIES nor ENTITY_TAGS. Of the bounds types only
    ``TYPE_SPHERE`` is applied -- the one the service requires -- so ``TYPE_BOX`` and
    ``TYPE_CONVEX_HULL``, which ENTITY_BOUNDS_BOX and ENTITY_BOUNDS_CONVEX would advertise, are
    refused with ``RESULT_FEATURE_UNSUPPORTED``. Answering such a query with every entity would
    hand the caller a list it asked to have narrowed.
    """
    if filters is None:
        return None
    unsupported = []
    if list(getattr(filters, "categories", []) or []):
        unsupported.append("'categories'")
    if list(getattr(getattr(filters, "tags", None), "tags", []) or []):
        unsupported.append("'tags'")
    bounds = getattr(filters, "bounds", None)
    if bounds is not None and bounds.type not in (Bounds.TYPE_EMPTY, Bounds.TYPE_SPHERE):
        unsupported.append(f"'bounds' of type {bounds.type}")
    if unsupported:
        return Result.RESULT_FEATURE_UNSUPPORTED, (
            f"filters {', '.join(unsupported)} cannot be applied: entities here carry no "
            "category or tags, and only TYPE_SPHERE bounds are applied (get_simulator_features "
            "advertises none of ENTITY_CATEGORIES, ENTITY_TAGS, ENTITY_BOUNDS_BOX and "
            "ENTITY_BOUNDS_CONVEX). Filter by name with 'filter' or by a sphere."
        )
    return None


def _sphere_of(bounds):
    """``(center, radius)`` of a ``TYPE_SPHERE`` filter, ``None`` for no bounds filter.

    Raises :class:`ValueError` naming what is wrong with bounds that define no volume.
    """
    points = list(getattr(bounds, "points", []) or []) if bounds is not None else []
    if bounds is None or bounds.type == Bounds.TYPE_EMPTY:
        if points:
            raise ValueError(f"TYPE_EMPTY bounds carry no points, got {len(points)}")
        return None
    if len(points) != 2:
        raise ValueError(
            f"TYPE_SPHERE bounds take two points (the center, then the radius in x), got "
            f"{len(points)}"
        )
    center = (points[0].x, points[0].y, points[0].z)
    radius = points[1].x
    if not all(math.isfinite(v) for v in (*center, radius)):
        raise ValueError("TYPE_SPHERE bounds must be finite")
    if radius < 0.0:
        raise ValueError(f"TYPE_SPHERE radius must not be negative, got {radius}")
    return center, radius


def _overlaps_sphere(model, data, body, center, radius) -> bool:
    """Does any geom of entity body *body* (and its subtree) overlap the sphere? Physics thread.

    Each geom is tested by its bounding sphere (``geom_rbound`` about ``geom_xpos``), so the test
    is conservative: it never misses a geom that overlaps, and may include one whose bounding
    sphere reaches the query sphere while its surface does not. A plane, whose ``geom_rbound`` is
    zero because it is unbounded, is the half-space MuJoCo collides it as.
    """
    import mujoco
    import numpy as np

    c = np.asarray(center, dtype=float)
    for gid in entity_geom_ids(model, body):
        offset = c - data.geom_xpos[gid]
        if int(model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_PLANE:
            normal = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            if float(offset @ normal) <= radius:
                return True
        elif float(np.linalg.norm(offset)) <= radius + float(model.geom_rbound[gid]):
            return True
    return False


def _already_at(state, pose) -> bool:
    """Is the body described by *state* already at *pose*?

    What an entity with no free joint is asked instead of "can you move there": it cannot, so the
    only honourable answer is whether it is there already. The quaternion is compared through its
    dot product because ``q`` and ``-q`` are the same rotation, which a component-wise test would
    call a mismatch.
    """
    pos, quat = pose
    at = state.get("pos")
    rot = state.get("quat")
    if at is None or rot is None:
        return False
    if any(abs(a - b) > _POSE_EPS for a, b in zip(at, pos, strict=True)):
        return False
    return abs(sum(a * b for a, b in zip(rot, quat, strict=True))) > 1.0 - _POSE_EPS


def _resource_of(msg):
    """``(uri, resource_string)`` of a request or a spawnable, in either interface shape.

    simulation_interfaces 1.x (Jazzy) carries both flat; 2.x moved them into a ``Resource``
    sub-message (``entity_resource``). Reading only the flat fields would miss every 2.x request.
    """
    src = getattr(msg, "entity_resource", msg)
    return getattr(src, "uri", ""), getattr(src, "resource_string", "")


def _spawnable(uri, description):
    s = Spawnable()
    getattr(s, "entity_resource", s).uri = uri
    s.description = description
    return s


def _spawnables(entities):
    """The entities ``SpawnEntity`` can select right now: the declared ones that are ABSENT.

    Spawning here is activation, so a spawnable is CONSUMED by spawning it -- unlike a simulator
    whose uri names a model file and spawns a copy per call. A present entity's name is not a uri
    a spawn accepts (it answers "already present"), and the service asks for the uris that are
    valid, so listing it would offer one that cannot be used.

    Together with ``GetEntities``, which lists the present ones, this covers everything the world
    declared.
    """
    return [_spawnable(e.name, e.kind) for e in entities if not e.present]


def _spawn_target(req, names):
    """``(name, None)`` for the declared entity a spawn request selects, or ``(None, refusal)``.

    ``uri`` is the service's selector, and here it names one of the entities ``GetSpawnables``
    lists. ``name`` may be empty or repeat it; any other name would be a rename, and a compiled
    entity keeps the name its world gave it. With no ``uri`` the request selects by ``name`` alone:
    the service calls that invalid, but callers written before ``GetSpawnables`` rely on it, and it
    cannot select anything other than what it names.
    """
    uri, _ = _resource_of(req)
    if not uri:
        if not req.name:
            return None, (
                SpawnEntity.Response.NO_RESOURCE,
                "the request names no entity: set 'uri' to one get_spawnables lists.",
            )
        return req.name, None
    if uri not in names:
        return None, (
            SpawnEntity.Response.UNSUPPORTED_FORMAT,
            f"'uri' {uri!r} is not an entity this world declared, and this simulator loads no "
            f"geometry (get_simulator_features advertises no spawn_formats). Declared: "
            f"{', '.join(sorted(names)) or 'none'}; get_spawnables lists the absent ones a spawn "
            "can select.",
        )
    if req.name and req.name != uri:
        return None, (
            SpawnEntity.Response.NAME_INVALID,
            f"'name' {req.name!r} would rename {uri!r}, and a compiled entity keeps the name its "
            "world gave it. Leave 'name' empty or repeat the uri.",
        )
    return uri, None


def _absent(name) -> Result:
    """``RESULT_NOT_FOUND`` for an entity the world compiled but holds absent.

    The service identifies an entity as ``GetEntities`` or ``SpawnEntity`` returns it, and neither
    returns an absent one: ``GetEntities`` does not list it and ``GetSpawnables`` offers it for
    spawning. So by the service's terms it does not exist, and naming it is "no match for input".
    """
    return Result(
        result=Result.RESULT_NOT_FOUND,
        error_message=(
            f"entity {name!r} is absent, and an absent entity can only be spawned: "
            "get_spawnables lists it, and spawn_entity brings it in."
        ),
    )


def _unsupported_spawn_request(req):
    """``(result_code, message)`` for a request this simulator cannot serve as asked, else ``None``.

    EVERY field of the request is either honoured or named here, and that is the point rather than
    tidiness: a field read off the request and discarded under a ``RESULT_OK`` is the failure this
    module is shaped against -- a trial that believes it spawned something. A
    field added to the service later must join one list or the other.

    The codes are the service's OWN extended ones where it defines a fitting one, which
    ``Result.msg`` asks of an implementation rather than leaving to taste; the generic
    ``RESULT_FEATURE_UNSUPPORTED`` is for a call option this simulator does not offer, which is
    what that code is for.

    ``uri`` and ``name`` are not here: they select the entity, which :func:`_spawn_target` resolves.

    Why none of these is a gap to fill: geometry, because the model is compiled once, which
    ``get_simulator_features`` already says by advertising no ``spawn_formats``; a namespace,
    because an entity's name is settled when the model compiles; renaming, because spawning here
    SELECTS an entity by name, so an existing name is the required case rather than the collision
    ``allow_renaming`` resolves; and a frame, because the service requires one the simulator
    knows, and this one knows only the world frame the empty default already names.
    """
    if _resource_of(req)[1]:
        return SpawnEntity.Response.UNSUPPORTED_FORMAT, (
            "this simulator spawns entities the world compiled and loads no geometry, so "
            "'resource_string' cannot be honoured (get_simulator_features advertises no "
            "spawn_formats). Declare the entity in the world and spawn it by 'uri'."
        )
    if getattr(req, "entity_namespace", ""):
        return Result.RESULT_FEATURE_UNSUPPORTED, (
            "'entity_namespace' cannot be honoured: an entity's name is settled when the model "
            "compiles, so this simulator cannot place one under a namespace. Name it in the world."
        )
    if getattr(req, "allow_renaming", False):
        return Result.RESULT_FEATURE_UNSUPPORTED, (
            "'allow_renaming' cannot be honoured: spawning here selects the entity the world "
            "compiled under this name, so an existing name is what the request needs rather than "
            "a collision to rename around. Ask for the name you want."
        )
    frame_id = _foreign_frame(req.initial_pose)
    if frame_id:
        return SpawnEntity.Response.INVALID_POSE, (
            f"initial_pose is stated in frame {frame_id!r}, which this simulator does not know. "
            "It places entities in the world frame, which the empty default already names."
        )
    return None


class SimInterfacesPlugin(Plugin):
    # Pure service surface over an already-built world -- no geometry, no state of its own -- so a
    # scene consumer (render, review, export) may drop it. See Plugin.transport_only.
    transport_only = True

    CONFIG_SCHEMA = {
        "node_name": Field(
            str,
            default="roqsim_interfaces",
            doc="name of the node the services are served on, where no ros2_bridge node is shared",
        ),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._ctx = None
        self._node: Node | None = None
        self._own_node = False
        self._executor = None
        self._thread = None
        self._we_inited_rclpy = False
        # One event per ResetSimulation call waiting for the reset it asked for (see _reset).
        self._reset_lock = threading.Lock()
        self._reset_waiters: list[threading.Event] = []

    def configure(self, ctx) -> None:
        self._ctx = ctx
        node = ctx.blackboard.get("ros2_node")
        if node is None:
            if not rclpy.ok():
                rclpy.init()
                self._we_inited_rclpy = True
            node = Node(self.settings.node_name)
            self._own_node = True
        self._node = node

        node.create_service(GetSimulatorFeatures, "get_simulator_features", self._get_features)
        node.create_service(GetEntities, "get_entities", self._get_entities)
        node.create_service(GetSpawnables, "get_spawnables", self._get_spawnables)
        # Spawning here is ACTIVATION, not creation: the model is compiled once and never
        # rebuilt, so these two make an entity the world already carries perceivable or not.
        node.create_service(SpawnEntity, "spawn_entity", self._spawn_entity)
        node.create_service(DeleteEntity, "delete_entity", self._delete_entity)
        node.create_service(GetEntityState, "get_entity_state", self._get_entity_state)
        node.create_service(SetEntityState, "set_entity_state", self._set_entity_state)
        node.create_service(GetSimulationState, "get_simulation_state", self._get_sim_state)
        node.create_service(SetSimulationState, "set_simulation_state", self._set_sim_state)
        node.create_service(StepSimulation, "step_simulation", self._step)
        node.create_service(ResetSimulation, "reset_simulation", self._reset)

        if self._own_node:
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()

    def shutdown(self, ctx) -> None:
        if self._own_node:
            if self._executor is not None:
                self._executor.shutdown()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if self._node is not None:
                self._node.destroy_node()
            if self._we_inited_rclpy and rclpy.ok():
                rclpy.shutdown()

    # -- services -----------------------------------------------------------------------------
    def _get_features(self, req, resp):
        f = SimulatorFeatures()
        f.features = [
            # Both are served over presence, not creation -- see _spawn_entity. Advertised
            # because a caller's question is "can I make this entity appear", and here it can.
            SimulatorFeatures.SPAWNING,
            SimulatorFeatures.SPAWNABLES,
            SimulatorFeatures.DELETING,
            SimulatorFeatures.ENTITY_STATE_GETTING,
            SimulatorFeatures.ENTITY_STATE_SETTING,
            SimulatorFeatures.SIMULATION_RESET,
            SimulatorFeatures.SIMULATION_RESET_STATE,
            SimulatorFeatures.SIMULATION_STATE_GETTING,
            SimulatorFeatures.SIMULATION_STATE_SETTING,
            SimulatorFeatures.SIMULATION_STATE_PAUSE,
            SimulatorFeatures.STEP_SIMULATION_SINGLE,
            SimulatorFeatures.STEP_SIMULATION_MULTIPLE,
        ]
        # No format: nothing is loaded. Spawning selects an entity the world compiled, so a
        # caller offering MJCF would be refused -- saying "mjcf" here would invite exactly that.
        f.spawn_formats = []
        f.custom_info = (
            "roqsim bridge (M1 subset); spawn/delete activate entities the world compiled, "
            "spawn selecting one by the uri get_spawnables lists and placing a free-jointed one "
            "at initial_pose (world frame only) -- the model is never rebuilt at runtime"
        )
        resp.features = f
        return resp

    def _get_entities(self, req, resp):
        """The present entities, narrowed by the request's name ``filter`` and sphere ``bounds``.

        ``filter`` is a POSIX Extended regular expression searched in each name (:func:`_posix_ere`).
        ``bounds`` of ``TYPE_SPHERE`` keeps the entities whose geometry overlaps the sphere, given
        in the world frame (:func:`_overlaps_sphere`); an entity with no body has no geometry and
        overlaps nothing. A filter this simulator cannot apply is refused by name
        (:func:`_unsupported_entity_filters`). A malformed filter -- a pattern that is not an ERE,
        bounds that define no volume -- is ``RESULT_OPERATION_FAILED`` naming what is wrong:
        ``Result.msg`` has no code for invalid input and ``GetEntities`` defines none of its own,
        and ``RESULT_FEATURE_UNSUPPORTED`` would send the caller to the feature list for a feature
        that is supported.
        """
        filters = getattr(req, "filters", None)
        unsupported = _unsupported_entity_filters(filters)
        if unsupported:
            code, message = unsupported
            resp.result = Result(result=code, error_message=message)
            return resp
        pattern = getattr(filters, "filter", "") if filters is not None else ""
        try:
            regex = _posix_ere(pattern) if pattern else None
        except (re.error, ValueError) as exc:
            return self._refuse_filter(
                resp, f"'filter' {pattern!r} is not a POSIX Extended regular expression: {exc}"
            )
        try:
            sphere = _sphere_of(getattr(filters, "bounds", None))
        except ValueError as exc:
            return self._refuse_filter(resp, f"'bounds' define no sphere: {exc}")
        # Present ones only: an absent entity is compiled into the model but nothing can see
        # or touch it, so listing it would make this disagree with every sensor.
        names = self._ctx.entities.names(present_only=True)
        if regex is not None:
            names = [name for name in names if regex.search(name)]
        if sphere is not None:
            center, radius = sphere
            bodies = {name: getattr(self._ctx.entities.get(name), "body", None) for name in names}
            # Recorded positively: run_on_physics also returns True for a command that raised.
            outcome = {}

            def _overlapping(ctx):
                outcome["kept"] = [
                    name
                    for name in names
                    if bodies[name]
                    and _overlaps_sphere(ctx.model, ctx.data, bodies[name], center, radius)
                ]

            if not run_on_physics(self._ctx, _overlapping) or "kept" not in outcome:
                resp.result = Result(
                    result=Result.RESULT_OPERATION_FAILED,
                    error_message=f"the simulation did not answer the bounds query within "
                    f"{DEFAULT_TIMEOUT_S} s or raised doing so; its log carries the reason.",
                )
                return resp
            names = outcome["kept"]
        resp.entities = names
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    @staticmethod
    def _refuse_filter(resp, message):
        resp.result = Result(result=Result.RESULT_OPERATION_FAILED, error_message=message)
        return resp

    def _get_spawnables(self, req, resp):
        resp.spawnables = _spawnables(self._ctx.entities.all())
        # The service asks for unrecognised sources to be named without failing the call; this
        # simulator searches none, because everything spawnable was compiled in.
        sources = list(getattr(req, "sources", []))
        resp.result = Result(
            result=Result.RESULT_OK,
            error_message=f"sources not searched: {', '.join(sources)}" if sources else "",
        )
        return resp

    def _spawn_entity(self, req, resp):
        """Make an entity the world compiled perceivable again, at the pose the request states.

        Not creation. roqsim does not recompile the model at runtime, so there is no body to add:
        a world declares everything a trial may bring in, and this selects one. A request for a
        name the world does not carry is refused rather than approximated -- the alternative is
        a trial that believes it spawned something.

        ``initial_pose`` is applied, in the same physics transaction as the presence flip, so the
        entity is never perceivable at a pose nobody asked for. The service states that pose
        unconditionally and a default request carries the origin, so THAT is what a caller sending
        no pose asks for and what it gets; to bring an entity back where it was, state where that
        is. An entity the model welded cannot be moved at all, so for it the request succeeds only
        if it is already at the pose asked for, and is refused otherwise rather than appearing
        somewhere else under a RESULT_OK.
        """
        name, unsupported = _spawn_target(req, self._ctx.entities.names())
        unsupported = unsupported or _unsupported_spawn_request(req)
        if unsupported:
            code, message = unsupported
            resp.result = Result(result=code, error_message=message)
            return resp
        # ``initial_pose`` is stated unconditionally -- there is no "unset": a default request
        # carries the origin and, because ``geometry_msgs/Quaternion`` declares ``w 1``, the
        # identity. So every request asks for a pose, and this reads the one it asks for.
        pose, invalid = _pose_from(req.initial_pose.pose)
        if invalid:
            resp.result = Result(
                result=SpawnEntity.Response.INVALID_POSE, error_message=f"initial_pose: {invalid}"
            )
            return resp
        return self._set_presence(name, True, resp, verb="spawn", pose=pose)

    def _delete_entity(self, req, resp):
        """Make an entity absent: invisible to sensors, untouchable, and unlisted.

        Its pose does not move. Parking it out of sight instead would leave a free body
        accelerating under gravity for as long as it stays away, so it would return with
        whatever velocity it had accumulated. An entity already absent is not found
        (:func:`_absent`), as it is for every service that names one.
        """
        return self._set_presence(req.entity, False, resp, verb="delete")

    def _set_presence(self, name, present, resp, *, verb, pose=None):
        entity = self._ctx.entities.get(name)
        if entity is None:
            resp.result = Result(
                result=Result.RESULT_NOT_FOUND,
                error_message=(
                    f"no entity {name!r}. This simulator cannot create one: which entities "
                    "exist is settled when the model compiles, so the world must declare it."
                ),
            )
            return resp
        if not present and not entity.present:
            resp.result = _absent(name)
            return resp
        if present and entity.present:
            # A spawn of a present entity asks for a name a listed entity already holds, which
            # SpawnEntity.srv answers with its own NAME_NOT_UNIQUE (renaming is never allowed here).
            resp.result = Result(
                result=SpawnEntity.Response.NAME_NOT_UNIQUE,
                error_message=f"entity {name!r} is already present",
            )
            return resp
        # Physics-thread only, like every other write to model/data -- and WAITED FOR, so RESULT_OK
        # means the entity really has appeared. Posting and answering OK immediately would report
        # success before the flip has run, so a stalled simulator accepts spawns
        # that never happen and the caller has no way to tell.
        #
        # Pose first, then presence, in ONE transaction: placing an entity that is already
        # perceivable would show it at the compiled pose for the steps in between.
        #
        # Each outcome is recorded POSITIVELY, never inferred from a flag that stayed unset:
        # run_on_physics sets its event in a `finally`, so a raising command still returns True,
        # and reading "no outcome" as one particular failure would explain an exception as a
        # missing free joint.
        outcome = {}

        def _apply(ctx):
            if pose is not None and not place_body(ctx, entity, pose[0], pose[1]):
                state = self._read_body(ctx, entity.body) if entity.body else {}
                if not _already_at(state, pose):
                    outcome["welded_at"] = state.get("pos")
                    return
            set_present(ctx, entity, present)
            outcome["done"] = True

        if not run_on_physics(self._ctx, _apply):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"the simulation did not apply {verb} {name!r} within "
                f"{DEFAULT_TIMEOUT_S} s (is the physics thread stalled?)",
            )
            return resp
        if "welded_at" in outcome:
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=(
                    f"{verb} {name!r} asks for a pose the entity cannot take: the world compiled "
                    f"it as welded scenery at {outcome['welded_at']}, with neither a mocap body "
                    f"nor a free joint to write. Ask for that pose, or place it: "
                    f"{PLACEABLE_MODES_HINT}"
                ),
            )
            return resp
        if not outcome.get("done"):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"{verb} {name!r} did not complete in the simulation; "
                "its log carries the reason.",
            )
            return resp
        if hasattr(resp, "entity_name"):
            resp.entity_name = name
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _get_entity_state(self, req, resp):
        entity = self._ctx.entities.get(req.entity)
        if entity is None or entity.body is None:
            resp.result = Result(
                result=Result.RESULT_NOT_FOUND, error_message=f"no entity {req.entity!r}"
            )
            return resp
        if not entity.present:
            resp.result = _absent(req.entity)
            return resp
        holder = {}
        run_on_physics(self._ctx, lambda c: holder.update(self._read_body(c, entity.body)))
        if not holder:
            resp.result = Result(result=Result.RESULT_OPERATION_FAILED)
            return resp
        s = resp.state
        s.pose.position.x, s.pose.position.y, s.pose.position.z = holder["pos"]
        (s.pose.orientation.w, s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z) = (
            holder["quat"]
        )
        s.twist.linear.x, s.twist.linear.y, s.twist.linear.z = holder["lin"]
        s.twist.angular.x, s.twist.angular.y, s.twist.angular.z = holder["ang"]
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _set_entity_state(self, req, resp):
        entity = self._ctx.entities.get(req.entity)
        if entity is None:
            resp.result = Result(
                result=Result.RESULT_NOT_FOUND, error_message=f"no entity {req.entity!r}"
            )
            return resp
        # Not placed while absent: a pose written now would be one nobody can observe, and
        # spawn_entity states the pose an entity appears at.
        if not entity.present:
            resp.result = _absent(req.entity)
            return resp
        # The pose and twist are stated in the header's frame, and this simulator knows only the
        # world frame, which the empty default names.
        frame_id = _foreign_frame(req.state)
        if frame_id:
            resp.result = Result(
                result=SetEntityState.Response.INVALID_POSE,
                error_message=f"the state is stated in frame {frame_id!r}, which this simulator "
                "does not know. It places entities in the world frame, which the empty default "
                "already names.",
            )
            return resp
        pose, invalid = _pose_from(req.state.pose)
        if invalid:
            resp.result = Result(result=SetEntityState.Response.INVALID_POSE, error_message=invalid)
            return resp
        # The twist is part of the state: `EntityState` carries one and the GETTER reports one, so
        # answering RESULT_OK while ignoring whatever was asked for would let a caller read a
        # velocity it could not set. World frame, like the pose, and what the getter reports.
        #
        # Absent rather than zero when the request has no twist at all, which a real EntityState
        # always does but a partial caller may not: `None` is what `place_body` already reads as
        # "zero the velocity", so a pose write never fails for want of a field it does not use.
        twist = getattr(req.state, "twist", None)
        vel = None
        if twist is not None:
            tl, ta = twist.linear, twist.angular
            vel = (tl.x, tl.y, tl.z, ta.x, ta.y, ta.z)
        # The acceleration is ignored, which the service allows -- except that a non-zero one, like
        # a non-zero twist, fails for an entity that cannot move on from its pose.
        accel = getattr(req.state, "acceleration", None)
        acc = ()
        if accel is not None:
            al, aa = accel.linear, accel.angular
            acc = (al.x, al.y, al.z, aa.x, aa.y, aa.z)
        if not all(math.isfinite(v) for v in (*(vel or ()), *acc)):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message="the twist or acceleration carries a NaN or infinite component.",
            )
            return resp
        asked_motion = [
            field for field, values in (("twist", vel or ()), ("acceleration", acc)) if any(values)
        ]
        # Recorded POSITIVELY, and each failure told apart, exactly as the spawn door does it:
        # run_on_physics sets its event in a `finally`, so a command that raised still returns
        # True, and reading "no outcome" as one particular failure would explain an exception as
        # a missing free joint. A welded entity asked for the pose it already holds, at rest,
        # succeeds -- the caller got what it asked for, and only a MOVE is what a weld refuses.
        # A non-zero twist or acceleration on an entity without a free joint fails before
        # anything is written, as SetEntityState.srv requires of a static object: a mocap body is
        # held where it is put and welded scenery not even that, so neither can carry one.
        outcome = {}

        def _apply(ctx):
            if asked_motion and not carries_velocity(ctx, entity):
                outcome["static"] = True
                return
            if not place_body(ctx, entity, pose[0], pose[1], vel):
                state = self._read_body(ctx, entity.body) if entity.body else {}
                if not _already_at(state, pose):
                    outcome["welded_at"] = state.get("pos")
                    return
            outcome["done"] = True

        if not run_on_physics(self._ctx, _apply):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"the simulation did not place {req.entity!r} within "
                f"{DEFAULT_TIMEOUT_S} s (is the physics thread stalled?)",
            )
            return resp
        if "welded_at" in outcome:
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=(
                    f"{req.entity!r} cannot be moved: the world compiled it as welded scenery "
                    f"at {outcome['welded_at']}, with neither a mocap body nor a free joint to "
                    f"write. {PLACEABLE_MODES_HINT}"
                ),
            )
            return resp
        if outcome.get("static"):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=(
                    f"{req.entity!r} cannot take a non-zero {' or '.join(asked_motion)}: it has no "
                    "free joint, so it holds whatever pose it is given (a mocap body) or the one "
                    "the world compiled (welded scenery), and nothing was written. Send a zero "
                    "twist and acceleration, or give it 'motion: physics' in the world."
                ),
            )
            return resp
        if not outcome.get("done"):
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"placing {req.entity!r} did not complete in the simulation; "
                "its log carries the reason.",
            )
            return resp
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _get_sim_state(self, req, resp):
        resp.state = SimulationState(state=_STATE_TO_MSG[self._ctx.control.state])
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _set_sim_state(self, req, resp):
        target = _MSG_TO_STATE.get(req.state.state)
        if target is None:
            resp.result = Result(result=Result.RESULT_OPERATION_FAILED)
            return resp
        self._ctx.control.set_state(target)
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _step(self, req, resp):
        """Take the requested steps while paused, and answer once they have run.

        ``StepSimulation.srv`` returns "only once stepping is complete", and a caller reads the
        world straight after it: an answer before the steps ran would let it read the state from
        before them. The wait has no fixed budget, since a long request legitimately takes long;
        it fails when the simulation stops advancing (no step within
        :data:`~roqsim_ros_bridge.physics.DEFAULT_TIMEOUT_S`) or leaves the pause first.
        """
        control = self._ctx.control
        if control.state != ctl.PAUSED:
            resp.result = Result(
                result=Result.RESULT_INCORRECT_STATE, error_message="step requires PAUSED state"
            )
            return resp
        steps = int(req.steps)
        if steps < 1:
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"steps must be at least 1, got {steps}",
            )
            return resp
        done = threading.Event()
        outcome = {}

        def finished(completed: bool) -> None:
            outcome["completed"] = completed
            done.set()

        control.request_steps(steps, on_done=finished)
        last = self._ctx.sim_time
        while not done.wait(DEFAULT_TIMEOUT_S):
            now = self._ctx.sim_time
            if now == last:
                resp.result = Result(
                    result=Result.RESULT_OPERATION_FAILED,
                    error_message=f"the simulation took no step within {DEFAULT_TIMEOUT_S} s "
                    f"while {steps} were requested (is the physics thread stalled?)",
                )
                return resp
            last = now
        if not outcome["completed"]:
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message="the simulation left the paused state before the requested steps ran",
            )
            return resp
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def _reset(self, req, resp):
        """Reset the world, and answer once the reset has run (:meth:`on_reset`).

        Only a driver that honours the run control resets on request; under any other the reset
        never comes, and the call fails rather than answering ``RESULT_OK`` for a world that kept
        its state.
        """
        done = threading.Event()
        with self._reset_lock:
            self._reset_waiters.append(done)
        self._ctx.control.request_reset()
        if not done.wait(DEFAULT_TIMEOUT_S):
            with self._reset_lock:
                if done in self._reset_waiters:
                    self._reset_waiters.remove(done)
            resp.result = Result(
                result=Result.RESULT_OPERATION_FAILED,
                error_message=f"the simulation did not reset within {DEFAULT_TIMEOUT_S} s (is the "
                "physics thread stalled, or does this run's driver own resets?)",
            )
            return resp
        resp.result = Result(result=Result.RESULT_OK)
        return resp

    def on_reset(self, ctx) -> None:
        with self._reset_lock:
            waiters, self._reset_waiters = self._reset_waiters, []
        for done in waiters:
            done.set()

    # -- physics-thread helpers ---------------------------------------------------------------
    @staticmethod
    def _read_body(ctx, body_name):
        import mujoco

        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return {}
        twist = body_twist(ctx.model, ctx.data, bid)
        return {
            "pos": [float(v) for v in ctx.data.xpos[bid]],
            "quat": [float(v) for v in ctx.data.xquat[bid]],
            "ang": list(twist.angular),
            "lin": list(twist.linear),
        }

    def validate_config(self, config: dict) -> list[str]:
        return []
