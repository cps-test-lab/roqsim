"""ROS 2 transport bridge: wire the world's declared interface (``ctx.interface``) to ROS 2.

This is the ROS 2 backend of :class:`roqsim.bridge.BridgeBase`. It owns no per-robot knowledge:
the set of topics, their message types, frames, and rates all come from the :class:`~roqsim.
context.Endpoint`s that the robots' plugins register. Message classes are resolved from the type
*string* on each endpoint (via :mod:`roqsim_ros_bridge.registry`), so there are no hardcoded message
imports here and a new topic needs zero bridge edits -- just an endpoint on the producer.

One bridge serves the whole world: each endpoint carries its own ``namespace`` (declared by the
producer plugin), which this backend attaches to the endpoint's topic, TF frames, and action name.
Endpoints whose hint block carries ``action`` (an action type string) are served as action servers
via the handlers in :mod:`roqsim_ros_bridge.actions` (e.g. ``FollowJointTrajectory`` for arms).
Other packages add their own action handlers / message converters by advertising a module in the
``roqsim_ros_bridge.extensions`` entry-point group, which this bridge imports at start-up (see
:func:`roqsim_ros_bridge.extensions.load_extensions`) -- so serving e.g. nav2's
``NavigateThroughPoses`` needs no edit here and no nav2 dependency in this package.

Two things are ROS-intrinsic rather than robot endpoints and stay built in: ``/clock`` (this bridge is
the sim's time source; every other node runs with ``use_sim_time:=true``) and the dynamic ``tf``
(odom->base_link), which is derived from an ``odom`` endpoint whose hint sets ``emit_tf``.

Concurrency (see roqsim docs/architecture.rst §7): an ``rclpy`` MultiThreadedExecutor spins on a
worker thread; inbound subscriptions decode to a neutral payload and marshal the write onto the
physics thread via ``ctx.post``. Publishing happens in ``post_step`` on the physics thread (rclpy
publish is thread-safe), so the physics thread stays the sole writer of ``data``.

Config::

    ros2_bridge:
      namespace: ""            # optional GLOBAL outer prefix (node namespace) for the whole sim;
                               # per-robot scoping comes from each endpoint's own namespace
      node_name: roqsim_bridge
      clock_rate_hz: step      # "step" (default) = one /clock per physics step; 0 disables /clock;
                               # a number gates it, but see the commensurability rule below
      reuse_messages: true     # reuse one message object per topic (safe for inter-process subs)
      rates: {scan: 10.0}      # optional per-endpoint publish-rate overrides (by endpoint name)
      owner: null              # serve only these entities' endpoints (name or list; omit = all)
      domain_id: null          # run this bridge on its OWN ROS_DOMAIN_ID (its own rclpy Context), so
                               # ONE sim process can expose several robots on isolated domains -- run
                               # one ros2_bridge instance per robot, each with its owner + domain_id
      strip_namespace: null    # namespace(s) to DROP from topics/frames so the served robot presents
                               # CLEAN local names (/odom, base_link) on its domain -- a name or list.
                               # Non-stripped owners (e.g. walkers) keep their prefixed frames.

**The /clock grid must divide every gated publish period.** A subscriber running on sim time cannot
resolve an event finer than the last ``/clock`` it received, so ``/clock``'s own period is the grid
every stamp -- and every rosbag receive time -- snaps to. When a publisher's period is not a whole
number of grid steps, its arrival times alias: an 18 ms ground-truth pose on a 10 ms grid arrives at
an alternating 20/20/20/20/10 ms, which reads as a robot whose speed doubles every fifth sample even
though it is moving at a constant rate. That is why the default is one tick per physics step -- the
finest grid available, and one that divides every gated period by construction. Set a rate only with
the arithmetic in hand; ``configure()`` warns for each output that does not divide.

Multi-robot pattern: one MuJoCo world with prefixed robots (``turtlebot/``, ``spot/``) and two
instances -- ``{owner: [turtlebot, ped_a, ped_b], domain_id: 1, strip_namespace: turtlebot}`` and
``{owner: spot, domain_id: 2, strip_namespace: spot}`` -- gives each robot a clean single-robot ROS
graph on its own domain (nav2 needs no namespacing), all from the one shared physics world.

Merged ``joint_states``: a robot with several controllers declares one ``joint_states`` endpoint
per controller, and where those are scoped apart by namespace nothing publishes the topic a
``robot_state_publisher`` / ``move_group`` over their combined ``robot_description`` listens on. The
bridge closes that with one extra merged publisher per group, in addition to each endpoint's own
topic. Which endpoints form a group is DECLARED (``merged_joint_states``), because it is a fact about
the stack's ``robot_description``s, not about the world: ``"auto"`` (default) groups by entity, so
several arms on one robot merge and separate robots stay separate; ``true`` merges everything this
bridge serves into one ``/joint_states`` (two entities, one combined description); ``false`` disables
it; a list of ``{topic, owners}`` states the groups outright. See ``_joint_state_groups``.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock as ClockMsg
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from roqsim.bridge import BridgeBase, _RateGate
from roqsim_ros_bridge import registry as reg
from roqsim_ros_bridge.actions import get_action_handler
from roqsim_ros_bridge.extensions import load_extensions
from roqsim_ros_bridge.services import get_service_handler

#: ``clock_rate_hz`` value meaning "one tick per physics step" -- the default. Spelled rather than
#: numeric because 0 already means "no /clock at all" here, unlike _RateGate's rate<=0.
_CLOCK_EVERY_STEP = "step"

#: The backend type string a joint-state-producing endpoint declares. Endpoints of this type are
#: candidates for the merged ``/joint_states`` publisher below.
_JOINT_STATE_TYPE = "sensor_msgs.msg.JointState"


def _merge_joint_state_payloads(payloads: list[tuple]) -> tuple:
    """Concatenate several ``(names, positions, velocities[, efforts])`` payloads into one.

    Order follows the order ``payloads`` is given in (the endpoints' own registration order), which
    is what lets a caller line this up against another consumer of the same order -- e.g. a URDF
    exporter that also reads the endpoints in registration order.

    Effort is included only if at least one source reports it (real drivers do; a wheel/locomotion
    source may not), and a source that omits it is padded with zeros rather than dropped, so every
    joint still has an effort entry once any of them does -- a message can't state effort for some
    joints and omit it for others.
    """
    names: list = []
    positions: list = []
    velocities: list = []
    efforts: list = []
    any_effort = False
    for payload in payloads:
        n, p, v, *rest = payload
        names.extend(n)
        positions.extend(p)
        velocities.extend(v)
        if rest:
            efforts.extend(rest[0])
            any_effort = True
        else:
            efforts.extend([0.0] * len(n))
    return (names, positions, velocities, efforts) if any_effort else (names, positions, velocities)


def _common_ns(namespaces) -> str:
    """The deepest namespace every one of ``namespaces`` sits in ("" if they share nothing)."""
    parts = [ns.strip("/").split("/") if ns.strip("/") else [] for ns in namespaces]
    common: list[str] = []
    # zip stops at the shortest, which is exactly the depth a common prefix can reach.
    for segments in zip(*parts, strict=False):
        if len(set(segments)) > 1:
            break
        common.append(segments[0])
    return "/".join(common)


def _gate_period(gate: _RateGate, dt: float) -> float:
    """The period a gate ACTUALLY fires at, which is its requested period rounded UP to the physics
    grid -- ``due()`` is only ever evaluated at step boundaries. A 60 Hz gate on a 2 ms step fires
    every 9 steps (18 ms), not every 16.67 ms, and it is the 18 that has to divide the /clock grid.
    """
    if gate.rate_hz <= 0.0:
        return dt
    return math.ceil((1.0 / gate.rate_hz - 1e-9) / dt) * dt


def _join_ns(*parts: str) -> str:
    """Join namespace/prefix parts, skipping empties (no leading/trailing slashes)."""
    return "/".join(p for p in parts if p)


def _resolve_topic(namespace: str, topic: str) -> str:
    """Resolve an endpoint's ROS topic, honouring an absolute hardwired topic.

    A ``topic`` beginning with ``/`` is *absolute*: it is used verbatim, so it bypasses the
    endpoint's ``namespace`` (and rclpy leaves absolute names untouched by the node namespace too).
    This is how a producer hardwires a topic to match external/hardware names -- e.g. a camera under
    ``namespace: ur10e`` publishing exactly ``/camera/color/image_raw`` (see ``Plugin.topic_override``
    / the ``topics:`` config). A relative topic is scoped under the namespace as before.
    """
    return topic if topic.startswith("/") else _join_ns(namespace, topic)


@dataclass
class _Pub:
    publisher: Any
    msg_type: Any
    convert: Any  # fill(msg, payload, stamp, hints)
    hints: dict
    msg: Any  # reused message instance, or None when reuse is disabled
    emit_tf: bool


class _NamespacedTfPublisher:
    """A drop-in for tf2_ros's broadcasters that publishes TF to a namespaced *topic* — ``/<ns>/tf``
    and ``/<ns>/tf_static`` — while leaving frame ids untouched.

    Why this exists: tf2_ros's ``TransformBroadcaster`` / ``StaticTransformBroadcaster`` hardwire the
    absolute ``/tf`` and ``/tf_static`` and expose no topic override, so a single robot's TF always
    lands on the global tree. Stacks that follow the nav2 multi-robot convention (each robot's nav2 is
    launched with the ``/tf -> tf`` remap, so TF lives under ``/<robot>/tf``) then cannot see it, and
    tooling that assumes that convention — e.g. scenario_execution's ``NamespacedTransformListener``,
    which subscribes ``<namespace>/tf`` — hangs waiting for a transform that is being published one
    topic over. This publisher lets the bridge match that convention when asked (``tf_namespace``),
    without namespacing the node (which would also scope ``/scan``, ``/odom``, ... and break the
    stack's topic remaps).

    For static transforms it accumulates by child frame and republishes the whole latched set on every
    send, exactly as ``StaticTransformBroadcaster`` does, so a late subscriber on the transient-local
    topic still receives every mount transform rather than only the last one sent.
    """

    def __init__(self, node: Node, topic: str, *, static: bool) -> None:
        qos = (
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            if static
            else QoSProfile(depth=100)
        )
        self._pub = node.create_publisher(TFMessage, topic, qos)
        self._static = static
        self._latched: dict[str, Any] = {}

    def sendTransform(self, transform) -> None:  # noqa: N802 — match the tf2_ros broadcaster API
        transforms = transform if isinstance(transform, list) else [transform]
        if self._static:
            for t in transforms:
                self._latched[t.child_frame_id] = t
            transforms = list(self._latched.values())
        self._pub.publish(TFMessage(transforms=transforms))


class Ros2Bridge(BridgeBase):
    BACKEND = "ros2"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        config = dict(config or {})
        # Optional GLOBAL prefix (the rclpy node namespace): scopes the whole sim, e.g. /sim1/...
        # Per-robot scoping is NOT set here -- it rides on each endpoint's own `namespace`.
        self._namespace = config.get("namespace", "")
        self._frame_prefix = config.get("frame_prefix", self._namespace)
        # Optional TOPIC namespace for /tf and /tf_static only (not frames, not other topics): when set,
        # TF is published on /<tf_namespace>/tf(_static) instead of the global /tf. This matches the
        # nav2 per-robot convention (/tf -> tf remap) that scenario_execution and namespaced Nav2
        # bringups expect. Frame ids are unchanged — use frame_prefix for those. See _NamespacedTfPublisher.
        self._tf_namespace = str(config.get("tf_namespace", "")).strip("/")
        # Optional own ROS_DOMAIN_ID (an isolated rclpy Context) so several bridges can run in one sim
        # process, one per robot domain. None => the default context / env ROS_DOMAIN_ID (single-bridge).
        self._domain_id = config.get("domain_id")
        # Namespace(s) to strip so the served robot presents clean local names on its domain.
        strip = config.get("strip_namespace")
        self._strip: set[str] = (
            set() if strip is None else ({strip} if isinstance(strip, str) else set(strip))
        )
        # Ground-truth topic namespace. When this sim acts as the `/gt` ground-truth system (see
        # docs/ground_truth.rst), published *output* topics get `gt.prefix` (e.g. /gt) so consumers can
        # tell true poses from real perception -- EXCEPT topics in `gt.exempt`, which stay canonical.
        # The rule: prefix = a pure-GT stream with no real equivalent (the object's true /gt/tf); exempt
        # = a stream that mirrors a real topic (robot telemetry, or perception's own message name).
        gt = config.get("gt") or {}
        self._gt_prefix = str(gt.get("prefix", "")).rstrip("/")
        self._gt_exempt: set[str] = set(gt.get("exempt") or [])
        super().__init__(config, name=name, entity=entity, label=label)
        self._context: Context | None = None
        self._node: Node | None = None
        self._action_servers: list[ActionServer] = []
        self._services: list = []
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._we_inited_rclpy = False
        self._tf: TransformBroadcaster | None = None
        self._static_tf: StaticTransformBroadcaster | None = None
        self._clock_pub = None
        self._clock_msg = ClockMsg()
        # One (publisher, gate, endpoints) per merged joint-state group (see
        # _setup_merged_joint_states); empty when nothing needs merging, which is the single-robot,
        # single-controller case.
        self._merged_joint_states: list[tuple[_Pub, _RateGate, list]] = []
        # "step" (the default) publishes one tick per physics step: the finest grid available, and
        # the only setting that divides every gated period whatever they are. 0 keeps its older
        # meaning of "no /clock at all", which is why "every step" needed a name of its own rather
        # than reusing _RateGate's rate<=0.
        clock_rate = self.config.get("clock_rate_hz", _CLOCK_EVERY_STEP)
        self._clock_enabled = clock_rate == _CLOCK_EVERY_STEP or float(clock_rate) > 0.0
        self._clock_gate = _RateGate(-1.0 if clock_rate == _CLOCK_EVERY_STEP else float(clock_rate))
        self._reuse = bool(self.config.get("reuse_messages", True))
        # Whether to publish producers' fixed sensor-mount transforms on /tf_static. Turn off in a
        # world that runs a robot_state_publisher over the robot's URDF, which publishes the same
        # links itself (see the emit site in _make_publisher).
        self._publish_static_tf = bool(self.config.get("publish_static_tf", True))

    def _eff_ns(self, ep) -> str:
        """The endpoint's effective namespace for topic/frame scoping — ``""`` if it is stripped."""
        return "" if ep.namespace in self._strip else ep.namespace

    def configure(self, ctx) -> None:
        # After super(), because the check reads the gates _bind() built, and the merge needs
        # ctx.interface fully populated -- true only once _bind() (called by super()) has run.
        super().configure(ctx)
        self._warn_on_clock_aliasing(ctx)
        self._setup_merged_joint_states(ctx)

    def _joint_state_endpoints(self, ctx) -> list:
        """Every joint-state output endpoint this bridge instance serves, in registration order."""
        out = []
        for ep in ctx.interface.all():
            if ep.direction != "out":
                continue
            hints = ep.backend.get(self.BACKEND)
            if hints is None or hints.get("type") != _JOINT_STATE_TYPE:
                continue
            if self._owners is not None and ep.owner not in self._owners:
                continue
            out.append(ep)
        return out

    def _ep_topic(self, ep) -> str:
        """The topic an endpoint publishes on by itself -- what a merged group must not duplicate."""
        hints = ep.backend.get(self.BACKEND, {})
        return self._gt_topic(_resolve_topic(self._eff_ns(ep), hints.get("topic", ep.name)))

    def _joint_state_groups(self, sources: list, logger=None) -> list[tuple[str, list]]:
        """Which joint-state endpoints belong on ONE merged topic, and what that topic is.

        The grouping is a statement about the ROS graph -- how many ``robot_description``s the stack
        runs -- which the world model alone cannot answer: two arms may be two independent robots,
        each with its own ``robot_state_publisher`` and ``move_group``, or two planning groups of one
        combined description (what ``arm_controller``'s ``joint_prefix`` exists for). So it is
        declared, via ``merged_joint_states``:

        * ``"auto"`` (the default) groups by ``owner``, i.e. by spawned entity: several controllers on
          one robot (a dual-arm entity, an arm plus a torso) merge into that robot's own scope, and
          separate robots stay separate. Safe without any declaration, because endpoints of one entity
          describe one physical robot whatever the stack above does.
        * ``true`` merges every endpoint this bridge serves into one ``/joint_states`` -- the combined
          ``robot_description`` case, across entities.
        * ``false`` disables merging.
        * A list of ``{topic: ..., owners: [...]}`` states the groups outright, for a world whose
          descriptions cut across entities in some other way.

        A group whose endpoints already resolve to ONE topic needs no merged publisher: they meet on
        the wire, and both ``robot_state_publisher`` and MoveIt's current-state monitor accumulate
        partial joint states. Only a group scoped apart (each endpoint under its own namespace) needs
        one.
        """
        decl = self.config.get("merged_joint_states", "auto")
        if decl is False:
            return []
        if decl is True:
            return [(self._gt_topic("joint_states"), sources)]
        if isinstance(decl, list):
            groups = []
            for entry in decl:
                owners = set(entry["owners"])
                members = [ep for ep in sources if ep.owner in owners]
                if members:
                    groups.append((self._gt_topic(entry.get("topic", "joint_states")), members))
            return groups
        if decl != "auto":
            raise ValueError(
                f"ros2 bridge: merged_joint_states must be 'auto', true, false or a list of "
                f"{{topic, owners}} groups, got {decl!r}"
            )
        by_owner: dict[str, list] = {}
        for ep in sources:
            by_owner.setdefault(ep.owner, []).append(ep)
        if len(by_owner) > 1 and logger is not None:
            logger.warning(
                "ros2 bridge: %d entities publish joint states here; each is merged into its own "
                "scope. If they share ONE robot_description, declare it -- merged_joint_states: true "
                "(one /joint_states across all of them) or a list of {topic, owners} groups -- "
                "otherwise a combined description gets no TF and move_group no current state.",
                len(by_owner),
            )
        groups = []
        for members in by_owner.values():
            # The merged topic sits in the scope the group shares -- the deepest namespace common
            # to its endpoints, which is the robot's own scope when its controllers are namespaced
            # below it (``dual/left``, ``dual/right`` -> ``dual``). Nothing in common puts it at the
            # root, where an unnamespaced robot's stack looks for it anyway.
            groups.append(
                (
                    self._gt_topic(
                        _join_ns(_common_ns(self._eff_ns(ep) for ep in members), "joint_states")
                    ),
                    members,
                )
            )
        return groups

    def _setup_merged_joint_states(self, ctx) -> None:
        """Publish one merged ``joint_states`` per declared group (see :meth:`_joint_state_groups`),
        in ADDITION to each endpoint's own topic, which keeps publishing unchanged.

        MoveIt's convention -- and this substrate's own exporter contract (see
        ``roqsim.export_moveit``, which documents the joint names it emits as "the ones that reach
        ``/joint_states``") -- is one ``robot_description`` implies one merged joint-state stream. A
        robot with several controllers declares one ``joint_states`` endpoint PER controller, and
        where those are scoped apart by namespace nothing publishes the topic
        ``robot_state_publisher`` and MoveIt's planning-scene monitor actually listen on. This closes
        that gap without guessing across robots: what belongs together is declared, and the default
        groups by entity, which is true independently of the stack.
        """
        sources = self._joint_state_endpoints(ctx)
        if len(sources) < 2:
            return
        msg_type = reg.resolve_type(_JOINT_STATE_TYPE)
        for topic, members in self._joint_state_groups(sources, getattr(ctx, "logger", None)):
            if len(members) < 2 or len({self._ep_topic(ep) for ep in members}) < 2:
                continue
            publisher = self._node.create_publisher(msg_type, topic, 10)
            handle = _Pub(
                publisher=publisher,
                msg_type=msg_type,
                convert=reg.get_converter(_JOINT_STATE_TYPE),
                hints={},
                msg=msg_type() if self._reuse else None,
                emit_tf=False,
            )
            gate = _RateGate(max(ep.rate_hz for ep in members))
            self._merged_joint_states.append((handle, gate, members))

    def _publish_merged_joint_states(self, stamp, t: float) -> None:
        for handle, gate, members in self._merged_joint_states:
            if not gate.due(t):
                continue
            payloads = [p for ep in members if (p := ep.read()) is not None]
            if payloads:
                self._publish(handle, _merge_joint_state_payloads(payloads), stamp)

    def _warn_on_clock_aliasing(self, ctx) -> None:
        """Warn for each output whose publish period is not a whole number of /clock ticks.

        Such an output arrives on an alternating spacing rather than a steady one -- not jitter, a
        systematic alias -- and every consumer that differentiates it (a speed, a rate, anything per
        second) reads that alias as the robot's behaviour. It is invisible in the data: the
        displacement between samples is constant and correct, only the timestamps are wrong, so the
        run looks healthy and the metric does not. Cheap to say here, expensive to find later.
        """
        if not self._clock_enabled:
            return
        grid = _gate_period(self._clock_gate, ctx.dt)
        grid_steps = round(grid / ctx.dt)
        for out in self._outputs:
            steps = round(_gate_period(out.gate, ctx.dt) / ctx.dt)
            if steps % grid_steps:
                ctx.logger.warning(
                    "bridge: %r publishes every %.4g s but /clock ticks every %.4g s, which does "
                    "not divide it -- its stamps will alternate between %.4g s and %.4g s apart. "
                    "Set clock_rate_hz: %s, or pick a rate whose period is a multiple of the grid.",
                    out.endpoint.name,
                    steps * ctx.dt,
                    grid,
                    (steps // grid_steps + 1) * grid,
                    (steps // grid_steps) * grid,
                    _CLOCK_EVERY_STEP,
                )

    # -- backend hooks ----------------------------------------------------------------------------
    def _setup(self, ctx) -> None:
        # Import other packages' action handlers / converters before any endpoint is wired, so a
        # producer's action or message type resolves regardless of which package supplies it.
        load_extensions()
        node_name = self.config.get("node_name", "roqsim_bridge")
        if self._domain_id is not None:
            # Own isolated context on a specific ROS_DOMAIN_ID (multi-robot: one bridge per domain).
            self._context = Context()
            rclpy.init(context=self._context, domain_id=int(self._domain_id))
            self._we_inited_rclpy = True
            node = Node(node_name, namespace=self._namespace, context=self._context)
            self._executor = MultiThreadedExecutor(context=self._context)
        else:
            if not rclpy.ok():
                rclpy.init()
                self._we_inited_rclpy = True
            node = Node(node_name, namespace=self._namespace)
            self._executor = MultiThreadedExecutor()
        self._node = node
        # Share the node so a co-loaded sim_interfaces plugin reuses it (one executor).
        ctx.blackboard.set("ros2_node", node)

        if self._clock_enabled:
            # Absolute name: /clock is the domain's time source, never namespaced.
            self._clock_pub = node.create_publisher(ClockMsg, "/clock", 10)

        self._executor.add_node(node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _make_tf_broadcaster(self, *, static: bool):
        """A TF broadcaster whose topic is namespaced when ``tf_namespace`` is set, and the plain
        tf2_ros broadcaster (global ``/tf``) otherwise — so existing worlds are byte-for-byte
        unchanged and only a stack that opts in gets ``/<ns>/tf``."""
        if self._tf_namespace:
            topic = f"/{self._tf_namespace}/{'tf_static' if static else 'tf'}"
            return _NamespacedTfPublisher(self._node, topic, static=static)
        return (StaticTransformBroadcaster if static else TransformBroadcaster)(self._node)

    def _gt_topic(self, topic: str) -> str:
        """Apply the ground-truth namespace to an output topic: prefix with ``gt.prefix`` unless the
        topic is exempt (canonical). No-op when no ``gt.prefix`` is configured. The result is absolute
        so it is unaffected by the node namespace."""
        if not self._gt_prefix or topic in self._gt_exempt:
            return topic
        return self._gt_prefix + (topic if topic.startswith("/") else "/" + topic)

    def _make_output(self, ep, hints: dict) -> _Pub:
        msg_type = reg.resolve_type(hints["type"])
        # The endpoint's own namespace scopes its topic (relative, so any global node namespace
        # still applies on top): ep.namespace="ur10e" -> /ur10e/joint_states. An absolute hardwired
        # topic (leading "/") is used verbatim, bypassing the namespace (see _resolve_topic).
        topic = self._gt_topic(_resolve_topic(self._eff_ns(ep), hints.get("topic", ep.name)))
        qos = int(hints.get("qos", 10))
        publisher = self._node.create_publisher(msg_type, topic, qos)
        # Let an expensive producer (e.g. a rendered camera) skip work when nobody's listening --
        # generic, not camera-specific; cheap endpoints (lidar, odom) just never check it.
        ep.has_subscribers = lambda p=publisher: p.get_subscription_count() > 0
        emit_tf = bool(hints.get("emit_tf", False))
        if emit_tf and self._tf is None:
            self._tf = self._make_tf_broadcaster(static=False)
        # Carry the namespace-derived frame prefix into the converter so frame ids are namespaced
        # (empty when this endpoint's namespace is stripped, so its frames are clean, e.g. base_link).
        frame_prefix = _join_ns(self._frame_prefix, self._eff_ns(ep))
        hints = {**hints, "frame_prefix": frame_prefix}
        # A producer may ship a fixed sensor-mount transform (base -> its frame) as plain numbers;
        # publish it once on the latched /tf_static so consumers get the frame without a URDF/RSP.
        #
        # Set ``publish_static_tf: false`` when the world DOES run a robot_state_publisher over the
        # robot's URDF: that publishes the same base -> sensor links from the model, and two
        # publishers for one static transform is a TF conflict rather than redundancy. The default
        # stays true, because a world without an RSP has no other source for these frames.
        st = hints.get("static_tf") if self._publish_static_tf else None
        if st:
            if self._static_tf is None:
                self._static_tf = self._make_tf_broadcaster(static=True)
            parent = reg.namespaced(frame_prefix, st["parent"])
            child = reg.namespaced(frame_prefix, hints["frame_id"])
            self._static_tf.sendTransform(
                reg.make_static_tf(
                    reg.to_time_msg(0.0), parent, child, st["translation"], st["rotation"]
                )
            )
        return _Pub(
            publisher=publisher,
            msg_type=msg_type,
            convert=reg.get_converter(hints["type"]),
            hints=hints,
            msg=msg_type() if self._reuse else None,
            emit_tf=emit_tf,
        )

    def _make_input(self, ep, hints: dict, on_payload) -> None:
        # A `service` hint means this input is a command whose OUTCOME the caller needs: serve a
        # service whose reply policy comes from the same handler registry the actions use. A topic
        # cannot say whether the command took effect, and an action's feedback and cancellation are
        # machinery an instantaneous command has no use for.
        if "service" in hints:
            srv_type = reg.resolve_type(hints["service"])
            handler = get_service_handler(hints["service"])
            self._services.append(
                self._node.create_service(
                    srv_type,
                    _join_ns(self._eff_ns(ep), hints.get("name", ep.name)),
                    # The endpoint rides along so the handler can resolve its producer's state
                    # (its `state_key` hint) without knowing which producer it is serving.
                    lambda req, resp, e=ep: handler(req, resp, self._ctx, on_payload, e),
                    callback_group=ReentrantCallbackGroup(),
                )
            )
            return
        # An `action` hint means this input is a goal-driven interaction, not a stream: serve an
        # ActionServer whose execution policy comes from the handler registry (roqsim_ros_bridge.actions).
        if "action" in hints:
            action_type = reg.resolve_type(hints["action"])
            handler = get_action_handler(hints["action"])
            self._action_servers.append(
                ActionServer(
                    self._node,
                    action_type,
                    _join_ns(self._eff_ns(ep), hints.get("name", ep.name)),
                    # The endpoint rides along so the handler can resolve its producer's state
                    # (e.g. ctx.blackboard.get(f"walker:{endpoint.owner}")) without a hardcoded key.
                    execute_callback=lambda gh, e=ep: handler(gh, self._ctx, on_payload, e),
                    goal_callback=lambda _g: GoalResponse.ACCEPT,
                    cancel_callback=lambda _g: CancelResponse.ACCEPT,
                    callback_group=ReentrantCallbackGroup(),
                )
            )
            return
        msg_type = reg.resolve_type(hints["type"])
        topic = _resolve_topic(self._eff_ns(ep), hints.get("topic", ep.name))
        qos = int(hints.get("qos", 10))
        decode = reg.get_decoder(hints["type"])
        self._node.create_subscription(msg_type, topic, lambda m: on_payload(decode(m)), qos)

    def _shutting_down(self) -> bool:
        """True once rclpy has invalidated our context -- i.e. the process is on its way out.

        A shutting-down context is not an error. On SIGINT/SIGTERM rclpy invalidates the context from
        its signal handler, but the physics loop owns the thread and finishes the step it is in, so
        the next publish lands on a dead context and raises. Left to propagate, that aborts the
        process with an RCLError traceback which reads exactly like a mid-run crash -- it was
        repeatedly misdiagnosed as one, while the run had in fact completed and was being torn down.
        Publishes are skipped from here on; anything else still raises.
        """
        return not (self._context.ok() if self._context is not None else rclpy.ok())

    def _publish(self, handle: _Pub, payload, stamp) -> None:
        if self._shutting_down():
            return
        msg = handle.msg if handle.msg is not None else handle.msg_type()
        handle.convert(msg, payload, stamp, handle.hints)
        handle.publisher.publish(msg)
        if handle.emit_tf and self._tf is not None:
            self._tf.sendTransform(
                reg.make_tf(
                    payload,
                    stamp,
                    reg.frame(handle.hints, "frame_id", "odom"),
                    reg.frame(handle.hints, "child_frame_id", "base_link"),
                )
            )

    def _now(self, t: float):
        return reg.to_time_msg(t)

    def _tick(self, ctx, t: float, stamp) -> None:
        if self._shutting_down():
            return
        if self._clock_pub is not None and self._clock_gate.due(t):
            self._clock_msg.clock = stamp
            self._clock_pub.publish(self._clock_msg)
        if self._merged_joint_states:
            self._publish_merged_joint_states(stamp, t)

    def on_reset(self, ctx) -> None:
        super().on_reset(ctx)
        self._clock_gate.reset()
        for _, gate, _ in self._merged_joint_states:
            gate.reset()

    def _teardown(self, ctx) -> None:
        for server in self._action_servers:
            server.destroy()
        for service in self._services:
            self._node.destroy_service(service)
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._we_inited_rclpy:
            if self._context is not None:
                if self._context.ok():
                    rclpy.shutdown(context=self._context)
            elif rclpy.ok():
                rclpy.shutdown()
