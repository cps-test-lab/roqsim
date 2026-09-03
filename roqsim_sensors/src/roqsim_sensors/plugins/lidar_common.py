"""Shared machinery for every ray-casting range sensor: 2D laser scanners and 3D lidars alike.

:class:`RayCastSensorPlugin` owns everything the devices had in common and had drifted apart on --
config keys and their validation, site/``exclude_body`` resolution, the reusable ray buffers, the
static mount TF, the ``rate_hz`` gate, the range window, the noise model, and endpoint registration.
A device then declares only what actually distinguishes it:

* :meth:`~RayCastSensorPlugin._build_directions` -- its ray pattern, in the site frame.
* :meth:`~RayCastSensorPlugin._payload` -- its wire type (``LaserScan`` vs ``PointCloud``).
* a handful of ``DEFAULT_*`` class attributes -- its datasheet.

This mirrors how :mod:`camera_common` + :mod:`depth_camera` already layer the cameras, and it exists
for the same reason: the duplicated copies had diverged in ways that were bugs rather than choices.
Two are fixed by being written once here.

**A near return is either clamped or dropped, and that is a device property, not an accident.** A
``LaserScan`` is a fixed-length array, so a return inside ``range_min`` is clamped up to it and the
array keeps its shape; a point cloud is a list of real returns, so a blind-zone return is simply not
a point. :data:`RayCastSensorPlugin.CLAMP_NEAR_RETURNS` names that difference instead of leaving it
implicit in two hand-written expressions.

**``max_range`` is enforced here, for everyone.** ``mj_multiRay``'s ``cutoff`` is a culling hint and
not a clamp -- it can still report a hit beyond it. The 2D lidar had always applied the window; the
3D lidars had not, so a Mid-360 with a 40 m range could emit points from further away.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim import raycast
from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

from ..live_config import FaultableSensorMixin

#: MuJoCo's own name for body 0, and the parent of a mount transform that has no body to hang from.
#: See :meth:`RayCastSensorPlugin._mount_tf`.
WORLD_FRAME = "world"


class RayCastSensorPlugin(FaultableSensorMixin, Plugin):
    """Base for a ``post_step`` range sensor built on :func:`roqsim.raycast.cast`."""

    parallel_safe = True  # post_step only reads data + writes its own payload buffer

    #: Endpoint role name, and the key a world's ``topics:`` map overrides it by.
    ENDPOINT_NAME = "scan"
    #: Backend-neutral payload type the bridge resolves. No ROS import here.
    ROS_TYPE = "sensor_msgs.msg.LaserScan"
    #: Topic used when the world declares no ``topics:`` override.
    DEFAULT_TOPIC = "scan"
    #: Name this plugin reports itself under in errors, so a subclass says its own.
    PLUGIN_LABEL = "lidar"

    DEFAULT_SITE = "lidar"
    DEFAULT_RANGE_MIN = 0.164
    DEFAULT_MAX_RANGE = 20.0
    DEFAULT_RATE_HZ = 10.0
    DEFAULT_EXCLUDE_BODY = "base_link"

    #: True for a fixed-length scan (clamp a blind-zone return up to ``range_min``), False for a
    #: point cloud (drop it). See the module docstring.
    CLAMP_NEAR_RETURNS = True

    #: Keys a ``fault:`` block may write WHILE THE RUN IS IN PROGRESS -> the attribute each lives in.
    #: Every row is read inside ``post_step`` on the frame it is used (see the noise block at the end
    #: of this file), so a write takes effect on the very next cast and reads back honestly.
    #: ``max_range`` -> ``range_max`` because the config key and the attribute have never had the
    #: same name, and a fault naming the attribute would silently write nothing.
    LIVE_WRITABLE = {
        "range_stddev": "range_stddev",
        "dropout_percent": "dropout_percent",
        "max_range": "range_max",
        "range_min": "range_min",
        "rate_hz": "rate_hz",
    }

    #: Refused by name, with the reason, rather than left to the undeclared-key message -- these are
    #: the keys someone reaches for first. Each is consumed once, at ``configure``, and baked into a
    #: buffer, an id or a frame name: writing it later changes nothing while reading back as though
    #: it had, which is what ``geom_size`` is refused for on the physics channel.
    REFUSED_WRITES = {
        "site": "it is resolved to a site id at configure, so a later write moves no rays.",
        "frame_id": "it is stamped into the payload and the static mount TF at configure; changing "
        "it mid-run would relabel frames a consumer has already built a TF tree from.",
        "exclude_body": "it is resolved to a body id at configure.",
        "emit_static_tf": "the static TF is published once, at configure.",
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.site = self.config.get("site", self.DEFAULT_SITE)
        # ROS frame the payload is stamped in, and the child of the static mount TF (one value, so
        # the two cannot disagree). Defaults to the site the rays are actually cast from; a model
        # whose real description names the frame differently declares it in its manifest (e.g. the
        # TurtleBot 4's URDF calls it `rplidar_link`, Livox's driver `livox_frame`). Hardwired per
        # plugin instead, one robot's scan goes out stamped in another robot's frame.
        self.frame_id = self.config.get("frame_id", self.site)
        self.range_min = float(self.config.get("range_min", self.DEFAULT_RANGE_MIN))
        self.range_max = float(self.config.get("max_range", self.DEFAULT_MAX_RANGE))
        # Cast rate: rays are cast (and the payload published) at this rate, not every physics step.
        # Casting every step at e.g. 500 Hz is ~50x more work than any consumer asked for.
        self.rate_hz = float(self.config.get("rate_hz", self.DEFAULT_RATE_HZ))
        self._last_cast = float("-inf")
        self.exclude_body = self.config.get("exclude_body", self.DEFAULT_EXCLUDE_BODY)
        self.range_stddev = float(self.config.get("range_stddev", 0.0))
        self.dropout_percent = float(self.config.get("dropout_percent", 0.0))
        # Publish base body -> sensor frame as a static TF (derived from the same site the rays are
        # cast from). On by default; disable when an external robot_state_publisher owns it.
        self.emit_static_tf = bool(self.config.get("emit_static_tf", True))
        self._site_id = -1
        self._bodyexclude = -1
        self._local_dirs: np.ndarray | None = None  # (nray, 3) unit directions, site frame
        self._hits: raycast.RayHits | None = None
        self._payload_value = None  # latest payload, read by the endpoint
        # Faulted values + the switch. After the attributes above, since it reads them for nominal.
        self._fault_init()

    @property
    def latest(self):
        """The most recent payload -- exactly what this plugin's output endpoint serves.

        A read accessor rather than a private buffer, because a caller that holds the *plugin* (a
        test, an embedding driver) cannot always go through the endpoint: several sensors on one
        ``SimContext`` all register a ``scan``/``cloud`` endpoint, so "the scan endpoint" does not
        identify which one computed it.
        """
        return self._payload_value

    # -- subclass contract --------------------------------------------------------------------

    @property
    def num_rays(self) -> int:
        raise NotImplementedError

    def _build_directions(self) -> np.ndarray:
        """(nray, 3) unit ray directions in the site frame."""
        raise NotImplementedError

    def _payload(self, dist: np.ndarray, valid: np.ndarray):
        """Build the wire payload from per-ray ``dist`` and its ``valid`` mask.

        ``dist`` is metres along each ray (meaningless where ``valid`` is False) and already carries
        the range window, the near-return policy and any noise.
        """
        raise NotImplementedError

    def _validate_extra(self, config: dict) -> list[str]:
        """Device-specific config errors, appended to the shared ones."""
        return []

    def _ros2_hints_extra(self) -> dict:
        return {}

    # -- lifecycle ----------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("max_range", self.DEFAULT_MAX_RANGE)) <= 0:
            errors.append("'max_range' must be > 0")
        if float(config.get("range_min", self.DEFAULT_RANGE_MIN)) < 0:
            errors.append("'range_min' must be >= 0")
        if float(config.get("rate_hz", self.DEFAULT_RATE_HZ)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("range_stddev", 0.0)) < 0:
            errors.append("'range_stddev' must be >= 0")
        if not 0.0 <= float(config.get("dropout_percent", 0.0)) <= 100.0:
            errors.append("'dropout_percent' must be in [0, 100]")
        return errors + self._validate_extra(config) + self.validate_fault(config)

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        # Transport scope for the endpoint: own config wins, else inherited from the spawn.
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model
        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, prefix + self.site)
        if self._site_id < 0:
            raise RuntimeError(f"{self.PLUGIN_LABEL}: site {prefix + self.site!r} not found")
        self._bodyexclude = self._resolve_exclude_body(m, prefix)

        self._local_dirs = np.ascontiguousarray(self._build_directions(), dtype=np.float64)
        # Allocated once, so the per-cast path does not allocate -- this fires at rate_hz for the
        # life of the run, and for a 3D lidar it is 20k+ rays a frame.
        self._hits = raycast.buffers(self.num_rays)

        ros2_hints = {
            "type": self.ROS_TYPE,
            "topic": self.topic_override(self.ENDPOINT_NAME) or self.DEFAULT_TOPIC,
            "frame_id": self.frame_id,
            **self._ros2_hints_extra(),
        }
        if self.emit_static_tf:
            ros2_hints["static_tf"] = self._mount_tf(m, prefix)

        # The fault switch, if this sensor declares one. Registered here, beside the scan endpoint,
        # so both are in ctx.interface before a bridge binds it.
        self.register_fault_endpoints(ctx, ns)

        # Declared as a backend-neutral output endpoint (no ROS import here). The bridge resolves the
        # type string and publishes at rate; ``namespace`` scopes topic and frames.
        ctx.interface.add(
            Endpoint(
                name=self.ENDPOINT_NAME,
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._payload_value,
                rate_hz=self.rate_hz,
                backend={"ros2": ros2_hints},
            )
        )

    def _resolve_exclude_body(self, m, prefix: str) -> int:
        """Body id whose geoms the rays skip, or ``-1`` for "exclude nothing".

        An empty ``exclude_body`` means the latter explicitly. A body the *world asked for by name*
        and that does not resolve is an error: silently casting through the chassis it meant to skip
        is the kind of failure that shows up as inexplicable lidar returns much later.

        The class default is deliberately not held to that. A sensor mounted on the worldbody -- a
        static scanner on a tripod, or a bare test scene -- has no ``base_link`` and needs none, so
        the default resolving to nothing is an ordinary world rather than a mistake.
        """
        if not self.exclude_body:
            return -1
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.exclude_body)
        if bid < 0 and "exclude_body" in self.config:
            raise RuntimeError(
                f"{self.PLUGIN_LABEL}: exclude_body {prefix + self.exclude_body!r} not found. "
                f"Set 'exclude_body' to a body of this robot, or to '' to exclude nothing."
            )
        return bid

    def on_reset(self, ctx: SimContext) -> None:
        # sim_time restarts at 0 on reset; clear the gate so the first post-reset step casts again.
        self._last_cast = float("-inf")
        # And back to nominal: a fault applied in one trial must not survive into the next of the
        # same process, or the control cell silently becomes a faulted one.
        self.on_reset_fault()

    def _mount_tf(self, m, prefix: str) -> dict:
        """Static mount transform (base body -> sensor site) as plain numbers, for a bridge.

        Computed from the model on a throwaway ``MjData`` at the reference pose. The base<-site
        transform is rigid, so it is independent of where the robot stands; deriving it from the same
        site the rays are cast from keeps the published frame consistent with the payload by
        construction. No ROS types here -- ``roqsim`` stays ROS-free.
        """
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        # Body 0 is ``world`` (origin, identity), which is the right reference for a site mounted on
        # the worldbody -- the transform is then simply the site's world pose. Not a fallback for
        # tidiness: read as an index, ``self._bodyexclude`` of -1 selects ``xpos[-1]``, the *last*
        # body in the model, and publishes that unrelated body's transform under the declared
        # parent's name.
        #
        # The PARENT NAME follows the reference. ``_bodyexclude`` is -1 whenever
        # nothing was excluded: either the world said so (``exclude_body: ''``) or the class default
        # ``base_link`` is absent, which is the ordinary case for a scanner on a tripod or a mast.
        # Naming the parent after ``exclude_body`` regardless published numbers measured from the
        # world under the header of a frame that does not exist -- an orphaned sensor frame, and in
        # a world where some OTHER robot does have a ``base_link``, one bolted onto that robot at a
        # pose measured from somewhere else entirely. For the explicit ``''`` spelling it published
        # an empty ``frame_id``, which tf2 drops outright, so the frame never appeared at all.
        world_mounted = self._bodyexclude < 0
        ref = 0 if world_mounted else self._bodyexclude
        base_pos = d0.xpos[ref]
        base_mat = d0.xmat[ref].reshape(3, 3)
        site_pos = d0.site_xpos[self._site_id]
        site_mat = d0.site_xmat[self._site_id].reshape(3, 3)
        rel_pos = base_mat.T @ (site_pos - base_pos)
        rel_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rel_quat, np.ascontiguousarray(base_mat.T @ site_mat).reshape(-1))
        return {
            # Bare name; the bridge applies any namespace prefix.
            "parent": WORLD_FRAME if world_mounted else self.exclude_body,
            "translation": [float(v) for v in rel_pos],
            "rotation": [float(v) for v in rel_quat],  # (w, x, y, z)
        }

    def post_step(self, ctx: SimContext) -> None:
        # Cast at the sensor's own rate, not every physics step; the endpoint reads the latest value.
        if ctx.sim_time - self._last_cast < 1.0 / self.rate_hz:
            return
        self._last_cast = ctx.sim_time
        m, d = ctx.model, ctx.data
        origin = d.site_xpos[self._site_id]
        rot = d.site_xmat[self._site_id].reshape(3, 3)
        # Site-frame directions rotated into the world. `_local_dirs @ rot.T` is the world direction
        # of each ray; the payload is built back in the site frame, so both live off one array.
        raycast.cast(
            m,
            d,
            origin,
            self._local_dirs @ rot.T,
            cutoff=self.range_max,
            bodyexclude=self._bodyexclude,
            out=self._hits,
        )
        dist = self._hits.dist
        # The range window. `cutoff` above is a culling hint, not a clamp, so a hit beyond
        # `range_max` is still reported and is filtered here -- for every device, once.
        valid = (dist >= 0.0) & (dist <= self.range_max)
        if self.CLAMP_NEAR_RETURNS:
            # Fixed-length scan: a blind-zone return keeps its slot, pushed out to range_min.
            dist = np.maximum(dist, self.range_min)
        else:
            # Point cloud: a blind-zone return is not a point.
            valid = valid & (dist >= self.range_min)

        if self.range_stddev > 0.0 or self.dropout_percent > 0.0:
            # One generator per (sensor, step), not per draw: counter-based, so the same noise is
            # reproducible from a recording without replaying the run. Keyed on this plugin's own
            # name so two sensors on one robot get independent streams.
            rng = ctx.rng_for(self.name or self.PLUGIN_LABEL)
            # Copy before writing: `dist` may still be the reused cast buffer.
            dist = dist.copy()
            if self.range_stddev > 0.0:
                dist[valid] += rng.normal(0.0, self.range_stddev, size=int(valid.sum()))
            if self.dropout_percent > 0.0:
                # Randomly drop this percentage of the potential returns per frame.
                n_drop = int(round(self.num_rays * self.dropout_percent / 100.0))
                if n_drop > 0:
                    drop = rng.choice(self.num_rays, size=n_drop, replace=False)
                    valid = valid.copy()
                    valid[drop] = False
        self._payload_value = self._payload(dist, valid)
