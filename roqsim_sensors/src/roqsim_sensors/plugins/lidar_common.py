"""Shared machinery for every ray-casting range sensor: 2D laser scanners and 3D lidars alike.

:class:`RayCastSensorPlugin` owns everything the devices have in common --
config keys and their validation, site/``exclude_body`` resolution, the reusable ray buffers, the
static mount TF, the ``rate_hz`` gate, the range window, the noise model, and endpoint registration.
A device then declares only what actually distinguishes it:

* :meth:`~RayCastSensorPlugin._build_directions` -- its ray pattern, in the site frame.
* :meth:`~RayCastSensorPlugin._payload` -- its wire type (``LaserScan`` vs ``PointCloud``).
* a handful of ``DEFAULT_*`` class attributes -- its datasheet.

This mirrors how :mod:`camera_common` + :mod:`depth_camera` already layer the cameras, and it exists
for the same reason: duplicated copies diverge in ways that are bugs rather than choices. Two rules
are written once here.

**A return is classified against the physical detection limits, once, here.** A cast hit nearer
than :attr:`RayCastSensorPlugin.detection_min` is *too close*: the device cannot measure it. A hit
beyond :attr:`RayCastSensorPlugin.detection_max`, or no hit at all, is *no return*. What each becomes
on the wire is the device's own format: :meth:`RayCastSensorPlugin._payload` receives the measured
returns and the too-close mask separately. A ``LaserScan`` is a fixed-length array, so every ray keeps
its slot and a too-close ray carries the value the device's driver publishes for it (REP 117's
``-inf`` unless the device model says otherwise); a too-close ray is never raised to ``range_min`` and
never published as a measured distance. A point cloud is a list of real returns, so a too-close return
is not a point. For a point cloud the detection limits are ``range_min`` and ``max_range``.

**``max_range`` is enforced here, for everyone.** ``mj_multiRay``'s ``cutoff`` is a culling hint and
not a clamp -- it can still report a hit beyond it. Without the
window, a Mid-360 with a 40 m range would emit points from further away.

Config (every ray-cast device; each device's module adds its ray pattern and datasheet defaults)::

    <plugin short name>:
      site: lidar                # site the rays are cast from (default: the device's DEFAULT_SITE)
      frame_id: lidar            # frame the payload is stamped in, and the static TF's child
                                 #   (default: site)
      range_min: 0.164           # m; the nearest distance the device reports
      max_range: 20.0            # m; the farthest, enforced here rather than left to the cast
      rate_hz: 10.0              # cast and publish rate, not the physics rate
      exclude_body: ""           # body the rays skip -- the device's own housing (default: none)
      range_stddev: 0.0          # Gaussian range sigma (m)
      range_stddev_relative: 0.0 # sigma as a fraction of the distance, at and beyond
                                 #   range_stddev_relative_from (0 = constant sigma)
      range_stddev_relative_from: 0.0   # m; nearer than this the sigma is range_stddev
      range_resolution: 0.0      # quantisation step of a published distance (m); 0 = continuous
      dropout_percent: 0.0       # percent of returns dropped, drawn per cast
      emit_static_tf: true       # publish tf_parent -> frame_id; false where the mount publishes it
      tf_parent: ""              # body the static TF hangs from (default: the carrier's root body)
      lazy: false                # cast and publish only while something subscribes
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
    #: Nothing: a scanner excludes only its own housing, which a device model names.
    DEFAULT_EXCLUDE_BODY = ""

    #: Keys a ``fault:`` block may write WHILE THE RUN IS IN PROGRESS -> the attribute each lives in.
    #: Every row is read inside ``post_step`` on the frame it is used (see the noise block at the end
    #: of this file), so a write takes effect on the very next cast and reads back honestly.
    #: ``max_range`` -> ``range_max`` because the config key and the attribute do not share a
    #: name, and a fault naming the attribute would silently write nothing.
    LIVE_WRITABLE = {
        "range_stddev": "range_stddev",
        "range_stddev_relative": "range_stddev_relative",
        "range_stddev_relative_from": "range_stddev_relative_from",
        "range_resolution": "range_resolution",
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
        "tf_parent": "the static TF is published once, at configure.",
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
        # Range-dependent sigma: at and beyond `range_stddev_relative_from` metres the sigma is this
        # fraction of the true distance, nearer it is `range_stddev`. 0 = constant sigma.
        self.range_stddev_relative = float(self.config.get("range_stddev_relative", 0.0))
        self.range_stddev_relative_from = float(self.config.get("range_stddev_relative_from", 0.0))
        # Quantisation step of a published distance (m); 0 = continuous.
        self.range_resolution = float(self.config.get("range_resolution", 0.0))
        self.dropout_percent = float(self.config.get("dropout_percent", 0.0))
        # Publish base body -> sensor frame as a static TF (derived from the same site the rays are
        # cast from). On by default; disable when an external robot_state_publisher owns it.
        self.emit_static_tf = bool(self.config.get("emit_static_tf", True))
        # The body that static TF hangs from. Unset, it is the root body of the entity carrying the
        # sensor (a robot's base), else the resolved `exclude_body`, else the world -- see _mount_tf.
        self.tf_parent = self.config.get("tf_parent", "")
        # Opt out of casting AND publishing while nothing subscribes (``Endpoint.lazy``). Off by
        # default: a scan is cheap and a consumer in-process reads `latest` without subscribing.
        # A robot manifest sets it on the small sensors only its own stack reads, so a world that
        # never launches that stack pays nothing for them.
        self.lazy = bool(self.config.get("lazy", False))
        self._endpoint: Endpoint | None = None
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

    @property
    def detection_min(self) -> float:
        """Nearest distance the device measures; a nearer hit is too close. See the module docstring."""
        return self.range_min

    @property
    def detection_max(self) -> float:
        """Farthest distance the device measures; a farther hit is no return."""
        return self.range_max

    def _payload(self, dist: np.ndarray, valid: np.ndarray, near: np.ndarray):
        """Build the wire payload from per-ray ``dist`` and two disjoint masks.

        ``valid`` marks a measured return inside the detection limits, ``near`` a hit nearer than
        ``detection_min``; a ray in neither is no return. ``dist`` is metres along each ray, carrying
        any noise and quantisation where either mask is set, and is meaningless elsewhere.
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
        if float(config.get("range_stddev_relative", 0.0)) < 0:
            errors.append("'range_stddev_relative' must be >= 0")
        if float(config.get("range_stddev_relative_from", 0.0)) < 0:
            errors.append("'range_stddev_relative_from' must be >= 0")
        if float(config.get("range_resolution", 0.0)) < 0:
            errors.append("'range_resolution' must be >= 0")
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
            ros2_hints["static_tf"] = self._mount_tf(m, prefix, entity)

        # The fault switch, if this sensor declares one. Registered here, beside the scan endpoint,
        # so both are in ctx.interface before a bridge binds it.
        self.register_fault_endpoints(ctx, ns)

        # Declared as a backend-neutral output endpoint (no ROS import here). The bridge resolves the
        # type string and publishes at rate; ``namespace`` scopes topic and frames.
        self._endpoint = Endpoint(
            name=self.ENDPOINT_NAME,
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=lambda: self._payload_value,
            rate_hz=self.rate_hz,
            backend={"ros2": ros2_hints},
            lazy=self.lazy,
        )
        ctx.interface.add(self._endpoint)

    def _resolve_exclude_body(self, m, prefix: str) -> int:
        """Body id whose geoms the rays skip, or ``-1`` for "exclude nothing".

        Nothing is the default. A scanner excludes only its own housing, so a device model names that
        body (``exclude_body: mount``), and robot geometry in the scan plane is a real return. A named
        body that does not resolve is an error: silently casting through the housing it meant to skip
        is the kind of failure that shows up as inexplicable lidar returns much later.
        """
        if not self.exclude_body:
            return -1
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.exclude_body)
        if bid < 0:
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

    def _mount_tf(self, m, prefix: str, entity=None) -> dict:
        """Static mount transform (parent body -> sensor site) as plain numbers, for a bridge.

        Computed from the model on a throwaway ``MjData`` at the reference pose. The parent<-site
        transform is rigid, so it is independent of where the robot stands; deriving it from the same
        site the rays are cast from keeps the published frame consistent with the payload by
        construction. No ROS types here -- ``roqsim`` stays ROS-free.

        The parent is, in order: ``tf_parent`` when set; else the root body of the entity carrying
        the sensor (*entity*'s ``body``, a robot's base), the link a vendor description hangs a
        scanner's frame from; else the resolved ``exclude_body``; else ``world`` for a sensor nothing
        carries and that excludes nothing, whose transform is then its world pose. A named parent or
        a carrier body that is not in the model raises: a transform measured from one body and
        published under another's name is a frame bolted onto the wrong thing.
        """
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        carrier = entity.body if entity is not None and entity.body else ""
        if self.tf_parent:
            ref = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.tf_parent)
            if ref < 0:
                raise RuntimeError(
                    f"{self.PLUGIN_LABEL}: tf_parent {prefix + self.tf_parent!r} not found. Set it "
                    f"to a body of this robot, or leave it unset to hang the frame off the root "
                    f"body of the entity carrying the sensor."
                )
            parent = self.tf_parent
        elif carrier:
            ref = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, carrier)
            if ref < 0:
                raise RuntimeError(
                    f"{self.PLUGIN_LABEL}: the carrying entity {entity.name!r} names body "
                    f"{carrier!r}, which is not in the model. Set 'tf_parent' to the body the frame "
                    f"hangs from."
                )
            # Bare name, like every frame a bridge publishes; the bridge applies the namespace.
            parent = carrier.removeprefix(prefix)
        elif self._bodyexclude >= 0:
            ref = self._bodyexclude
            parent = self.exclude_body
        else:
            # Body 0 is ``world`` (origin, identity). Never index with -1: ``xpos[-1]`` is the last
            # body in the model, whose transform would be published under the parent's name.
            ref = 0
            parent = WORLD_FRAME
        base_pos = d0.xpos[ref]
        base_mat = d0.xmat[ref].reshape(3, 3)
        site_pos = d0.site_xpos[self._site_id]
        site_mat = d0.site_xmat[self._site_id].reshape(3, 3)
        rel_pos = base_mat.T @ (site_pos - base_pos)
        rel_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rel_quat, np.ascontiguousarray(base_mat.T @ site_mat).reshape(-1))
        return {
            # Bare name; the bridge applies any namespace prefix.
            "parent": parent,
            "translation": [float(v) for v in rel_pos],
            "rotation": [float(v) for v in rel_quat],  # (w, x, y, z)
        }

    def post_step(self, ctx: SimContext) -> None:
        # Cast at the sensor's own rate, not every physics step; the endpoint reads the latest value.
        if ctx.sim_time - self._last_cast < 1.0 / self.rate_hz:
            return
        if (
            self.lazy
            and self._endpoint is not None
            and self._endpoint.has_subscribers is not None
            and not self._endpoint.has_subscribers()
        ):
            # Nobody listening and the sensor opted out: the cast is the whole cost, so skip it too.
            # `has_subscribers is None` (no transport) is "assume yes", as for the cameras.
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
        # Classified on the TRUE distance, against the physical limits. `cutoff` above is a culling
        # hint, not a clamp, so a hit beyond the far limit is still reported and is filtered here --
        # for every device, once.
        hit = (dist >= 0.0) & (dist <= self.detection_max)
        near = hit & (dist < self.detection_min)
        valid = hit & ~near

        noisy = (
            self.range_stddev > 0.0
            or self.range_stddev_relative > 0.0
            or self.dropout_percent > 0.0
        )
        if noisy or self.range_resolution > 0.0:
            # Copy before writing: `dist` is still the reused cast buffer.
            dist = dist.copy()
        if noisy:
            # One generator per (sensor, step), not per draw: counter-based, so the same noise is
            # reproducible from a recording without replaying the run. Keyed on this plugin's own
            # name so two sensors on one robot get independent streams.
            rng = ctx.rng_for(self.name or self.PLUGIN_LABEL)
            if self.range_stddev > 0.0 or self.range_stddev_relative > 0.0:
                true = dist[hit]
                sigma = np.full(true.shape, self.range_stddev)
                if self.range_stddev_relative > 0.0:
                    far = true >= self.range_stddev_relative_from
                    sigma[far] = self.range_stddev_relative * true[far]
                # A measured distance is never negative, however near the surface and wide the sigma.
                dist[hit] = np.maximum(true + rng.standard_normal(true.shape) * sigma, 0.0)
            if self.dropout_percent > 0.0:
                # Randomly drop this percentage of the potential returns per frame: a dropped ray is
                # no return, whatever it would have been.
                n_drop = int(round(self.num_rays * self.dropout_percent / 100.0))
                if n_drop > 0:
                    drop = rng.choice(self.num_rays, size=n_drop, replace=False)
                    valid[drop] = False
                    near[drop] = False
        if self.range_resolution > 0.0:
            dist[hit] = np.round(dist[hit] / self.range_resolution) * self.range_resolution
        self._payload_value = self._payload(dist, valid, near)
