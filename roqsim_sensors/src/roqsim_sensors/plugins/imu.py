# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Sensor plugin: a strap-down IMU -- angular rate, proper acceleration, and attitude at a site.

The one sensor a mobile robot carries that the substrate had no plugin for. Every wheeled platform
here ships an IMU in reality, and the stacks these experiments run are built on it: ``robot_localization``
fuses ``sensor_msgs/Imu`` with wheel odometry, AMCL's motion model degrades without it, and a legged
or aerial controller reads it every cycle. Without the plugin a world could only publish *odometry*,
so an experiment whose independent variable is sensor quality had no rate channel to degrade and a
paper's "EKF over wheel odom + IMU" could not be reconstructed at all.

Nothing here is new physics. MuJoCo already computes all three signals; a site carries an
``<accelerometer>``/``<gyro>``/``<framequat>`` triple that reads them, and this plugin turns that
triple into a first-class observable -- a rate-limited endpoint with covariances, a blackboard reader
for in-process consumers, a static mount transform so the frame exists in TF without a URDF, and a
``fault:`` block so the IMU can be degraded *during* a run.

**Proper acceleration, not coordinate acceleration.** MuJoCo's accelerometer includes the
gravitational reaction, exactly as a real accelerometer does: a level, stationary robot reads
``+9.81 m/s^2`` on z, and a robot in free fall reads zero. That is the convention every ROS consumer
of ``sensor_msgs/Imu`` assumes (REP 145), so it is passed through untouched -- no gravity is added or
removed here. ``tests/test_imu.py`` pins it, because it is the one property a reader cannot verify by
looking at this file.

**A device bolted to the world still reads 1 g.** MuJoCo's accelerometer is a proper-acceleration
sensor computed from constraint forces, so a body resting on the floor reads ``+9.81`` (measured) and
a body in free fall reads ``0`` (measured) -- but a body **welded to the worldbody**, which is what a
fixed camera mount or a wall-mounted sensor is, reads ``0`` as well: it has no degrees of freedom, so
no acceleration is ever computed for it. Left alone, a tripod-mounted D435i would tell a stack it was
falling, forever.

For that one case the answer is closed-form rather than a model of anything: a body that cannot move
has proper acceleration exactly ``-g`` expressed in the sensor frame, so the plugin evaluates it
(detected by ``body_weldid == 0``, which is precisely "welded to the world", nested fixed children
included) and logs that it did. Nothing else is substituted: a body held by a *joint*, even a
motionless one, has MuJoCo compute its acceleration properly, and that reading is used untouched.
Gyro and attitude need no such handling -- a welded body cannot rotate, and ``0 rad/s`` is right.

**Attitude is ground truth, and says so.** ``framequat`` is the site's true world orientation, not
the output of an on-board fusion filter: there is no drift, no yaw bias, and no magnetometer.
A real strap-down IMU without magnetic heading has *unobservable* yaw, so a stack fed this signal is
being handed information the hardware could not give it. Two consequences, both explicit:

* ``orientation: false`` (a rate-and-acceleration-only IMU) publishes the ROS "orientation not
  provided" marker -- ``orientation_covariance[0] = -1`` -- which is the honest configuration for
  reconstructing a paper that fused raw rates.
* ``yaw_stddev`` / ``orientation_stddev`` perturb the attitude when a paper's IMU was imperfect but
  the experiment does not turn on *how* it was imperfect.

The choice is the experiment's and belongs in the world, so both are config with a documented
default rather than a behaviour this plugin picks. Prefixing the signal as ground truth (see
:doc:`ground_truth`) would be wrong here: a stack subscribes to ``imu/data``, and the truth-ness is a
property of the *attitude channel*, which the covariance already states.

**Where it goes.** An IMU is bolted to a link, so this plugin creates its own site on that body at a
configured offset -- no hand-authored MJCF site needed, which is what makes it usable from a robot
manifest. Naming an existing ``site:`` instead uses that one, for a model that already ships the
mount the vendor documents.

Config::

    imu:
      # The entity is the one this entry is NESTED UNDER; declaring it at the top of a document is
      # refused (`requires_owner`) -- an IMU measures a body's motion, so it belongs to something.
      body: ""                  # body the IMU is bolted to; default: the entity's registered base body
      site: ""                  # measure at an EXISTING site instead (then `body`/`pos`/`rpy` are unused)
      pos: [0.0, 0.0, 0.0]      # mount offset in the body frame (m)
      rpy: [0.0, 0.0, 0.0]      # mount orientation, fixed-axis XYZ (rad); or `quat: [w, x, y, z]`
      frame_id: imu_link        # the frame the reading is stamped in (default: '<label>_link')
      topic: imu/data           # the endpoint's RELATIVE topic, so a device can match its driver's
                                #   layout (the D435i's IMU is `camera/imu`); `topics: {imu: /abs}`
                                #   still hardwires an absolute one and wins over this
      rate_hz: 100.0            # endpoint publish rate
      orientation: true         # publish attitude; false -> orientation_covariance[0] = -1
      accel_stddev: 0.0         # m/s^2, additive Gaussian white noise, per axis
      gyro_stddev: 0.0          # rad/s, likewise
      accel_bias: [0, 0, 0]     # m/s^2, systematic -- a real accelerometer's is not zero-mean
      gyro_bias: [0, 0, 0]      # rad/s, likewise (a rate bias is what makes integrated yaw drift)
      orientation_stddev: 0.0   # rad, small-angle noise about each axis
      yaw_stddev: 0.0           # rad, EXTRA noise about the vertical axis only (see above)
      fault: {gyro_stddev: 0.4} # optional: the values it takes while degraded (set_sensor_override)

Endpoint ``imu`` (out) reads an :class:`ImuReading` and carries a ``sensor_msgs/Imu`` backend hint on
``imu/data`` -- the topic ``robot_localization`` and a standalone IMU driver both use -- plus the
static ``body -> frame_id`` transform. An :class:`ImuReader` is published on the blackboard under
``imu:<address>`` for in-process consumers.

**Nothing is computed while nothing is listening.** The reading is assembled in the endpoint's
``read()`` rather than in a ``post_step``, and the endpoint is ``lazy``, so a bridge does not even
read it while the transport reports no subscriber (``BridgeBase._skip_unsubscribed``). That matters
here because an IMU is the fastest sensor on the robot -- at 200 Hz, several of them, each drawing
noise from a fresh counter-based generator, is real work to do for nobody -- and because a device
whose manifest carries an IMU by default (the D435i does) must cost a world that ignores it nothing.
Two properties make ``lazy`` safe for this endpoint, and both are why it is opt-in per endpoint:
the publish has no side effect beyond the message (the mount transform is latched once at bind
time, not derived per read), and an in-process consumer goes through the blackboard reader, which is
not gated by anyone's subscriptions. Where no transport can report subscribers at all, the
convention is "assume yes", so the endpoint stays live.

**Covariances are the declared noise, squared, and nothing else.** The reported variance is
``stddev**2`` per channel (bias is systematic and deliberately not folded in: a covariance is what a
filter uses to weight the *random* part, and inflating it to cover a bias tells the filter the bias
will average out). A perfect sensor therefore reports zero variance, which is truthful and which some
filters refuse; a world that wants a filter-friendly floor states the stddev it wants assumed, in the
world, where the run's provenance records it.

**Noise draws come from** ``ctx.rng_for``, keyed ``imu:<address>``, like the lidar's and the FT
sensor's: reproducible from the run's seed, identical for two readers in the same step (the endpoint
and the blackboard reader are two by construction), and repeatable across a reset.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

from ..live_config import FaultableSensorMixin

_log = logging.getLogger(__name__)


@dataclass
class ImuReading:
    """Neutral payload for the ``imu`` endpoint: what a strap-down IMU reports at one instant.

    ``orientation`` is (w, x, y, z) in the world frame; the rates and accelerations are in the
    sensor frame, which is what a strap-down device measures and what REP 145 expects. The three
    variances are per-axis and isotropic; ``orientation_valid`` False is the ROS "not provided"
    marker rather than a zero quaternion, which a consumer cannot distinguish from level.
    """

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_acceleration: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_valid: bool = True
    orientation_variance: float = 0.0
    angular_velocity_variance: float = 0.0
    linear_acceleration_variance: float = 0.0


@dataclass
class ImuReader:
    """Blackboard handle published under ``imu:<address>``; read on the physics thread."""

    name: str
    frame: str
    read: Callable[[], ImuReading]


def _quat_from_rpy(rpy) -> list[float]:
    """(w, x, y, z) from fixed-axis XYZ roll/pitch/yaw (the ROS/URDF convention).

    ``mju_euler2Quat`` with the sequence ``"XYZ"`` (upper case: fixed axes) is that convention
    exactly -- checked against the hand-rolled half-angle form the other mount plugins carry, which
    is why this one does not carry a fourth copy of it.
    """
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray([float(v) for v in rpy], dtype=float), "XYZ")
    return [float(v) for v in quat]


def _vec3(value) -> list[float]:
    """A three-vector as plain floats.

    A list rather than a numpy array because the fault mixin coerces a written value with
    ``type(nominal)(value)``, and ``np.ndarray([...])`` is a shape constructor -- a faulted bias
    would come back as garbage. Adding a list to the numpy reading works either way.
    """
    return [float(v) for v in value] if value is not None else [0.0, 0.0, 0.0]


class ImuPlugin(FaultableSensorMixin, Plugin):
    """See the module docstring."""

    parallel_safe = True  # post-compile, read-only: reads data.sensordata only
    #: An IMU measures the motion of the body it is bolted to, so it belongs to an entity.
    requires_owner = True

    #: What a `fault:` block may change mid-run -> the attribute holding it. Every one of these is
    #: read fresh on each `read()`, so writing it takes effect on the next sample.
    LIVE_WRITABLE = {
        "accel_stddev": "accel_stddev",
        "gyro_stddev": "gyro_stddev",
        "accel_bias": "accel_bias",
        "gyro_bias": "gyro_bias",
        "orientation_stddev": "orientation_stddev",
        "yaw_stddev": "yaw_stddev",
        "orientation": "orientation",
    }
    #: The keys someone reaches for first, and why writing them at run time cannot work.
    REFUSED_WRITES = {
        "site": "the site is resolved to an id at configure; a later write moves no sensor.",
        "body": "the mount is built into the model before compile.",
        "pos": "likewise -- the mount pose is geometry, not a per-frame value.",
        "rpy": "likewise.",
        "frame_id": "a consumer that saw the frame change mid-run reads it as two sensors.",
        "rate_hz": "the endpoint's rate gate is fixed when the bridge binds it.",
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.owner = self.entity
        self.body = self.config.get("body", "")
        self.site = self.config.get("site", "")
        self.rate_hz = float(self.config.get("rate_hz", 100.0))
        self.frame_id = self.config.get("frame_id") or f"{self.label}_link"
        # A relative topic, namespaced by the bridge like every other endpoint's default. Distinct
        # from `topics:`, which hardwires an ABSOLUTE topic and bypasses the namespace: a device that
        # simply names its channels differently from a standalone IMU (`camera/imu`) should not have
        # to give up the namespace to say so, which is what an absolute override would cost it.
        self.topic = str(self.config.get("topic") or "imu/data")
        self.orientation = bool(self.config.get("orientation", True))
        self.accel_stddev = float(self.config.get("accel_stddev", 0.0))
        self.gyro_stddev = float(self.config.get("gyro_stddev", 0.0))
        self.accel_bias = _vec3(self.config.get("accel_bias"))
        self.gyro_bias = _vec3(self.config.get("gyro_bias"))
        self.orientation_stddev = float(self.config.get("orientation_stddev", 0.0))
        self.yaw_stddev = float(self.config.get("yaw_stddev", 0.0))
        # Faulted values + the switch. After the attributes above, since it reads them for nominal.
        self._fault_init()
        self._ctx: SimContext | None = None
        self._accel_adr = -1
        self._gyro_adr = -1
        self._quat_adr = -1
        self._site_id = -1
        self._mount_bid = -1  # body the static mount transform is published relative to
        #: Is the sensor's body welded to the world? Then MuJoCo computes no acceleration for it and
        #: the proper acceleration is exactly -g in the sensor frame (see the module docstring).
        self._world_fixed = False
        self._resolved_site = ""  # set in build(), reused in configure()
        self._own_site = False  # did this plugin create the site?

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 100.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if "topic" in config and (
            not isinstance(config["topic"], str) or not config["topic"].strip()
        ):
            errors.append("'topic' must be a non-empty relative topic, e.g. 'camera/imu'")
        elif isinstance(config.get("topic"), str) and config["topic"].startswith("/"):
            # An absolute value here would look like it worked while silently escaping the
            # namespace, which is the one thing `topics:` exists to do deliberately.
            errors.append(
                "'topic' is relative (it is scoped by the entity's namespace); for an absolute "
                "topic use topics: {imu: /your/topic}"
            )
        for key in ("accel_stddev", "gyro_stddev", "orientation_stddev", "yaw_stddev"):
            if float(config.get(key, 0.0)) < 0:
                errors.append(f"'{key}' must be >= 0")
        for key in ("accel_bias", "gyro_bias"):
            if key in config and len(config[key]) != 3:
                errors.append(f"'{key}' must be three numbers [x, y, z]")
        if "pos" in config and len(config["pos"]) != 3:
            errors.append("'pos' must be three numbers [x, y, z] (a mount offset has a height)")
        if "quat" in config and len(config["quat"]) != 4:
            errors.append("'quat' must be four numbers [w, x, y, z]")
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be three numbers [roll, pitch, yaw]")
        if "quat" in config and "rpy" in config:
            errors.append("set 'quat' or 'rpy', not both -- two spellings of one orientation")
        if config.get("site") and (
            "pos" in config or "rpy" in config or "quat" in config or config.get("body")
        ):
            errors.append(
                "'site' names an existing mount, so 'body'/'pos'/'rpy'/'quat' would be ignored: "
                "either name the site the model ships, or give the body and offset to build one."
            )
        if "seed" in config:
            # Silently ignoring it would leave a world believing it pinned the noise stream.
            errors.append(
                "'seed' is not an imu setting: noise is drawn from the RUN's seed (`roqsim sim "
                "--seed`, or the campaign's) via ctx.rng_for, so every sensor in a run is "
                "reproducible together. Remove the key."
            )
        return errors + self.validate_fault(config)

    # -- lifecycle ----------------------------------------------------------------------------

    def build(self, spec, ctx: SimContext) -> None:
        """Add the site (unless one was named) and the three MJCF sensors that read it.

        Entities register in ``configure``, after every ``build``, so the owner's prefix is not
        available yet: an explicit ``prefix:`` wins, else a body/site is matched by suffix -- the
        same resolution ``force_torque`` does, for the same reason.
        """
        prefix = self.config.get("prefix")
        if self.site:
            matches = [
                s.name
                for s in spec.sites
                if (
                    s.name == f"{prefix}{self.site}"
                    if prefix is not None
                    else s.name.endswith(self.site)
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"imu[{self.label}]: expected exactly one site matching {self.site!r}, found "
                    f"{matches}. Set 'prefix:' when a world carries more than one of this robot."
                )
            site_name = matches[0]
        else:
            body_name = self._resolve_body(spec, prefix)
            parent = spec.body(body_name)
            if parent is None:  # pragma: no cover - _resolve_body already matched it
                raise RuntimeError(f"imu[{self.label}]: body {body_name!r} not found")
            site = parent.add_site()
            # Named off the ADDRESS, not the label: two robots each carrying an `imu` have the same
            # label, and a duplicate site name is a compile error that names neither of them.
            site_name = f"{body_name}_{self.label}_site"
            site.name = site_name
            site.pos = [float(v) for v in self.config.get("pos", [0.0, 0.0, 0.0])]
            site.quat = (
                [float(v) for v in self.config["quat"]]
                if "quat" in self.config
                else _quat_from_rpy(self.config.get("rpy", [0.0, 0.0, 0.0]))
            )
            # A mount frame, not geometry: small, and left in the default site group so a world's
            # own site-visualisation setting decides whether mounts are drawn.
            site.size = [0.005, 0.005, 0.005]
            self._own_site = True
        self._resolved_site = site_name

        existing = {s.name for s in spec.sensors}
        for suffix, kind in (
            ("accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
            ("gyro", mujoco.mjtSensor.mjSENS_GYRO),
            ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ):
            name = f"{site_name}_{suffix}"
            if name in existing:
                # A vendor MJCF may ship its own IMU sensors on the site it documents. A second
                # triple would compile and double the sensordata layout, so the existing one wins.
                continue
            sensor = spec.add_sensor()
            sensor.name = name
            sensor.type = kind
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = site_name

    def _resolve_body(self, spec, prefix: str | None) -> str:
        """The body the IMU is bolted to: an explicit ``body:``, else the owner's base body."""
        if self.body:
            wanted = self.body
        else:
            # The entity's registered base body is not known until `configure`, so the conventional
            # base link is matched here and a miss names both candidates rather than guessing.
            wanted = "base_link"
        names = [b.name for b in spec.bodies]
        if prefix is not None:
            if f"{prefix}{wanted}" in names:
                return f"{prefix}{wanted}"
            raise RuntimeError(f"imu[{self.label}]: body {prefix}{wanted!r} not found")
        matches = [
            n for n in names if n == wanted or n.endswith(f"/{wanted}") or n.endswith(wanted)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"imu[{self.label}]: expected exactly one body matching {wanted!r}, found {matches}."
                f" Set 'body:' to the link the IMU is bolted to (and 'prefix:' when a world carries "
                f"more than one of this robot)."
            )
        return matches[0]

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.owner)
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self._resolved_site)
        if self._site_id < 0:
            raise RuntimeError(
                f"imu[{self.label}]: site {self._resolved_site!r} missing after compile"
            )
        for suffix, attr in (("accel", "_accel_adr"), ("gyro", "_gyro_adr"), ("quat", "_quat_adr")):
            sid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_SENSOR, f"{self._resolved_site}_{suffix}"
            )
            if sid < 0:
                raise RuntimeError(
                    f"imu[{self.label}]: sensor {self._resolved_site}_{suffix} missing after compile"
                )
            setattr(self, attr, int(m.sensor_adr[sid]))
        # The site's own body is what the mount transform is relative to, whether this plugin built
        # the site or the model shipped it -- so the published frame cannot disagree with the reading.
        self._mount_bid = int(m.site_bodyid[self._site_id])
        # weldid 0 is the world's weld group: this body has no degrees of freedom at all.
        self._world_fixed = int(m.body_weldid[self._mount_bid]) == 0
        if self._world_fixed:
            # Logged, not silent: the reading no longer comes from the MJCF sensor, and a reader of
            # the run's log should be able to see which branch produced it.
            _log.info(
                "imu[%s]: %s is welded to the world, so MuJoCo computes no acceleration for it; "
                "reporting the closed-form proper acceleration (-gravity in the sensor frame)",
                self.address,
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, self._mount_bid),
            )

        key = f"imu:{self.address}"
        if ctx.blackboard.get(key) is not None:
            raise RuntimeError(
                f"imu: blackboard key {key!r} is already registered. Two IMUs need distinct "
                f"labels, else a consumer silently reads the wrong one."
            )
        ctx.blackboard.set(key, ImuReader(name=self.label, frame=self.frame_id, read=self.read))

        # The fault switch, if this sensor declares one -- before the reading endpoint, so both are
        # in ctx.interface when a bridge binds them.
        self.register_fault_endpoints(ctx, ns)
        ctx.interface.add(
            Endpoint(
                name="imu",
                direction="out",
                owner=self.owner,
                namespace=ns,
                read=self.read,
                rate_hz=self.rate_hz,
                # Read only while something subscribes -- see the module docstring.
                lazy=True,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.Imu",
                        # `imu/data` by default, where robot_localization and a standalone driver
                        # look; `topic:` is how a device states its own layout.
                        "topic": self.topic_override("imu") or self.topic,
                        "frame_id": self.frame_id,
                        "static_tf": self._mount_tf(m),
                    }
                },
            )
        )

    def _mount_tf(self, m) -> dict:
        """Static ``mount body -> frame_id`` transform as plain numbers, for a bridge.

        Read off a throwaway ``MjData`` at the reference pose: the body<-site transform is rigid, so
        it does not depend on where the robot stands, and taking it from the same site the sensors
        read keeps the published frame consistent with the payload by construction.
        """
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        base_pos = d0.xpos[self._mount_bid]
        base_mat = d0.xmat[self._mount_bid].reshape(3, 3)
        rel_pos = base_mat.T @ (d0.site_xpos[self._site_id] - base_pos)
        rel_quat = np.zeros(4)
        site_mat = d0.site_xmat[self._site_id].reshape(3, 3)
        mujoco.mju_mat2Quat(rel_quat, np.ascontiguousarray(base_mat.T @ site_mat).reshape(-1))
        return {
            # Bare name: the bridge applies any namespace prefix, and the model's body name is
            # already prefixed per robot, so it is stripped back to what TF expects.
            "parent": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, self._mount_bid).split("/")[
                -1
            ],
            "translation": [float(v) for v in rel_pos],
            "rotation": [float(v) for v in rel_quat],  # (w, x, y, z)
        }

    def on_reset(self, ctx: SimContext) -> None:
        self.on_reset_fault()

    # -- the reading --------------------------------------------------------------------------

    def read(self) -> ImuReading:
        """The current reading. Runs on the physics thread; called once per due tick per reader."""
        d = self._ctx.data
        gyro = np.array(d.sensordata[self._gyro_adr : self._gyro_adr + 3], dtype=float)
        quat = np.array(d.sensordata[self._quat_adr : self._quat_adr + 4], dtype=float)
        if self._world_fixed:
            # site_xmat rotates sensor -> world, so its transpose takes -g into the sensor frame.
            rot = np.array(d.site_xmat[self._site_id]).reshape(3, 3)
            accel = rot.T @ -np.asarray(self._ctx.model.opt.gravity, dtype=float)
        else:
            accel = np.array(d.sensordata[self._accel_adr : self._accel_adr + 3], dtype=float)

        accel = accel + self.accel_bias
        gyro = gyro + self.gyro_bias
        if self.accel_stddev or self.gyro_stddev or self.orientation_stddev or self.yaw_stddev:
            # One counter-based generator per (sensor, step): every reader in this step sees the same
            # reading, and it is reproducible from a recording without replaying the run.
            rng = self._ctx.rng_for(f"imu:{self.address}")
            if self.accel_stddev:
                accel = accel + rng.normal(0.0, self.accel_stddev, 3)
            if self.gyro_stddev:
                gyro = gyro + rng.normal(0.0, self.gyro_stddev, 3)
            if self.orientation and (self.orientation_stddev or self.yaw_stddev):
                quat = self._perturb(quat, rng)

        return ImuReading(
            orientation=[float(v) for v in quat],
            angular_velocity=[float(v) for v in gyro],
            linear_acceleration=[float(v) for v in accel],
            orientation_valid=self.orientation,
            # stddev**2, per channel. Bias is deliberately not folded in -- see the module docstring.
            orientation_variance=self.orientation_stddev**2 + self.yaw_stddev**2,
            angular_velocity_variance=self.gyro_stddev**2,
            linear_acceleration_variance=self.accel_stddev**2,
        )

    def _perturb(self, quat: np.ndarray, rng) -> np.ndarray:
        """Rotate the true attitude by a small random rotation, isotropic plus extra yaw.

        Applied as a rotation composed onto the truth rather than as noise added to the four
        components: adding to a quaternion and renormalising biases the attitude towards the
        identity, so a tilted robot would read as slightly more level the noisier the sensor got.
        """
        axis_angle = np.zeros(3)
        if self.orientation_stddev:
            axis_angle += rng.normal(0.0, self.orientation_stddev, 3)
        if self.yaw_stddev:
            axis_angle[2] += rng.normal(0.0, self.yaw_stddev)
        delta = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            delta,
            axis_angle / (np.linalg.norm(axis_angle) or 1.0),
            float(np.linalg.norm(axis_angle)),
        )
        out = np.zeros(4)
        # World-frame perturbation: yaw noise must be about the world's vertical, not the sensor's,
        # or a robot on a ramp would have its "yaw" error tilt with it.
        mujoco.mju_mulQuat(out, delta, quat)
        return out
