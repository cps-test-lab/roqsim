"""Sensor plugin: a six-axis force/torque sensor at a site.

The substrate's *contact-force* observable. Every other sensor here reports geometry —
where things are (lidar, cameras, fiducials, ground-truth pose). None of them reports what a robot
is pushing against, and for a contact-rich manipulation task that is the whole measurement: an
insertion, a polishing pass, or a compliant assembly is judged by its wrench, not by its trajectory.

MuJoCo computes the constraint wrench already; a ``<force>``/``<torque>`` sensor pair on a site
reads it, and this plugin turns that pair into a first-class observable — a rate-limited endpoint, a
blackboard reader for in-process controllers, and an optional per-trial log. Nothing here is novel
physics; it is plumbing.

**Where the sensor goes matters more than it looks.** A site sensor reads the wrench the site's body
receives from its parent: the load of that body and of every body below it in the kinematic tree,
and of nothing else. A tool is in the reading only if it hangs from that body or below it -- where a
real FT sensor is bolted, between the flange and the tool. A site on the flange body itself reads a
tool mounted there, plus that body's own weight, which a tare removes. A tool on a body *beside* the
sensor's is outside the reading altogether: the ``ur5e``'s flange carries both its sensor stack
(``tool0``, with ``fts_site``) and its bare ``attachment_site``, and a tool welded at the latter
reads as nothing at ``fts_site`` -- not its weight, not a push on it, not a contact -- which looks
like a quiet trial rather than a broken world.

**A tool the sensor cannot see.** So ``configure`` refuses a sensor on an entity whose mounted tool
lies outside that subtree. The entity names its tool in ``meta["end_effector"]`` (``site``, where
it is mounted, and ``bodies``, its root bodies; ``spawn_arm`` records both), and the tool is seen
when the body carrying its mount site is in the sensed subtree, or when the sensor sits in the tool
itself, as a fingertip sensor does. Measured on MuJoCo 3.14.0
(``roqsim_manipulation/tests/test_tool_is_in_the_wrench.py``): a 0.5 kg tool on the ``ur5e``
reads 0.000 N at ``fts_site`` when welded at ``attachment_site`` and its full 4.905 N at
``tool_site``, the site the model's manifest declares; on every other arm with a mount site, a
sensor at that site reads the tool's weight and a push on it. There is no key that accepts the
first case: a sensor that reads none of its tool measures nothing a trial asks of it.

Config -- a component of the entry that spawns the arm whose prefix and namespace it inherits, since ownership is where the entry
sits rather than a config key::

    force_torque:
      site: fts_site            # REQUIRED: MJCF site to measure at (prefixed with the arm's prefix)
      frame: base               # sensor | base | world -- the frame the wrench is REPORTED in
      invert: true              # negate the reading (report the force the ENVIRONMENT applies to the
                                #   tool, the sign convention a real FT sensor and its users assume;
                                #   MuJoCo's site sensor reports the opposite). Whichever is chosen,
                                #   the blackboard reader says which it is in `measures`, so a
                                #   consumer never has to assume.
      tare_at_s: null           # sim time (s) at which to capture the zero offset, once per
                                #   episode; null (default) never tares. The `tare` service and
                                #   `WrenchReader.tare()` are the better doors -- see "Taring"
                                #   below, and note the offset is only valid at the pose it was
                                #   captured at.
      noise_force_stddev: 0.0   # N, additive Gaussian white noise on the three force channels
      noise_torque_stddev: 0.0  # Nm, likewise on the three torque channels
      bias_force: 0.0           # N, bound of a per-episode zero offset, uniform per channel
      bias_torque: 0.0          # Nm, likewise
      drift_force: 0.0          # N/s, bound of a per-episode linear drift rate, uniform per channel
      drift_torque: 0.0         # Nm/s, likewise
      range_force: null         # N, the measuring range: each force channel saturates at +-range
      range_torque: null        # Nm, likewise for torque; null (default) never saturates
      rate_hz: 100.0            # endpoint publish rate
      namespace: ""             # transport scope (default: inherited from the entity)
      topics: {wrench: /ft}     # optional absolute-topic hardwire
      controller_name: ft_broadcaster  # name of the broadcaster it registers with the controller
                                #   manager (default: <entry label>_broadcaster)

Endpoint ``tare`` (a command) is that zero button, a ``std_srvs/Trigger`` service on
``<name>/tare`` over ROS; it takes no argument and its reply is what lets a scenario fail rather
than measure against an offset it only assumed was applied.

Endpoint ``wrench`` (out) reads a :class:`roqsim.types.Wrench`, a ``geometry_msgs/WrenchStamped`` on
``<name>/wrench`` over ROS, stamped in the frame ``frame`` names. A ``WrenchReader`` is published on the blackboard
under ``ft:<entry label>`` for in-process consumers — the admittance controller is one — exposing
``read()`` and the resolved ``frame``.

**Frames.** ``sensor`` returns MuJoCo's raw site-frame reading. ``base`` rotates it into the owning
entity's base body frame, and ``world`` into the world frame. The choice is not cosmetic for
metrics that split the wrench into an insertion axis and the plane orthogonal to it: ``|F_z|`` and
``||F_x, F_y||`` are frame-dependent, and a tool that tilts reports a different split in its own
frame than in the world's. The ``WrenchStamped`` header names that frame as TF knows it: the site's
name without the entity's MJCF prefix, published with the fixed transform from the body it sits on;
the entity's root body, likewise unprefixed; or ``world``.

**Taring: zeroing the tool's own load.** The sensor reads everything below the cut, which for a
loaded flange is mostly the tool's own weight -- so a contact task measuring a 5 N push starts from
20 N of tool.

Zeroing is a **command**, the way it is on real hardware: an FT driver exposes a service taking no
argument (``zero_ftsensor``) and this exposes the same thing three ways onto one implementation --
the ``tare`` endpoint (``std_srvs/Trigger`` over ROS), ``WrenchReader.tare()`` for an in-process
controller, and ``tare_at_s`` for a world that wants it done once at a stated time without anything
to press the button. Prefer one of the first two: a time is a number that has to stay in step with
a scenario's own timing, and if the approach runs long it fires mid-motion and zeroes against a
contact.

All three re-arm on ``on_reset``: an offset carried into the next episode is a measurement of the
previous one, and repetitions of a trial would not be repetitions.

**A tare is not gravity compensation, and the difference is a trap.** The offset is captured in
the raw sensor frame and subtracted there, exactly as a real FT sensor's zero button works -- so it
is valid at *the pose the tool was in when it was captured*. Rotate the tool ninety degrees and the
weight reappears, up to twice the tool's load in the worst case, because the tool's weight is fixed
in the WORLD frame while the sensor frame turns with it. Compensating at every pose needs the
tool's mass and centre of mass estimated, which is a calibration, not a tare; roqsim does not do
it, and a tare that quietly claimed to would leave a contact controller chasing a bias that grows
with the tool's tilt. Tare at the pose you are about to make contact in, or tare per approach.

The offset is the reading BEFORE noise is added, so what is left after taring is the noise alone
rather than the noise plus one sample's worth of it -- a real tare averages many samples, and this
is that average exactly. Capture happens on the first read at or after ``tare_at_s``, so a sensor
nobody reads is never tared and one read at 100 Hz tares within a step of the time asked for.

**A flex's contacts are added to the reading.** MuJoCo's site force/torque sensor reads the sensed
subtree's momentum balance against its external forces and leaves a contact with a flex out of them
(:mod:`roqsim.flex`, "the wrench a flex's contacts put into a subtree"). Measured on MuJoCo 3.14.0
(``tests/test_force_torque_flex.py``): a rigid probe pressing a flex block read its weight alone, and
a support carrying 1.5 N of a pinned elastic cantilever was absent from the reading. The rest of
what a flex does already reaches the sensor -- its weight, its inertia and its elastic forces, which
act on its vertex DOFs as joint forces between each vertex body and the body it hangs from.

So the plugin adds the missing term: every contact that involves a flex and has exactly one side in
the sensed subtree enters with its force and its moment about the site
(:class:`roqsim.flex.FlexContactWrench`). That covers a soft tool below the sensor touching anything
-- an elastic block pinned to a flange pressing a table -- and a rigid tool pressing a soft object.
The correction is applied to the raw pair, before the tare and the frame, so the reading is what a
real sensor at that cut measures. A flex lying partly inside the subtree and partly outside is
refused: its contacts cannot be assigned to one side of the cut.

**Bias, drift and range are the transducer's, so they come before the tare.** A real sensor's zero
offset and its slow thermal drift are in what it measures, and its zero button removes whatever of
them has accumulated -- which is why a contact task tares at the pose it is about to press in. So
``bias_*`` (a constant per episode) and ``drift_*`` (a rate per episode, the offset growing
linearly from zero at the episode's start) are added to the raw pair, and a tare captures them with
it. Both are drawn once per episode and per channel, uniformly within the stated bound, from
``ctx.rng_for(..., per_episode=True)``: a function of the run's seed and the episode, so a value at
any sim time is reproducible from that time alone, as the noise is. Linear rather than a random
walk for that reason -- a walk's value depends on every step before it -- and because a trial lasts
minutes, over which a thermal drift is close to a line. ``range_*`` saturates each raw channel, bias
and drift included, as the transducer's range does; the noise is added after, in the reported
frame. All default to zero or none.

**Noise is per-sensor config, deliberately.** There is no generic error-model framework in roqsim (see
``docs/architecture.rst`` §9); a sensor that wants noise declares its own, as the lidar's
``range_stddev`` does. The default is zero: a noise model that appears without being asked for is a
silent change to every metric derived from the signal.

The draws come from :meth:`roqsim.context.SimContext.rng_for` -- the run's seed, not a per-sensor one --
for the same reason the lidars use it, plus one specific to a wrench: it is a pure function of
``(seed, sim_time, sensor)``, so **two readers in the same step see the same wrench**. This sensor has
two by construction (the ``wrench`` endpoint and the blackboard ``WrenchReader`` an in-process
controller polls), and with a stateful generator each read would have advanced the stream -- the
controller and the recorded signal would disagree about the force at one instant, which is
indistinguishable from a controller bug. It also makes the noise reproducible from a recorded state
without replaying the run, and repeats identically after ``on_reset`` because ``sim_time`` restarts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.controllers import ACTIVE, Controller, registry_for
from roqsim.flex import FlexContactWrench
from roqsim.plugin import Plugin
from roqsim.presence import entity_body_ids
from roqsim.types import Wrench

_FRAMES = ("sensor", "base", "world")


@dataclass
class WrenchReader:
    """Blackboard handle published under ``ft:<name>``; consumed on the physics thread.

    ``read()`` returns ``(force[3], torque[3])`` as numpy arrays in ``frame``. ``frame`` is carried
    with the reader because a consumer that integrates the wrench into a motion command has to know
    which frame it is commanding in, and getting that wrong produces a controller that pushes in a
    plausible-looking wrong direction rather than one that fails.

    ``measures`` is carried for exactly the same reason: a wrench has a
    direction as well as a frame, and the two conventions are negatives of each other. It is
    ``"environment_on_tool"`` (what a real FT sensor and its users assume, this sensor's default)
    or ``"tool_on_environment"`` (MuJoCo's raw site sensor). A consumer comparing a measured wrench
    against a target it *commands* must put both in one convention first; subtracting one from the
    other turns a contact controller's negative feedback into positive, which does not look like a
    sign error -- it looks like the contact getting away from the controller.

    ``tare()`` captures the current reading as the zero offset, the way a real sensor's zero button
    does -- and with the same limit: the offset is valid at the pose it was captured at, not at
    every pose. See "Taring" in the module docstring before reaching for it. ``None`` on a reader
    built by something other than this plugin.
    """

    name: str
    frame: str
    read: Callable[[], tuple[np.ndarray, np.ndarray]]
    measures: str = "environment_on_tool"
    tare: Callable[[], None] | None = None


class ForceTorquePlugin(Plugin):
    parallel_safe = True  # post-compile read-only: reads data.sensordata / xmat

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        # No config `name:` of its own: this instance is identified by its label, like every
        # other entry, so the document, the blackboard key and the topic cannot disagree about
        # what this sensor is called.
        self.site = self.config.get("site", "")
        self.owner = self.entity
        self.frame = self.config.get("frame", "sensor")
        self.invert = bool(self.config.get("invert", True))
        self.noise_f = float(self.config.get("noise_force_stddev", 0.0))
        self.noise_t = float(self.config.get("noise_torque_stddev", 0.0))
        self.bias_f = float(self.config.get("bias_force", 0.0))
        self.bias_t = float(self.config.get("bias_torque", 0.0))
        self.drift_f = float(self.config.get("drift_force", 0.0))
        self.drift_t = float(self.config.get("drift_torque", 0.0))
        rf, rt = self.config.get("range_force"), self.config.get("range_torque")
        self.range_f = None if rf is None else float(rf)
        self.range_t = None if rt is None else float(rt)
        #: This episode's draws: (bias force, bias torque, drift force, drift torque), each [3].
        self._transducer: tuple[np.ndarray, ...] | None = None
        self._transducer_episode = -1
        tare_at = self.config.get("tare_at_s")
        self.tare_at_s = None if tare_at is None else float(tare_at)
        #: The captured zero, in the RAW sensor frame and before the sign convention is applied --
        #: which is where it has to live: an offset stored after the rotation would be re-rotated
        #: on every read and would drift with the tool, and one stored after `invert` would flip
        #: with a config change that is meant to affect only how the reading is reported.
        self._offset_force = np.zeros(3)
        self._offset_torque = np.zeros(3)
        self._tared = False
        self.rate_hz = float(self.config.get("rate_hz", 100.0))
        self._ctx: SimContext | None = None
        self._force_adr = -1
        self._torque_adr = -1
        self._site_id = -1
        self._ref_bid = -1  # body whose frame the wrench is rotated into (frame: base)
        self._resolved_site = ""  # set in build(), reused in configure()
        self._flex_contacts: FlexContactWrench | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if not config.get("site"):
            errors.append("'site' is required: name the MJCF site the wrench is measured at")
        if config.get("frame", "sensor") not in _FRAMES:
            errors.append(f"'frame' must be one of {', '.join(_FRAMES)}")
        if float(config.get("rate_hz", 100.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        for key in (
            "noise_force_stddev",
            "noise_torque_stddev",
            "bias_force",
            "bias_torque",
            "drift_force",
            "drift_torque",
        ):
            if float(config.get(key, 0.0)) < 0:
                errors.append(f"'{key}' must be >= 0")
        for key in ("range_force", "range_torque"):
            if config.get(key) is not None and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0, or null for a sensor that never saturates")
        if config.get("tare_at_s") is not None and float(config["tare_at_s"]) < 0:
            errors.append("'tare_at_s' must be >= 0: it is a sim time, not an offset")
        if "flex_reaction" in config:
            # A world that states it expects a reading blind to flex contacts; the reading now
            # carries them, so the statement no longer describes the run and must not pass silently.
            errors.append(
                "'flex_reaction' is not a force_torque setting: the reading includes the contacts "
                "of a flex in or against the sensed subtree. Remove the key, and re-read any result "
                "that relied on the sensor being blind to them."
            )
        if "seed" in config:
            # Silently ignoring it would leave a world believing it pinned the noise stream.
            errors.append(
                "'seed' is not a force_torque setting: noise is drawn from the RUN's seed "
                "(`sim.seed` in the world, or `roqsim sim --seed`, or the seed the scenario "
                "adapter resolves) via ctx.rng_for, so every sensor in a run is reproducible "
                "together. Remove the key."
            )
        return errors

    def build(self, spec, ctx: SimContext) -> None:
        """Add the ``<force>``/``<torque>`` sensor pair, unless the model already carries them.

        A vendor MJCF may ship its own FT sensors on the same site (the tool adapter is part of the
        model, after all). Adding a second pair would compile fine and double the sensordata layout,
        so an existing pair on this site wins and the plugin just reads it.
        """
        # Entities register in `configure`, after every build hook, so the arm's prefix is not
        # available here; resolve the site by suffix instead (peg_in_hole has the same problem).
        prefix = self.config.get("prefix")
        if prefix is not None:
            matches = [s for s in spec.sites if s.name == f"{prefix}{self.site}"]
        else:
            matches = [s for s in spec.sites if s.name.endswith(self.site)]
        if len(matches) != 1:
            raise RuntimeError(
                f"force_torque[{self.name}]: expected exactly one site matching {self.site!r}, found "
                f"{[s.name for s in matches]}. A six-axis FT sensor is measured at one site; set "
                f"`prefix:` when a world carries more than one arm."
            )
        site_name = matches[0].name
        self._resolved_site = site_name
        existing = {s.name for s in spec.sensors}
        if f"{site_name}_force" in existing and f"{site_name}_torque" in existing:
            return
        # spec.sensors is validated at compile; a missing site raises there with the site name, which
        # is a better error than anything this plugin could produce pre-compile.
        for suffix, kind in (
            ("force", mujoco.mjtSensor.mjSENS_FORCE),
            ("torque", mujoco.mjtSensor.mjSENS_TORQUE),
        ):
            sensor = spec.add_sensor()
            sensor.name = f"{site_name}_{suffix}"
            sensor.type = kind
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = site_name

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.owner)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        site_name = self._resolved_site or f"{prefix}{self.site}"

        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise RuntimeError(
                f"force_torque[{self.name}]: site {site_name!r} not found. A six-axis FT sensor must "
                f"be measured at a site on the body it is bolted to."
            )
        for suffix, attr in (("force", "_force_adr"), ("torque", "_torque_adr")):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{site_name}_{suffix}")
            if sid < 0:
                raise RuntimeError(
                    f"force_torque[{self.name}]: sensor {site_name}_{suffix!r} missing after compile"
                )
            setattr(self, attr, int(m.sensor_adr[sid]))
        self._refuse_a_tool_it_cannot_see(m, site_name, entity)
        self._flex_contacts = FlexContactWrench(m, self._sensed_bodies(m))

        if self.frame == "base":
            body_name = entity.body if entity and entity.body else f"{prefix}base"
            self._ref_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if self._ref_bid < 0:
                raise RuntimeError(
                    f"force_torque[{self.name}]: frame 'base' needs the entity's base body, but "
                    f"{body_name!r} was not found. Nest this sensor under the entry that spawns "
                    f"the entity, or use frame 'world'."
                )

        # Keyed on the LABEL, like every other identity: a document with two entries answering to
        # one label is already refused when it loads, so reaching this means a caller constructed
        # two directly.
        key = f"ft:{self.label}"
        if ctx.blackboard.get(key) is not None:
            raise RuntimeError(
                f"force_torque: blackboard key {key!r} is already registered. Two FT sensors need "
                f"distinct labels, else a controller silently reads the wrong one."
            )
        ctx.blackboard.set(
            key,
            WrenchReader(
                name=self.label,
                frame=self.frame,
                read=self.read,
                measures="environment_on_tool" if self.invert else "tool_on_environment",
                tare=self.tare,
            ),
        )

        # A broadcaster: it claims NO command interface, which is exactly why it can read the same
        # hardware a command controller is driving without blocking it.
        registry_for(ctx).register(
            Controller(
                name=self.config.get("controller_name", f"{self.name}_broadcaster"),
                type="force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster",
                reads=(f"{self.name}/force.x", f"{self.name}/force.y", f"{self.name}/force.z"),
                state=ACTIVE,
                namespace=ns,
                owner=self.owner,
            )
        )

        # The frame the wrench is stated in, as TF names it: bare, like every frame a bridge
        # publishes (it applies the namespace), so never the MJCF name with the entity's prefix.
        # `sensor` is the site's own frame, which nothing else publishes, so it comes with the
        # fixed transform from the body it is on; `base` is the entity's root body. The topic is
        # under the sensor's name, since one arm may carry several.
        ros2 = {"topic": f"{self.name}/wrench"}
        if self.frame == "sensor":
            ros2["frame_id"] = site_name.removeprefix(prefix)
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[self._site_id]))
            ros2["static_tf"] = {
                "parent": body.removeprefix(prefix),
                "translation": [float(v) for v in m.site_pos[self._site_id]],
                "rotation": [float(v) for v in m.site_quat[self._site_id]],  # (w, x, y, z)
            }
        elif self.frame == "base":
            ros2["frame_id"] = mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_BODY, self._ref_bid
            ).removeprefix(prefix)
        else:
            ros2["frame_id"] = "world"
        self._wrench_ros2 = ros2

    def _sensed_bodies(self, m) -> set[int]:
        """The bodies whose load this sensor reads: its site's body and everything below it."""
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[self._site_id]))
        return set(entity_body_ids(m, body))

    def _refuse_a_tool_it_cannot_see(self, m, site_name: str, entity) -> None:
        """Refuse a sensor on an entity whose mounted tool hangs outside the subtree it reads.

        The owning entity names its tool in ``meta["end_effector"]`` (``spawn_arm`` does). The tool
        is seen when the body carrying its mount site is in the sensed subtree -- the tool hangs
        below it then -- or when the sensor sits in the tool itself, a fingertip sensor.
        """
        tool = (entity.meta.get("end_effector") if entity else None) or {}
        if not tool:
            return
        mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tool["site"])
        if mount < 0:
            raise RuntimeError(
                f"force_torque[{self.name}]: entity {self.owner!r} names its tool's mount site "
                f"{tool['site']!r}, which the compiled model does not have"
            )
        sensed = self._sensed_bodies(m)
        in_tool = set()
        for root in tool.get("bodies", []):
            in_tool.update(entity_body_ids(m, root))
        if int(m.site_bodyid[mount]) in sensed or sensed & in_tool:
            return
        sensor_body = mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[self._site_id])
        )
        mount_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[mount]))
        raise RuntimeError(
            f"force_torque[{self.name}]: site {site_name!r} reads the subtree of body "
            f"{sensor_body!r}, and the tool mounted at {tool['site']!r} (on body {mount_body!r}) "
            f"is not in it -- the sensor would read none of the tool's weight, load or contacts. "
            f"Mount the tool at a site on {sensor_body!r} or below it (spawn_arm's "
            f"`end_effector.site`, which defaults to the site the arm model's manifest declares), "
            f"or measure at a site the tool hangs below."
        )

    def _raw(self) -> tuple[np.ndarray, np.ndarray]:
        """The transducer's pair in the sensor frame: MuJoCo's site pair with the flex contacts it
        leaves out added, then this episode's bias and drift, saturated at the range.

        MuJoCo's site sensor reports the subtree's balance as ``-(external wrench)``, so a missing
        external contact wrench ``(F, M)`` (world, at the site) is subtracted, rotated into the site.
        """
        d = self._ctx.data
        force = np.array(d.sensordata[self._force_adr : self._force_adr + 3], dtype=float)
        torque = np.array(d.sensordata[self._torque_adr : self._torque_adr + 3], dtype=float)
        if self._flex_contacts is not None and self._flex_contacts.active:
            f_world, m_world = self._flex_contacts(d, d.site_xpos[self._site_id])
            rot = np.array(d.site_xmat[self._site_id]).reshape(3, 3)
            force = force - rot.T @ f_world
            torque = torque - rot.T @ m_world
        if self.bias_f or self.bias_t or self.drift_f or self.drift_t:
            bias_f, bias_t, drift_f, drift_t = self._episode_transducer()
            t = self._ctx.sim_time
            force = force + bias_f + drift_f * t
            torque = torque + bias_t + drift_t * t
        if self.range_f is not None:
            force = np.clip(force, -self.range_f, self.range_f)
        if self.range_t is not None:
            torque = np.clip(torque, -self.range_t, self.range_t)
        return force, torque

    def _episode_transducer(self) -> tuple[np.ndarray, ...]:
        """This episode's bias and drift rate per channel, drawn once from the episode's key."""
        episode = int(self._ctx.episode)
        if self._transducer is None or self._transducer_episode != episode:
            rng = self._ctx.rng_for(self.draw_key("transducer"), per_episode=True)
            self._transducer = (
                rng.uniform(-self.bias_f, self.bias_f, 3),
                rng.uniform(-self.bias_t, self.bias_t, 3),
                rng.uniform(-self.drift_f, self.drift_f, 3),
                rng.uniform(-self.drift_t, self.drift_t, 3),
            )
            self._transducer_episode = episode
        return self._transducer

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """``(force[3], torque[3])`` in the configured frame. Runs on the physics thread."""
        force, torque = self._raw()
        if not self._tared and self.tare_at_s is not None and self._ctx.sim_time >= self.tare_at_s:
            # Captured here, from the raw pair, so the offset is the reading's MEAN: the noise is
            # added below and is what remains after the subtraction. A tare taken after the noise
            # would bake one sample's draw into every reading for the rest of the episode.
            self._offset_force, self._offset_torque = force.copy(), torque.copy()
            self._tared = True
        force = force - self._offset_force
        torque = torque - self._offset_torque
        if self.invert:
            force, torque = -force, -torque
        if self.frame != "sensor":
            rot = np.array(self._ctx.data.site_xmat[self._site_id]).reshape(3, 3)
            if self.frame == "base":
                # world = R_site @ v; base = R_base^T @ world
                base_rot = np.array(self._ctx.data.xmat[self._ref_bid]).reshape(3, 3)
                rot = base_rot.T @ rot
            force, torque = rot @ force, rot @ torque
        if self.noise_f or self.noise_t:
            # One generator per (sensor, step) -- counter-based, so every reader in this step draws the
            # same wrench and the value is reproducible from a recording. Keyed on the instance's
            # address, not its name, so two arms' unnamed sensors draw independent streams.
            rng = self._ctx.rng_for(self.draw_key("noise"))
            if self.noise_f:
                force = force + rng.normal(0.0, self.noise_f, 3)
            if self.noise_t:
                torque = torque + rng.normal(0.0, self.noise_t, 3)
        return force, torque

    @endpoint.out(rate="rate_hz", ros2=lambda self: self._wrench_ros2)
    def wrench(self) -> Wrench:
        """The wrench, in the frame ``frame`` names."""
        force, torque = self.read()
        return Wrench(force, torque)

    # A command without parameters, which ROS carries as a `Trigger` service: this is the zero
    # button, which a real FT driver also exposes as a service taking no argument
    # (`zero_ftsensor`). A caller needs the outcome -- a scenario that tared and carried on
    # regardless would measure against an offset it only assumed was applied.
    @endpoint.command(ros2=lambda self: {"name": f"{self.name}/tare"})
    def tare(self) -> None:
        """Zero the sensor at the tool's CURRENT pose and load. Physics thread only.

        What a real sensor's zero button does, with the same limit: this cancels the load as it is
        right now, not the tool's weight at every pose. See "Taring" in the module docstring.

        Re-tares an already-tared sensor, deliberately: an approach that tares per contact is the
        way to use this on a tool that turns, and refusing the second call would make that the one
        thing it cannot do.
        """
        # From the raw pair rather than through `read`, which has already subtracted whatever
        # offset is standing -- taring twice would otherwise capture the residual and leave the
        # first tare's offset in place forever.
        self._offset_force, self._offset_torque = self._raw()
        self._tared = True

    def on_reset(self, ctx: SimContext) -> None:
        """Forget the zero, so the next episode captures its own.

        An offset carried across a reset is a measurement of the previous episode, and the whole
        point of a repetition is that it repeats -- the same reason the noise is keyed on the
        episode. A `tare_at_s` sensor re-arms and tares again at that time.
        """
        self._offset_force = np.zeros(3)
        self._offset_torque = np.zeros(3)
        self._tared = False
