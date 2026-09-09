"""Sensor plugin: a six-axis force/torque sensor at a site.

The substrate's first *contact-force* observable. Every sensor here so far reports geometry —
where things are (lidar, cameras, fiducials, ground-truth pose). None of them reports what a robot
is pushing against, and for a contact-rich manipulation task that is the whole measurement: an
insertion, a polishing pass, or a compliant assembly is judged by its wrench, not by its trajectory.

MuJoCo computes the constraint wrench already; a ``<force>``/``<torque>`` sensor pair on a site
reads it, and this plugin turns that pair into a first-class observable — a rate-limited endpoint, a
blackboard reader for in-process controllers, and an optional per-trial log. Nothing here is novel
physics. It is the plumbing that was missing.

**Where the sensor goes matters more than it looks.** A site sensor measures the wrench transmitted
*through* that site's body from its children, so it must sit on a body between the tool flange and
whatever touches the world — exactly where a real FT sensor is bolted. Put it on the flange itself
and the tool's own contacts are on the wrong side of the cut, and the sensor reads nothing.

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
      rate_hz: 100.0            # endpoint publish rate
      namespace: ""             # transport scope (default: inherited from the entity)
      topics: {wrench: /ft}     # optional absolute-topic hardwire

Endpoint ``tare`` (in) is that zero button as a service; it takes no argument and its reply is
what lets a scenario fail rather than measure against an offset it only assumed was applied.

Endpoint ``wrench`` (out) reads ``(force[3], torque[3])`` and carries a
``geometry_msgs/WrenchStamped`` backend hint. A ``WrenchReader`` is published on the blackboard
under ``ft:<entry label>`` for in-process consumers — the admittance controller is one — exposing
``read()`` and the resolved ``frame``.

**Frames.** ``sensor`` returns MuJoCo's raw site-frame reading. ``base`` rotates it into the owning
entity's base body frame, and ``world`` into the world frame. The choice is not cosmetic for
metrics that split the wrench into an insertion axis and the plane orthogonal to it: ``|F_z|`` and
``||F_x, F_y||`` are frame-dependent, and a tool that tilts reports a different split in its own
frame than in the world's.

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

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

_FRAMES = ("sensor", "base", "world")


@dataclass
class WrenchReader:
    """Blackboard handle published under ``ft:<name>``; consumed on the physics thread.

    ``read()`` returns ``(force[3], torque[3])`` as numpy arrays in ``frame``. ``frame`` is carried
    with the reader because a consumer that integrates the wrench into a motion command has to know
    which frame it is commanding in, and getting that wrong produces a controller that pushes in a
    plausible-looking wrong direction rather than one that fails.

    ``measures`` is carried for exactly the same reason, and was missing for longer: a wrench has a
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

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if not config.get("site"):
            errors.append("'site' is required: name the MJCF site the wrench is measured at")
        if config.get("frame", "sensor") not in _FRAMES:
            errors.append(f"'frame' must be one of {', '.join(_FRAMES)}")
        if float(config.get("rate_hz", 100.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        for key in ("noise_force_stddev", "noise_torque_stddev"):
            if float(config.get(key, 0.0)) < 0:
                errors.append(f"'{key}' must be >= 0")
        if config.get("tare_at_s") is not None and float(config["tare_at_s"]) < 0:
            errors.append("'tare_at_s' must be >= 0: it is a sim time, not an offset")
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

        ctx.interface.add(
            Endpoint(
                name="tare",
                direction="in",
                owner=self.owner,
                namespace=ns,
                # The payload is ignored: zeroing takes no argument, and the call IS the request.
                write=lambda _payload=None: ctx.post(lambda _ctx: self.tare()),
                backend={
                    "ros2": {
                        # A service, not a topic, and `Trigger` rather than `SetBool`: this is the
                        # zero button, which a real FT driver also exposes as a service taking no
                        # argument (`zero_ftsensor`). A caller needs the outcome -- a scenario that
                        # tared and carried on regardless would measure against an offset it only
                        # assumed was applied.
                        "service": "std_srvs.srv.Trigger",
                        "name": self.topic_override("tare") or f"{self.name}/tare",
                    }
                },
            )
        )

        ctx.interface.add(
            Endpoint(
                name="wrench",
                direction="out",
                owner=self.owner,
                namespace=ns,
                read=self.read_pair,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.WrenchStamped",
                        "topic": self.topic_override("wrench") or f"{self.name}/wrench",
                        "frame_id": site_name if self.frame == "sensor" else self.frame,
                    }
                },
            )
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """``(force[3], torque[3])`` in the configured frame. Runs on the physics thread."""
        d = self._ctx.data
        force = np.array(d.sensordata[self._force_adr : self._force_adr + 3], dtype=float)
        torque = np.array(d.sensordata[self._torque_adr : self._torque_adr + 3], dtype=float)
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
            # same wrench and the value is reproducible from a recording. Keyed with an `ft:` prefix so
            # a lidar named `ft` cannot share the stream.
            rng = self._ctx.rng_for(f"ft:{self.name}")
            if self.noise_f:
                force = force + rng.normal(0.0, self.noise_f, 3)
            if self.noise_t:
                torque = torque + rng.normal(0.0, self.noise_t, 3)
        return force, torque

    def read_pair(self):
        """Endpoint ``read``: the same wrench as plain lists, for a transport-neutral payload."""
        force, torque = self.read()
        return (force.tolist(), torque.tolist())

    def tare(self) -> None:
        """Zero the sensor at the tool's CURRENT pose and load. Physics thread only.

        What a real sensor's zero button does, with the same limit: this cancels the load as it is
        right now, not the tool's weight at every pose. See "Taring" in the module docstring.

        Re-tares an already-tared sensor, deliberately: an approach that tares per contact is the
        way to use this on a tool that turns, and refusing the second call would make that the one
        thing it cannot do.
        """
        d = self._ctx.data
        # From the raw pair rather than through `read`, which has already subtracted whatever
        # offset is standing -- taring twice would otherwise capture the residual and leave the
        # first tare's offset in place forever.
        self._offset_force = np.array(
            d.sensordata[self._force_adr : self._force_adr + 3], dtype=float
        )
        self._offset_torque = np.array(
            d.sensordata[self._torque_adr : self._torque_adr + 3], dtype=float
        )
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
