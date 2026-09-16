"""Bridge plugin: PX4 SITL flies this airframe, over PX4's simulator-agnostic MAVLink HIL API.

This is what makes roqsim stand in for Gazebo as PX4's physics backend, so an experiment can fly
the flight stack people actually deploy instead of a Python controller that resembles one.

**The analysis, because the choice of interface is the whole design.** PX4's default SITL simulator
is gz-sim (Gazebo Harmonic): PX4 ships a ``gz_bridge`` module that attaches to it *natively* over
gz-transport, exchanging gz messages for IMU, motor commands and clock. That bridge is specific to
gz and reusable by nothing else -- there is no way for a non-gz simulator to present itself to it.
Every other simulator PX4 supports (jMAVSim, Gazebo Classic, AirSim, FlightGear) instead attaches
through the documented, *simulator-agnostic* **Simulator MAVLink API**: the simulator listens on
TCP 4560, PX4 SITL connects out to it, and the two exchange a fixed message set. That is the
interface this plugin speaks, and it is the only one that makes roqsim a first-class PX4 backend
rather than a fork of PX4.

What is therefore **in scope here**: producing the vehicle's sensed state (IMU, magnetometer,
barometer, GNSS via the :mod:`roqsim_aerial.plugins.gnss` plugin), sending ground truth, receiving
motor commands, and driving lockstep. What is **PX4's job and not ours**: mixing (PX4's airframe
config decides which rotor is which -- the reason the x500 model's actuator order is load-bearing),
attitude and position control, EKF2 state estimation, arming logic, failsafes, and the offboard
interface ROS 2 talks to. We supply physics; PX4 supplies the autopilot. Nothing in this file
should ever grow a control law.

**Deliberately not simulated, compared with what gz gives PX4** -- these are the documented gaps,
not oversights: camera and depth streams into PX4, optical flow, downward distance/range sensing,
and RC input. The messages that would carry them are ``HIL_OPTICAL_FLOW``, ``DISTANCE_SENSOR`` and
``RC_CHANNELS``; an experiment on flow-aided or terrain-relative navigation needs them added here
before it means anything. Airspeed (``diff_pressure``) is sent as zero, so a fixed-wing airframe is
out of scope too.

Config::

    px4_sitl:
      port: 4560                 # PX4's documented simulator port; PX4 connects OUT to us
      bind: "127.0.0.1"          # loopback: PX4 SITL runs beside the simulator
      lockstep: true
      body: x500                 # the flown body (default: the entity root)
      imu_rate: 250.0            # Hz -- HIL_SENSOR cadence
      mag_field: {north: 0.21, east: 0.0, down: 0.43}     # gauss, world NED
      baro: {sea_level_pressure: 1013.25, temperature: 20.0}   # hPa, degC
      sensor_noise: {accel: 0.02, gyro: 0.002, mag: 0.002, baro: 0.05}   # 1-sigma
      connect_timeout: 60.0      # s to wait for PX4 before failing
      ground_truth: true         # send HIL_STATE_QUATERNION

**PX4 is the client, we are the server.** Confirmed against PX4's own SITL startup path, which
runs ``simulator_mavlink start -c <port>`` (or ``-h``/``-t`` with ``PX4_SIM_HOSTNAME`` /
``PX4_SIM_HOST_ADDR``): the module dials out to the simulator. The port is ``4560 + instance``, so
the default here is right for a single vehicle and a second PX4 instance wants ``port: 4561``.

**250 Hz is not a guess.** The same startup path sets ``IMU_INTEG_RATE`` to 250, which is the rate
PX4 integrates HIL_SENSOR at; the default here matches it. That path also sets
``SENS_GPS0_DELAY``/``SENS_GPS1_DELAY`` to 10 ms, i.e. EKF2 is told to expect a fix that is 10 ms
old -- so timestamping ``HIL_GPS`` with the current sim time, as this plugin does, is already
consistent with PX4's assumption and no artificial delay belongs here.

**Thrust numbers live in the model, never here.** ``HIL_ACTUATOR_CONTROLS`` is normalized 0..1 and
goes straight through ``MotorsHandle.set_normalized``; what a 1.0 is worth in newtons is the MJCF's
``ctrlrange``, which ``multirotor_motors`` reads and exposes as ``max_thrust``. A number duplicated
here would be one that could disagree with the airframe being flown.

**``imu_rate`` must divide sensibly into the sim rate.** Sensors are sent on whole ticks, so the
achieved rate is ``1 / (ceil(1 / (imu_rate * dt)) * dt)``: at the usual ``dt = 0.002`` s an
``imu_rate`` of 250 Hz lands exactly (every 2nd tick), 300 Hz silently becomes 250. The plugin logs
the rate it will actually achieve rather than the one that was asked for.

**Zero-noise sensors are not the neutral choice.** It is tempting to send the physics engine's
exact values and call the result "the ideal case", but EKF2 is *tuned* against noisy inputs: its
process and measurement covariances assume a certain amount of jitter, and a perfect IMU is an
out-of-distribution input for it -- innovation gates behave differently, covariances collapse, and
the estimator's behaviour stops being the one that flies on hardware. The defaults here are small
but non-zero for that reason; set them to zero deliberately and knowing that the run no longer says
anything about the estimator.

**The magnetic field default** ``[0.21, 0.0, 0.43]`` G (north, east, down) is a representative
mid-latitude northern-hemisphere IGRF value (~0.48 G total, ~64 degrees inclination, declination
taken as zero). It is a *stand-in for the datum's actual field*, not a computed one: this plugin
does not carry an IGRF model, so an experiment whose result depends on declination must set the
field for its datum explicitly.

**Frames.** MuJoCo is ENU (world) / FLU (body); MAVLink HIL is NED (world) / FRD (body). Every
conversion goes through :func:`enu_to_ned`, :func:`flu_to_frd` and :func:`quat_enu_flu_to_ned_frd`,
and nowhere else. This is not fussiness: a sign error here produces a drone that flies confidently
in the wrong direction, arms and takes off exactly as it should, and is the single most common way
this integration is got wrong. One helper pair means one place to check.

**Threading follows architecture.rst section 7 exactly.** The socket is opened in ``configure``
(which is where that section puts socket setup) and served by one background thread, and that
thread **never touches** ``ctx.data``, the motors handle, or any plugin state the physics thread
reads. It parses frames and enqueues the result with ``ctx.post(...)``; the engine drains the queue
at the start of the next ``pre_step``, on the physics thread, in FIFO order. So the actuator update
lands at a defined point in the tick rather than whenever the OS scheduled the reader.

**Lockstep is architecture.rst section 10** -- the designed synchronous mode, of which this plugin
is the first real user. It registers a **consumer gate** (``ctx.register_gate``), pending until the
expected input arrives via ``ctx.post``, exactly as that section specifies. The engine does not yet
wait on gates ("today ``register_gate``/``gates`` exist and are reset each ``reset()``, but nothing
waits on them"), so the wait itself is implemented here, in ``post_step``, against that gate --
with the timeout and the deadlock diagnostic section 10 requires, naming the gate that never fired.
It is not a second concurrency scheme: the command queue is still the substrate, and this only adds
the wait. **Blocking inside ``post_step`` is listed as an anti-pattern "except deliberately in sync
mode"** -- this is that exception, taken deliberately, and it is the reason the block is confined to
one clearly named place. When the engine grows its own gate wait, this method becomes a
``gate.satisfy`` and nothing else changes.

**Lockstep arms only after PX4's first answer, and that is not a convenience.** PX4 SITL runs on a
``lockstep_scheduler`` whose clock is set from the timestamps in the ``HIL_SENSOR`` messages we
send: nothing in PX4 makes progress until sim time advances, *including its own startup script* and
therefore every module that could ever publish ``actuator_outputs`` -- the topic
``HIL_ACTUATOR_CONTROLS`` is emitted from. Blocking on the very first batch is consequently a true
deadlock, not a slow start: the bridge waits for controls that require a boot that requires the
sensor stream the bridge has just stopped. Verified against PX4 v1.18.0-beta2, which sits at
``ERROR [simulator_mavlink] poll timeout 0, 25`` forever. PX4's own gazebo-classic bridge carries
exactly this gate (``received_first_actuator_``). So the first ticks free-run; from the first
``HIL_ACTUATOR_CONTROLS`` onward every tick is locked, and the gate stays armed across a reset
because PX4 did not reboot.

What is locked to what: on each tick that carries a sensor batch, the plugin sends ``HIL_SENSOR``
(plus ``HIL_GPS`` at the receiver's own rate) and then blocks until PX4's ``HIL_ACTUATOR_CONTROLS``
for that batch has been received and queued; PX4 in turn blocks until the next ``HIL_SENSOR``.
Neither side runs ahead, so the number of physics steps between two actuator updates is fixed by
construction. **With ``lockstep: false`` the run is NOT reproducible**: how many steps elapse
between two actuator updates then depends on host load and OS scheduling, so two runs of the
identical world with the identical seed diverge, and repetitions stop being samples of anything.
Non-lockstep exists only for interactive flying, where a stall in one process should not freeze the
other. (The sensor noise itself is reproducible either way -- it draws from ``ctx.rng_for``, keyed
on ``(seed, episode, sim_time)`` per section 9, like every other stochastic plugin here. It is the
*coupling* that lockstep makes deterministic.)

**MAVLink encoding is pymavlink's.** Imported lazily in :meth:`configure` so that loading a world
containing this plugin (to render it, to export it) does not require the dependency, and so the
error names the package. Hand-rolling a serialiser for a dialect that is generated from XML and
versioned upstream would be a maintenance liability disguised as a saved dependency.
"""

from __future__ import annotations

import logging
import socket
import threading

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin

logger = logging.getLogger(__name__)

#: MAV_MODE_FLAG_SAFETY_ARMED. PX4 keeps streaming controls while disarmed; the flag is the only
#: thing distinguishing "spin at idle" from "do not spin".
MAV_MODE_FLAG_SAFETY_ARMED = 128

#: HIL_SENSOR ``fields_updated`` bits: accel xyz (0-2), gyro xyz (3-5), mag xyz (6-8),
#: abs_pressure (9), pressure_alt (11), temperature (12). Diff pressure (10) is left clear because
#: this plugin does not model airspeed.
FIELDS_ACCEL_GYRO_MAG = 0x1FF
FIELDS_BARO = (1 << 9) | (1 << 11) | (1 << 12)
FIELDS_UPDATED = FIELDS_ACCEL_GYRO_MAG | FIELDS_BARO

_DEFAULTS = {
    "port": 4560,
    "bind": "127.0.0.1",
    "lockstep": True,
    "imu_rate": 250.0,
    "connect_timeout": 60.0,
    "ground_truth": True,
    "gyro_sensor": "body_gyro",
    "accel_sensor": "body_linacc",
    "quat_sensor": "body_quat",
}
_DEFAULT_MAG = {"north": 0.21, "east": 0.0, "down": 0.43}
_DEFAULT_BARO = {"sea_level_pressure": 1013.25, "temperature": 20.0}
_DEFAULT_NOISE = {"accel": 0.02, "gyro": 0.002, "mag": 0.002, "baro": 0.05}

#: How long a lockstep tick waits for PX4's answer before giving up (section 10's ``sync.timeout_s``,
#: which this plugin cannot yet read because the engine does not drive sync mode). Long enough to
#: survive a garbage-collection pause on either side, short enough that a dead autopilot fails the
#: run in half a minute rather than hanging a campaign slot until somebody notices.
LOCKSTEP_TIMEOUT_S = 30.0


# -- frame conversions -----------------------------------------------------------------------
# MuJoCo world is ENU (x East, y North, z Up); MAVLink world is NED (x North, y East, z Down).
# MuJoCo bodies are FLU (x Forward, y Left, z Up); MAVLink bodies are FRD (x Forward, y Right,
# z Down). The two transforms are DIFFERENT -- the world one swaps x and y, the body one does not --
# and conflating them is the classic failure: it yields a drone whose yaw looks right and whose roll
# is mirrored.


def enu_to_ned(v) -> np.ndarray:
    """World vector: (East, North, Up) -> (North, East, Down)."""
    e, n, u = (float(x) for x in v)
    return np.array([n, e, -u])


def flu_to_frd(v) -> np.ndarray:
    """Body vector: (Forward, Left, Up) -> (Forward, Right, Down)."""
    f, left, up = (float(x) for x in v)
    return np.array([f, -left, -up])


#: The two conversions as matrices, for rotating an attitude rather than a vector. Both are proper
#: rotations (det +1): T_WORLD is 180 degrees about the East+North diagonal, T_BODY is 180 degrees
#: about the body's forward axis.
T_WORLD = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
T_BODY = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def quat_enu_flu_to_ned_frd(quat) -> np.ndarray:
    """Attitude quaternion (w, x, y, z): ENU<-FLU -> NED<-FRD.

    A rotation is a map between two frames, so BOTH ends must be converted:
    ``R_ned<-frd = T_WORLD @ R_enu<-flu @ T_BODY^-1``, and each T is its own inverse.
    """
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    converted = T_WORLD @ rot.reshape(3, 3) @ T_BODY
    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(converted, dtype=float).reshape(9))
    return out


#: Standard gravity, m/s^2. HIL_STATE_QUATERNION reports acceleration in milli-g, not in m/s^2 --
#: the one field in this message set that is not SI -- so the conversion needs the constant.
STANDARD_GRAVITY = 9.80665


def clamp_i16(value: float) -> int:
    """Saturate to int16. HIL_STATE_QUATERNION's acceleration fields are int16, and a landing-gear
    contact easily produces a spike outside them; a ground-truth message that raises rather than
    saturating would abort a run over a number nothing flies on."""
    return int(max(-32768, min(32767, round(value))))


def clamp_u16(value: float) -> int:
    """Saturate to uint16, for HIL_GPS's unsigned accuracy and ground-speed fields."""
    return int(max(0, min(65535, round(value))))


def pressure_from_altitude(altitude_m: float, sea_level_hpa: float) -> float:
    """ISA barometric formula up to 11 km, in hPa. PX4 derives its baro altitude by inverting it."""
    return float(sea_level_hpa * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588)


class _SocketWriter:
    """The file-like object pymavlink's ``MAVLink`` writes encoded frames into."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def write(self, buf) -> None:
        self._sock.sendall(buf)


class Px4SitlPlugin(Plugin):
    #: Flies one entity's airframe, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    #: NOT parallel-safe, and not merely because it writes: it binds a FIXED TCP port, which is a
    #: process-wide (indeed host-wide) singleton. Two of these in one process is a port conflict,
    #: and the failure is made explicit in `configure` rather than left to a stray EADDRINUSE.
    parallel_safe = False

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self._sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._closing = threading.Event()
        self._mav = None
        self._armed = False
        self._controls = [0.0] * 16
        self._tick = 0
        self._next_gps = 0.0
        self._ctx: SimContext | None = None
        self._gate = None
        #: Set by the receive thread when a HIL_ACTUATOR_CONTROLS has been parsed AND its
        #: application posted. The physics thread waits on this, never on the socket.
        self._controls_arrived = threading.Event()
        #: Set once PX4 has EVER answered. Lockstep cannot be armed before that: see `post_step`.
        self._ever_answered = threading.Event()

    def cfg(self, key):
        return self.config.get(key, _DEFAULTS[key])

    # -- config ----------------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if "port" in config and not 1 <= int(config["port"]) <= 65535:
            errors.append("'port' must be a TCP port in [1, 65535]")
        for key in ("imu_rate", "connect_timeout"):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        mag = config.get("mag_field")
        if mag is not None:
            if not isinstance(mag, dict):
                errors.append("'mag_field' must be a mapping {north, east, down} in gauss")
            elif set(mag) - {"north", "east", "down"}:
                errors.append("'mag_field' keys are 'north', 'east', 'down' (gauss, world NED)")
        baro = config.get("baro")
        if baro is not None:
            if not isinstance(baro, dict):
                errors.append("'baro' must be a mapping {sea_level_pressure, temperature}")
            elif set(baro) - {"sea_level_pressure", "temperature"}:
                errors.append("'baro' keys are 'sea_level_pressure' (hPa) and 'temperature' (degC)")
        noise = config.get("sensor_noise")
        if noise is not None:
            if not isinstance(noise, dict):
                errors.append("'sensor_noise' must be a mapping {accel, gyro, mag, baro}")
            else:
                unknown = set(noise) - set(_DEFAULT_NOISE)
                if unknown:
                    errors.append(f"unknown 'sensor_noise' keys: {sorted(unknown)}")
                for key, value in noise.items():
                    if key in _DEFAULT_NOISE and float(value) < 0:
                        errors.append(f"'sensor_noise.{key}' must be >= 0")
        if "seed" in config:
            errors.append(
                "'seed' is not a key here: sensor noise draws from the run's seed (sim.seed / "
                "roqsim sim --seed) through ctx.rng_for, so the whole world reproduces together"
            )
        return errors

    # -- lifecycle -------------------------------------------------------------------------------
    def _resolve_body(self, entity, prefix: str) -> str:
        """The flown body, as a name in the COMPILED model.

        ``Entity.body`` is the root body ``spawn_robot`` resolved against the compiled model, so it
        is already prefixed; a config ``body:`` names a body in the MODEL's namespace and takes the
        prefix. Applying it to both would look for a doubly-prefixed body that cannot exist.
        """
        configured = self.config.get("body")
        if configured:
            return prefix + str(configured)
        if entity is not None and entity.body:
            return entity.body
        raise RuntimeError(
            f"px4_sitl ({self.name}): no body to fly -- entity {self.robot!r} has no root body, "
            f"so name one in this plugin's 'body:'"
        )

    def configure(self, ctx: SimContext) -> None:
        self._mavlink_module = self._load_mavlink()

        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        model = ctx.model

        body = self._resolve_body(entity, prefix)
        self._bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self._bid < 0:
            raise RuntimeError(
                f"px4_sitl ({self.name}): body {body!r} is not in the compiled model"
            )

        self._sensor_adr = {}
        for role in ("gyro_sensor", "accel_sensor", "quat_sensor"):
            sensor_name = prefix + str(self.cfg(role))
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sid < 0:
                raise RuntimeError(
                    f"px4_sitl ({self.name}): the airframe has no {sensor_name!r} sensor. PX4's "
                    f"EKF2 is driven by HIL_SENSOR, so the MJCF must carry gyro, accelerometer and "
                    f"framequat sensors on the IMU site; name them in '{role}:' if they differ."
                )
            self._sensor_adr[role] = (int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))

        # The two handles this bridge consumes. Both are hard requirements, and both are checked
        # HERE rather than at the first tick, because a drone that has taken off before the failure
        # surfaces has already invalidated the trial.
        self._motors = ctx.blackboard.get(f"motors:{self.robot}")
        if self._motors is None:
            raise RuntimeError(
                f"px4_sitl ({self.name}): no 'motors:{self.robot}' handle. HIL_ACTUATOR_CONTROLS "
                f"is a per-rotor command, so this airframe needs the 'multirotor_motors' plugin to "
                f"give it per-rotor actuation for PX4's mixer to drive -- without it there is "
                f"nothing for the mixer's outputs to mean. Add multirotor_motors to this entity's "
                f"components, before px4_sitl."
            )
        self._gnss = ctx.blackboard.get(f"gnss:{self.robot}")
        if self._gnss is None:
            raise RuntimeError(
                f"px4_sitl ({self.name}): no 'gnss:{self.robot}' handle. HIL_GPS is not optional "
                f"for EKF2 in its default configuration -- with EKF2_GPS_CTRL at its default the "
                f"estimator will not complete its position alignment, the vehicle never becomes "
                f"armable, and the run looks like a PX4 fault. Add the 'gnss' plugin (it also "
                f"carries the world's datum) to this entity's components, before px4_sitl."
            )

        dt = ctx.dt
        self._sensor_period_ticks = max(1, int(round(1.0 / (float(self.cfg("imu_rate")) * dt))))
        achieved = 1.0 / (self._sensor_period_ticks * dt)
        if abs(achieved - float(self.cfg("imu_rate"))) > 1e-6:
            logger.warning(
                "px4_sitl (%s): imu_rate %.1f Hz does not divide the %.4f s timestep; sending "
                "HIL_SENSOR every %d ticks, i.e. at %.1f Hz.",
                self.name,
                float(self.cfg("imu_rate")),
                dt,
                self._sensor_period_ticks,
                achieved,
            )

        mag = {**_DEFAULT_MAG, **(self.config.get("mag_field") or {})}
        self._mag_ned = np.array([mag["north"], mag["east"], mag["down"]], dtype=float)
        self._baro = {**_DEFAULT_BARO, **(self.config.get("baro") or {})}
        self._noise = {**_DEFAULT_NOISE, **(self.config.get("sensor_noise") or {})}
        self._datum_alt = float(self._gnss.datum["alt"])

        self._ctx = ctx
        # Section 10's consumer gate: pending until the expected input arrives via ctx.post. The
        # engine does not wait on gates yet, so post_step does -- but the seam gets its user, and a
        # future engine-side wait needs no change here.
        self._gate = ctx.register_gate(f"px4_sitl:{self.name}:controls", "consumer")

        self._listen()
        self._await_connection()

    def on_reset(self, ctx: SimContext) -> None:
        self._tick = 0
        self._next_gps = 0.0
        self._controls_arrived.clear()
        # `_ever_answered` is NOT cleared: PX4 did not reboot, so the lockstep it is already
        # driving must stay armed across an episode boundary.
        # The armed state and the last controls are NOT cleared: PX4 is a separate process that did
        # not reset, and pretending it disarmed would command zero thrust to a vehicle its
        # autopilot still believes is flying.

    def post_step(self, ctx: SimContext) -> None:
        """Send this tick's sensors and, in lockstep, wait for PX4's answer before returning.

        ``post_step`` and not ``pre_step``: the sensors must describe the state the step produced,
        and the controls that come back are applied by ``multirotor_motors`` on the next tick --
        which is exactly the one-sample actuation delay a real autopilot has.
        """
        self._tick += 1
        if self._conn is None:
            return
        if self._tick % self._sensor_period_ticks:
            return

        self._controls_arrived.clear()
        if self._gate is not None:
            self._gate.reset()
        self._send_sensors(ctx)
        if self._gnss.read_fix()["valid"] and ctx.sim_time + 1e-12 >= self._next_gps:
            self._next_gps = ctx.sim_time + 1.0 / max(float(self._gnss.rate), 1e-6)
            self._send_gps(ctx)
        if bool(self.cfg("ground_truth")):
            self._send_ground_truth(ctx)

        if not bool(self.cfg("lockstep")):
            return
        if not self._ever_answered.is_set():
            # PX4 HAS NOT BOOTED YET, and blocking here would guarantee it never does. PX4 SITL runs
            # on a lockstep_scheduler whose clock is driven by the HIL_SENSOR timestamps we send:
            # every module -- including the startup script that starts them -- only makes progress
            # as that clock advances. HIL_ACTUATOR_CONTROLS is emitted from `actuator_outputs`,
            # which does not exist until the control chain is running. So blocking on the first
            # batch is a true deadlock: we wait for controls that require a boot that requires the
            # sensor stream we have stopped sending. PX4's own gazebo-classic bridge has exactly
            # this gate (`received_first_actuator_`) for exactly this reason. Free-run until the
            # autopilot answers once; from then on every tick is locked.
            return
        # The deliberate sync-mode block (section 10). Nothing else in this plugin waits.
        if not self._controls_arrived.wait(LOCKSTEP_TIMEOUT_S):
            gate = self._gate.name if self._gate is not None else "px4_sitl:controls"
            raise RuntimeError(
                f"px4_sitl ({self.name}): lockstep timed out after {LOCKSTEP_TIMEOUT_S:.0f} s. "
                f"Gate {gate!r} never fired: no HIL_ACTUATOR_CONTROLS arrived on port "
                f"{self._port} for the sensor batch sent at t={ctx.sim_time:.3f} s. PX4 has "
                f"stopped answering, and the simulation cannot advance without it. Failing rather "
                f"than free-running, which would silently turn a locked run into an unreproducible "
                f"one."
            )

    def shutdown(self, ctx: SimContext) -> None:
        self._closing.set()
        for sock in (self._conn, self._sock):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        self._conn = self._sock = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None

    # -- the socket ------------------------------------------------------------------------------
    def _load_mavlink(self):
        """The MAVLink 2 dialect module. Imported here so a world can be loaded without pymavlink."""
        try:
            from pymavlink.dialects.v20 import common as mavlink2
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                f"px4_sitl ({self.name}) needs the 'pymavlink' package to speak PX4's simulator "
                f"MAVLink API; it is not installed. Install pymavlink in the environment running "
                f"roqsim. (Hand-rolling the encoding is not an option: the dialect is generated "
                f"from upstream XML and versioned with PX4.)"
            ) from exc
        return mavlink2

    def _listen(self) -> None:
        port, host = int(self.cfg("port")), str(self.cfg("bind"))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR so a re-run is not blocked by the previous run's TIME_WAIT. It does NOT
        # mask a live listener: a second px4_sitl on the same port still fails below, which is the
        # collision we want named.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"px4_sitl ({self.name}): cannot listen on {host}:{port} ({exc}). PX4's simulator "
                f"port is a host-wide singleton, so this is either a second px4_sitl in this world "
                f"or another simulation still holding it. Give each vehicle its own 'port:' and "
                f"point its PX4 instance at it."
            ) from exc
        sock.listen(1)
        self._sock = sock
        self._port = sock.getsockname()[1]  # resolved, so port 0 (tests) reports the real one
        self._accept_thread = threading.Thread(
            target=self._accept, name=f"px4_sitl-{self.name}", daemon=True
        )
        self._accept_thread.start()
        logger.info("px4_sitl (%s): listening on %s:%d for PX4 SITL", self.name, host, self._port)

    def _accept(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:  # closed during shutdown
            return
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn = conn
        self._mav = self._mavlink_module.MAVLink(_SocketWriter(conn), srcSystem=1, srcComponent=1)
        # Robust parsing: a partial frame at connect time is normal, and the default parser raises
        # on it rather than resynchronising.
        self._mav.robust_parsing = True
        conn.settimeout(0.2)  # so the loop below can notice shutdown
        self._connected.set()
        # The same thread goes on to serve the connection: one thread for the socket, and it never
        # touches anything the physics thread reads (architecture.rst section 7).
        self._receive()

    def _await_connection(self) -> None:
        """Block the sim until PX4 is attached. A drone flying with no flight stack looks exactly
        like a drone whose flight stack crashed, and the difference is a whole run's worth of
        debugging -- so the run does not start until the autopilot is there."""
        timeout = float(self.cfg("connect_timeout"))
        if not self._connected.wait(timeout):
            raise RuntimeError(
                f"px4_sitl ({self.name}): no PX4 SITL connected to port {self._port} within "
                f"{timeout:.0f} s. The simulator listens and PX4 connects out to it, so start PX4 "
                f"with its simulator host/port pointed here (PX4_SIM_HOSTNAME / the instance's "
                f"simulator TCP port)."
            )
        logger.info("px4_sitl (%s): PX4 connected on port %d", self.name, self._port)

    # -- outbound --------------------------------------------------------------------------------
    def _state(self, ctx: SimContext):
        data = ctx.data
        adr, dim = self._sensor_adr["gyro_sensor"]
        gyro_flu = np.array(data.sensordata[adr : adr + dim], dtype=float)
        adr, dim = self._sensor_adr["accel_sensor"]
        accel_flu = np.array(data.sensordata[adr : adr + dim], dtype=float)
        adr, dim = self._sensor_adr["quat_sensor"]
        quat = np.array(data.sensordata[adr : adr + dim], dtype=float)
        pos_enu = np.array(data.xpos[self._bid], dtype=float)
        vel_enu = np.array(data.cvel[self._bid][3:6], dtype=float)
        return gyro_flu, accel_flu, quat, pos_enu, vel_enu

    def _send_sensors(self, ctx: SimContext) -> None:
        gyro_flu, accel_flu, quat, pos_enu, _ = self._state(ctx)
        rng = ctx.rng_for(f"px4_sitl:{self.name}")

        gyro = flu_to_frd(gyro_flu) + self._noise["gyro"] * rng.standard_normal(3)
        accel = flu_to_frd(accel_flu) + self._noise["accel"] * rng.standard_normal(3)

        # The magnetometer measures the WORLD field expressed in the BODY frame, so the body-frame
        # value is the field rotated by the inverse attitude -- not the field converted axis-wise.
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot_ned_frd = T_WORLD @ rot.reshape(3, 3) @ T_BODY
        mag = rot_ned_frd.T @ self._mag_ned + self._noise["mag"] * rng.standard_normal(3)

        altitude = self._datum_alt + float(pos_enu[2])
        pressure = pressure_from_altitude(altitude, float(self._baro["sea_level_pressure"]))
        pressure += self._noise["baro"] * float(rng.standard_normal())

        self._mav.hil_sensor_send(
            int(ctx.sim_time * 1e6),
            *(float(v) for v in accel),
            *(float(v) for v in gyro),
            *(float(v) for v in mag),
            pressure,
            0.0,  # diff_pressure: no airspeed model, and the bit is left clear
            altitude,
            float(self._baro["temperature"]),
            FIELDS_UPDATED,
            0,  # sensor instance id; PX4 locks its lockstep to id 0
        )

    def _send_gps(self, ctx: SimContext) -> None:
        fix = self._gnss.read_fix()
        self._mav.hil_gps_send(
            int(ctx.sim_time * 1e6),
            int(fix["fix_type"]),
            int(round(fix["lat"] * 1e7)),
            int(round(fix["lon"] * 1e7)),
            int(round(fix["alt"] * 1e3)),
            clamp_u16(fix["eph"] * 100),
            clamp_u16(fix["epv"] * 100),
            clamp_u16(fix["vel"] * 100),
            clamp_i16(fix["vel_n"] * 100),
            clamp_i16(fix["vel_e"] * 100),
            clamp_i16(fix["vel_d"] * 100),
            clamp_u16(fix["cog"] * 100),
            int(fix["satellites"]),
        )

    def _send_ground_truth(self, ctx: SimContext) -> None:
        """``HIL_STATE_QUATERNION``: the true state, for logging and for comparing EKF2 against.

        PX4 does not fly on this -- it is the reference an estimation experiment measures error
        against, which is the reason to spend the bandwidth."""
        gyro_flu, accel_flu, quat, pos_enu, vel_enu = self._state(ctx)
        gyro = flu_to_frd(gyro_flu)
        accel = flu_to_frd(accel_flu)
        vel_ned = enu_to_ned(vel_enu)
        fix = self._gnss.read_fix()
        self._mav.hil_state_quaternion_send(
            int(ctx.sim_time * 1e6),
            [float(v) for v in quat_enu_flu_to_ned_frd(quat)],
            float(gyro[0]),
            float(gyro[1]),
            float(gyro[2]),
            int(round(fix["lat"] * 1e7)),
            int(round(fix["lon"] * 1e7)),
            int(round((self._datum_alt + float(pos_enu[2])) * 1e3)),
            clamp_i16(vel_ned[0] * 100),
            clamp_i16(vel_ned[1] * 100),
            clamp_i16(vel_ned[2] * 100),
            0,  # indicated airspeed: not modelled
            0,  # true airspeed: not modelled
            clamp_i16(accel[0] / STANDARD_GRAVITY * 1000.0),
            clamp_i16(accel[1] / STANDARD_GRAVITY * 1000.0),
            clamp_i16(accel[2] / STANDARD_GRAVITY * 1000.0),
        )

    # -- inbound ---------------------------------------------------------------------------------
    def _receive(self) -> None:
        """The socket thread. Parses frames and POSTS the result; touches nothing else.

        Per architecture.rst section 7 this thread must not write anything the physics thread reads
        -- not ``ctx.data``, and not the motors handle either, whose commanded value is read on the
        physics thread. Everything it learns goes through ``ctx.post``.
        """
        while not self._closing.is_set():
            conn = self._conn
            if conn is None:
                return
            try:
                chunk = conn.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                # PX4 hung up. Say so once; a lockstep tick then fails on its timeout with the
                # gate diagnostic, which is the message that explains the run.
                logger.error(
                    "px4_sitl (%s): PX4 closed the connection on port %d; the vehicle has no "
                    "autopilot for the rest of this run.",
                    self.name,
                    self._port,
                )
                return
            for msg in self._parse(chunk):
                if msg.get_type() != "HIL_ACTUATOR_CONTROLS":
                    continue
                controls = [float(v) for v in msg.controls]
                armed = bool(int(msg.mode) & MAV_MODE_FLAG_SAFETY_ARMED)
                self._ctx.post(lambda ctx, c=controls, a=armed: self._apply_controls(c, a))
                self._ever_answered.set()
                self._controls_arrived.set()

    def _parse(self, chunk: bytes) -> list:
        """pymavlink's parser, byte at a time -- ``parse_char`` yields at most one message."""
        messages = []
        for byte in chunk:
            msg = self._mav.parse_char(bytes([byte]))
            if msg is not None:
                messages.append(msg)
        return messages

    def _apply_controls(self, controls: list[float], armed: bool) -> None:
        """Physics-thread only: runs from the command queue at the start of ``pre_step``."""
        self._controls, self._armed = controls, armed
        n = self._rotor_count()
        if armed:
            values = [min(max(v, 0.0), 1.0) for v in controls[:n]]
        else:
            # Disarmed means disarmed. PX4 keeps streaming outputs while disarmed and their values
            # are not meaningful thrust; a vehicle that spun them up would take off on the pad.
            values = [0.0] * n
        self._motors.set_normalized(values)
        if self._gate is not None:
            self._gate.satisfy()

    def _rotor_count(self) -> int:
        """How many of PX4's 16 outputs this airframe actually has -- ``MotorsHandle.count``.

        Read off the handle rather than assumed: HIL_ACTUATOR_CONTROLS always carries 16 outputs
        whatever the airframe, so the number of rotors is the airframe's property, not the
        message's, and a hex would silently fly on four of its six.
        """
        return int(self._motors.count)
