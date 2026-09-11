"""px4_sitl: the frames, the two hard dependencies, and the lockstep -- without a real PX4.

The far end here is a socket this test drives itself, speaking real MAVLink 2 through pymavlink.
That is deliberate: what can go wrong in this bridge is the *wire* (frames, units, message
sequence) and the *blocking*, and both are testable against a client we control. Booting PX4 would
test PX4.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim_aerial.plugins.gnss import GnssHandle
from roqsim_aerial.plugins.px4_sitl import (
    clamp_i16,
    MAV_MODE_FLAG_SAFETY_ARMED,
    Px4SitlPlugin,
    enu_to_ned,
    flu_to_frd,
    quat_enu_flu_to_ned_frd,
)

mavlink2 = pytest.importorskip(
    "pymavlink.dialects.v20.common",
    reason="the wire tests speak real MAVLink 2; pymavlink is the roqsim_aerial[px4] extra",
)

SCENE = """
<mujoco model="px4_test">
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="x500" pos="0 0 1">
      <freejoint name="base_free"/>
      <geom name="core" type="box" size="0.15 0.15 0.05" mass="2"/>
      <site name="imu" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <gyro name="body_gyro" site="imu"/>
    <accelerometer name="body_linacc" site="imu"/>
    <framequat name="body_quat" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""

DATUM = {"lat": 47.397742, "lon": 8.545594, "alt": 488.0}


@dataclass
class FakeMotors:
    """The ``motors:<robot>`` handle's shape, recording what was commanded."""

    name: str = "drone"
    count: int = 4
    max_thrust: float = 8.175
    commanded: list = field(default_factory=list)

    def set_normalized(self, values):
        self.commanded = list(values)

    def read_normalized(self):
        return list(self.commanded)


def _fix(valid=True):
    return {
        "lat": DATUM["lat"],
        "lon": DATUM["lon"],
        "alt": DATUM["alt"],
        "eph": 0.5,
        "epv": 1.0,
        "vel_n": 0.0,
        "vel_e": 0.0,
        "vel_d": 0.0,
        "vel": 0.0,
        "cog": 0.0,
        "fix_type": 3 if valid else 0,
        "satellites": 12,
        "valid": valid,
    }


def _ctx(*, motors=True, gnss=True, seed=7):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.seed = seed
    # A real Entity, exactly as spawn_robot builds one: the root body is the first-class `body`
    # attribute, and `meta` carries prefix/namespace only.
    ctx.entities.add(Entity(name="drone", kind="robot", body="x500", meta={"prefix": ""}))
    if motors:
        ctx.blackboard.set("motors:drone", FakeMotors())
    if gnss:
        ctx.blackboard.set(
            "gnss:drone",
            GnssHandle(name="drone", read_fix=_fix, rate=10.0, datum=dict(DATUM)),
        )
    return ctx


def _free_port() -> int:
    """A port that was free a moment ago. The plugin binds with SO_REUSEADDR, so the gap is safe."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakePx4:
    """A PX4 stand-in: connects out to the simulator, reads HIL_*, answers with actuator controls."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        self.mav = mavlink2.MAVLink(_Writer(self.sock), srcSystem=1, srcComponent=1)
        self.mav.robust_parsing = True

    def read_until(self, msg_type: str, timeout: float = 5.0):
        self.sock.settimeout(timeout)
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError(f"simulator closed before sending {msg_type}")
            for byte in chunk:
                msg = self.mav.parse_char(bytes([byte]))
                if msg is not None and msg.get_type() == msg_type:
                    return msg

    def send_controls(self, controls, *, armed: bool) -> None:
        values = list(controls) + [0.0] * (16 - len(controls))
        self.mav.hil_actuator_controls_send(
            0, values, MAV_MODE_FLAG_SAFETY_ARMED if armed else 0, 0
        )

    def close(self) -> None:
        self.sock.close()


class _Writer:
    def __init__(self, sock):
        self._sock = sock

    def write(self, buf):
        self._sock.sendall(buf)


def _plugin(config, *, name="px4"):
    plugin = Px4SitlPlugin(config, name=name, entity="drone")
    errors = plugin.validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return plugin


def _connected(ctx, config=None):
    """A configured plugin with a fake PX4 attached. ``configure`` blocks until it connects, so the
    client has to be dialling while it does -- exactly the real startup order."""
    port = _free_port()
    plugin = _plugin({"port": port, "connect_timeout": 10.0, **(config or {})})
    box = {}

    def dial():
        for _ in range(200):
            try:
                box["px4"] = FakePx4(port)
                return
            except OSError:
                threading.Event().wait(0.02)

    thread = threading.Thread(target=dial, daemon=True)
    thread.start()
    plugin.configure(ctx)
    thread.join(timeout=5.0)
    assert "px4" in box, "the fake PX4 never connected"
    plugin.on_reset(ctx)
    return plugin, box["px4"], port


# -- frames (no socket, no pymavlink dependency in the logic) -------------------------------------
def test_enu_to_ned_on_a_hand_worked_vector():
    # (East 1, North 2, Up 3) is (North 2, East 1, Down -3).
    assert list(enu_to_ned([1.0, 2.0, 3.0])) == [2.0, 1.0, -3.0]


def test_flu_to_frd_on_a_hand_worked_vector():
    # The BODY transform is not the world one: forward is unchanged, left becomes -right.
    assert list(flu_to_frd([1.0, 2.0, 3.0])) == [1.0, -2.0, -3.0]


def test_the_two_transforms_are_not_the_same():
    """Conflating them is the classic failure: yaw looks right and roll is mirrored."""
    assert not np.allclose(enu_to_ned([1.0, 2.0, 3.0]), flu_to_frd([1.0, 2.0, 3.0]))


def test_a_ninety_degree_enu_yaw_is_zero_ned_yaw():
    """ENU yaw is measured from East, NED yaw (heading) from North. A drone yawed 90 degrees in ENU
    points North, which is heading zero -- so the converted attitude must be the identity."""
    half = np.pi / 4
    quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # +90 deg about ENU up
    assert np.allclose(quat_enu_flu_to_ned_frd(quat), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_a_level_enu_attitude_is_a_level_ned_attitude():
    """Identity in ENU/FLU is heading East, i.e. NED yaw +90 -- and level (no roll, no pitch)."""
    out = quat_enu_flu_to_ned_frd([1.0, 0.0, 0.0, 0.0])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, np.asarray(out, dtype=float))
    forward = rot.reshape(3, 3)[:, 0]
    assert np.allclose(forward, [0.0, 1.0, 0.0], atol=1e-12)  # body forward points East in NED
    assert out[0] == pytest.approx(np.cos(np.pi / 4), abs=1e-12)


def test_a_roll_keeps_its_sign_and_a_yaw_reverses():
    """The body transform flips the sign of roll and yaw rates; forward is untouched."""
    assert list(flu_to_frd([0.5, 0.0, 0.0])) == [0.5, 0.0, 0.0]  # roll rate: unchanged
    assert list(flu_to_frd([0.0, 0.0, 0.5])) == [0.0, 0.0, -0.5]  # yaw rate: ENU up -> NED down


# -- the two hard dependencies --------------------------------------------------------------------
def test_missing_motors_handle_is_refused_by_name():
    with pytest.raises(RuntimeError, match="multirotor_motors"):
        _plugin({"port": _free_port()}).configure(_ctx(motors=False))


def test_missing_gnss_handle_is_refused_by_name():
    """HIL_GPS is not optional for EKF2's default configuration, so a missing receiver is a failure
    at configure time and not a vehicle that silently never becomes armable."""
    with pytest.raises(RuntimeError, match="gnss"):
        _plugin({"port": _free_port()}).configure(_ctx(gnss=False))


def test_a_missing_imu_sensor_is_refused_by_name():
    scene = SCENE.replace('<gyro name="body_gyro" site="imu"/>', "")
    model = mujoco.MjModel.from_xml_string(scene)
    ctx = _ctx()
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(ctx.model, ctx.data)
    with pytest.raises(RuntimeError, match="body_gyro"):
        _plugin({"port": _free_port()}).configure(ctx)


def test_no_px4_within_the_timeout_fails_naming_the_port():
    port = _free_port()
    with pytest.raises(RuntimeError, match=f"port {port}"):
        _plugin({"port": port, "connect_timeout": 0.2}).configure(_ctx())


def test_a_second_bridge_on_one_port_is_a_named_collision():
    ctx = _ctx()
    first = _plugin({"port": _free_port(), "connect_timeout": 0.2})
    with pytest.raises(RuntimeError, match="no PX4 SITL connected"):
        first.configure(ctx)  # binds, then times out waiting -- the listener is still held
    try:
        second = _plugin({"port": first._port, "connect_timeout": 0.2}, name="px4b")
        with pytest.raises(RuntimeError, match="cannot listen"):
            second.configure(_ctx())
    finally:
        first.shutdown(ctx)


# -- the wire ------------------------------------------------------------------------------------
def test_the_bridge_listens_and_a_client_can_connect():
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx)
    try:
        assert plugin._conn is not None
    finally:
        px4.close()
        plugin.shutdown(ctx)


class _Tick:
    """One post_step, run off the main thread so the test itself can play PX4.

    Only ONE thread may ever drive the FakePx4: its MAVLink parser is a single buffer, and two
    readers on it interleave into garbage. So the *plugin* goes on the worker and the wire stays on
    the main thread, which is also the arrangement that lets the test observe the block directly.
    """

    def __init__(self, plugin, ctx):
        self._plugin, self._ctx = plugin, ctx
        self.returned = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._plugin.post_step(self._ctx)
        except BaseException as exc:  # surfaced by join(), never swallowed
            self.error = exc
        finally:
            self.returned.set()

    def join(self, timeout=5.0):
        self._thread.join(timeout)
        if self.error is not None:
            raise self.error
        assert self.returned.is_set(), "post_step never returned after PX4 answered"


def _boot(plugin, px4, ctx):
    """Play PX4's boot: one free-running tick, answered once.

    Lockstep cannot be armed before PX4 has EVER answered -- PX4's clock is driven by our
    HIL_SENSOR timestamps, so blocking on the first batch stops the very boot that would produce
    the first HIL_ACTUATOR_CONTROLS. Every lockstep assertion below therefore starts here.
    """
    tick = _Tick(plugin, ctx)
    px4.read_until("HIL_SENSOR")
    tick.join()  # must NOT have blocked
    px4.send_controls([0.0] * 4, armed=False)
    _wait_for(lambda: plugin._ever_answered.is_set())
    ctx.drain_commands()


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition never became true")


def test_the_first_tick_does_not_block_because_px4_has_not_booted_yet():
    """The deadlock this gate exists to prevent, pinned as a test.

    PX4 SITL runs on a lockstep scheduler clocked by the HIL_SENSOR timestamps we send; its startup
    script, and therefore every module that could ever publish `actuator_outputs`, only makes
    progress as that clock advances. A bridge that blocks on the first sensor batch waits for
    controls that require a boot that requires the sensor stream it has just stopped. Observed
    against real PX4 v1.18.0-beta2, which sat at `poll timeout 0, 25` forever.
    """
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": False})
    try:
        tick = _Tick(plugin, ctx)
        px4.read_until("HIL_SENSOR")
        assert tick.returned.wait(2.0), (
            "post_step blocked before PX4 ever answered -- that is the boot deadlock"
        )
        tick.join()
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_a_lockstep_tick_blocks_until_controls_arrive_and_then_commands_the_motors():
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": False})
    motors = ctx.blackboard.get("motors:drone")
    try:
        _boot(plugin, px4, ctx)
        tick = _Tick(plugin, ctx)
        sensor = px4.read_until("HIL_SENSOR")
        # PX4 locks to instance 0, and the batch must declare accel, gyro, mag and baro.
        assert sensor.id == 0
        assert sensor.fields_updated & 0x1FF == 0x1FF
        # The tick is still blocked: the sensors are out and no controls have been sent.
        assert not tick.returned.wait(0.2), "post_step returned unanswered -- lockstep is not locking"

        px4.send_controls([0.1, 0.2, 0.3, 0.4], armed=True)
        tick.join()

        # Nothing has touched the motors yet: the receive thread only POSTED the command, so the
        # motors still carry the zeros the disarmed boot handshake left there.
        assert motors.commanded == pytest.approx([0.0] * 4)
        assert ctx.drain_commands() == 1
        assert motors.commanded == pytest.approx([0.1, 0.2, 0.3, 0.4])
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_the_gate_is_the_section_10_seam_and_it_fires():
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": False})
    try:
        gate = next(g for g in ctx.gates() if "px4" in g.name)
        assert gate.role == "consumer"
        _boot(plugin, px4, ctx)
        tick = _Tick(plugin, ctx)
        px4.read_until("HIL_SENSOR")
        px4.send_controls([0.0] * 4, armed=True)
        tick.join()
        assert not gate.is_satisfied()  # the command is queued, not yet run
        ctx.drain_commands()
        assert gate.is_satisfied()  # satisfied on the physics thread, where section 10 puts it
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_disarmed_controls_command_zero():
    """PX4 keeps streaming outputs while disarmed; a bridge that passed them through would spin the
    rotors up on the pad."""
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": False})
    motors = ctx.blackboard.get("motors:drone")
    try:
        _boot(plugin, px4, ctx)
        tick = _Tick(plugin, ctx)
        px4.read_until("HIL_SENSOR")
        px4.send_controls([0.9, 0.9, 0.9, 0.9], armed=False)
        tick.join()
        ctx.drain_commands()
        assert motors.commanded == [0.0, 0.0, 0.0, 0.0]
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_hil_gps_carries_the_receivers_fix():
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": False})
    try:
        tick = _Tick(plugin, ctx)
        gps = px4.read_until("HIL_GPS")  # sent in the same batch, before the block
        px4.send_controls([0.0] * 4, armed=True)
        tick.join()
        assert gps.lat == round(DATUM["lat"] * 1e7)
        assert gps.lon == round(DATUM["lon"] * 1e7)
        assert gps.fix_type == 3
        assert gps.satellites_visible == 12
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_ground_truth_reports_the_attitude_in_ned():
    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": True})
    try:
        tick = _Tick(plugin, ctx)
        truth = px4.read_until("HIL_STATE_QUATERNION")
        px4.send_controls([0.0] * 4, armed=True)
        tick.join()
        # The drone is level and unrotated in ENU, which is heading East in NED.
        assert truth.attitude_quaternion[0] == pytest.approx(np.cos(np.pi / 4), abs=1e-5)
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_ground_truth_acceleration_is_milli_g_and_saturates():
    """HIL_STATE_QUATERNION's xacc/yacc/zacc are int16 in MILLI-G -- the one non-SI field in this
    message set, and the only place a m/s^2 value must be divided by g.

    Encoding m/s^2 * 1000 instead overflows the int16 the moment the landing gear touches down: a
    real run against PX4 v1.18 died in pymavlink's struct.pack on the first contact tick. So both
    halves are pinned: the unit, and the saturation that keeps a contact spike from aborting a run
    over a number nothing flies on.
    """
    assert clamp_i16(1e9) == 32767
    assert clamp_i16(-1e9) == -32768

    ctx = _ctx()
    plugin, px4, _ = _connected(ctx, {"imu_rate": 250.0, "ground_truth": True})
    try:
        # A level airframe standing on its gear: the accelerometer reads +1 g along body up (FLU),
        # which is -1000 mG once converted to FRD. Encoded as m/s^2 * 1000 the field would read
        # about -9810 -- so this number is what separates the two conventions.
        adr = int(ctx.model.sensor_adr[
            mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")
        ])
        ctx.data.sensordata[adr : adr + 3] = [0.0, 0.0, 9.80665]

        tick = _Tick(plugin, ctx)
        truth = px4.read_until("HIL_STATE_QUATERNION")
        px4.send_controls([0.0] * 4, armed=True)
        tick.join()
        assert truth.zacc == -1000
        assert truth.xacc == 0 and truth.yacc == 0
    finally:
        px4.close()
        plugin.shutdown(ctx)


def test_teardown_releases_the_port():
    """A leaked listener makes the NEXT run fail with 'address in use', which is a miserable thing
    to debug in a campaign."""
    ctx = _ctx()
    plugin, px4, port = _connected(ctx)
    px4.close()
    plugin.shutdown(ctx)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))  # raises if the listener leaked


# -- config --------------------------------------------------------------------------------------
def test_refuses_a_seed_of_its_own():
    assert any("seed" in e for e in Px4SitlPlugin({}).validate_config({"seed": 3}))


def test_refuses_an_unknown_sensor_noise_key():
    errors = Px4SitlPlugin({}).validate_config({"sensor_noise": {"lidar": 0.1}})
    assert any("lidar" in e for e in errors), errors


# -- the entity contract -------------------------------------------------------------------------
def test_the_body_comes_from_the_entity_not_from_a_meta_key():
    """Regression: `Entity.body`, not a meta key. Only a real Entity can catch a wrong key name."""
    ctx = _ctx()
    assert "root_body" not in ctx.entities.get("drone").meta
    plugin = _plugin({"port": _free_port(), "connect_timeout": 0.2})
    with pytest.raises(RuntimeError, match="no PX4 SITL connected"):
        plugin.configure(ctx)  # gets all the way to the connect wait, i.e. the body resolved
    try:
        assert plugin._bid == mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "x500")
    finally:
        plugin.shutdown(ctx)


def test_an_entity_with_no_body_at_all_still_fails_loudly():
    ctx = _ctx()
    ctx.entities.remove("drone")
    ctx.entities.add(Entity(name="drone", kind="robot", body=None, meta={"prefix": ""}))
    with pytest.raises(RuntimeError, match="no body to fly"):
        _plugin({"port": _free_port(), "connect_timeout": 0.2}).configure(ctx)
