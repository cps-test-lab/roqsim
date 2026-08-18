"""The interchangeable end effectors: they compile, they open and close, and they hold a payload.

Each gripper is checked three ways, because each catches a different class of mistake:

* **aperture** -- the measured travel between the jaw inner faces against the datasheet. This is what
  catches a wrong mesh scale, a wrong joint range, or an inverted tendon sign, all of which still
  compile and still move.
* **command polarity** -- that ``gripper_open`` really opens. ``arm_controller`` maps the manifest's
  open/close values onto the actuator's ctrlrange low/high end, so a sign error in either the tendon
  coefficients or the manifest produces a gripper that closes when told to open. Nothing else notices.
* **hold under gravity** -- the jaws are closed on a parcel in zero g, then gravity is switched on and
  the slip is measured. A friction grasp that looks right statically can still creep out of the jaws
  (the humanoid pick measured 0.12 m/s of it before ``noslip_iterations`` was raised), so the assertion
  is on millimetres of slip over ten seconds rather than on contact existing.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.models import apply_assets, resolve_model

# model, jaw geoms, jaw half-thickness across the grasp axis, driver joints, actuator,
# (ctrl_open, ctrl_close), home (rest) jaw position, jaw half-height, datasheet aperture in mm
GRIPPERS = [
    pytest.param(
        "robotiq_2f85",
        ("left_pad1", "right_pad1"),
        0.004,
        ("robotiq_85_left_knuckle_joint", "right_driver_joint"),
        "fingers_actuator",
        (0.0, 255.0),
        0.0,
        0.019,
        85.0,
        id="robotiq_2f85",
    ),
    pytest.param(
        "schunk_pg70",
        ("finger_left_pad", "finger_right_pad"),
        0.005,
        ("finger_left_joint", "finger_right_joint"),
        "finger_actuator",
        (-0.0301, 0.001),
        0.0301,
        0.005,
        61.2,  # PILZ's URDF stroke, NOT the PG+70 datasheet's 70 mm -- see the port log
        id="schunk_pg70",
    ),
]

# A parcel narrow enough for the tighter of the two grippers, so one payload serves both tests.
PARCEL_HALF = (0.03, 0.0225, 0.01875)
PARCEL_MASS = 0.211

_TEST_WORLD = """<mujoco model="grasp_test">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"
          noslip_iterations="10"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="parcel" pos="0 0 {pz}">
      <freejoint/>
      <geom type="box" size="{hx} {hy} {hz}" mass="{mass}" condim="4"
            friction="1.2 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.005 1"/>
    </body>
  </worldbody>
</mujoco>"""


def _gripper_spec(model):
    asset = resolve_model(model)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    return spec


def _jaw_gap(model, data, geoms, half_thickness):
    """Distance between the two jaws' inner faces, along whichever axis they separate on."""
    a, b = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms)
    return float(np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b])) - 2 * half_thickness


@pytest.mark.parametrize(
    "model,geoms,half,joints,actuator,ctrl,home,jaw_half_z,aperture_mm", GRIPPERS
)
def test_gripper_aperture_matches_spec(
    model, geoms, half, joints, actuator, ctrl, home, jaw_half_z, aperture_mm
):
    """Open/closed travel matches the description it was ported from, to within a millimetre."""
    m = _gripper_spec(model).compile()
    d = mujoco.MjData(m)
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
    assert m.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_TENDON, (
        "a gripper must be driven through a tendon, not a joint: that non-joint transmission is "
        "what makes arm_controller expose it as a GripperCommand action instead of as arm joints"
    )

    measured = {}
    for value, label in ((ctrl[0], "open"), (ctrl[1], "closed")):
        d.ctrl[aid] = value
        for _ in range(4000):
            mujoco.mj_step(m, d)
        measured[label] = _jaw_gap(m, d, geoms, half) * 1000.0

    assert measured["open"] == pytest.approx(aperture_mm, abs=1.0)
    assert measured["closed"] == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize(
    "model,geoms,half,joints,actuator,ctrl,home,jaw_half_z,aperture_mm", GRIPPERS
)
def test_gripper_open_command_opens(
    model, geoms, half, joints, actuator, ctrl, home, jaw_half_z, aperture_mm
):
    """The ctrl value the manifest calls `gripper_open` widens the jaws, not narrows them.

    Both grippers reach their open end at the LOW end of ctrlrange, which for the Schunk means
    negative tendon coefficients. Get that backwards and every GripperCommand runs inverted.
    """
    m = _gripper_spec(model).compile()
    d = mujoco.MjData(m)
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
    assert m.actuator_ctrlrange[aid][0] == pytest.approx(min(ctrl)), (
        "gripper_open must map to the low end of ctrlrange (arm_controller.set_gripper maps it there)"
    )

    d.ctrl[aid] = ctrl[1]
    for _ in range(3000):
        mujoco.mj_step(m, d)
    closed = _jaw_gap(m, d, geoms, half)
    d.ctrl[aid] = ctrl[0]
    for _ in range(3000):
        mujoco.mj_step(m, d)
    assert _jaw_gap(m, d, geoms, half) > closed + 0.02


@pytest.mark.parametrize(
    "model,geoms,half,joints,actuator,ctrl,home,jaw_half_z,aperture_mm", GRIPPERS
)
def test_gripper_holds_parcel_under_gravity(
    model, geoms, half, joints, actuator, ctrl, home, jaw_half_z, aperture_mm
):
    """Closed on a parcel, the grasp does not creep once gravity is switched on.

    The parcel is gripped near its TOP edge rather than through its centre. That is not a convenience:
    both grippers have short jaws set close to the palm face (the PG+70's are 10 mm blocks 2.5 mm off
    it), so a parcel centred on the jaw plane intersects the palm's own collision geometry and is
    ejected on the first step. Gripping the upper band is what a short-jaw gripper physically does.
    """
    palm_z = 0.5
    jaw_plane = {"robotiq_2f85": 0.1488, "schunk_pg70": 0.0789}[model]
    parcel_z = palm_z - (jaw_plane + PARCEL_HALF[2] - jaw_half_z)

    world = mujoco.MjSpec.from_string(
        _TEST_WORLD.format(
            pz=parcel_z, hx=PARCEL_HALF[0], hy=PARCEL_HALF[1], hz=PARCEL_HALF[2], mass=PARCEL_MASS
        )
    )
    frame = world.worldbody.add_frame()
    frame.pos = [0.0, 0.0, palm_z]
    frame.quat = [0.0, 1.0, 0.0, 0.0]  # flip so the approach axis points down at the parcel
    world.attach(_gripper_spec(model), prefix="grip_", frame=frame)
    m = world.compile()
    d = mujoco.MjData(m)

    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_" + actuator)
    parcel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "parcel")
    # Start from the open stance, as spawn_arm's home / arm_controller.on_reset would. Without this
    # the jaws begin shut around the parcel and MuJoCo resolves the interpenetration by firing it out.
    for jname in joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_" + jname)
        d.qpos[m.jnt_qposadr[jid]] = home
    mujoco.mj_forward(m, d)

    m.opt.gravity[2] = 0.0
    d.ctrl[aid] = ctrl[0]
    for _ in range(500):
        mujoco.mj_step(m, d)
    d.ctrl[aid] = ctrl[1]
    for _ in range(1500):
        mujoco.mj_step(m, d)

    assert d.ncon > 0, "the jaws never touched the parcel"
    held_z = float(d.xpos[parcel][2])
    m.opt.gravity[2] = -9.81
    for _ in range(5000):  # 10 s
        mujoco.mj_step(m, d)

    slip_mm = (held_z - float(d.xpos[parcel][2])) * 1000.0
    assert abs(slip_mm) < 2.0, f"parcel slipped {slip_mm:.2f} mm in 10 s"
