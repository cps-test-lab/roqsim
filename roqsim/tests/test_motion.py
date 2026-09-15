"""When a recording starts moving -- and the spawn drop that makes the naive answer wrong.

The first two tests are the reason this module exists. A robot is spawned a little above the floor
and falls onto it, so *something* is moving at t=0 in almost every recording; a detector that reads
"the robot's velocity" therefore reports every run as starting immediately, which is exactly the dead
air ``--from onset`` exists to remove. Measured on recorded runs the drop reaches 0.8 m/s in some
worlds and 0.02 m/s in others, so no threshold separates a fall from a drive -- only direction does.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from roqsim.capture import STATE_FIELDS, STATE_SPEC, RecordingError, record_dtype
from roqsim.motion import MotionError, actuated_joints, base_free_joint, motion_onset
from roqsim.recording import open_recording

FPS = 25

# A base on a free joint with two driven wheels: the shape every mobile robot in the substrate has,
# and the one that can fall.
_MOBILE_XML = """
<mujoco>
  <option timestep="0.04"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body name="base_link" pos="0 0 .2">
      <freejoint name="base_free"/>
      <geom type="box" size=".2 .15 .05"/>
      <body name="wl" pos="0 .16 0">
        <joint name="wheel_left" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size=".05 .02" quat=".707 .707 0 0"/>
      </body>
      <body name="wr" pos="0 -.16 0">
        <joint name="wheel_right" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size=".05 .02" quat=".707 .707 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="wheel_left" kv="10"/>
    <velocity joint="wheel_right" kv="10"/>
  </actuator>
</mujoco>
"""

# A bolted-down arm -- which cannot fall -- standing beside a free-floating parcel, which can. The
# parcel is the trap: it is a free body, but it is the root of no actuated joint.
_FIXED_XML = """
<mujoco>
  <option timestep="0.04"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body name="arm_base" pos="0 0 0">
      <body name="link" pos="0 0 .2">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".04" fromto="0 0 0 .4 0 0"/>
      </body>
    </body>
    <body name="parcel" pos="1 0 .5">
      <freejoint name="parcel_free"/>
      <geom type="box" size=".1 .1 .1"/>
    </body>
  </worldbody>
  <actuator><position joint="shoulder" kp="20"/></actuator>
</mujoco>
"""


def _write(tmp_path, xml, qvel_of, *, seconds=6.0, actuators=None, name="run.npz"):
    """A recording of ``xml`` whose velocities are whatever ``qvel_of(t)`` says.

    Written by hand rather than simulated so a test can state the motion it is testing -- "falls for
    one second, then drives" -- instead of tuning a world until it happens to produce that.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    size = mujoco.mj_stateSize(model, STATE_SPEC)
    times = np.arange(1, int(seconds * FPS) + 1) / FPS
    samples = np.zeros(len(times), dtype=record_dtype(size, False))
    samples["t"] = times
    samples["w"] = times
    for i, t in enumerate(times):
        samples["s"][i, 0] = t
        samples["s"][i, 1 + model.nq : 1 + model.nq + model.nv] = qvel_of(float(t), model)
    meta = {
        "format_version": 2,
        "state_size": size,
        "state_fields": list(STATE_FIELDS),
        "capture_fps": [FPS, 1],
        "world": "synthetic.yaml",
        "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)},
        "actuators": actuators if actuators is not None else {},
    }
    path = tmp_path / name
    np.savez(path, meta=np.array(json.dumps(meta)), samples=samples)
    return open_recording(path), model


def _dofs(model, joint):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return int(model.jnt_dofadr[jid])


_WHEELS = {"base": [{"joint": "wheel_left"}, {"joint": "wheel_right"}]}
_ARM = {"arm": [{"joint": "shoulder"}]}


# -- the spawn drop: the bug this file exists to prevent -------------------------------------------


def test_a_base_that_falls_then_drives_starts_at_the_drive(tmp_path):
    """The load-bearing case. Falls at 0.8 m/s for a second, stands still, then drives at 0.5 m/s.

    0.8 is the drop measured on real recordings, and it is *larger* than the drive that follows -- so
    a detector ranking motion by speed picks the fall. Only direction separates them.
    """

    def qvel(t, model):
        v = np.zeros(model.nv)
        adr = _dofs(model, "base_free")
        if t <= 1.0:
            v[adr + 2] = -0.8  # falling: vz
            v[adr + 3] = 0.3  # and rocking about x as it lands
        elif t >= 3.0:
            v[adr] = 0.5  # driving: vx
        return v

    rec, model = _write(tmp_path, _MOBILE_XML, qvel, actuators=_WHEELS)
    onset = motion_onset(rec, model=model, pre_roll=0.0)

    assert onset.moved
    assert onset.kind == "mobile"
    assert onset.channel == "lin"
    assert onset.detected == pytest.approx(3.0, abs=1 / FPS)


def test_the_fall_alone_is_not_movement(tmp_path):
    """A run that only ever falls and settles never moved, and says so rather than reporting t=0."""

    def qvel(t, model):
        v = np.zeros(model.nv)
        if t <= 1.0:
            v[_dofs(model, "base_free") + 2] = -0.8
        return v

    rec, model = _write(tmp_path, _MOBILE_XML, qvel, actuators=_WHEELS)
    onset = motion_onset(rec, model=model)

    assert not onset.moved
    assert onset.detected is None
    assert onset.time == pytest.approx(rec.span[0])


def test_any_sees_the_fall_which_is_why_it_is_the_fallback(tmp_path):
    """The negative control: proves the test above is testing the *selection*, not the threshold.

    ``select="any"`` reads every velocity, so the same recording reports the drop. It exists for
    recordings whose world can no longer be rebuilt, and reports ``kind="any"`` to say so.
    """

    def qvel(t, model):
        v = np.zeros(model.nv)
        if t <= 1.0:
            v[_dofs(model, "base_free") + 2] = -0.8
        return v

    rec, _ = _write(tmp_path, _MOBILE_XML, qvel, actuators=_WHEELS)
    onset = motion_onset(rec, select="any")

    assert onset.moved
    assert onset.kind == "any"
    assert onset.detected == pytest.approx(rec.span[0], abs=1 / FPS)


# -- choosing the signal ---------------------------------------------------------------------------


def test_a_free_joint_that_is_not_the_robot_does_not_make_it_mobile(tmp_path):
    """A parcel on a conveyor is a free body too. Reaching the base *through the actuators* is what
    keeps it from being mistaken for one -- it is the root of no actuated joint."""
    rec, model = _write(tmp_path, _FIXED_XML, lambda t, m: np.zeros(m.nv), actuators=_ARM)

    assert base_free_joint(model, actuated_joints(rec)) is None
    assert motion_onset(rec, model=model).kind == "fixed"


def test_a_driven_base_is_mobile(tmp_path):
    rec, model = _write(tmp_path, _MOBILE_XML, lambda t, m: np.zeros(m.nv), actuators=_WHEELS)

    free = base_free_joint(model, actuated_joints(rec))
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, free) == "base_free"
    assert motion_onset(rec, model=model).kind == "mobile"


def test_a_fixed_base_reads_its_actuated_joints_and_ignores_the_parcel(tmp_path):
    """The parcel moves the whole time; the arm moves at t=2. A fixed base reports the arm."""

    def qvel(t, model):
        v = np.zeros(model.nv)
        v[_dofs(model, "parcel_free")] = 2.0  # the conveyor runs throughout
        if t >= 2.0:
            v[_dofs(model, "shoulder")] = 0.4
        return v

    rec, model = _write(tmp_path, _FIXED_XML, qvel, actuators=_ARM)
    onset = motion_onset(rec, model=model, pre_roll=0.0)

    assert onset.kind == "fixed"
    assert onset.joints == ("shoulder",)
    assert onset.detected == pytest.approx(2.0, abs=1 / FPS)


def test_planar_on_a_fixed_base_is_refused(tmp_path):
    rec, model = _write(tmp_path, _FIXED_XML, lambda t, m: np.zeros(m.nv), actuators=_ARM)

    with pytest.raises(MotionError, match="no free joint"):
        motion_onset(rec, select="planar", model=model)


def test_named_joints_override_the_choice(tmp_path):
    """Naming joints reads those, whatever the robot is -- here the parcel rather than the arm."""

    def qvel(t, model):
        v = np.zeros(model.nv)
        if t >= 1.0:
            v[_dofs(model, "parcel_free")] = 2.0
        return v

    rec, model = _write(tmp_path, _FIXED_XML, qvel, actuators=_ARM)
    onset = motion_onset(rec, select=["parcel_free"], model=model, pre_roll=0.0)

    assert onset.select == "joints"
    assert onset.detected == pytest.approx(1.0, abs=1 / FPS)


# -- the threshold ---------------------------------------------------------------------------------


def test_a_single_sample_spike_is_not_the_run_starting(tmp_path):
    """Measured on a real run: one sample of 0.0248 rad/s and nothing after it. Judging by the peak
    alone would call that moved and clip a robot that never drove."""

    def qvel(t, model):
        v = np.zeros(model.nv)
        if abs(t - 1.0) < 1e-9:
            v[_dofs(model, "base_free")] = 0.6
        return v

    rec, model = _write(tmp_path, _MOBILE_XML, qvel, actuators=_WHEELS)
    assert not motion_onset(rec, model=model).moved


def test_the_floor_keeps_a_stationary_run_from_bootstrapping_its_own_noise(tmp_path):
    """Without an absolute floor, ``fraction * peak`` of a parked robot is ~0 and any value crosses."""

    def qvel(t, model):
        v = np.zeros(model.nv)
        v[_dofs(model, "base_free")] = 1e-7 * (1 + (t > 2))
        return v

    rec, model = _write(tmp_path, _MOBILE_XML, qvel, actuators=_WHEELS)
    onset = motion_onset(rec, model=model)

    assert not onset.moved
    assert onset.peaks["lin"] < onset.thresholds["lin"]


def test_pre_roll_opens_the_clip_before_the_motion_but_never_before_the_run(tmp_path):
    def qvel(t, model):
        v = np.zeros(model.nv)
        if t >= 0.2:
            v[_dofs(model, "base_free")] = 0.5
        return v

    rec, model = _write(tmp_path, _MOBILE_XML, qvel, actuators=_WHEELS)
    onset = motion_onset(rec, model=model, pre_roll=0.5)

    assert onset.detected == pytest.approx(0.2, abs=1 / FPS)
    assert onset.time == pytest.approx(rec.span[0])


# -- reading a recording without rebuilding it -----------------------------------------------------


def test_qvel_is_readable_when_the_world_cannot_be_rebuilt(tmp_path):
    """The reason the accessors exist. ``build()`` needs the world; the samples do not.

    A quarter of the archived runs name a model provider that is no longer installed, which leaves the
    file perfectly readable and every restoring path refusing.
    """
    rec, model = _write(tmp_path, _MOBILE_XML, lambda t, m: np.zeros(m.nv), actuators=_WHEELS)

    with pytest.raises(RecordingError):
        rec.build()  # 'synthetic.yaml' is no world
    assert rec.qvel.shape == (len(rec), model.nv)
    assert rec.qpos.shape == (len(rec), model.nq)
    assert motion_onset(rec, select="any").kind == "any"


def test_samples_are_handed_out_read_only(tmp_path):
    rec, _ = _write(tmp_path, _MOBILE_XML, lambda t, m: np.zeros(m.nv), actuators=_WHEELS)

    with pytest.raises(ValueError):
        rec.samples["t"][0] = 99.0


def test_the_onset_names_the_robot_so_a_picture_can_be_framed_on_it(tmp_path):
    """``body`` is the root the actuated joints hang from -- the arm, not the parcel beside it."""
    fixed, _ = _write(
        tmp_path, _FIXED_XML, lambda t, m: np.zeros(m.nv), actuators=_ARM, name="a.npz"
    )
    mobile, model = _write(
        tmp_path, _MOBILE_XML, lambda t, m: np.zeros(m.nv), actuators=_WHEELS, name="b.npz"
    )

    assert motion_onset(fixed, model=mujoco.MjModel.from_xml_string(_FIXED_XML)).body == "arm_base"
    assert motion_onset(mobile, model=model).body == "base_link"
    # "any" identifies no robot, so it offers no body to frame on rather than guessing one.
    assert motion_onset(mobile, select="any").body is None
