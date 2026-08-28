"""The Interbotix WidowX XSeries turret: a 2-DOF pan/tilt, and a self-collision that hid as effort.

The finding this file pins is ``test_the_chain_neighbours_do_not_collide``. The vendor's collision
geometry *is* its visual mesh, and the convex hulls of the base and the pan overlap where the pan
seats into the base — 2.6 mm of penetration at the home pose. The constraint force that produced
exactly cancelled the pan servo's 3 N·m, so the joint would not move **while reporting full actuator
effort**: it looked like an under-powered servo and was a contact.

MuJoCo's ``filterparent`` does not save this, because it is skipped when the parent is the world body
and ``base_link`` is welded to world — the same trap the M1013 port documents. ``base_link``/
``tilt_link`` is a grandparent pair and was never auto-excluded either.

The second thing worth pinning is that this needed **no plugin work at all**:
``test_arm_controller_drives_a_two_joint_chain``. An "arm" to ``spawn_arm``/``arm_controller`` is a
chain of position-controlled joints, and two is a chain. The platform ledger predicted `capability:
none` and that held for the smallest actuated mechanism in the substrate.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim_manipulation_assets.models import MODELS_DIR

MODEL = MODELS_DIR / "wxxms" / "wxxms.xml"
#: From the expanded Interbotix wxxms description @ bd5fd2df, not measured from our model.
TOTAL_MASS = 1.0677
#: The vendor's own joint limits and effort ceiling.
PAN_RANGE = (-3.14, 3.14)
TILT_RANGE = (-np.pi / 2, np.pi / 2)
EFFORT = 3.0


def _model():
    return mujoco.MjModel.from_xml_path(str(MODEL))


def test_mass_matches_the_vendor_description():
    m = _model()
    assert m.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)


def test_the_vendors_limits_and_effort_are_kept():
    m = _model()
    for name, (lo, hi) in (("pan", PAN_RANGE), ("tilt", TILT_RANGE)):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"no joint {name!r}"
        assert m.jnt_range[jid][0] == pytest.approx(lo, abs=1e-4)
        assert m.jnt_range[jid][1] == pytest.approx(hi, abs=1e-4)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"no actuator {name!r}"
        assert m.actuator_forcerange[aid][1] == pytest.approx(EFFORT), (
            "the force ceiling is Interbotix's own 3 N.m effort limit, not a tuning choice"
        )


def test_the_chain_neighbours_do_not_collide():
    """The finding -- see the module docstring. Asserted as an absence of contact at the home pose."""
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    pairs = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1),
         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2))
        for i in range(d.ncon)
    ]
    assert not pairs, (
        f"the turret is in contact with itself at the home pose: {pairs}. MuJoCo's filterparent is "
        f"skipped when the parent is the world body, so the <contact><exclude> pairs are what keep "
        f"the base's and pan's convex hulls from locking the mechanism."
    )


def test_a_blocked_joint_would_have_looked_like_a_weak_servo():
    """Regression for the specific way that defect presented, which is why it took measuring.

    With the exclusions removed the pan actuator saturated at its full 3 N.m and the joint did not
    move -- symptoms of an under-powered servo. The distinguishing signal is ``qfrc_constraint``:
    something was pushing back exactly as hard. This asserts that nothing does.
    """
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = [np.pi / 2, -np.pi / 4]
    for _ in range(3000):
        mujoco.mj_step(m, d)
    assert np.abs(d.qfrc_constraint).max() < 0.05, (
        f"a constraint is resisting the servos ({np.round(d.qfrc_constraint, 3)}); if it matches the "
        f"actuator force the mechanism is fighting its own collision geometry, not lacking torque"
    )


@pytest.mark.parametrize(
    "target", [(0.0, 0.0), (np.pi / 2, -np.pi / 4), (-np.pi / 2, np.pi / 4), (1.2, -0.6)]
)
def test_arm_controller_drives_a_two_joint_chain(target):
    """No plugin work was needed: two position-controlled joints are a chain like six are."""
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = target
    for _ in range(4000):
        mujoco.mj_step(m, d)
    for got, want, name in zip(d.qpos, target, ("pan", "tilt"), strict=True):
        assert abs(float(got) - want) < 0.01, (
            f"{name} settled at {float(got):+.4f} against a commanded {want:+.4f}")


def test_the_surface_site_is_the_vendors_mounting_frame():
    """The reason this mechanism exists: something bolts to it and gets aimed."""
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "surface")
    assert sid >= 0, "no `surface` site -- a pan/tilt with nothing to mount on is furniture"
    home = d.site_xpos[sid].copy()
    assert home[2] > 0.1, f"the mounting frame should sit above the base, got z={home[2]:.4f}"

    # Panning a quarter turn must swing the mount off the x axis onto y: it is the aiming that
    # matters, so it is the aiming that is checked, not just that a joint moved.
    d.ctrl[:] = [np.pi / 2, -np.pi / 4]
    for _ in range(4000):
        mujoco.mj_step(m, d)
    aimed = d.site_xpos[sid]
    assert abs(aimed[1]) > 0.01, (
        f"after a +90 deg pan the mount is at {np.round(aimed, 4)}; its offset should have rotated "
        f"onto the y axis"
    )
    assert aimed[2] < home[2], "a negative tilt should lower the mount"
