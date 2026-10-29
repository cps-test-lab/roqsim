"""An arm standing on a riser: Neobotix's cabinet between an MPO-500 and a UR5e (``mount.riser``).

What goes wrong without it is invisible in every number that does not look at geometry: an arm whose
``pose`` puts its base 0.77 m above the base's frame on a deck that tops out at 0.39 m floats, its
weight still loads the base, and only a render shows the gap. These tests pin the three properties
that make the riser real:

* the arm hangs off the riser and the riser off the base, so all three ride the base's free joint;
* the arm's base stands ON the riser's top plate (within the vendor's 3 mm seat), and the riser on
  the deck, at the poses a robot description states -- both in the mount body's frame;
* nothing of the riser is touched by the arm's base in contact -- parent and child -- and a riser that
  is not rigid is refused.

The numbers are Neobotix's, from ``mp_components`` (jazzy @ c4ba0a7): the cabinet at base_link
(0.025, 0, 0.35), the UR's base_link at (0.133, 0, 0.416) in the cabinet, yawed -pi/2.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

CABINET = [0.025, 0.0, 0.35]
ARM_IN_CABINET = [0.133, 0.0, 0.416]
ARM = [CABINET[0] + ARM_IN_CABINET[0], 0.0, CABINET[2] + ARM_IN_CABINET[2]]
#: The spawn yaw for the vendor's -pi/2: the ur5e model's root body is a half turn from its spawn
#: frame (docs/plugins.rst).
YAW = 1.5708


def _world(tmp_path, riser=None):
    pose = {"position": dict(zip("xyz", ARM, strict=True)), "orientation": {"yaw": YAW}}
    mount = {
        "robot": "base",
        "body": "base_link",
        "riser": riser
        or {"model": "mpo_cabinet", "pose": {"position": dict(zip("xyz", CABINET, strict=True))}},
    }
    components = [
        {"spawn_robot": {"model": "mpo_500"}, "name": "base"},
        {
            "spawn_arm": {
                "model": "ur5e",
                "prefix": "ur5e_",
                "mount": mount,
                "pose": pose,
                "end_effector": {
                    "model": "robotiq_2f85",
                    "site": "tool0_site",
                    "pose": {"position": {"z": 0.011}},
                },
            },
            "name": "arm",
        },
    ]
    return load_config_from_dict(
        {"sim": {"timestep": 0.002}, "components": components}, base_dir=tmp_path
    )


@pytest.fixture
def engine(tmp_path):
    e = Engine(_world(tmp_path))
    e.ctx.seed = 0
    e.setup()
    yield e
    e.shutdown()


def _id(m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    assert i >= 0, f"{name} is missing"
    return i


def test_the_arm_hangs_off_the_riser_and_the_riser_off_the_base(engine):
    m = engine.ctx.model
    base = _id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    riser = _id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_riser_cabinet_link")
    arm = _id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_base")
    assert m.body_parentid[riser] == base
    assert m.body_parentid[arm] == riser
    assert list(m.jnt_type).count(mujoco.mjtJoint.mjJNT_FREE) == 1


def test_the_poses_are_the_descriptions_in_the_mount_frame(engine):
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    base = _id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rot = d.xmat[base].reshape(3, 3)

    def in_base(body):
        i = _id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        return rot.T @ (d.xpos[i] - d.xpos[base]), rot.T @ d.xmat[i].reshape(3, 3)

    pos, _ = in_base("ur5e_riser_cabinet_link")
    np.testing.assert_allclose(pos, CABINET, atol=1e-6)
    pos, r = in_base("ur5e_base")
    np.testing.assert_allclose(pos, ARM, atol=1e-6)
    # The ur5e MJCF's root body `base` is a half turn about z from the arm's spawn frame (its
    # quat="0 0 0 -1"), so the body stands at the stated yaw plus pi.
    yaw = np.arctan2(r[1, 0], r[0, 0])
    assert np.cos(yaw - (YAW + np.pi)) == pytest.approx(1.0, abs=1e-9)
    # The vendor's UR base_link is yawed -pi/2, so at zero joints the arm reaches along the base's -y.
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name.startswith("ur5e_"):
            d.qpos[m.jnt_qposadr[j]] = 0.0
    mujoco.mj_kinematics(m, d)
    pos, _ = in_base("ur5e_wrist_1_link")
    assert pos[1] - ARM[1] < -0.8 and abs(pos[0] - ARM[0]) < 0.2, pos


def test_the_arm_stands_on_the_top_plate(engine):
    """The riser's collider top is within the vendor's 3 mm seat of the arm's base, and the riser
    stands on the deck rather than in the air."""
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    g = _id(m, mujoco.mjtObj.mjOBJ_GEOM, "ur5e_riser_cabinet_collision")
    top = d.geom_xpos[g][2] + m.geom_size[g][2]
    bottom = d.geom_xpos[g][2] - m.geom_size[g][2]
    base = _id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    arm = _id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_base")
    assert 0.0 <= top - d.xpos[arm][2] <= 0.004, (
        f"arm base {d.xpos[arm][2]:.4f}, plate top {top:.4f}"
    )
    # The collider starts at the MPO-500's top cover (z 0.386 in base_link): the cabinet's lowest
    # 36 mm stand inside the base's body (mpo_cabinet's generator, COLLISION_FLOOR).
    assert bottom - d.xpos[base][2] == pytest.approx(0.386, abs=1e-6)


def test_the_arm_and_riser_do_not_touch_in_contact(engine):
    m, d = engine.ctx.model, engine.ctx.data
    for _ in range(200):
        engine.step()
    riser = _id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_riser_cabinet_link")
    base = _id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    for c in d.contact[: d.ncon]:
        bodies = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
        assert riser not in bodies, f"riser in contact with {[m.body(b).name for b in bodies]}"
        assert not (bodies & {base} and any(m.body(b).name.startswith("ur5e_") for b in bodies)), (
            f"arm and base in contact: {[m.body(b).name for b in bodies]}"
        )


def test_a_riser_with_joints_is_refused(tmp_path):
    with pytest.raises(Exception, match="rigid|joints"):
        e = Engine(_world(tmp_path, riser={"model": "robotiq_2f85"}))
        e.ctx.seed = 0
        e.setup()


def test_an_unknown_riser_is_refused(tmp_path):
    with pytest.raises(Exception, match="mount.riser"):
        Engine(_world(tmp_path, riser={"model": "no_such_riser"}))


def test_an_empty_riser_model_is_refused(tmp_path):
    """A riser that names nothing would leave the arm floating where the riser should be."""
    with pytest.raises(Exception, match="'mount.riser.model' is empty"):
        Engine(_world(tmp_path, riser={"model": ""}))


def test_a_riser_stated_as_a_name_is_refused_once_by_the_schema(tmp_path):
    with pytest.raises(Exception, match="riser") as exc:
        Engine(_world(tmp_path, riser="mpo_cabinet"))
    assert "could not be constructed" not in str(exc.value)
