"""Every humanoid's manifest-declared range sensor: what its own scan hits of itself, and where it sits.

The rule a lidar follows is that it never excludes robot geometry, so the scan is cast here from the
plugin's own ray pattern and site with no exclusion but the device's own housing (a device model
names it, ``exclude_body: mount``), and two properties are pinned per robot:

* **No ray starts inside robot geometry.** A first surface met from inside (normal . ray > 0) means
  the scan origin lies within a geom, and a published scan would read that geom's inner face on every
  such ray -- unless something robot-sized is excluded, which is what this test refuses to do.
* **The robot bodies hit from outside, with their ray counts.** These are real returns of a sensor
  that sees part of its own robot; a change to the model, the stance or the mount shows up here.

No manifest here excludes robot geometry. A manifest that did would be pinned in ``STILL_EXCLUDES``,
so the exclusion is a visible exception rather than a default nobody chose.

The Unitree G1s mount the Livox Mid-360 device where Unitree's description puts it, inside the head,
and their head mesh carries the opening the sensor looks out of (external/convert/g1_head_window.py).
For those, the mount itself is pinned too: the scan site is the vendor joint origin on
``torso_link``, and the published transforms are that chain.

Each robot is spawned as a world spawns it -- ``spawn_robot`` with a prefix, its manifest's controller
setting the stance at reset (a humanoid's legs, the G2's hanging arms) -- into the default world
(``empty_room``: a floor and perimeter walls), so every non-world body is the robot's or its sensor's.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.pose import rpy_to_quat

PREFIX = "r_"

#: model -> (sim timestep, address of its range sensor, {robot body hit from outside: ray count}).
CASES = {
    # Returns off base_link, which carries the 12-DoF model's welded shoulders and arms: the field
    # reaches -55 deg below the horizon, and the arms hang inside it.
    "unitree_g1": (0.002, "robot.mid360.livox_mid360", {"base_link": 777}),
    # The same mount riding the waist chain: shoulders, wrists and grippers below the head.
    "unitree_g1_dex1": (
        0.002,
        "robot.mid360.livox_mid360",
        {
            "left_shoulder_pitch_link": 224,
            "right_shoulder_pitch_link": 224,
            "left_wrist_yaw_link": 44,
            "right_wrist_yaw_link": 42,
            "left_wrist_pitch_link": 6,
            "right_wrist_pitch_link": 6,
            "left_dex1_base_link": 55,
            "right_dex1_base_link": 56,
            "left_dex1_finger_link_1": 16,
            "right_dex1_finger_link_1": 9,
            "left_dex1_finger_link_2": 4,
            "right_dex1_finger_link_2": 1,
        },
    ),
    # The waist is a jointed link, not the pelvis, so its surface 0.062 m out around the whole fan is
    # a real return by the rule a lidar follows. Whether the site belongs at this height is an open
    # question about the mount pose, not about the exclusion.
    "oli": (0.001, "robot.lidar", {"waist_pitch_link": 360}),
    # The torso column stands behind the chassis-front site and fills the rear of the fan, and the
    # chassis itself (base_link) returns from outside over another sector. Both are real returns in
    # the published scan.
    "agibot_g2": (0.002, "robot.lidar", {"body_link1": 89, "base_link": 98}),
}

#: model -> the body its range sensor excludes: its own device housing, and nothing of the robot.
EXCLUDES = {"unitree_g1": "mid360_mount", "unitree_g1_dex1": "mid360_mount"}

#: model -> the robot body its manifest still excludes from the published scan.
STILL_EXCLUDES: dict[str, str] = {}

#: unitree_ros g1_description @ f3772ce, g1_29dof_rev_1_0.urdf:572-576 (and
#: g1_29dof_mode_15_with_dex1_1.urdf:548-552): mid360_joint on torso_link.
MID360_JOINT = ([0.0002835, 0.00003, 0.428434], [3.141592653589793, 0.05112069379091391, 0.0])

#: model -> the static transforms its mount publishes, as (parent, child, translation, rpy).
G1_TRANSFORMS = {
    "unitree_g1": [
        # The 12-DoF MJCF welds the waist; its manifest declares torso_link (the waist joints' origins).
        ("base_link", "torso_link", [-0.0039635, 0.0, 0.044], [0.0, 0.0, 0.0]),
        ("torso_link", "mid360_link", *MID360_JOINT),
    ],
    "unitree_g1_dex1": [("torso_link", "mid360_link", *MID360_JOINT)],
}

RANGE_SENSORS = ("LidarPlugin", "LivoxMid360Plugin")


def _spawn(model: str, timestep: float) -> Engine:
    world = {
        "sim": {"timestep": timestep},
        "components": [{"spawn_robot": {"model": model, "prefix": PREFIX}, "name": "robot"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1  # a test driving an Engine is the driver, and the seed is driver-owned
    engine.setup()
    engine.reset()
    engine.step()  # the rate gate starts open, so the first step casts
    return engine


def _body(model, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))


def _range_sensor(engine: Engine, address: str):
    sensors = [p for p in engine.plugins if type(p).__name__ in RANGE_SENSORS]
    assert [p.address for p in sensors] == [address]
    return sensors[0]


@pytest.mark.parametrize("model", list(CASES))
def test_the_scan_meets_no_robot_geometry_from_inside(model):
    timestep, address, expected_outside = CASES[model]
    engine = _spawn(model, timestep)
    try:
        m, d = engine.ctx.model, engine.ctx.data
        sensor = _range_sensor(engine, address)
        foreign = [_body(m, b) for b in range(1, m.nbody) if not _body(m, b).startswith(PREFIX)]
        assert not foreign, f"bodies not of the spawned robot: {foreign}"
        bid = sensor._bodyexclude
        excluded = _body(m, bid).removeprefix(PREFIX) if bid >= 0 else None
        assert excluded == (EXCLUDES.get(model) or STILL_EXCLUDES.get(model)), (
            f"the scan excludes {excluded!r}"
        )
        if excluded is not None and model in EXCLUDES:
            # A device's housing is a body of its own, with nothing of the robot under it.
            assert m.body_parentid[bid] != bid and not any(
                m.body_parentid[b] == bid for b in range(m.nbody)
            )

        # The plugin's own rays from its own site: `_local_dirs @ rot.T`, as post_step casts them.
        origin = d.site_xpos[sensor._site_id].copy()
        dirs = sensor._build_directions() @ d.site_xmat[sensor._site_id].reshape(3, 3).T
        hits = raycast.cast(
            m,
            d,
            origin,
            dirs,
            cutoff=sensor.range_max,
            bodyexclude=bid,
            out=raycast.buffers(len(dirs), normals=True),
        )
        np.testing.assert_array_equal(hits.geomid, sensor._hits.geomid)

        on_robot = (hits.geomid >= 0) & (m.geom_bodyid[np.maximum(hits.geomid, 0)] != 0)
        from_inside = on_robot & (np.einsum("ij,ij->i", hits.normal, dirs) > 0)
        inside = Counter(_body(m, m.geom_bodyid[g]) for g in hits.geomid[from_inside])
        assert not inside, f"rays start inside robot geometry: {dict(inside)}"

        outside = Counter(_body(m, m.geom_bodyid[g]) for g in hits.geomid[on_robot])
        assert dict(outside) == {PREFIX + b: n for b, n in expected_outside.items()}
    finally:
        engine.shutdown()


def _frame_pose(m, d, name: str) -> tuple[np.ndarray, np.ndarray]:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        return d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"{name} is neither a site nor a body"
    return d.xpos[bid], d.xmat[bid].reshape(3, 3)


def _mat(quat) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=np.float64))
    return out.reshape(3, 3)


@pytest.mark.parametrize("model", list(G1_TRANSFORMS))
def test_the_g1_mid360_is_at_the_vendor_joint_origin_and_publishes_that_chain(model):
    timestep, address, _ = CASES[model]
    engine = _spawn(model, timestep)
    try:
        m, d = engine.ctx.model, engine.ctx.data
        sensor = _range_sensor(engine, address)
        assert sensor.frame_id == "mid360_link"

        # The rays start at mid360_joint's origin in torso_link, in the device's point-cloud axes.
        xyz, rpy = MID360_JOINT
        tpos, tmat = _frame_pose(m, d, PREFIX + "torso_link")
        np.testing.assert_allclose(d.site_xpos[sensor._site_id], tpos + tmat @ xyz, atol=1e-9)
        np.testing.assert_allclose(
            d.site_xmat[sensor._site_id].reshape(3, 3), tmat @ _mat(rpy_to_quat(*rpy)), atol=1e-9
        )

        published = [
            t
            for e in engine.ctx.interface.all()
            if e.name == "frames"
            for t in e.backend["ros2"]["static_tf"]
        ]
        assert [(t["parent"], t["child"]) for t in published] == [
            (p, c) for p, c, _, _ in G1_TRANSFORMS[model]
        ]
        for tf, (_, _, pos, angles) in zip(published, G1_TRANSFORMS[model], strict=True):
            np.testing.assert_allclose(tf["translation"], pos, atol=1e-9)
            np.testing.assert_allclose(_mat(tf["rotation"]), _mat(rpy_to_quat(*angles)), atol=1e-9)
        cloud = next(e for e in engine.ctx.interface.all() if e.name == "cloud")
        assert cloud.backend["ros2"]["frame_id"] == "mid360_link"
        assert "static_tf" not in cloud.backend["ros2"]
    finally:
        engine.shutdown()
