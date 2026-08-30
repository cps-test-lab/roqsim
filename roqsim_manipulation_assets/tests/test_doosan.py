"""The Doosan M1013: Doosan's numbers, an assembled arm, and a chain that does not chatter.

Three of these tests pin things this port got wrong first, none of which move a number the obvious
checks look at.

``test_meshes_assemble_into_a_continuous_arm`` guards the failure that reached a render before it was
noticed: the arm arrived **in pieces**, with a 43 cm gap between link_3 and link_4 and its base
buried below the floor -- while the mass audit, the joint limits and the reach all passed, because
they are computed from the kinematic chain and the site, not from where the geometry sits.

``test_rests_without_chattering`` guards the opposite kind of invisibility. With one stiff servo
setting for all six joints, joints 5 and 6 sat pinned at their force limit flipping velocity sign
every step -- ``kv*dt/I`` of 2.9, past the explicit-damping threshold. Position error stayed under
0.03 deg, so a gravity-hold test passed while the distal joints buzzed.

``test_self_collision_matches_the_vendor_geometry`` pins the collision decision. Doosan's own
``*_collision`` meshes are byte-for-byte copies of its full-detail visual CAD, so collision here is
the convex hull of each decimated mesh. Fitted primitives were tried and measured 5x too
conservative; the number below is the vendor geometry's own rate.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

#: From Doosan's expanded dsr_description2 @ 816ecb5d, not measured from our model.
TOTAL_MASS = 34.230
JOINTS = [f"joint_{i}" for i in range(1, 7)]
#: Doosan publishes 1300 mm reach and 33 kg for the M1013.
PUBLISHED_REACH = 1.300
#: Measured against the vendor's own meshes used as collision, over 1500 sampled configurations.
VENDOR_SELF_COLLISION = 0.117


def _model():
    return mujoco.MjModel.from_xml_path(str(resolve_model("roqsim_manipulation_assets:m1013").path))


def test_mass_matches_the_vendor_description():
    assert _model().body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)


def test_reach_matches_the_datasheet():
    model, data = _model(), None
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    rng = np.random.default_rng(0)
    reach = 0.0
    for _ in range(6000):
        data.qpos[:] = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        mujoco.mj_forward(model, data)
        reach = max(reach, float(np.hypot(*data.site_xpos[site][:2])))
    assert reach == pytest.approx(PUBLISHED_REACH, abs=0.02)


def test_meshes_assemble_into_a_continuous_arm():
    """The visual chain must be gapless and stand on the floor.

    Both failed at first, and silently. ``dae2obj`` writes Z-up OBJs while ``reduce-mesh`` imported
    OBJ with Blender's Y-up default, so every decimated mesh came back rotated 90 degrees about x:
    the base sank 349 mm below the floor and a 43 cm hole opened between link_3 and link_4. Nothing
    else noticed -- the kinematics matched the URDF to float precision throughout.
    """
    model = _model()
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    spans = {}
    for geom in range(model.ngeom):
        if model.geom_group[geom] != 2 or model.geom_dataid[geom] < 0:
            continue
        mesh = model.geom_dataid[geom]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
        body = model.geom_bodyid[geom]
        world = (verts @ rot.reshape(3, 3).T + model.geom_pos[geom]) @ np.array(
            data.xmat[body]
        ).reshape(3, 3).T + data.xpos[body]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        lo, hi = float(world[:, 2].min()), float(world[:, 2].max())
        spans[name] = (min(spans.get(name, (lo, hi))[0], lo), max(spans.get(name, (lo, hi))[1], hi))

    floor = min(lo for lo, _ in spans.values())
    assert abs(floor) < 0.02, f"lowest visual point is z={floor:+.3f}; the arm is not on the floor"

    chain = ["base_link"] + [f"link_{i}" for i in range(1, 7)]
    for lower, upper in zip(chain, chain[1:]):
        gap = spans[upper][0] - spans[lower][1]
        assert gap < 0.05, (
            f"{gap:.3f} m gap between {lower} and {upper} at the zero pose -- the arm is in pieces"
        )


def test_rests_without_chattering():
    """Held at its home keyframe the arm must be still, not merely in the right place."""
    model = _model()
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    for _ in range(6000):
        mujoco.mj_step(model, data)
    velocity = float(np.abs(data.qvel).max())
    assert velocity < 1e-3, (
        f"max |qvel| {velocity:.3e} rad/s after 12 s. The distal joints chatter when one stiff servo "
        f"setting is used for the whole chain; gains scale with the vendor's effort limits instead."
    )
    sag = float(np.degrees(np.abs(data.qpos - model.key_qpos[0])).max())
    assert sag < 1.0, f"sagged {sag:.3f} deg under gravity"


def test_joint_limits_are_enforced():
    model = _model()
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.jnt_range[:, 1] + 2.0
    for _ in range(4000):
        mujoco.mj_step(model, data)
    overshoot = float((data.qpos - model.jnt_range[:, 1]).max())
    assert overshoot < np.radians(1.0), f"overshot a limit by {np.degrees(overshoot):.3f} deg"


def test_self_collision_matches_the_vendor_geometry():
    """Collision is the convex hull of each decimated mesh, so the rate must be the vendor's own.

    Fitted primitives, as the xArm uses, were measured at 62.4% against this geometry's 11.7% --
    a single capsule is a poor envelope for a 0.62 m upper arm, and shrinking the radius to
    compensate stops containing the mesh long before it reaches the right rate.
    """
    model = _model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    colliding = 0
    trials = 800
    for _ in range(trials):
        data.qpos[:] = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        if any(data.contact[c].dist < -1e-4 for c in range(data.ncon)):
            colliding += 1
    rate = colliding / trials
    assert abs(rate - VENDOR_SELF_COLLISION) < 0.05, (
        f"self-collision {rate:.1%} against the vendor geometry's {VENDOR_SELF_COLLISION:.1%}"
    )


def test_spawns_and_holds_through_the_substrate(tmp_path):
    world = {
        "sim": {},
        "components": [{
            "spawn_arm": {"model": "m1013", "prefix": "m_"},
            "name": "m",
            "components": [{"arm_controller": {}}],
        }],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        adr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"m_{j}")]
            for j in JOINTS
        ]
        start = np.array([float(data.qpos[a]) for a in adr])
        for _ in range(3000):
            engine.step()
        drift = np.degrees(np.abs(np.array([float(data.qpos[a]) for a in adr]) - start)).max()
        assert drift < 2.0, f"arm drifted {drift:.2f} deg when spawned through the substrate"
    finally:
        engine.shutdown()
