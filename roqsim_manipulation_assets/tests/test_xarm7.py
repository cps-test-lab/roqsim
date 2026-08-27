"""The xArm 7 model: Menagerie's kinematics unchanged, and a collision envelope that is ours.

The port is a *transformation* of MuJoCo Menagerie's ``ufactory_xarm7``, so the test that earns its
place is the one asserting nothing drifted in transit: masses, inertias, joint limits and the
forward kinematics are upstream's, and if a regenerated MJCF ever disagrees the upstream arm changed
-- a decision to make, not a test to update quietly.

The collision envelope is the one part that is genuinely ours. Upstream collides against nine
full-detail convex hulls; this model uses primitives fitted by
``external/convert/build_xarm7_mjcf.py``, calibrated so self-collision reports at the same rate as
the package's own reference arms. That calibration is what ``test_self_collision_rate_matches_siblings``
pins, because it is invisible to every other check: the arm loads, compiles, holds and moves either
way, and only a planner would notice the envelope had silently inflated.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import apply_assets, resolve_model

#: From MuJoCo Menagerie's ufactory_xarm7 @ da76818e (BSD-3-Clause), which took them from the vendor
#: URDF. Not measured from our model -- that would make the test tautological.
TOTAL_MASS = 11.3171
JOINT_RANGES = {
    "joint1": (-6.28319, 6.28319),
    "joint2": (-2.059, 2.0944),
    "joint3": (-6.28319, 6.28319),
    "joint4": (-0.19198, 3.927),
    "joint5": (-6.28319, 6.28319),
    "joint6": (-1.69297, 3.14159),
    "joint7": (-6.28319, 6.28319),
}
#: UFACTORY publishes a 700 mm working radius; the flange reaches 770 mm because `attachment_site`
#: sits at the tool flange, past the wrist the datasheet measures to. Upstream reaches exactly the
#: same distance, so this is inherited, not introduced -- see the port log.
FLANGE_RADIUS = 0.770


def _model():
    asset = resolve_model("roqsim_manipulation_assets:xarm7")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    return spec.compile()


def test_mass_matches_upstream():
    assert _model().body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)


@pytest.mark.parametrize("joint,limits", JOINT_RANGES.items())
def test_joint_limits_match_upstream(joint, limits):
    model = _model()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    assert jid >= 0, f"{joint} is missing"
    assert model.jnt_range[jid] == pytest.approx(np.array(limits), abs=1e-5)


def test_no_option_block():
    # Timestep, integrator, solver and contact overrides are properties of the experiment and live
    # in the world's `sim:` block. Upstream pins integrator="implicitfast"; a model that carried it
    # would silently override every world it is spawned into. Parse rather than grep: the header
    # comment says the word "<option>" and a substring check matches its own documentation.
    import xml.etree.ElementTree as ET

    asset = resolve_model("roqsim_manipulation_assets:xarm7")
    assert ET.parse(asset.path).getroot().find("option") is None


def test_base_is_at_the_origin():
    # Upstream stands the arm on a 0.12 m plinth for its own scene.xml. spawn_arm places the arm via
    # an attach frame, so a baked offset would double up with whatever the world asks for.
    model = _model()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_base")
    assert model.body_pos[bid] == pytest.approx(np.zeros(3), abs=1e-9)


def test_flange_reach():
    model, data = _model(), None
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    assert site >= 0, "attachment_site is what a gripper welds to; it must exist"
    rng = np.random.default_rng(0)
    reach = 0.0
    for _ in range(4000):
        data.qpos[:] = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        mujoco.mj_forward(model, data)
        reach = max(reach, float(np.hypot(*data.site_xpos[site][:2])))
    assert reach == pytest.approx(FLANGE_RADIUS, abs=0.01)


def test_gravity_hold(tmp_path):
    """Held at the home keyframe under gravity, the arm must not sag."""
    plugins = [{
        "spawn_arm": {"model": "xarm7", "prefix": "xarm7_"},
        "name": "xarm7",
        "components": [{"arm_controller": {}}],
    }]
    engine = Engine(load_config_from_dict({"sim": {}, "components": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        qadr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"xarm7_{j}")]
            for j in JOINT_RANGES
        ]
        start = data.qpos[qadr].copy()
        for _ in range(1500):
            engine.step()
        drift = np.degrees(np.abs(data.qpos[qadr] - start)).max()
        assert drift < 2.0, f"arm sagged {drift:.2f} deg under gravity"
    finally:
        engine.shutdown()


def test_self_collision_rate_matches_siblings():
    """The fitted collision envelope must not be more conservative than the package's own arms.

    A capsule is a poor envelope for the xArm's prismatic links, so the radius is calibrated (see the
    converter). Left uncalibrated it reports self-collision on 35% of random configurations against
    upstream's 8%, and a motion planner would then refuse poses that are physically fine. ur10e and
    ur5e measure ~17.5% the same way; this asserts xarm7 is in that band rather than drifting up.
    """
    model = _model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    colliding = 0
    trials = 1500
    for _ in range(trials):
        data.qpos[:] = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        if any(data.contact[c].dist < -1e-4 for c in range(data.ncon)):
            colliding += 1
    rate = colliding / trials
    assert rate < 0.22, (
        f"self-collision on {rate:.1%} of random poses; the reference arms sit at ~17.5% and "
        f"upstream's exact hulls at 8%. The collision primitives have inflated."
    )
