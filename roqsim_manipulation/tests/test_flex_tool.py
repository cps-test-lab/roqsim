"""An arm's tool may be a flex: it attaches, compiles, and hangs under its own weight.

A soft pad or a compliant finger is written as MuJoCo's own ``<flexcomp>`` in the end-effector MJCF.
What has to hold for it: the pad survives the attach whichever way its pins are written, the world
compiles under the integrator ``auto`` picks for it, and gravity compensation -- which holds the arm
where it was sent -- leaves the pad's vertices to their weight, so it sags the way a real pad on a
still arm does. Measured on the running arm rather than read off ``body_gravcomp``. And mounted
where the arm's manifest puts a tool, a pad that makes contact hangs below the arm's force/torque
site, so the sensor's refusal of a flex contact it cannot see applies to it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

_PAD = 'type="grid" count="4 2 2" spacing=".015 .015 .015" dim="3" radius=".002" mass=".05"'
_MATERIAL = (
    '<edge equality="false"/><elasticity young="2e4" poisson="0.3" damping="0.002"/>'
    '<contact selfcollide="none" contype="0" conaffinity="0"/>'
)

#: A mount plate with a pad cantilevered off it, its first layer pinned to the plate.
PINNED_IN_A_BODY = f"""<mujoco><worldbody><body name="pad_mount">
  <geom type="box" size=".02 .02 .005" mass=".05"/>
  <flexcomp name="pad" {_PAD} pos=".03 0 .015"><pin gridrange="0 0 0 0 1 1"/>{_MATERIAL}</flexcomp>
</body></worldbody></mujoco>"""

#: The same pad written under <worldbody>, pinned there -- to the flange, once attached.
PINNED_AT_TOP = f"""<mujoco><worldbody>
  <flexcomp name="pad" {_PAD} pos=".03 0 .015"><pin gridrange="0 0 0 0 1 1"/>{_MATERIAL}</flexcomp>
</worldbody></mujoco>"""

HOLD_S = 1.0


def _arm(tmp_path, xml):
    (tmp_path / "pad_tool.xml").write_text(xml, encoding="utf-8")
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "ur5e_",
                        "end_effector": {"model": "pad_tool.xml"},
                    },
                    "name": "arm",
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _hold(engine):
    """Command the pose the arm stands in, step, and return (flange drop, mean pad drop) in metres."""
    model, data = engine.ctx.model, engine.ctx.data
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)
    site = model.site("ur5e_attachment_site").id
    flange0 = float(data.site_xpos[site][2])
    pad0 = data.flexvert_xpos[:, 2].copy()
    for _ in range(int(HOLD_S / model.opt.timestep)):
        engine.step()
    return flange0 - float(data.site_xpos[site][2]), float(np.mean(pad0 - data.flexvert_xpos[:, 2]))


@pytest.mark.parametrize("xml", [PINNED_IN_A_BODY, PINNED_AT_TOP], ids=["in_a_body", "at_top"])
def test_a_pinned_flex_tool_attaches_and_compiles_under_auto(tmp_path, xml):
    engine = _arm(tmp_path, xml)
    model = engine.ctx.model
    assert model.nflex == 1
    assert engine.integrator.resolved == "discrete"
    assert engine.ctx.entities.get("arm").meta["flexes"] == ["ur5e_pad"]
    # Its pins hold on the tool, under the arm's prefix.
    pins = {model.body(int(b)).name for b in model.flex_vertbodyid if model.body_jntnum[b] == 0}
    assert pins and all(name.startswith("ur5e_") for name in pins)


def test_the_arm_holds_its_pose_and_the_pad_sags_under_its_own_weight(tmp_path):
    engine = _arm(tmp_path, PINNED_IN_A_BODY)
    model = engine.ctx.model
    flange_drop, pad_drop = _hold(engine)
    assert abs(flange_drop) < 1e-3, "the arm is compensated, the flex tool's weight included"
    # A compensated pad stays exactly where it was modelled (measured: 0.0); this one bends down,
    # by 9 mm on average.
    assert pad_drop > 1e-3
    vertex_bodies = [int(b) for b in model.flex_vertbodyid if model.body_jntnum[b] > 0]
    assert vertex_bodies and all(model.body_gravcomp[b] == 0.0 for b in vertex_bodies)


#: The same pad, making contact -- the case a force_torque above it cannot measure.
COLLIDING_PAD = PINNED_IN_A_BODY.replace(
    'contype="0" conaffinity="0"', 'contype="1" conaffinity="1"'
)


def _sensed(tmp_path, **ft):
    (tmp_path / "pad_tool.xml").write_text(COLLIDING_PAD, encoding="utf-8")
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "ur5e_",
                        "end_effector": {"model": "pad_tool.xml"},
                    },
                    "name": "arm",
                    "components": [{"force_torque": {"site": "fts_site", **ft}, "name": "ft"}],
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    return engine


def test_a_colliding_pad_mounted_by_default_is_below_the_sensor_and_refused(tmp_path):
    """Mounted where the arm's manifest puts a tool, the pad is under ``fts_site``: its contacts
    would be missing from the wrench, so the flex refusal applies to it."""
    with pytest.raises(RuntimeError, match=r"'ur5e_pad' hangs below it"):
        _sensed(tmp_path)


def test_a_colliding_pad_is_accepted_once_the_world_states_it(tmp_path):
    engine = _sensed(tmp_path, flex_reaction="excluded")
    assert engine.ctx.entities.get("arm").meta["end_effector"]["site"] == "ur5e_tool_site"
