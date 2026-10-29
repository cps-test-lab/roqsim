"""``fzi_cartesian_controllers``: FZI's motion and compliance controllers on a UR5e, with FZI's
parameters, switched like ros2_control controllers, commanded in ur_description's base_link."""

from __future__ import annotations

import mujoco
import numpy as np
from roqsim_manipulation.plugins.fzi_cartesian_controllers import FZICartesianControllersPlugin

from roqsim.config import load_config_from_dict
from roqsim.controllers import INACTIVE, registry_for
from roqsim.engine import Engine
from roqsim.plugin import Plugin

HOME = [0.0, -1.3798, 1.8479, -1.6902, -1.571, -1.571]
GAINS = {a: {"p": 10.0} for a in ("trans_x", "trans_y", "trans_z")} | {
    a: {"p": 1.5} for a in ("rot_x", "rot_y", "rot_z")
}
FORCE_GAINS = {a: {"p": 0.05} for a in ("trans_x", "trans_y", "trans_z")} | {
    a: {"p": 1.5} for a in ("rot_x", "rot_y", "rot_z")
}
SOLVER = {"error_scale": 1.0, "iterations": 10}


class _Block(Plugin):
    """A static block whose top lies 20 mm below the flange at HOME."""

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        self.body = spec.worldbody.add_body(name="press_block", pos=[0, 0, 0])
        g = self.body.add_geom(name="press_block_geom")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.05, 0.05, 0.02]


TOOL = """<mujoco><worldbody><body name="probe">
  <geom type="box" size=".02 .02 .02" pos="0 0 .02" mass="0.2"/>
</body></worldbody></mujoco>"""


def _engine(block: bool = False, tmp_path=None) -> Engine:
    arm = {
        "model": "ur5e",
        "prefix": "ur5e_",
        "home": HOME,
        "actuators": {"control": "position", "p": 50000, "d": 450},
    }
    if tmp_path is not None:
        (tmp_path / "probe.xml").write_text(TOOL)
        arm["end_effector"] = {"model": "probe.xml"}
    components = [
        {
            "spawn_arm": arm,
            "name": "ur5e",
            "components": [
                {"force_torque": {"site": "fts_site", "frame": "sensor"}, "name": "ft"},
                {
                    "fzi_cartesian_controllers": {
                        "ft": "ft",
                        "robot_base_link": {"body": "base", "yaw": 0.0},
                        "controllers": {
                            "cartesian_motion_controller": {
                                "type": "motion",
                                "solver": SOLVER,
                                "pd_gains": GAINS,
                            },
                            "cartesian_compliance_controller": {
                                "type": "compliance",
                                "solver": SOLVER,
                                "pd_gains": FORCE_GAINS,
                                "stiffness": {
                                    "trans_x": 2000,
                                    "trans_y": 2000,
                                    "trans_z": 0,
                                    "rot_x": 50,
                                    "rot_y": 50,
                                    "rot_z": 50,
                                },
                            },
                        },
                    },
                    "name": "fzi",
                },
            ],
        }
    ]
    if block:
        components.append({f"{__name__}:_Block": {}})
    engine = Engine(
        load_config_from_dict(
            {"sim": {"timestep": 0.002}, "components": components}, base_dir=tmp_path
        )
    )
    engine.ctx.seed = 0
    engine.setup()
    for _ in range(100):
        engine.step()
    return engine


def _fzi(engine) -> FZICartesianControllersPlugin:
    return next(p for p in engine.plugins if isinstance(p, FZICartesianControllersPlugin))


def _switch(engine, name: str) -> None:
    ok, msg = registry_for(engine.ctx).switch(
        activate=[name], deactivate=[], strictness=2, namespace="", sim_time=engine.ctx.sim_time
    )
    assert ok, msg


def _flange(engine) -> np.ndarray:
    m, d = engine.ctx.model, engine.ctx.data
    return np.array(d.site_xpos[m.site("ur5e_tool0_site").id])


def test_the_controllers_are_listed_under_fzis_types_and_wait_inactive():
    listed = {c.name: (c.type, c.state) for c in registry_for(_engine().ctx).all()}
    assert listed["cartesian_motion_controller"] == (
        "cartesian_motion_controller/CartesianMotionController",
        INACTIVE,
    )
    assert listed["cartesian_compliance_controller"] == (
        "cartesian_compliance_controller/CartesianComplianceController",
        INACTIVE,
    )


def test_motion_reaches_a_target_given_in_base_link():
    engine = _engine()
    fzi = _fzi(engine)
    start = _flange(engine)
    _switch(engine, "cartesian_motion_controller")
    # base_link is the base body, itself a half turn from the world here: +x in base_link is -x.
    p_base, R_base = fzi._base_frame(engine.ctx.data)
    target_world = start + np.array([-0.03, 0.0, 0.02])
    target_base = R_base.T @ (target_world - p_base)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_base.T @ fzi._ee_mat).reshape(9))
    fzi.target_frame("cartesian_motion_controller", target_base, q)
    for _ in range(1500):
        engine.step()
    assert np.linalg.norm(_flange(engine) - target_world) < 0.001


def test_a_target_sent_while_inactive_is_dropped():
    engine = _engine()
    fzi = _fzi(engine)
    fzi.target_frame(
        "cartesian_motion_controller", np.array([0.3, 0.3, 0.3]), np.array([1.0, 0, 0, 0])
    )
    start = _flange(engine)
    _switch(engine, "cartesian_motion_controller")
    for _ in range(500):
        engine.step()
    assert np.linalg.norm(_flange(engine) - start) < 0.002  # held where it was activated


def _press(tmp_path, servo_p: float, servo_d: float) -> tuple[float, float]:
    """(mean, spread) of the wrist's vertical reading in the last second of a 10 N compliant press.

    A rigid probe on the flange, a block 10 mm below it, no stiffness along z, gains a stiff contact
    tolerates (one iteration, p 0.002 in translation, 0.01 in rotation).
    """
    (tmp_path / "probe.xml").write_text(TOOL)
    gains = {a: {"p": 0.002} for a in ("trans_x", "trans_y", "trans_z")} | {
        a: {"p": 0.01} for a in ("rot_x", "rot_y", "rot_z")
    }
    arm = {
        "model": "ur5e",
        "prefix": "ur5e_",
        "home": HOME,
        "end_effector": {"model": "probe.xml"},
        "actuators": {"control": "position", "p": servo_p, "d": servo_d},
    }
    fzi = {
        "ft": "ft",
        "robot_base_link": {"body": "base", "yaw": 0.0},
        "controllers": {
            "cartesian_compliance_controller": {
                "type": "compliance",
                "solver": {"error_scale": 1.0, "iterations": 1},
                "pd_gains": gains,
                "stiffness": {
                    "trans_x": 2000,
                    "trans_y": 2000,
                    "trans_z": 0,
                    "rot_x": 50,
                    "rot_y": 50,
                    "rot_z": 50,
                },
            }
        },
    }
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.0004},
            "components": [
                {
                    "spawn_arm": arm,
                    "name": "ur5e",
                    "components": [
                        {"force_torque": {"site": "fts_site", "frame": "sensor"}, "name": "ft"},
                        {"fzi_cartesian_controllers": fzi, "name": "fzi"},
                    ],
                },
                {f"{__name__}:_Block": {}},
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    for _ in range(500):
        engine.step()
    m, d = engine.ctx.model, engine.ctx.data
    m.geom_solref[m.geom("press_block_geom").id] = [0.02, 1.0]
    tip = np.array(d.site_xpos[m.site("ur5e_tool_site").id])
    m.body_pos[m.body("press_block").id] = tip - np.array([0.0, 0.0, 0.04 + 0.02 + 0.01])
    mujoco.mj_forward(m, d)
    plugin = _fzi(engine)
    _switch(engine, "cartesian_compliance_controller")
    _, R_base = plugin._base_frame(d)
    plugin.target_wrench(
        "cartesian_compliance_controller", R_base.T @ np.array([0.0, 0.0, -10.0]), np.zeros(3)
    )
    reader, site = engine.ctx.blackboard.get("ft:ft"), m.site("ur5e_fts_site").id
    readings = []
    for i in range(15000):  # 6 s at 0.4 ms; the last second is read
        engine.step()
        if i >= 12500:
            f, _ = reader.read()
            readings.append((np.array(d.site_xmat[site]).reshape(3, 3) @ f)[2])
    return float(np.mean(readings)), float(np.std(readings))


def test_a_compliant_press_settles_and_its_force_error_is_the_servos(tmp_path):
    """FZI's law re-reads the measured joints every cycle and commands a step from there, so it
    presses with the commanded force only on an arm that realises each commanded position, as a
    UR's own servo does. A simulated servo yields under the contact, and what it yields is lost at
    the next re-read: the settled force falls short of the target by the servo's compliance, and
    a stiffer servo closes the gap."""
    soft, soft_spread = _press(tmp_path, 50_000, 450)
    stiff, stiff_spread = _press(tmp_path, 200_000, 900)
    assert soft_spread < 1.0 and stiff_spread < 1.0  # settled, not chattering
    assert 0.0 < soft < stiff < 10.5  # pressing, short of the target, closer with the stiffer servo
    assert stiff > 5.0


def test_robot_base_link_is_the_base_body_turned_by_the_stated_yaw():
    engine = _engine()
    fzi = _fzi(engine)
    m, d = engine.ctx.model, engine.ctx.data
    p, R = fzi._base_frame(d)
    base = m.body("ur5e_base").id
    assert np.allclose(p, d.xpos[base]) and np.allclose(R, d.xmat[base].reshape(3, 3))
    assert np.allclose(
        R, np.diag([-1.0, -1.0, 1.0]), atol=1e-9
    )  # the UR5e MJCF's base, a half turn


def test_config_is_checked():
    cfg = {"controllers": {"c": {"type": "teleport"}}}
    assert any(
        "type is one of" in e for e in FZICartesianControllersPlugin(cfg).validate_config(cfg)
    )
    cfg = {"controllers": {"c": {"type": "compliance"}}}
    assert any(
        "'ft' must name" in e for e in FZICartesianControllersPlugin(cfg).validate_config(cfg)
    )
