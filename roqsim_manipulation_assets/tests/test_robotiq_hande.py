# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The Robotiq Hand-E against Robotiq's own numbers.

The shared gripper tests (test_grippers.py, test_gripper_command_tracks.py,
test_gripper_max_effort.py) check what every gripper must do. This file checks what only the Hand-E's
datasheet says: its mass properties, the shape of its travel, its speed, and how it sits on a UR5e.
The source of every expected value is Robotiq's Hand-E instruction manual for Universal Robots,
revision 2022/01/10, cited by section.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.models import apply_assets, resolve_model

LEFT, RIGHT = "hande_left_finger_joint", "hande_right_finger_joint"
#: Robotiq's convention: each finger joint is its travel, 0 open to 0.025 m closed.
OPEN_Q, CLOSED_Q = 0.0, 0.025
PADS = ("hande_left_pad", "hande_right_pad")
PAD_HALF_THICKNESS = 0.003

#: 5.2.3: 1068 g, CoM (0, 0, 59.0) mm; 5.2.1: 1070 g with the coupling.
MASS_KG = 1.070
COM_Z_M = 0.0590
#: 5.2.3, Fig. 5-13: inertia with the fingers fully open, kg mm^2. The manual names no reference
#: point; read as the mounting face, see robotiq_hande.xml.
INERTIA_KG_MM2 = (5341.0, 5447.0, 636.0)


def _spec() -> mujoco.MjSpec:
    asset = resolve_model("robotiq_hande")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    return spec


def _set_fingers(model, data, q: float) -> None:
    for name in (LEFT, RIGHT):
        data.qpos[model.joint(name).qposadr[0]] = q
    mujoco.mj_forward(model, data)


def test_mass_centre_of_mass_and_inertia_are_the_manuals():
    model = _spec().compile()
    data = mujoco.MjData(model)
    _set_fingers(model, data, OPEN_Q)  # the manual states the inertia with the fingers open

    mass = float(model.body_mass.sum())
    com = sum(model.body_mass[b] * data.xipos[b] for b in range(model.nbody)) / mass
    inertia = np.zeros((3, 3))
    for b in range(1, model.nbody):
        rot = data.ximat[b].reshape(3, 3)
        r = data.xipos[b]
        inertia += rot @ np.diag(model.body_inertia[b]) @ rot.T
        inertia += model.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))

    assert mass == pytest.approx(MASS_KG, abs=0.005)
    assert com[:2] == pytest.approx([0.0, 0.0], abs=1e-4)
    assert com[2] == pytest.approx(COM_Z_M, abs=0.001)
    assert np.diag(inertia) * 1e6 == pytest.approx(INERTIA_KG_MM2, rel=0.05)


# --- the travel: aperture, pad alignment, coupling and symmetry -----------------------------------

#: Register counts along the travel, and how far the measured aperture may sit from the linear map.
REGISTER = (0, 64, 128, 192, 255)
APERTURE_TOL_MM = 0.5
SETTLE_STEPS = 1500  # 3 s at 2 ms: the full stroke takes 0.17 s at 150 mm/s


def _world(integrator: str = "implicitfast") -> tuple[mujoco.MjModel, mujoco.MjData]:
    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{integrator.upper()}")
    world.option.gravity = [0.0, 0.0, 0.0]
    world.attach(_spec(), prefix="", frame=world.worldbody.add_frame())
    model = world.compile()
    return model, mujoco.MjData(model)


def _gap_mm(model, data) -> float:
    """Distance between the two pads' inner faces, along the finger axis."""
    a, b = (data.geom(g).xpos for g in PADS)
    return (float(np.linalg.norm(a - b)) - 2 * PAD_HALF_THICKNESS) * 1000.0


def _settle(model, data, ctrl: float, steps: int = SETTLE_STEPS) -> None:
    data.ctrl[model.actuator("fingers_actuator").id] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)


def test_aperture_follows_the_register_linearly():
    """0 is 50 mm open, 255 closed, 0.196 mm per count between (manual: about 0.2 mm per count)."""
    model, data = _world()
    _set_fingers(model, data, OPEN_Q)
    for count in REGISTER:
        _settle(model, data, count)
        assert _gap_mm(model, data) == pytest.approx(
            50.0 * (1 - count / 255), abs=APERTURE_TOL_MM
        ), f"register {count}"


def test_pads_stay_parallel_and_face_each_other():
    """Each pad's inner face points at the other pad, along the finger axis, at every opening.

    The closed end is left out: there the pads touch and the line between them has no direction.
    """
    model, data = _world()
    _set_fingers(model, data, OPEN_Q)
    for count in REGISTER[:-1]:
        _settle(model, data, count)
        left, right = (data.geom(g) for g in PADS)
        # The left finger sits at -x and closes along +x, so its pad's inner face is the box's +x
        # face; the right one's is its -x face.
        n_left = left.xmat.reshape(3, 3)[:, 0]
        n_right = -right.xmat.reshape(3, 3)[:, 0]
        towards = (right.xpos - left.xpos) / np.linalg.norm(right.xpos - left.xpos)
        assert float(n_left @ towards) == pytest.approx(1.0, abs=1.5e-4), count  # within 1 degree
        assert float(n_left @ n_right) == pytest.approx(-1.0, abs=1.5e-4), count
        assert abs(float(towards[0])) == pytest.approx(1.0, abs=1.5e-4), count


def test_the_linkage_is_one_coupling_and_drives_both_fingers():
    """One joint equality and no loop closure; opening from closed moves BOTH pads."""
    model, data = _world()
    kinds = [int(model.eq_type[i]) for i in range(model.neq)]
    assert kinds == [mujoco.mjtEq.mjEQ_JOINT], "a two-slide gripper is one coupling, no connect"

    _set_fingers(model, data, CLOSED_Q)  # start at the end the command will leave
    before = [data.geom(g).xpos.copy() for g in PADS]
    _settle(model, data, 0)
    moved = [
        float(np.linalg.norm(data.geom(g).xpos - p)) for g, p in zip(PADS, before, strict=True)
    ]
    assert all(m == pytest.approx(0.025, abs=5e-4) for m in moved), moved


def test_pads_are_symmetric_about_the_pinch_site():
    """The grasp centre a driver reaches is midway between the pads and inside their band."""
    model, data = _world()
    for q in (OPEN_Q, 0.0125):
        _set_fingers(model, data, q)
        left, right = (data.geom(g).xpos for g in PADS)
        pinch = data.site("pinch").xpos
        assert (left + right) / 2 == pytest.approx(pinch, abs=1e-6)
        assert _gap_mm(model, data) == pytest.approx(50.0 - 2000.0 * q, abs=1e-6)


# --- speed and servo stability -------------------------------------------------------------------

#: 5.2.1: finger speed 20 to 150 mm/s; the model's top speed is the full-force one.
TOP_SPEED_M_S = 0.150


def test_fingers_close_at_the_datasheet_top_speed():
    model, data = _world()
    _set_fingers(model, data, OPEN_Q)
    dof = model.joint(LEFT).dofadr[0]
    data.ctrl[model.actuator("fingers_actuator").id] = 255
    speeds = []
    for _ in range(150):  # 0.3 s, longer than the stroke takes
        mujoco.mj_step(model, data)
        speeds.append(abs(float(data.qvel[dof])))
    assert max(speeds) == pytest.approx(TOP_SPEED_M_S, rel=0.1)
    assert max(speeds) <= TOP_SPEED_M_S * 1.1


@pytest.mark.parametrize("integrator", ["euler", "implicitfast"])
def test_the_servo_holds_both_stops_without_ripple(integrator):
    """A stiff position servo on a light jaw is where a limit cycle hides (see schunk_pg70.xml)."""
    model, data = _world(integrator)
    _set_fingers(model, data, 0.0125)
    adr = model.joint(LEFT).qposadr[0]
    for count, target in ((0, OPEN_Q), (255, CLOSED_Q)):
        _settle(model, data, count, steps=1000)
        trace = []
        for _ in range(250):
            mujoco.mj_step(model, data)
            trace.append(float(data.qpos[adr]))
        assert np.mean(trace) == pytest.approx(target, abs=2e-4), f"register {count}"
        assert max(trace) - min(trace) < 5e-5, f"register {count}: ripple"


# --- mounted on a UR5e ---------------------------------------------------------------------------

#: The UR5e's tool_site is the outer face of its FT adapter stack, this far out from tool0.
STACK_M = 0.059
#: A joint held by a drive that carries its own weight sits within this of its target
#: (test_ur5e_actuator_gains.py, HELD_RAD).
HELD_RAD = 1e-3


@pytest.fixture(scope="module")
def ur5e_with_hande():
    from pathlib import Path

    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"timestep": 0.002},
                "components": [
                    {
                        "spawn_arm": {
                            "model": "ur5e",
                            "prefix": "ur5e_",
                            "end_effector": {"model": "robotiq_hande"},
                        },
                        "name": "ur5e",
                    }
                ],
            },
            base_dir=Path("."),
        )
    )
    engine.setup()
    engine.reset()
    for _ in range(1500):  # 3 s
        engine.step()
    yield engine
    engine.shutdown()


def _in_tool0(engine, point) -> np.ndarray:
    model, data = engine.ctx.model, engine.ctx.data
    tool0 = model.body("ur5e_tool0").id
    rot = data.xmat[tool0].reshape(3, 3)
    return rot.T @ (np.asarray(point) - data.xpos[tool0])


def test_the_coupling_face_sits_on_the_tool_site(ur5e_with_hande):
    data = ur5e_with_hande.ctx.data
    face = data.body("ur5e_hand_e_coupling").xpos
    assert np.linalg.norm(face - data.site("ur5e_tool_site").xpos) < 5e-4
    assert _in_tool0(ur5e_with_hande, face) == pytest.approx([0.0, 0.0, STACK_M], abs=5e-4)


def test_tcp_and_pinch_are_where_the_world_yaml_says(ur5e_with_hande):
    """Robotiq's TCP 157.0 mm from the coupling face (5.2.3); pinch mid-pad, 146.75 mm."""
    data = ur5e_with_hande.ctx.data
    tcp = _in_tool0(ur5e_with_hande, data.site("ur5e_tcp").xpos)
    pinch = _in_tool0(ur5e_with_hande, data.site("ur5e_pinch").xpos)
    assert tcp == pytest.approx([0.0, 0.0, STACK_M + 0.157], abs=1e-4)
    assert pinch == pytest.approx([0.0, 0.0, STACK_M + 0.14675], abs=1e-4)


def test_the_fingers_close_along_tool0_x(ur5e_with_hande):
    """No rotation on attach: the fingers move along the flange's x, as in Robotiq's description."""
    data = ur5e_with_hande.ctx.data
    left, right = (data.geom(f"ur5e_{g}").xpos for g in PADS)
    axis = _in_tool0(ur5e_with_hande, right) - _in_tool0(ur5e_with_hande, left)
    assert axis / np.linalg.norm(axis) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_the_arm_still_reaches_home_and_touches_nothing(ur5e_with_hande):
    engine = ur5e_with_hande
    model, data = engine.ctx.model, engine.ctx.data
    _, position, _, _ = engine.ctx.blackboard.require("arm:ur5e").read_state()
    target = engine.plugins[0]._home_vector()[:6]
    assert np.abs(np.asarray(position[:6]) - np.asarray(target)).max() < HELD_RAD

    gripper = {b for b in range(model.nbody) if "hande" in model.body(b).name}
    touching = [
        (model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name)
        for c in data.contact[: data.ncon]
        if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} & gripper
    ]
    assert not touching, f"the mounted gripper is in contact: {touching}"


# --- a held load: picked off the floor by the UR5e and carried --------------------------------------

UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
#: Stances solved by damped least squares on the spawned model: `pinch` over the box, the approach
#: axis straight down and the fingers along world y, at the grasp height and 132 mm above it.
GRASP = dict(zip(UR5E_JOINTS, (-0.3007, -1.4796, 2.2129, -2.3040, -1.5708, -0.3007), strict=True))
LIFT = dict(zip(UR5E_JOINTS, (-0.3007, -1.6646, 2.0034, -1.9097, -1.5708, -0.3007), strict=True))
#: graspable_box at scale 0.6: 48 x 36 x 30 mm, its 36 mm side along world y, between the fingers;
#: it keeps its modelled 0.5 kg.
BOX_SCALE, BOX_HALF_Z = 0.6, 0.015
BOX_XY = (-0.45, 0.0)


def test_the_ur5e_picks_a_box_and_holds_it():
    """Lifted, and still between both pads after a dwell, not merely nudged.

    The arm spawns at the grasp stance: driven there from home, the fingers would sweep the box away
    first, and a check made only at the lift would score that as a pick.
    """
    from pathlib import Path

    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    engine = Engine(
        load_config_from_dict(
            {
                "sim": {
                    "timestep": 0.002,
                    "cone": "elliptic",
                    "impratio": 10,
                    "noslip_iterations": 10,
                },
                "components": [
                    {
                        "spawn_arm": {
                            "model": "ur5e",
                            "prefix": "ur5e_",
                            "end_effector": {"model": "robotiq_hande"},
                        },
                        "name": "ur5e",
                        "components": [{"arm_controller": {"rest": GRASP}}],
                    },
                    {
                        "spawn_model": {
                            "model": "graspable_box",
                            "motion": "physics",
                            "scale": BOX_SCALE,
                            "pose": {"position": {"x": BOX_XY[0], "y": BOX_XY[1], "z": BOX_HALF_Z}},
                        },
                        "name": "box",
                    },
                ],
            },
            base_dir=Path("."),
        )
    )
    engine.setup()
    engine.reset()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        controller = next(p for p in engine.plugins if type(p).__name__ == "ArmControllerPlugin")
        box = model.body("graspable_box").id
        for _ in range(1000):
            engine.step()
        assert float(data.xpos[box][2]) == pytest.approx(BOX_HALF_Z, abs=0.002), (
            "the box moved before the grasp: the fingers swept into it"
        )
        controller._gripper_ctrl_target = 255.0
        for _ in range(1000):
            engine.step()
        controller._target = LIFT
        for _ in range(3000):
            engine.step()
        lifted = float(data.xpos[box][2])
        assert lifted - BOX_HALF_Z > 0.1, f"the box reached only z = {lifted:.3f}: not picked up"

        pads = {model.geom(f"ur5e_{g}").id for g in PADS}
        held = [lifted]
        for _ in range(4):  # 4 s dwell
            for _ in range(500):
                engine.step()
            held.append(float(data.xpos[box][2]))
            touching = {
                g
                for c in data.contact[: data.ncon]
                for g in (int(c.geom1), int(c.geom2))
                if g in pads and box in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            }
            assert touching == pads, "the box is no longer between both pads"
        creep = (held[0] - held[-1]) / (2000 * engine.ctx.dt)
        assert abs(creep) < 0.001, f"the box crept {creep * 1000:.2f} mm/s out of the jaws"
    finally:
        engine.shutdown()
