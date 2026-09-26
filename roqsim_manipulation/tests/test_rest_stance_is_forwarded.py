"""A `rest:` stance is in the kinematics before the next plugin's on_reset reads them.

`arm_controller._apply_rest` writes joint positions during on_reset; the engine's closing forward
pass runs only after every plugin's hook. A Cartesian controller declared after the arm anchors on
`site_xpos` in its own on_reset, so without a forward pass in between it anchored on the pose from
BEFORE the stance and pulled the tool toward it from the first step.
"""

from __future__ import annotations

import numpy as np
from roqsim_manipulation.plugins.cartesian_admittance import CartesianAdmittancePlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

REST = {"shoulder_lift_joint": -0.5, "elbow_joint": 0.5}


def _engine():
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.001, "gravity": [0.0, 0.0, 0.0]},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {"arm_controller": {"rest": REST}},
                        {
                            "cartesian_admittance": {
                                "site": "tool_site",
                                "controller_type": "cartesian_motion_controller",
                                "rate_hz": 500.0,
                            }
                        },
                    ],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine


def test_the_cartesian_anchor_is_the_tool_pose_at_rest():
    engine = _engine()
    try:
        cart = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))
        anchored = cart._rest_pos
        actual = cart.read_pose()[0]  # after the engine's closing forward pass: the true pose
        assert np.allclose(anchored, actual, atol=1e-6), (
            f"anchored {np.linalg.norm(anchored - actual):.3f} m from the tool's reset pose"
        )
    finally:
        engine.shutdown()


def test_an_uncommanded_arm_at_rest_stays_where_it_was_reset():
    engine = _engine()
    try:
        cart = next(p for p in engine.plugins if isinstance(p, CartesianAdmittancePlugin))
        start = cart.read_pose()[0].copy()
        for _ in range(500):
            engine.step()
        moved = float(np.linalg.norm(cart.read_pose()[0] - start))
        assert moved < 2e-3, f"the tool moved {moved:.4f} m with no goal"
    finally:
        engine.shutdown()
