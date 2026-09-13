# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A straight Cartesian move under the motion law stays on its line.

The servo lags the joint target by a whole configuration during a move, so the step the law adds
to that target has to be resolved at the target, not at the lagging arm. Resolved at the arm, the
target walks off the commanded line, and a tool sent straight down arrives having swept sideways --
against a task whose clearance is a millimetre, that is the difference between entering a bore and
scraping its wall.
"""

from __future__ import annotations

import numpy as np

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: The move: straight down from the home pose, far enough for the clamp to saturate for most of it.
GOAL_DZ = -0.08

#: Largest sideways excursion of the controlled site from the vertical through its start. What is
#: left is the servo's own lag along the joint path, which no Cartesian law removes.
MAX_LATERAL_M = 1.5e-3

#: Where the move must end: it arrives, rather than trading the line for the goal.
MAX_FINAL_M = 5e-4

RUN_S = 3.0


def test_a_straight_descent_stays_on_its_line():
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.001, "gravity": [0.0, 0.0, 0.0]},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [
                        {
                            "cartesian_admittance": {
                                "site": "tool_site",
                                "law": "position",
                                "rate_hz": 500.0,
                                "kp": [4.0, 4.0, 4.0, 2.0, 2.0, 2.0],
                            }
                        },
                    ],
                },
            ],
        }
    )
    engine = Engine(cfg)
    try:
        engine.setup()
        engine.reset()
        handle = engine.ctx.blackboard.get("cartesian:ur5e")
        start, mat = handle.read_pose()
        goal = np.array(start) + [0.0, 0.0, GOAL_DZ]
        handle.set_goal(goal.tolist(), mat.reshape(9))

        lateral = 0.0
        for _ in range(int(RUN_S / engine.ctx.model.opt.timestep)):
            engine.step()
            pos, _ = handle.read_pose()
            lateral = max(lateral, float(np.linalg.norm(pos[:2] - start[:2])))
        final = float(np.linalg.norm(handle.read_pose()[0] - goal))
    finally:
        engine.shutdown()

    assert lateral < MAX_LATERAL_M, (
        f"a straight {abs(GOAL_DZ) * 1e3:.0f} mm descent swept {lateral * 1e3:.2f} mm sideways"
    )
    assert final < MAX_FINAL_M, f"the descent ended {final * 1e3:.2f} mm from its goal"
