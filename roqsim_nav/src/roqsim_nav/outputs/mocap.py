"""The ``mocap`` output: move a body by writing its pose.

For anything with no degrees of freedom that still has to go somewhere -- a pallet, a cart, a crate
that crosses an aisle. The body is a MuJoCo mocap body (``spawn_model: {mocap: true}``), so it costs
the solver nothing and nothing can push it off course, while remaining collision geometry the robot
under test perceives and collides with. That combination is exactly what a *controlled* obstacle is:
present to the robot, immovable by it.

The position is integrated from the preferred velocity and the heading is turned toward travel at a
bounded rate, because snapping a body's yaw looks like a glitch and, more importantly, makes its
footprint jump discontinuously for anything sensing it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..control import approach_angle, yaw_from_quat, yaw_to_quat
from . import NavOutput, OutputUnavailable


class MocapOutput(NavOutput):
    """Writes ``mocap_pos``/``mocap_quat`` for the owning entity's body."""

    kinematics = "holonomic"  # a written pose can move any direction; nothing is being steered

    def __init__(self, config: dict):
        super().__init__(config)
        #: rad/s the body re-faces its direction of travel at. 0 means snap.
        self.yaw_rate = float(self.config.get("yaw_rate", 3.0))
        self._mocapid = -1

    def attach(self, ctx, entity) -> None:
        if not entity.body:
            raise OutputUnavailable(f"entity {entity.name!r} has no body to move")
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if bid < 0:
            raise OutputUnavailable(f"body {entity.body!r} not found in the compiled model")
        self._mocapid = int(ctx.model.body_mocapid[bid])
        if self._mocapid < 0:
            raise OutputUnavailable(
                f"body {entity.body!r} of entity {entity.name!r} is not a mocap body, so its pose "
                f"cannot be written. Spawn it with `spawn_model: {{mocap: true}}`."
            )

    def emit(self, ctx, pref_vel: np.ndarray, yaw: float, dt: float) -> None:
        data = ctx.data
        pos = data.mocap_pos[self._mocapid]
        speed = float(math.hypot(pref_vel[0], pref_vel[1]))
        # Keep z: the prop owns its own height (its spawn pose put it there), and re-deriving it
        # here would silently disagree with a `pos: [x, y, z]` a world wrote.
        data.mocap_pos[self._mocapid] = [
            pos[0] + float(pref_vel[0]) * dt,
            pos[1] + float(pref_vel[1]) * dt,
            pos[2],
        ]
        if speed > 1e-3:
            target = math.atan2(float(pref_vel[1]), float(pref_vel[0]))
            step = self.yaw_rate * dt if self.yaw_rate > 0.0 else math.inf
            yaw = approach_angle(yaw, target, step)
        data.mocap_quat[self._mocapid] = yaw_to_quat(yaw)

    def pose(self, ctx) -> tuple[float, float, float]:
        pos = ctx.data.mocap_pos[self._mocapid]
        return float(pos[0]), float(pos[1]), yaw_from_quat(ctx.data.mocap_quat[self._mocapid])

    def stop(self, ctx) -> None:
        """Nothing to do: a pose-written body is already at rest whenever it is not being written."""
