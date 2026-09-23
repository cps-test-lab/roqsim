"""The ``drive`` output: move a real robot by commanding its own drive plugin.

For an opponent whose *dynamics* matter -- a second robot in the aisle whose wheels really turn, that
slips, that a collision actually stops. The navigator does not touch actuators: it calls the
``drive(vx, vy, w)`` its owner's controller published on the blackboard, which is the very same entry
point the ROS bridge writes ``/cmd_vel`` into. Everything below that is unchanged and stays the drive
plugin's business -- the inverse kinematics, the acceleration ramp, the wheel-encoder odometry. The
only difference from a nav2-driven robot is where the twist came from.

That seam is why this output is ten lines of actual work: every locomotion plugin in the substrate
already publishes one, and none of them needed changing.

The base's geometry comes from the handle's own ``kinematics`` declaration rather than from a list of
robot names here, so a differential base, a mecanum base, a car and an out-of-tree drive are each
shaped correctly without this module knowing they exist.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import RobotHandle
from roqsim.kinematics import body_twist

from ..control import LAWS, yaw_from_quat
from . import NavOutput, OutputUnavailable


class DriveOutput(NavOutput):
    """Commands a body-frame twist through the owning entity's :class:`RobotHandle`."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._handle: RobotHandle | None = None
        self._bid = -1
        self._kinematics = str(self.config.get("kinematics", "auto"))
        self.gain = float(self.config.get("heading_gain", 2.0))
        self.max_w = float(self.config.get("max_angular_vel", 1.5))
        self.turn_in_place = float(self.config.get("turn_in_place", 0.8))
        self.min_speed = float(self.config.get("min_speed", 0.15))
        self.face = str(self.config.get("face", "travel"))

    @property
    def kinematics(self) -> str:
        return self._kinematics

    def attach(self, ctx, entity) -> None:
        handle = ctx.blackboard.get(f"robot:{entity.name}")
        if handle is None:
            raise OutputUnavailable(
                f"entity {entity.name!r} publishes no RobotHandle, so there is nothing to command. "
                f"A drive plugin (diff_drive, omni_drive, ackermann_drive, or a legged locomotion "
                f"controller) must be a component of it -- for a robot model that usually arrives "
                f"from its manifest."
            )
        self._handle = handle
        if self._kinematics == "auto":
            # Declared by the drive, never guessed here: see RobotHandle.kinematics.
            self._kinematics = getattr(handle, "kinematics", "unicycle")
        if self._kinematics not in LAWS:
            raise OutputUnavailable(
                f"entity {entity.name!r} declares kinematics {self._kinematics!r}, which has no "
                f"control law here. Known: {', '.join(sorted(LAWS))}."
            )
        self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if self._bid < 0:
            raise OutputUnavailable(f"body {entity.body!r} not found in the compiled model")

    def emit(self, ctx, pref_vel: np.ndarray, yaw: float, dt: float) -> None:
        law = LAWS[self._kinematics]
        kwargs = {"gain": self.gain, "max_w": self.max_w}
        if self._kinematics == "unicycle":
            kwargs["turn_in_place"] = self.turn_in_place
        elif self._kinematics == "ackermann":
            kwargs["min_speed"] = self.min_speed
        else:
            kwargs["face"] = self.face
        vx, vy, w = law(pref_vel, yaw, **kwargs)
        # No clamping to the platform's own limits: `drive()` already clips to its max_linear_vel /
        # max_angular_vel and owns the wheel acceleration ramp. Doing it here too would put a
        # robot's physical limits in two places, and let a navigator's cap silently override one.
        self._handle.drive(vx, vy, w)

    def pose(self, ctx) -> tuple[float, float, float]:
        """Ground truth from the compiled model, deliberately not ``read_odom``.

        Two independent reasons. It would be the wrong *frame*: ``read_odom`` returns the drive
        plugin's own dead-reckoned integral, zeroed at every reset, while goals and the planner's
        grid are in world coordinates -- and nothing here publishes the map-to-odom transform that
        would bridge them. And it would be the wrong *instrument*: an opponent is apparatus, so its
        trajectory must not become a function of wheel slip, which is a function of contacts, which
        includes contacts with the robot under test. A mover with realistic localisation error is a
        different experiment, and belongs in a plugin that says so.
        """
        data = ctx.data
        x, y = float(data.xpos[self._bid][0]), float(data.xpos[self._bid][1])
        return x, y, yaw_from_quat(data.xquat[self._bid])

    def twist(self, ctx):
        """Ground-truth :class:`~roqsim.kinematics.Twist`, for a caller that wants what the base is
        actually doing -- as opposed to what it was told to do, which is the drive plugin's ramp."""
        return body_twist(ctx.model, ctx.data, self._bid)

    def stop(self, ctx) -> None:
        if self._handle is not None:
            self._handle.drive(0.0, 0.0, 0.0)
