"""What every velocity-commanded controller shares: a command watchdog and a spawn-pose odom frame.

A robot commanded by a twist -- a wheeled base, a legged locomotion policy, a quadrotor flown by
velocity -- keeps two promises to the stack driving it, whatever moves it:

* **A command expires.** :class:`CommandWatchdog` is ``cmd_vel_timeout``: a command is good for that
  many seconds of sim time, and then the controller stops the robot. ``0`` (the default) holds a
  command until the next one. The stamp is cleared on reset, so no trial inherits the last one's.
* **Odometry starts at zero where the robot was spawned.** The ``odom`` frame is the pose at reset,
  so a stack sees the same numbers whatever the spawn pose. A wheeled base gets there by integrating
  its wheels from zero; a floating base whose controller reads its pose from the simulator gets there
  through :class:`SpawnFrame`.
"""

from __future__ import annotations

import math

import numpy as np

from .pose import yaw_of


class CommandWatchdog:
    """``cmd_vel_timeout``: the last command is stale once it is older than ``timeout`` seconds."""

    KEY = "cmd_vel_timeout"

    def __init__(self, timeout: float = 0.0):
        self.timeout = float(timeout)
        self.clear()

    @classmethod
    def from_config(cls, config: dict) -> CommandWatchdog:
        return cls(config.get(cls.KEY, 0.0))

    @classmethod
    def validate(cls, config: dict) -> list[str]:
        if float(config.get(cls.KEY, 0.0)) < 0:
            return [f"'{cls.KEY}' must be >= 0 (0 = no watchdog)"]
        return []

    def stamp(self, ctx) -> None:
        """Record a command. ``ctx`` is None before configure, when sim time is 0."""
        self._last = ctx.sim_time if ctx is not None else 0.0

    def clear(self) -> None:
        self._last = -math.inf

    def expired(self, ctx) -> bool:
        return self.timeout > 0.0 and ctx.sim_time - self._last > self.timeout


class SpawnFrame:
    """The ``odom`` frame of a floating base: its pose at reset, horizontally, gravity-aligned.

    The origin is the spawn position in x and y and the axes are turned to the spawn heading; heights
    are not offset, so a standing robot's base keeps its height above the floor and a drone's ``z`` is
    its altitude. Roll and pitch at spawn are ignored, since ``odom`` is a planar frame.
    """

    def __init__(self):
        self._x = self._y = self._yaw = 0.0

    def capture(self, pos, quat) -> None:
        """Take the frame from a world position and ``(w, x, y, z)`` orientation."""
        self._x, self._y = float(pos[0]), float(pos[1])
        self._yaw = yaw_of(quat)

    def position(self, pos) -> tuple[float, float, float]:
        """A world position in this frame."""
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        dx, dy = float(pos[0]) - self._x, float(pos[1]) - self._y
        return (c * dx + s * dy, -s * dx + c * dy, float(pos[2]))

    def yaw(self, quat) -> float:
        """The heading of a world ``(w, x, y, z)`` orientation, in this frame."""
        a = yaw_of(quat) - self._yaw
        return math.atan2(math.sin(a), math.cos(a))

    def world_position(self, pos) -> tuple[float, float, float]:
        """A position in this frame, in the world: the inverse of :meth:`position`."""
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        x, y = float(pos[0]), float(pos[1])
        return (self._x + c * x - s * y, self._y + s * x + c * y, float(pos[2]))

    def world_yaw(self, yaw: float) -> float:
        """A heading in this frame, in the world."""
        a = float(yaw) + self._yaw
        return math.atan2(math.sin(a), math.cos(a))

    def orientation(self, quat) -> tuple[float, float, float, float]:
        """A world ``(w, x, y, z)`` orientation in this frame, tilt kept."""
        h = -0.5 * self._yaw
        cw, cz = math.cos(h), math.sin(h)
        w, x, y, z = (float(v) for v in quat)
        # q_z(-yaw0) * q
        return (cw * w - cz * z, cw * x - cz * y, cw * y + cz * x, cw * z + cz * w)


def planar_odom(frame: SpawnFrame, data, body: int, dof: int) -> tuple[float, ...]:
    """``(x, y, yaw, vx, vy, w, z)`` of a free-floating base in ``frame``, the twist in the body frame.

    ``body`` is the base body and ``dof`` the first dof of its free joint, whose linear velocity is
    world-frame and angular velocity body-frame. The trailing ``z`` is the base height.
    """
    quat = data.xquat[body]
    x, y, z = frame.position(data.xpos[body])
    world_yaw = yaw_of(quat)
    vgx, vgy = data.qvel[dof : dof + 2]
    c, s = np.cos(world_yaw), np.sin(world_yaw)
    return (
        x,
        y,
        frame.yaw(quat),
        float(c * vgx + s * vgy),
        float(-s * vgx + c * vgy),
        float(data.qvel[dof + 5]),
        z,
    )
