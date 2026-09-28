"""The neutral data types endpoints carry, and the unit aliases their fields and parameters use.

A plugin's endpoint returns or takes these, and a bridge maps each one to its transport once -- the
ROS bridge to one message type per type, with converters both ways -- so a plugin states *what* it
publishes and never how a transport spells it::

    from roqsim import endpoint
    from roqsim.types import AngularSpeed, Odometry, Speed, Twist

    class Base(Plugin):
        @endpoint.stream(Twist)
        def cmd_vel(self, vx: Speed, vy: Speed = 0.0, wz: AngularSpeed = 0.0) -> None:
            ...

        @endpoint.out(rate="odom_rate_hz")
        def odom(self) -> Odometry:
            ...

**Unit aliases** are ``typing.Annotated`` types carrying a :class:`~roqsim.endpoint.Unit`, spelled
as a config :class:`~roqsim.schema.Field`'s ``unit`` is: ``Speed`` is ``Annotated[float,
Unit("m/s")]``. The vector aliases are ``numpy`` arrays of a fixed :class:`~roqsim.endpoint.Shape`
(``Point3`` is three ``float64`` in metres; ``Quaternion`` is ``(w, x, y, z)``, MuJoCo's order). A
parameter or a field annotated with one is described with its unit and checked against its shape.

**The types** are dataclasses. Vectors are ``numpy`` arrays and every frame is the one the producer
states in its endpoint's hints unless a field says otherwise. A type is here when two or more
plugins publish or take it; any other dataclass is a payload too (see :mod:`roqsim.endpoint`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from .endpoint import Shape, Unit

__all__ = [
    "Acceleration",
    "Acceleration3",
    "Angle",
    "AngularSpeed",
    "AngularVelocity3",
    "CameraInfo",
    "Degrees",
    "Duration",
    "Force",
    "Force3",
    "Frequency",
    "Image",
    "Imu",
    "JointPositions",
    "JointState",
    "LaserScan",
    "Length",
    "Mass",
    "Odometry",
    "Point3",
    "PointCloud",
    "Pose",
    "Quaternion",
    "Shape",
    "Speed",
    "Torque",
    "Torque3",
    "Twist",
    "Unit",
    "Vector3",
    "Velocity3",
    "Wrench",
]

# -- scalar unit aliases ---------------------------------------------------------------------------
Length = Annotated[float, Unit("m")]
Speed = Annotated[float, Unit("m/s")]
Acceleration = Annotated[float, Unit("m/s^2")]
Angle = Annotated[float, Unit("rad")]
Degrees = Annotated[float, Unit("deg")]
AngularSpeed = Annotated[float, Unit("rad/s")]
Force = Annotated[float, Unit("N")]
Torque = Annotated[float, Unit("N*m")]
Mass = Annotated[float, Unit("kg")]
Duration = Annotated[float, Unit("s")]
Frequency = Annotated[float, Unit("Hz")]

# -- vector aliases --------------------------------------------------------------------------------
Vector3 = Annotated[NDArray[np.float64], Shape(3)]
Point3 = Annotated[NDArray[np.float64], Shape(3), Unit("m")]
Quaternion = Annotated[NDArray[np.float64], Shape(4)]
Velocity3 = Annotated[NDArray[np.float64], Shape(3), Unit("m/s")]
AngularVelocity3 = Annotated[NDArray[np.float64], Shape(3), Unit("rad/s")]
Acceleration3 = Annotated[NDArray[np.float64], Shape(3), Unit("m/s^2")]
Force3 = Annotated[NDArray[np.float64], Shape(3), Unit("N")]
Torque3 = Annotated[NDArray[np.float64], Shape(3), Unit("N*m")]


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _identity() -> np.ndarray:
    return _vec(1.0, 0.0, 0.0, 0.0)


def _zero3() -> np.ndarray:
    return _vec(0.0, 0.0, 0.0)


def yaw_quaternion(yaw: float) -> np.ndarray:
    """The ``(w, x, y, z)`` quaternion of a rotation by *yaw* about +z."""
    return _vec(math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


# -- types -----------------------------------------------------------------------------------------
@dataclass
class Twist:
    """A body-frame velocity.

    Attributes:
        vx: forward speed
        vy: leftward speed
        vz: upward speed
        wx: roll rate
        wy: pitch rate
        wz: yaw rate
    """

    vx: Speed = 0.0
    vy: Speed = 0.0
    vz: Speed = 0.0
    wx: AngularSpeed = 0.0
    wy: AngularSpeed = 0.0
    wz: AngularSpeed = 0.0


@dataclass
class Pose:
    """A position and orientation in a frame.

    Attributes:
        position: origin of the pose
        orientation: quaternion (w, x, y, z)
        frame_id: frame the pose is stated in; empty for the endpoint's own
    """

    position: Point3 = field(default_factory=_zero3)
    orientation: Quaternion = field(default_factory=_identity)
    frame_id: str = ""


@dataclass
class Odometry:
    """A pose in the odometry frame and the body-frame twist.

    Attributes:
        position: body origin, odometry frame
        orientation: quaternion (w, x, y, z), odometry frame
        linear: linear velocity, body frame
        angular: angular velocity, body frame
    """

    position: Point3 = field(default_factory=_zero3)
    orientation: Quaternion = field(default_factory=_identity)
    linear: Velocity3 = field(default_factory=_zero3)
    angular: AngularVelocity3 = field(default_factory=_zero3)

    @classmethod
    def planar(
        cls,
        x: float,
        y: float,
        yaw: float,
        vx: float,
        vy: float = 0.0,
        wz: float = 0.0,
        z: float = 0.0,
    ) -> Odometry:
        """A ground robot's odometry: a pose in the plane, at height *z*, and a planar twist."""
        return cls(_vec(x, y, z), yaw_quaternion(yaw), _vec(vx, vy, 0.0), _vec(0.0, 0.0, wz))


@dataclass
class JointState:
    """Positions and velocities of named joints, and their efforts where known.

    Attributes:
        names: joint names, in the order of every array
        positions: rad for a revolute joint, m for a prismatic one
        velocities: rad/s or m/s
        efforts: N*m or N; ``None`` when not reported
    """

    names: list[str]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    efforts: NDArray[np.float64] | None = None


@dataclass
class JointPositions:
    """Target positions for named joints.

    Attributes:
        names: joint names, in the order of ``positions``
        positions: rad for a revolute joint, m for a prismatic one
    """

    names: list[str]
    positions: NDArray[np.float64]


@dataclass
class Wrench:
    """A force and a torque, in the frame the endpoint states.

    Attributes:
        force: force (x, y, z)
        torque: torque (x, y, z)
    """

    force: Force3 = field(default_factory=_zero3)
    torque: Torque3 = field(default_factory=_zero3)


@dataclass
class Imu:
    """What a strap-down IMU reports at one instant.

    Attributes:
        orientation: quaternion (w, x, y, z), world frame
        angular_velocity: sensor frame
        linear_acceleration: proper acceleration (gravity included), sensor frame
        orientation_valid: false when the device reports no attitude
        orientation_variance: per axis
        angular_velocity_variance: per axis, (rad/s)^2
        linear_acceleration_variance: per axis, (m/s^2)^2
    """

    orientation: Quaternion = field(default_factory=_identity)
    angular_velocity: AngularVelocity3 = field(default_factory=_zero3)
    linear_acceleration: Acceleration3 = field(default_factory=_zero3)
    orientation_valid: bool = True
    orientation_variance: float = 0.0
    angular_velocity_variance: float = 0.0
    linear_acceleration_variance: float = 0.0


@dataclass
class LaserScan:
    """One planar sweep.

    Attributes:
        ranges: one range per ray from angle_min to angle_max; inf is no return
        angle_min: angle of the first ray
        angle_max: angle of the last ray
        angle_increment: angle between rays
        range_min: shortest range reported
        range_max: longest range reported
    """

    ranges: Annotated[NDArray[np.float32], Shape(None), Unit("m")]
    angle_min: Angle
    angle_max: Angle
    angle_increment: Angle
    range_min: Length
    range_max: Length


@dataclass
class Image:
    """A rendered frame.

    Attributes:
        data: (height, width) or (height, width, channels) pixels
        encoding: the pixel encoding, as ROS spells it (rgb8, mono8, 16UC1, 32FC1)
    """

    data: NDArray
    encoding: str = "rgb8"


@dataclass
class CameraInfo:
    """Pinhole intrinsics of a camera.

    Attributes:
        width: pixels
        height: pixels
        fx: focal length, pixels
        fy: focal length, pixels
        cx: principal point, pixels
        cy: principal point, pixels
        d: plumb_bob distortion coefficients
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    d: list[float] = field(default_factory=lambda: [0.0] * 5)


@dataclass
class PointCloud:
    """One frame of returns as points in the sensor's frame.

    Attributes:
        points: (N, 3) x, y, z of each finite return
    """

    points: Annotated[NDArray[np.float32], Shape(None, 3), Unit("m")]
