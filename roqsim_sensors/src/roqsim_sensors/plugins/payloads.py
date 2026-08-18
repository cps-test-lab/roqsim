"""Transport-neutral sensor payloads: what an output endpoint's ``read()`` hands to a transport.

These are the substrate's side of a contract with the ROS 2 bridge: ``roqsim_ros_bridge``'s converters
(``registry.py``) read exactly these attribute names off whatever an endpoint returns, and nothing
here imports ROS -- that is what keeps a sensor runnable with no bridge loaded.

**One type per wire format, shared by every sensor that emits it.** A second point-cloud sensor
started by declaring its own class with the same single ``points`` field, which is how a wire format
acquires two definitions that must be kept in step by hand: the field a converter reads is then
documented in whichever producer the reader happened to open. Two things follow from putting them
here instead -- a new sensor emits an existing format by importing it rather than by re-deriving what
the converter needs, and a change to a format is one edit with every producer visible from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LaserScan:
    """One planar sweep. ``sensor_msgs/LaserScan``; ``inf`` in ``ranges`` means "no return"."""

    ranges: np.ndarray  # (N,) float, sweeping angle_min -> angle_max
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float


@dataclass
class PointCloud:
    """A single frame of returns as XYZ points, finite hits only. ``sensor_msgs/PointCloud2``.

    ``points`` is the wire layout already: an (N, 3) float32 array is x,y,z contiguous per point, so
    the bridge copies it with one ``tobytes()`` rather than boxing per-point floats. Producers differ
    in the frame it is expressed in -- a lidar emits its sensor frame, a depth camera the ROS optical
    frame -- which the endpoint's ``frame_id`` hint carries, not this type.
    """

    points: np.ndarray  # (N, 3) float32
