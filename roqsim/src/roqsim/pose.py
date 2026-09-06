"""Reading a spawn pose written the way ``simulation_interfaces`` states one.

A world's ``pose:`` key carries a ``geometry_msgs/PoseStamped``, the type ``SpawnEntity.srv``
gives its ``initial_pose`` -- ``position`` as x/y/z metres, ``orientation`` as an x/y/z/w
quaternion, under an optional ``header.frame_id``::

    pose:
      position:    {x: 1.5, y: 2.0}
      orientation: {x: 0.0, y: 0.0, z: 0.383, w: 0.924}

That shape rather than a friendlier one because the same pose reaches an entity two ways -- a world
declaring where it starts, and a ``SpawnEntity`` call placing it during a trial -- and the two must
be the same pose written the same way. A world spelling orientation as a yaw and the service
spelling it as a quaternion means every producer of poses needs a conversion whose direction depends
on which door it came in at, and a value that reads correctly in the wrong one is the failure that
follows: ``orientation: {yaw: 1.57}`` is a *valid-looking* quaternion message with x, y, z and w all
defaulted to zero, so it would arrive as a rotation nobody asked for rather than as an error. It is
rejected by name here for exactly that reason.

Two deliberate departures, both because a document can express something a message cannot:

``header.frame_id``
    optional, and only the world frame is accepted -- the same answer the bridge's spawn service
    gives, since this simulator knows no other frame to state a pose in.
``position.z``
    optional, and omitting it means *unstated* rather than zero. The service has no way to say that
    (a default-constructed request carries the origin), but a wheeled robot authored with its base
    at the origin and its wheels below it needs the model's own resting height, and z=0 buries it by
    a wheel radius. A caller that means the floor says ``z: 0.0`` and gets it.
"""

from __future__ import annotations

import math

#: Frame names that mean the world frame. Empty is the message's own default for it; ``world`` is
#: spelled out because that is the name the service description gives that frame.
_WORLD_FRAMES = ("", "world")

#: Below this a quaternion has no direction to normalise, so it is not a rotation.
_MIN_QUAT_NORM = 1e-9


class PoseError(ValueError):
    """A ``pose:`` value that is not a pose. Carries the reason, for a plugin's validator."""


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseError(f"{where} must be a number, got {value!r}")
    return float(value)


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise PoseError(f"{where} must be a mapping, got {value!r}")
    return value


def parse_pose(value) -> tuple[list[float | None], list[float]]:
    """``(position, quaternion)`` from a ``geometry_msgs/PoseStamped``-shaped mapping.

    *position* is ``[x, y, z]`` with ``z`` **None** when the document did not state one, which is
    what tells a caller to use the model's own resting height. *quaternion* is ``[w, x, y, z]``,
    normalised, in MuJoCo's order rather than the message's.

    A ``PoseStamped`` (``header``/``pose``) and the bare ``Pose`` inside it are both accepted, so
    the value can be pasted from either level of the service request.
    """
    outer = _mapping(value, "'pose'")
    if "pose" in outer or "header" in outer:
        frame = str(_mapping(outer.get("header", {}), "'pose.header'").get("frame_id", "") or "")
        if frame not in _WORLD_FRAMES:
            raise PoseError(
                f"'pose.header.frame_id' is {frame!r}, which this simulator does not know. It "
                "places entities in the world frame, which the empty default already names."
            )
        outer = _mapping(outer.get("pose", {}), "'pose.pose'")

    unknown = set(outer) - {"position", "orientation"}
    if unknown:
        raise PoseError(
            f"'pose' has no key(s) {sorted(unknown)!r}. It is a geometry_msgs/PoseStamped, as "
            "SpawnEntity states one: 'position' (x/y/z) and 'orientation' (x/y/z/w), optionally "
            "under 'header'/'pose'."
        )

    position = _mapping(outer.get("position", {}), "'pose.position'")
    if extra := set(position) - {"x", "y", "z"}:
        raise PoseError(f"'pose.position' has no key(s) {sorted(extra)!r}; it is x, y and z")
    for axis in ("x", "y"):
        if axis not in position:
            raise PoseError(f"'pose.position.{axis}' is required")
    pos = [
        _number(position["x"], "'pose.position.x'"),
        _number(position["y"], "'pose.position.y'"),
        _number(position["z"], "'pose.position.z'") if "z" in position else None,
    ]

    return pos, _parse_orientation(outer.get("orientation"))


def _parse_orientation(value) -> list[float]:
    """``[w, x, y, z]``, normalised. Absent means the identity, as the message's defaults do."""
    if value is None:
        return [1.0, 0.0, 0.0, 0.0]
    orientation = _mapping(value, "'pose.orientation'")
    if extra := set(orientation) - {"x", "y", "z", "w"}:
        # Named rather than lumped in with any other stray key: a yaw is what a reader reaches for,
        # and every component of the quaternion it is not defaults to zero -- so accepting it would
        # apply a rotation of (0, 0, 0, 0) instead of the heading that was written.
        if "yaw" in extra:
            raise PoseError(
                "'pose.orientation' is a quaternion (x/y/z/w), not a yaw -- this is the pose "
                "SpawnEntity states. For a heading about z, write "
                "{x: 0, y: 0, z: sin(yaw/2), w: cos(yaw/2)}, or use the 'yaw' key beside 'pos'."
            )
        raise PoseError(f"'pose.orientation' has no key(s) {sorted(extra)!r}; it is x, y, z and w")

    quat = [
        _number(orientation.get(key, 0.0), f"'pose.orientation.{key}'")
        for key in ("w", "x", "y", "z")
    ]
    norm = math.sqrt(sum(c * c for c in quat))
    if norm < _MIN_QUAT_NORM:
        raise PoseError(
            "'pose.orientation' is a zero-length quaternion, which is not a rotation. "
            "The identity is {x: 0, y: 0, z: 0, w: 1}."
        )
    # Normalised because MuJoCo reads a free joint's quaternion as a unit one; a near-unit value
    # would otherwise scale the body's orientation.
    return [c / norm for c in quat]


def yaw_of(quat) -> float:
    """The heading of ``[w, x, y, z]``: the rotation about z, in radians.

    For a pose that is a heading this is exact. For one that is not, it is the z component of the
    rotation and the rest is dropped -- so a caller that can only apply a heading must say so rather
    than call this and assume the pose was flat.
    """
    w, x, y, z = (float(c) for c in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
