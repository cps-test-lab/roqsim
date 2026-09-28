"""Which frame ids the bridge namespaces. Free of ROS imports, so the rule is testable without rclpy."""

from __future__ import annotations

#: Frames every robot shares: the simulator's world and the map robots are localised in. A robot's
#: own frames, ``odom`` among them, are per robot and get its namespace.
GLOBAL_FRAMES = frozenset({"world", "map"})


def namespaced(prefix: str, name: str) -> str:
    """Prefix a robot's frame id with the bridge namespace so multi-robot TF trees stay unique.

    A global frame is returned bare: ``<ns>/world`` is a frame nothing publishes.
    """
    return f"{prefix}/{name}" if prefix and name not in GLOBAL_FRAMES else name
