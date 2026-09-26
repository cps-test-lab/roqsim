"""Every ROS package the bridge imports is declared in its ``package.xml``.

``rosdep install`` reads that file and nothing else, so an import it does not list is satisfied only
by whatever the base image happens to carry. The detection converters import ``vision_msgs`` lazily,
at the first publish of a detector endpoint, which is where a workspace built from a clean
``rosdep`` would find out: mid-run, with an ImportError on the physics thread.
"""

from __future__ import annotations

import pathlib
import re

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
ROS_MODULE = re.compile(r"^(rclpy|tf2_ros|[a-z0-9_]+_(?:msgs|srvs|interfaces))$")
IMPORT = re.compile(r"^\s*(?:from|import)\s+([a-z0-9_]+)", re.M)
DECLARED = re.compile(r"<(?:exec_depend|depend)>([^<]+)</")


def _imported() -> set[str]:
    names: set[str] = set()
    for source in (PACKAGE / "roqsim_ros_bridge").rglob("*.py"):
        for match in IMPORT.finditer(source.read_text(encoding="utf-8")):
            if ROS_MODULE.match(match.group(1)):
                names.add(match.group(1))
    return names


def _declared() -> set[str]:
    return set(DECLARED.findall((PACKAGE / "package.xml").read_text(encoding="utf-8")))


def test_every_imported_ros_package_is_declared():
    undeclared = sorted(_imported() - _declared())
    assert not undeclared, f"imported but not in package.xml: {undeclared}"
