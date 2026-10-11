"""Every package of this workspace declares in its ``package.xml`` the ROS packages it uses.

``rosdep install`` reads that file and nothing else, so a use it does not list is satisfied only by
whatever the base image happens to carry. The detection converters import ``vision_msgs`` lazily, at
the first publish of a detector endpoint, which is where a workspace built from a clean ``rosdep``
would find out: mid-run, with an ImportError on the physics thread. A launch file finds out the same
way, when it imports ``nav2_common`` or names a package whose node or share directory it starts.

What counts as a use:

* an import of a ROS Python package, in a module, a launch file or a test;
* a type the code names as a string (``"std_srvs.srv.Trigger"`` on a handler, a converter or a
  decoder), which is imported through ``importlib`` when it is first resolved;
* a package a launch file names: a node's ``package=`` and a ``get_package_share_directory``.

A test's uses may be declared as ``test_depend``; everything else needs an ``exec_depend`` (or a
``depend``).

The packages are found next to this one, so a package added to the workspace is checked with no
edit here.
"""

from __future__ import annotations

import pathlib
import re

import pytest

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
PACKAGES = sorted(p for p in WORKSPACE.iterdir() if (p / "package.xml").is_file())

ROS_MODULE = re.compile(
    r"^(rclpy|tf2_ros|ament_index_python|launch|launch_ros|nav2_common|nav2_simple_commander"
    r"|[a-z0-9_]+_(?:msgs|srvs|interfaces))$"
)
IMPORT = re.compile(r"^\s*(?:from|import)\s+([a-z0-9_]+)", re.M)
TYPE_STRING = re.compile(r"\b([a-z0-9_]+)\.(?:msg|srv|action)\.[A-Z]")
NAMED_PACKAGE = re.compile(r'(?:\bpackage=|get_package_share_directory\()\s*"([a-z0-9_]+)"')
DECLARED = re.compile(r"<(exec_depend|depend|test_depend)>([^<]+)</")


def _used(package: pathlib.Path, sources) -> set[str]:
    names: set[str] = set()
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for name in [*IMPORT.findall(text), *TYPE_STRING.findall(text)]:
            if ROS_MODULE.match(name):
                names.add(name)
        names.update(NAMED_PACKAGE.findall(text))
    names.discard(package.name)
    return names


def _declared(package: pathlib.Path) -> tuple[set[str], set[str]]:
    """``(run, test)``: what the package may use at run time, and in its tests."""
    entries = DECLARED.findall((package / "package.xml").read_text(encoding="utf-8"))
    run = {name for kind, name in entries if kind != "test_depend"}
    return run, run | {name for kind, name in entries if kind == "test_depend"}


@pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
def test_every_ros_package_used_is_declared(package):
    tests = set((package / "test").rglob("*.py"))
    shipped = set(package.rglob("*.py")) - tests
    run, test = _declared(package)
    undeclared = sorted(_used(package, shipped) - run)
    assert not undeclared, f"{package.name} uses but does not exec_depend on: {undeclared}"
    undeclared = sorted(_used(package, tests) - test)
    assert not undeclared, f"{package.name}'s tests use but do not declare: {undeclared}"


def test_the_workspace_is_found():
    assert "roqsim_ros_bridge" in {p.name for p in PACKAGES}
