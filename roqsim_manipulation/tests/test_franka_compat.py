"""The pinned releases are a row of Franka's own compatibility tables.

libfranka's ``docs/compatibility_matrix.rst`` gives, per libfranka version, the robot system version
and the robot server version it needs; franka_ros2's gives, per franka_ros2 release, the libfranka and
franka_description it needs. The pins in ``SOURCES.toml`` must satisfy both, and the simulated
server must speak the robot server version that libfranka row names.
"""

from __future__ import annotations

import re

from _franka_upstream import LIBFRANKA, ROS2, sources, text
from roqsim_manipulation.franka.protocol import VERSION


def _v(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", s))


def _pins() -> dict[str, str]:
    return {r["dir"]: r["tag"] for r in sources()["repo"]}


def _requirement_rows(path) -> list[list[str]]:
    """The ``list-table`` rows whose cells read ``>= x.y.z`` (or ``>=vX.Y.Z``), minima only."""
    rows: list[list[str]] = []
    for line in text(path).splitlines():
        m = re.match(r"^\s*(\*\s+)?-\s+(.*)$", line)
        if not m:
            continue
        if m.group(1):
            rows.append([])
        if rows:
            rows[-1].append(m.group(2).strip())
    out = []
    for row in rows:
        cells = []
        for cell in row:
            v = re.search(r">=\s*v?(\d+(?:\.\d+)*)", cell)
            cells.append(v.group(1) if v else cell)
        if cells and re.match(r"\d", cells[0]) and ">=" in row[0]:
            out.append(cells)
    return out


def test_libfranka_pin_and_server_version():
    pin = _v(_pins()["libfranka"])
    rows = _requirement_rows(LIBFRANKA / "docs/compatibility_matrix.rst")
    # the first row whose libfranka minimum the pin meets is the one that applies
    row = next(r for r in rows if len(r) >= 3 and _v(r[0]) <= pin)
    assert int(re.match(r"\d+", row[2]).group()) == VERSION, (
        row
    )  # the robot server version the simulated FCI speaks
    assert _v(row[1]) == (5, 9, 0)  # the FR3 system version this pin needs


def test_franka_ros2_pin_needs_the_pinned_libfranka_and_description():
    pins = _pins()
    rows = _requirement_rows(ROS2 / "franka_ros2/doc/compatibility_matrix.rst")
    ros2 = _v(pins["franka_ros2"])
    row = next(r for r in rows if len(r) >= 3 and _v(r[0]) <= ros2)
    assert _v(pins["libfranka"]) >= _v(row[1])
    assert _v(pins["franka_description"]) >= _v(row[2])
