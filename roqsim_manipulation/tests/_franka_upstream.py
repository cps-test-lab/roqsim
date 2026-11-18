"""Reading the vendored Franka files (``upstream/franka``, pinned in ``SOURCES.toml``) for the
``test_franka*`` tests: never a value typed from them, always the file."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

import yaml

UPSTREAM = Path(__file__).resolve().parent / "upstream" / "franka"
LIBFRANKA = UPSTREAM / "libfranka"
COMMON = UPSTREAM / "libfranka-common" / "include" / "research_interface" / "robot"
DESCRIPTION = UPSTREAM / "franka_description"
ROS2 = UPSTREAM / "franka_ros2"
TOOL = Path(__file__).resolve().parents[2] / "external" / "franka_upstream.py"


def sources() -> dict:
    return tomllib.loads((UPSTREAM / "SOURCES.toml").read_text(encoding="utf-8"))


def upstream_tool() -> ModuleType:
    """The checkout's ``external/franka_upstream.py``, which no installed package carries."""
    spec = importlib.util.spec_from_file_location("franka_upstream", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def struct_body(src: str, name: str) -> str:
    """The body of ``struct <name> { ... };`` (braces balanced)."""
    m = re.search(r"struct\s+" + re.escape(name) + r"\b[^{;]*\{", src)
    if m is None:
        raise KeyError(f"struct {name} not found")
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[m.end() : i - 1]


def enum_values(src: str, name: str) -> list[str]:
    m = re.search(r"enum\s+class\s+" + re.escape(name) + r"\b[^{]*\{([^}]*)\}", src)
    if m is None:
        raise KeyError(f"enum {name} not found")
    return [v.strip().split("=")[0].strip() for v in m.group(1).split(",") if v.strip()]


#: C++ member types -> (struct code, count factor)
_TYPES = {
    "uint64_t": ("Q", 1),
    "float": ("f", 1),
    "double": ("d", 1),
    "bool": ("?", 1),
    "uint8_t": ("B", 1),
    "MotionGeneratorMode": ("B", 1),
    "ControllerMode": ("B", 1),
    "RobotMode": ("B", 1),
}


def data_members(body: str) -> list[tuple[str, str, int]]:
    """``(name, struct code, count)`` of a packed struct's data members, in order (nested classes,
    functions and constructors skipped)."""
    body = strip_comments(body)
    # drop nested class/struct definitions and function bodies
    out_body, depth, i = "", 0, 0
    while i < len(body):
        ch = body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out_body += ";"
        elif depth == 0:
            out_body += ch
        i += 1
    members = []
    for decl in out_body.split(";"):
        decl = " ".join(decl.split())
        if (
            not decl
            or "(" in decl
            or decl.startswith(("template", "public", "class", "using", "enum"))
        ):
            continue
        decl = decl.replace("public:", "").replace("const ", "").strip()
        m = re.match(
            r"(?:std::)?array<\s*(?:std::)?array<\s*(\w+)\s*,\s*(\d+)\s*>\s*,\s*(\d+)\s*>\s+(\w+)",
            decl,
        )
        if m:
            code = _TYPES[m.group(1)][0]
            members.append((m.group(4), code, int(m.group(2)) * int(m.group(3))))
            continue
        m = re.match(r"(?:std::)?array<\s*(\w+)\s*,\s*(\d+)\s*>\s+(\w+)", decl)
        if m:
            members.append((m.group(3), _TYPES[m.group(1)][0], int(m.group(2))))
            continue
        m = re.match(r"floatarray<\s*(\d+)\s*>\s+(\w+)", decl)
        if m:
            members.append((m.group(2), "f", int(m.group(1))))
            continue
        m = re.match(r"(\w+)\s+(\w+)(?:\{\})?$", decl)
        if m and m.group(1) in _TYPES:
            members.append((m.group(2), _TYPES[m.group(1)][0], 1))
    return members


def joint_limits() -> dict:
    return yaml.safe_load(text(DESCRIPTION / "robots/fr3/joint_limits.yaml"))


def constexpr(src: str, name: str) -> str:
    m = re.search(r"constexpr[^;=]*\b" + re.escape(name) + r"\b\s*(?:=|\{)(.*?);", src, flags=re.S)
    if m is None:
        raise KeyError(name)
    return m.group(1)


def eval_constexpr(src: str, name: str, env: dict) -> list[float] | float:
    """A ``constexpr`` double or ``std::array<double, N>`` of ``rate_limiting.h``, evaluated."""
    expr = constexpr(src, name).strip()
    expr = expr.strip("{}").strip()
    expr = expr.replace("std::numeric_limits<double>::epsilon()", "2.220446049250313e-16")
    expr = re.sub(r"(\w+)\[(\d+)\]", r"\1[\2]", expr)
    parts = [p for p in re.split(r",(?![^\[]*\])", expr) if p.strip()]
    values = [float(eval(p.strip().strip("{}"), {}, env)) for p in parts]  # noqa: S307 - pinned file
    return values if len(values) > 1 else values[0]
