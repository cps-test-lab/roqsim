"""Reading the vendored Franka files (``upstream/franka``, pinned in ``SOURCES.toml``) for the
``test_franka*`` tests: never a value typed from them, always the file."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import yaml

UPSTREAM = Path(__file__).resolve().parent / "upstream" / "franka"
LIBFRANKA = UPSTREAM / "libfranka"
DESCRIPTION = UPSTREAM / "franka_description"
TOOL = Path(__file__).resolve().parents[2] / "external" / "franka_upstream.py"


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
