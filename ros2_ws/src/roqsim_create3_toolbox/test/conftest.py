"""Make the launch files importable as modules for the launch-description test."""

import importlib.util
import sys
import types
from pathlib import Path

_LAUNCH = Path(__file__).resolve().parents[1] / "launch"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _LAUNCH / f"{name}.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pkg = types.ModuleType("roqsim_create3_toolbox_launch")
try:
    pkg.create3_nodes = _load("create3_nodes")
    pkg.turtlebot4_nodes = _load("turtlebot4_nodes")
    sys.modules["roqsim_create3_toolbox_launch"] = pkg
except Exception:  # no ROS overlay: the launch test skips on its own import
    pass
