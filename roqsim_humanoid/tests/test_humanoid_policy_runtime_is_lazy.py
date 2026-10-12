"""Importing a locomotion plugin loads no inference runtime.

``roqsim plugins list`` and ``describe`` import every registered plugin class, so a runtime imported
with the module would be loaded by every tool that only lists what is installed.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_locomotion_plugins_loads_neither_torch_nor_onnxruntime():
    code = (
        "import sys, roqsim\n"
        "import roqsim_humanoid.plugins.g1_locomotion, roqsim_humanoid.plugins.oli_locomotion\n"
        "print(sorted(m for m in ('torch', 'onnxruntime') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
