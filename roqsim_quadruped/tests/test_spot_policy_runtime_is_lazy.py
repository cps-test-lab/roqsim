"""Importing ``spot_locomotion`` loads no inference runtime.

``roqsim plugins list`` and ``describe`` import every registered plugin class, so a runtime imported
with the module would be loaded by every tool that only lists what is installed.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_spot_locomotion_does_not_load_torch():
    code = (
        "import sys, roqsim\n"
        "import roqsim_quadruped.plugins.spot_locomotion\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout
