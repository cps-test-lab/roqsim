"""Writes orca_rvo2.json: rvo2's velocities for test_orca.py's fixed scenes, every agent at every
step, so the port is compared with RVO2 where rvo2 is not installed.

    python roqsim_nav/tests/data/orca_rvo2.py    (with rvo2 and roqsim_nav importable)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from test_orca import SCENES, rvo2_trace  # noqa: E402

if __name__ == "__main__":
    traces = {name: rvo2_trace(name) for name in sorted(SCENES)}
    (HERE / "orca_rvo2.json").write_text(json.dumps(traces, separators=(",", ":")) + "\n")
