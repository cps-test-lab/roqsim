# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the scene-builder tests: a throwaway X display for the GUI ones.

Ported from ``mcp-media-review``'s image-review suite, which is where this pattern already lives --
the two packages share the annotation-window design and, per the note atop ``annotate_ui``, share it
by copying rather than by depending on each other.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def virtual_display():
    """Run the GUI tests on a throwaway Xvfb display.

    Not on the real one: a window has to be MAPPED for its geometry to be measurable, and mapping it
    on the developer's display flashes windows across their screen mid-run. Module-scoped because the
    fixture boots a real X server -- one per test class would start (and leak) several.
    """
    # Tk is the other half of the same prerequisite, and it was the unguarded half: a machine with
    # Xvfb but without python3-tk got past this fixture and died on `import tkinter` inside each test.
    pytest.importorskip("tkinter", reason="python3-tk is not installed; the GUI tests need real Tk")
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb not installed; skipping rather than opening windows on a real display")
    for num in range(90, 100):
        sock = Path(f"/tmp/.X11-unix/X{num}")
        if sock.exists():
            continue
        proc = subprocess.Popen(
            ["Xvfb", f":{num}", "-screen", "0", "1280x1024x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.05)
        else:
            proc.terminate()
            continue
        previous = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = f":{num}"
        try:
            yield f":{num}"
        finally:
            if previous is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = previous
            proc.terminate()
            proc.wait(timeout=5)
        return
    pytest.skip("no free X display number for Xvfb")
