# SPDX-License-Identifier: Apache-2.0
"""Run a scene-builder window as a subprocess and read back its result JSON.

Every annotation window (3D scene review, 2D floorplan sketch) is opened out-of-process via the
``roqsim-scene-builder`` CLI, because tkinter owns the main thread and cannot run inside the FastMCP
worker thread; keeping the GUI out-of-process also means a rendering crash can never take the MCP
server down. The window writes its verdict/sketch JSON to a temp file which this helper reads back.

Both MCP tool modules (:mod:`scene_review`, :mod:`floorplan_sketch`) share this one launcher so the
tempfile/timeout/no-result handling lives in exactly one place.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_S = 600.0


def run_window_subprocess(
    subcommand: str,
    extra_args: list[str],
    timeout_s: float | None = None,
    target: str | None = None,
) -> dict:
    """Spawn ``roqsim-scene-builder <subcommand> [target] <extra_args> --json-out <tmp>`` and return it.

    Args:
        subcommand: the CLI subcommand that opens the window (``review-scene`` / ``sketch-floorplan``).
        extra_args: subcommand-specific options (``--message``, ``--initial-json`` …). ``--json-out`` is
            appended here and must not be included by the caller.
        timeout_s: seconds to wait for a result before raising ``TimeoutError`` (default 600).
        target: the subcommand's positional argument (a scene ref) when it takes one; ``None`` for a
            window that has no target (the floorplan sketch). Also used as the error label.

    Returns:
        The dict the window wrote (its result JSON).

    Raises:
        TimeoutError: if no result arrives within the timeout.
        RuntimeError: if the window closed with no verdict/result or failed to start.
    """
    wait_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    label = target or subcommand

    with tempfile.TemporaryDirectory(prefix="scene-builder-") as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, "-m", "roqsim_scene_builder.cli", subcommand]
        if target is not None:
            cmd.append(target)
        cmd += [*extra_args, "--json-out", str(out)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            _, stderr = proc.communicate(timeout=wait_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            raise TimeoutError(f"No human result within {wait_s:.0f}s for {label}") from exc

        if not out.is_file():
            tail = (stderr or "").strip().splitlines()[-5:]
            detail = "\n".join(tail)
            raise RuntimeError(
                f"{subcommand} produced no verdict/result for {label} "
                f"(window exit {proc.returncode}).\n{detail}"
            )
        return json.loads(out.read_text(encoding="utf-8"))
