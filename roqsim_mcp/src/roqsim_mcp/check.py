# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""The ``check_world`` MCP tool: ``roqsim check --json``, for a client that has no shell.

An agent that writes a world through MCP alone must be able to ask whether it loads before anything
runs it; the answer already exists as ``roqsim check``, so this is that command and nothing more.

It runs the command in a **subprocess** rather than calling :func:`roqsim.check.check_world` in this
server's process, unlike the catalog tools beside it, which only read registries. A check compiles
the model and runs every plugin's ``build``, ``configure`` and ``on_reset``: anything one of them
prints lands on this server's stdout, which on the stdio transport IS the protocol stream, and a
MuJoCo compile that aborts would take the long-lived server down with it. The subprocess keeps both
out, and the command stays the one implementation.
"""

from __future__ import annotations

import json
import subprocess
import sys


def check_world(world: str) -> dict:
    """Load a world as far as it goes and report every problem at once, as ``roqsim check --json``.

    Use it after writing or editing a world, before running it. Nothing is stepped.

    Args:
        world: a world YAML path, or a ``<package>:<world>`` ref -- what ``roqsim check`` takes.

    Returns:
        ``roqsim check``'s report: ``{"target", "ok", "reached", "problems", "warnings", "inputs",
        "world", "derived"}``. ``ok`` is false when any stage found a problem, each problem naming its
        ``stage``, ``message`` and ``hint``; ``warnings`` never change ``ok``.

    Raises:
        ValueError: if ``world`` is empty.
        RuntimeError: if the check itself could not run -- the message is the command's own.
    """
    if not world or not world.strip():
        raise ValueError("world: give a world YAML path or a '<package>:<world>' ref")
    proc = subprocess.run(  # noqa: S603 - argv, no shell
        [sys.executable, "-m", "roqsim.check", world, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    # 0 and 1 are both a verdict with its report on stdout; anything else is a check that did not run.
    if proc.returncode in (0, 1):
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    raise RuntimeError(detail[-1] if detail else f"roqsim check failed (exit {proc.returncode})")
