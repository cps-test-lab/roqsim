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

"""The exit statuses every ``roqsim`` tool shares: one table, so a caller branches the same way on all.

A script that runs ``roqsim sim``, ``render``, ``state``, ``check``, ``health`` or an exporter keys on
the status rather than on stderr text, so "the input I named does not exist" has to be the same
number from each of them. Every tool takes its codes from here and states the ones it can return in
its ``--help`` epilog, built by :func:`epilog` so the text cannot drift from the numbers. The table is
documented once, in ``docs/quickstart.rst`` ("Exit status").

``1`` is deliberately not a verdict. It is the status Python gives an uncaught exception, so a tool
never returns it on purpose: seeing it means the tool crashed, and the traceback on stderr says where.

An error class states its own status with an ``exit_status`` class attribute, read by
:func:`for_error`, so a tool (or the command tree around it) maps an exception it caught without a
list of class names here. A class that states none is a bad input, because an error a tool catches
and reports is one it understands, and those are about what it was asked to do.

Nothing is imported here: ``roqsim health`` and the command tree read this table and must stay cheap.
"""

from __future__ import annotations

import sys

#: The tool did what it was asked, and any verdict it gives is a pass.
OK = 0
#: An uncaught exception -- a crash, with a traceback. Never returned on purpose.
CRASH = 1
#: The input or the request is wrong: a file, world, model or recording that does not exist or does
#: not load, a flag or value that cannot be honoured.
BAD_INPUT = 2
#: No GL context: no offscreen backend for a render, or no display for a window.
NO_GL = 3
#: A recording that exists cannot be read, or its world cannot be rebuilt from its provenance.
RECORDING = 4
#: The tool ran and its verdict is a failure: a check that found a problem, an error-level health
#: finding, an export whose round trip disagrees with its source.
FINDING = 5

#: One line per status, in order: the text of the ``--help`` epilogs and of the documented table.
MEANINGS: dict[int, str] = {
    OK: "success",
    CRASH: "an unexpected error (a crash, with its traceback on stderr)",
    BAD_INPUT: "bad input or a wrong request",
    NO_GL: "no GL context (set MUJOCO_GL=egl or osmesa, or give a window a display)",
    RECORDING: "a recording that cannot be read or rebuilt",
    FINDING: "a check or health verdict that failed",
}


def for_error(err: BaseException) -> int:
    """The status for an error a tool caught and reports: its class's ``exit_status``, else
    :data:`BAD_INPUT`."""
    return getattr(type(err), "exit_status", BAD_INPUT)


def fail(prog: str, err: BaseException) -> int:
    """Report `err` as one ``<prog>: <reason>`` line on stderr and return its status."""
    print(f"{prog}: {err}", file=sys.stderr)
    return for_error(err)


def epilog(*codes: int, note: str = "") -> str:
    """A ``--help`` epilog naming :data:`OK` and each of `codes` with its meaning from the table.

    Every epilog also names :data:`CRASH`, since any tool can crash. `note` is appended, for what a
    tool adds about its own output (where the report goes, what a warning does to the status).
    """
    unknown = set(codes) - set(MEANINGS)
    if unknown:
        raise ValueError(f"exit status {sorted(unknown)} is not in roqsim.exit_status.MEANINGS")
    shown = sorted({OK, CRASH, *codes})
    text = "exit status: " + "; ".join(f"{code} {MEANINGS[code]}" for code in shown) + "."
    return f"{text} {note}".strip()
