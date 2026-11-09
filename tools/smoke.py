#!/usr/bin/env python3
"""Headless-run every shipped world for a fixed number of steps -- ``make smoke``.

A world passes when ``roqsim sim`` exits ``0`` on it. It compiles the MJCF, loads every plugin and
resolves every mesh and texture, which is what smoke is for.

A world with a scene (a Python scenario or an OpenSCENARIO ``scenario:`` entry) must also finish
that scene within the run: ``roqsim sim`` exits ``6`` when it stops before a scene has ended, and
smoke reports that world as a scene that did not finish, so a passing smoke still means every
shipped scene runs to its end. A scene that needs longer than the default budget says so in its
world file, with a comment line the world loader ignores::

    # smoke-steps: 3000

That world then runs for that many steps instead of the default. It is a comment, not a world key,
because a world document refuses keys it does not know, and the budget is smoke's alone.

Two failures are reported as skipped rather than failed: a world that needs an asset that is
generated rather than committed (named in ``.gitignore``, built by ``make external-resources``),
and one whose renderer draws on a CUDA GPU on a machine that has none.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: The exit status of ``roqsim sim`` stopping before a scene of its world ended
#: (``roqsim.exit_status.UNFINISHED``). A literal, so smoke runs without importing roqsim.
UNFINISHED = 6

#: A world's own step budget: a whole line ``# smoke-steps: <n>``.
STEPS_LINE = re.compile(r"^#\s*smoke-steps\s*:(.*)$", re.MULTILINE)

MISSING_FILE = re.compile(r"Error opening file '([^']*)'")
NEEDS_CUDA = "draws on a CUDA GPU"
#: The lines of a failed world's output smoke prints.
LOG_TAIL = 20


def world_steps(world: Path, default: int) -> int:
    """The steps ``world`` runs for: its ``# smoke-steps:`` line, else ``default``."""
    lines = STEPS_LINE.findall(world.read_text())
    if not lines:
        return default
    if len(lines) > 1:
        raise ValueError(f"{world}: more than one '# smoke-steps:' line")
    value = lines[0].strip()
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"{world}: '# smoke-steps: {value}' is not a positive whole number")
    return int(value)


def has_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    return (
        nvidia_smi is not None and subprocess.run([nvidia_smi], capture_output=True).returncode == 0
    )


def run(roqsim: str, world: Path, steps: int, extra: list[str]) -> tuple[int, str]:
    argv = [roqsim, "sim", str(world), "--headless", "--pacing", "asap", "--steps", str(steps)]
    done = subprocess.run(
        [*argv, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return done.returncode, done.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roqsim", required=True, help="the roqsim command to run worlds with")
    parser.add_argument("--steps", type=int, required=True, help="default steps per world")
    parser.add_argument("--gitignore", type=Path, default=Path(".gitignore"))
    parser.add_argument(
        "world", nargs="*", type=Path, help="package worlds, run as those of --worlds are"
    )
    parser.add_argument(
        "--worlds",
        nargs="*",
        default=[],
        type=Path,
        help="package worlds, always run with --no-communication",
    )
    parser.add_argument(
        "--ros-worlds",
        nargs="*",
        default=[],
        type=Path,
        help="ros2_ws worlds, run as authored only with --ros",
    )
    parser.add_argument("--ros", action="store_true", help="ROS is sourced and ros2_ws built")
    args = parser.parse_args(argv)
    args.worlds = [*args.world, *args.worlds]

    ignored = args.gitignore.read_text() if args.gitignore.is_file() else ""
    mute = [] if args.ros else ["--no-communication"]
    groups = [(w, ["--no-communication"]) for w in args.worlds]
    groups += [(w, mute) for w in args.ros_worlds]

    failed, unfinished, skipped, nogpu = [], [], [], []
    for i, (world, extra) in enumerate(groups):
        if i == len(args.worlds):
            how = (
                "as authored (ROS sourced)"
                if args.ros
                else "with --no-communication (ROS not sourced)"
            )
            print(f"-- ros2_ws worlds {how}")
        print(f"== {world}", flush=True)
        try:
            steps = world_steps(world, args.steps)
        except ValueError as err:
            print(err)
            failed.append(world)
            continue
        status, log = run(args.roqsim, world, steps, extra)
        if status == 0:
            continue
        missing = MISSING_FILE.search(log)
        if status == UNFINISHED:
            unfinished.append((world, steps))
        elif missing and Path(missing.group(1)).name in ignored:
            skipped.append(world)
        elif NEEDS_CUDA in log and not has_gpu():
            nogpu.append(world)
        else:
            print("".join(log.splitlines(keepends=True)[-LOG_TAIL:]), end="")
            failed.append(world)

    if skipped:
        print(
            "\nSKIPPED -- needs an asset that is generated, not committed "
            "(external/external_assets.yaml):"
        )
        for w in skipped:
            print(f"  {w}")
    if nogpu:
        print("\nSKIPPED -- its renderer draws on a CUDA GPU, and this machine has none:")
        for w in nogpu:
            print(f"  {w}")
    if unfinished:
        print("\nUNFINISHED -- its scene did not finish within the steps it ran for:")
        for w, steps in unfinished:
            print(f"  {w}: its scene did not finish within {steps} steps")
        print(
            "A shipped scene must end within its smoke run: shorten the scene, or give the world "
            "a line '# smoke-steps: <n>' with the steps it needs."
        )
    if failed:
        print(f"\nFAILED ({len(groups)} worlds run):")
        for w in failed:
            print(f"  {w}")
    if failed or unfinished:
        return 1
    print(
        f"\n{len(groups)} worlds run headless for {args.steps} steps, or their own "
        "'# smoke-steps:'; none failed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
