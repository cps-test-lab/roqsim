"""List every file a world is defined by, as one line of JSON.

A world is never one file. It is the YAML, whatever it ``extends``, the MJCF that chain
settles on, and the meshes and textures that MJCF names -- all referenced by paths
*relative to each other*. Anything that has to move a world somewhere else, or decide
whether something compiled from it is stale, needs the whole set::

    roqsim scenes inputs worlds/depot.yaml
    {"world": "/abs/worlds/depot.yaml", "inputs": ["/abs/worlds/depot.yaml", ...]}

Why a command and not an import: the caller is often *not* a roqsim process. A campaign
runner staging a world into a container has no reason to have roqsim installed --
knowing which files travel would be its only use for it -- so it asks the image that does.
One line of JSON on stdout is the same machine contract ``roqsim render`` uses.

Paths are absolute and every one exists: the answer is what to copy, not what to check.
A world that cannot be fully resolved yields what *was* resolved rather than failing, so a
caller about to report a different error is not pre-empted by this one -- pass
``--require-complete`` when an incomplete answer would be worse than no answer, which is
the case when the files are about to be shipped somewhere the originals are unreachable.
Then every part of the walk that gave up -- an ``extends`` that does not resolve, a world
that does not load, a plugin that cannot be asked -- is a failure, named on stderr, rather
than a line missing from a list that looks exactly like a complete one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roqsim.config import world_sources
from roqsim.world import resolve_world_yaml_ref


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim scenes inputs",
        description="List every file a world is defined by, as JSON on stdout.",
    )
    parser.add_argument(
        "world", help="world YAML path, or a package ref such as 'roqsim_scenes:depot'"
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail if any part of the world did not resolve, instead of listing what did",
    )
    args = parser.parse_args(argv)

    target, packaged = args.world, False
    if ":" in target and not Path(target).exists():
        # A package ref, resolved by the same helper `roqsim sim` uses, so both accept
        # exactly the same spellings.
        try:
            resolved = resolve_world_yaml_ref(target)
        except FileNotFoundError as err:
            print(f"cannot resolve world {target!r}: {err}", file=sys.stderr)
            return 1
        if resolved is None:
            print(
                f"{target!r} is not a world ref (no such roqsim.worlds provider)", file=sys.stderr
            )
            return 1
        target, packaged = str(resolved), True
    if not Path(target).exists():
        print(f"world {target!r} does not exist", file=sys.stderr)
        return 1

    world = Path(target).resolve()
    skipped: list = []
    inputs = world_sources(world, skipped=skipped)
    if args.require_complete and world not in inputs:
        skipped.insert(0, f"world {world} did not resolve")
    if args.require_complete and skipped:
        # Named, every one of them: a caller that asked for a complete answer is staging these
        # files somewhere the originals are not, and "something is missing" does not say which
        # world, which parent or which plugin to go and fix.
        for problem in skipped:
            print(problem, file=sys.stderr)
        return 1
    # ``packaged`` says the files arrive with an installed package rather than needing to
    # travel with whatever is staging them. The list is reported either way -- a caller
    # asking what a world depends on wants it; a caller deciding what to copy reads the
    # flag first.
    print(
        json.dumps({"world": str(world), "packaged": packaged, "inputs": [str(p) for p in inputs]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
