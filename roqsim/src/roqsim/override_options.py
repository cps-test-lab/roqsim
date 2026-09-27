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

"""The ``--set`` / ``--override`` pair, for every command that loads a world a run would load.

``roqsim sim``, ``roqsim check``, ``roqsim render``, ``roqsim scenes describe``, every ``roqsim
export`` that compiles a world and ``roqsim sensors coverage`` each answer a question about the
world a run with a given set of overrides builds. They are only answers about the *same* world while
all of them spell the overrides the same way and merge them in the same order, so the options and
the merge are defined here once rather than per command::

    parser = argparse.ArgumentParser(...)
    add_override_options(parser)
    args = parser.parse_args(argv)
    overrides = overrides_from_options(args)   # the nested dict load_config takes

Files are merged first, in order, then ``--set`` over them: a saved override set plus one ad-hoc
tweak is the obvious way to use the two together, and the tweak is what should win.

A command that also compiles a bare MJCF or a model refuses these options for it, with
:func:`refuse_world_options`, rather than ignoring them.
"""

from __future__ import annotations

import argparse
import sys

from . import exit_status
from .config import deep_merge, overrides_from_dotlist, overrides_from_files


def add_override_options(parser: argparse.ArgumentParser) -> None:
    """Add ``--set PATH=VALUE`` (as ``overrides``) and ``--override FILE`` (as ``override_files``).

    Both are repeatable and default to ``None``; :func:`overrides_from_options` reads them back.
    """
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="PATH=VALUE",
        help="override a world value, e.g. --set components.floorplan.floor.reflectance=0.3 "
        "(repeatable)",
    )
    parser.add_argument(
        "--override",
        dest="override_files",
        action="append",
        metavar="FILE",
        help="a YAML file of world overrides -- the file spelling of --set, for anything "
        "structured enough that flattening it onto a command line loses it (repeatable; "
        "later files and --set win)",
    )


def overrides_from_options(args: argparse.Namespace) -> dict:
    """The nested override dict *args* spells: every ``--override`` file in order, then ``--set``.

    Raises :class:`roqsim.plugin.PluginError` for a file that cannot be read, is not YAML or is not
    a mapping, and for a ``--set`` that is not ``path=value``.
    """
    return deep_merge(
        overrides_from_files(args.override_files), overrides_from_dotlist(args.overrides)
    )


#: The options that act on a world YAML, and the ``argparse`` dest each is read back from.
WORLD_OPTIONS = {
    "--set": "overrides",
    "--override": "override_files",
    "--skip-plugins": "skip_plugins",
    "--settle-steps": "settle_steps",
}


def refuse_world_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace, why: str
) -> None:
    """Exit with :data:`roqsim.exit_status.BAD_INPUT` if *args* gives any of :data:`WORLD_OPTIONS`.

    Call it when the source is not a world YAML; *why* says what the source is instead and what to
    use. An option the parser does not define counts as not given.
    """
    given = [flag for flag, dest in WORLD_OPTIONS.items() if getattr(args, dest, None)]
    if given:
        parser.print_usage(sys.stderr)
        parser.exit(
            exit_status.BAD_INPUT,
            f"{parser.prog}: error: {', '.join(given)} "
            f"{'acts' if len(given) == 1 else 'act'} on a world YAML, and {why}\n",
        )
