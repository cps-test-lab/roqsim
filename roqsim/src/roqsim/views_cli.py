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

"""The conventions of a views file, and a ready example file for a preset (`roqsim render --views`).

    roqsim views describe                  # the keys, the presets and the units
    roqsim views describe colmap > v.yaml  # ...and an example file to fill in: it is valid as written
    roqsim views describe ros --json       # the same as one JSON document

The text is a views file's own YAML, the conventions as comments, so the output of a preset is a
file ``roqsim render --check`` accepts. The presets are :data:`roqsim.views.PRESETS`, the one
declaration ``roqsim render --help``, a refusal and the docs are made from.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import exit_status
from .views import PRESETS, ViewsError, conventions_text, describe_conventions, example_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim views",
        description="The conventions of a views file (roqsim render --views), and an example file.",
        epilog=exit_status.epilog(exit_status.BAD_INPUT, note="2 includes an unknown preset."),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_describe = sub.add_parser(
        "describe",
        help="The keys, the presets and the units; with a preset, a ready example views file.",
    )
    p_describe.add_argument(
        "preset", nargs="?", help=f"a preset to give an example file for: {', '.join(PRESETS)}"
    )
    p_describe.add_argument("--json", action="store_true", help="one JSON document instead")
    args = parser.parse_args(argv)
    try:
        if args.json:
            print(json.dumps(describe_conventions(args.preset), indent=1))
            return exit_status.OK
        text = conventions_text()
        if args.preset is not None:
            text += "\n" + example_file(args.preset)[1]
        sys.stdout.write(text)
    except ViewsError as err:
        return exit_status.fail("roqsim views", err)
    return exit_status.OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
